"""Wildlife observer CLI. No PTZ, property-write, or global disconnect commands are sent; step plans are logged only."""
import argparse
import asyncio
import json
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from .core import CONTEXT_EVENTS, Config, Journal, Marker, ProtocolError, RunLock, connect, guarded_start, guarded_stop, log, metadata_summary, reconcile, status, trigger_context, wake_event
from .vision import MEGADETECTOR_CLASSES, Decoder, Detector, PROMPTS, model_labels
from .tracking import AnimalTracker
from .ptz import StepPlanner, StepPolicy
from .follow import FollowController, follow_available


def redacted_structure(value, key="", depth=0):
    """Retain shapes and safe event names; redact every untrusted string."""
    if depth > 8:
        return "[depth-limit]"
    if isinstance(value, dict):
        return {str(k)[:80]: redacted_structure(v, str(k), depth + 1) for k, v in list(value.items())[:64]
                if str(k).lower() not in {"buffer", "image", "picture", "captcha", "token", "password"}}
    if isinstance(value, list):
        return [redacted_structure(v, key, depth + 1) for v in value[:32]]
    if isinstance(value, str):
        if key in {"event", "source"} and value in {"device", "driver", "station", "server", "livestream started", "livestream stopped"} | CONTEXT_EVENTS:
            return value
        return "[redacted]"
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return "[unsupported]"


async def selected_device(client, config):
    devices = (client.state or {}).get("devices") or []
    if config.serial not in devices:
        raise ProtocolError("Selected serial absent from WS device list")
    properties = await client.command("device.get_properties", serialNumber=config.serial)
    values = properties.get("properties") or {}
    model = values.get("model")
    if isinstance(model, dict):
        model = model.get("value")
    if not isinstance(model, str) or not model:
        raise ProtocolError("Selected device model unavailable")
    if model not in {"T8170", "SoloCam S340"}:
        raise ProtocolError("Selected device is not an identified SoloCam S340")
    return model, values


async def collect_stream(client, config, *, detector=None, duration=None, keep_detector=False, follow=None):
    marker = Marker(config.data_dir)
    decoder = None
    feed_task = None
    ownership_token = None
    frame_count = 0
    detection_count = 0
    inference_count = 0
    timings = {}
    began = time.monotonic()
    phase = "precheck"
    tracker = AnimalTracker() if detector else None
    if follow:
        planner = follow.planner  # closed loop, real steps
    else:
        # Dry run: the camera never moves, so closed-loop verification would always lock; simulate spacing instead.
        planner = StepPlanner(StepPolicy(closed_loop=False)) if detector else None
    previous_tracking_state = None
    proposal_count = 0
    planned_steps = 0
    labels = {}
    camera_summary = {"known_frames": 0, "unknown_frames": 0, "moving_frames": 0, "max_shift": 0.0}
    previous_moving = None
    night = False
    try:
        if marker.exists():
            await reconcile(client, marker, config.serial)
        if marker.exists():
            raise ProtocolError("Prior stream ownership needs manual reconciliation")
        if await status(client, config.serial):
            log("skip_existing_stream")
            return {"frames": 0, "detections": 0, "skipped": True}
        if detector:
            # Cold model loading happens before stream acquisition and before target timers.
            phase = "model_load"
            loading = time.monotonic()
            await detector.start()
            timings["model_start_seconds"] = round(time.monotonic() - loading, 3)
            log("model_ready", model=Path(config.model_path).name, **timings)
        phase = "start_command"
        acquiring = time.monotonic()
        ownership_token = await guarded_start(client, marker, config.serial)
        if not ownership_token:
            return {"frames": 0, "detections": 0, "skipped": True}
        timings["start_ack_seconds"] = round(time.monotonic() - acquiring, 3)
        log("stream_start_ack", video_confirmed=False, **timings)
        phase = "first_video_chunk"
        chunk_task = asyncio.create_task(client.video.get())
        try:
            done, _ = await asyncio.wait({chunk_task, client.reader_task}, timeout=config.startup_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if client.reader_task in done:
                raise ConnectionError("WS reader stopped during stream startup")
            if chunk_task not in done:
                raise TimeoutError("First video chunk deadline")
            chunk, metadata = chunk_task.result()
            timings["first_video_seconds"] = round(time.monotonic() - acquiring, 3)
        finally:
            if not chunk_task.done():
                chunk_task.cancel()
                await asyncio.gather(chunk_task, return_exceptions=True)
        decoder = Decoder(metadata.get("videoCodec"))
        phase = "decode"
        await decoder.start()
        log("stream_metadata", codec=decoder.fmt,
            width=metadata.get("videoWidth") if isinstance(metadata.get("videoWidth"), int) else None,
            height=metadata.get("videoHeight") if isinstance(metadata.get("videoHeight"), int) else None)

        async def feeder():
            await decoder.feed(chunk)
            while not client.closed:
                part, meta = await asyncio.wait_for(client.video.get(), config.frame_seconds)
                if {"H264": "h264", "H265": "hevc", "HEVC": "hevc"}.get(str(meta.get("videoCodec")).upper()) != decoder.fmt:
                    raise ProtocolError("Video codec changed midstream")
                await decoder.feed(part)
            raise ConnectionError("WS closed")

        feed_task = asyncio.create_task(feeder())

        async def next_frame(until):
            """The next decoded frame, or None at `until`; feeder/reader failures propagate."""
            if feed_task.done():
                await feed_task
            if client.reader_task.done():
                raise ConnectionError("WS reader stopped during stream")
            frame_task = asyncio.create_task(decoder.frame(max(.01, min(config.frame_seconds, until - time.monotonic()))))
            try:
                done, _ = await asyncio.wait({frame_task, feed_task, client.reader_task},
                                             timeout=max(0, until - time.monotonic()),
                                             return_when=asyncio.FIRST_COMPLETED)
                if client.reader_task in done:
                    raise ConnectionError("WS reader stopped during stream")
                if feed_task in done:
                    await feed_task
                if frame_task not in done:
                    return None
                return frame_task.result()
            finally:
                if not frame_task.done():
                    frame_task.cancel()
                    await asyncio.gather(frame_task, return_exceptions=True)

        async def return_home():
            """Return by verified reverse steps while the owned stream still shows the view."""
            until = time.monotonic() + follow.return_seconds()
            while True:
                frame = await next_frame(until)
                if frame is None:
                    follow.fail("return_deadline")
                    break
                detector.submit(frame)
                await detector.result(config.frame_seconds)
                if await follow.return_step(time.monotonic(), getattr(detector, "camera", None)):
                    break
            log("follow_return", verified=follow.return_verified, offset=follow.offset)

        started = time.monotonic()
        last_target = None
        limit = duration or config.session_seconds
        deadline = started + limit
        extended = waited_out = False
        while time.monotonic() < deadline:
            try:
                frame = await next_frame(deadline)
            except (TimeoutError, asyncio.TimeoutError):
                if extended and time.monotonic() - started > limit:
                    # Past the normal session length the stream only stays for the recording; end quietly.
                    log("night_hold", ended="video_stopped")
                    break
                raise
            if frame is None:
                break
            frame_count += 1
            if frame_count == 1:
                timings["first_decoded_frame_seconds"] = round(time.monotonic() - acquiring, 3)
            if detector:
                phase = "inference"
                frame_received_at = time.monotonic()
                detector.submit(frame)
                # Full inference allowance even at the session deadline: clipping it to the remaining
                # session time made a session that ran its full length fail (and skip the follow return).
                found = await detector.result(config.frame_seconds)
                inference_count += 1
                phase = "decode"
                camera = getattr(detector, "camera", None)
                night = night or bool(camera and camera.get("grayscale"))
                if night and not extended and detector and config.night_hold_seconds > 0:
                    # The stream stays open for Eufy's recording anyway (stopping it cuts the IR light),
                    # so keep watching and following for that whole time. On 2026-10-11 02:28 analysis
                    # stopped after 11 s while the stream ran another 115 s unobserved.
                    extended = True
                    deadline = max(deadline, acquiring + config.night_hold_seconds)
                    log("night_session", observe_seconds=round(deadline - time.monotonic(), 1))
                if camera and camera.get("shift") is not None:
                    camera_summary["known_frames"] += 1
                    camera_summary["moving_frames"] += bool(camera["moving"])
                    camera_summary["max_shift"] = max(camera_summary["max_shift"], *(abs(v) for v in camera["shift"]))
                elif inference_count > 1:
                    camera_summary["unknown_frames"] += 1
                moving = camera.get("moving") if camera else None
                if moving is not previous_moving and inference_count > 1:
                    # View motion not commanded by this service: native tracking, wind, or an unknown scene change.
                    log("camera_motion", moving=moving, shift=camera.get("shift") if camera else None,
                        response=camera.get("response") if camera else None)
                    previous_moving = moving
                now = time.monotonic()
                decision = tracker.update(found, now=now, observed_at=frame_received_at, camera=camera)
                if decision["action"] != "none" or decision["state"] != previous_tracking_state:
                    log("tracking_dry_run", **decision)
                previous_tracking_state = decision["state"]
                if decision["action"] == "move":
                    proposal_count += 1
                if follow:
                    await follow.act(decision, now, camera)
                else:
                    step = planner.plan(decision, now, camera)
                    if step["command"]:
                        planned_steps += 1
                        log("ptz_dry_run", would_send=step["command"], ledger=step["ledger"],
                            executable=False, direction_mapping="unverified")
                for item in found:
                    entry = labels.setdefault(item["label"], {"detections": 0, "max_score": 0.0})
                    entry["detections"] += 1
                    entry["max_score"] = max(entry["max_score"], item["score"])
                if found:
                    detection_count += len(found)
                    log("animal_observed", detections=len(found),
                        target_id=decision["target_id"], label=decision["label"])
                # Present = the tracker matched its target in this very frame, at least twice so far;
                # an isolated noise blob or a held-but-unseen target does not keep the session alive.
                target = tracker.target
                if target is not None and target["last_seen"] == frame_received_at and target["matched"] >= 2:
                    last_target = time.monotonic()
                elif last_target is None and time.monotonic() - started >= config.initial_target_seconds:
                    if not extended:
                        log("initial_target_timeout")
                        break
                    if not waited_out:
                        waited_out = True
                        log("initial_target_timeout", continuing="night_session")
                elif last_target is not None and time.monotonic() - last_target >= config.target_loss_seconds:
                    log("target_lost")
                    if not extended:
                        break
                    # Night: go home now, then keep watching until the recording is over.
                    if follow and follow.returning_needed():
                        phase = "follow_return"
                        await return_home()
                        phase = "decode"
                    last_target = None
        if follow and follow.returning_needed():
            phase = "follow_return"
            await return_home()
        if night and config.night_hold_seconds > 0:
            # Stopping mid-recording at night switched the IR light off; outlast Eufy's clip instead.
            phase = "night_hold"
            held = time.monotonic()
            hold_until = acquiring + config.night_hold_seconds
            ended = "deadline"
            try:
                while time.monotonic() < hold_until and await next_frame(hold_until) is not None:
                    pass
            except (TimeoutError, asyncio.TimeoutError):
                # Video stopped arriving: the hold has nothing left to protect; stop normally.
                ended = "video_stopped"
            log("night_hold", seconds=round(time.monotonic() - held, 1), ended=ended)
        return {"frames": frame_count, "inferences": inference_count, "detections": detection_count, "skipped": False,
                "night": night,
                "labels": labels, "camera_motion": camera_summary if detector else None,
                "tracking_proposals": proposal_count, "planned_steps": planned_steps,
                "control_enabled": follow is not None, "follow": follow.summary() if follow else None,
                "dropped_video_chunks": client.video_dropped}
    finally:
        if follow:
            follow.close()
        if tracker:
            log("tracking_dry_run", **tracker.finish())
            log("ptz_dry_run", **planner.finish())
        log("stream_diagnostic", phase=phase, frames=frame_count, inferences=inference_count,
            elapsed_seconds=round(time.monotonic() - began, 3), timings=timings,
            stream_signals=client.stream_signals, dropped_video_chunks=client.video_dropped)
        client.video_enabled = False
        if feed_task:
            feed_task.cancel()
            await asyncio.gather(feed_task, return_exceptions=True)
        try:
            if decoder:
                await decoder.close()
        finally:
            try:
                if detector and not keep_detector:
                    detector.close()
            finally:
                if ownership_token:
                    await guarded_stop(client, marker, config.serial, ownership_token)


class WarmModel:
    """One loaded model worker kept across sessions and reconnects; replaced after any failure."""
    def __init__(self, config):
        self.config = config
        self.detector = None

    async def acquire(self):
        if self.detector is not None and self.detector.alive():
            try:
                await self.detector.reset()
                return self.detector
            except Exception as error:
                log("warm_model_reset_failed", error=type(error).__name__)
                self.close()
        self.close()
        detector = Detector(self.config.model_path, self.config.confidence)
        began = time.monotonic()
        await detector.start()
        self.detector = detector
        log("model_ready_warm", load_seconds=round(time.monotonic() - began, 3), model=Path(self.config.model_path).name)
        return detector

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None


async def power_state(client, serial):
    """Battery percent and raw charging status; None when unavailable (never blocks a session)."""
    values = (await client.command("device.get_properties", serialNumber=serial)).get("properties") or {}
    def number(key):
        raw = values.get(key)
        raw = raw.get("value") if isinstance(raw, dict) else raw
        return raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
    return {"battery": number("battery"), "charging_status": number("chargingStatus")}


async def run(config, action, capture_seconds=30, show_serials=False):
    models = WarmModel(config) if action == "run" and config.mode == "observe" and config.warm_model else None
    try:
        await _run(config, action, capture_seconds, show_serials, models)
    finally:
        if models:
            models.close()


async def _run(config, action, capture_seconds, show_serials, models):
    marker = Marker(config.data_dir)
    journal = Journal(config.data_dir)
    backoff = 1.0
    while True:
        client = None
        connected_at = None
        try:
            client = await connect(config)
            connected_at = time.monotonic()
            log("ws_connected", **client.version, **metadata_summary(client.state))
            if action == "inventory":
                entries = []
                for serial in (client.state or {}).get("devices") or []:
                    props = await client.command("device.get_properties", serialNumber=serial)
                    meta = await client.command("device.get_properties_metadata", serialNumber=serial)
                    commands = await client.command("device.get_commands", serialNumber=serial)
                    values = props.get("properties") or {}
                    def safe_value(key):
                        value = values.get(key)
                        return value.get("value") if isinstance(value, dict) else value
                    entries.append({"serial": serial if show_serials else "[redacted]",
                                    "model": safe_value("model"), "hardware_version": safe_value("hardwareVersion"),
                                    "software_version": safe_value("softwareVersion"),
                                    "property_names": sorted(str(k) for k in values),
                                    "metadata_property_names": sorted(str(k) for k in (meta.get("properties") or {})),
                                    "supported_commands": sorted(str(k) for k in (commands.get("commands") or []))})
                print(json.dumps({"source": "real_ws", "devices": entries, "summary": metadata_summary(client.state)}, sort_keys=True))
                return
            if action != "capture-events":
                model, values = await selected_device(client, config)
                log("selected_device", model=model, serial_redacted=True)
                if marker.exists():
                    await reconcile(client, marker, config.serial)
            if action == "reconcile":
                print(json.dumps({"ownership_uncertain": marker.exists(), "stream_running": await status(client, config.serial)}))
                return
            if action == "probe-stream":
                print(json.dumps(await collect_stream(client, config, duration=min(capture_seconds, 30))))
                return
            if action == "capture-events":
                until = time.monotonic() + min(capture_seconds, 300)
                count = 0
                while time.monotonic() < until:
                    event_task = asyncio.create_task(client.events.get())
                    try:
                        done, _ = await asyncio.wait({event_task, client.reader_task},
                                                     timeout=until - time.monotonic(),
                                                     return_when=asyncio.FIRST_COMPLETED)
                        if client.reader_task in done:
                            raise ConnectionError("WS reader stopped during capture")
                        if event_task not in done:
                            break
                        event = event_task.result()
                    finally:
                        if not event_task.done():
                            event_task.cancel()
                            await asyncio.gather(event_task, return_exceptions=True)
                    log("captured_event", structure=redacted_structure(event))
                    count += 1
                log("capture_complete", count=count, source="real_ws")
                return
            if models:
                await models.acquire()  # Cold load once, before the first event, not inside a session.
            unavailable = follow_available(client, config)
            log("follow_mode", enabled=config.follow, available=unavailable is None, reason=unavailable)
            # One session at a time. Events accumulated during a session are stale and discarded.
            while True:
                event_task = asyncio.create_task(client.events.get())
                try:
                    done, _ = await asyncio.wait({event_task, client.reader_task}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not event_task.done():
                        event_task.cancel()
                        await asyncio.gather(event_task, return_exceptions=True)
                if client.reader_task in done:
                    raise ConnectionError("WS reader stopped")
                event = event_task.result()
                if event.get("source") == "device" and event.get("serialNumber") == config.serial:
                    log("device_event", event=event["event"] if event.get("event") in CONTEXT_EVENTS else "other")
                if config.mode == "listen" or not wake_event(event, config.serial):
                    continue
                if marker.exists():
                    # The stream may have stopped since; a verified stopped stream clears the marker.
                    try:
                        await reconcile(client, marker, config.serial)
                    except ProtocolError as error:
                        log("ownership_reconciliation_refused", error=type(error).__name__)
                if marker.exists():
                    log("wake_blocked_uncertain_ownership")
                    continue
                triggered = time.monotonic()
                power = await power_state(client, config.serial)
                if power["battery"] is not None and power["battery"] < config.min_battery:
                    # Battery/solar camera: every session is an extra live stream.
                    log("session_skipped_low_battery", minimum=config.min_battery, **power)
                    journal.write("session_skipped_low_battery", minimum=config.min_battery, **power)
                    continue
                log("state", value="TRIGGERED")
                try:
                    log("state", value="OBSERVING")
                    detector = await models.acquire() if models else Detector(config.model_path, config.confidence)
                    follow = FollowController(client, config) if unavailable is None else None
                    outcome = await collect_stream(client, config, detector=detector, keep_detector=models is not None,
                                                   follow=follow)
                    trigger = trigger_context(client.device_events, triggered - 5)
                    log("session_complete", **outcome, trigger=trigger, power=power, warm_model=models is not None)
                    journal.write("session_complete", started=round(time.time() - (time.monotonic() - triggered), 3),
                                  **outcome, trigger=trigger, power=power)
                except Exception as error:
                    log("session_failed", error=type(error).__name__)
                    journal.write("session_failed", error=type(error).__name__,
                                  started=round(time.time() - (time.monotonic() - triggered), 3), power=power)
                    if models:
                        models.close()  # A fresh worker is loaded at the next acquire.
                while not client.events.empty():
                    client.events.get_nowait()
                log("state", value="COOLDOWN")
                done, _ = await asyncio.wait({client.reader_task}, timeout=config.cooldown_seconds)
                if done:
                    raise ConnectionError("WS reader stopped during cooldown")
                while not client.events.empty():
                    client.events.get_nowait()
                log("state", value="IDLE")
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        except Exception as error:
            log("connection_failure", error=type(error).__name__)
            if action != "run":
                raise
            if connected_at is not None and time.monotonic() - connected_at >= 60:
                backoff = 1.0
            await asyncio.sleep(min(backoff, 30) + random.random() * .5)
            backoff = min(backoff * 2, 30)
        finally:
            if client:
                await client.close()
        if action != "run":
            return


# MegaDetector V6 compact, the URL PytorchWildlife 1.3.0 uses for MDV6-yolov10-c; pinned by hash.
MEGADETECTOR_URL = "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1"
MEGADETECTOR_SHA256 = "21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7"


def megadetector_install(path):
    """Explicit network/download action, never called by run."""
    import hashlib
    import urllib.request
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".part.pt")  # ultralytics loads only .pt names
    digest = hashlib.sha256()
    with urllib.request.urlopen(MEGADETECTOR_URL, timeout=60) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            handle.write(chunk)
    if digest.hexdigest() != MEGADETECTOR_SHA256:
        partial.unlink()
        raise RuntimeError("Downloaded MegaDetector checksum mismatch")
    from ultralytics import YOLO
    try:
        if model_labels(YOLO(str(partial)).names)[0] != "animal":
            raise RuntimeError("Downloaded MegaDetector classes unexpected")
    except Exception:
        partial.unlink()
        raise
    partial.replace(destination)
    print(json.dumps({"model_installed": str(destination), "bytes": destination.stat().st_size, "classes": list(MEGADETECTOR_CLASSES)}))


def model_install(path):
    """Explicit network/download action, never called by run. The YOLO-World alternative to MegaDetector."""
    from ultralytics.utils.downloads import attempt_download_asset
    from ultralytics import YOLOWorld
    source = Path(attempt_download_asset("yolov8s-worldv2.pt"))
    model = YOLOWorld(str(source))
    model.set_classes(list(PROMPTS))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(destination))
    # Reload the prepared checkpoint, including its text embeddings and names.
    from ultralytics import YOLO
    prepared = YOLO(str(destination))
    if list(prepared.names.values()) != list(PROMPTS):
        raise RuntimeError("Prepared model prompts did not persist")
    print(json.dumps({"model_installed": str(destination), "bytes": destination.stat().st_size, "prompts": list(PROMPTS)}))


def main():
    parser = argparse.ArgumentParser(description="Eufy wildlife observation; camera controls locked")
    parser.add_argument("action", nargs="?", choices=("run", "inventory", "capture-events", "probe-stream", "reconcile", "model-install", "preflight", "model-check", "trial", "pilot", "ptz-probe", "replay", "ptz-ledger-clear", "archive-clips"), default="run")
    parser.add_argument("--path", default="/data/clips", help="Replay/archive: a recorded clip or a directory of clips")
    parser.add_argument("--out", default="/data/archive", help="archive-clips only: where trimmed animal segments go")
    parser.add_argument("--candidates", help="archive-clips only: save frames from the discarded parts here for labelling")
    parser.add_argument("--keep-undetected", action="store_true", help="archive-clips only: copy clips with no detection whole into --out")
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--show-serials", action="store_true", help="Print serials only to your private terminal for device selection")
    parser.add_argument("--preload", action="store_true", help="Trial only: warm model on a synthetic frame before READY")
    parser.add_argument("--max-sessions", type=int, default=3, help="Pilot only: 1–3 event-triggered sessions")
    parser.add_argument("--axis", choices=("pan", "tilt"), default="pan", help="ptz-probe only: left/right or up-to-limit/down")
    parser.add_argument("--max-up", type=int, default=5, help="ptz-probe tilt only: 1–8 up steps while searching the limit")
    parser.add_argument("--model", choices=("megadetector", "world"), default="megadetector",
                        help="model-install only: MegaDetector V6 compact, or the eight-prompt YOLO-World checkpoint")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("--seconds must be between 1 and 300")
    if not 1 <= args.max_sessions <= 3:
        parser.error("--max-sessions must be between 1 and 3")
    if not 1 <= args.max_up <= 8:
        parser.error("--max-up must be between 1 and 8")
    if args.action == "model-install":
        install = megadetector_install if args.model == "megadetector" else model_install
        install(os.environ.get("WILDLIFE_MODEL_PATH", Config.model_path))
        return
    config = Config.from_env(require_serial=args.action not in {"inventory", "capture-events", "preflight", "model-check", "trial", "pilot", "ptz-probe", "replay", "ptz-ledger-clear", "archive-clips"})
    config.model_path = os.environ.get("WILDLIFE_MODEL_PATH", config.model_path)
    if args.action == "run":
        if os.environ.get("WILDLIFE_ENABLED") != "1":
            parser.error("WILDLIFE_ENABLED=1 is required for service run")
        if config.mode == "observe" and not Path(config.model_path).is_file():
            parser.error("Prepared model missing; run model-install explicitly")
    async def service():
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        stopping = False
        def request_stop():
            nonlocal stopping
            if not stopping:
                stopping = True
                task.cancel()
        for signum in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(signum, request_stop)
            except NotImplementedError:
                pass
        if args.action in {"preflight", "model-check", "trial", "pilot"}:
            from .supervised import preflight, model_check, trial, pilot
            if args.action == "preflight":
                await preflight(config)
            elif args.action == "model-check":
                await model_check(config)
            elif args.action == "pilot":
                await pilot(config, args.seconds, args.max_sessions)
            else:
                await trial(config, args.seconds, preload=args.preload)
        elif args.action == "ptz-ledger-clear":
            # Supervised: only after the user confirms in the owner App that the camera is home.
            from .follow import ledger_clear
            await ledger_clear(config)
        elif args.action == "archive-clips":
            # Offline: trims recordings to animal segments; sources are never modified or deleted.
            from .replay import archive
            await archive(config, args.path, args.out, args.candidates, args.keep_undetected)
        elif args.action == "replay":
            # Offline: recorded clips only, no WS or camera.
            from .replay import replay
            await replay(config, args.path)
        elif args.action == "ptz-probe":
            # Supervised hardware probe only; never part of run/trial/pilot.
            from .probe import ProbeLimits, ptz_probe
            await ptz_probe(config, axis=args.axis, limits=ProbeLimits(max_up_steps=args.max_up))
        else:
            await run(config, args.action, args.seconds, args.show_serials)
    try:
        # Keep the descriptor open until all async cleanup has finished.
        guard = RunLock(config.data_dir) if args.action in {"run", "trial", "pilot", "probe-stream", "reconcile", "ptz-probe", "ptz-ledger-clear"} else nullcontext()
        with guard:
            asyncio.run(service())
    except asyncio.CancelledError:
        pass
    except Exception as error:
        log("command_failed", error=type(error).__name__)
        sys.exit(1)


if __name__ == "__main__":
    main()
