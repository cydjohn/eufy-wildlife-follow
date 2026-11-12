"""Read-only preparation and explicitly armed, single-attempt acceptance trial."""
import asyncio
import sys
import time
from datetime import datetime, timezone

from .core import Marker, ProtocolError, connect, log, metadata_summary, status, trigger_context, wake_event
from .vision import Detector, FRAME_BYTES

# Versions whose source and live behavior this project inspected. A watchtower
# update of eufy-security-ws:latest changes the API under the observer.
VERIFIED_VERSIONS = {"serverVersion": "3.1.0", "driverVersion": "4.1.0"}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


async def inspect_selected(client, config):
    from .__main__ import selected_device
    if not config.serial:
        devices = (client.state or {}).get("devices") or []
        if len(devices) != 1:
            raise ProtocolError("Explicit selection required unless exactly one device exists")
        config.serial = next(iter(devices))
    model, values = await selected_device(client, config)
    def value(key):
        raw = values.get(key)
        raw = raw.get("value") if isinstance(raw, dict) else raw
        return raw if isinstance(raw, (bool, int, float)) else None
    driver = await client.command("driver.is_connected")
    push = await client.command("driver.is_push_connected")
    report = {
        "model": model, "versions": client.version,
        "versions_verified": all((client.version or {}).get(k) == v for k, v in VERIFIED_VERSIONS.items()),
        "inventory": metadata_summary(client.state),
        "driver_connected": driver.get("connected"),
        "push_connected": push.get("connected"),
        "stream_running": await status(client, config.serial),
        "ownership_marker_present": (config.data_dir / "stream-ownership.json").exists(),
        "settings": {key: value(key) for key in (
            "motionDetection", "motionTracking", "audioRecording", "notificationType",
            "recordingClipLength", "recordingEndClipMotionStops", "recordingRetriggerInterval")},
        "automatic_ptz": False,
    }
    log("preflight", **report)
    return report


async def preflight(config):
    client = await connect(config)
    try:
        return await inspect_selected(client, config)
    finally:
        await client.close()


async def warm_model(detector):
    began = time.monotonic()
    await detector.start()
    load_seconds = time.monotonic() - began
    measurements = []
    for _ in range(2):
        began = time.monotonic()
        detector.submit(bytes(FRAME_BYTES))
        detections = await detector.result(60)
        measurements.append({"seconds": round(time.monotonic() - began, 3), "detections": len(detections)})
    report = {"source": "synthetic_blank", "load_seconds": round(load_seconds, 3),
              "first_inference": measurements[0], "warm_inference": measurements[1]}
    log("model_check", **report)
    return report


async def model_check(config):
    detector = Detector(config.model_path, config.confidence)
    try:
        return await warm_model(detector)
    finally:
        detector.close()


async def confirm_start():
    loop = asyncio.get_running_loop()
    answer = loop.create_future()
    def readable():
        if not answer.done():
            answer.set_result(sys.stdin.readline().strip() == "START")
    try:
        loop.add_reader(sys.stdin.fileno(), readable)
    except (OSError, ValueError):
        return False
    try:
        return await answer
    finally:
        loop.remove_reader(sys.stdin.fileno())


async def trial(config, seconds=120, *, preload=False, confirm=None):
    """READY opens no socket. START must follow explicit human readiness in chat.

    Only ARMED means the fresh window is active. There is no retry and no second
    session. A no-event window remains inconclusive until stimulus is confirmed.
    """
    from .__main__ import collect_stream
    detector = Detector(config.model_path, config.confidence)
    client = None
    try:
        if preload:
            await warm_model(detector)
        log("READY", capture_active=False, awaiting="explicit_user_readiness_then_stdin_START",
            model_preloaded=preload, automatic_ptz=False)
        if not await (confirm or confirm_start)():
            log("trial_cancelled", capture_started=False)
            return
        client = await connect(config)
        report = await inspect_selected(client, config)
        if report["driver_connected"] is not True or report["push_connected"] is not True:
            raise ProtocolError("Cloud/push readiness not established")
        if report["stream_running"] or report["ownership_marker_present"]:
            raise ProtocolError("Trial blocked by existing stream or ownership marker")
        while not client.events.empty():
            client.events.get_nowait()
        armed = time.monotonic()
        log("ARMED", window_seconds=seconds, instruction="Walk once through the camera view now and stay 5–10 seconds", model_preloaded=preload)
        deadline = armed + seconds
        while time.monotonic() < deadline:
            event_task = asyncio.create_task(client.events.get())
            try:
                done, _ = await asyncio.wait({event_task, client.reader_task},
                    timeout=max(0, deadline - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
                if client.reader_task in done:
                    raise ConnectionError("WS reader stopped while armed")
                if event_task not in done:
                    break
                event = event_task.result()
            finally:
                if not event_task.done():
                    event_task.cancel()
                    await asyncio.gather(event_task, return_exceptions=True)
            if not wake_event(event, config.serial):
                continue
            log("trial_trigger", source="real_device", event=event["event"], state=True, utc=utc_now(),
                seconds_since_armed=round(time.monotonic() - armed, 3))
            result = await collect_stream(client, config, detector=detector, duration=min(config.session_seconds, 15))
            stopped = not await status(client, config.serial)
            marker_present = Marker(config.data_dir).exists()
            passed = result.get("inferences", 0) > 0 and stopped and not marker_present
            log("trial_complete", **result, stream_stopped=stopped, ownership_marker_present=marker_present,
                pipeline_passed=passed, recording_and_notification="awaiting_user_check",
                stimulus="awaiting_user_time_confirmation", model_preloaded=preload)
            return result
        log("trial_no_event", verdict="inconclusive", stimulus="awaiting_user_confirmation",
            stream_started=False)
    except Exception as error:
        log("trial_failed", error=type(error).__name__, retry=False)
        raise
    finally:
        detector.close()
        if client:
            try:
                log("trial_cleanup", stream_running=await status(client, config.serial),
                    ownership_marker_present=(config.data_dir / "stream-ownership.json").exists())
            except Exception as error:
                log("trial_cleanup_unverified", error=type(error).__name__)
            await client.close()


async def pilot(config, seconds=300, max_sessions=3, *, confirm=None):
    """Natural-event observation; explicit readiness, fixed budget, no error retry."""
    from .__main__ import collect_stream
    if not 0 < seconds <= 300 or not 1 <= max_sessions <= 3:
        raise ValueError("Pilot exceeds its time/session budget")
    client = None
    attempts = 0
    successes = 0
    total_inferences = 0
    total_detections = 0
    categories = {"person_associated": 0, "vehicle_associated": 0, "motion_only": 0}
    motion_only_with_candidates = 0
    try:
        log("READY", mode="pilot", capture_active=False, window_seconds=seconds,
            max_sessions=max_sessions, awaiting="explicit_user_readiness_then_stdin_START", automatic_ptz=False)
        if not await (confirm or confirm_start)():
            log("pilot_cancelled", capture_started=False)
            return
        client = await connect(config)
        report = await inspect_selected(client, config)
        if report["driver_connected"] is not True or report["push_connected"] is not True:
            raise ProtocolError("Cloud/push readiness not established")
        if report["stream_running"] or report["ownership_marker_present"]:
            raise ProtocolError("Pilot blocked by existing stream or ownership marker")
        while not client.events.empty():
            client.events.get_nowait()
        deadline = time.monotonic() + seconds
        log("PILOT_ARMED", window_seconds=seconds, max_sessions=max_sessions,
            no_walk_required=True, automatic_ptz=False)
        while attempts < max_sessions and time.monotonic() < deadline:
            event_task = asyncio.create_task(client.events.get())
            try:
                done, _ = await asyncio.wait({event_task, client.reader_task},
                    timeout=max(0, deadline - time.monotonic()), return_when=asyncio.FIRST_COMPLETED)
                if client.reader_task in done:
                    raise ConnectionError("WS disconnected during pilot")
                if event_task not in done:
                    break
                event = event_task.result()
            finally:
                if not event_task.done():
                    event_task.cancel()
                    await asyncio.gather(event_task, return_exceptions=True)
            if not wake_event(event, config.serial):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            attempts += 1
            triggered = time.monotonic()
            log("pilot_trigger", attempt=attempts, source="real_device", event=event["event"], utc=utc_now())
            try:
                result = await asyncio.wait_for(collect_stream(client, config,
                    detector=Detector(config.model_path, config.confidence), duration=min(15, config.session_seconds)), remaining)
            except asyncio.TimeoutError:
                log("pilot_stopped", reason="session_or_window_timeout", retry=False)
                break
            if result.get("skipped"):
                log("pilot_stopped", reason="existing_stream", retry=False)
                break
            stopped = not await status(client, config.serial)
            marker_present = (config.data_dir / "stream-ownership.json").exists()
            if not stopped or marker_present or result.get("inferences", 0) == 0:
                raise ProtocolError("Pilot session or cleanup unverified")
            successes += 1
            total_inferences += result["inferences"]
            total_detections += result["detections"]
            context = trigger_context(client.device_events, triggered - 5)
            categories[context["category"]] += 1
            motion_only_with_candidates += context["category"] == "motion_only" and result["detections"] > 0
            log("pilot_session", attempt=attempts, **result, trigger=context,
                stream_stopped=True, ownership_marker_present=False)
            if attempts >= max_sessions:
                break
            while not client.events.empty():
                client.events.get_nowait()
            wait = max(0, min(config.cooldown_seconds, deadline - time.monotonic()))
            done, _ = await asyncio.wait({client.reader_task}, timeout=wait)
            if done:
                raise ConnectionError("WS disconnected during pilot cooldown")
            while not client.events.empty():
                client.events.get_nowait()
        log("pilot_complete", attempts=attempts, successes=successes,
            inferences=total_inferences, detections=total_detections, trigger_categories=categories,
            motion_only_sessions_with_candidates=motion_only_with_candidates,
            animal_evidence="candidate_detections_require_review" if total_detections else "not_established",
            no_event_is_failure=False, automatic_ptz=False)
    except Exception as error:
        log("pilot_failed", error=type(error).__name__, attempts=attempts, retry=False)
        raise
    finally:
        if client:
            try:
                log("pilot_cleanup", stream_running=await status(client, config.serial),
                    ownership_marker_present=(config.data_dir / "stream-ownership.json").exists())
            except Exception as error:
                log("pilot_cleanup_unverified", error=type(error).__name__)
            await client.close()
