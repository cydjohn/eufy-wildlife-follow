"""Supervised pan/tilt probe inside an owned short stream.

pan:  one LEFT, then one RIGHT, each gated by a STEP line.
tilt: START authorises a bounded plan — UP until the view stops moving (the S340's
      mechanical upper limit, at most `max_up_steps`), then DOWN by the steps that
      actually moved, and a residual check against the starting view.

Never reachable from `run`, `trial` or `pilot`. It answers what one
`device.pan_and_tilt` does on this S340: whether it moves at all (shared-account
permission), fixed step or continuous motion, step size/duration/settle, the
direction sign, and whether the reverse step returns to the starting view.

Every step needs its own STEP line after the user's readiness. Any unexpected
result ends the plan. There is no stop command, so a camera that does not
settle is reported for manual recovery in the owner App, never "corrected".
Native settings are read, never written.
"""
import asyncio
import sys
import time
from dataclasses import dataclass

from .core import CommandRejected, Marker, ProtocolError, connect, guarded_start, guarded_stop, log, status
from .motion import CameraMotion, prepare
from .supervised import inspect_selected, utc_now
from .vision import Decoder, HEIGHT, WIDTH

PTZ_COMMAND = "device.pan_and_tilt"
PLAN = (("left", 1), ("right", 2))  # eufy PanTiltDirection LEFT/RIGHT only
DIRECTIONS = {"up": 3, "down": 4}    # tilt mode; ROTATE360 (0) never
EXPECTED = {"left": (0, 1), "right": (0, -1), "up": (1, 1), "down": (1, -1)}  # (axis, content sign)


@dataclass(frozen=True)
class ProbeLimits:
    baseline_frames: int = 3       # consecutive measurable still frames before any step
    baseline_seconds: float = 10
    confirm_seconds: float = 60    # STEP must arrive in this time, else the plan ends
    still_seconds: float = 3       # view must be still right before a step
    onset_seconds: float = 4       # no measurable motion by then: no_measurable_motion
    settle_frames: int = 3         # consecutive still frames after motion
    settle_seconds: float = 10     # still moving by then: unsettled
    result_seconds: float = 5      # wait for the WS "command result" event
    stream_seconds: float = 150    # whole owned stream
    returned_tolerance: float = .02
    max_up_steps: int = 5          # tilt mode: stop searching for the upper limit after this many moves


async def stdin_confirm(word, timeout):
    loop = asyncio.get_running_loop()
    answer = loop.create_future()
    def readable():
        if not answer.done():
            answer.set_result(sys.stdin.readline().strip() == word)
    try:
        loop.add_reader(sys.stdin.fileno(), readable)
    except (OSError, ValueError):
        return False
    try:
        return await (asyncio.wait_for(answer, timeout) if timeout else answer)
    except asyncio.TimeoutError:
        return False
    finally:
        loop.remove_reader(sys.stdin.fileno())


def compare(before, after):
    """Direct shift between two prepared views; None when not measurable."""
    if before is None or after is None:
        return None
    motion = CameraMotion()
    motion.update(before)
    return motion.update(after)["shift"]


class View:
    """Owned-stream frames with whole-view motion, bounded by the WS reader and feeder."""
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.decoder = None
        self.feed_task = None
        self.motion = CameraMotion()
        self.last_small = None

    async def start(self):
        client = self.client
        chunk_task = asyncio.create_task(client.video.get())
        try:
            done, _ = await asyncio.wait({chunk_task, client.reader_task}, timeout=self.config.startup_seconds,
                                         return_when=asyncio.FIRST_COMPLETED)
            if client.reader_task in done:
                raise ConnectionError("WS reader stopped during stream startup")
            if chunk_task not in done:
                raise TimeoutError("First video chunk deadline")
            chunk, metadata = chunk_task.result()
        finally:
            if not chunk_task.done():
                chunk_task.cancel()
                await asyncio.gather(chunk_task, return_exceptions=True)
        self.decoder = Decoder(metadata.get("videoCodec"), fps=5)
        await self.decoder.start()

        async def feeder():
            await self.decoder.feed(chunk)
            while not client.closed:
                part, meta = await asyncio.wait_for(client.video.get(), self.config.frame_seconds)
                if {"H264": "h264", "H265": "hevc", "HEVC": "hevc"}.get(str(meta.get("videoCodec")).upper()) != self.decoder.fmt:
                    raise ProtocolError("Video codec changed midstream")
                await self.decoder.feed(part)
            raise ConnectionError("WS closed")
        self.feed_task = asyncio.create_task(feeder())

    async def frame(self, timeout):
        frame_task = asyncio.create_task(self.decoder.frame(timeout))
        try:
            done, _ = await asyncio.wait({frame_task, self.feed_task, self.client.reader_task},
                                         timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if self.client.reader_task in done:
                raise ConnectionError("WS reader stopped during probe")
            if self.feed_task in done:
                await self.feed_task
            if frame_task not in done:
                raise asyncio.TimeoutError("Frame deadline")
            frame = frame_task.result()
        finally:
            if not frame_task.done():
                frame_task.cancel()
                await asyncio.gather(frame_task, return_exceptions=True)
        at = time.monotonic()
        small = prepare(frame, WIDTH, HEIGHT)
        self.last_small = small
        return at, self.motion.update(small)

    async def close(self):
        if self.feed_task:
            self.feed_task.cancel()
            await asyncio.gather(self.feed_task, return_exceptions=True)
        if self.decoder:
            await self.decoder.close()


async def wait_still(view, frames, seconds):
    """True after `frames` consecutive measurable still frames within `seconds`."""
    count = 0
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        _, motion = await view.frame(min(2, max(.05, until - time.monotonic())))
        count = count + 1 if motion["moving"] is False else 0
        if count >= frames:
            return True
    return False


async def await_confirm(view, confirm, word, seconds, deadline):
    """Keep consuming frames while waiting, so measurements after STEP are current."""
    task = asyncio.create_task(confirm(word, seconds))
    try:
        while not task.done():
            if time.monotonic() > deadline:
                raise TimeoutError("Probe stream budget")
            try:
                await view.frame(1)
            except asyncio.TimeoutError:
                pass
        return task.result()
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def measure(view, sent_at, limits, deadline):
    onset = last_moving = None
    still = unknown = frames = 0
    integrated = [0.0, 0.0]
    outcome = None
    while outcome is None:
        if time.monotonic() > deadline:
            raise TimeoutError("Probe stream budget")
        at, motion = await view.frame(2)
        frames += 1
        elapsed = at - sent_at
        if motion["shift"] is None:
            unknown += 1
            still = 0  # Cannot confirm settling through an unmeasurable frame.
        else:
            integrated = [a + b for a, b in zip(integrated, motion["shift"])]
            if motion["moving"]:
                onset = elapsed if onset is None else onset
                last_moving = elapsed
                still = 0
            elif onset is not None:
                still += 1
        if onset is not None and still >= limits.settle_frames:
            outcome = "settled"
        elif onset is None and elapsed >= limits.onset_seconds:
            outcome = "no_measurable_motion"
        elif onset is not None and elapsed >= limits.settle_seconds:
            outcome = "unsettled"
    return {"outcome": outcome, "frames": frames, "unknown_frames": unknown,
            "onset_seconds": round(onset, 3) if onset is not None else None,
            "motion_seconds": round(last_moving - onset, 3) if onset is not None else None,
            "integrated_shift": [round(v, 4) for v in integrated]}


def drain_events(client, seen):
    """Collect the probe-relevant event facts; never raw payloads."""
    result = None
    while not client.events.empty():
        event = client.events.get_nowait()
        if event.get("serialNumber") != client.config.serial:
            continue
        name = event.get("event")
        if name == "command result" and event.get("command") == "pan_and_tilt":
            code = event.get("returnCode")
            result = code if isinstance(code, int) else "malformed"
        elif name in {"motion detected", "person detected"} and event.get("state") is True:
            seen[name] = seen.get(name, 0) + 1
    return result


async def ptz_probe(config, *, confirm=None, limits=None, axis="pan"):
    if axis not in ("pan", "tilt"):
        raise ValueError("Probe axis is pan or tilt")
    confirm = confirm or stdin_confirm
    limits = limits or ProbeLimits()
    plan = [label for label, _ in PLAN] if axis == "pan" else [f"up×≤{limits.max_up_steps}", "down×moved"]
    log("READY", mode="ptz-probe", axis=axis, capture_active=False, plan=plan,
        awaiting="explicit_user_readiness_then_stdin_START", automatic_ptz=False)
    if not await confirm("START", None):
        log("probe_cancelled", camera_commands_sent=False)
        return None
    client = await connect(config)
    marker = Marker(config.data_dir)
    view = token = None
    native_events = {}
    report = {"source": "real_device", "axis": axis, "steps": [], "pan_commands_sent": 0, "stopped": None}
    try:
        pre = await inspect_selected(client, config)
        report["native_motion_tracking"] = pre["settings"]["motionTracking"]
        if pre["versions_verified"] is not True:
            raise ProtocolError("WS/client versions differ from the inspected ones")
        if pre["driver_connected"] is not True or pre["push_connected"] is not True:
            raise ProtocolError("Cloud/push readiness not established")
        if pre["stream_running"] or pre["ownership_marker_present"]:
            raise ProtocolError("Probe blocked by existing stream or ownership marker")
        commands = await client.command("device.get_commands", serialNumber=config.serial)
        if "pan_and_tilt" not in (commands.get("commands") or []):
            raise ProtocolError("pan_and_tilt not advertised for the selected device")
        token = await guarded_start(client, marker, config.serial)
        if not token:
            raise ProtocolError("A stream appeared; the probe never shares one")
        deadline = time.monotonic() + limits.stream_seconds
        view = View(client, config)
        await view.start()
        if not await wait_still(view, limits.baseline_frames, limits.baseline_seconds):
            report["stopped"] = "baseline_view_not_still_or_unmeasurable"
            return report
        baseline = view.last_small
        drain_events(client, native_events)
        client.extra_commands = frozenset({PTZ_COMMAND})

        async def step_once(label, direction):
            """Send one step and measure it; returns the recorded step (rejection included)."""
            before = view.last_small
            drain_events(client, native_events)
            sent_at = time.monotonic()
            step = {"step": label, "direction": direction, "utc": utc_now()}
            report["pan_commands_sent"] += 1
            try:
                await client.command(PTZ_COMMAND, serialNumber=config.serial, direction=direction)
            except CommandRejected:
                step["ack"] = "rejected"
                report["steps"].append(step)
                log("probe_step", **step)
                return step
            step["ack"] = "accepted"
            step["ack_seconds"] = round(time.monotonic() - sent_at, 3)
            step.update(await measure(view, sent_at, limits, deadline))
            result = drain_events(client, native_events)
            while result is None and time.monotonic() - sent_at < limits.result_seconds:
                await asyncio.sleep(.1)
                result = drain_events(client, native_events)
            step["command_result"] = result
            step["direct_shift"] = compare(before, view.last_small)
            if step["direct_shift"] is not None:
                # Content moves opposite to the view: LEFT → +x, RIGHT → −x, UP → +y, DOWN → −y.
                axis, sign = EXPECTED[label]
                step["matches_assumed_direction"] = step["direct_shift"][axis] * sign > 0
            report["steps"].append(step)
            log("probe_step", **step)
            return step

        def failed(step):
            """Reason this step ends the plan, or None when it settled as expected."""
            if step["ack"] == "rejected":
                return "command_rejected"
            if step.get("command_result") not in (0, None):
                return "device_error_result"
            if step["outcome"] != "settled":
                return step["outcome"]
            if step.get("matches_assumed_direction") is False:
                return "unexpected_direction"
            return None

        if axis == "pan":
            for label, direction in PLAN:
                log("PROBE_STEP_READY", step=label, awaiting="stdin_STEP", steps_done=len(report["steps"]))
                if not await await_confirm(view, confirm, "STEP", limits.confirm_seconds, deadline):
                    report["stopped"] = "step_not_confirmed"
                    break
                if not await wait_still(view, 2, limits.still_seconds):
                    report["stopped"] = "view_not_still_before_step"
                    break
                reason = failed(await step_once(label, direction))
                if reason:
                    report["stopped"] = reason
                    break
            complete = len(report["steps"]) == len(PLAN) and report["stopped"] is None
        else:
            # START authorised the whole bounded plan: up until the view stops moving, then back down.
            moved, sizes = 0, []
            report["upper_limit_reached"] = False
            for _ in range(limits.max_up_steps):
                if not await wait_still(view, 2, limits.still_seconds):
                    report["stopped"] = "view_not_still_before_step"
                    break
                step = await step_once("up", DIRECTIONS["up"])
                if (step["ack"] == "accepted" and step.get("command_result") in (0, None)
                        and step["outcome"] == "no_measurable_motion" and step["unknown_frames"] == 0):
                    # Accepted, every frame measurable and still: the mechanical upper limit.
                    report["upper_limit_reached"] = True
                    break
                reason = failed(step)
                if reason:
                    report["stopped"] = reason
                    break
                moved += 1
                sizes.append(abs(step["direct_shift"][1]))
            report["up_steps_moved"] = moved
            report["tilt_step_sizes"] = [round(v, 4) for v in sizes]
            if sizes:
                ordered = sorted(sizes)
                report["tilt_step_median"] = round(ordered[len(ordered) // 2], 4)
                # A shorter last step means the limit stopped it part-way; the reverse step may overshoot.
                report["partial_last_step"] = len(sizes) > 1 and sizes[-1] < .6 * report["tilt_step_median"]
            if report["stopped"] is None:
                for _ in range(moved):
                    if not await wait_still(view, 2, limits.still_seconds):
                        report["stopped"] = "view_not_still_before_step"
                        break
                    reason = failed(await step_once("down", DIRECTIONS["down"]))
                    if reason:
                        report["stopped"] = reason
                        break
            complete = report["stopped"] is None
        if complete and report["steps"]:
            residual = compare(baseline, view.last_small)
            report["residual_shift"] = residual
            report["returned_to_start"] = (residual is not None
                                           and max(abs(residual[0]), abs(residual[1])) <= limits.returned_tolerance)
        elif complete:
            report["returned_to_start"] = None  # nothing moved
        return report
    except Exception as error:
        report["stopped"] = f"error:{type(error).__name__}"
        raise
    finally:
        client.extra_commands = frozenset()
        try:
            if view:
                await view.close()
        finally:
            if token:
                await guarded_stop(client, marker, config.serial, token)
            try:
                report["stream_running_after"] = await status(client, config.serial)
                report["ownership_marker_present"] = marker.exists()
            except Exception as error:
                report["cleanup"] = f"unverified:{type(error).__name__}"
            report["native_events_during_probe"] = native_events
            if report.get("upper_limit_reached") and report.get("up_steps_moved") == 0 and report["stopped"] is None:
                report["position"] = "unchanged_already_at_upper_limit"
            elif report.get("returned_to_start"):
                report["position"] = "start_verified"
            elif report["pan_commands_sent"] == 0:
                report["position"] = "no_command_sent"
            else:
                report["position"] = "offset_or_unknown_check_in_owner_app"
            report["automatic_ptz"] = False
            log("probe_complete", **report)
            await client.close()
