"""Bounded, observation-only Eufy WebSocket client."""
import asyncio
import fcntl
import hashlib
import json
import math
import os
import random
import re
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import websockets

ALLOWED_COMMANDS = {
    "set_api_schema", "start_listening", "driver.is_connected",
    "driver.is_push_connected", "device.get_properties",
    "device.get_properties_metadata", "device.get_commands",
    "station.get_properties", "station.get_properties_metadata",
    "station.get_commands", "station.is_connected",
    "device.is_livestreaming", "device.start_livestream",
    "device.stop_livestream",
}
WAKE_EVENTS = {"motion detected", "pet detected", "dog detected", "cat detected", "radar motion detected"}
# Native classifications recorded as trigger context only; they never wake a session.
# T8170 and T8173 detection types are human, vehicle and all-other-motion (no pet type).
CONTEXT_EVENTS = WAKE_EVENTS | {"person detected", "stranger person detected", "vehicle detected"}
SERIAL_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def log(kind, **fields):
    # Callers must only pass fixed, allowlisted fields; never raw WS payloads.
    print(json.dumps({"time": round(time.time(), 3), "kind": kind, **fields}, separators=(",", ":")), flush=True)


class Journal:
    """Durable per-session summaries on the data mount; container logs vanish when it is recreated.

    One JSON line per session (fixed, allowlisted fields only — never images, serials or raw
    payloads). Rotates to `sessions.jsonl.1` past `limit` bytes, keeping one old file.
    """
    def __init__(self, directory, limit=2_000_000):
        self.path = Path(directory) / "sessions.jsonl"
        self.limit = limit

    def write(self, kind, **fields):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > self.limit:
                os.replace(self.path, self.path.with_suffix(".jsonl.1"))
            with self.path.open("a") as file:
                file.write(json.dumps({"time": round(time.time(), 3), "kind": kind, **fields},
                                      separators=(",", ":")) + "\n")
        except OSError as error:
            log("journal_unwritable", error=type(error).__name__)  # never fails a session


class ProtocolError(RuntimeError):
    pass


class CommandRejected(ProtocolError):
    """An explicit negative result from WS, distinct from an ambiguous response."""


class RunLock:
    """Serialize local collectors sharing the data mount; never an external lease."""
    def __init__(self, directory):
        self.path = Path(directory) / ".observer.lock"
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            self.file = None
            raise ProtocolError("Another local observer owns this data directory") from None
        return self

    def __exit__(self, *args):
        if self.file:
            self.file.close()
            self.file = None


@dataclass
class Config:
    url: str
    serial: str
    data_dir: Path
    mode: str = "observe"
    session_seconds: float = 30
    initial_target_seconds: float = 12
    target_loss_seconds: float = 8
    cooldown_seconds: float = 30
    command_seconds: float = 8
    frame_seconds: float = 8
    startup_seconds: float = 12
    # Was 16: S340 sessions dropped chunks in 32 of 118, and the E42's 1080p stream dropped 10 in
    # its first 10 s (2026-11-24), stalling HEVC decode until the next keyframe.
    max_video_chunks: int = 64
    schema: int = 13
    model_path: str = "/data/models/megadetector-v6-yolov10-c.pt"
    confidence: float = 0.1    # model output floor; tracker hysteresis (tracking.py) decides what counts
    warm_model: bool = True    # persistent run keeps one loaded model across sessions (~1 GB resident)
    min_battery: float = 25    # skip a session (no stream) below this battery percent; unknown never blocks
    follow: bool = False       # closed-loop pan following (follow.py); off unless explicitly enabled
    # At night (infrared frames) stopping our stream switched the IR light off for the rest of Eufy's
    # recording (2026-10-07 05:12:23), so keep the stream until this long after it started.
    night_hold_seconds: float = 130
    # Measured by `ptz-probe` (pan) on this camera; 0 uses the model's measured default (follow.PAN_STEPS)
    # and keeps following off for a model without one.
    pan_step: float = 0           # content shift per pan step, normalized image width
    device_model: str = None      # set once the selected device is identified
    # Measured by `ptz-probe --axis tilt`; 0 keeps tilt following off.
    tilt_step: float = 0          # content shift per tilt step, normalized image height
    tilt_up_steps: int = 0        # steps from the start view up to the mechanical upper limit
    max_steps: int = 16           # forward follow steps per session (StepPolicy.max_steps)
    max_net_steps: int = 5        # furthest excursion from the start view, per axis
    stop_confirm_seconds: float = 5  # how long to wait for an acknowledged stop to show as stopped

    @classmethod
    def from_env(cls, require_serial=True):
        serial = os.environ.get("WILDLIFE_DEVICE_SERIAL", "")
        if (require_serial or serial) and not SERIAL_RE.fullmatch(serial):
            raise ValueError("WILDLIFE_DEVICE_SERIAL must be a selected device serial")
        mode = os.environ.get("WILDLIFE_MODE", "observe")
        if mode not in {"listen", "observe"}:
            raise ValueError("Only listen and observe modes exist")
        url = os.environ.get("WILDLIFE_WS_URL", "ws://eufy-security-ws:3000")
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("WILDLIFE_WS_URL must use ws:// or wss://")
        config = cls(url=url, serial=serial, data_dir=Path(os.environ.get("WILDLIFE_DATA_DIR", "/data")), mode=mode)
        for field in ("session_seconds", "initial_target_seconds", "target_loss_seconds", "cooldown_seconds", "command_seconds", "frame_seconds", "startup_seconds", "confidence"):
            value = os.environ.get("WILDLIFE_" + field.upper())
            if value is not None:
                parsed = float(value)
                if not math.isfinite(parsed) or parsed <= 0 or parsed > (300 if field != "confidence" else 1):
                    raise ValueError(f"Invalid {field}")
                setattr(config, field, parsed)
        warm = os.environ.get("WILDLIFE_WARM_MODEL", "1")
        if warm not in {"0", "1"}:
            raise ValueError("WILDLIFE_WARM_MODEL must be 0 or 1")
        config.warm_model = warm == "1"
        follow = os.environ.get("WILDLIFE_FOLLOW", "0")
        if follow not in {"0", "1"}:
            raise ValueError("WILDLIFE_FOLLOW must be 0 or 1")
        config.follow = follow == "1"
        hold = os.environ.get("WILDLIFE_NIGHT_HOLD_SECONDS")
        if hold is not None:
            parsed = float(hold)
            if not math.isfinite(parsed) or not 0 <= parsed <= 300:
                raise ValueError("Invalid night_hold_seconds")
            config.night_hold_seconds = parsed
        pan = os.environ.get("WILDLIFE_PAN_STEP")
        if pan is not None:
            parsed = float(pan)
            if not math.isfinite(parsed) or not 0 <= parsed < .5:
                raise ValueError("Invalid pan_step")
            config.pan_step = parsed
        tilt = os.environ.get("WILDLIFE_TILT_STEP")
        if tilt is not None:
            parsed = float(tilt)
            if not math.isfinite(parsed) or not 0 <= parsed < .5:
                raise ValueError("Invalid tilt_step")
            config.tilt_step = parsed
        up = os.environ.get("WILDLIFE_TILT_UP_STEPS")
        if up is not None:
            if not up.isdigit() or not 0 <= int(up) <= 8:
                raise ValueError("Invalid tilt_up_steps")
            config.tilt_up_steps = int(up)
        for field, upper in (("max_steps", 32), ("max_net_steps", 8)):
            value = os.environ.get("WILDLIFE_" + field.upper())
            if value is not None:
                if not value.isdigit() or not 1 <= int(value) <= upper:
                    raise ValueError(f"Invalid {field}")
                setattr(config, field, int(value))
        if config.max_net_steps > config.max_steps:
            raise ValueError("max_net_steps cannot exceed max_steps")
        battery = os.environ.get("WILDLIFE_MIN_BATTERY")
        if battery is not None:
            parsed = float(battery)
            if not math.isfinite(parsed) or not 0 <= parsed <= 100:
                raise ValueError("Invalid min_battery")
            config.min_battery = parsed
        return config


class Marker:
    def __init__(self, directory: Path):
        self.path = directory / "stream-ownership.json"
        directory.mkdir(parents=True, exist_ok=True)

    def exists(self):
        return self.path.exists()

    def read(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def target(serial):
        return hashlib.sha256(serial.encode()).hexdigest()

    def write(self, state: str, token=None, serial=None):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as file:
            json.dump({"state": state, "token": token,
                       "target": self.target(serial) if serial else None, "created": time.time()}, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def clear(self):
        self.path.unlink(missing_ok=True)
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class EufySocket:
    def __init__(self, ws, config: Config):
        self.ws = ws
        self.config = config
        self.pending = {}
        self.events = asyncio.Queue(maxsize=64)
        self.video = asyncio.Queue(maxsize=config.max_video_chunks)
        self.reader_task = None
        self.version = None
        self.state = None
        self.closed = False
        self.video_dropped = 0
        self.video_enabled = False
        self.stream_signals = {}
        self.device_events = deque(maxlen=64)  # (monotonic, name) of selected-device state=true events
        # Empty for every observer path. Only the supervised ptz-probe widens it, after its own gates.
        self.extra_commands = frozenset()

    def clear_video(self):
        while not self.video.empty():
            self.video.get_nowait()

    async def open(self):
        greeting = json.loads(await asyncio.wait_for(self.ws.recv(), self.config.command_seconds))
        if greeting.get("type") != "version":
            raise ProtocolError("Missing WS version greeting")
        minimum, maximum = greeting.get("minSchemaVersion"), greeting.get("maxSchemaVersion")
        if not isinstance(minimum, int) or not isinstance(maximum, int) or not minimum <= self.config.schema <= maximum:
            raise ProtocolError("WS schema 13 unavailable")
        self.version = {key: greeting.get(key) for key in ("driverVersion", "serverVersion", "minSchemaVersion", "maxSchemaVersion")}
        self.reader_task = asyncio.create_task(self._reader())
        await self.command("set_api_schema", schemaVersion=self.config.schema)
        response = await self.command("start_listening")
        self.state = response.get("state", {})
        return self

    async def _reader(self):
        try:
            async for raw in self.ws:
                if isinstance(raw, bytes) or len(raw) > 8_000_000:
                    continue
                packet = json.loads(raw)
                if packet.get("type") == "result":
                    future = self.pending.pop(str(packet.get("messageId")), None)
                    if future and not future.done():
                        future.set_result(packet)
                    continue
                if packet.get("type") != "event" or not isinstance(packet.get("event"), dict):
                    continue
                event = packet["event"]
                if (self.video_enabled and event.get("serialNumber") == self.config.serial
                        and event.get("event") in {"livestream started", "livestream stopped"}):
                    name = event["event"]
                    self.stream_signals[name] = self.stream_signals.get(name, 0) + 1
                if event.get("source") == "device" and event.get("event") == "livestream video data":
                    if self.video_enabled and event.get("serialNumber") == self.config.serial:
                        try:
                            chunk = decode_chunk(event.get("buffer"))
                            if self.video.full():
                                self.video.get_nowait()
                                self.video_dropped += 1
                            self.video.put_nowait((chunk, event.get("metadata") or {}))
                        except (ValueError, TypeError):
                            log("invalid_video_chunk")
                    continue
                if (event.get("source") == "device" and event.get("serialNumber") == self.config.serial
                        and event.get("event") in CONTEXT_EVENTS and event.get("state") is True):
                    self.device_events.append((time.monotonic(), event["event"]))
                if self.events.full():
                    self.events.get_nowait()
                self.events.put_nowait(event)
        except Exception as error:
            log("ws_reader_error", error=type(error).__name__)
        finally:
            self.closed = True
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("WS closed"))
            self.pending.clear()

    async def command(self, command, **arguments):
        if command not in ALLOWED_COMMANDS and command not in self.extra_commands:
            raise ValueError("Command is outside observation allowlist")
        if command == "device.pan_and_tilt" and arguments.get("direction") not in (1, 2, 3, 4):
            raise ValueError("Only single LEFT/RIGHT/UP/DOWN steps exist; ROTATE360 is never sent")
        if self.closed:
            raise ConnectionError("WS closed")
        message_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[message_id] = future
        try:
            await self.ws.send(json.dumps({"messageId": message_id, "command": command, **arguments}))
            packet = await asyncio.wait_for(future, self.config.command_seconds)
            if packet.get("success") is False:
                raise CommandRejected(f"{command} rejected")
            if packet.get("success") is not True:
                raise ProtocolError("Malformed command acknowledgement")
            result = packet.get("result", {})
            if not isinstance(result, dict):
                raise ProtocolError("Malformed command result")
            return result
        finally:
            self.pending.pop(message_id, None)

    async def close(self):
        await self.ws.close()
        if self.reader_task:
            try:
                await asyncio.wait_for(self.reader_task, 2)
            except asyncio.TimeoutError:
                self.reader_task.cancel()


def decode_chunk(buffer):
    # Eufy WS serializes Node Buffer as {type: Buffer, data: [byte,...]}.
    if not isinstance(buffer, dict) or buffer.get("type") != "Buffer" or not isinstance(buffer.get("data"), list):
        raise ValueError("Unsupported video buffer form")
    data = buffer["data"]
    if len(data) > 2_000_000:
        raise ValueError("Oversize video chunk")
    return bytes(data)


def wake_event(event, serial):
    return (event.get("source") == "device" and event.get("serialNumber") == serial
            and event.get("event") in WAKE_EVENTS and event.get("state") is True)


def trigger_context(history, since):
    """Native event names seen since a monotonic time; a person event marks a human-associated trigger.

    motion_only means no native person/vehicle classification arrived. It is a
    candidate for non-human motion (animal, vegetation, light), not animal proof.
    """
    names = sorted({name for at, name in history if at >= since})
    person = any(name in {"person detected", "stranger person detected"} for name in names)
    vehicle = "vehicle detected" in names
    category = "person_associated" if person else "vehicle_associated" if vehicle else "motion_only"
    return {"native_events": names, "category": category}


def metadata_summary(state):
    """Inventory output with only counts; never serials or names."""
    return {"devices": len(state.get("devices") or []), "stations": len(state.get("stations") or []),
            "driver_connected": (state.get("driver") or {}).get("connected"),
            "push_connected": (state.get("driver") or {}).get("pushConnected")}


async def connect(config):
    ws = await websockets.connect(config.url, max_size=8_000_000, ping_interval=20, open_timeout=config.command_seconds)
    client = EufySocket(ws, config)
    try:
        return await client.open()
    except BaseException:
        await client.close()
        raise


async def status(client, serial):
    result = await client.command("device.is_livestreaming", serialNumber=serial)
    if not isinstance(result.get("livestreaming"), bool):
        raise ProtocolError("Unknown livestream status")
    return result["livestreaming"]


async def guarded_start(client, marker, serial):
    if marker.exists():
        raise ProtocolError("Unreconciled prior stream ownership")
    if await status(client, serial):
        log("skip_existing_stream")
        return None
    client.clear_video()
    client.stream_signals.clear()
    client.video_enabled = True
    token = uuid.uuid4().hex
    # Marker before send: acknowledgement timeout is still an uncertain start.
    try:
        marker.write("start_pending", token, serial)
    except BaseException:
        client.video_enabled = False
        raise
    try:
        await client.command("device.start_livestream", serialNumber=serial)
    except CommandRejected:
        # Definitive command rejection. Recheck before clearing; leave marker on ambiguity.
        try:
            if not await status(client, serial):
                marker.clear()
        except Exception:
            pass
        client.video_enabled = False
        raise
    except BaseException:
        client.video_enabled = False
        raise
    marker.write("owned", token, serial)
    return token


async def guarded_stop(client, marker, serial, token):
    client.video_enabled = False
    client.clear_video()
    record = marker.read()
    if (not token or not record or record.get("state") != "owned"
            or record.get("token") != token or record.get("target") != Marker.target(serial)):
        return
    try:
        if await status(client, serial):
            await client.command("device.stop_livestream", serialNumber=serial)
        # stop_livestream is asynchronous ({async: true}); the state can lag the acknowledgement.
        # 2026-10-11: an immediate check still saw the stream and left the marker for two days.
        until = time.monotonic() + client.config.stop_confirm_seconds
        while True:
            if not await status(client, serial):
                marker.clear()
                log("owned_stream_cleaned")
                return
            if time.monotonic() >= until:
                break
            await asyncio.sleep(.3)
        log("stream_cleanup_uncertain", error="still_streaming")
    except Exception as error:
        log("stream_cleanup_uncertain", error=type(error).__name__)


async def reconcile(client, marker, serial):
    if not marker.exists():
        return
    record = marker.read()
    if not record or record.get("target") != Marker.target(serial):
        raise ProtocolError("Ownership marker belongs to another or unknown device")
    if not await status(client, serial):
        marker.clear()
        log("ownership_reconciled_stopped")
    else:
        log("ownership_reconciliation_required")
