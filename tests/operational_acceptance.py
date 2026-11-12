"""Offline Linux-container acceptance: loopback fake WS, real CLI/model/FFmpeg.

Run with --network none, 2 CPUs/4 GiB and a read-only model mount. No real
camera addresses, credentials, images or persistent ownership directory used.
"""
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import websockets


def report(kind, **fields):
    print(json.dumps({"source": "synthetic_operational", "kind": kind, **fields}), flush=True)


def container_cpu_seconds():
    fields = dict(line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines())
    return int(fields["usage_usec"]) / 1_000_000


async def eventually(predicate, seconds=15):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        if predicate():
            return
        await asyncio.sleep(.05)
    raise AssertionError("Condition deadline")


class Camera:
    def __init__(self, video):
        self.video = video
        self.clients = set()
        self.streaming = False
        self.commands = []
        self.send_video = True

    async def serve(self, ws):
        self.clients.add(ws)
        await ws.send(json.dumps({"type": "version", "driverVersion": "synthetic",
            "serverVersion": "synthetic", "minSchemaVersion": 0, "maxSchemaVersion": 21}))
        try:
            async for raw in ws:
                p = json.loads(raw)
                command = p["command"]
                self.commands.append(command)
                result = {}
                if command == "start_listening":
                    result = {"state": {"devices": ["synthetic123"], "stations": [],
                        "driver": {"connected": True, "pushConnected": True}}}
                elif command == "device.get_properties":
                    result = {"properties": {"model": "T8170"}}
                elif command == "device.is_livestreaming":
                    result = {"livestreaming": self.streaming}
                elif command == "device.start_livestream":
                    self.streaming = True
                elif command == "device.stop_livestream":
                    self.streaming = False
                await ws.send(json.dumps({"type": "result", "messageId": p["messageId"],
                    "success": True, "result": result}))
                if command == "device.start_livestream" and self.send_video:
                    await self.event(ws, "livestream started")
                    await self.event(ws, "livestream video data", buffer={"type": "Buffer", "data": list(self.video)},
                        metadata={"videoCodec": "H264", "videoWidth": 640, "videoHeight": 360})
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)

    async def event(self, ws, name, **extra):
        await ws.send(json.dumps({"type": "event", "event": {"source": "device",
            "serialNumber": "synthetic123", "event": name, **extra}}))

    async def wake(self):
        for ws in list(self.clients):
            await self.event(ws, "motion detected", state=True)


class Process:
    def __init__(self, url, directory, **options):
        self.env = {**os.environ, "WILDLIFE_ENABLED": "1", "WILDLIFE_WS_URL": url,
            "WILDLIFE_DEVICE_SERIAL": "synthetic123", "WILDLIFE_DATA_DIR": str(directory),
            "WILDLIFE_MODEL_PATH": os.environ.get("ACCEPTANCE_MODEL", "/models/megadetector-v6-yolov10-c.pt"), "WILDLIFE_SESSION_SECONDS": "3",
            "WILDLIFE_INITIAL_TARGET_SECONDS": "2", "WILDLIFE_COOLDOWN_SECONDS": ".1",
            "WILDLIFE_COMMAND_SECONDS": "1", "WILDLIFE_STARTUP_SECONDS": "1",
            "WILDLIFE_FRAME_SECONDS": "2",
            # The synthetic black video looks like infrared; hold is checked separately below.
            "WILDLIFE_NIGHT_HOLD_SECONDS": "0", **options}
        self.records = []
        self.raw = []
        self.peak_rss = 0
        self.peak_cgroup = 0
        self.proc = None

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(sys.executable, "-m", "wildlife_tracker", "run",
            env=self.env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            start_new_session=True)
        self.reader = asyncio.create_task(self.read())
        self.sampler = asyncio.create_task(self.sample())
        return self

    async def read(self):
        async for line in self.proc.stdout:
            text = line.decode(errors="replace")
            try:
                self.records.append(json.loads(text))
            except ValueError:
                self.raw.append(text)

    def count(self, kind):
        return sum(r.get("kind") == kind for r in self.records)

    async def sample(self):
        while self.proc.returncode is None:
            try:
                parent = psutil.Process(self.proc.pid)
                total = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True) if p.is_running())
                self.peak_rss = max(self.peak_rss, total)
                current = Path("/sys/fs/cgroup/memory.current")
                if current.exists():
                    self.peak_cgroup = max(self.peak_cgroup, int(current.read_text()))
            except psutil.Error:
                pass
            await asyncio.sleep(.1)

    async def stop(self, hard=False):
        if self.proc.returncode is None:
            if hard:
                # Simulate whole-container abrupt loss, including model/FFmpeg.
                os.killpg(self.proc.pid, signal.SIGKILL)
            else:
                self.proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(self.proc.wait(), 40)
        await self.reader
        await self.sampler


async def main(warm):
    """warm=False: session-scoped model (original gates). warm=True: one persistent, reused worker."""
    report("mode", warm_model=warm)
    encoded = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "color=c=black:size=640x360:rate=6:duration=20", "-c:v", "libx264", "-threads", "1",
        "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
    camera = Camera(encoded)
    processes = []
    with tempfile.TemporaryDirectory(prefix="wildlife-offline-") as tmp:
        directory = Path(tmp)
        marker = directory / "stream-ownership.json"
        async with websockets.serve(camera.serve, "127.0.0.1", 0) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async def launch(**options):
                p = await Process(url, directory, WILDLIFE_WARM_MODEL="1" if warm else "0", **options).start()
                processes.append(p)
                return p

            try:
                p = await launch()
                await eventually(lambda: p.count("selected_device") == 1)
                if warm:
                    await eventually(lambda: p.count("model_ready_warm") == 1, 90)  # idle sample after the one cold load
                proc = psutil.Process(p.proc.pid)
                def tree():
                    return [proc] + proc.children(recursive=True)
                t0, cpu0 = time.monotonic(), sum(sum(c.cpu_times()[:2]) for c in tree())
                await asyncio.sleep(15)
                elapsed = time.monotonic() - t0
                idle_cpu = 100 * (sum(sum(c.cpu_times()[:2]) for c in tree()) - cpu0) / elapsed
                report("idle", seconds=round(elapsed, 2), cpu_percent_one_core=round(idle_cpu, 3),
                    rss_mib=round(proc.memory_info().rss / 2**20, 2), children=len(proc.children(recursive=True)),
                    tree_rss_mib=round(sum(c.memory_info().rss for c in tree()) / 2**20, 2))
                warm_worker = None

                # A second real CLI process cannot acquire the local lock or connect.
                command_count = len(camera.commands)
                competing = await launch()
                await asyncio.wait_for(competing.proc.wait(), 5)
                await competing.stop()
                assert competing.proc.returncode == 1 and len(camera.commands) == command_count
                report("single_instance", passed=True)

                for attempt in range(3):
                    active_begin, cpu_begin = time.monotonic(), container_cpu_seconds()
                    await camera.wake()
                    await eventually(lambda: p.count("session_complete") == attempt + 1, 30)
                    await eventually(lambda: sum(r.get("kind") == "state" and r.get("value") == "IDLE" for r in p.records) >= attempt + 1)
                    assert not marker.exists() and not camera.streaming
                    result = [r for r in p.records if r.get("kind") == "session_complete"][-1]
                    assert result["inferences"] > 0
                    await asyncio.sleep(.25)
                    # The multiprocessing resource tracker may remain. Cold mode: no heavy worker
                    # may remain. Warm mode: exactly one, and the same process every session.
                    children = proc.children(recursive=True)
                    heavy = [c for c in children if "resource_tracker" not in " ".join(c.cmdline())]
                    if warm:
                        assert len(heavy) == 1, heavy
                        warm_worker = warm_worker or heavy[0].pid
                        assert heavy[0].pid == warm_worker
                        assert result.get("warm_model") is True
                    else:
                        assert not heavy
                    report("session", index=attempt + 1, frames=result["frames"], inferences=result["inferences"],
                        idle_rss_mib=round(proc.memory_info().rss / 2**20, 2), heavy_children=len(heavy),
                        cpu_percent_one_core=round(100 * (container_cpu_seconds() - cpu_begin) / (time.monotonic() - active_begin), 2))
                report("resources", tree_peak_rss_mib=round(p.peak_rss / 2**20, 2),
                    container_peak_current_mib=round(p.peak_cgroup / 2**20, 2))

                camera.send_video = False
                await camera.wake()
                await eventually(lambda: p.count("session_failed") == 1, 20)
                assert not camera.streaming and not marker.exists()
                report("ack_without_video", failed_bounded=True, owned_stream_stopped=True)
                await p.stop()

                # Infrared-looking video: the owned stream is held until night_hold after it started.
                camera.send_video = True
                # Frame deadline above the hold: the synthetic camera sends one burst, a real one keeps streaming.
                p = await launch(WILDLIFE_NIGHT_HOLD_SECONDS="4", WILDLIFE_FRAME_SECONDS="8")
                await eventually(lambda: p.count("selected_device") == 1)
                await camera.wake()
                await eventually(lambda: p.count("session_complete") == 1, 30)
                started = [r for r in p.records if r.get("kind") == "stream_start_ack"][-1]["time"]
                result = [r for r in p.records if r.get("kind") == "session_complete"][-1]
                assert result["night"] is True and result["time"] - started >= 3.5
                assert not camera.streaming and not marker.exists()
                await p.stop()
                report("night_hold", held_seconds=round(result["time"] - started, 2), stream_stopped=True)

                # SIGTERM while real model + decoder are active must release all children and stream.
                camera.send_video = True
                p = await launch(WILDLIFE_INITIAL_TARGET_SECONDS="30", WILDLIFE_SESSION_SECONDS="30")
                await eventually(lambda: p.count("selected_device") == 1)
                await camera.wake()
                await eventually(lambda: p.count("stream_metadata") == 1, 20)
                child_ids = [c.pid for c in psutil.Process(p.proc.pid).children(recursive=True)]
                began = time.monotonic()
                await p.stop()
                await eventually(lambda: not any(psutil.pid_exists(pid) for pid in child_ids), 5)
                assert not camera.streaming and not marker.exists() and p.proc.returncode == 0
                report("sigterm_active", seconds=round(time.monotonic() - began, 3), stream_stopped=True, children_reaped=True)

                # Disconnect during active stream retains uncertainty; reconnect never takes over.
                p = await launch(WILDLIFE_INITIAL_TARGET_SECONDS="30", WILDLIFE_SESSION_SECONDS="30")
                await eventually(lambda: p.count("selected_device") == 1)
                await camera.wake()
                await eventually(lambda: p.count("stream_metadata") == 1, 20)
                starts = camera.commands.count("device.start_livestream")
                stops = camera.commands.count("device.stop_livestream")
                for ws in list(camera.clients):
                    await ws.close()
                await eventually(lambda: p.count("selected_device") >= 2, 10)
                await camera.wake()
                await eventually(lambda: p.count("wake_blocked_uncertain_ownership") >= 1)
                assert marker.exists() and camera.streaming
                assert camera.commands.count("device.start_livestream") == starts
                assert camera.commands.count("device.stop_livestream") == stops
                await p.stop()
                report("disconnect_reconnect", uncertainty_preserved=True, no_takeover=True)

                # A fresh invocation also refuses a prior process's running stream.
                p = await launch()
                await eventually(lambda: p.count("ownership_reconciliation_required") >= 1)
                await camera.wake()
                await eventually(lambda: p.count("wake_blocked_uncertain_ownership") >= 1)
                assert camera.commands.count("device.start_livestream") == starts
                assert camera.commands.count("device.stop_livestream") == stops
                await p.stop()
                camera.streaming = False  # Synthetic device later stops by itself; never a service stop.
                p = await launch()
                await eventually(lambda: p.count("ownership_reconciled_stopped") == 1)
                assert not marker.exists()
                report("restart_reconciliation", running_blocks=True, stopped_clears=True)

                # Abrupt loss releases the OS lock but leaves the durable marker.
                await camera.wake()
                await eventually(lambda: p.count("stream_metadata") == 1, 20)
                await p.stop(hard=True)
                assert marker.exists() and camera.streaming
                starts = camera.commands.count("device.start_livestream")
                stops = camera.commands.count("device.stop_livestream")
                p = await launch()
                await eventually(lambda: p.count("ownership_reconciliation_required") == 1)
                await camera.wake()
                await eventually(lambda: p.count("wake_blocked_uncertain_ownership") >= 1)
                assert camera.commands.count("device.start_livestream") == starts
                assert camera.commands.count("device.stop_livestream") == stops
                await p.stop()
                report("abrupt_loss_restart", marker_preserved=True, no_takeover=True, lock_reacquired=True)
                allowed = {"set_api_schema", "start_listening", "device.get_properties", "device.is_livestreaming",
                    "device.start_livestream", "device.stop_livestream"}
                assert set(camera.commands) <= allowed
                report("complete", passed=True, camera_commands=sorted(set(camera.commands)))
            except Exception:
                report("failure_context", records=p.records[-20:], local_output=p.raw[-5:])
                raise
            finally:
                for p in processes:
                    if p.proc.returncode is None:
                        await p.stop(hard=True)


if __name__ == "__main__":
    asyncio.run(main("--warm" in sys.argv))
