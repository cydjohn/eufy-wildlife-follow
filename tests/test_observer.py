import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("websockets", types.SimpleNamespace(connect=None))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from wildlife_tracker.core import Config, EufySocket, Marker, ProtocolError, guarded_start, guarded_stop, reconcile, wake_event
from wildlife_tracker.__main__ import collect_stream, main, run, selected_device
from wildlife_tracker.vision import Decoder, FRAME_BYTES, LABELS, PROMPTS, model_labels


class FakeWS:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait(json.dumps({"type": "version", "driverVersion": "4.1.0", "serverVersion": "3.1.0", "minSchemaVersion": 0, "maxSchemaVersion": 21}))
        self.commands = []
        self.stream = False
        self.closed = False
        self.video_payload = None
        self.start_ack = True
        self.model = "T8170"
        self.battery = None
        self.stop_lag = 0        # seconds the stream still reports running after an acknowledged stop
        self.stops_at = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def recv(self):
        return await self.incoming.get()

    async def send(self, raw):
        packet = json.loads(raw)
        command = packet["command"]
        self.commands.append(command)
        result = {}
        if command == "start_listening":
            result = {"state": {"devices": ["selected123"], "stations": [], "driver": {"connected": True}}}
        if command == "device.is_livestreaming":
            if self.stops_at is not None and asyncio.get_running_loop().time() >= self.stops_at:
                self.stream, self.stops_at = False, None
            result = {"livestreaming": self.stream}
        if command == "device.get_properties":
            result = {"properties": {"model": {"value": self.model}}}
            if self.battery is not None:
                result["properties"]["battery"] = {"value": self.battery}
        if command == "device.get_properties_metadata":
            result = {"properties": {"motionTracking": {"type": "boolean"}}}
        if command == "device.get_commands":
            result = {"commands": ["startLivestream"]}
        if command == "device.start_livestream":
            if self.start_ack is not False:
                self.stream = True
            if self.video_payload is not None and self.start_ack is not False:
                self.incoming.put_nowait(json.dumps({"type": "event", "event": {"source": "device",
                    "serialNumber": "selected123", "event": "livestream video data",
                    "buffer": {"type": "Buffer", "data": list(self.video_payload)},
                    "metadata": {"videoCodec": "H264", "videoWidth": 640, "videoHeight": 360}}}))
        if command == "device.stop_livestream":
            if self.stop_lag:
                self.stops_at = asyncio.get_running_loop().time() + self.stop_lag  # asynchronous stop
            else:
                self.stream = False
        reply = {"type": "result", "messageId": packet["messageId"], "result": result}
        if command != "device.start_livestream" or self.start_ack is not None:
            reply["success"] = self.start_ack if command == "device.start_livestream" else True
        self.incoming.put_nowait(json.dumps(reply))

    async def close(self):
        self.closed = True
        self.incoming.put_nowait(None)


class ObserverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Config("ws://fake", "selected123", pathlib.Path(self.tmp.name))
        self.ws = FakeWS()
        self.client = await EufySocket(self.ws, self.config).open()
        self.marker = Marker(self.config.data_dir)

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def test_stream_ownership_and_cleanup(self):
        token = await guarded_start(self.client, self.marker, self.config.serial)
        self.assertTrue(token)
        self.assertTrue(self.marker.exists())
        await guarded_stop(self.client, self.marker, self.config.serial, token)
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.ws.stream)
        self.assertEqual(self.ws.commands.count("device.start_livestream"), 1)
        self.assertEqual(self.ws.commands.count("device.stop_livestream"), 1)
        self.assertFalse(any("pan_and_tilt" in c or "set_property" in c or "disconnect" in c for c in self.ws.commands))

    async def test_asynchronous_stop_is_confirmed_before_declaring_uncertain(self):
        self.ws.stop_lag = .6
        self.config.stop_confirm_seconds = 3
        token = await guarded_start(self.client, self.marker, self.config.serial)
        await guarded_stop(self.client, self.marker, self.config.serial, token)
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.ws.stream)

    async def test_stream_that_never_stops_stays_uncertain(self):
        self.ws.stop_lag = 100
        self.config.stop_confirm_seconds = .5
        token = await guarded_start(self.client, self.marker, self.config.serial)
        with patch("wildlife_tracker.core.log") as logged:
            await guarded_stop(self.client, self.marker, self.config.serial, token)
        self.assertTrue(self.marker.exists())
        self.assertIn(("stream_cleanup_uncertain", {"error": "still_streaming"}),
                      [(c.args[0], c.kwargs) for c in logged.call_args_list])

    async def test_existing_stream_never_stopped(self):
        self.ws.stream = True
        self.assertFalse(await guarded_start(self.client, self.marker, self.config.serial))
        await guarded_stop(self.client, self.marker, self.config.serial, None)
        self.assertNotIn("device.stop_livestream", self.ws.commands)
        self.assertTrue(self.ws.stream)

    async def test_uncertain_marker_blocks_start(self):
        self.marker.write("start_pending")
        with self.assertRaises(ProtocolError):
            await guarded_start(self.client, self.marker, self.config.serial)

    async def test_prior_marker_never_authorizes_stop(self):
        self.ws.stream = True
        self.marker.write("owned", "prior-process", self.config.serial)
        await guarded_stop(self.client, self.marker, self.config.serial, "other-process")
        self.assertTrue(self.ws.stream)
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_changed_target_cannot_clear_prior_marker(self):
        self.marker.write("start_pending", "prior-process", "different123")
        with self.assertRaises(ProtocolError):
            await reconcile(self.client, self.marker, self.config.serial)
        self.assertTrue(self.marker.exists())
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_non_s340_selection_fails_closed(self):
        self.ws.model = "T8214"
        with self.assertRaises(ProtocolError):
            await selected_device(self.client, self.config)
        self.assertNotIn("device.start_livestream", self.ws.commands)

    async def test_only_explicit_rejection_can_clear_pending_start(self):
        self.ws.start_ack = False
        with self.assertRaises(ProtocolError):
            await guarded_start(self.client, self.marker, self.config.serial)
        self.assertFalse(self.marker.exists())
        self.ws.start_ack = None
        with self.assertRaises(ProtocolError):
            await guarded_start(self.client, self.marker, self.config.serial)
        self.assertEqual(self.marker.read()["state"], "start_pending")
        self.assertTrue(self.ws.stream)

    async def test_cancelled_start_keeps_ambiguous_marker(self):
        original_send = self.ws.send
        async def delayed_start(raw):
            if json.loads(raw)["command"] == "device.start_livestream":
                self.ws.stream = True
                await asyncio.sleep(60)
            else:
                await original_send(raw)
        self.ws.send = delayed_start
        task = asyncio.create_task(guarded_start(self.client, self.marker, self.config.serial))
        await asyncio.sleep(.05)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.marker.read()["state"], "start_pending")
        await guarded_stop(self.client, self.marker, self.config.serial, "any-token")
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_cancelled_session_stops_only_its_stream(self):
        self.config.startup_seconds = 30
        task = asyncio.create_task(collect_stream(self.client, self.config))
        for _ in range(100):
            if "device.start_livestream" in self.ws.commands:
                break
            await asyncio.sleep(.01)
        self.assertIn("device.start_livestream", self.ws.commands)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.ws.commands.count("device.stop_livestream"), 1)

    async def test_disconnect_during_startup_cleans_owned_stream(self):
        self.config.startup_seconds = 30
        task = asyncio.create_task(collect_stream(self.client, self.config))
        for _ in range(100):
            if "device.start_livestream" in self.ws.commands:
                break
            await asyncio.sleep(.01)
        await self.ws.close()
        with self.assertRaises(ConnectionError):
            await task
        # The socket is gone, so cleanup cannot confirm a stop. The durable marker remains.
        self.assertTrue(self.marker.exists())
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_disconnect_during_active_decode_releases_worker(self):
        decoding = asyncio.Event()
        closed = asyncio.Event()
        class WaitingDecoder:
            fmt = "h264"
            def __init__(self, codec):
                assert codec == "H264"
            async def start(self):
                pass
            async def feed(self, chunk):
                pass
            async def frame(self, timeout):
                decoding.set()
                await asyncio.sleep(60)
            async def close(self):
                closed.set()
        self.ws.video_payload = b"video"
        self.config.session_seconds = 30
        with patch("wildlife_tracker.__main__.Decoder", WaitingDecoder):
            task = asyncio.create_task(collect_stream(self.client, self.config))
            await asyncio.wait_for(decoding.wait(), 2)
            await self.ws.close()
            with self.assertRaises(ConnectionError):
                await asyncio.wait_for(task, 2)
        self.assertTrue(closed.is_set())
        self.assertTrue(self.marker.exists())
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_unsolicited_idle_video_is_discarded(self):
        event = {"type": "event", "event": {"source": "device", "serialNumber": self.config.serial,
                 "event": "livestream video data", "buffer": {"type": "Buffer", "data": [1, 2]},
                 "metadata": {"videoCodec": "H264"}}}
        self.ws.incoming.put_nowait(json.dumps(event))
        await asyncio.sleep(.02)
        self.assertTrue(self.client.video.empty())
        token = await guarded_start(self.client, self.marker, self.config.serial)
        self.ws.incoming.put_nowait(json.dumps(event))
        await asyncio.sleep(.02)
        self.assertEqual(self.client.video.qsize(), 1)
        await guarded_stop(self.client, self.marker, self.config.serial, token)
        self.assertTrue(self.client.video.empty())

    async def test_synthetic_event_to_decode_inference_and_stop(self):
        # Synthetic WS event/video and a fake inference result; real FFmpeg decodes the H.264.
        # 6 s of input: FFmpeg holds back trailing frames, and 3 s gave only ~3 frames (the confirm threshold).
        self.ws.video_payload = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=12:duration=6", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
        self.config.session_seconds = 1.5
        self.config.cooldown_seconds = .01
        class FakeDetector:
            frames = 0
            async def start(self):
                pass
            def submit(self, frame):
                assert len(frame) == FRAME_BYTES
                self.frames += 1
            async def result(self, timeout):
                return [{"label": "deer", "score": .8, "area": .1, "center": [.8, .5]}]
            def alive(self):
                return True
            async def reset(self):
                pass
            def close(self):
                pass
        async def connect_existing(_):
            return self.client
        detector = FakeDetector()
        with patch("wildlife_tracker.__main__.connect", connect_existing), patch("wildlife_tracker.__main__.Detector", return_value=detector), \
             patch("wildlife_tracker.__main__.log") as recorded:
            task = asyncio.create_task(run(self.config, "run"))
            await asyncio.sleep(.01)
            self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {"source": "device",
                "serialNumber": "selected123", "event": "motion detected", "state": True}}))
            for _ in range(300):
                if "device.stop_livestream" in self.ws.commands and not self.marker.exists():
                    break
                await asyncio.sleep(.01)
            self.assertIn("device.stop_livestream", self.ws.commands)
            self.assertGreater(detector.frames, 0)
            self.assertFalse(self.ws.stream)
            self.assertFalse(self.marker.exists())
            decisions = [c.kwargs for c in recorded.call_args_list if c.args[0] == "tracking_dry_run"]
            self.assertTrue(any(d["action"] == "move" for d in decisions))
            self.assertTrue(all(d["control_enabled"] is False for d in decisions))
            self.assertTrue(all("pan" not in c and "preset" not in c and "set_property" not in c for c in self.ws.commands))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

    async def test_events_and_storm_queue_bound(self):
        self.assertTrue(wake_event({"source": "device", "serialNumber": "selected123", "event": "motion detected", "state": True}, "selected123"))
        self.assertFalse(wake_event({"source": "device", "serialNumber": "another", "event": "pet detected", "state": True}, "selected123"))
        for _ in range(90):
            self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {"source": "device", "serialNumber": "selected123", "event": "motion detected", "state": True}}))
        await asyncio.sleep(.05)
        self.assertEqual(self.client.events.qsize(), 64)

    async def test_schema_rejected(self):
        ws = FakeWS()
        ws.incoming = asyncio.Queue()
        ws.incoming.put_nowait(json.dumps({"type": "version", "minSchemaVersion": 14, "maxSchemaVersion": 21}))
        with self.assertRaises(ProtocolError):
            await EufySocket(ws, self.config).open()

    async def test_decoder_with_real_ffmpeg(self):
        encoded = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=12:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
        decoder = Decoder("H264")
        await decoder.start()
        try:
            decoder.process.stdin.write(encoded)
            await decoder.process.stdin.drain()
            decoder.process.stdin.close()
            frame = await decoder.frame(8)
            self.assertEqual(len(frame), FRAME_BYTES)
            self.assertGreater(len(set(frame)), 1)
        finally:
            await decoder.close()

    def test_prompt_coverage(self):
        self.assertEqual(set(LABELS), {"deer", "bobcat", "coyote", "raccoon", "rabbit", "squirrel", "cat", "dog"})

    def test_model_labels_keep_only_animals(self):
        self.assertEqual(model_labels(dict(enumerate(PROMPTS))), LABELS)
        self.assertEqual(model_labels({0: "animal", 1: "person", 2: "vehicle"}), ("animal", None, None))
        for names in ({0: "person"}, {0: "animal", 1: "vehicle", 2: "person"}, dict(enumerate(LABELS))):
            with self.assertRaises(ValueError):
                model_labels(names)


    async def test_decoder_shutdown_with_backpressured_output(self):
        encoded = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:size=640x360:rate=6:duration=20",
            "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
        decoder = Decoder("H264")
        await decoder.start()
        await decoder.feed(encoded)
        await asyncio.sleep(.2)  # Let raw stdout fill without consuming frames.
        await asyncio.wait_for(decoder.close(), 4)
        self.assertIsNone(decoder.process)

    def test_config_rejects_nonfinite_deadlines(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value), patch.dict(os.environ,
                    {"WILDLIFE_DEVICE_SERIAL": "selected123", "WILDLIFE_FRAME_SECONDS": value}, clear=True):
                with self.assertRaises(ValueError):
                    Config.from_env()

    def test_run_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {"WILDLIFE_DEVICE_SERIAL": "selected123", "WILDLIFE_ENABLED": "0"}, clear=True), \
             patch.object(sys, "argv", ["wildlife_tracker", "run"]):
            with self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
