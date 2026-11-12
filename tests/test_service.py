"""Persistent-service behavior (warm model, battery guard) and recorded-clip replay.

Fake WS and fake detector; real FFmpeg decodes synthetic H.264. No camera.
"""
import asyncio
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_observer as fixtures
from wildlife_tracker.__main__ import run
from wildlife_tracker.core import Config, Journal
from wildlife_tracker.replay import archive, replay, segments

STILL = {"shift": (0.0, 0.0), "response": .9, "moving": False}


def synthetic_h264(seconds=3, container="h264"):
    return subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
        f"testsrc=size=640x360:rate=12:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-f", container, "pipe:1"])


class FakeDetector:
    instances = []

    def __init__(self, *args, fail_first=False):
        self.loaded = self.closed = False
        self.resets = 0
        self.fail = fail_first
        self.camera = None
        FakeDetector.instances.append(self)

    async def start(self):
        self.loaded = True  # A real Detector.start() returns at once when already alive.

    def alive(self):
        return self.loaded and not self.closed

    async def reset(self):
        self.resets += 1

    def submit(self, frame):
        pass

    async def result(self, timeout):
        if self.fail:
            self.fail = False
            raise RuntimeError("synthetic inference failure")
        self.camera = STILL
        return [{"label": "deer", "score": .8, "area": .1, "center": [.8, .5]}]

    def close(self):
        self.closed = True


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    def setUp(self):
        FakeDetector.instances = []

    def wake(self):
        self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {"source": "device",
            "serialNumber": "selected123", "event": "motion detected", "state": True}}))

    async def serve(self, wakes, factory=FakeDetector):
        """Each wake waits until its session finished, failed or was skipped."""
        ends = ("session_complete", "session_failed", "session_skipped_low_battery")
        self.ws.video_payload = synthetic_h264()
        self.config.session_seconds = 1
        self.config.cooldown_seconds = .01
        async def connect_existing(_):
            return self.client
        with patch("wildlife_tracker.__main__.connect", connect_existing), \
             patch("wildlife_tracker.__main__.Detector", factory), \
             patch("wildlife_tracker.__main__.log") as recorded:
            task = asyncio.create_task(run(self.config, "run"))
            await asyncio.sleep(.05)
            for index in range(wakes):
                self.wake()
                for _ in range(500):
                    await asyncio.sleep(.01)
                    if sum(self.count(recorded.call_args_list, kind) for kind in ends) > index:
                        break
                await asyncio.sleep(.05)  # cooldown passes before the next wake
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        return [c for c in recorded.call_args_list]

    @staticmethod
    def count(calls, kind):
        return sum(1 for c in calls if c.args and c.args[0] == kind)

    async def test_warm_model_loaded_once_and_reset_per_session(self):
        calls = await self.serve(2)
        self.assertEqual(self.count(calls, "session_complete"), 2)
        self.assertEqual(len(FakeDetector.instances), 1)
        worker = FakeDetector.instances[0]
        self.assertEqual(worker.resets, 2)
        self.assertTrue(worker.closed)  # run() closes the warm worker on exit
        self.assertEqual(self.count(calls, "model_ready_warm"), 1)
        completes = [c.kwargs for c in calls if c.args[0] == "session_complete"]
        self.assertTrue(all(c["warm_model"] and c["power"] == {"battery": None, "charging_status": None} for c in completes))
        self.assertEqual(self.ws.commands.count("device.start_livestream"), 2)
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    def journal(self):
        path = self.config.data_dir / "sessions.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    async def test_sessions_are_journaled_on_the_data_mount(self):
        await self.serve(2)
        entries = self.journal()
        self.assertEqual([e["kind"] for e in entries], ["session_complete"] * 2)
        for entry in entries:
            self.assertLessEqual(entry["started"], entry["time"])
            self.assertGreater(entry["inferences"], 0)
            self.assertIn("trigger", entry)
            self.assertNotIn("selected123", json.dumps(entry))  # never the serial

    def test_journal_rotates_and_never_raises(self):
        journal = Journal(self.config.data_dir, limit=200)
        for i in range(20):
            journal.write("session_complete", index=i)
        self.assertTrue((self.config.data_dir / "sessions.jsonl.1").exists())
        self.assertLess((self.config.data_dir / "sessions.jsonl").stat().st_size, 400)
        with patch("wildlife_tracker.core.log") as logged:
            Journal(self.config.data_dir / "sessions.jsonl" / "not-a-dir").write("x")
        self.assertEqual(logged.call_args.args[0], "journal_unwritable")

    async def test_leftover_marker_is_rechecked_on_wake(self):
        # 2026-10-11: a marker left by one uncertain cleanup blocked every later event until restart.
        self.marker.write("owned", "earlier-session", self.config.serial)
        calls = await self.serve(1)
        self.assertEqual(self.count(calls, "session_complete"), 1)
        self.assertEqual(self.count(calls, "wake_blocked_uncertain_ownership"), 0)
        self.assertFalse(self.marker.exists())

    async def test_marker_with_running_stream_still_blocks(self):
        self.marker.write("owned", "earlier-session", self.config.serial)
        self.ws.stream = True
        self.ws.video_payload = synthetic_h264()
        async def connect_existing(_):
            return self.client
        with patch("wildlife_tracker.__main__.connect", connect_existing), \
             patch("wildlife_tracker.__main__.Detector", FakeDetector), \
             patch("wildlife_tracker.__main__.log") as recorded:
            task = asyncio.create_task(run(self.config, "run"))
            await asyncio.sleep(.05)
            self.wake()
            await asyncio.sleep(.3)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(self.count(recorded.call_args_list, "wake_blocked_uncertain_ownership"), 1)
        self.assertTrue(self.marker.exists())
        self.assertNotIn("device.start_livestream", self.ws.commands)
        self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_session_reaching_its_limit_is_not_a_failure(self):
        # A target present for the whole session used to hit a near-zero inference timeout at the deadline.
        from wildlife_tracker.__main__ import collect_stream
        self.ws.video_payload = synthetic_h264(6)
        class SlowDetector(FakeDetector):
            async def result(self, timeout):
                await asyncio.wait_for(asyncio.sleep(.15), timeout)  # honours the caller's timeout
                return await super().result(timeout)
        with patch("wildlife_tracker.__main__.log"):
            result = await collect_stream(self.client, self.config, detector=SlowDetector(), duration=.5)
        self.assertGreaterEqual(result["inferences"], 3)
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    async def test_low_battery_skips_without_stream(self):
        self.ws.battery = 10
        calls = await self.serve(1)
        skipped = [c.kwargs for c in calls if c.args[0] == "session_skipped_low_battery"]
        self.assertEqual(skipped[0]["battery"], 10)
        self.assertEqual([e["kind"] for e in self.journal()], ["session_skipped_low_battery"])
        self.assertNotIn("device.start_livestream", self.ws.commands)

    async def test_battery_above_minimum_runs_session(self):
        self.ws.battery = 80
        calls = await self.serve(1)
        self.assertEqual(self.count(calls, "session_complete"), 1)
        self.assertEqual([c.kwargs for c in calls if c.args[0] == "session_complete"][0]["power"]["battery"], 80)

    async def test_failed_session_replaces_warm_worker(self):
        first = [True]
        def factory(*args):
            fail, first[0] = first[0], False
            return FakeDetector(fail_first=fail)
        calls = await self.serve(2, factory)
        self.assertEqual(self.count(calls, "session_failed"), 1)
        self.assertEqual(self.count(calls, "session_complete"), 1)
        self.assertEqual(len(FakeDetector.instances), 2)
        self.assertTrue(FakeDetector.instances[0].closed)
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    def test_service_settings_validated(self):
        base = {"WILDLIFE_DEVICE_SERIAL": "selected123"}
        with patch.dict(os.environ, base, clear=True):
            config = Config.from_env()
            self.assertEqual((config.warm_model, config.min_battery), (True, 25))
            self.assertEqual((config.max_steps, config.max_net_steps), (16, 5))
        for key, value in (("WILDLIFE_WARM_MODEL", "yes"), ("WILDLIFE_MIN_BATTERY", "101"),
                           ("WILDLIFE_MIN_BATTERY", "nan"), ("WILDLIFE_MAX_STEPS", "0"),
                           ("WILDLIFE_MAX_NET_STEPS", "9"), ("WILDLIFE_MAX_STEPS", "-1")):
            with patch.dict(os.environ, {**base, key: value}, clear=True):
                with self.assertRaises(ValueError):
                    Config.from_env()
        with patch.dict(os.environ, {**base, "WILDLIFE_WARM_MODEL": "0", "WILDLIFE_MIN_BATTERY": "0"}, clear=True):
            config = Config.from_env()
            self.assertEqual((config.warm_model, config.min_battery), (False, 0))
        with patch.dict(os.environ, {**base, "WILDLIFE_MAX_STEPS": "4", "WILDLIFE_MAX_NET_STEPS": "5"}, clear=True):
            with self.assertRaises(ValueError):
                Config.from_env()


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeDetector.instances = []
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.config = Config("ws://unused", "", self.dir)

    async def asyncTearDown(self):
        self.tmp.cleanup()

    async def test_recorded_clip_runs_live_pipeline_offline(self):
        # MP4 needs a seekable output, so FFmpeg writes the file directly.
        subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            "testsrc=size=640x360:rate=12:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(self.dir / "clip.mp4")])
        (self.dir / "notes.txt").write_text("ignored")
        with patch("wildlife_tracker.replay.Detector", FakeDetector), patch("wildlife_tracker.replay.log") as recorded:
            totals = await replay(self.config, self.dir)
        self.assertEqual(totals["clips"], 1)
        clip = [c.kwargs for c in recorded.call_args_list if c.args[0] == "replay_clip"][0]
        self.assertEqual((clip["clip"], clip["dominant_label"]), ("clip.mp4", "deer"))
        self.assertGreaterEqual(clip["frames"], 8)
        self.assertGreater(clip["tracking_frames"], 0)
        self.assertGreaterEqual(clip["planned_steps"], 1)
        self.assertEqual(clip["first_detection_seconds"], 0)
        worker = FakeDetector.instances[0]
        self.assertEqual(worker.resets, 1)
        self.assertTrue(worker.closed)

    def test_segments_pad_merge_and_clamp(self):
        self.assertEqual(segments([5, 5.33, 5.67, 7, 20, 20.33], 120, lead=0), [(3.0, 9.0), (18.0, 22.33)])
        self.assertEqual(segments([.33, 119.5], 120, lead=0), [(0.0, 2.33), (117.5, 120.0)])
        self.assertEqual(segments([], 120, lead=0), [])
        self.assertEqual(segments([10, 13.5], 120, gap=3, lead=0), [(8.0, 15.5)])  # overlap: one cut

    def test_first_seconds_always_kept(self):
        # The recording starts at the triggering motion; a missed animal there must not be cut away.
        self.assertEqual(segments([], 120), [(0.0, 10.0)])
        self.assertEqual(segments([9], 15), [(0.0, 11.0)])
        self.assertEqual(segments([40], 120), [(0.0, 10.0), (38.0, 42.0)])
        self.assertEqual(segments([], 4), [(0.0, 4.0)])

    def write_clip(self, name="clip.mp4", seconds=3):
        subprocess.check_call(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
            f"testsrc=size=640x360:rate=12:duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(self.dir / name)])
        return self.dir / name

    async def test_archive_keeps_only_animal_segments_and_never_touches_sources(self):
        source = self.write_clip()
        before = source.read_bytes()
        out = self.dir / "archive"
        with patch("wildlife_tracker.replay.Detector", FakeDetector), patch("wildlife_tracker.replay.log"):
            report = await archive(self.config, source, out)
            again = await archive(self.config, source, out)
        self.assertEqual(report["archived"], ["clip.mp4"])
        self.assertEqual(again["skipped_existing"], ["clip.mp4"])
        meta = json.loads((out / "clip.json").read_text())
        self.assertEqual(len(meta["segments"]), 1)
        segment = out / meta["segments"][0]["file"]
        self.assertGreater(segment.stat().st_size, 0)
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                         "-of", "csv=p=0", str(segment)]).decode()
        self.assertGreater(float(probe), 1.5)
        self.assertEqual(source.read_bytes(), before)

    async def test_archive_leaves_clip_without_animal_untouched(self):
        source = self.write_clip("empty.mp4")
        class Blind(FakeDetector):
            async def result(self, timeout):
                self.camera = STILL
                return []
        out = self.dir / "archive"
        with patch("wildlife_tracker.replay.Detector", Blind), patch("wildlife_tracker.replay.log"):
            report = await archive(self.config, source, out)
        self.assertEqual(report["no_animal_kept"], ["empty.mp4"])
        self.assertEqual(list(out.iterdir()), [])
        self.assertTrue(source.exists())

    async def test_inbox_archive_copies_undetected_clip_and_saves_candidate_frames(self):
        # A recording the user downloaded most likely shows an animal the model missed: keep it whole.
        source = self.write_clip("missed.mp4", seconds=25)
        before = source.read_bytes()
        class Blind(FakeDetector):
            async def result(self, timeout):
                self.camera = STILL
                return []
        out, frames = self.dir / "archive", self.dir / "candidates"
        with patch("wildlife_tracker.replay.Detector", Blind), patch("wildlife_tracker.replay.log"):
            report = await archive(self.config, source, out, candidates=frames, keep_undetected=True)
            again = await archive(self.config, source, out, candidates=frames, keep_undetected=True)
        self.assertEqual(report["no_animal_kept"], ["missed.mp4"])
        self.assertEqual(again["skipped_existing"], ["missed.mp4"])
        self.assertEqual((out / "missed_undetected.mp4").read_bytes(), before)
        self.assertEqual(json.loads((out / "missed.json").read_text())["segments"], [])
        jpgs = sorted(p.name for p in frames.glob("*.jpg"))
        self.assertEqual(jpgs, ["missed_t0000.0.jpg", "missed_t0010.0.jpg", "missed_t0020.0.jpg"])
        self.assertEqual((frames / jpgs[0]).read_bytes()[:2], b"\xff\xd8")
        manifest = json.loads((frames / "missed_candidates.json").read_text())
        self.assertEqual((manifest["model_detection"], len(manifest["frames"])), (False, 3))
        self.assertEqual(source.read_bytes(), before)

    async def test_candidate_frames_never_come_from_kept_segments(self):
        source = self.write_clip("deer.mp4", seconds=25)
        class LateDeer(FakeDetector):
            async def result(self, timeout):
                self.camera = STILL
                self.frame_index = getattr(self, "frame_index", -1) + 1
                if 36 <= self.frame_index <= 45:  # 12–15 s at 3 fps
                    return [{"label": "deer", "score": .8, "area": .05, "center": [.5, .5]}]
                return []
        out, frames = self.dir / "archive", self.dir / "candidates"
        with patch("wildlife_tracker.replay.Detector", LateDeer), patch("wildlife_tracker.replay.log"):
            report = await archive(self.config, source, out, candidates=frames, keep_undetected=True)
        self.assertEqual(report["archived"], ["deer.mp4"])
        meta = json.loads((out / "deer.json").read_text())
        self.assertEqual([(s["start"], s["end"]) for s in meta["segments"]], [(0.0, 17.0)])
        # 0 s and 10 s fall inside the kept segment; only 20 s was discarded footage.
        self.assertEqual([p.name for p in frames.glob("*.jpg")], ["deer_t0020.0.jpg"])
        self.assertFalse((out / "deer_undetected.mp4").exists())

    async def test_missing_or_empty_path_refused(self):
        with self.assertRaises(FileNotFoundError):
            await replay(self.config, self.dir / "absent")
        with self.assertRaises(FileNotFoundError):
            await replay(self.config, self.dir)
