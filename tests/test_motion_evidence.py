"""Movement evidence (background subtraction) and the night stream hold.

Synthetic scenes only. The thresholds come from recorded S340 clips: a small
infrared deer the model never named, but a continuous moving blob; a changing
timestamp watermark in the top band; bushes that flicker in place.
"""
import asyncio
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_observer as fixtures
from wildlife_tracker.motion import MOTION_LABEL, MotionBlobs, is_grayscale
from wildlife_tracker.tracking import AnimalTracker, TrackingPolicy, candidates
from wildlife_tracker.vision import FRAME_BYTES, HEIGHT, WIDTH

STILL = {"shift": (0.0, 0.0), "response": .9, "moving": False}


def scene(seed=4):
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((HEIGHT, WIDTH, 3)) * 160).astype(np.uint8), (0, 0), 2)


def with_box(base, x, y, size=14, value=250):
    image = base.copy()
    image[y:y + size, x:x + size] = value
    return image.tobytes()


def blob(x, y, area=.002):
    return {"label": MOTION_LABEL, "score": .35, "area": area, "center": [x, y]}


class MotionBlobTests(unittest.TestCase):
    def run_frames(self, frames, camera=STILL):
        detector = MotionBlobs()
        return [detector.update(f, WIDTH, HEIGHT, camera) for f in frames]

    def test_small_moving_animal_is_located(self):
        base = scene()
        results = self.run_frames([with_box(base, 300 + 6 * i, 150) for i in range(12)])
        found = [r[0] for r in results[4:] if r]
        self.assertGreaterEqual(len(found), 6)
        for item in found:
            self.assertEqual(item["label"], MOTION_LABEL)
            self.assertAlmostEqual(item["center"][1], (150 + 7) / HEIGHT, delta=.05)

    def test_still_scene_and_watermark_band_yield_nothing(self):
        base = scene()
        # Only the top 7% changes, like Eufy's per-second timestamp.
        frames = [with_box(base, 500, 4, size=12, value=40 * (i % 5)) for i in range(12)]
        self.assertEqual([r for r in self.run_frames(frames) if r], [])

    def test_whole_frame_change_is_not_an_animal(self):
        base = scene()
        frames = [base.tobytes()] * 6 + [(base // 10).tobytes()] * 2  # IR light switching off
        self.assertEqual(self.run_frames(frames)[-2], [])

    def test_moving_view_resets_background(self):
        base = scene()
        detector = MotionBlobs()
        for i in range(6):
            detector.update(with_box(base, 300 + 6 * i, 150), WIDTH, HEIGHT, STILL)
        self.assertEqual(detector.update(base.tobytes(), WIDTH, HEIGHT, {"shift": (.05, 0), "moving": True}), [])
        self.assertEqual(detector.seen, 1)

    def test_grayscale_detects_infrared_frames(self):
        import numpy as np
        gray = np.repeat(scene()[..., :1], 3, axis=2).tobytes()
        self.assertTrue(is_grayscale(gray, WIDTH, HEIGHT))
        self.assertFalse(is_grayscale(scene().tobytes(), WIDTH, HEIGHT))


class MotionTrackingTests(unittest.TestCase):
    def test_blob_inside_model_box_is_one_candidate(self):
        merged = candidates([{"label": "dog", "score": .5, "area": .04, "center": [.60, .60]},
                             blob(.63, .66)], .12)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["model_score"], .5)
        self.assertIn(MOTION_LABEL, merged[0]["votes"])

    def test_fragments_of_one_moving_body_merge(self):
        self.assertEqual(len(candidates([blob(.50, .50), blob(.53, .52, .001)], .12)), 1)

    def test_flickering_bush_never_turns_the_camera(self):
        tracker = AnimalTracker()
        decisions = [tracker.update([blob(.85 + (.004 if i % 2 else 0), .5)], i / 3) for i in range(30)]
        self.assertTrue(any(d["state"] == "TRACKING" for d in decisions))
        self.assertFalse(any(d["action"] == "move" for d in decisions))
        self.assertEqual(decisions[-1]["reason"], "awaiting_confident_detection")

    def test_travelling_movement_only_target_turns_the_camera_only_when_enabled(self):
        # Off by default since MegaDetector: such tracks were insects or rain near the lens (2026-11-03/04).
        tracker = AnimalTracker()
        decisions = [tracker.update([blob(.78 + .012 * i, .5)], i / 3) for i in range(8)]
        self.assertFalse(any(d["action"] == "move" for d in decisions))
        self.assertIn("awaiting_confident_detection", {d["reason"] for d in decisions})
        tracker = AnimalTracker(TrackingPolicy(motion_only_moves=True))
        decisions = [tracker.update([blob(.78 + .012 * i, .5)], i / 3) for i in range(8)]
        self.assertTrue(any(d["action"] == "move" and d["horizontal"] == "right" for d in decisions))
        self.assertEqual(decisions[-1]["label"], MOTION_LABEL)  # no species claimed

    def test_movement_track_turns_the_camera_once_the_model_confirms_it(self):
        tracker = AnimalTracker()
        for i in range(4):
            tracker.update([blob(.80 + .01 * i, .5)], i / 3)
        decision = tracker.update([blob(.84, .5), {"label": "animal", "score": .7, "area": .02, "center": [.84, .5]}], 4 / 3)
        self.assertEqual((decision["action"], decision["horizontal"]), ("move", "right"))

    def test_intermittent_travelling_noise_never_turns_the_camera(self):
        # Heavy rain 2026-10-15: a blob track that travels but is matched only every other frame.
        tracker = AnimalTracker()
        decisions = []
        for i in range(30):
            found = [blob(.70 + .01 * i, .5)] if i % 2 == 0 else []
            decisions.append(tracker.update(found, i / 3))
        self.assertFalse(any(d["action"] == "move" for d in decisions))

    def test_recorded_night_deer_blob_track(self):
        # Blob positions from the 2026-10-07 05:12 infrared clip (model detected nothing).
        path = [(.53, .15, .0024), (.52, .15, .0023), (.52, .14, .0021), (.51, .13, .0014), (.50, .12, .0016),
                (.49, .10, .0013), (.49, .11, .0016), (.48, .11, .0021), (.48, .11, .0020), (.47, .10, .0016)]
        tracker = AnimalTracker()
        decisions = [tracker.update([blob(x, y, a)], 9.67 + i / 3) for i, (x, y, a) in enumerate(path)]
        self.assertEqual(len({d["target_id"] for d in decisions}), 1)
        self.assertEqual(decisions[-1]["state"], "TRACKING")
        # It walked up and away near the top edge: the only off-centre axis is vertical (tilt disabled).
        self.assertTrue(all(d["horizontal"] == "center" for d in decisions))


class NightHoldTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    async def session(self, grayscale, hold, duration=.6):
        from wildlife_tracker.__main__ import collect_stream
        self.ws.video_payload = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=12:duration=6", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
        self.config.initial_target_seconds = .3
        self.config.night_hold_seconds = hold
        stops = []
        class Detector:
            camera = None
            async def start(self):
                pass
            def submit(self, frame):
                pass
            async def result(self, timeout):
                self.camera = {**STILL, "grayscale": grayscale}
                return []
            def close(self):
                pass
        began = asyncio.get_running_loop().time()
        with patch("wildlife_tracker.__main__.log"):
            # Short session: the synthetic stream yields only a burst of frames, so end on the deadline.
            result = await collect_stream(self.client, self.config, detector=Detector(), duration=duration)
        return result, asyncio.get_running_loop().time() - began

    async def test_infrared_session_holds_stream_until_recording_ends(self):
        result, elapsed = await self.session(grayscale=True, hold=1.5)
        self.assertTrue(result["night"])
        self.assertGreaterEqual(elapsed, 1.4)
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    async def test_video_stopping_during_hold_ends_session_normally(self):
        # The fake camera sends one burst then nothing: observation ends first (0.2 s), then the
        # feeder's 0.8 s chunk deadline expires inside the 4 s hold.
        self.config.frame_seconds = .8
        result, elapsed = await self.session(grayscale=True, hold=4, duration=.2)
        self.assertTrue(result["night"])
        self.assertLess(elapsed, 3.5)
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    async def test_daylight_session_stops_without_hold(self):
        result, elapsed = await self.session(grayscale=False, hold=1.5)
        self.assertFalse(result["night"])
        self.assertLess(elapsed, 1.4)
        self.assertFalse(self.ws.stream)
