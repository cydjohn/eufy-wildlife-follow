"""Follow-path preparation: synthetic motion, planning bounds and trigger context.

None of these tests move a camera or measure real step size, direction or
animal recall. They pin behavior that a later supervised hardware stage needs.
"""
import asyncio
import json
import random
import subprocess
import unittest
from unittest.mock import AsyncMock, patch

import test_observer as fixtures
from wildlife_tracker.core import trigger_context
from wildlife_tracker.ptz import DIRECTIONS, StepPlanner, StepPolicy
from wildlife_tracker.tracking import AnimalTracker, TrackingPolicy
from wildlife_tracker.supervised import pilot
from wildlife_tracker.vision import FRAME_BYTES, HEIGHT, WIDTH

STILL = {"shift": (0.0, 0.0), "response": .9, "moving": False}
OPEN = StepPolicy(closed_loop=False)


def animal(x=.8, y=.5, label="deer", score=.8, area=.1):
    return {"label": label, "score": score, "area": area, "center": [x, y]}


def move(horizontal="right", vertical="center"):
    return {"action": "move", "horizontal": horizontal, "vertical": vertical}


class CameraMotionTests(unittest.TestCase):
    def frames(self, shift):
        import cv2
        import numpy as np
        rng = np.random.default_rng(3)
        scene = cv2.GaussianBlur((rng.random((HEIGHT + 40, WIDTH + 80, 3)) * 255).astype(np.uint8), (0, 0), 1)
        first = np.ascontiguousarray(scene[20:20 + HEIGHT, 40:40 + WIDTH])
        # View turned right by `shift` pixels: content moves left in the image.
        second = np.ascontiguousarray(scene[20:20 + HEIGHT, 40 + shift:40 + shift + WIDTH])
        return first.tobytes(), second.tobytes()

    def estimate(self, first, second):
        from wildlife_tracker.motion import CameraMotion, prepare
        motion = CameraMotion()
        self.assertIsNone(motion.update(prepare(first, WIDTH, HEIGHT))["moving"])
        return motion.update(prepare(second, WIDTH, HEIGHT))

    def test_pan_reports_content_shift_sign_and_moving(self):
        result = self.estimate(*self.frames(32))
        self.assertTrue(result["moving"])
        self.assertAlmostEqual(result["shift"][0], -32 / WIDTH, delta=.006)
        self.assertAlmostEqual(result["shift"][1], 0, delta=.006)

    def test_still_view_is_known_still(self):
        first, _ = self.frames(0)
        result = self.estimate(first, first)
        self.assertIs(result["moving"], False)

    def test_textureless_view_is_unknown_not_still(self):
        blank = bytes(FRAME_BYTES)
        result = self.estimate(blank, blank)
        self.assertIsNone(result["shift"])
        self.assertIsNone(result["moving"])


class EgoMotionTrackingTests(unittest.TestCase):
    def acquire(self, tracker, x=.8):
        for t in (0, .33, .66):
            decision = tracker.update([animal(x)], t, camera=STILL)
        self.assertEqual(decision["action"], "move")
        return decision["target_id"]

    def test_measured_view_shift_preserves_association(self):
        uncompensated = AnimalTracker(TrackingPolicy(smoothing=1))
        self.acquire(uncompensated)
        self.assertEqual(uncompensated.update([animal(.55)], 1)["reason"], "target_not_matched")
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        original = self.acquire(tracker)
        result = tracker.update([animal(.55)], 1, camera={"shift": (-.25, 0), "moving": True})
        self.assertEqual(result["target_id"], original)
        self.assertEqual((result["reason"], result["action"]), ("camera_moving", "stop"))
        self.assertEqual(tracker.target["raw_center"], (.55, .5))

    def test_no_new_move_while_view_moves_then_resume_when_still(self):
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        self.acquire(tracker)
        moving = {"shift": (-.03, 0), "moving": True}
        self.assertEqual(tracker.update([animal(.77)], 1, camera=moving)["action"], "stop")
        self.assertEqual(tracker.update([animal(.74)], 1.33, camera=moving)["action"], "none")
        self.assertEqual(tracker.update([animal(.74)], 1.66, camera=STILL)["action"], "move")

    def test_oversized_shift_holds_instead_of_guessing(self):
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        self.acquire(tracker)
        result = tracker.update([animal(.4)], 1, camera={"shift": (-.4, 0), "moving": True})
        self.assertEqual((result["state"], result["reason"]), ("HOLD", "camera_shift_too_large"))

    def test_track_survives_a_pan_while_movement_evidence_relearns(self):
        # Recorded 2026-10-30 night deer, a moving blob only: nothing is detected while the view
        # pans, nor for ~1.7 s after, while the background is relearned. The same track resumes.
        tracker = AnimalTracker(TrackingPolicy(smoothing=1, motion_only_moves=True))
        blob = lambda x: animal(x, label="motion", score=.35, area=.005)
        for i, x in enumerate((.85, .84, .83, .82, .81, .80)):
            decision = tracker.update([blob(x)], i / 3, camera=STILL)
        self.assertEqual(decision["action"], "move")
        original = decision["target_id"]
        pan = {"shift": (-.07, 0), "moving": True}
        for t, camera in ((2, pan), (7 / 3, pan), (8 / 3, STILL), (3, STILL), (10 / 3, STILL), (11 / 3, STILL)):
            self.assertNotEqual(tracker.update([], t, camera=camera)["state"], "LOST")
        resumed = [tracker.update([blob(x)], t, camera=STILL) for t, x in ((4, .66), (13 / 3, .65))]
        self.assertEqual([d["target_id"] for d in resumed], [original, original])
        # Pan frames do not count against the blob's persistence, so it may still turn the camera.
        self.assertEqual((resumed[-1]["reason"], resumed[-1]["action"]), ("target_off_center", "move"))

    def test_a_view_that_keeps_moving_still_loses_the_track(self):
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        self.acquire(tracker)
        states = [tracker.update([], 1 + i / 3, camera={"shift": (-.01, 0), "moving": True})["state"]
                  for i in range(13)]
        self.assertEqual(states.index("LOST"), 11)  # 4 s after the deer was last seen at 0.66

    def test_blob_track_continues_as_a_larger_model_box(self):
        # 2026-10-30: a blob-tracked deer boxed as "dog" 10× its blob area was rejected every other frame.
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        for t in (0, .33, .66):
            original = tracker.update([animal(.5, label="motion", score=.35, area=.005)], t, camera=STILL)["target_id"]
        result = tracker.update([animal(.51, label="dog", score=.5, area=.05)], 1, camera=STILL)
        self.assertEqual((result["target_id"], result["state"]), (original, "TRACKING"))
        # Box against box still keeps the size check.
        self.assertEqual(tracker.update([animal(.51, label="dog", area=.5)], 1.33, camera=STILL)["reason"],
                         "target_not_matched")

    def test_megadetector_animal_boxes_are_followed(self):
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        for t in (0, .33, .66):
            result = tracker.update([animal(.8, label="animal", score=.7), animal(.2, label="vehicle", score=.9)], t, camera=STILL)
        self.assertEqual((result["state"], result["label"], result["action"]), ("TRACKING", "animal", "move"))
        self.assertGreater(result["center"][0], .5)

    def test_unknown_motion_does_not_compensate(self):
        tracker = AnimalTracker(TrackingPolicy(smoothing=1))
        self.acquire(tracker)
        tracker.update([animal(.79)], 1, camera={"shift": None, "moving": None})
        self.assertEqual(tracker.target["raw_center"], (.79, .5))


class StepPlannerTests(unittest.TestCase):
    def test_unknown_or_moving_view_never_steps(self):
        planner = StepPlanner()
        for camera in (None, {"moving": None}, {"moving": True}):
            self.assertIsNone(planner.plan(move(), 10, camera)["command"])
        self.assertEqual(planner.steps, 0)

    def test_open_loop_spacing_and_executable_false(self):
        planner = StepPlanner(OPEN)
        first = planner.plan(move(), 10, STILL)
        self.assertEqual(first["command"], {"command": "device.pan_and_tilt", "direction": 2, "label": "right"})
        self.assertIs(first["executable"], False)
        self.assertEqual(planner.plan(move(), 11, STILL)["reason"], "step_interval")
        self.assertEqual(planner.plan(move(), 12.5, STILL)["reason"], "step")

    def test_excursion_budget_and_reverse_ledger(self):
        planner = StepPlanner(StepPolicy(closed_loop=False, max_net_steps=3))
        reasons = [planner.plan(move(), t * 2.5, STILL)["reason"] for t in range(5)]
        self.assertEqual(reasons, ["step"] * 3 + ["excursion_budget"] * 2)
        self.assertEqual(planner.plan(move("left"), 20, STILL)["reason"], "step")
        summary = planner.finish()
        self.assertEqual(summary["reverse_steps"], ["left", "left"])
        self.assertEqual((summary["return_home"], summary["executable"]), ("blocked_unverified", False))

    def test_tilt_requires_explicit_enable_and_pan_has_priority(self):
        self.assertEqual(StepPlanner(OPEN).plan(move("center", "down"), 0, STILL)["reason"], "tilt_disabled")
        planner = StepPlanner(StepPolicy(closed_loop=False, tilt_enabled=True))
        self.assertEqual(planner.plan(move("center", "down"), 0, STILL)["command"]["direction"], 4)
        self.assertEqual(planner.plan(move("left", "down"), 2.5, STILL)["command"]["label"], "left")

    def test_random_proposals_bounded_and_never_rotate360(self):
        for policy in (StepPolicy(closed_loop=False, tilt_enabled=True, max_up_steps=3),
                       StepPolicy(tilt_enabled=True, tilt_expected_step=.11, max_up_steps=3)):
            planner = StepPlanner(policy)
            rng = random.Random(11)
            now, sent = 0, []
            for _ in range(2000):
                now += rng.uniform(.05, .6)
                decision = move(rng.choice(["left", "right", "center"]), rng.choice(["up", "down", "center"]))
                camera = rng.choice([STILL, {"shift": (-.118, 0), "moving": True}, {"shift": None, "moving": None}])
                plan = planner.plan(decision, now, camera)
                if plan["command"]:
                    sent.append((now, plan["command"]["direction"]))
                self.assertLessEqual(max(map(abs, planner.ledger.values())), policy.max_net_steps)
            self.assertLessEqual(len(sent), policy.max_steps)
            self.assertTrue(set(d for _, d in sent) <= set(DIRECTIONS.values()))
            self.assertNotIn(0, DIRECTIONS.values())
            if not policy.closed_loop:
                self.assertTrue(all(b - a >= 2.5 for (a, _), (b, _) in zip(sent, sent[1:])))

    def test_invalid_step_policy_rejected(self):
        for values in ({"step_tolerance": 1}, {"max_steps": 0}, {"max_net_steps": True},
                       {"step_interval": float("inf")}, {"tilt_enabled": 1}, {"closed_loop": "yes"},
                       {"settle_frames": 0}, {"expected_step": .6}):
            with self.assertRaises(ValueError):
                StepPolicy(**values)


class ClosedLoopStepTests(unittest.TestCase):
    """Next step only after the commanded motion is seen, matches the measured step, and settles."""
    def stepped(self):
        planner = StepPlanner()
        self.assertEqual(planner.plan(move(), 0, STILL)["reason"], "step")
        return planner

    def test_verified_step_unblocks_next(self):
        planner = self.stepped()
        self.assertEqual(planner.plan(move(), 1.0, STILL)["reason"], "awaiting_step_motion")
        # RIGHT turns the view right: content moves left by about one measured step.
        self.assertEqual(planner.plan(move(), 1.8, {"shift": (-.118, .004), "moving": True})["reason"],
                         "awaiting_step_settle")
        self.assertEqual(planner.plan(move(), 2.1, STILL)["reason"], "awaiting_step_settle")
        self.assertEqual(planner.plan(move(), 2.4, STILL)["reason"], "step")
        self.assertIsNone(planner.locked)

    def test_no_observed_motion_locks_session(self):
        planner = self.stepped()
        self.assertEqual(planner.plan(move(), 3.0, STILL)["reason"], "awaiting_step_motion")
        self.assertEqual(planner.plan(move(), 3.6, STILL)["reason"], "step_motion_not_observed")
        self.assertIsNone(planner.plan(move(), 20, STILL)["command"])
        self.assertEqual(planner.finish()["locked"], "step_motion_not_observed")

    def test_recorded_steps_with_blur_drift_are_accepted(self):
        # 2026-11-04 06:05: 0.156 along with 0.034 cross-axis from one blurred frame; daytime steps measure short.
        for shift in ((-.156, .034), (-.045, 0), (-.19, -.05)):
            planner = self.stepped()
            planner.plan(move(), 1.8, {"shift": shift, "moving": True})
            planner.plan(move(), 2.1, STILL)
            self.assertEqual(planner.plan(move(), 2.4, STILL)["reason"], "step", shift)
            self.assertEqual(planner.measured["along"], -shift[0])

    def test_wrong_direction_or_size_locks_session(self):
        for shift in ((.118, 0), (-.03, 0), (-.25, 0), (-.118, .08)):
            planner = self.stepped()
            planner.plan(move(), 1.8, {"shift": shift, "moving": True})
            planner.plan(move(), 2.1, STILL)
            self.assertEqual(planner.plan(move(), 2.4, STILL)["reason"], "step_shift_mismatch", shift)

    def test_motion_that_does_not_settle_locks_session(self):
        planner = self.stepped()
        moving = {"shift": (-.04, 0), "moving": True}
        for t in (1.8, 2.5, 3.2):
            planner.plan(move(), t, moving)
        self.assertEqual(planner.plan(move(), 3.9, moving)["reason"], "step_not_settled")

    def test_unmeasurable_frame_resets_settle_count(self):
        planner = self.stepped()
        planner.plan(move(), 1.8, {"shift": (-.118, 0), "moving": True})
        planner.plan(move(), 2.1, STILL)
        self.assertEqual(planner.plan(move(), 2.4, {"shift": None, "moving": None})["reason"], "awaiting_step_settle")
        planner.plan(move(), 2.7, STILL)
        self.assertEqual(planner.plan(move(), 3.0, STILL)["reason"], "step")


class TriggerContextTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    def event(self, name, serial="selected123", state=True):
        self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {
            "source": "device", "serialNumber": serial, "event": name, "state": state}}))

    async def test_reader_records_only_selected_true_native_events(self):
        self.event("person detected", "another")
        self.event("person detected", state=False)
        self.event("motion detected")
        await asyncio.sleep(.02)
        self.assertEqual(trigger_context(self.client.device_events, 0)["category"], "motion_only")
        self.event("person detected")
        await asyncio.sleep(.02)
        context = trigger_context(self.client.device_events, 0)
        self.assertEqual(context, {"native_events": ["motion detected", "person detected"],
                                   "category": "person_associated"})
        self.assertEqual(trigger_context(self.client.device_events, float("inf"))["native_events"], [])

    async def test_ptz_and_settings_commands_are_structurally_refused(self):
        for command in ("device.pan_and_tilt", "device.preset_position", "device.set_property", "device.calibrate"):
            with self.assertRaises(ValueError):
                await self.client.command(command, serialNumber="selected123")
        self.assertFalse(any("pan" in c or "preset" in c or "set_property" in c or "calibrate" in c for c in self.ws.commands))

    async def test_pilot_reports_category_and_motion_only_candidates(self):
        self.config.cooldown_seconds = .001
        results = [{"frames": 3, "inferences": 3, "detections": 0, "skipped": False},
                   {"frames": 3, "inferences": 3, "detections": 2, "skipped": False}]
        inspected = AsyncMock(return_value={"driver_connected": True, "push_connected": True,
                                            "stream_running": False, "ownership_marker_present": False})
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", inspected), \
             patch("wildlife_tracker.__main__.collect_stream", AsyncMock(side_effect=results)), \
             patch("wildlife_tracker.supervised.log") as recorded:
            task = asyncio.create_task(pilot(self.config, 2, 2, confirm=AsyncMock(return_value=True)))
            await asyncio.sleep(.03)
            self.event("person detected")
            self.event("motion detected")
            # Simulate the person event ageing out of the 5-second context window.
            await asyncio.sleep(.05)
            self.client.device_events.clear()
            self.event("motion detected")
            await asyncio.wait_for(task, 2)
        sessions = [c.kwargs for c in recorded.call_args_list if c.args[0] == "pilot_session"]
        self.assertEqual([s["trigger"]["category"] for s in sessions], ["person_associated", "motion_only"])
        triggers = [c.kwargs for c in recorded.call_args_list if c.args[0] == "pilot_trigger"]
        self.assertTrue(all(t["utc"].endswith("Z") for t in triggers))
        summary = [c.kwargs for c in recorded.call_args_list if c.args[0] == "pilot_complete"][0]
        self.assertEqual(summary["trigger_categories"], {"person_associated": 1, "vehicle_associated": 0, "motion_only": 1})
        self.assertEqual(summary["motion_only_sessions_with_candidates"], 1)
        self.assertIs(summary["automatic_ptz"], False)


class DryRunStepIntegrationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    async def run_session(self, camera):
        from wildlife_tracker.__main__ import collect_stream
        self.ws.video_payload = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=size=640x360:rate=12:duration=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1"])
        class FakeDetector:
            def __init__(self):
                self.camera = None
            async def start(self):
                pass
            def submit(self, frame):
                pass
            async def result(self, timeout):
                self.camera = camera
                return [animal()]
            def close(self):
                pass
        with patch("wildlife_tracker.__main__.log") as recorded:
            result = await collect_stream(self.client, self.config, detector=FakeDetector(), duration=1.5)
        return result, recorded

    async def test_still_view_logs_would_send_without_sending(self):
        result, recorded = await self.run_session(STILL)
        plans = [c.kwargs for c in recorded.call_args_list if c.args[0] == "ptz_dry_run"]
        self.assertTrue(any(p.get("would_send", {}).get("label") == "right" for p in plans))
        self.assertGreaterEqual(result["planned_steps"], 1)
        self.assertEqual(result["labels"]["deer"]["max_score"], .8)
        self.assertGreater(result["camera_motion"]["known_frames"], 0)
        self.assertIs(result["control_enabled"], False)
        self.assertTrue(all("pan" not in c and "preset" not in c and "set_property" not in c for c in self.ws.commands))
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())

    async def test_moving_view_plans_no_step(self):
        result, recorded = await self.run_session({"shift": (-.05, 0), "response": .8, "moving": True})
        self.assertEqual(result["planned_steps"], 0)
        self.assertGreater(result["camera_motion"]["moving_frames"], 0)
        self.assertTrue(any(c.args[0] == "camera_motion" and c.kwargs["moving"] for c in recorded.call_args_list))


class ClosedLoopTiltTests(unittest.TestCase):
    """UP/DOWN with a measured step and the S340's mechanical upper limit (slope opposite)."""
    POLICY = StepPolicy(tilt_enabled=True, tilt_expected_step=.11, max_up_steps=2)
    UP_SHIFT = {"shift": (0.0, .11), "response": .9, "moving": True}  # view up → content down

    def up(self, planner, t):
        return planner.plan(move("center", "up"), t, STILL)

    def test_verified_up_step(self):
        planner = StepPlanner(self.POLICY)
        self.assertEqual(self.up(planner, 0)["command"]["direction"], 3)
        planner.plan(move("center", "up"), 1.8, self.UP_SHIFT)
        planner.plan(move("center", "up"), 2.1, STILL)
        self.assertEqual(self.up(planner, 2.4)["reason"], "step")
        self.assertEqual(planner.ledger["tilt"], -2)

    def test_upper_limit_is_not_a_fault(self):
        planner = StepPlanner(self.POLICY)
        self.up(planner, 0)
        for t in (1, 2, 3, 3.6):
            result = self.up(planner, t)
        self.assertIsNone(planner.locked)
        self.assertTrue(planner.up_limit)
        self.assertEqual(planner.ledger["tilt"], 0)  # the step that did nothing is not counted
        self.assertEqual(result["reason"], "upper_limit")
        self.assertEqual(planner.plan(move("left"), 4, STILL)["reason"], "step")  # pan still follows

    def test_unmeasurable_wait_is_still_uncertain(self):
        planner = StepPlanner(self.POLICY)
        self.up(planner, 0)
        planner.plan(move("center", "up"), 1, {"shift": None, "moving": None})
        for t in (2, 3, 3.6):
            self.up(planner, t)
        self.assertEqual(planner.locked, "step_motion_not_observed")

    def test_measured_up_steps_cap(self):
        planner = StepPlanner(self.POLICY)
        t = 0
        for _ in range(2):
            self.up(planner, t)
            planner.plan(move("center", "up"), t + 1.8, self.UP_SHIFT)
            planner.plan(move("center", "up"), t + 2.1, STILL)
            planner.plan(move("center", "up"), t + 2.4, STILL)
            t += 3
        self.assertEqual(planner.ledger["tilt"], -2)
        self.assertEqual(self.up(planner, t)["reason"], "upper_limit")
        self.assertEqual(planner.plan(move("center", "down"), t, STILL)["command"]["direction"], 4)

    def test_wrong_tilt_direction_locks(self):
        planner = StepPlanner(self.POLICY)
        self.up(planner, 0)
        planner.plan(move("center", "up"), 1.8, {"shift": (0.0, -.11), "moving": True})
        planner.plan(move("center", "up"), 2.1, STILL)
        planner.plan(move("center", "up"), 2.4, STILL)
        self.assertEqual(planner.locked, "step_shift_mismatch")

    def test_tilt_needs_measured_step(self):
        with self.assertRaises(ValueError):
            StepPolicy(tilt_enabled=True)
        for values in ({"max_up_steps": 9}, {"tilt_expected_step": .6}, {"max_up_steps": True}):
            with self.assertRaises(ValueError):
                StepPolicy(**values)
        self.assertEqual(StepPlanner(StepPolicy(tilt_enabled=True, tilt_expected_step=.1)).plan(
            move("center", "up"), 0, STILL)["reason"], "upper_limit")  # 0 measured up steps: never up
