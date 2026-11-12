"""Supervised pan probe against a fake WS and a synthetic panning scene.

The scene shifts when the fake camera receives pan_and_tilt; real phase
correlation measures it. This verifies the probe's gates, bounds and cleanup,
not the real S340 step size, permission or direction.
"""
import asyncio
import json
import pathlib
import unittest
from unittest.mock import AsyncMock, patch

import test_observer as fixtures
from wildlife_tracker.core import EufySocket, ProtocolError
from wildlife_tracker.probe import ProbeLimits, ptz_probe
from wildlife_tracker.vision import HEIGHT, WIDTH

LIMITS = ProbeLimits(baseline_seconds=2, confirm_seconds=1, still_seconds=1, onset_seconds=.5,
                     settle_seconds=.6, result_seconds=.3, stream_seconds=10)
STEP_PIXELS = 48


class Scene:
    def __init__(self, step_pixels=STEP_PIXELS, tilt_pixels=40, up_limit_steps=None):
        self.step_pixels = step_pixels
        self.tilt_pixels = tilt_pixels
        import cv2
        import numpy as np
        rng = np.random.default_rng(5)
        # Wide enough that "continuous" motion (16 px/frame) outlasts the settle limit.
        self.image = cv2.GaussianBlur((rng.random((HEIGHT + 400, WIDTH + 800, 3)) * 255).astype(np.uint8), (0, 0), 1)
        self.offset = self.target = 400
        self.voffset = self.vtarget = 200
        # Like the S340 facing a slope: UP stops moving after this many steps from the start view.
        self.top = 0 if up_limit_steps is None else self.voffset - up_limit_steps * tilt_pixels
        self.mode = "step"

    def command(self, direction):
        if direction in (3, 4):
            if self.mode == "step":
                # UP turns the view up: crop moves up, content moves down.
                self.vtarget = max(self.top, self.vtarget + (-1 if direction == 3 else 1) * self.tilt_pixels)
            return
        sign = -1 if direction == 1 else 1  # LEFT turns the view left: crop moves left, content moves right.
        if self.mode == "step":
            self.target += sign * self.step_pixels
        elif self.mode == "continuous":
            self.target = 0 if sign < 0 else 800

    def frame(self):
        self.offset += max(-16, min(16, self.target - self.offset))
        self.voffset += max(-16, min(16, self.vtarget - self.voffset))
        return self.image[self.voffset:self.voffset + HEIGHT, self.offset:self.offset + WIDTH].tobytes()


class PanWS(fixtures.FakeWS):
    def __init__(self, scene):
        super().__init__()
        self.scene = scene
        self.video_payload = b"\x00\x00\x01"
        self.directions = []
        self.reject_pan = False

    async def send(self, raw):
        packet = json.loads(raw)
        command = packet["command"]
        if command == "device.get_commands":
            self.commands.append(command)
            self.incoming.put_nowait(json.dumps({"type": "result", "messageId": packet["messageId"], "success": True,
                                                 "result": {"commands": ["start_livestream", "pan_and_tilt"]}}))
            return
        if command == "device.pan_and_tilt":
            self.commands.append(command)
            self.directions.append(packet["direction"])
            if not self.reject_pan:
                self.scene.command(packet["direction"])
            self.incoming.put_nowait(json.dumps({"type": "result", "messageId": packet["messageId"],
                                                 "success": not self.reject_pan, "result": {"async": True}}))
            if not self.reject_pan:
                self.incoming.put_nowait(json.dumps({"type": "event", "event": {"source": "device",
                    "serialNumber": "selected123", "event": "command result", "command": "pan_and_tilt", "returnCode": 0}}))
            return
        await super().send(raw)


class SceneDecoder:
    def __init__(self, scene):
        self.scene = scene
        self.fmt = "h264"
        self.offsets = []  # view offset of every served frame, for world-anchored fake detections

    async def start(self):
        pass

    async def feed(self, chunk):
        pass

    async def frame(self, timeout):
        await asyncio.sleep(.05)
        frame = self.scene.frame()
        self.offsets.append((self.scene.offset, self.scene.voffset))
        return frame

    async def close(self):
        pass


class ProbeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile
        from wildlife_tracker.core import Config, Marker
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Config("ws://fake", "selected123", pathlib.Path(self.tmp.name))
        self.scene = Scene()
        self.ws = PanWS(self.scene)
        self.client = await EufySocket(self.ws, self.config).open()
        self.marker = Marker(self.config.data_dir)
        self.inspected = {"driver_connected": True, "push_connected": True, "stream_running": False,
                          "ownership_marker_present": False, "versions_verified": True,
                          "settings": {"motionTracking": True}}

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def probe(self, answers=("START", "STEP"), axis="pan", limits=None):
        async def confirm(word, timeout):
            return word in answers
        with patch("wildlife_tracker.probe.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.probe.inspect_selected", AsyncMock(return_value=self.inspected)), \
             patch("wildlife_tracker.probe.Decoder", lambda codec, fps: SceneDecoder(self.scene)), \
             patch("wildlife_tracker.probe.log"):
            return await asyncio.wait_for(ptz_probe(self.config, confirm=confirm, limits=limits or LIMITS, axis=axis), 60)

    def assert_clean(self):
        self.assertFalse(self.ws.stream)
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.client.extra_commands, frozenset())

    async def test_ready_without_start_never_connects(self):
        with patch("wildlife_tracker.probe.connect", new_callable=AsyncMock) as connect, \
             patch("wildlife_tracker.probe.log"):
            self.assertIsNone(await ptz_probe(self.config, confirm=AsyncMock(return_value=False), limits=LIMITS))
            connect.assert_not_awaited()

    async def test_left_then_right_measured_and_returned(self):
        report = await self.probe()
        self.assertEqual(self.ws.directions, [1, 2])
        left, right = report["steps"]
        self.assertEqual((left["outcome"], right["outcome"]), ("settled", "settled"))
        self.assertAlmostEqual(left["integrated_shift"][0], STEP_PIXELS / WIDTH, delta=.012)
        self.assertAlmostEqual(right["integrated_shift"][0], -STEP_PIXELS / WIDTH, delta=.012)
        self.assertTrue(left["matches_assumed_direction"] and right["matches_assumed_direction"])
        self.assertEqual((left["command_result"], left["ack"]), (0, "accepted"))
        self.assertIsNotNone(left["onset_seconds"])
        self.assertTrue(report["returned_to_start"])
        self.assertEqual(report["position"], "start_verified")
        self.assertIs(report["automatic_ptz"], False)
        self.assert_clean()

    async def test_no_motion_ends_plan_after_first_step(self):
        self.scene.mode = "ignore"
        report = await self.probe()
        self.assertEqual(self.ws.directions, [1])
        self.assertEqual(report["stopped"], "no_measurable_motion")
        self.assertEqual(report["position"], "offset_or_unknown_check_in_owner_app")
        self.assert_clean()

    async def test_continuous_motion_reported_without_further_commands(self):
        self.scene.mode = "continuous"
        report = await self.probe()
        self.assertEqual(self.ws.directions, [1])
        self.assertEqual(report["stopped"], "unsettled")
        self.assert_clean()

    async def test_unconfirmed_step_sends_no_pan(self):
        report = await self.probe(answers=("START",))
        self.assertEqual(self.ws.directions, [])
        self.assertEqual((report["stopped"], report["position"]), ("step_not_confirmed", "no_command_sent"))
        self.assertIn("device.start_livestream", self.ws.commands)
        self.assert_clean()

    async def test_rejected_pan_stops_plan(self):
        self.ws.reject_pan = True
        report = await self.probe()
        self.assertEqual(self.ws.directions, [1])
        self.assertEqual(report["stopped"], "command_rejected")
        self.assert_clean()

    async def test_unverified_versions_refused_before_stream(self):
        self.inspected["versions_verified"] = False
        with self.assertRaises(ProtocolError):
            await self.probe()
        self.assertNotIn("device.start_livestream", self.ws.commands)
        self.assert_clean()

    async def test_existing_stream_refused_and_never_stopped(self):
        self.inspected["stream_running"] = True
        self.ws.stream = True
        with self.assertRaises(ProtocolError):
            await self.probe()
        self.assertTrue(self.ws.stream)
        self.assertNotIn("device.stop_livestream", self.ws.commands)


class TiltProbeTests(unittest.IsolatedAsyncioTestCase):
    """Up until the scene's mechanical limit, then back down by the steps that moved."""
    asyncSetUp = ProbeTests.asyncSetUp
    asyncTearDown = ProbeTests.asyncTearDown
    probe = ProbeTests.probe
    assert_clean = ProbeTests.assert_clean
    # settle 1.5 s: a 40 px step takes ~3 synthetic frames plus 3 still frames, slower in the container.
    TILT_LIMITS = ProbeLimits(baseline_seconds=2, still_seconds=1, onset_seconds=.5, settle_seconds=1.5,
                              result_seconds=.3, stream_seconds=40, max_up_steps=5)

    async def tilt(self, limit, max_up=5):
        self.scene.top = self.scene.voffset - limit * self.scene.tilt_pixels
        limits = ProbeLimits(**{**vars(self.TILT_LIMITS), "max_up_steps": max_up})
        return await self.probe(answers=("START",), axis="tilt", limits=limits)

    async def test_up_to_limit_then_back_down(self):
        report = await self.tilt(limit=3)
        self.assertEqual(self.ws.directions, [3, 3, 3, 3, 4, 4, 4])
        self.assertTrue(report["upper_limit_reached"])
        self.assertEqual(report["up_steps_moved"], 3)
        self.assertAlmostEqual(report["tilt_step_median"], 40 / HEIGHT, delta=.015)
        self.assertTrue(all(s["matches_assumed_direction"] for s in report["steps"] if s["outcome"] == "settled"))
        self.assertTrue(report["returned_to_start"])
        self.assertEqual(report["position"], "start_verified")
        self.assertEqual(self.scene.voffset, 200)
        self.assert_clean()

    async def test_already_at_upper_limit_moves_nothing(self):
        report = await self.tilt(limit=0)
        self.assertEqual(self.ws.directions, [3])
        self.assertEqual((report["up_steps_moved"], report["position"]), (0, "unchanged_already_at_upper_limit"))
        self.assert_clean()

    async def test_search_bounded_without_reaching_limit(self):
        report = await self.tilt(limit=6, max_up=2)
        self.assertEqual(self.ws.directions, [3, 3, 4, 4])
        self.assertFalse(report["upper_limit_reached"])
        self.assertTrue(report["returned_to_start"])
        self.assert_clean()

    async def test_tilt_probe_stops_on_unexpected_direction(self):
        self.scene.tilt_pixels = -40  # camera moves the other way than assumed
        self.scene.top = -10_000
        report = await self.tilt(limit=3)
        self.assertEqual(self.ws.directions, [3])
        self.assertEqual(report["stopped"], "unexpected_direction")
        self.assert_clean()


class CommandGuardTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    async def test_pan_needs_probe_widening_and_single_steps_only(self):
        self.assertEqual(self.client.extra_commands, frozenset())
        with self.assertRaises(ValueError):
            await self.client.command("device.pan_and_tilt", serialNumber="selected123", direction=1)
        self.client.extra_commands = frozenset({"device.pan_and_tilt"})
        for direction in (0, 5, -1, None, "3"):  # ROTATE360 and anything not a single step
            with self.assertRaises(ValueError):
                await self.client.command("device.pan_and_tilt", serialNumber="selected123", direction=direction)
        self.assertNotIn("device.pan_and_tilt", self.ws.commands)

    def test_only_probe_and_follow_modules_widen_commands(self):
        package = pathlib.Path(__file__).resolve().parents[1] / "wildlife_tracker"
        writers = sorted(p.name for p in package.glob("*.py") if "extra_commands = frozenset({" in p.read_text())
        self.assertEqual(writers, ["follow.py", "probe.py"])
