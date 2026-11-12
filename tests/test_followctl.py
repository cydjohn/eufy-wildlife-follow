"""Closed-loop following through a full observer session against a panning synthetic scene.

The fake camera pans by the measured S340 step (≈0.119 of the width) on each
command. A deer is fixed in world coordinates, so the detector reports where it
appears in the current view. This verifies control logic, ledger and return —
not real animal recall or real camera behavior.
"""
import asyncio
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import test_probe as scene_fixtures
from wildlife_tracker.__main__ import collect_stream
from wildlife_tracker.core import Config, EufySocket, Marker
from wildlife_tracker.follow import FollowController, Ledger, follow_available, ledger_clear
from wildlife_tracker.vision import HEIGHT, WIDTH

STEP = 76  # ≈ 0.119 × 640, the measured pan step
START = 400


class WorldDetector:
    """Deer at a fixed world position; view motion derived from the served frame offsets."""
    def __init__(self, decoder, world_x, visible_for=3.0, on_frame=None, world_y=None, visible_from=0.0, night=False):
        self.decoder = decoder
        self.world_x = world_x
        self.world_y = world_y  # None: stays at mid-height
        self.visible_from = time.monotonic() + visible_from
        self.night = night
        self.visible_until = time.monotonic() + visible_for
        self.on_frame = on_frame
        self.previous = None
        self.camera = None

    async def start(self):
        pass

    def submit(self, frame):
        self.offset, self.voffset = self.decoder.offsets[-1]

    async def result(self, timeout):
        if self.previous is None:
            self.camera = {"shift": None, "response": None, "moving": None}
        else:
            dx = (self.previous[0] - self.offset) / WIDTH   # view right → content left
            dy = (self.previous[1] - self.voffset) / HEIGHT  # view up → content down
            self.camera = {"shift": (round(dx, 4), round(dy, 4)), "response": .9,
                           "moving": max(abs(dx), abs(dy)) >= .02}
        self.previous = (self.offset, self.voffset)
        self.camera["grayscale"] = self.night
        if self.on_frame:
            self.on_frame(self)
        if not self.visible_from <= time.monotonic() <= self.visible_until:
            return []
        x = (self.world_x - self.offset) / WIDTH
        y = .5 if self.world_y is None else (self.world_y - self.voffset) / HEIGHT
        return [{"label": "deer", "score": .8, "area": .08, "center": [round(x, 3), round(y, 3)]}] if 0 <= x <= 1 and 0 <= y <= 1 else []

    def close(self):
        pass


class FollowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Config("ws://fake", "selected123", pathlib.Path(self.tmp.name))
        self.config.initial_target_seconds = 5
        self.config.target_loss_seconds = .6
        self.scene = scene_fixtures.Scene(step_pixels=STEP)
        self.ws = scene_fixtures.PanWS(self.scene)
        self.client = await EufySocket(self.ws, self.config).open()
        self.decoder = scene_fixtures.SceneDecoder(self.scene)
        self.ledger = Ledger(self.config.data_dir)

    async def asyncTearDown(self):
        await self.client.close()
        self.tmp.cleanup()

    async def session(self, visible_for=3.0, on_frame=None, world_view_x=.8, world_view_y=None, visible_from=0.0, night=False):
        world_y = None if world_view_y is None else self.scene.voffset + world_view_y * HEIGHT
        detector = WorldDetector(self.decoder, START + world_view_x * WIDTH, visible_for, on_frame, world_y,
                                 visible_from, night)
        follow = FollowController(self.client, self.config)
        with patch("wildlife_tracker.__main__.Decoder", lambda codec: self.decoder), \
             patch("wildlife_tracker.__main__.log"), patch("wildlife_tracker.follow.log"):
            result = await asyncio.wait_for(collect_stream(self.client, self.config, detector=detector,
                                                           duration=10, follow=follow), 30)
        return result, follow

    def assert_stream_clean(self):
        self.assertFalse(self.ws.stream)
        self.assertFalse(Marker(self.config.data_dir).exists())
        self.assertEqual(self.client.extra_commands, frozenset())

    async def test_follows_off_center_deer_then_returns_home(self):
        result, follow = await self.session()
        self.assertEqual(self.ws.directions, [2, 2, 1, 1])  # two right to center, two verified left back
        self.assertEqual(self.scene.offset, START)
        summary = result["follow"]
        self.assertEqual((summary["steps_sent"], summary["offset"], summary["return_verified"]), (4, 0, True))
        self.assertIsNone(summary["uncertain"])
        self.assertFalse(self.ledger.exists())
        self.assertIs(result["control_enabled"], True)
        self.assert_stream_clean()

    async def test_tilts_up_to_mechanical_limit_then_returns_by_moved_steps(self):
        # Deer high on the slope; the camera can tilt up only one step from its start view.
        self.config.tilt_step, self.config.tilt_up_steps = 40 / HEIGHT, 3
        self.scene.top = self.scene.voffset - 40
        result, follow = await self.session(world_view_x=.5, world_view_y=.12)
        self.assertEqual(self.ws.directions, [3, 3, 4])  # up, up (limit: no motion), one verified down
        summary = result["follow"]
        self.assertTrue(summary["up_limit_reached"])
        self.assertIsNone(summary["uncertain"])
        self.assertEqual((summary["tilt_offset"], summary["return_verified"]), (0, True))
        self.assertEqual(self.scene.voffset, 200)
        self.assertFalse(self.ledger.exists())
        self.assert_stream_clean()

    async def test_tilt_stays_off_until_measured(self):
        result, _ = await self.session(world_view_x=.5, world_view_y=.12)
        self.assertEqual(self.ws.directions, [])

    def test_step_limits_come_from_config(self):
        self.config.max_steps, self.config.max_net_steps = 12, 4
        policy = FollowController(self.client, self.config).planner.policy
        self.assertEqual((policy.max_steps, policy.max_net_steps), (12, 4))

    async def test_night_session_follows_an_animal_that_arrives_late(self):
        # 2026-10-11 02:28: analysis stopped after the first seconds while the stream stayed open.
        self.config.initial_target_seconds = .3
        self.config.night_hold_seconds = 7
        self.config.frame_seconds = 30  # the fake camera sends a single chunk; a real one keeps streaming
        began = time.monotonic()
        result, follow = await self.session(visible_from=1.0, visible_for=4.0, night=True)
        self.assertTrue(result["night"])
        self.assertEqual(self.ws.directions, [2, 2, 1, 1])  # followed, then home, inside the held stream
        self.assertEqual((result["follow"]["offset"], result["follow"]["return_verified"]), (0, True))
        self.assertGreaterEqual(time.monotonic() - began, 6.5)  # stream outlasts the recording
        self.assertFalse(self.ledger.exists())
        self.assert_stream_clean()

    async def test_centered_deer_needs_no_step(self):
        result, _ = await self.session(world_view_x=.55)
        self.assertEqual(self.ws.directions, [])
        self.assertFalse(self.ledger.exists())
        self.assert_stream_clean()

    async def test_person_event_yields_control_to_native_tracking(self):
        self.client.device_events.append((time.monotonic(), "person detected"))
        result, _ = await self.session()
        self.assertEqual(self.ws.directions, [])
        self.assertEqual(result["follow"]["no_follow"], "person_present_native_tracking")
        self.assert_stream_clean()

    async def test_unobserved_step_locks_and_keeps_ledger(self):
        self.scene.mode = "ignore"
        result, _ = await self.session(visible_for=6)
        self.assertEqual(self.ws.directions, [2])
        self.assertEqual(result["follow"]["uncertain"], "step_motion_not_observed")
        self.assertIs(result["follow"]["return_verified"], None)
        self.assertEqual(self.ledger.read()["state"], "uncertain")
        self.assert_stream_clean()
        # The next session starts from wherever the camera points now instead of staying blocked.
        resumed = FollowController(self.client, self.config)
        self.assertIsNone(resumed.uncertain)
        self.assertFalse(self.ledger.exists())
        self.assertEqual(self.client.extra_commands, frozenset({"device.pan_and_tilt"}))

    async def test_unexplained_motion_away_from_start_never_returns(self):
        moved = []
        def native(detector):
            # Once the single step is verified (ledger "known"), the view moves without our command.
            state = (self.ledger.read() or {}).get("state")
            if self.ws.directions == [2] and state == "known" and not moved:
                self.scene.target = START + STEP + 160
                moved.append(True)
        # At 0.72 one step (≈0.119) centers the deer, so exactly one step precedes the native motion.
        result, _ = await self.session(visible_for=6, on_frame=native, world_view_x=.72)
        self.assertTrue(moved)
        self.assertNotIn(1, self.ws.directions)  # no reverse step from an unknown position
        self.assertEqual(result["follow"]["uncertain"], "unexplained_view_motion_away_from_start")
        self.assertTrue(self.ledger.exists())
        self.assert_stream_clean()

    async def test_rejected_step_is_uncertain(self):
        self.ws.reject_pan = True
        result, _ = await self.session()
        self.assertEqual(self.ws.directions, [2])
        self.assertEqual(result["follow"]["uncertain"], "send_failed:CommandRejected")
        self.assertEqual(self.ledger.read()["state"], "uncertain")
        self.assert_stream_clean()

    async def test_ledger_clear_requires_home_confirmation(self):
        self.ledger.write(1, "uncertain")
        with patch("wildlife_tracker.follow.log"):
            self.assertFalse(await ledger_clear(self.config, confirm=AsyncMock(return_value=False)))
            self.assertTrue(self.ledger.exists())
            self.assertTrue(await ledger_clear(self.config, confirm=AsyncMock(return_value=True)))
        self.assertFalse(self.ledger.exists())

    def test_follow_availability_and_setting(self):
        self.assertEqual(follow_available(self.client, self.config), "disabled")
        self.config.follow = True
        self.assertIsNone(follow_available(self.client, self.config))
        self.client.version = {**self.client.version, "serverVersion": "3.2.0"}
        self.assertEqual(follow_available(self.client, self.config), "unverified_ws_versions")
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {"WILDLIFE_DEVICE_SERIAL": "selected123", "WILDLIFE_FOLLOW": value}, clear=True):
                self.assertIs(Config.from_env().follow, expected)
        with patch.dict(os.environ, {"WILDLIFE_DEVICE_SERIAL": "selected123", "WILDLIFE_FOLLOW": "yes"}, clear=True):
            with self.assertRaises(ValueError):
                Config.from_env()
        with patch.dict(os.environ, {"WILDLIFE_DEVICE_SERIAL": "selected123"}, clear=True):
            self.assertIs(Config.from_env().follow, False)
