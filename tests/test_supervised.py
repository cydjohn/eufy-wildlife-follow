"""Synthetic readiness/one-shot checks; never connects to a real camera."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_observer as fixtures
from wildlife_tracker.supervised import trial, pilot


class TrialTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown
    async def test_ready_does_not_connect_before_confirmation(self):
        confirmation = asyncio.Event()
        async def confirm():
            await confirmation.wait()
            return False
        with patch("wildlife_tracker.supervised.connect", new_callable=AsyncMock) as connect:
            task = asyncio.create_task(trial(self.config, .05, confirm=confirm))
            await asyncio.sleep(.02)
            connect.assert_not_awaited()
            self.assertFalse(self.marker.exists())
            confirmation.set()
            await task
            connect.assert_not_awaited()

    async def test_single_trigger_no_retries_and_verified_cleanup(self):
        async def inspected(client, config):
            return {"driver_connected": True, "push_connected": True,
                    "stream_running": False, "ownership_marker_present": False}
        async def collect(client, config, **kwargs):
            self.assertTrue(kwargs["detector"])
            return {"frames": 1, "inferences": 1, "detections": 0, "skipped": False}
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", inspected), \
             patch("wildlife_tracker.__main__.collect_stream", AsyncMock(side_effect=collect)) as collected:
            task = asyncio.create_task(trial(self.config, 1, confirm=AsyncMock(return_value=True)))
            await asyncio.sleep(.02)
            for _ in range(2):
                self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {
                    "source": "device", "serialNumber": self.config.serial,
                    "event": "motion detected", "state": True}}))
            await task
            collected.assert_awaited_once()
            self.assertTrue(self.client.closed)

    async def test_existing_stream_blocks_trial_without_stop(self):
        self.ws.stream = True
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", AsyncMock(return_value={
                 "driver_connected": True, "push_connected": True,
                 "stream_running": True, "ownership_marker_present": False})), \
             patch("wildlife_tracker.__main__.collect_stream", new_callable=AsyncMock) as collected:
            with self.assertRaises(RuntimeError):
                await trial(self.config, .05, confirm=AsyncMock(return_value=True))
            collected.assert_not_awaited()
            self.assertTrue(self.ws.stream)
            self.assertNotIn("device.stop_livestream", self.ws.commands)

    async def test_no_event_is_inconclusive_and_starts_no_stream(self):
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", AsyncMock(return_value={
                 "driver_connected": True, "push_connected": True,
                 "stream_running": False, "ownership_marker_present": False})), \
             patch("wildlife_tracker.supervised.log") as logged:
            await trial(self.config, .02, confirm=AsyncMock(return_value=True))
            self.assertNotIn("device.start_livestream", self.ws.commands)
            outcomes = [call.kwargs for call in logged.call_args_list if call.args[0] == "trial_no_event"]
            self.assertEqual(outcomes[0]["verdict"], "inconclusive")

    async def test_cancel_while_ready_releases_without_connecting(self):
        async def pending():
            await asyncio.Event().wait()
        with patch("wildlife_tracker.supervised.connect", new_callable=AsyncMock) as connect:
            task = asyncio.create_task(trial(self.config, confirm=pending))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            connect.assert_not_awaited()


class PilotTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.ObserverTests.asyncSetUp
    asyncTearDown = fixtures.ObserverTests.asyncTearDown

    def inspected(self):
        return AsyncMock(return_value={"driver_connected": True, "push_connected": True,
            "stream_running": False, "ownership_marker_present": False})

    def wake(self):
        self.ws.incoming.put_nowait(json.dumps({"type": "event", "event": {
            "source": "device", "serialNumber": self.config.serial,
            "event": "motion detected", "state": True}}))

    async def test_pilot_no_confirmation_never_connects(self):
        with patch("wildlife_tracker.supervised.connect", new_callable=AsyncMock) as connect:
            await pilot(self.config, .1, confirm=AsyncMock(return_value=False))
            connect.assert_not_awaited()

    async def test_pilot_caps_sessions_and_never_requests_walk(self):
        self.config.cooldown_seconds = .001
        collected = AsyncMock(return_value={"frames": 1, "inferences": 1, "detections": 0, "skipped": False})
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", self.inspected()), \
             patch("wildlife_tracker.__main__.collect_stream", collected), \
             patch("wildlife_tracker.supervised.log") as recorded:
            task = asyncio.create_task(pilot(self.config, 1, 2, confirm=AsyncMock(return_value=True)))
            for _ in range(3):
                await asyncio.sleep(.03)
                self.wake()
            await task
            self.assertEqual(collected.await_count, 2)
            armed = [c.kwargs for c in recorded.call_args_list if c.args[0] == "PILOT_ARMED"]
            self.assertTrue(armed[0]["no_walk_required"])
            summary = [c.kwargs for c in recorded.call_args_list if c.args[0] == "pilot_complete"][0]
            self.assertEqual(summary["successes"], 2)
            self.assertEqual(summary["animal_evidence"], "not_established")

    async def test_pilot_failure_exits_without_retry(self):
        collected = AsyncMock(side_effect=RuntimeError("synthetic failure"))
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", self.inspected()), \
             patch("wildlife_tracker.__main__.collect_stream", collected):
            task = asyncio.create_task(pilot(self.config, 1, confirm=AsyncMock(return_value=True)))
            await asyncio.sleep(.02)
            self.wake()
            with self.assertRaises(RuntimeError):
                await task
            collected.assert_awaited_once()
            self.assertTrue(self.client.closed)

    async def test_pilot_deadline_cancels_session_and_finishes_cleanup(self):
        cleaned = asyncio.Event()
        async def waiting(*args, **kwargs):
            try:
                await asyncio.sleep(60)
            finally:
                cleaned.set()
        with patch("wildlife_tracker.supervised.connect", AsyncMock(return_value=self.client)), \
             patch("wildlife_tracker.supervised.inspect_selected", self.inspected()), \
             patch("wildlife_tracker.__main__.collect_stream", waiting):
            task = asyncio.create_task(pilot(self.config, .1, confirm=AsyncMock(return_value=True)))
            await asyncio.sleep(.02)
            self.wake()
            await asyncio.wait_for(task, 1)
            self.assertTrue(cleaned.is_set())
            self.assertTrue(self.client.closed)
