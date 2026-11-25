"""Animal-following control: closed-loop pan steps inside an owned observer session.

Off unless WILDLIFE_FOLLOW=1, and only with the inspected WS/client versions.
Pan only, by the camera's measured step (S340 built in; others via WILDLIFE_PAN_STEP). Every step is written to a durable ledger
before it is sent and verified by observed view motion afterwards; once the
target is lost the session returns by verified reverse steps before the stream
stops.

Fail closed in two levels:
- no_follow: no new forward steps (native person event, unexplained view motion
  at the start position). A verified return is still allowed.
- uncertain: position no longer known (failed verification, unexplained motion
  away from the start, send error). No more steps in this session, and no return.
  The start view is only ever relative, so the next session takes wherever the
  camera now points as its start; `ptz-ledger-clear` remains for manual recovery.
"""
import json
import os
import time
from pathlib import Path

from .core import log
from .ptz import StepPlanner, StepPolicy
from .supervised import VERIFIED_VERSIONS

PTZ_COMMAND = "device.pan_and_tilt"
PERSON_EVENTS = {"person detected", "stranger person detected"}


class Ledger:
    """Durable net pan/tilt offset from the start view, fsynced like the stream marker."""
    def __init__(self, directory):
        self.path = Path(directory) / "ptz-ledger.json"

    def exists(self):
        return self.path.exists()

    def read(self):
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"state": "unreadable"} if self.path.exists() else None

    def write(self, pan, state, tilt=0):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as file:
            json.dump({"pan": pan, "tilt": tilt, "state": state, "updated": time.time()}, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.path)
        self._sync_dir()

    def clear(self):
        self.path.unlink(missing_ok=True)
        self._sync_dir()

    def _sync_dir(self):
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


# Measured pan content shift per step, normalized image width (ptz-probe, one camera each).
# A model missing here follows only once WILDLIFE_PAN_STEP gives its own measurement.
PAN_STEPS = {"T8170": .118, "SoloCam S340": .118}


def pan_step(config):
    """This camera's measured pan step, or None when it was never measured."""
    return config.pan_step or PAN_STEPS.get(config.device_model)


def follow_available(client, config):
    """Reason following cannot run in this session, or None."""
    if not config.follow:
        return "disabled"
    if pan_step(config) is None:
        return "pan_step_unmeasured"
    if any((client.version or {}).get(k) != v for k, v in VERIFIED_VERSIONS.items()):
        return "unverified_ws_versions"
    return None


class FollowController:
    def __init__(self, client, config, policy=None):
        self.client = client
        self.config = config
        # Closed loop; tilt only once its step and the steps to the upper limit were measured (ptz-probe --axis tilt).
        self.planner = StepPlanner(policy or StepPolicy(
            expected_step=pan_step(config) or StepPolicy.expected_step,
            tilt_enabled=config.tilt_step > 0, tilt_expected_step=config.tilt_step, max_up_steps=config.tilt_up_steps,
            max_steps=config.max_steps, max_net_steps=config.max_net_steps))
        self.ledger = Ledger(config.data_dir)
        self.started = time.monotonic()
        self.no_follow = None
        self.uncertain = None
        self.steps_sent = 0
        self.return_verified = None
        if self.ledger.exists():
            # A previous session lost the position. Waiting for the user cost four days of following
            # (2026-11-07 lock, a deer passed unfollowed on 11-11), so start from the current view instead.
            log("follow_rezeroed", previous=self.ledger.read())
            self.ledger.clear()
        client.extra_commands = frozenset({PTZ_COMMAND})

    @property
    def offset(self):
        return self.planner.ledger["pan"]

    @property
    def tilt(self):
        return self.planner.ledger["tilt"]

    @property
    def moved(self):
        return self.offset != 0 or self.tilt != 0

    def _write(self, state):
        self.ledger.write(self.offset, state, tilt=self.tilt)

    def close(self):
        self.client.extra_commands = frozenset()

    def fail(self, reason):
        """The caller could not finish verification (e.g. return deadline): position unknown."""
        self._mark_uncertain(reason)
        self.return_verified = False

    def _mark_uncertain(self, reason):
        if not self.uncertain:
            self.uncertain = reason
            if self.moved or self.planner.pending is not None:
                self._write("uncertain")
            log("follow_locked", level="uncertain", reason=reason, offset=self.offset, tilt=self.tilt)

    def _stop_following(self, reason):
        if not self.no_follow:
            self.no_follow = reason
            log("follow_locked", level="no_follow", reason=reason, offset=self.offset)

    def _guard(self, camera):
        if any(name in PERSON_EVENTS and at >= self.started - 5 for at, name in self.client.device_events):
            self._stop_following("person_present_native_tracking")
        if (camera or {}).get("moving") and self.planner.pending is None:
            # Not our step: native tracking, wind or a scene change moved the view.
            if self.moved:
                self._mark_uncertain("unexplained_view_motion_away_from_start")
            else:
                self._stop_following("unexplained_view_motion")

    async def _drive(self, decision, now, camera, returning):
        was_pending = self.planner.pending is not None
        plan = self.planner.plan(decision, now, camera, returning=returning)
        if self.planner.measured:
            log("follow_step_measured", returning=returning, verified=not self.planner.locked, **self.planner.measured)
            self.planner.measured = None
        if self.planner.locked:
            self._mark_uncertain(self.planner.locked)
            return plan
        if was_pending and self.planner.pending is None and not plan["command"]:
            # Verified (or, for UP at the mechanical limit, provably not moved and not counted).
            if self.moved:
                self._write("known")
            else:
                self.ledger.clear()
        if plan["command"]:
            # Ledger first: a crash after sending must never look like the start position.
            self._write("pending")
            try:
                await self.client.command(PTZ_COMMAND, serialNumber=self.config.serial,
                                          direction=plan["command"]["direction"])
            except Exception as error:
                # Rejected, timed out or disconnected: whether the camera moved is unknown.
                self._mark_uncertain(f"send_failed:{type(error).__name__}")
                return plan
            self.steps_sent += 1
            log("follow_step", label=plan["command"]["label"], offset=self.offset, tilt=self.tilt, returning=returning)
        return plan

    async def act(self, decision, now, camera):
        """Forward following for one frame."""
        if self.uncertain:
            return None
        self._guard(camera)
        if self.uncertain:
            return None
        if self.no_follow:
            decision = {"action": "none"}  # Still verify any pending step.
        return await self._drive(decision, now, camera, returning=False)

    def returning_needed(self):
        return not self.uncertain and (self.moved or self.planner.pending is not None)

    async def return_step(self, now, camera):
        """Feed one frame while returning; True once home is verified or return is impossible."""
        if self.uncertain:
            self.return_verified = False
            return True
        if not self.moved and self.planner.pending is None:
            self.ledger.clear()
            self.return_verified = True
            return True
        if (camera or {}).get("moving") and self.planner.pending is None:
            self._mark_uncertain("unexplained_view_motion_during_return")
            self.return_verified = False
            return True
        # Pan back first, then tilt (UP is the negative tilt direction, so a negative ledger returns DOWN).
        horizontal = "left" if self.offset > 0 else "right" if self.offset < 0 else "center"
        vertical = "center" if horizontal != "center" else "down" if self.tilt < 0 else "up" if self.tilt > 0 else "center"
        await self._drive({"action": "move", "horizontal": horizontal, "vertical": vertical}, now, camera, returning=True)
        if self.uncertain:
            self.return_verified = False
            return True
        if not self.moved and self.planner.pending is None:
            self.ledger.clear()
            self.return_verified = True
            return True
        return False

    def return_seconds(self):
        p = self.planner.policy
        return (abs(self.offset) + abs(self.tilt) + 1) * (p.motion_timeout + p.settle_timeout) + 2

    def summary(self):
        """Session report; `ledger_present` true means the next session restarts from the current view."""
        return {"enabled": True, "steps_sent": self.steps_sent, "offset": self.offset, "tilt_offset": self.tilt,
                "up_limit_reached": self.planner.up_limit,
                "no_follow": self.no_follow, "uncertain": self.uncertain,
                "return_verified": self.return_verified, "ledger_present": self.ledger.exists()}


async def ledger_clear(config, confirm=None):
    """Supervised: clear a left-over ledger only after the user confirms in the owner App that the camera is home."""
    from .probe import stdin_confirm
    ledger = Ledger(config.data_dir)
    state = ledger.read()
    log("READY", mode="ptz-ledger-clear", ledger=state,
        awaiting="user_confirms_camera_at_home_in_app_then_stdin_HOME")
    if state is None:
        log("ledger_absent")
        return False
    if not await (confirm or stdin_confirm)("HOME", None):
        log("ledger_kept", ledger=state)
        return False
    ledger.clear()
    log("ledger_cleared", previous=state)
    return True
