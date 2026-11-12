"""Discrete pan/tilt step planning from tracker proposals. Plans only, never device I/O.

Measured on this S340 (ptz-probe, 2026-10-02 UTC, one sample): one
`device.pan_and_tilt` LEFT/RIGHT is one fixed step that settles by itself, about
0.118–0.119 of the image width; LEFT moves content right; visible motion starts
1.5–1.8 s after the command (includes stream latency) and lasts ≤0.4 s; the
reverse step returned to the start. There is no stop command. ROTATE360 (0) is a
cruise and is never planned; presets/calibrate/settings are outside this module.

Closed loop (default): after a step, the next one waits until the commanded
motion is seen with the expected direction and size and the view is still
again. No observed motion, a mismatched shift or no settling locks the planner
for the session. Every plan is `executable=False`; the observer allowlist does
not contain pan_and_tilt.
"""
import math
from dataclasses import dataclass

# eufy-security-client PanTiltDirection; ROTATE360=0 deliberately absent.
DIRECTIONS = {"left": 1, "right": 2, "up": 3, "down": 4}
AXIS = {"left": ("pan", -1), "right": ("pan", 1), "up": ("tilt", -1), "down": ("tilt", 1)}


@dataclass(frozen=True)
class StepPolicy:
    closed_loop: bool = True       # False only for dry-run logs where the camera never moves
    step_interval: float = 2.5     # open-loop spacing: measured latency 1.5–1.8 s + motion + settle
    motion_timeout: float = 3.5    # closed loop: commanded motion must be seen by then
    settle_timeout: float = 2.0    # closed loop: still again within this after motion is first seen
    settle_frames: int = 2         # consecutive measurable still frames
    expected_step: float = .118    # measured pan content shift per step (normalized width)
    # Accept 0.35×–1.65× of expected_step, and this much cross-axis drift. Was 0.5×–1.5× and 0.03:
    # 2026-11-04 06:05 a verified-looking step (0.156 along) summed 0.034 across from one blurred,
    # low-response frame and locked following; night steps measure 0.13–0.16, daytime ones less.
    step_tolerance: float = .65
    maximum_cross_axis: float = .06
    # Widened 2026-10-30 from 8/3: a deer herd hit the 3-step excursion at 04:22, and all 8 returns verified.
    max_steps: int = 16            # per session (return steps are exempt)
    max_net_steps: int = 5         # excursion from the session start, per axis
    tilt_enabled: bool = False     # requires a measured tilt_expected_step
    tilt_expected_step: float = 0  # measured content shift per tilt step (normalized height); 0 = unknown
    max_up_steps: int = 0          # measured steps from the start view to the mechanical upper limit

    def __post_init__(self):
        if (isinstance(self.tilt_expected_step, bool) or not isinstance(self.tilt_expected_step, (int, float))
                or not math.isfinite(self.tilt_expected_step) or not 0 <= self.tilt_expected_step < .5):
            raise ValueError("tilt_expected_step must be 0 (unknown) or a measured fraction below 0.5")
        if isinstance(self.max_up_steps, bool) or not isinstance(self.max_up_steps, int) or not 0 <= self.max_up_steps <= 8:
            raise ValueError("max_up_steps must be 0–8")
        for value in (self.step_interval, self.motion_timeout, self.settle_timeout, self.expected_step,
                      self.step_tolerance, self.maximum_cross_axis):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Step timing and geometry must be positive and finite")
        if self.step_tolerance >= 1 or self.expected_step >= .5:
            raise ValueError("Step geometry out of range")
        for value in (self.max_steps, self.max_net_steps, self.settle_frames):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 20:
                raise ValueError("Step budgets must be small positive integers")
        if not isinstance(self.tilt_enabled, bool) or not isinstance(self.closed_loop, bool):
            raise ValueError("Flags must be boolean")
        if self.tilt_enabled and self.closed_loop and self.tilt_expected_step == 0:
            raise ValueError("Closed-loop tilt needs a measured tilt_expected_step")


class StepPlanner:
    """One axis per step, bounded excursion, never while the view is moving or unknown."""
    def __init__(self, policy=None):
        self.policy = policy or StepPolicy()
        self.ledger = {"pan": 0, "tilt": 0}
        self.steps = 0
        self.last_step = -math.inf
        self.pending = None   # closed loop: the step whose motion has not yet been verified
        self.locked = None    # reason following stopped for this session
        self.up_limit = False # an UP step was accepted but the view provably did not move
        self.measured = None  # the last finished step's measured shift, for the caller to log once

    def _output(self, reason, label=None):
        command = {"command": "device.pan_and_tilt", "direction": DIRECTIONS[label], "label": label} if label else None
        return {"command": command, "reason": reason, "ledger": dict(self.ledger), "steps": self.steps,
                "executable": False, "closed_loop": self.policy.closed_loop, "locked": self.locked}

    def _observe(self, now, camera):
        """Advance verification of the pending step; returns a blocking reason or None."""
        p, pending = self.policy, self.pending
        shift = (camera or {}).get("shift")
        moving = (camera or {}).get("moving")
        if pending["onset"] is None:
            if moving:
                pending["onset"] = now
            else:
                pending["still"] += moving is False
                pending["unmeasured"] = pending["unmeasured"] or moving is None
            if pending["onset"] is None and now - pending["sent"] > p.motion_timeout:
                if (pending["axis"] == "tilt" and pending["delta"] < 0 and not pending["unmeasured"]
                        and pending["still"] >= p.settle_frames):
                    # Mechanical upper limit (the slope opposite): the step did nothing. Not a fault.
                    self.ledger["tilt"] -= pending["delta"]
                    self.pending = None
                    self.up_limit = True
                    return None
                self.locked = "step_motion_not_observed"
                return self.locked
            if pending["onset"] is not None:
                pending["still"] = 0
        if pending["onset"] is not None:
            if shift is not None:
                pending["shift"] = [a + b for a, b in zip(pending["shift"], shift)]
            pending["still"] = pending["still"] + 1 if moving is False else 0
            if pending["still"] >= p.settle_frames:
                axis_index = 0 if pending["axis"] == "pan" else 1
                along = pending["shift"][axis_index] * -pending["delta"]  # content moves opposite to the view
                across = abs(pending["shift"][1 - axis_index])
                expected = p.expected_step if pending["axis"] == "pan" else p.tilt_expected_step
                # The last step before the upper limit may stop part-way; accept down to 0.2× for UP.
                floor = .2 if pending["axis"] == "tilt" and pending["delta"] < 0 else 1 - p.step_tolerance
                low, high = expected * floor, expected * (1 + p.step_tolerance)
                self.pending = None
                self.measured = {"axis": pending["axis"], "along": round(along, 4), "across": round(across, 4),
                                 "accepted": [round(low, 4), round(high, 4)]}
                if not low <= along <= high or across > p.maximum_cross_axis:
                    self.locked = "step_shift_mismatch"
                    return self.locked
                return None
            if now - pending["onset"] > p.settle_timeout:
                self.locked = "step_not_settled"
                return self.locked
        return "awaiting_step_motion" if pending["onset"] is None else "awaiting_step_settle"

    def plan(self, decision, now, camera=None, returning=False):
        """returning=True: a reverse step toward the session start, exempt from the session step budget."""
        if not math.isfinite(now):
            raise ValueError("Step planning requires a finite timestamp")
        if self.locked:
            return self._output(self.locked)
        if self.pending:
            blocked = self._observe(now, camera)
            if blocked:
                return self._output(blocked)
        if decision.get("action") != "move":
            return self._output("no_move_proposal")
        moving = (camera or {}).get("moving")
        if moving is None:
            return self._output("camera_motion_unknown")
        if moving:
            return self._output("camera_moving")
        if not self.policy.closed_loop and now - self.last_step < self.policy.step_interval:
            return self._output("step_interval")
        if self.steps >= self.policy.max_steps and not returning:
            return self._output("session_step_budget")
        label = decision.get("horizontal")
        if label not in ("left", "right"):
            label = decision.get("vertical")
            if label not in ("up", "down"):
                return self._output("no_axis")
            if not self.policy.tilt_enabled:
                return self._output("tilt_disabled")
        axis, delta = AXIS[label]
        if label == "up" and (self.up_limit or -self.ledger["tilt"] >= self.policy.max_up_steps) and not returning:
            return self._output("upper_limit")
        if returning and abs(self.ledger[axis] + delta) >= abs(self.ledger[axis]):
            return self._output("not_toward_start")
        if abs(self.ledger[axis] + delta) > self.policy.max_net_steps:
            return self._output("excursion_budget")
        self.ledger[axis] += delta
        self.steps += 1
        self.last_step = now
        if self.policy.closed_loop:
            self.pending = {"sent": now, "axis": axis, "delta": delta, "onset": None, "still": 0,
                            "unmeasured": False, "shift": [0.0, 0.0]}
        return self._output("step", label)

    def finish(self):
        """Reverse steps that would undo the excursion; informational until verified."""
        reverse = []
        for axis, (negative, positive) in (("pan", ("left", "right")), ("tilt", ("up", "down"))):
            count = self.ledger[axis]
            reverse += [negative if count > 0 else positive] * abs(count)
        return {"ledger": dict(self.ledger), "steps": self.steps, "reverse_steps": reverse,
                "locked": self.locked, "return_home": "blocked_unverified", "executable": False}
