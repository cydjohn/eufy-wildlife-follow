"""Temporal animal-following policy. Produces proposals only, never device I/O."""
import math
from dataclasses import dataclass

from .motion import MOTION_LABEL
from .vision import ANIMAL_LABELS


@dataclass(frozen=True)
class TrackingPolicy:
    # Recorded S340 clips (deer day/night, coyote) show stable boxes but labels flipping among
    # dog/cat/squirrel/deer with scores 0.1–0.6, so association ignores species and uses
    # hysteresis: start ≥0.3, continue ≥0.12, and no move until the track has scored ≥0.4.
    acquire_confidence: float = .3
    continue_confidence: float = .12
    move_confidence: float = .4
    duplicate_distance: float = .03   # boxes this close are one object with several labels
    confirm_frames: int = 3
    resume_frames: int = 2
    association_distance: float = .22
    ambiguity_margin: float = .04
    maximum_area_ratio: float = 3
    smoothing: float = .5
    enter_dead_zone: float = .15
    exit_dead_zone: float = .08
    proposal_interval: float = .75
    loss_seconds: float = 2
    # After the view moves, frames this long are not held against a track's persistence:
    # blurred, and MotionBlobs is still relearning the background (settle_frames at 3 fps ≈ 1 s).
    view_settle_seconds: float = 1.5
    maximum_unseen_seconds: float = 4  # a view that keeps moving cannot hold a track forever
    maximum_frame_gap: float = 1
    maximum_result_age: float = 1.5
    maximum_camera_shift: float = .3
    # Movement-only evidence (motion.MotionBlobs) may turn the camera only once the track has
    # matched this many frames and actually travelled; a swaying bush flickers in place.
    motion_confirm_frames: int = 5
    minimum_travel: float = .04
    # ...and was matched in most frames since it appeared. 2026-10-15 03:50: in heavy rain an
    # intermittent noise track (≈5 of 13 frames) triggered a step; real animals match nearly every frame.
    motion_persistence: float = .7
    motion_merge_distance: float = .05
    motion_area_ratio: float = 6
    # Whether a track the model never scored ≥ move_confidence may turn the camera on movement alone.
    # Off since MegaDetector: on 14 recorded clips all 15 such moves chased insects or rain near the
    # lens (2026-11-03/04 nights) and none was an animal; identity kept through pans was unchanged (22/34).
    motion_only_moves: bool = False

    def __post_init__(self):
        values = {k: v for k, v in vars(self).items() if k != "motion_only_moves"}
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("Tracking policy values must be positive and finite")
        if not isinstance(self.motion_only_moves, bool):
            raise ValueError("motion_only_moves must be boolean")
        if (not isinstance(self.confirm_frames, int) or not isinstance(self.resume_frames, int)
                or not isinstance(self.motion_confirm_frames, int) or self.motion_confirm_frames < self.confirm_frames
                or self.minimum_travel >= .5 or self.motion_area_ratio < self.maximum_area_ratio
                or self.motion_persistence > 1
                or self.confirm_frames < 2 or self.resume_frames < 2
                or not 0 < self.exit_dead_zone < self.enter_dead_zone < .5
                or not self.continue_confidence <= self.acquire_confidence <= self.move_confidence <= 1
                or self.duplicate_distance >= self.ambiguity_margin
                or self.smoothing > 1 or self.maximum_area_ratio < 1
                or self.maximum_camera_shift >= 1
                or self.maximum_unseen_seconds < self.loss_seconds
                or self.ambiguity_margin >= self.association_distance):
            raise ValueError("Invalid conservative tracking policy")


def _same_object(item, kept, duplicate_distance, motion_distance):
    distance = math.dist(item["center"], kept["center"])
    motion = (item["label"] == MOTION_LABEL, kept["label"] == MOTION_LABEL)
    if all(motion):
        return distance <= motion_distance  # fragments of one moving body
    if any(motion):
        # A moving blob inside a model box is the same animal; their centroids differ.
        model_area = kept["area"] if motion[0] else item["area"]
        return distance <= max(duplicate_distance, .5 * math.sqrt(model_area), motion_distance)
    ratio = item["area"] / kept["area"]
    return distance <= duplicate_distance and .67 <= ratio <= 1.5


def candidates(detections, threshold, duplicate_distance=.03, motion_distance=.05):
    """Valid animal boxes and moving blobs ≥ threshold, with same-object duplicates merged.

    A merged candidate keeps the highest score and records every label's best score,
    so species can be voted over the track instead of trusted per frame. `model_score`
    is the best score from the model itself (0 for movement-only evidence).
    """
    valid = []
    for item in detections:
        try:
            label, score, area = item["label"], item["score"], item["area"]
            x, y = item["center"]
            if label not in ANIMAL_LABELS and label != MOTION_LABEL:
                continue
            if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (score, area, x, y)):
                continue
            if not (threshold <= score <= 1 and 0 < area <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
                continue
            valid.append({"label": label, "score": score, "area": area, "center": (x, y)})
        except (KeyError, TypeError, ValueError):
            continue
    merged = []
    for item in sorted(valid, key=lambda d: (-d["score"], d["label"], d["center"])):
        model_score = 0 if item["label"] == MOTION_LABEL else item["score"]
        for kept in merged:
            if _same_object(item, kept, duplicate_distance, motion_distance):
                kept["votes"][item["label"]] = max(kept["votes"].get(item["label"], 0), item["score"])
                kept["model_score"] = max(kept["model_score"], model_score)
                break
        else:
            merged.append({**item, "votes": {item["label"]: item["score"]}, "model_score": model_score})
    return sorted(merged, key=lambda d: (-d["score"] * d["area"], d["label"], d["center"]))


class AnimalTracker:
    """Conservative single-target geometry, not identity recognition or PTZ control.

    Uncertain crossing/occlusion pauses movement proposals. No return command,
    duration, angle, serial or camera connection exists in this policy.
    A measured whole-frame shift (motion.CameraMotion) moves the stored target
    with the view, so camera motion from any controller does not break the
    association; while the view is moving, no new move is proposed.
    """
    def __init__(self, policy=None):
        self.policy = policy or TrackingPolicy()
        self.target = None
        self.sequence = 0
        self.last_update = None
        self.last_move = -math.inf
        self.last_direction = (0, 0)
        self.axes = (0, 0)
        self.hits = 0
        self.confirmed = False
        self.held = False
        self.view_moved_at = -math.inf

    def _output(self, state, reason, action="none", direction=(0, 0)):
        target = self.target
        # Species is a score-weighted vote over the whole track, and stays tentative.
        label = None
        if target:
            named = {k: v for k, v in target["tally"].items() if k != MOTION_LABEL} or target["tally"]
            label = max(named, key=lambda k: (named[k], k))
        return {"state": state, "reason": reason, "action": action,
                "target_id": target["id"] if target else None,
                "label": label,
                # Smoothed position in the current view (0–1), logged for where-animals-go statistics.
                "center": [round(v, 3) for v in target["center"]] if target else None,
                "horizontal": {-1: "left", 0: "center", 1: "right"}[direction[0]],
                "vertical": {-1: "up", 0: "center", 1: "down"}[direction[1]],
                "control_enabled": False, "return_home": "blocked_unverified"}

    def _stop(self, state, reason):
        action = "stop" if self.last_direction != (0, 0) else "none"
        self.last_direction = self.axes = (0, 0)
        return self._output(state, reason, action)

    def _hold(self, reason):
        self.hits = 0
        self.held = True
        return self._stop("HOLD", reason)

    def _axis(self, error, previous):
        p = self.policy
        if previous == 1 and error > p.exit_dead_zone:
            return 1
        if previous == -1 and error < -p.exit_dead_zone:
            return -1
        return 1 if error > p.enter_dead_zone else -1 if error < -p.enter_dead_zone else 0

    def update(self, detections, now, observed_at=None, camera=None):
        if (not math.isfinite(now) or (self.last_update is not None and now < self.last_update)):
            raise ValueError("Tracking requires finite monotonic timestamps")
        observed_at = now if observed_at is None else observed_at
        if not math.isfinite(observed_at) or observed_at > now:
            raise ValueError("Observation timestamp must be finite and not in the future")
        gap = self.last_update is not None and now - self.last_update > self.policy.maximum_frame_gap
        self.last_update = now
        if (camera or {}).get("moving"):
            self.view_moved_at = observed_at
        # While the view moves the animal is blurred, and movement evidence must relearn the
        # background afterwards (motion.MotionBlobs), so the loss clock restarts when the view
        # stops. Recorded 2026-10-30 deer pans: the blob returned 1–1.7 s after the motion.
        unseen_since = max(self.target["last_seen"], self.view_moved_at) if self.target else None
        if self.target and (now - unseen_since >= self.policy.loss_seconds
                            or now - self.target["last_seen"] >= self.policy.maximum_unseen_seconds):
            lost = self._stop("LOST", "target_timeout")
            self.target = None
            self.hits = 0
            self.confirmed = self.held = False
            return lost  # Do not switch to a different animal in the same update.
        if self.target and observed_at - self.view_moved_at >= self.policy.view_settle_seconds:
            self.target["updates"] += 1  # every observable frame since acquisition, matched or not
        shift = (camera or {}).get("shift")
        if self.target and shift is not None:
            if max(abs(shift[0]), abs(shift[1])) > self.policy.maximum_camera_shift:
                return self._hold("camera_shift_too_large")
            # Content moved with the view; keep the stored track in image coordinates.
            for key in ("center", "raw_center", "origin"):
                self.target[key] = tuple(v + d for v, d in zip(self.target[key], shift))
        if now - observed_at > self.policy.maximum_result_age:
            return self._hold("stale_result") if self.target else self._output("EMPTY", "stale_result")
        if self.target and observed_at <= self.target["last_seen"]:
            return self._hold("repeated_or_old_frame")
        if gap and self.target:
            return self._hold("frame_gap")
        p = self.policy
        threshold = p.continue_confidence if self.target else p.acquire_confidence
        available = candidates(detections, threshold, p.duplicate_distance, p.motion_merge_distance)
        if not self.target:
            if not available:
                return self._output("EMPTY", "no_qualified_animal")
            chosen = available[0]
            self.sequence += 1
            self.target = {**chosen, "id": self.sequence, "last_seen": observed_at,
                           "raw_center": chosen["center"], "origin": chosen["center"],
                           "model_best": chosen["model_score"], "matched": 1, "updates": 1,
                           "tally": dict(chosen["votes"]),
                           "areas": {"motion" if chosen["label"] == MOTION_LABEL else "model": chosen["area"]}}
            self.hits, self.confirmed, self.held = 1, False, False
            return self._output("ACQUIRING", "confirming_target")

        matches = []
        for candidate in available:
            # Species labels flip on real footage; association is geometric only.
            distance = math.dist(candidate["center"], self.target["raw_center"])
            # Sizes compare only like with like: a moving blob covers part of the body, so its area
            # says nothing about a model box (recorded 2026-10-30: a deer tracked as a blob was then
            # boxed as "dog" 10× larger and rejected every other frame). Blobs fragment; more slack.
            kind = "motion" if candidate["label"] == MOTION_LABEL else "model"
            reference = self.target["areas"].get(kind)
            bound = p.motion_area_ratio if kind == "motion" else p.maximum_area_ratio
            sized = reference is None or 1 / bound <= candidate["area"] / reference <= bound
            if distance <= p.association_distance and sized:
                matches.append((distance, candidate))
        matches.sort(key=lambda pair: pair[0])
        if not matches:
            return self._hold("target_not_matched")
        if len(matches) > 1 and matches[1][0] - matches[0][0] <= self.policy.ambiguity_margin:
            return self._hold("ambiguous_targets")
        chosen = matches[0][1]
        alpha = self.policy.smoothing
        center = tuple(alpha * v + (1 - alpha) * old for v, old in zip(chosen["center"], self.target["center"]))
        tally = self.target["tally"]
        for label, score in chosen["votes"].items():
            tally[label] = tally.get(label, 0) + score
        self.target.update({k: chosen[k] for k in ("label", "score", "area")}, center=center,
                           raw_center=chosen["center"], last_seen=observed_at,
                           model_best=max(self.target["model_best"], chosen["model_score"]),
                           matched=self.target["matched"] + 1)
        self.target["areas"]["motion" if chosen["label"] == MOTION_LABEL else "model"] = chosen["area"]
        self.hits += 1
        needed = self.policy.resume_frames if self.confirmed else self.policy.confirm_frames
        if self.hits < needed:
            return self._output("HOLD" if self.held else "ACQUIRING", "confirming_target")
        self.confirmed, self.held = True, False
        if (camera or {}).get("moving"):
            return self._stop("TRACKING", "camera_moving")
        travelled = math.dist(self.target["origin"], self.target["raw_center"])
        persistent = self.target["matched"] / self.target["updates"] >= p.motion_persistence
        moving_animal = (p.motion_only_moves and self.target["matched"] >= p.motion_confirm_frames
                         and travelled >= p.minimum_travel and persistent)
        if self.target["model_best"] < p.move_confidence and not moving_animal:
            # Neither a confident model detection nor a persistent, travelling moving region yet.
            return self._output("TRACKING", "awaiting_confident_detection")
        direction = tuple(self._axis(value - .5, prior) for value, prior in zip(center, self.axes))
        self.axes = direction
        if direction == (0, 0):
            return self._stop("TRACKING", "inside_dead_zone")
        if any(old * new < 0 for old, new in zip(self.last_direction, direction)):
            return self._stop("TRACKING", "direction_reversal")
        if any(old != 0 and new == 0 for old, new in zip(self.last_direction, direction)):
            return self._stop("TRACKING", "axis_entered_dead_zone")
        if now - self.last_move < self.policy.proposal_interval:
            return self._output("TRACKING", "rate_limited")
        self.last_move, self.last_direction = now, direction
        return self._output("TRACKING", "target_off_center", "move", direction)

    def finish(self):
        result = self._stop("ENDED", "session_ended")
        self.target = None
        self.hits = 0
        self.confirmed = self.held = False
        return result
