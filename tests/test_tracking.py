"""Synthetic trajectories test the policy, not wildlife detector accuracy."""
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wildlife_tracker.tracking import AnimalTracker, TrackingPolicy


def animal(x=.8, y=.5, label="deer", score=.8, area=.1):
    return {"label": label, "score": score, "area": area, "center": [x, y]}


class TrackingTests(unittest.TestCase):
    def tracker(self, **options):
        return AnimalTracker(TrackingPolicy(smoothing=1, **options))

    def acquire(self, tracker, value=None):
        value = animal() if value is None else value
        outputs = [tracker.update([value], t) for t in (0, .33, .66)]
        self.assertEqual([x["action"] for x in outputs[:2]], ["none", "none"])
        return outputs[-1]

    def test_confirm_before_follow_and_finish_proposes_stop(self):
        tracker = self.tracker()
        decision = self.acquire(tracker)
        self.assertEqual((decision["action"], decision["horizontal"]), ("move", "right"))
        self.assertIs(decision["control_enabled"], False)
        self.assertEqual(tracker.finish()["action"], "stop")
        self.assertEqual(tracker.finish()["action"], "none")

    def test_keep_original_when_larger_distant_animal_appears(self):
        tracker = self.tracker()
        original = self.acquire(tracker)["target_id"]
        result = tracker.update([animal(.15, area=.4, score=.99), animal(.78)], 1)
        self.assertEqual(result["target_id"], original)
        self.assertEqual(tracker.target["raw_center"], (.78, .5))

    def test_crossing_ambiguity_stops_and_requires_fresh_confirmation(self):
        tracker = self.tracker()
        self.acquire(tracker)
        result = tracker.update([animal(.78), animal(.82)], 1)
        self.assertEqual((result["reason"], result["action"]), ("ambiguous_targets", "stop"))
        self.assertEqual(tracker.update([animal(.8)], 1.33)["action"], "none")
        self.assertEqual(tracker.update([animal(.8)], 1.66)["action"], "move")

    def test_missing_frame_stops_without_immediate_target_switch(self):
        tracker = self.tracker()
        original = self.acquire(tracker)["target_id"]
        result = tracker.update([animal(.1, label="raccoon")], 1)
        self.assertEqual((result["state"], result["action"]), ("HOLD", "stop"))
        self.assertEqual(result["target_id"], original)
        self.assertEqual(tracker.update([], 1.33)["action"], "none")
        self.assertEqual(tracker.update([], 2.67)["state"], "LOST")
        replacement = tracker.update([animal(.1, label="raccoon")], 3)
        self.assertEqual(replacement["state"], "ACQUIRING")
        self.assertNotEqual(replacement["target_id"], original)
        self.assertEqual(replacement["return_home"], "blocked_unverified")

    def test_size_jump_does_not_steal_target(self):
        tracker = self.tracker()
        self.acquire(tracker)
        self.assertEqual(tracker.update([animal(area=.8)], 1)["reason"], "target_not_matched")

    def test_species_flip_keeps_geometric_track_and_votes_label(self):
        # Real S340 footage: one deer labelled dog/cat/squirrel frame to frame.
        tracker = self.tracker()
        original = self.acquire(tracker, animal(label="deer"))["target_id"]
        for t, label in ((1, "dog"), (1.33, "cat"), (1.66, "squirrel")):
            result = tracker.update([animal(.79, label=label, score=.5)], t)
            self.assertEqual(result["target_id"], original)
        self.assertEqual(result["label"], "deer")  # 3 × 0.8 outweighs single 0.5 votes

    def test_multi_label_duplicate_box_is_one_object(self):
        tracker = self.tracker()
        self.acquire(tracker)
        result = tracker.update([animal(.79, label="dog", score=.5), animal(.79, label="deer", score=.45)], 1)
        self.assertNotEqual(result["reason"], "ambiguous_targets")
        self.assertEqual(result["state"], "TRACKING")

    def test_low_scores_continue_but_never_start_or_move(self):
        tracker = self.tracker()
        for t in (0, .33, .66, 1):
            self.assertEqual(tracker.update([animal(score=.2)], t)["state"], "EMPTY")  # below 0.3 to start
        tracker = self.tracker()
        tracker.update([animal(score=.35)], 0)
        for t in (.33, .66, 1):
            result = tracker.update([animal(score=.15)], t)  # ≥0.12 continues the track
        self.assertEqual((result["state"], result["reason"], result["action"]),
                         ("TRACKING", "awaiting_confident_detection", "none"))
        self.assertEqual(tracker.update([animal(score=.45)], 1.33)["action"], "move")  # best now ≥0.4

    def test_recorded_deer_sequence_is_tracked(self):
        # Detection numbers from recorded clip 15b8ff38 (deer walking left), floor 0.1, 3 FPS.
        sequence = [(5.67, "cat", .10, .70, .59, .023), (6.00, "cat", .12, .68, .61, .030),
                    (6.33, "dog", .18, .66, .63, .040), (6.67, "dog", .41, .63, .66, .041),
                    (7.00, "cat", .20, .61, .66, .060), (7.33, "squirrel", .15, .56, .69, .048),
                    (7.67, "dog", .63, .53, .73, .068), (8.00, "squirrel", .31, .48, .75, .061),
                    (8.33, "squirrel", .24, .45, .78, .070), (8.67, "squirrel", .45, .40, .80, .068),
                    (9.00, "dog", .60, .36, .84, .084), (9.33, "dog", .49, .32, .86, .079),
                    (9.67, "dog", .17, .28, .87, .073), (10.00, "squirrel", .25, .24, .89, .064),
                    (10.33, "dog", .37, .20, .90, .056), (10.67, "dog", .30, .17, .92, .049),
                    (11.00, "cat", .24, .12, .92, .039)]
        tracker = AnimalTracker()
        states = [tracker.update([animal(x, y, label, score, area)], t)
                  for t, label, score, x, y, area in sequence]
        self.assertEqual(len({s["target_id"] for s in states if s["target_id"]}), 1)
        self.assertGreaterEqual(sum(s["state"] == "TRACKING" for s in states), 6)
        self.assertTrue(any(s["action"] == "move" and s["horizontal"] == "left" for s in states))

    def test_dead_zone_hysteresis_suppresses_boundary_jitter(self):
        tracker = self.tracker()
        self.acquire(tracker, animal(.67))
        # Stay moving until entering the smaller inner zone.
        self.assertNotEqual(tracker.update([animal(.60)], 1)["action"], "stop")
        self.assertEqual(tracker.update([animal(.57)], 1.33)["action"], "stop")
        for i, x in enumerate((.59, .64, .60, .64)):
            self.assertEqual(tracker.update([animal(x)], 1.66 + i * .33)["action"], "none")

    def test_reversal_passes_through_stop(self):
        tracker = self.tracker(association_distance=.5)
        self.acquire(tracker, animal(.7))
        result = tracker.update([animal(.3)], 1.5)
        self.assertEqual((result["action"], result["reason"]), ("stop", "direction_reversal"))
        result = tracker.update([animal(.3)], 1.83)
        self.assertEqual((result["action"], result["horizontal"]), ("move", "left"))

    def test_proposal_rate_bound_over_long_jittered_sequence(self):
        tracker = self.tracker()
        rng = random.Random(7)
        times = []
        now = 0
        for _ in range(1000):
            now += rng.uniform(.01, .15)
            result = tracker.update([animal(.8 + rng.uniform(-.015, .015))], now)
            self.assertFalse(result["control_enabled"])
            if result["action"] == "move":
                times.append(now)
        self.assertGreater(len(times), 10)
        self.assertTrue(all(b - a >= .75 for a, b in zip(times, times[1:])))

    def test_single_axis_centering_stops_old_diagonal_before_rate_limit(self):
        tracker = self.tracker(association_distance=.5)
        self.acquire(tracker, animal(.8, .8))
        result = tracker.update([animal(.8, .57)], 1)
        self.assertEqual((result["action"], result["reason"]), ("stop", "axis_entered_dead_zone"))

    def test_frame_gap_and_stale_inference_pause(self):
        tracker = self.tracker()
        self.acquire(tracker)
        self.assertEqual(tracker.update([animal()], 1.8)["reason"], "frame_gap")
        tracker = self.tracker()
        self.acquire(tracker)
        self.assertEqual(tracker.update([animal()], 2.2, observed_at=.5)["reason"], "stale_result")
        self.assertEqual(tracker.update([animal()], 2.8)["state"], "LOST")

    def test_duplicate_frame_cannot_confirm_and_bad_clock_is_rejected(self):
        tracker = self.tracker()
        tracker.update([animal()], 0)
        self.assertEqual(tracker.update([animal()], .2, observed_at=0)["reason"], "repeated_or_old_frame")
        for now, observed in ((math.nan, 0), (.1, .1), (1, 2)):
            with self.assertRaises(ValueError):
                tracker.update([animal()], now, observed)

    def test_invalid_low_confidence_and_non_animal_inputs_never_acquire(self):
        tracker = self.tracker()
        values = [animal(score=.2), animal(label="person"), animal(x=math.nan), animal(area=0),
                  animal(x=2), {"label": "deer"}, None]
        for i in range(5):
            result = tracker.update(values, i * .3)
            self.assertEqual((result["state"], result["action"]), ("EMPTY", "none"))

    def test_invalid_policy_rejected(self):
        for values in ({"confirm_frames": 1}, {"smoothing": math.nan}, {"exit_dead_zone": .2}, {"proposal_interval": 0},
                       {"acquire_confidence": .5}, {"continue_confidence": .35}, {"duplicate_distance": .05}):
            with self.assertRaises(ValueError):
                TrackingPolicy(**values)
