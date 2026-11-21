import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_model import (decision_summary, episode_stats, freshness,  # noqa: E402
                             movement_summary, screen_state, time_in_state)


def ep(i, dur, rec, analyzed=True, cue=100):
    return {"id": i, "timestamp": f"14:0{i}:00", "duration_sec": dur, "pre_freeze_cadence": 98,
            "cue_bpm": cue, "recovery_sec": rec, "analyzed": analyzed}


BASE = {"status": "WALKING", "freeze_index": 0.3, "bpm": 100, "haptic": 0,
        "alert_level": "normal", "caretaker_brief": "", "episodes": []}


class ScreenState(unittest.TestCase):
    def test_steady_when_nothing_happening(self):
        s = screen_state(BASE, age_sec=0.2)
        self.assertEqual(s["key"], "steady")
        self.assertEqual(s["headline"], "All steady")

    def test_offline_wins_over_everything(self):
        st = dict(BASE, alert_level="alert", episodes=[ep(1, 6.0, None, False)])
        s = screen_state(st, age_sec=3.5)
        self.assertEqual(s["key"], "offline")
        self.assertIn("No signal", s["headline"])

    def test_offline_boundary_is_three_seconds(self):
        self.assertNotEqual(screen_state(BASE, age_sec=2.99)["key"], "offline")
        self.assertEqual(screen_state(BASE, age_sec=3.0)["key"], "offline")

    def test_ongoing_short_freeze_is_freeze(self):
        st = dict(BASE, status="FREEZE", episodes=[ep(1, 2.0, None, False)])
        s = screen_state(st, age_sec=0.1)
        self.assertEqual(s["key"], "freeze")
        self.assertIn("2.0 s", s["detail"])

    def test_ongoing_freeze_over_4s_escalates_to_alert(self):
        st = dict(BASE, status="FREEZE", episodes=[ep(1, 4.1, None, False)])
        self.assertEqual(screen_state(st, age_sec=0.1)["key"], "alert")

    def test_alert_level_from_advisor_is_alert(self):
        st = dict(BASE, alert_level="alert")
        self.assertEqual(screen_state(st, age_sec=0.1)["key"], "alert")

    def test_watch_level(self):
        st = dict(BASE, alert_level="watch")
        s = screen_state(st, age_sec=0.1)
        self.assertEqual(s["key"], "watch")
        self.assertEqual(s["headline"], "Keep an eye out")

    def test_missing_fields_do_not_crash(self):
        self.assertEqual(screen_state({}, age_sec=0.1)["key"], "steady")


class Freshness(unittest.TestCase):
    def test_live(self):
        self.assertEqual(freshness(0.4), "Live, updated 0.4 s ago")

    def test_seconds(self):
        self.assertEqual(freshness(12), "Last update 12 s ago")

    def test_minutes(self):
        self.assertEqual(freshness(150), "Last update 2 min ago")

    def test_never(self):
        self.assertEqual(freshness(float("inf")), "Waiting for first data")


class EpisodeStats(unittest.TestCase):
    def test_empty(self):
        s = episode_stats([])
        self.assertEqual(s["count"], 0)
        self.assertIsNone(s["avg_recovery"])
        self.assertIsNone(s["last_vs_avg"])

    def test_average_excludes_ongoing(self):
        s = episode_stats([ep(1, 2, 2.0), ep(2, 3, 4.0), ep(3, 1, None, False)])
        self.assertEqual(s["count"], 3)
        self.assertEqual(s["completed"], 2)
        self.assertAlmostEqual(s["avg_recovery"], 3.0)

    def test_last_compared_with_earlier_average(self):
        s = episode_stats([ep(1, 2, 3.0), ep(2, 2, 3.0), ep(3, 2, 1.5)])
        self.assertEqual(s["last_vs_avg"], "faster")
        s = episode_stats([ep(1, 2, 1.5), ep(2, 2, 1.5), ep(3, 2, 3.0)])
        self.assertEqual(s["last_vs_avg"], "slower")
        s = episode_stats([ep(1, 2, 2.0), ep(2, 2, 2.1)])
        self.assertEqual(s["last_vs_avg"], "similar")

    def test_single_episode_has_no_comparison(self):
        self.assertIsNone(episode_stats([ep(1, 2, 2.0)])["last_vs_avg"])


class TimeInState(unittest.TestCase):
    def test_fractions_weighted_by_duration(self):
        hist = [(0.0, "WALKING"), (6.0, "FREEZE"), (8.0, "STATIONARY")]
        f = time_in_state(hist, now=10.0, window=10.0)
        self.assertAlmostEqual(f["WALKING"], 0.6)
        self.assertAlmostEqual(f["FREEZE"], 0.2)
        self.assertAlmostEqual(f["STATIONARY"], 0.2)

    def test_window_clips_old_history(self):
        hist = [(0.0, "FREEZE"), (90.0, "WALKING")]
        f = time_in_state(hist, now=100.0, window=20.0)
        self.assertAlmostEqual(f["FREEZE"], 0.5)
        self.assertAlmostEqual(f["WALKING"], 0.5)

    def test_offline_counts_as_no_signal(self):
        f = time_in_state([(0.0, "OFFLINE")], now=5.0, window=5.0)
        self.assertAlmostEqual(f["OFFLINE"], 1.0)

    def test_empty_history(self):
        self.assertEqual(time_in_state([], now=5.0, window=5.0), {})


class DecisionSummary(unittest.TestCase):
    def test_changes(self):
        e = {"before": {"bpm": 100, "haptic": 0}, "decision": {"bpm": 92, "haptic": 1,
             "signal_uncertain": False}}
        self.assertEqual(decision_summary(e), "Tempo 100 → 92 bpm, wrist vibration on")

    def test_no_change(self):
        e = {"before": {"bpm": 92, "haptic": 1}, "decision": {"bpm": 92, "haptic": 1}}
        self.assertEqual(decision_summary(e), "Kept the current cue")

    def test_uncertain_flag(self):
        e = {"before": {"bpm": 92, "haptic": 1},
             "decision": {"bpm": 92, "haptic": 1, "signal_uncertain": True}}
        self.assertEqual(decision_summary(e), "Kept the current cue, sensor signal flagged as unclear")



class Movement(unittest.TestCase):
    def test_none_without_motion_block(self):
        self.assertIsNone(movement_summary(BASE))

    def test_counts_and_pace(self):
        st = dict(BASE, motion={"steps": 1240, "cadence_spm": 96, "turns": 5, "turns_left": 3,
                                "turns_right": 2, "turning": False})
        m = movement_summary(st)
        self.assertEqual(m["items"], [("Steps", "1,240", "this session"),
                                      ("Walking pace", "96", "steps/min"),
                                      ("Turns", "5", "3 left, 2 right")])
        self.assertIsNone(m["turn_pattern"])

    def test_unknown_pace_is_a_dash_not_zero(self):
        m = movement_summary(dict(BASE, motion={"steps": 0, "cadence_spm": None, "turns": 0}))
        self.assertEqual(m["items"][1][1:], ("–", "not walking now"))

    def test_turn_pattern(self):
        eps = [dict(ep(1, 2.0, 1.0), during_turn=True), dict(ep(2, 2.0, 1.0), during_turn=False),
               dict(ep(3, 2.0, 1.0), during_turn=True)]
        m = movement_summary(dict(BASE, episodes=eps, motion={"steps": 10, "turns": 2}))
        self.assertEqual(m["turn_pattern"], "2 of 3 freezes began during a turn")

    def test_old_episodes_without_field(self):
        m = movement_summary(dict(BASE, episodes=[ep(1, 2.0, 1.0)], motion={"steps": 10, "turns": 1}))
        self.assertIsNone(m["turn_pattern"])

if __name__ == "__main__":
    unittest.main()
