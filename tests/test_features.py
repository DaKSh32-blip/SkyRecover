"""The most important tests in the project: the model must never see the future."""

import numpy as np
import pandas as pd

from skyrecover.features import build_features


def flights(rows):
    t = pd.Timestamp("2024-01-05")
    out = []
    for i, r in enumerate(rows):
        dep = t + pd.Timedelta(minutes=r["dep"])
        out.append({
            "flight_id": i, "date": t, "airline": "DL", "tail_number": r.get("tail", "N1"),
            "flight_no": 100 + i, "origin": r["o"], "dest": r["d"],
            "sched_dep": dep, "sched_arr": dep + pd.Timedelta(minutes=r["block"]),
            "block_minutes": r["block"], "dep_delay": r["dd"], "arr_delay": r["ad"],
            "cancelled": False, "distance": 500,
            "cause_carrier": 0, "cause_weather": 0, "cause_nas": 0, "cause_security": 0,
            "cause_late_aircraft": 0,
        })
    return pd.DataFrame(out)


def test_inbound_arrival_hidden_if_it_lands_after_prediction_time():
    # Inbound lands 10:00 + 50 min late = 10:50.
    # Next flight departs 11:30, so prediction time is 10:30. It hasn't landed yet.
    f = build_features(flights([
        {"o": "MCO", "d": "ATL", "dep": 8 * 60, "block": 120, "dd": 50, "ad": 50},
        {"o": "ATL", "d": "BOS", "dep": 11 * 60 + 30, "block": 150, "dd": 40, "ad": 40},
    ])).set_index("flight_no")
    assert np.isnan(f.at[101, "inbound_arr_delay_known"])   # not known yet
    assert f.at[101, "inbound_dep_delay_known"] == 50       # but it DID depart late, that's known


def test_inbound_arrival_visible_once_it_has_landed():
    # Same, but the next flight departs at 13:00 (prediction 12:00). Inbound landed at 10:50.
    f = build_features(flights([
        {"o": "MCO", "d": "ATL", "dep": 8 * 60, "block": 120, "dd": 50, "ad": 50},
        {"o": "ATL", "d": "BOS", "dep": 13 * 60, "block": 150, "dd": 0, "ad": 0},
    ])).set_index("flight_no")
    assert f.at[101, "inbound_arr_delay_known"] == 50


def test_airport_recent_delay_ignores_flights_that_have_not_left_yet():
    rows = [
        # another plane at ATL: departs 9:00 with a 30 min delay (left 9:30)
        {"tail": "N2", "o": "ATL", "d": "MCO", "dep": 9 * 60, "block": 90, "dd": 30, "ad": 30},
        # another plane at ATL: departs 11:00 with a 200 min delay (leaves 14:20, in the future)
        {"tail": "N3", "o": "ATL", "d": "LGA", "dep": 11 * 60, "block": 120, "dd": 200, "ad": 200},
        # our flight: departs 11:30, prediction at 10:30
        {"tail": "N1", "o": "ATL", "d": "BOS", "dep": 11 * 60 + 30, "block": 150, "dd": 0, "ad": 0},
    ]
    f = build_features(flights(rows)).set_index("flight_no")
    assert f.at[102, "origin_recent_delay"] == 30   # only the flight that already left
    assert f.at[102, "origin_recent_count"] == 1


def test_targets():
    f = build_features(flights([
        {"o": "ATL", "d": "BOS", "dep": 600, "block": 150, "dd": 20, "ad": 10},
        {"tail": "N2", "o": "ATL", "d": "MCO", "dep": 700, "block": 90, "dd": -5, "ad": -10},
    ])).set_index("flight_no")
    assert f.at[100, "target_late"] == 1 and f.at[100, "target_delay"] == 20
    assert f.at[101, "target_late"] == 0 and f.at[101, "target_delay"] == 0   # early counts as 0
