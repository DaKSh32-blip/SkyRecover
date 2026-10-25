import numpy as np
import pandas as pd

from skyrecover.features import build_features
from skyrecover.forecast import time_split, train_and_evaluate


def fake_flights(days=40, planes=30, seed=0):
    """Small random schedule where late inbound planes cause late departures."""
    rng = np.random.default_rng(seed)
    airports = ["ATL", "MCO", "BOS", "LGA", "DTW"]
    rows, fid = [], 0
    for day in pd.date_range("2024-01-01", periods=days):
        for p in range(planes):
            here, t, carried = "ATL", day + pd.Timedelta(hours=6 + p % 4), 0.0
            for _leg in range(4):
                dest = rng.choice([a for a in airports if a != here])
                block = int(rng.integers(60, 180))
                dep_delay = max(-5.0, carried + rng.exponential(8) - 5)
                arr_delay = dep_delay + rng.normal(0, 5)
                rows.append(dict(
                    flight_id=fid, date=day, airline="DL", tail_number=f"N{p}", flight_no=fid,
                    origin=here, dest=dest, sched_dep=t, sched_arr=t + pd.Timedelta(minutes=block),
                    block_minutes=block, dep_delay=round(dep_delay), arr_delay=round(arr_delay),
                    cancelled=False, distance=block * 7, cause_carrier=0, cause_weather=0,
                    cause_nas=0, cause_security=0, cause_late_aircraft=0))
                fid += 1
                ground = int(rng.integers(40, 90))
                carried = max(0.0, arr_delay - (ground - 35))
                t = t + pd.Timedelta(minutes=block + ground)
                here = dest
    return pd.DataFrame(rows)


def test_time_split_never_overlaps():
    f = build_features(fake_flights())
    s = time_split(f, "2024-02-01")
    assert s.train["sched_dep"].max() < s.valid["sched_dep"].min()
    assert s.valid["sched_dep"].max() < s.test["sched_dep"].min()


def test_training_runs_and_day_of_beats_always_on_time():
    results, trained = train_and_evaluate(build_features(fake_flights()), test_from="2024-02-01")
    m = results["models"]
    assert set(m) >= {"always_on_time", "airport_hour_average", "ripple_formula", "planning", "day_of"}
    assert m["day_of"]["mae_minutes"] < m["always_on_time"]["mae_minutes"]
