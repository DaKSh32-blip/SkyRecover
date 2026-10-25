import pandas as pd

from skyrecover.ripple import add_rotations, propagate


def plane_day():
    """One plane, 3 flights: ATL->MCO->ATL->BOS, with known ground times."""
    t = pd.Timestamp("2024-01-05")
    rows = [
        # id, origin, dest, dep (h:m), arr (h:m), arr_delay
        (0, "ATL", "MCO", (8, 0), (9, 30), 60.0),
        (1, "MCO", "ATL", (10, 20), (11, 50), 0.0),   # ground 50 min -> slack 15
        (2, "ATL", "BOS", (13, 30), (16, 0), 0.0),    # ground 100 min -> slack 65
    ]
    return pd.DataFrame([{
        "flight_id": i, "tail_number": "N1", "flight_no": 100 + i, "origin": o, "dest": d,
        "sched_dep": t + pd.Timedelta(hours=dh, minutes=dm),
        "sched_arr": t + pd.Timedelta(hours=ah, minutes=am),
        "arr_delay": a, "cancelled": False, "date": t,
    } for i, o, d, (dh, dm), (ah, am), a in rows])


def test_links_follow_the_same_plane():
    r = add_rotations(plane_day()).set_index("flight_id")
    assert r.at[0, "prev_flight_id"] == -1
    assert r.at[1, "prev_flight_id"] == 0
    assert r.at[0, "next_flight_id"] == 1
    assert r.at[2, "next_flight_id"] == -1


def test_slack_is_ground_time_minus_min_turn():
    r = add_rotations(plane_day()).set_index("flight_id")
    assert r.at[1, "ground_minutes"] == 50
    assert r.at[1, "slack_minutes"] == 15
    assert r.at[2, "slack_minutes"] == 65


def test_pushed_delay_uses_inbound_lateness():
    # flight 0 landed 60 late, slack before flight 1 is 15 -> 45 pushed
    r = add_rotations(plane_day()).set_index("flight_id")
    assert r.at[1, "pushed_delay"] == 45


def test_propagation_stops_when_slack_absorbs_it():
    r = add_rotations(plane_day())
    hit = propagate(r, 0, 60)
    # 60 - 15 = 45 onto flight 1, then 45 - 65 < 0 so flight 2 is safe
    assert list(hit["flight_id"]) == [1]
    assert hit["added_delay"].iloc[0] == 45


def test_big_delay_reaches_further():
    hit = propagate(add_rotations(plane_day()), 0, 120)
    # 120 - 15 = 105 onto flight 1, 105 - 65 = 40 onto flight 2
    assert list(hit["added_delay"]) == [105, 40]


def test_no_link_when_plane_is_somewhere_else():
    df = plane_day()
    df.loc[1, "origin"] = "TPA"  # plane landed in MCO, can't leave from TPA
    r = add_rotations(df).set_index("flight_id")
    assert r.at[1, "prev_flight_id"] == -1


def test_negative_slack_is_clamped_to_zero():
    df = plane_day()
    df.loc[1, "sched_dep"] = df.loc[0, "sched_arr"] + pd.Timedelta(minutes=25)  # tighter than 35
    r = add_rotations(df).set_index("flight_id")
    assert r.at[1, "slack_minutes"] == 0
