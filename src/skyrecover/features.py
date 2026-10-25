"""Turn the flights table into model features.

THE RULE: every feature must be something an airline would actually KNOW
at prediction time, which we set to 60 minutes before scheduled departure.
Using information from after that moment is "data leakage": the model would
look great in testing and fail in real life.

Two kinds of features:
  * SCHEDULE features: known days in advance (hour, route, how tight the turn is)
  * LIVE features: known only on the day, as things happen (is the inbound
    plane already late? is the airport having a bad afternoon?)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyrecover import config
from skyrecover.ripple import add_rotations

PREDICTION_LEAD_MINUTES = 60

SCHEDULE_FEATURES = [
    "dep_hour", "day_of_week", "origin", "dest", "distance",
    "leg_of_day", "slack_minutes", "origin_hour_departures", "dest_hour_arrivals",
]
LIVE_FEATURES = [
    "inbound_dep_delay_known", "inbound_arr_delay_known", "inbound_pushed_known",
    "origin_recent_delay", "origin_recent_count",
]
CATEGORICAL = ["origin", "dest"]


def _schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    df["dep_hour"] = df["sched_dep"].dt.hour
    df["day_of_week"] = df["sched_dep"].dt.dayofweek
    # (No "month" feature on purpose: with only a few months of data, a model
    # tested on a month it never trained on can't learn anything real from it.)

    # Which flight of the plane's day is this? Later legs have had more
    # chances to pick up delay along the way.
    df = df.sort_values(["tail_number", "sched_dep"])
    df["leg_of_day"] = df.groupby(["tail_number", df["sched_dep"].dt.date]).cumcount() + 1

    # How busy is the schedule? More Delta departures from the same airport in
    # the same hour = more competition for runways, gates and ground crews.
    hour_key = df["sched_dep"].dt.floor("h")
    df["origin_hour_departures"] = df.groupby(["origin", hour_key])["flight_id"].transform("count")
    arr_hour = df["sched_arr"].dt.floor("h")
    df["dest_hour_arrivals"] = df.groupby(["dest", arr_hour])["flight_id"].transform("count")
    return df


def _inbound_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """What do we know about the inbound plane, 60 minutes before we depart?"""
    pred_time = df["sched_dep"] - pd.Timedelta(minutes=PREDICTION_LEAD_MINUTES)
    prev = df.set_index("flight_id")[["sched_arr", "block_minutes", "dep_delay", "arr_delay"]]
    has_prev = df["prev_flight_id"] >= 0
    p = prev.reindex(df["prev_flight_id"].where(has_prev)).reset_index(drop=True)
    p.index = df.index

    # Time-zone care (same trap as the arrival-time bug): the inbound flight's
    # departure is recorded on the PREVIOUS airport's clock, but prediction
    # time is on THIS airport's clock. Its scheduled arrival is already on our
    # clock, so: departure on our clock = scheduled arrival - scheduled block time.
    inbound_sched_dep_here = p["sched_arr"] - pd.to_timedelta(p["block_minutes"], unit="m")
    inbound_actual_dep = inbound_sched_dep_here + pd.to_timedelta(p["dep_delay"], unit="m")
    inbound_actual_arr = p["sched_arr"] + pd.to_timedelta(p["arr_delay"], unit="m")

    # Only use the inbound plane's delay if it had ALREADY happened by prediction time
    dep_known = has_prev & (inbound_actual_dep <= pred_time)
    arr_known = has_prev & (inbound_actual_arr <= pred_time)
    df["inbound_dep_delay_known"] = p["dep_delay"].where(dep_known)
    df["inbound_arr_delay_known"] = p["arr_delay"].where(arr_known)

    # Best live guess of the delay the inbound plane will push onto us:
    # use the landing delay if it's already landed, otherwise its departure delay.
    best = df["inbound_arr_delay_known"].fillna(df["inbound_dep_delay_known"]).where(dep_known)
    df["inbound_pushed_known"] = (best - df["slack_minutes"]).clip(lower=0)
    return df


def _origin_recent_delay(df: pd.DataFrame, window_hours: int = 3) -> pd.DataFrame:
    """Is this airport having a bad day RIGHT NOW?

    Average departure delay of flights that already left the same airport in
    the last few hours before prediction time. This is how the model "sees"
    storms and ground stops without needing weather data yet.
    """
    pred_time = df["sched_dep"] - pd.Timedelta(minutes=PREDICTION_LEAD_MINUTES)
    df["origin_recent_delay"] = np.nan
    df["origin_recent_count"] = 0

    flown = df[df["dep_delay"].notna()]
    actual_dep = flown["sched_dep"] + pd.to_timedelta(flown["dep_delay"], unit="m")
    window = np.timedelta64(window_hours, "h")

    for origin, idx in df.groupby("origin").groups.items():
        mask = flown["origin"] == origin
        events_t = actual_dep[mask].to_numpy()
        events_d = flown.loc[mask, "dep_delay"].to_numpy()
        order = np.argsort(events_t)
        events_t, events_d = events_t[order], events_d[order]
        csum = np.concatenate([[0.0], np.cumsum(events_d)])

        q = pred_time.loc[idx].to_numpy()
        hi = np.searchsorted(events_t, q, side="right")       # departed by prediction time
        lo = np.searchsorted(events_t, q - window, side="right")
        count = hi - lo
        total = csum[hi] - csum[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(count > 0, total / count, np.nan)
        df.loc[idx, "origin_recent_delay"] = avg
        df.loc[idx, "origin_recent_count"] = count
    return df


def build_features(flights: pd.DataFrame) -> pd.DataFrame:
    """Flights table -> one row per non-cancelled flight with features + targets."""
    df = add_rotations(flights)
    df = _schedule_features(df)
    df = _inbound_live_features(df)
    df = _origin_recent_delay(df)

    df = df[~df["cancelled"] & df["dep_delay"].notna()].copy()
    df["target_delay"] = df["dep_delay"].clip(lower=0, upper=config.DELAY_CAP_MINUTES)
    df["target_late"] = (df["dep_delay"] >= config.DELAY_THRESHOLD_MINUTES).astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df.sort_values("sched_dep").reset_index(drop=True)
