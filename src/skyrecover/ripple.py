"""The Ripple Engine: follow each aircraft through its day and see how delays spread.

Core idea (think of a bus doing a loop):
  * A plane has some spare time on the ground between flights, called SLACK.
    slack = (time between landing and next departure) - (minimum turn time)
  * If a plane lands late by MORE than its slack, the next flight leaves late too.
  * That leftover delay keeps passing down the chain until a big enough
    slack soaks it up, or the plane's day ends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyrecover import config

MAX_LINK_HOURS = 12  # ground time longer than this = overnight break, not a connection


def add_rotations(flights: pd.DataFrame, min_turn: int = config.MIN_TURN_MINUTES) -> pd.DataFrame:
    """For every flight, find the SAME plane's previous and next flight.

    Adds columns:
      prev_flight_id / next_flight_id : the linked flights (-1 if none)
      ground_minutes  : scheduled time on the ground before THIS flight
      slack_minutes   : ground_minutes - min_turn (how much delay it can absorb)
      inbound_arr_delay : how late the plane's previous flight arrived
      pushed_delay    : the delay the inbound plane forces onto this flight
                        = max(0, inbound_arr_delay - slack)
    """
    df = flights.sort_values(["tail_number", "sched_dep"]).copy()
    by_plane = df.groupby("tail_number", sort=False)

    prev_id = by_plane["flight_id"].shift()
    prev_dest = by_plane["dest"].shift()
    prev_arr = by_plane["sched_arr"].shift()
    prev_arr_delay = by_plane["arr_delay"].shift()
    prev_cancelled = by_plane["cancelled"].shift()

    ground = (df["sched_dep"] - prev_arr).dt.total_seconds() / 60

    # A real connection = same plane, lands where the next flight leaves from,
    # on the ground for a physically possible, same-day amount of time,
    # and the inbound flight actually flew.
    linked = (
        prev_id.notna()
        & (prev_dest == df["origin"])
        & (ground >= 0)
        & (ground <= MAX_LINK_HOURS * 60)
        & (prev_cancelled == False)  # noqa: E712  (pandas needs == here)
    )

    df["prev_flight_id"] = np.where(linked, prev_id, -1).astype(int)
    df["ground_minutes"] = ground.where(linked)
    # Slack can't be negative: if the airline scheduled a turn shorter than our
    # 35-minute average, that plane/airport can clearly turn faster, so it just
    # has zero spare time (it can't ADD delay to an on-time arrival).
    df["slack_minutes"] = (ground - min_turn).clip(lower=0).where(linked)
    df["inbound_arr_delay"] = prev_arr_delay.where(linked)
    df["pushed_delay"] = (df["inbound_arr_delay"] - df["slack_minutes"]).clip(lower=0)

    # next link is just the reverse of the prev link
    nxt = df.loc[df["prev_flight_id"] >= 0, ["prev_flight_id", "flight_id"]]
    next_map = dict(zip(nxt["prev_flight_id"], nxt["flight_id"]))
    df["next_flight_id"] = df["flight_id"].map(next_map).fillna(-1).astype(int)

    return df.sort_values("flight_id").reset_index(drop=True)


def propagate(rot: pd.DataFrame, flight_id: int, extra_arrival_delay: float) -> pd.DataFrame:
    """What if `flight_id` arrived `extra_arrival_delay` minutes late?

    Walks down the plane's chain and returns every later flight that gets
    knocked late, with how many minutes. Stops as soon as slack absorbs it.
    """
    by_id = rot.set_index("flight_id")
    rows = []
    carried = float(extra_arrival_delay)
    current = flight_id
    while carried > 0:
        nxt = int(by_id.at[current, "next_flight_id"])
        if nxt < 0:
            break  # plane's day is over
        slack = by_id.at[nxt, "slack_minutes"]
        carried = max(0.0, carried - slack)
        if carried <= 0:
            break  # slack soaked it all up
        f = by_id.loc[nxt]
        rows.append({
            "flight_id": nxt, "flight_no": f["flight_no"], "origin": f["origin"],
            "dest": f["dest"], "sched_dep": f["sched_dep"],
            "slack_minutes": slack, "added_delay": carried,
        })
        current = nxt  # the delay passes through the flight roughly unchanged
    return pd.DataFrame(rows, columns=[
        "flight_id", "flight_no", "origin", "dest", "sched_dep", "slack_minutes", "added_delay"])


def top_spreaders(rot: pd.DataFrame, n: int = 10, min_delay: float = 30) -> pd.DataFrame:
    """Which real late arrivals caused the most downstream delay?

    For every flight that actually arrived `min_delay`+ minutes late, run the
    ripple forward and total up the minutes it pushed onto later flights.
    """
    by_id = rot.set_index("flight_id")
    next_id = by_id["next_flight_id"].to_dict()
    slack = by_id["slack_minutes"].to_dict()

    results = []
    late = rot[(rot["arr_delay"] >= min_delay) & (~rot["cancelled"])]
    for fid, delay in zip(late["flight_id"], late["arr_delay"]):
        carried, cur, hit, total = float(delay), fid, 0, 0.0
        while True:
            nxt = next_id.get(cur, -1)
            if nxt < 0:
                break
            carried = max(0.0, carried - slack[nxt])
            if carried <= 0:
                break
            hit += 1
            total += carried
            cur = nxt
        if hit:
            results.append((fid, delay, hit, total))

    out = pd.DataFrame(results, columns=["flight_id", "arr_delay", "flights_hit", "minutes_pushed"])
    out = out.merge(
        rot[["flight_id", "date", "flight_no", "tail_number", "origin", "dest", "sched_dep"]],
        on="flight_id")
    return out.sort_values("minutes_pushed", ascending=False).head(n).reset_index(drop=True)


def validate_against_bts(rot: pd.DataFrame) -> dict:
    """Check our ripple estimate against what the airline itself reported.

    BTS asks airlines to say WHY each delayed flight was late. One cause is
    "Late Aircraft" = the plane showed up late from its previous flight.
    That's exactly what our `pushed_delay` estimates, so we can compare.
    BTS only fills in causes for flights that ARRIVED 15+ minutes late.
    """
    d = rot[(rot["arr_delay"] >= config.DELAY_THRESHOLD_MINUTES) & rot["pushed_delay"].notna()]
    ours = d["pushed_delay"]
    bts = d["cause_late_aircraft"]
    return {
        "flights_compared": int(len(d)),
        "correlation": round(float(np.corrcoef(ours, bts)[0, 1]), 3),
        "our_total_minutes": int(ours.sum()),
        "bts_total_minutes": int(bts.sum()),
        "agree_on_ripple_or_not": round(float(((ours > 0) == (bts > 0)).mean()), 3),
    }
