"""The Recovery Optimizer: when something breaks, re-plan the operation.

The question it answers:
    "A plane is stuck (or a hub is slowed down). Which plane should fly which
     flight, when should each flight leave, and which flights (if any) should
     we cancel, so that total delay is as small as possible?"

How: a CONSTRAINT PROGRAMMING model, solved with Google OR-Tools (CP-SAT).
Instead of telling the computer HOW to find the plan, we describe:
  * the DECISIONS  : which aircraft flies each flight, how late it leaves, cancel or not
  * the RULES      : a plane can only leave from where it is, needs a minimum
                     turn between flights, can't be in two places at once
  * the GOAL       : minimize delay minutes + a penalty per cancellation
                     + a penalty for planes ending the day in the wrong city
and the solver searches millions of combinations for the best plan.

Each aircraft's day is modeled as a ROUTE through flights (like a delivery truck
visiting stops), which is the classic "vehicle routing" trick in CP-SAT.

Times: every connection happens AT ONE AIRPORT (land at X, leave from X), and
both times are already on X's local clock thanks to the time-zone fix in
clean.py. So no time-zone conversion is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from ortools.sat.python import cp_model

from skyrecover import config

MAX_DELAY = 180          # beyond this, a flight should be cancelled instead
CANCEL_PENALTY = 300     # a cancellation "costs" as much as 300 minutes of delay
WRONG_END_PENALTY = 120  # plane finishes the day in the wrong city (hurts tomorrow)
SWAP_PENALTY = 20        # each swap = moving crews, gates, bags; real ops centers avoid needless swaps


@dataclass
class Flight:
    fid: int
    flight_no: int
    origin: str
    dest: str
    dep: int                 # scheduled departure, minutes after midnight (local)
    arr: int                 # scheduled arrival, minutes after midnight (local)
    tail: str                # aircraft originally planned
    root_delay: int = 0      # delay that happens no matter what (e.g. ground delay at hub)


@dataclass
class Aircraft:
    tail: str
    start_airport: str
    ready: int               # earliest time it can leave start_airport
    planned_end: str         # where it's supposed to finish the day


@dataclass
class Problem:
    flights: list[Flight]
    aircraft: list[Aircraft]
    min_turn: int = config.MIN_TURN_MINUTES


@dataclass
class Plan:
    status: str
    rows: list[dict] = field(default_factory=list)   # one per flight
    bound: float | None = None    # best score the solver PROVED is possible

    @property
    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary(self) -> dict:
        t = self.table
        flown = t[~t["cancelled"]]
        return {
            "flights": len(t),
            "cancelled": int(t["cancelled"].sum()),
            "delay_minutes": int(flown["delay"].sum()),
            "flights_delayed_15": int((flown["delay"] >= 15).sum()),
            "flights_reassigned": int((flown["new_tail"] != flown["tail"]).sum()),
            "wrong_end_of_day": int(self.rows[0].get("_wrong_end", 0)) if self.rows else 0,
            "score": int(flown["delay"].sum() + CANCEL_PENALTY * t["cancelled"].sum()
                         + SWAP_PENALTY * (flown["new_tail"] != flown["tail"]).sum()
                         + WRONG_END_PENALTY * (self.rows[0].get("_wrong_end", 0) if self.rows else 0)),
        }


def solve(problem: Problem, allow_swaps: bool = True, time_limit: float = 20.0) -> Plan:
    m = cp_model.CpModel()
    F, A, turn = problem.flights, problem.aircraft, problem.min_turn
    n = len(F)

    delay = [m.NewIntVar(f.root_delay, max(f.root_delay, MAX_DELAY), f"delay_{i}") for i, f in enumerate(F)]
    cancel = [m.NewBoolVar(f"cancel_{i}") for i in range(n)]
    assign = {}           # (flight index, aircraft index) -> bool
    wrong_end = []

    for a_idx, ac in enumerate(A):
        # Nodes: 0 = "depot" (start/end of the plane's day), 1..n = flights
        arcs = []
        used_here = []
        for i, f in enumerate(F):
            if not allow_swaps and f.tail != ac.tail:
                continue
            x = m.NewBoolVar(f"assign_{i}_{a_idx}")
            assign[(i, a_idx)] = x
            used_here.append(i)
            arcs.append((i + 1, i + 1, x.Not()))   # self-loop = this plane skips this flight

        if not used_here:
            continue

        # depot -> first flight: plane must start where it is, when it's ready
        ends = []
        for i in used_here:
            f = F[i]
            if f.origin == ac.start_airport and f.dep + MAX_DELAY >= ac.ready:
                lit = m.NewBoolVar(f"first_{i}_{a_idx}")
                arcs.append((0, i + 1, lit))
                m.Add(f.dep + delay[i] >= ac.ready).OnlyEnforceIf(lit)
            # last flight -> depot: remember where the plane ends
            end_lit = m.NewBoolVar(f"last_{i}_{a_idx}")
            arcs.append((i + 1, 0, end_lit))
            ends.append((end_lit, f.dest))

        # flight i -> flight j: same plane flies j right after i
        for i in used_here:
            for j in used_here:
                if i == j:
                    continue
                fi, fj = F[i], F[j]
                if fi.dest != fj.origin:
                    continue                      # plane would be in the wrong city
                if fj.dep + MAX_DELAY < fi.arr + turn:
                    continue                      # impossible even with max delay
                if fj.dep < fi.dep:
                    continue                      # keep the day moving forward
                lit = m.NewBoolVar(f"seq_{i}_{j}_{a_idx}")
                arcs.append((i + 1, j + 1, lit))
                # j leaves after i lands (+ its delay) + minimum turn
                m.Add(fj.dep + delay[j] >= fi.arr + delay[i] + turn).OnlyEnforceIf(lit)

        # plane not used at all today
        idle = m.NewBoolVar(f"idle_{a_idx}")
        arcs.append((0, 0, idle))
        m.AddCircuit(arcs)

        # wrong end-of-day city penalty
        bad = m.NewBoolVar(f"wrong_end_{a_idx}")
        for lit, city in ends:
            if city != ac.planned_end:
                m.AddImplication(lit, bad)
        if ac.start_airport != ac.planned_end:
            m.AddImplication(idle, bad)
        wrong_end.append(bad)

    # every flight is either flown by exactly one plane, or cancelled
    for i in range(n):
        planes = [assign[(i, a)] for a in range(len(A)) if (i, a) in assign]
        m.AddExactlyOne(planes + [cancel[i]])
        # a cancelled flight's delay doesn't count
        m.Add(delay[i] == F[i].root_delay).OnlyEnforceIf(cancel[i])

    swaps = [x for (i, a_idx), x in assign.items() if A[a_idx].tail != F[i].tail]
    m.Minimize(sum(delay) + CANCEL_PENALTY * sum(cancel)
               + SWAP_PENALTY * sum(swaps) + WRONG_END_PENALTY * sum(wrong_end))

    # Head start: tell the solver about the original plan (each flight on its
    # planned aircraft, delayed just enough). It's a valid starting point, so
    # the search starts from "the schedule as planned" and improves from there.
    start = {r["flight_id"]: r for r in do_nothing(problem).rows}
    for i, f in enumerate(F):
        m.AddHint(delay[i], min(max(start[f.fid]["delay"], f.root_delay), max(f.root_delay, MAX_DELAY)))
        for a_idx, ac in enumerate(A):
            if (i, a_idx) in assign:
                m.AddHint(assign[(i, a_idx)], ac.tail == f.tail)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 8
    status = solver.Solve(m)
    name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Plan(status=name)

    n_wrong = sum(solver.Value(b) for b in wrong_end)
    rows = []
    for i, f in enumerate(F):
        new_tail = None
        for a_idx, ac in enumerate(A):
            if (i, a_idx) in assign and solver.Value(assign[(i, a_idx)]):
                new_tail = ac.tail
        c = bool(solver.Value(cancel[i]))
        d = 0 if c else solver.Value(delay[i])
        rows.append({
            "flight_id": f.fid, "flight_no": f.flight_no, "origin": f.origin, "dest": f.dest,
            "sched_dep": f.dep, "new_dep": None if c else f.dep + d, "delay": d,
            "tail": f.tail, "new_tail": new_tail, "cancelled": c,
        })
    rows[0]["_wrong_end"] = n_wrong if rows else 0
    # If the time limit hit before the solver PROVED its plan is the best,
    # it still knows a lower bound: "no plan can score better than X".
    # That tells us how close to perfect our plan is.
    return Plan(status=name, rows=rows, bound=solver.BestObjectiveBound())


def do_nothing(problem: Problem) -> Plan:
    """Baseline: every plane keeps its original flights and just waits.

    No cancellations, no swaps, delays ripple forward as far as they go
    (this is the ripple engine's logic). Delays can exceed MAX_DELAY here,
    because nobody is making decisions.
    """
    by_tail = {}
    for i, f in enumerate(problem.flights):
        by_tail.setdefault(f.tail, []).append(i)
    ready = {ac.tail: ac.ready for ac in problem.aircraft}

    rows = [None] * len(problem.flights)
    for tail, idxs in by_tail.items():
        t_free = ready.get(tail, -10**9)          # when the plane can next leave
        for i in sorted(idxs, key=lambda k: problem.flights[k].dep):
            f = problem.flights[i]
            leave = max(f.dep + f.root_delay, t_free)
            d = leave - f.dep
            t_free = f.arr + d + problem.min_turn
            rows[i] = {
                "flight_id": f.fid, "flight_no": f.flight_no, "origin": f.origin, "dest": f.dest,
                "sched_dep": f.dep, "new_dep": leave, "delay": d,
                "tail": f.tail, "new_tail": f.tail, "cancelled": False,
            }
    rows[0]["_wrong_end"] = 0
    return Plan(status="BASELINE", rows=rows)


# ---------------------------------------------------------------------------
# Building a real scenario from the flights table
# ---------------------------------------------------------------------------

def _minutes(ts: pd.Series, day: pd.Timestamp) -> pd.Series:
    return ((ts - day).dt.total_seconds() // 60).astype(int)


def build_scenario(flights: pd.DataFrame, date: str, hub: str, start: str,
                   hours: float = 3, window_hours: float = 8, pool: int = 12,
                   aog_tail: str | None = None, ground_delay: int = 0) -> tuple[Problem, dict]:
    """Cut a realistic recovery problem out of one real day of Delta flights.

    Disruptions (pick one or both):
      * aog_tail     : this aircraft is stuck at the hub for `hours` from `start`
                       (if "auto", picks the hub aircraft with the most flights left)
      * ground_delay : every departure from the hub in the next `hours` gets at
                       least this many minutes of delay (like an FAA ground stop)

    To keep it solvable in seconds, only a POOL of aircraft is included: the
    disrupted one plus the planes passing through the hub with the most spare
    time. Real airline recovery tools also limit the swap pool.
    """
    day = pd.Timestamp(date)
    t0 = day + pd.Timedelta(start + ":00") if ":" in start else day + pd.Timedelta(hours=float(start))
    t_end = t0 + pd.Timedelta(hours=window_hours)
    d = flights[(flights["sched_dep"] >= t0) & (flights["sched_dep"] < t_end) & (~flights["cancelled"])].copy()

    # aircraft that are at the hub during the window (their first window flight leaves the hub)
    first = d.sort_values("sched_dep").groupby("tail_number").head(1)
    at_hub = first[first["origin"] == hub]
    if at_hub.empty:
        raise ValueError(f"No aircraft depart {hub} in that window.")

    counts = d.groupby("tail_number").size()
    if aog_tail == "auto":
        aog_tail = counts.reindex(at_hub["tail_number"]).idxmax()

    # rank the other hub planes by how much spare ground time they have (good swap partners)
    d_sorted = d.sort_values(["tail_number", "sched_dep"])
    nxt_dep = d_sorted.groupby("tail_number")["sched_dep"].shift(-1)
    gap = (nxt_dep - d_sorted["sched_arr"]).dt.total_seconds() / 60
    slack = gap.groupby(d_sorted["tail_number"]).sum()
    candidates = [t for t in slack.reindex(at_hub["tail_number"]).sort_values(ascending=False).index
                  if t != aog_tail]
    tails = ([aog_tail] if aog_tail else []) + candidates[: pool - (1 if aog_tail else 0)]

    sel = d[d["tail_number"].isin(tails)].sort_values("sched_dep")
    dep_m = _minutes(sel["sched_dep"], day)
    arr_m = _minutes(sel["sched_arr"], day)
    gd_end = (t0 + pd.Timedelta(hours=hours))

    fl = []
    for (_, r), dm, am in zip(sel.iterrows(), dep_m, arr_m):
        root = ground_delay if (ground_delay and r["origin"] == hub and r["sched_dep"] < gd_end) else 0
        fl.append(Flight(int(r["flight_id"]), int(r["flight_no"]), r["origin"], r["dest"],
                         int(dm), int(am), r["tail_number"], root))

    acs = []
    for tail in tails:
        mine = sel[sel["tail_number"] == tail]
        first_dep = int(_minutes(mine["sched_dep"].head(1), day).iloc[0])
        ready = first_dep
        if tail == aog_tail:
            ready = max(first_dep, int(_minutes(pd.Series([t0]), day).iloc[0] + hours * 60))
        acs.append(Aircraft(tail, mine["origin"].iloc[0], ready, mine["dest"].iloc[-1]))

    info = {"date": date, "hub": hub, "start": f"{t0:%H:%M}", "aog_tail": aog_tail,
            "aog_hours": hours if aog_tail else 0, "ground_delay": ground_delay,
            "aircraft": len(acs), "flights": len(fl)}
    return Problem(fl, acs), info


def optimality_gap(plan: Plan) -> float | None:
    """How far the plan's score could be from the best possible (0.0 = proven best)."""
    if plan.bound is None or not plan.rows:
        return None
    score = plan.summary()["score"]
    return 0.0 if score == 0 else max(0.0, (score - plan.bound) / score)


def compare(problem: Problem, time_limit: float = 20.0) -> dict[str, Plan]:
    """Run all three strategies on the same problem."""
    return {
        "do_nothing": do_nothing(problem),
        "delay_or_cancel_only": solve(problem, allow_swaps=False, time_limit=time_limit),
        "skyrecover": solve(problem, allow_swaps=True, time_limit=time_limit),
    }
