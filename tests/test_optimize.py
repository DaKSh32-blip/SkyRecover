"""Small recovery puzzles where we know the right answer."""

from skyrecover.optimize import Aircraft, Flight, Problem, do_nothing, solve


def two_plane_problem(stuck_until=600):
    # Plane A: ATL->MCO at 8:00 (lands 9:30), MCO->ATL at 10:10. A is stuck until `stuck_until`.
    # Plane B: sitting in ATL all morning, its only flight ATL->BOS at 13:00.
    flights = [
        Flight(0, 100, "ATL", "MCO", dep=480, arr=570, tail="A"),
        Flight(1, 101, "MCO", "ATL", dep=610, arr=700, tail="A"),
        Flight(2, 200, "ATL", "BOS", dep=780, arr=930, tail="B"),
    ]
    aircraft = [
        Aircraft("A", "ATL", ready=stuck_until, planned_end="ATL"),
        Aircraft("B", "ATL", ready=400, planned_end="BOS"),
    ]
    return Problem(flights, aircraft)


def test_do_nothing_waits_for_the_stuck_plane():
    plan = do_nothing(two_plane_problem(stuck_until=600)).table
    assert plan.loc[0, "delay"] == 120          # 8:00 flight waits until 10:00
    # ripples: lands 9:30 + 120 = 11:30, +35 turn = 12:05, scheduled 10:10 -> 115 late
    assert plan.loc[1, "delay"] == 115


def test_swap_rescues_the_morning():
    # B is free at ATL, can fly A's round trip (back by 11:40) and still make 13:00 to BOS.
    plan = solve(two_plane_problem(stuck_until=600))
    t = plan.table
    assert plan.status == "OPTIMAL"
    assert not t["cancelled"].any()
    assert t["delay"].sum() == 0
    assert list(t["new_tail"]) == ["B", "B", "B"]


def test_without_swaps_it_must_delay_or_cancel():
    t = solve(two_plane_problem(stuck_until=600), allow_swaps=False).table
    assert t["delay"].sum() + 300 * t["cancelled"].sum() > 0


def test_plane_cannot_teleport():
    # Only plane is in BOS; the flight leaves from ATL. It can't be flown.
    p = Problem([Flight(0, 1, "ATL", "MCO", 480, 570, "A")],
                [Aircraft("A", "BOS", ready=0, planned_end="BOS")])
    assert solve(p).table["cancelled"].iloc[0]


def test_minimum_turn_time_is_respected():
    # Second flight scheduled only 20 min after landing -> must be pushed to 35 min.
    p = Problem([
        Flight(0, 1, "ATL", "MCO", 480, 570, "A"),
        Flight(1, 2, "MCO", "ATL", 590, 680, "A"),
    ], [Aircraft("A", "ATL", ready=0, planned_end="ATL")])
    t = solve(p).table
    assert t.loc[1, "delay"] == 15


def test_root_delay_is_unavoidable():
    p = Problem([Flight(0, 1, "ATL", "MCO", 480, 570, "A", root_delay=45)],
                [Aircraft("A", "ATL", ready=0, planned_end="MCO")])
    assert solve(p).table.loc[0, "delay"] == 45
