"""Command line: `skyrecover <command>`. Run `skyrecover --help` to see them all."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from skyrecover import config


def cmd_download(args):
    from skyrecover.data.download import download
    download(args.year, args.months)


def cmd_prepare(args):
    from skyrecover.data.clean import prepare
    raw = sorted(config.RAW_DIR.glob("bts_*.zip"))
    if not raw:
        sys.exit("No raw files in data/raw. Run `skyrecover download` first.")
    df = prepare(raw, airline=args.airline)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FLIGHTS_FILE, index=False)
    print(f"Saved {len(df):,} {args.airline} flights to {config.FLIGHTS_FILE}")
    print(f"  {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}, "
          f"{df['tail_number'].nunique():,} aircraft, {df['origin'].nunique()} airports")


def load_flights() -> pd.DataFrame:
    if not config.FLIGHTS_FILE.exists():
        sys.exit(f"{config.FLIGHTS_FILE} not found. Run `skyrecover prepare` first.")
    return pd.read_parquet(config.FLIGHTS_FILE)


def cmd_ripple(args):
    from skyrecover.ripple import add_rotations, propagate, top_spreaders, validate_against_bts
    rot = add_rotations(load_flights())

    if args.flight_id is not None:
        f = rot.set_index("flight_id").loc[args.flight_id]
        print(f"What if DL{f['flight_no']} {f['origin']}->{f['dest']} "
              f"({f['sched_dep']:%b %d %H:%M}) lands {args.delay:.0f} min late?\n")
        hit = propagate(rot, args.flight_id, args.delay)
        if hit.empty:
            print("No ripple: the next flight's slack absorbs it.")
        else:
            print(hit.to_string(index=False))
            print(f"\n{len(hit)} later flights delayed, {hit['added_delay'].sum():,.0f} minutes total")
        return

    v = validate_against_bts(rot)
    print("Validation against airline-reported 'Late Aircraft' delays:")
    print(f"  flights compared:        {v['flights_compared']:,}")
    print(f"  correlation:             {v['correlation']}")
    print(f"  agree on ripple yes/no:  {v['agree_on_ripple_or_not']:.1%}")
    print(f"  minutes, ours vs BTS:    {v['our_total_minutes']:,} vs {v['bts_total_minutes']:,}\n")

    print(f"Top {args.top} delay spreaders:")
    ts = top_spreaders(rot, n=args.top)
    ts["sched_dep"] = ts["sched_dep"].dt.strftime("%b %d %H:%M")
    print(ts[["flight_id", "sched_dep", "flight_no", "origin", "dest", "arr_delay",
              "flights_hit", "minutes_pushed"]].to_string(index=False))


def cmd_train(args):
    from skyrecover.features import build_features
    from skyrecover.forecast import save, train_and_evaluate
    print("Building features (only info known 60 min before departure)...")
    feats = build_features(load_flights())
    print(f"Training on data before {args.test_from}, testing on {args.test_from} onward...")
    results, trained = train_and_evaluate(feats, test_from=args.test_from)
    save(results, trained)

    d = results["data"]
    print(f"\nTested on {d['test_flights']:,} flights it never saw ({d['test_range']})\n")
    print(f"{'model':<24}{'avg error (min)':>16}{'AUC':>8}{'precision':>11}{'recall':>8}")
    for name, m in results["models"].items():
        print(f"{name:<24}{m['mae_minutes']:>16}{m['auc']:>8}{m['precision']:>11}{m['recall']:>8}")
    print(f"\nSaved models to {config.MODELS_DIR}/ and the report to {config.REPORTS_DIR}/forecast_report.md")


def cmd_recover(args):
    from skyrecover.optimize import build_scenario, compare, optimality_gap
    if not args.aog_tail and not args.ground_delay:
        args.aog_tail = "auto"
    problem, info = build_scenario(
        load_flights(), args.date, args.hub, args.start, hours=args.hours,
        pool=args.pool, aog_tail=args.aog_tail, ground_delay=args.ground_delay)

    what = []
    if info["aog_tail"]:
        what.append(f"aircraft {info['aog_tail']} stuck at {args.hub} for {args.hours:g}h")
    if info["ground_delay"]:
        what.append(f"every {args.hub} departure delayed {info['ground_delay']} min for {args.hours:g}h")
    print(f"Scenario: {args.date} {info['start']}, " + " + ".join(what))
    print(f"Recovery pool: {info['aircraft']} aircraft, {info['flights']} flights\n")

    plans = compare(problem, time_limit=args.time_limit)
    labels = {"do_nothing": "Do nothing (wait)", "delay_or_cancel_only": "Delay/cancel, no swaps",
              "skyrecover": "SkyRecover (swaps allowed)"}
    print(f"{'strategy':<28}{'delay min':>10}{'cancelled':>11}{'reassigned':>12}{'score':>8}{'gap':>7}")
    for k, p in plans.items():
        if not p.rows:
            print(f"{labels[k]:<28} no solution ({p.status})")
            continue
        s = p.summary()
        gap = optimality_gap(p)
        gap_txt = "" if gap is None else f"{gap:.0%}"
        print(f"{labels[k]:<28}{s['delay_minutes']:>10}{s['cancelled']:>11}"
              f"{s['flights_reassigned']:>12}{s['score']:>8}{gap_txt:>7}")

    base = plans["do_nothing"].summary()["score"]
    best = plans["skyrecover"].summary()["score"] if plans["skyrecover"].rows else base
    if base:
        print(f"\nSkyRecover cuts the disruption score by {(base - best) / base:.0%} vs doing nothing.")
    print("score = delay minutes + 300 per cancellation + 20 per reassigned flight + 120 per plane ending in the wrong city")
    print("gap = how far the plan could be from the mathematically best one (0% = proven best)")

    if args.show_plan and plans["skyrecover"].rows:
        t = plans["skyrecover"].table
        changed = t[(t["delay"] > 0) | t["cancelled"] | (t["new_tail"] != t["tail"])].copy()
        for c in ["sched_dep", "new_dep"]:
            changed[c] = changed[c].map(lambda m: "" if pd.isna(m) else f"{int(m)//60 % 24:02d}:{int(m) % 60:02d}")
        print("\nChanged flights in the SkyRecover plan:")
        print(changed.drop(columns=["_wrong_end", "flight_id"], errors="ignore").to_string(index=False))


def cmd_export_demo(args):
    """Bundle what the website needs into data/demo/ (small enough for GitHub)."""
    import json
    import shutil

    import lightgbm as lgb

    from skyrecover.features import LIVE_FEATURES, SCHEDULE_FEATURES, build_features
    from skyrecover.ripple import add_rotations, validate_against_bts

    flights = load_flights()
    metrics_file = config.REPORTS_DIR / "forecast_metrics.json"
    model_delay = config.MODELS_DIR / "day_of_delay.txt"
    model_late = config.MODELS_DIR / "day_of_late.txt"
    if not (metrics_file.exists() and model_delay.exists()):
        sys.exit("Run `skyrecover train` first.")

    out = Path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.FLIGHTS_FILE, out / "flights.parquet")

    # Day-of model predictions for the TEST month only (flights it never trained on)
    feats = build_features(flights)
    test = feats[feats["sched_dep"] >= pd.Timestamp(args.test_from)]
    cols = SCHEDULE_FEATURES + LIVE_FEATURES
    pred = pd.DataFrame({
        "flight_id": test["flight_id"].to_numpy(),
        "prob_late": lgb.Booster(model_file=str(model_late)).predict(test[cols]),
        "pred_delay": lgb.Booster(model_file=str(model_delay)).predict(test[cols]).clip(0),
    })
    pred.to_parquet(out / "predictions.parquet", index=False)

    metrics = json.loads(metrics_file.read_text())
    metrics["ripple_validation"] = validate_against_bts(add_rotations(flights))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"Wrote {out}/ ({size:.1f} MB): flights, {len(pred):,} predictions, metrics")


def main(argv=None):
    p = argparse.ArgumentParser(prog="skyrecover", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download real BTS on-time data")
    d.add_argument("--year", type=int, required=True)
    d.add_argument("--months", type=int, nargs="+", required=True)
    d.set_defaults(func=cmd_download)

    pr = sub.add_parser("prepare", help="clean raw files into data/processed/flights.parquet")
    pr.add_argument("--airline", default="DL", help="airline code (default DL = Delta)")
    pr.set_defaults(func=cmd_prepare)

    r = sub.add_parser("ripple", help="validate the ripple model and show top delay spreaders")
    r.add_argument("--top", type=int, default=10)
    r.add_argument("--flight-id", type=int, help="simulate a delay on one flight")
    r.add_argument("--delay", type=float, default=60, help="minutes late (with --flight-id)")
    r.set_defaults(func=cmd_ripple)

    t = sub.add_parser("train", help="train and test the delay forecaster")
    t.add_argument("--test-from", default="2024-03-01", help="first date of the test period")
    t.set_defaults(func=cmd_train)

    rc = sub.add_parser("recover", help="simulate a disruption and compare recovery strategies")
    rc.add_argument("--date", default="2024-03-15")
    rc.add_argument("--hub", default="ATL")
    rc.add_argument("--start", default="14:00", help="when the disruption starts (HH:MM)")
    rc.add_argument("--hours", type=float, default=3, help="how long it lasts")
    rc.add_argument("--aog-tail", help="aircraft stuck on the ground ('auto' = pick one)")
    rc.add_argument("--ground-delay", type=int, default=0, help="minutes added to every hub departure")
    rc.add_argument("--pool", type=int, default=12, help="how many aircraft the optimizer may use")
    rc.add_argument("--time-limit", type=float, default=60, help="seconds per solve (stops early once proven best)")
    rc.add_argument("--show-plan", action="store_true", help="print every changed flight")
    rc.set_defaults(func=cmd_recover)

    ex = sub.add_parser("export-demo", help="bundle data + predictions for the website")
    ex.add_argument("--test-from", default="2024-03-01")
    ex.set_defaults(func=cmd_export_demo)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
