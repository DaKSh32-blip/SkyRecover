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

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
