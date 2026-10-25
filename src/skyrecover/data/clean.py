"""Turn raw BTS files into one clean flights table.

Raw BTS data is messy in a few specific ways this module fixes:
  * times are written as numbers like 5 (00:05) or 1430 (14:30), and even 2400
  * arrival can be "earlier" than departure when a flight lands after midnight
  * cancelled flights have no actual times
  * column names are long and inconsistent
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Raw BTS column -> our clean name. Only the columns we actually use.
COLUMNS = {
    "FlightDate": "date",
    "Reporting_Airline": "airline",
    "Tail_Number": "tail_number",
    "Flight_Number_Reporting_Airline": "flight_no",
    "Origin": "origin",
    "Dest": "dest",
    "CRSDepTime": "crs_dep_hhmm",
    "CRSArrTime": "crs_arr_hhmm",
    "CRSElapsedTime": "crs_elapsed",
    "DepDelay": "dep_delay",
    "ArrDelay": "arr_delay",
    "Cancelled": "cancelled",
    "Diverted": "diverted",
    "Distance": "distance",
    "CarrierDelay": "cause_carrier",
    "WeatherDelay": "cause_weather",
    "NASDelay": "cause_nas",
    "SecurityDelay": "cause_security",
    "LateAircraftDelay": "cause_late_aircraft",
}

# The final table every other module relies on
CLEAN_COLUMNS = [
    "flight_id", "date", "airline", "tail_number", "flight_no", "origin", "dest",
    "sched_dep", "sched_arr", "dep_delay", "arr_delay", "cancelled", "distance",
    "cause_carrier", "cause_weather", "cause_nas", "cause_security", "cause_late_aircraft",
]


def hhmm_to_minutes(hhmm: pd.Series) -> pd.Series:
    """1430 -> 870 minutes after midnight. 2400 is treated as 24:00 (= next day 00:00)."""
    v = pd.to_numeric(hhmm, errors="coerce")
    return (v // 100) * 60 + (v % 100)


def read_raw_zip(path: Path) -> pd.DataFrame:
    """Read the CSV inside one BTS zip, keeping only the columns we need."""
    with zipfile.ZipFile(path) as z:
        csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(csv_name) as f:
            return pd.read_csv(f, usecols=list(COLUMNS), low_memory=False)


def clean(raw: pd.DataFrame, airline: str | None = "DL") -> pd.DataFrame:
    """Raw BTS rows -> clean flights table (one row per scheduled flight)."""
    df = raw.rename(columns=COLUMNS)

    if airline:
        df = df[df["airline"] == airline]

    # Diverted flights landed somewhere else, so their arrival delay is meaningless.
    df = df[df["diverted"].fillna(0) == 0]
    # Without a tail number we can't follow the aircraft, so we can't model ripples.
    df = df[df["tail_number"].notna() & (df["tail_number"].astype(str).str.strip() != "")]

    date = pd.to_datetime(df["date"])
    dep_min = hhmm_to_minutes(df["crs_dep_hhmm"])
    df = df.assign(
        date=date.dt.normalize(),
        sched_dep=date + pd.to_timedelta(dep_min, unit="m"),
    )

    # Scheduled arrival, in LOCAL time at the destination airport.
    #
    # Gotcha: BTS times are local to each airport. A flight leaving Atlanta at
    # 10:00 Eastern and landing in Los Angeles after 4h50m arrives at 11:50
    # Pacific, NOT 14:50. If we just did "departure + flight time" we'd be mixing
    # time zones, and the plane's ground time in LA would come out 3 hours wrong.
    #
    # So we use the reported local arrival clock time (CRSArrTime) and only use
    # "departure + flight time" to figure out WHICH DAY it lands (red-eyes land
    # the next morning). Time-zone gaps are always < 12 hours, so rounding to
    # the nearest whole day picks the right date.
    elapsed = pd.to_numeric(df["crs_elapsed"], errors="coerce")
    arr_min = hhmm_to_minutes(df["crs_arr_hhmm"])
    elapsed = elapsed.fillna((arr_min - dep_min) % (24 * 60))
    naive_arr = df["sched_dep"] + pd.to_timedelta(elapsed, unit="m")   # departure-zone clock
    arr_same_day = date + pd.to_timedelta(arr_min, unit="m")           # arrival-zone clock, guessing same day
    day_shift = ((naive_arr - arr_same_day) / pd.Timedelta(days=1)).round()
    df["sched_arr"] = arr_same_day + pd.to_timedelta(day_shift, unit="D")

    df["cancelled"] = df["cancelled"].fillna(0).astype(bool)
    for c in ["cause_carrier", "cause_weather", "cause_nas", "cause_security", "cause_late_aircraft"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["tail_number"] = df["tail_number"].astype(str).str.strip()
    df["flight_no"] = pd.to_numeric(df["flight_no"], errors="coerce").astype("Int64")

    df = df.sort_values(["sched_dep", "tail_number"]).reset_index(drop=True)
    df["flight_id"] = np.arange(len(df))
    return df[CLEAN_COLUMNS]


def prepare(raw_files: list[Path], airline: str | None = "DL") -> pd.DataFrame:
    """Read every downloaded month and combine into one clean table."""
    frames = []
    for path in sorted(raw_files):
        print(f"  reading {path.name}")
        frames.append(clean(read_raw_zip(path), airline=airline))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["sched_dep", "tail_number"]).reset_index(drop=True)
    df["flight_id"] = np.arange(len(df))
    return df
