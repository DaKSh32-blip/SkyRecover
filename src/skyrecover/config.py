"""Shared settings in one place, so every module agrees on paths and rules."""

from pathlib import Path

# Project folders (relative to wherever you run the commands from)
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FLIGHTS_FILE = PROCESSED_DIR / "flights.parquet"
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# Minimum time a plane needs on the ground between flights
# (deplane, clean, refuel, board). Real values vary by aircraft and airport;
# 35 minutes is a reasonable narrow-body average for a first model.
MIN_TURN_MINUTES = 35

# A flight counts as "delayed" at 15+ minutes. This is the official
# US DOT definition used in all on-time statistics.
DELAY_THRESHOLD_MINUTES = 15

# Very large delays are rare and usually caused by something unusual
# (mechanical failure, crew timeout). We cap the training target so a few
# 10-hour outliers don't dominate the model.
DELAY_CAP_MINUTES = 300
