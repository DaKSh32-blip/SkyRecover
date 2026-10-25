"""Download real on-time flight records from the US Bureau of Transportation Statistics.

Every US airline must report every domestic flight: scheduled time, actual time,
delay, cancellation, and which aircraft (tail number) flew it. BTS publishes one
zip file per month. Each month is roughly 500,000+ flights across all airlines.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

from skyrecover import config

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)


def month_url(year: int, month: int) -> str:
    return BTS_URL.format(year=year, month=month)


def download_month(year: int, month: int, out_dir: Path = config.RAW_DIR) -> Path:
    """Download one month's zip (skips it if we already have it)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"bts_{year}_{month:02d}.zip"
    if target.exists() and zipfile.is_zipfile(target):
        print(f"  already have {target.name}, skipping")
        return target

    url = month_url(year, month)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = target.with_suffix(".part")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB at a time
                f.write(chunk)
    tmp.replace(target)

    if not zipfile.is_zipfile(target):
        target.unlink()
        raise RuntimeError(
            f"{url} did not return a zip file. BTS may have changed the link. "
            "See README section 'Getting the real data' for the manual download."
        )
    return target


def download(year: int, months: list[int], out_dir: Path = config.RAW_DIR) -> list[Path]:
    return [download_month(year, m, out_dir) for m in months]
