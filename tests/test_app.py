"""Smoke test: the dashboard loads and every tab renders without errors.

Skipped automatically if the demo bundle (data/demo) or streamlit isn't present.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
pytest.importorskip("streamlit")
if not (ROOT / "data" / "demo" / "flights.parquet").exists():
    pytest.skip("demo data not exported", allow_module_level=True)

from streamlit.testing.v1 import AppTest  # noqa: E402


def test_dashboard_renders_without_errors():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
