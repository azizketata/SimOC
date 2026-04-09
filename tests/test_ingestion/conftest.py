"""Shared fixtures for ingestion tests."""

from pathlib import Path

import pytest

from simoc.ingestion import OCELData, load_ocel

FIXTURE_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture(scope="session")
def sample_ocel_path() -> Path:
    """Path to the sample OCEL 2.0 test log."""
    return FIXTURE_DIR / "sample_order_process.json"


@pytest.fixture(scope="session")
def loaded_data(sample_ocel_path: Path) -> OCELData:
    """Load the sample log once for the entire test session."""
    return load_ocel(str(sample_ocel_path))
