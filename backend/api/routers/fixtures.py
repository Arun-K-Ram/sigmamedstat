"""
CRIP-X Fixtures Router

GET  /fixtures         → list all available fixtures
GET  /fixtures/{id}    → run a fixture through the pipeline
"""

from fastapi import APIRouter, HTTPException
from crip_x.ingestion.fixture_loader import (
    load_fixture, get_signal_array,
    get_neighboring_arrays, get_motion_array,
    list_fixtures, FIXTURES_DIR
)
from api.schemas.response import FixtureListResponse, FixtureSummary
from crip_x.utils.logger import get_logger

router = APIRouter(prefix="/fixtures", tags=["fixtures"])
logger = get_logger(__name__)


@router.get("", response_model=FixtureListResponse)
async def list_all_fixtures():
    """List all available test fixtures."""
    fixture_files = list_fixtures()
    summaries = []

    for filename in fixture_files:
        try:
            fixture = load_fixture(filename)
            summaries.append(FixtureSummary(
                fixture_id=fixture["fixture_id"],
                description=fixture["description"],
                signal_type=fixture["signal_type"],
                n_samples=len(fixture["signal"]),
                has_clinical_events=len(
                    fixture.get("clinical_events", [])
                ) > 0,
                has_motion_signal="motion_signal" in fixture,
                expected_outcome=fixture.get("expected_outcome", {}),
            ))
        except Exception as e:
            logger.warning(f"Could not load fixture {filename}: {e}")

    return FixtureListResponse(fixtures=summaries, total=len(summaries))


@router.get("/{fixture_id}")
async def get_fixture(fixture_id: str):
    """Get raw fixture data by ID."""
    fixture_files = list_fixtures()

    for filename in fixture_files:
        try:
            fixture = load_fixture(filename)
            if fixture["fixture_id"] == fixture_id:
                return fixture
        except Exception:
            continue

    raise HTTPException(
        status_code=404,
        detail=f"Fixture '{fixture_id}' not found"
    )