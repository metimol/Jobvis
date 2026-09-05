"""Tests for LLM-based Arbeitsagentur search query generation."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from app.models.profile import CVAnalysis, Profile
from app.schemas.job import BAJobListing
from app.services.query_generator import (
    BAQueryParams,
    generate_ba_query,
    generate_search_query,
)
from app.services.scheduler import MatchingSchedulerService


@pytest.mark.asyncio
async def test_llm_query_minijob_retail_marketing():
    """Verify LLM extracts 'was', 'arbeitszeit', and 'angebotsart' from 'I want a minijob in retail and marketing'."""
    mock_response = json.dumps(
        {
            "was": "Einzelhandel Marketing",
            "wo": None,
            "arbeitszeit": "mj",
            "angebotsart": 1,
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="I want a minijob in retail and marketing",
        cv_profile={"skills": ["Customer Service", "Sales"], "experience_years": 2.0},
        user_prefs={"location": "Berlin", "desired_job_type": "all"},
        llm=mock_llm,
    )

    assert isinstance(res, BAQueryParams)
    assert "Einzelhandel" in res.was or "Marketing" in res.was
    assert res.arbeitszeit == "mj"
    assert res.angebotsart == 1
    # Fallback to profile location when not specified in goals
    assert res.wo == "Berlin"


@pytest.mark.asyncio
async def test_llm_query_apprenticeship_with_location():
    """Verify LLM extracts angebotsart=4 (Ausbildung) and targeted location."""
    mock_response = json.dumps(
        {
            "was": "Fachinformatiker",
            "wo": "München",
            "arbeitszeit": "vz",
            "angebotsart": 4,
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="Ich suche eine Ausbildung zum Fachinformatiker in München",
        cv_profile={"skills": ["Python", "Linux"], "experience_years": 0.5},
        llm=mock_llm,
    )

    assert res.was == "Fachinformatiker"
    assert res.wo == "München"
    assert res.angebotsart == 4
    assert res.arbeitszeit == "vz"


@pytest.mark.asyncio
async def test_llm_query_part_time_accounting_in_cologne():
    """Verify LLM extracts arbeitszeit='tz' (Teilzeit) and location 'Köln'."""
    mock_response = json.dumps(
        {
            "was": "Buchhalterin",
            "wo": "Köln",
            "arbeitszeit": "tz",
            "angebotsart": 1,
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="Teilzeitstelle als Buchhalterin im Raum Köln",
        cv_profile={"skills": ["DATEV", "Buchhaltung"], "experience_years": 5.0},
        llm=mock_llm,
    )

    assert res.was == "Buchhalterin"
    assert res.wo == "Köln"
    assert res.arbeitszeit == "tz"
    assert res.angebotsart == 1


@pytest.mark.asyncio
async def test_llm_query_remote_software_developer():
    """Verify LLM extracts arbeitszeit='ho' (Homeoffice) and preserves wo=None even when profile has location."""
    mock_response = json.dumps(
        {
            "was": "Python Backend Developer",
            "wo": None,
            "arbeitszeit": "ho",
            "angebotsart": 1,
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="Senior Python Backend Developer (remote / Homeoffice)",
        cv_profile={"skills": ["Python", "FastAPI", "Docker"], "experience_years": 6.0},
        user_prefs={"location": "Hamburg", "desired_job_type": "all"},
        llm=mock_llm,
    )

    assert "Python" in res.was
    assert res.arbeitszeit == "ho"
    assert res.angebotsart == 1
    assert res.wo is None


@pytest.mark.asyncio
async def test_llm_query_ukrainian_natural_language():
    """Verify handling of Ukrainian goals translated to German standard BA parameters."""
    mock_response = json.dumps(
        {
            "was": "Fahrer",
            "wo": "Berlin",
            "arbeitszeit": "vz",
            "angebotsart": 1,
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="Шукаю роботу водієм у Берліні",
        cv_profile={"skills": ["Водій"], "experience_years": 5.0},
        llm=mock_llm,
    )

    assert res.was == "Fahrer"
    assert res.wo == "Berlin"
    assert res.arbeitszeit == "vz"
    assert res.angebotsart == 1


@pytest.mark.asyncio
async def test_llm_query_markdown_code_block_and_none_string_sanitization():
    """Verify markdown code block stripping and conversion of string 'None' to actual None."""
    markdown_response = (
        "```json\n"
        + json.dumps(
            {
                "was": "Pflegefachkraft",
                "wo": "None",
                "arbeitszeit": "vollzeit",
                "angebotsart": 1,
            }
        )
        + "\n```"
    )
    mock_llm = FakeListLLM(responses=[markdown_response])

    res = await generate_search_query(
        goals="Examinierte Pflegefachkraft gesucht",
        cv_profile={"skills": ["Altenpflege"], "experience_years": 3.0},
        user_prefs={"location": "Bremen"},
        llm=mock_llm,
    )

    assert res.was == "Pflegefachkraft"
    assert res.arbeitszeit == "vz"  # mapped from 'vollzeit'
    assert res.wo == "Bremen"  # 'None' string cleaned, fell back to profile location


@pytest.mark.asyncio
async def test_seamless_empty_and_missing_goals():
    """Verify empty or None goals seamlessly fall back to CV skills and preferences without invoking LLM."""
    # Subtest 1: None goals
    res_none = await generate_search_query(
        goals=None,
        cv_profile={"skills": ["Elektriker", "SPS-Programmierung"], "experience_years": 4.0},
        user_prefs={"location": "Hamburg", "desired_job_type": "vz"},
        llm=None,
    )
    assert res_none.was == "Elektriker SPS-Programmierung"
    assert res_none.wo == "Hamburg"
    assert res_none.arbeitszeit == "vz"
    assert res_none.angebotsart == 1

    # Subtest 2: Empty string goals
    res_empty = await generate_search_query(
        goals="   ",
        cv_profile={"skills": ["Koch", "Gastronomie"]},
        user_prefs={"location": "Dresden", "desired_job_type": "tz"},
        llm=None,
    )
    assert "Koch" in res_empty.was
    assert res_empty.wo == "Dresden"
    assert res_empty.arbeitszeit == "tz"
    assert res_empty.angebotsart == 1

    # Subtest 3: No CV skills, only keywords
    res_kw = await generate_search_query(
        goals="",
        cv_profile={"skills": [], "keywords": ["Pflegefachkraft", "Geriatrie"]},
        user_prefs={"location": "Bremen", "desired_job_type": "mj"},
        llm=None,
    )
    assert "Pflegefachkraft" in res_kw.was
    assert res_kw.wo == "Bremen"
    assert res_kw.arbeitszeit == "mj"
    assert res_kw.angebotsart == 1


@pytest.mark.asyncio
async def test_heuristic_fallback_when_llm_raises_error():
    """Verify that if LLM raises a network or runtime exception, the service seamlessly falls back to heuristics."""
    error_llm = RunnableLambda(
        lambda _x: (_ for _ in ()).throw(RuntimeError("Google GenAI 503 Service Unavailable"))
    )

    res = await generate_search_query(
        goals="I want a minijob in retail and marketing",
        cv_profile={"skills": ["Customer Service"], "experience_years": 1.0},
        user_prefs={"location": "Berlin", "desired_job_type": "all"},
        llm=error_llm,
    )

    assert isinstance(res, BAQueryParams)
    assert "Einzelhandel" in res.was or "Marketing" in res.was
    assert res.arbeitszeit == "mj"
    assert res.angebotsart == 1
    assert res.wo == "Berlin"


@pytest.mark.asyncio
async def test_scheduler_integration_with_llm_query():
    """Verify scheduler integration passes generated parameters to ArbeitsagenturClient search_jobs."""
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.rollback = AsyncMock()

    user_id = "test-user-llm-1"
    profile = Profile(
        user_id=user_id,
        goals="I want a minijob in retail and marketing",
        location="Berlin",
        radius_km=25,
        desired_job_type="all",
        german_level="B1",
    )
    cv = CVAnalysis(
        user_id=user_id,
        raw_text="Retail experience",
        skills=["Retail", "Marketing"],
        experience_years=2.0,
    )

    # Setup DB mock returns
    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_scalars.first.side_effect = [
        profile,  # Profile lookup
        cv,  # CVAnalysis lookup
        None,  # Settings lookup
        None,  # Job existence check
        None,  # MatchedJob existence check
    ]
    mock_scalars.all.return_value = []
    mock_db.execute.return_value = mock_result

    mock_ba = AsyncMock()
    mock_ba.search_jobs.return_value = [
        BAJobListing(
            ref_nr="TEST-REF-999",
            title="Aushilfe im Einzelhandel (Minijob)",
            employer="Retail Store GmbH",
            location="Berlin",
            working_time="Minijob",
            description="Minijob im Einzelhandel und Marketing.",
            external_url="https://jobboerse.arbeitsagentur.de/job/999",
        )
    ]

    mock_llm_response = json.dumps(
        {
            "was": "Einzelhandel Marketing",
            "wo": "Berlin",
            "arbeitszeit": "mj",
            "angebotsart": 1,
        }
    )
    mock_llm_sched = FakeListLLM(responses=[mock_llm_response])

    async def mock_gen(*args, **kwargs):
        kwargs["llm"] = mock_llm_sched
        return await generate_search_query(*args, **kwargs)

    with patch("app.services.query_generator.generate_search_query", side_effect=mock_gen):
        scheduler = MatchingSchedulerService()
        result = await scheduler.run_sync_for_user(user_id, mock_db, ba_client=mock_ba)

        assert result["status"] == "success"
        assert result["scraped"] == 1
        assert result["matched"] == 1

        # Verify search_jobs was invoked with targeted parameters
        mock_ba.search_jobs.assert_awaited_once()
        call_kwargs = mock_ba.search_jobs.call_args.kwargs
        assert (
            "Einzelhandel" in call_kwargs["query"]
            or "Retail" in call_kwargs["query"]
            or "Marketing" in call_kwargs["query"]
        )
        assert call_kwargs["location"] == "Berlin"
        assert call_kwargs["radius_km"] == 25
        assert call_kwargs["arbeitszeit"] == "mj"
        assert call_kwargs["angebotsart"] == 1


def test_convenience_helpers_and_alias():
    """Verify dictionary conversions, item access, containment, and convenience aliases."""
    params = BAQueryParams(was="Tester", wo="Berlin", arbeitszeit="vz", angebotsart=1)
    p_dict = params.to_dict()
    assert p_dict == {
        "was": "Tester",
        "wo": "Berlin",
        "arbeitszeit": "vz",
        "angebotsart": 1,
    }
    assert params["was"] == "Tester"
    assert params["wo"] == "Berlin"
    assert "was" in params
    assert params.get("arbeitszeit") == "vz"
    assert params.get("nonexistent", "fallback") == "fallback"
    assert generate_ba_query is generate_search_query
