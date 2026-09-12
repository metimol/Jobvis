"""Tests for Arbeitsagentur REST API integration, authentication, and job search client."""

import json
import urllib.parse
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.runnables import RunnableLambda

from app.models.profile import CVAnalysis, Profile
from app.models.user import User
from app.schemas.job import (
    BADetailedJob,
    BAJobListing,
    BASearchResponse,
    JobSearchParams,
)
from app.services.arbeitsagentur import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    ArbeitsagenturAPIError,
    ArbeitsagenturAuthError,
    ArbeitsagenturClient,
    ArbeitsagenturConnectionError,
    ArbeitsagenturNotFoundError,
    ArbeitsagenturRateLimitError,
    ArbeitsagenturTimeoutError,
)
from app.services.query_generator import (
    BAQueryParams,
    generate_ba_query,
    generate_search_query,
)
from app.services.scheduler import MatchingSchedulerService

pytestmark = pytest.mark.asyncio


# --- Core Arbeitsagentur Client Tests ---


@respx.mock
async def test_search_jobs_with_parameters():
    """Verify search_jobs translates filters into query params and sets required headers."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "stellenangebote": [
                    {
                        "refnr": "10000-1198765432-S",
                        "titel": "Python Backend Entwickler (m/w/d)",
                        "beruf": "Softwareentwickler/in",
                        "arbeitgeber": "Jobvis Tech GmbH",
                        "arbeitsort": {
                            "plz": "10117",
                            "ort": "Berlin",
                            "region": "Berlin",
                            "land": "Deutschland",
                        },
                        "arbeitszeit": "Vollzeit",
                        "arbeitszeitmodell": "vz",
                        "eintrittsdatum": "2026-10-01",
                        "aktuelleVeroeffentlichungsdatum": "2026-08-30",
                        "externeUrl": "https://example.com/job/1",
                    },
                    {
                        "hashId": "hash-998877",
                        "titel": "Frontend Engineer",
                        "arbeitgeber": "Web Solutions AG",
                        "arbeitsort": "80331 München",
                        "arbeitszeitmodell": "tz",
                    },
                ],
                "maxErgebnisse": 42,
                "page": 1,
                "size": 25,
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        jobs = await client.search_jobs(
            query="Python Entwickler",
            location="Berlin",
            radius_km=50,
            arbeitszeit="vz",
            page=1,
            size=25,
        )

    assert route.called
    request = route.calls.last.request
    assert request.headers["X-API-Key"] == DEFAULT_API_KEY
    assert "User-Agent" in request.headers
    assert request.url.params["was"] == "Python Entwickler"
    assert request.url.params["wo"] == "Berlin"
    assert request.url.params["umkreis"] == "50"
    assert request.url.params["arbeitszeit"] == "vz"
    assert request.url.params["page"] == "1"
    assert request.url.params["size"] == "25"

    assert len(jobs) == 2
    assert isinstance(jobs[0], BAJobListing)
    assert jobs[0].ref_nr == "10000-1198765432-S"
    assert jobs[0].title == "Python Backend Entwickler (m/w/d)"
    assert jobs[0].employer == "Jobvis Tech GmbH"
    assert jobs[0].location == "10117 Berlin"
    assert jobs[0].working_time == "vz"
    assert jobs[0].external_url == "https://example.com/job/1"

    assert jobs[1].ref_nr == "hash-998877"
    assert jobs[1].title == "Frontend Engineer"
    assert jobs[1].employer == "Web Solutions AG"
    assert jobs[1].location == "80331 München"
    assert jobs[1].working_time == "tz"
    assert jobs[1].external_url == "https://www.arbeitsagentur.de/jobsuche/jobdetail/hash-998877"


@respx.mock
async def test_search_jobs_with_combined_arbeitszeit():
    """Verify search with combined arbeitszeit parameters (vz,tz,mj)."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "stellenangebote": [],
                "maxErgebnisse": 0,
                "page": 1,
                "size": 50,
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        jobs = await client.search_jobs(
            query="Pflegekraft",
            location="Hamburg",
            arbeitszeit="vz,tz",
            size=50,
        )

    assert route.called
    request = route.calls.last.request
    assert request.url.params["arbeitszeit"] == "vz,tz"
    assert len(jobs) == 0


@respx.mock
async def test_search_jobs_response_envelope():
    """Verify search_jobs_response returns BASearchResponse envelope with total counts."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "stellenangebote": [
                    {
                        "refnr": "REF-100",
                        "titel": "Data Scientist",
                        "arbeitgeber": "AI Corp",
                        "arbeitsort": {"ort": "Köln", "plz": "50667"},
                    }
                ],
                "maxErgebnisse": 100,
                "page": 2,
                "size": 10,
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        params = JobSearchParams(was="Data", wo="Köln", page=2, size=10)
        res = await client.search_jobs_response(params)

    assert isinstance(res, BASearchResponse)
    assert res.max_ergebnisse == 100
    assert res.page == 2
    assert res.size == 10
    assert len(res.stellenangebote) == 1
    assert res.stellenangebote[0].location == "50667 Köln"


@respx.mock
async def test_search_jobs_handles_missing_fields_gracefully():
    """Verify parsing handles responses with missing or sparse fields."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "stellenangebote": [
                    {},  # Empty item
                    {"refnr": "MINIMAL-1"},
                ],
                "maxErgebnisse": 2,
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        jobs = await client.search_jobs()

    assert len(jobs) == 2
    assert jobs[0].ref_nr == ""
    assert jobs[0].title == "Unbenanntes Stellenangebot"
    assert jobs[0].employer is None
    assert jobs[1].ref_nr == "MINIMAL-1"
    assert jobs[1].external_url == "https://www.arbeitsagentur.de/jobsuche/jobdetail/MINIMAL-1"


@respx.mock
async def test_get_job_details_success():
    """Verify get_job_details parses full BADetailedJob payload."""
    code = "10000-1198765432-S"
    respx.get(f"{DEFAULT_BASE_URL}/pc/v4/jobdetails/{code}").mock(
        return_value=httpx.Response(
            200,
            json={
                "refnr": code,
                "titel": "Senior Cloud Architect",
                "arbeitgeber": "Enterprise Cloud Systems SE",
                "stellenbeschreibung": "Wir suchen ab sofort einen erfahrenen Cloud Architect...",
                "taetigkeiten": ["Architektur von AWS/Azure Lösungen", "Team-Mentoring"],
                "anforderungen": ["5+ Jahre Erfahrung mit Cloud", "Deutsch C1", "Englisch C1"],
                "arbeitsorte": [
                    {
                        "strasse": "Alexanderplatz 1",
                        "plz": "10178",
                        "ort": "Berlin",
                        "land": "Deutschland",
                    }
                ],
                "arbeitszeit": "Vollzeit, 40h/Woche",
                "verguetung": "85.000 - 95.000 EUR",
                "befristung": "Unbefristet",
                "eintrittsdatum": "2026-11-01",
                "kontakt": {
                    "name": "HR Department",
                    "email": "careers@enterprisecloud.de",
                },
                "externeUrl": "https://enterprisecloud.de/careers/123",
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        detail = await client.get_job_details(code)

    assert detail is not None
    assert isinstance(detail, BADetailedJob)
    assert detail.ref_nr == code
    assert detail.title == "Senior Cloud Architect"
    assert detail.employer == "Enterprise Cloud Systems SE"
    assert "erfahrenen Cloud Architect" in detail.description
    assert len(detail.tasks) == 2
    assert len(detail.requirements) == 3
    assert detail.location_str == "Alexanderplatz 1, 10178, Berlin"
    assert detail.working_time == "Vollzeit, 40h/Woche"
    assert detail.remuneration == "85.000 - 95.000 EUR"
    assert detail.contract_duration == "Unbefristet"
    assert detail.contact["email"] == "careers@enterprisecloud.de"


@respx.mock
async def test_get_job_details_not_found_returns_none():
    """Verify get_job_details returns None when 404 is encountered."""
    code = "NON-EXISTENT-JOB"
    respx.get(f"{DEFAULT_BASE_URL}/pc/v4/jobdetails/{code}").mock(
        return_value=httpx.Response(404, text="Job not found")
    )

    async with ArbeitsagenturClient() as client:
        detail = await client.get_job_details(code)

    assert detail is None


@respx.mock
async def test_empty_or_blank_ref_nr_returns_none():
    """Verify passing empty ref_nr immediately returns None without network call."""
    async with ArbeitsagenturClient() as client:
        assert await client.get_job_details("") is None
        assert await client.get_job_details("   ") is None


@respx.mock
async def test_retry_on_429_rate_limit_and_recover():
    """Verify client retries on 429 and succeeds on subsequent attempt."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(200, json={"stellenangebote": [], "maxErgebnisse": 0}),
    ]

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        jobs = await client.search_jobs(query="Tester")

    assert len(jobs) == 0
    assert route.call_count == 3


@respx.mock
async def test_exhausted_retries_on_429_raises_rate_limit_error():
    """Verify ArbeitsagenturRateLimitError is raised when 429 persists beyond max_retries."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(429, text="Too Many Requests"),
    ]

    async with ArbeitsagenturClient(max_retries=2, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturRateLimitError) as exc_info:
            await client.search_jobs(query="Tester")
        assert exc_info.value.status_code == 429


@respx.mock
async def test_retry_on_503_server_error():
    """Verify client retries on transient 503 errors and recovers."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(503, text="Service Unavailable"),
        httpx.Response(200, json={"stellenangebote": [], "maxErgebnisse": 0}),
    ]

    async with ArbeitsagenturClient(max_retries=2, backoff_factor=0.01) as client:
        jobs = await client.search_jobs(query="Developer")

    assert len(jobs) == 0
    assert route.call_count == 2


@respx.mock
async def test_auth_error_401_raises_immediately_without_retries():
    """Verify HTTP 401 raises ArbeitsagenturAuthError without wasting retries."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(401, text="Unauthorized: Invalid API Key")
    )

    async with ArbeitsagenturClient(max_retries=3) as client:
        with pytest.raises(ArbeitsagenturAuthError) as exc_info:
            await client.search_jobs(query="Developer")
        assert exc_info.value.status_code == 401

    assert route.call_count == 1


@respx.mock
async def test_network_connection_error_retries_and_raises():
    """Verify network connection failures trigger retries and raise ArbeitsagenturConnectionError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = httpx.ConnectError("Failed to resolve host")

    async with ArbeitsagenturClient(max_retries=2, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturConnectionError):
            await client.search_jobs(query="Developer")

    assert route.call_count == 3


@respx.mock
async def test_timeout_retries_and_raises():
    """Verify timeout errors trigger retries and raise ArbeitsagenturTimeoutError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = httpx.ReadTimeout("Request timed out")

    async with ArbeitsagenturClient(max_retries=2, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturTimeoutError):
            await client.search_jobs(query="Developer")

    assert route.call_count == 3


# --- Challenger Empirical Error & Resilience Tests ---


# ============================================================================
# 1. HTTP 400 Bad Request
# ============================================================================


@respx.mock
async def test_http_400_bad_request_raises_immediately_no_retries():
    """Verify HTTP 400 raises ArbeitsagenturAPIError with status 400 and does NOT retry."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(400, text="Bad Request: Invalid parameter combination")
    )

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturAPIError) as exc_info:
            await client.search_jobs(query="Developer")

        assert exc_info.value.status_code == 400
        assert "Bad Request" in exc_info.value.response_text
        assert route.call_count == 1  # No retries on 400


# ============================================================================
# 2. HTTP 401 & 403 Authentication / Authorization Errors
# ============================================================================


@respx.mock
async def test_http_401_unauthorized_raises_auth_error_no_retries():
    """Verify HTTP 401 raises ArbeitsagenturAuthError and does NOT retry."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(401, text="Unauthorized: Invalid API Key")
    )

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturAuthError) as exc_info:
            await client.search_jobs(query="Developer")

        assert exc_info.value.status_code == 401
        assert "Authentication failed (401)" in str(exc_info.value)
        assert route.call_count == 1


@respx.mock
async def test_http_403_forbidden_raises_auth_error_no_retries():
    """Verify HTTP 403 raises ArbeitsagenturAuthError and does NOT retry."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(403, text="Forbidden: Access Denied")
    )

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturAuthError) as exc_info:
            await client.search_jobs(query="Developer")

        assert exc_info.value.status_code == 403
        assert "Authentication failed (403)" in str(exc_info.value)
        assert route.call_count == 1


# ============================================================================
# 3. HTTP 404 Not Found
# ============================================================================


@respx.mock
async def test_http_404_search_jobs_raises_not_found_no_retries():
    """Verify HTTP 404 in search_jobs raises ArbeitsagenturNotFoundError without retrying."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(404, text="Endpoint Not Found")
    )

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturNotFoundError) as exc_info:
            await client.search_jobs(query="Developer")

        assert exc_info.value.status_code == 404
        assert route.call_count == 1


@respx.mock
async def test_http_404_get_job_details_returns_none_gracefully():
    """Verify HTTP 404 in get_job_details catches 404 and returns None."""
    code = "10000-NONEXISTENT-S"
    safe_code = urllib.parse.quote(code, safe="")
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v4/jobdetails/{safe_code}").mock(
        return_value=httpx.Response(404, text="Job Not Found")
    )

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        detail = await client.get_job_details(code)

        assert detail is None
        assert route.call_count == 1


# ============================================================================
# 4. HTTP 429 Rate Limit Handling & Backoff
# ============================================================================


@respx.mock
async def test_http_429_transient_retries_and_recovers():
    """Verify HTTP 429 retries with backoff and returns results when rate limit clears."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(429, text="Too Many Requests"),
        httpx.Response(
            200,
            json={
                "stellenangebote": [{"refnr": "RECOVERED-1", "titel": "Dev"}],
                "maxErgebnisse": 1,
            },
        ),
    ]

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        jobs = await client.search_jobs(query="Developer")

        assert len(jobs) == 1
        assert jobs[0].ref_nr == "RECOVERED-1"
        assert route.call_count == 3


@respx.mock
async def test_http_429_persistent_exhausts_retries_and_raises_rate_limit_error():
    """Verify persistent HTTP 429 exhausts exactly max_retries + 1 calls and raises ArbeitsagenturRateLimitError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(429, text="Too Many Requests")
    )

    max_retries = 2
    async with ArbeitsagenturClient(max_retries=max_retries, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturRateLimitError) as exc_info:
            await client.search_jobs(query="Developer")

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded (429)" in str(exc_info.value)
        assert route.call_count == max_retries + 1  # 1 initial + 2 retries = 3 calls


# ============================================================================
# 5. HTTP 500, 502, 503, 504 Server Errors
# ============================================================================


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@respx.mock
async def test_http_5xx_transient_retries_and_recovers(status_code: int):
    """Verify HTTP 5xx transient server errors retry and succeed."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(status_code, text=f"Server Error {status_code}"),
        httpx.Response(
            200,
            json={
                "stellenangebote": [{"refnr": "JOB-5XX", "titel": "Engineer"}],
                "maxErgebnisse": 1,
            },
        ),
    ]

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        jobs = await client.search_jobs(query="Engineer")

        assert len(jobs) == 1
        assert jobs[0].ref_nr == "JOB-5XX"
        assert route.call_count == 2


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
@respx.mock
async def test_http_5xx_persistent_exhausts_retries_and_raises_api_error(status_code: int):
    """Verify persistent HTTP 5xx errors exhaust exactly max_retries + 1 calls and raise ArbeitsagenturAPIError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(status_code, text=f"Fatal {status_code}")
    )

    max_retries = 2
    async with ArbeitsagenturClient(max_retries=max_retries, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturAPIError) as exc_info:
            await client.search_jobs(query="Dev")

        assert exc_info.value.status_code == status_code
        assert f"Server returned status {status_code}" in str(exc_info.value)
        assert route.call_count == max_retries + 1


# ============================================================================
# 6. Network Connection & Timeout Errors
# ============================================================================


@respx.mock
async def test_network_connection_transient_retries_and_recovers():
    """Verify transient network connection failure retries and recovers."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.ConnectError("Connection refused"),
        httpx.Response(200, json={"stellenangebote": [], "maxErgebnisse": 0}),
    ]

    async with ArbeitsagenturClient(max_retries=2, backoff_factor=0.01) as client:
        jobs = await client.search_jobs()
        assert len(jobs) == 0
        assert route.call_count == 2


@respx.mock
async def test_network_connection_persistent_raises_connection_error():
    """Verify persistent network connection failure raises ArbeitsagenturConnectionError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = httpx.ConnectError("Host unreachable")

    max_retries = 3
    async with ArbeitsagenturClient(max_retries=max_retries, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturConnectionError) as exc_info:
            await client.search_jobs()

        assert "Failed to connect to Arbeitsagentur API" in str(exc_info.value)
        assert route.call_count == max_retries + 1


@pytest.mark.parametrize(
    "timeout_exc",
    [
        httpx.ReadTimeout("Read timed out"),
        httpx.ConnectTimeout("Connect timed out"),
        httpx.WriteTimeout("Write timed out"),
        httpx.PoolTimeout("Pool exhausted"),
    ],
)
@respx.mock
async def test_various_timeouts_retry_and_raise_timeout_error(timeout_exc: Exception):
    """Verify all httpx.TimeoutException variants trigger retries and raise ArbeitsagenturTimeoutError."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = timeout_exc

    max_retries = 2
    async with ArbeitsagenturClient(max_retries=max_retries, backoff_factor=0.01) as client:
        with pytest.raises(ArbeitsagenturTimeoutError) as exc_info:
            await client.search_jobs()

        assert "Request timed out after" in str(exc_info.value)
        assert route.call_count == max_retries + 1


# ============================================================================
# 7. Zero Retries & Infinite Loop Prevention
# ============================================================================


@respx.mock
async def test_max_retries_zero_performs_exactly_one_call():
    """Verify that when max_retries=0, exactly 1 request is made on failure without spinning."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    async with ArbeitsagenturClient(max_retries=0) as client:
        with pytest.raises(ArbeitsagenturAPIError):
            await client.search_jobs()

        assert route.call_count == 1


@respx.mock
async def test_high_max_retries_terminates_strictly_without_infinite_loop():
    """Verify max_retries=5 terminates strictly at 6 calls."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(503, text="Service Unavailable")
    )

    async with ArbeitsagenturClient(max_retries=5, backoff_factor=0.001) as client:
        with pytest.raises(ArbeitsagenturAPIError):
            await client.search_jobs()

        assert route.call_count == 6


# ============================================================================
# 8. Malformed JSON, Empty Payloads, HTML Responses
# ============================================================================


@respx.mock
async def test_invalid_json_syntax_raises_arbeitsagentur_api_error():
    """Verify invalid JSON syntax on 200 OK raises ArbeitsagenturAPIError instead of raw crash."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            text="<HTML><BODY>502 Gateway Error</BODY></HTML>",
            headers={"Content-Type": "text/html"},
        )
    )

    async with ArbeitsagenturClient() as client:
        with pytest.raises(ArbeitsagenturAPIError) as exc_info:
            await client.search_jobs()

        assert "Invalid JSON response from BA API" in str(exc_info.value)


@respx.mock
async def test_empty_string_response_raises_api_error():
    """Verify 200 OK with empty string raises ArbeitsagenturAPIError."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(return_value=httpx.Response(200, text=""))

    async with ArbeitsagenturClient() as client:
        with pytest.raises(ArbeitsagenturAPIError) as exc_info:
            await client.search_jobs()

        assert "Invalid JSON response" in str(exc_info.value)


@respx.mock
async def test_empty_json_object_returns_empty_results():
    """Verify 200 OK with `{}` returns empty listings list and 0 maxErgebnisse."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(return_value=httpx.Response(200, json={}))

    async with ArbeitsagenturClient() as client:
        res = await client.search_jobs_response(JobSearchParams())

        assert isinstance(res, BASearchResponse)
        assert len(res.stellenangebote) == 0
        assert res.max_ergebnisse == 0


@respx.mock
async def test_null_stellenangebote_returns_empty_list():
    """Verify 200 OK with `{"stellenangebote": null, "maxErgebnisse": 0}` handles null safely."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(200, json={"stellenangebote": None, "maxErgebnisse": 0})
    )

    async with ArbeitsagenturClient() as client:
        jobs = await client.search_jobs()
        assert jobs == []


@respx.mock
async def test_heterogeneous_non_dict_items_in_stellenangebote():
    """Verify non-dict items in stellenangebote (strings, ints, nulls) are safely filtered out."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "stellenangebote": [
                    "corrupted_string_item",
                    12345,
                    None,
                    {"refnr": "VALID-1", "titel": "Valid Job"},
                    {"refnr": "VALID-2", "titel": "Second Valid Job"},
                ],
                "maxErgebnisse": 5,
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        jobs = await client.search_jobs()

        assert len(jobs) == 2
        assert jobs[0].ref_nr == "VALID-1"
        assert jobs[1].ref_nr == "VALID-2"


# ============================================================================
# 9. Schema Parsing Edge Cases (BAJobListing & BADetailedJob)
# ============================================================================


async def test_bajoblisting_from_api_dict_edge_cases():
    """Verify BAJobListing handles all alternative field keys and missing values."""
    # 1. Fully empty dict
    listing_empty = BAJobListing.from_api_dict({})
    assert listing_empty.ref_nr == ""
    assert listing_empty.title == "Unbenanntes Stellenangebot"
    assert listing_empty.employer is None
    assert listing_empty.location is None
    assert listing_empty.working_time is None
    assert listing_empty.description is None
    assert listing_empty.external_url is None

    # 2. Alternative keys (hashId, beruf, firma, ort string, arbeitszeitmodell, beschreibung, externeUrl)
    listing_alt = BAJobListing.from_api_dict(
        {
            "hashId": "hash-abc-123",
            "beruf": "Tischler/in",
            "firma": "Schreinerei Holz GmbH",
            "ort": "Hamburg",
            "arbeitszeitmodell": "tz",
            "beschreibung": "Schöne Holzarbeiten",
            "externeUrl": "https://holz.de/jobs/1",
            "modifikationsTimestamp": "2026-08-01T12:00:00Z",
        }
    )
    assert listing_alt.ref_nr == "hash-abc-123"
    assert listing_alt.title == "Tischler/in"
    assert listing_alt.employer == "Schreinerei Holz GmbH"
    assert listing_alt.location == "Hamburg"
    assert listing_alt.working_time == "tz"
    assert listing_alt.description == "Schöne Holzarbeiten"
    assert listing_alt.external_url == "https://holz.de/jobs/1"

    # 3. Location with region only
    listing_loc_region = BAJobListing.from_api_dict(
        {
            "refnr": "R-1",
            "arbeitsort": {"region": "Bayern", "plz": "", "ort": ""},
        }
    )
    assert listing_loc_region.location == "Bayern"

    # 4. Location with plz and ort
    listing_loc_plz_ort = BAJobListing.from_api_dict(
        {
            "refnr": "R-2",
            "arbeitsort": {"plz": "70173", "ort": "Stuttgart"},
        }
    )
    assert listing_loc_plz_ort.location == "70173 Stuttgart"

    # 5. Location as list of strings
    listing_loc_list_str = BAJobListing.from_api_dict({"location": ["Berlin"]})
    assert listing_loc_list_str.location == "Berlin"

    # 6. Location as dict with null fields
    listing_loc_null_fields = BAJobListing.from_api_dict(
        {"arbeitsort": {"plz": None, "ort": "Berlin", "region": None}}
    )
    assert listing_loc_null_fields.location == "Berlin"

    # 7. Location with integer plz
    listing_loc_int_plz = BAJobListing.from_api_dict(
        {"arbeitsort": {"plz": 10115, "ort": "Berlin"}}
    )
    assert listing_loc_int_plz.location == "10115 Berlin"

    # 8. Location with nested adresse dict
    listing_loc_nested_addr = BAJobListing.from_api_dict(
        {"arbeitsort": {"adresse": {"plz": "10115", "ort": "Berlin"}}}
    )
    assert listing_loc_nested_addr.location == "10115 Berlin"

    # 9. Location list with leading empty / None elements
    listing_loc_list_fallback = BAJobListing.from_api_dict({"location": [{}, None, "Hamburg"]})
    assert listing_loc_list_fallback.location == "Hamburg"

    # 10. Location list with nested adresse dict
    listing_loc_list_nested = BAJobListing.from_api_dict(
        {"location": [{"adresse": {"plz": 80331, "ort": "München"}}]}
    )
    assert listing_loc_list_nested.location == "80331 München"

    # 11. Location with only region
    listing_loc_region = BAJobListing.from_api_dict({"location": [{"region": "Bayern"}]})
    assert listing_loc_region.location == "Bayern"

    # 12. Plural key arbeitsorte
    listing_loc_arbeitsorte = BAJobListing.from_api_dict({"arbeitsorte": [{"ort": "Berlin"}]})
    assert listing_loc_arbeitsorte.location == "Berlin"

    # 13. Plural key locations (string list)
    listing_loc_locations = BAJobListing.from_api_dict({"locations": ["Hamburg"]})
    assert listing_loc_locations.location == "Hamburg"

    # 14. Employer as dictionary
    listing_dict_employer = BAJobListing.from_api_dict({"arbeitgeber": {"name": "Tech Corp GmbH"}})
    assert listing_dict_employer.employer == "Tech Corp GmbH"


async def test_badetailedjob_from_api_dict_edge_cases():
    """Verify BADetailedJob parses complex and sparse structures without errors."""
    # 1. Empty dict
    detailed_empty = BADetailedJob.from_api_dict({})
    assert detailed_empty.ref_nr == ""
    assert detailed_empty.title == "Unbekannt"
    assert detailed_empty.tasks == []
    assert detailed_empty.requirements == []
    assert detailed_empty.locations == []
    assert detailed_empty.location_str is None

    # 2. String tasks and requirements (newline separated)
    detailed_str_tasks = BADetailedJob.from_api_dict(
        {
            "refnr": "12345",
            "titel": "Kaufmann",
            "taetigkeiten": "Buchhaltung\nKundenbetreuung\nRechnungsstellung",
            "anforderungen": "Excel Kenntnisse\nDeutsch B2",
            "arbeitsort": {"plz": "60311", "ort": "Frankfurt am Main"},
        }
    )
    assert detailed_str_tasks.tasks == ["Buchhaltung", "Kundenbetreuung", "Rechnungsstellung"]
    assert detailed_str_tasks.requirements == ["Excel Kenntnisse", "Deutsch B2"]
    assert detailed_str_tasks.location_str == "60311 Frankfurt am Main"

    # 3. Multiple arbeitsorte with street, plz, ort
    detailed_multi_loc = BADetailedJob.from_api_dict(
        {
            "refnr": "MULTI-LOC",
            "titel": "Manager",
            "arbeitsorte": [
                {"strasse": "Hauptstr. 10", "plz": "50667", "ort": "Köln"},
                {"strasse": "Zweigstr. 5", "plz": "40213", "ort": "Düsseldorf"},
            ],
        }
    )
    assert len(detailed_multi_loc.locations) == 2
    assert detailed_multi_loc.location_str == "Hauptstr. 10, 50667, Köln"

    # 4. Arbeitsort with null plz and ort
    detailed_null_loc = BADetailedJob.from_api_dict({"arbeitsort": {"plz": None, "ort": None}})
    assert detailed_null_loc.location_str is None

    # 5. Arbeitsorte as list of strings
    detailed_str_loc = BADetailedJob.from_api_dict({"arbeitsorte": ["Berlin"]})
    assert detailed_str_loc.location_str == "Berlin"

    # 6. Arbeitsort with street and nested adresse
    detailed_nested_addr = BADetailedJob.from_api_dict(
        {
            "arbeitsort": {
                "adresse": {"strasse": "Friedrichstraße 12", "plz": "10115", "ort": "Berlin"}
            }
        }
    )
    assert detailed_nested_addr.location_str == "Friedrichstraße 12, 10115, Berlin"

    # 7. Dict tasks and requirements
    detailed_dict_reqs = BADetailedJob.from_api_dict(
        {
            "taetigkeiten": {"t1": "Architektur", "t2": "Entwicklung"},
            "anforderungen": {
                "sprache": [{"sprache": "Deutsch", "niveau": "C1"}],
                "erfahrung": "5 Jahre",
            },
        }
    )
    assert detailed_dict_reqs.tasks == ["Architektur", "Entwicklung"]
    assert "5 Jahre" in detailed_dict_reqs.requirements
    assert "Deutsch: C1" in detailed_dict_reqs.requirements

    # 8. Locations list with leading empty dict
    detailed_loc_fallback = BADetailedJob.from_api_dict({"locations": [{}, {"ort": "Berlin"}]})
    assert detailed_loc_fallback.location_str == "Berlin"

    # 9. Location with region only (dict and list)
    detailed_region_dict = BADetailedJob.from_api_dict({"arbeitsort": {"region": "Bayern"}})
    assert detailed_region_dict.location_str == "Bayern"

    detailed_region_list = BADetailedJob.from_api_dict({"locations": [{"region": "Hessen"}]})
    assert detailed_region_list.location_str == "Hessen"

    # 10. List of dicts in tasks and requirements
    detailed_list_dicts = BADetailedJob.from_api_dict(
        {
            "taetigkeiten": [{"beschreibung": "API Entwicklung"}],
            "anforderungen": [{"skill": "Python"}, {"skill": "Docker"}],
        }
    )
    assert detailed_list_dicts.tasks == ["API Entwicklung"]
    assert "Python" in detailed_list_dicts.requirements
    assert "Docker" in detailed_list_dicts.requirements

    # 11. Employer and Title as dictionary
    detailed_dict_meta = BADetailedJob.from_api_dict(
        {
            "titel": {"bezeichnung": "Senior Cloud Architect"},
            "arbeitgeber": {"name": "Cloud Solutions SE"},
        }
    )
    assert detailed_dict_meta.title == "Senior Cloud Architect"
    assert detailed_dict_meta.employer == "Cloud Solutions SE"


@respx.mock
async def test_get_job_details_encodes_special_characters_in_ref_nr():
    """Verify get_job_details properly URL-encodes special characters in ref_nr (e.g. slashes, spaces)."""
    raw_code = "10000/MÜNCHEN SPEC#1"
    safe_code = urllib.parse.quote(raw_code, safe="")

    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v4/jobdetails/{safe_code}").mock(
        return_value=httpx.Response(
            200,
            json={
                "refnr": raw_code,
                "titel": "Special Ref Engineer",
            },
        )
    )

    async with ArbeitsagenturClient() as client:
        detail = await client.get_job_details(raw_code)

        assert detail is not None
        assert detail.ref_nr == raw_code
        assert route.called


# ============================================================================
# 10. Client Lifecycle and Custom Injected Client
# ============================================================================


async def test_internal_client_closes_properly_on_exit():
    """Verify internal AsyncClient is closed when leaving async context manager."""
    client = ArbeitsagenturClient()
    async with client:
        _ = client.client  # instantiate internal client
        assert client._internal_client is not None
        assert not client._internal_client.is_closed

    assert client._internal_client is None


@respx.mock
async def test_external_client_injection_is_used():
    """Verify that when an external httpx.AsyncClient is provided, it is used directly."""
    external_client = httpx.AsyncClient()
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(200, json={"stellenangebote": [], "maxErgebnisse": 0})
    )

    try:
        async with ArbeitsagenturClient(client=external_client) as client:
            assert client.client is external_client
            jobs = await client.search_jobs()
            assert jobs == []
            assert route.called
    finally:
        await external_client.aclose()


# --- LLM Query Generator Tests ---


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
    # Subtest 1: None goals -> generates 2 targeted queries from CV skills
    res_none = await generate_search_query(
        goals=None,
        cv_profile={"skills": ["Elektriker", "SPS-Programmierung"], "experience_years": 4.0},
        user_prefs={"location": "Hamburg", "desired_job_type": "vz"},
        llm=None,
    )
    assert len(res_none) == 2
    assert res_none[0].was == "Elektriker"
    assert res_none[1].was == "SPS-Programmierung"
    assert res_none.wo == "Hamburg"
    assert res_none.arbeitszeit == "vz"
    assert res_none.angebotsart == 1

    # Subtest 2: Empty string goals -> generates 2 targeted queries from CV skills
    res_empty = await generate_search_query(
        goals="   ",
        cv_profile={"skills": ["Koch", "Gastronomie"]},
        user_prefs={"location": "Dresden", "desired_job_type": "tz"},
        llm=None,
    )
    assert len(res_empty) == 2
    assert "Koch" in res_empty.was
    assert res_empty[0].was == "Koch"
    assert res_empty[1].was == "Gastronomie"
    assert res_empty.wo == "Dresden"
    assert res_empty.arbeitszeit == "tz"
    assert res_empty.angebotsart == 1

    # Subtest 3: No CV skills, only keywords -> generates 2 targeted queries from keywords
    res_kw = await generate_search_query(
        goals="",
        cv_profile={"skills": [], "keywords": ["Pflegefachkraft", "Geriatrie"]},
        user_prefs={"location": "Bremen", "desired_job_type": "mj"},
        llm=None,
    )
    assert len(res_kw) == 2
    assert "Pflegefachkraft" in res_kw.was
    assert res_kw[0].was == "Pflegefachkraft"
    assert res_kw[1].was == "Geriatrie"
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
    user = User(id=user_id, email="test-user-llm-1@example.com")
    profile = Profile(
        user_id=user_id,
        goals="I want a minijob in retail and marketing",
        location="Berlin",
        radius_km=25,
        desired_job_type="all",
        german_level="B1",
        onboarding_completed=True,
        onboarding_step=8,
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
        user,  # User lookup
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
        return await generate_search_query(**kwargs)

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


async def test_convenience_helpers_and_alias():
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
    from app.services.query_generator import (
        BAQueryList,
        generate_ba_queries,
        generate_search_queries,
    )

    assert generate_search_queries is generate_search_query
    assert generate_ba_queries is generate_search_query

    q_list = BAQueryList(
        [params, BAQueryParams(was="Developer", wo="Hamburg", arbeitszeit="tz", angebotsart=1)]
    )
    assert len(q_list) == 2
    assert q_list.was == "Tester"
    assert q_list.wo == "Berlin"
    assert q_list["was"] == "Tester"
    assert "was" in q_list
    assert len(q_list.to_dict_list()) == 2
    assert q_list.to_dict_list()[1]["was"] == "Developer"


@pytest.mark.asyncio
async def test_llm_query_batch_multiple_queries():
    """Verify LLM generating a batch of 2-5 distinct targeted queries."""
    mock_response = json.dumps(
        {
            "queries": [
                {
                    "was": "Python Entwickler",
                    "wo": "Berlin",
                    "arbeitszeit": "vz",
                    "angebotsart": 1,
                },
                {
                    "was": "Backend Developer",
                    "wo": "Berlin",
                    "arbeitszeit": "vz",
                    "angebotsart": 1,
                },
                {
                    "was": "Data Engineer",
                    "wo": "Berlin",
                    "arbeitszeit": "vz",
                    "angebotsart": 1,
                },
            ]
        }
    )
    mock_llm = FakeListLLM(responses=[mock_response])

    res = await generate_search_query(
        goals="I want to work as a backend python or data engineer",
        cv_profile={"skills": ["Python", "FastAPI", "SQL", "Docker"], "experience_years": 4.0},
        user_prefs={"location": "Berlin", "desired_job_type": "vz"},
        llm=mock_llm,
    )

    assert len(res) == 3
    assert res[0].was == "Python Entwickler"
    assert res[1].was == "Backend Developer"
    assert res[2].was == "Data Engineer"
    # Backward compatible attributes refer to primary query
    assert res.was == "Python Entwickler"
    assert res.wo == "Berlin"
    assert res.arbeitszeit == "vz"
    assert res.angebotsart == 1
    # Check to_dict_list helper
    dict_list = res.to_dict_list()
    assert len(dict_list) == 3
    assert [d["was"] for d in dict_list] == [
        "Python Entwickler",
        "Backend Developer",
        "Data Engineer",
    ]


@pytest.mark.asyncio
async def test_heuristic_generates_2_to_5_queries():
    """Verify heuristic extractor produces 2 to 5 targeted queries covering goals and CV skills."""
    res = await generate_search_query(
        goals="I want a minijob in retail and marketing",
        cv_profile={"skills": ["Kundenservice", "Verkauf", "Kasse"], "keywords": ["Social Media"]},
        user_prefs={"location": "Frankfurt", "desired_job_type": "mj"},
        llm=None,
    )

    assert len(res) >= 2
    assert len(res) <= 5
    was_list = [q.was for q in res]
    # Primary queries from goals
    assert "Einzelhandel" in was_list
    assert "Marketing" in was_list
    # Additional queries from skills
    assert any(s in was_list for s in ["Kundenservice", "Verkauf", "Kasse"])
    assert all(q.wo == "Frankfurt" for q in res)
    assert all(q.arbeitszeit == "mj" for q in res)


@pytest.mark.asyncio
async def test_scheduler_integration_with_multi_query():
    """Verify scheduler queries BA API for each generated targeted query and deduplicates jobs."""
    from unittest.mock import MagicMock

    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.flush = AsyncMock()
    mock_db.rollback = AsyncMock()

    user_id = "test-user-multi-query"
    user = User(id=user_id, email="test-user-multi@example.com")
    profile = Profile(
        user_id=user_id,
        goals="Software Entwickler oder DevOps",
        location="Berlin",
        radius_km=25,
        desired_job_type="vz",
        german_level="B2",
        onboarding_completed=True,
        onboarding_step=8,
    )
    cv = CVAnalysis(
        user_id=user_id,
        raw_text="Python DevOps engineer",
        skills=["Python", "Docker", "DevOps"],
        experience_years=3.0,
    )

    mock_result = MagicMock()
    mock_scalars = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    lookup_queue = [user, profile, cv]
    mock_scalars.first.side_effect = lambda: lookup_queue.pop(0) if lookup_queue else None
    mock_scalars.all.return_value = []
    mock_db.execute.return_value = mock_result

    # Mock BA client returns different listings for each query call
    job1 = BAJobListing(
        ref_nr="REF-PYTHON-01",
        title="Python Developer",
        employer="Tech Corp",
        location="Berlin",
        working_time="vz",
    )
    job2 = BAJobListing(
        ref_nr="REF-DEVOPS-02",
        title="DevOps Engineer",
        employer="Cloud Systems",
        location="Berlin",
        working_time="vz",
    )
    # Overlapping job returned in another query batch
    job1_dup = BAJobListing(
        ref_nr="REF-PYTHON-01",
        title="Python Developer",
        employer="Tech Corp",
        location="Berlin",
        working_time="vz",
    )

    mock_ba = AsyncMock()
    mock_ba.search_jobs.side_effect = [
        [job1],
        [job2],
        [job1_dup],
    ]

    mock_llm_response = json.dumps(
        {
            "queries": [
                {"was": "Python Entwickler", "wo": "Berlin", "arbeitszeit": "vz", "angebotsart": 1},
                {"was": "DevOps Engineer", "wo": "Berlin", "arbeitszeit": "vz", "angebotsart": 1},
                {"was": "Backend Developer", "wo": "Berlin", "arbeitszeit": "vz", "angebotsart": 1},
            ]
        }
    )
    mock_llm = FakeListLLM(responses=[mock_llm_response])

    async def mock_gen(*args, **kwargs):
        kwargs["llm"] = mock_llm
        return await generate_search_query(**kwargs)

    with patch("app.services.query_generator.generate_search_query", side_effect=mock_gen):
        scheduler = MatchingSchedulerService()
        result = await scheduler.run_sync_for_user(user_id, mock_db, ba_client=mock_ba)

        assert result["status"] == "success"
        # 3 calls made (one per query)
        assert mock_ba.search_jobs.await_count == 3
        # 3 raw jobs scraped across all batches
        assert result["scraped"] == 3
        # Deduplicated to 2 unique jobs
        assert result["deduped"] == 2
