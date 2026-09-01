"""Automated tests for Arbeitsagentur REST API Client."""

import httpx
import pytest
import respx

from app.schemas.job import (
    BADetailedJob,
    BAJobListing,
    BASearchResponse,
    JobSearchParams,
)
from app.services.arbeitsagentur import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    ArbeitsagenturAuthError,
    ArbeitsagenturClient,
    ArbeitsagenturConnectionError,
    ArbeitsagenturRateLimitError,
    ArbeitsagenturTimeoutError,
)

pytestmark = pytest.mark.asyncio


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
