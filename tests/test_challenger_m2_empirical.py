"""Empirical Challenger 2 Test Suite for Milestone M2.

Stress-tests:
- ArbeitsagenturClient HTTP error handling: 400, 401, 403, 404, 429, 500, 502, 503, 504
- Network connection errors and timeouts (ReadTimeout, ConnectTimeout, WriteTimeout)
- Backoff retry verification (call counts, exponential scaling, no infinite loops)
- Malformed, empty, and non-JSON responses
- Heterogeneous, sparse, and edge-case payload schemas (missing fields, unexpected types, special characters)
- Client lifecycle and resource cleanup
"""

import urllib.parse

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
    DEFAULT_BASE_URL,
    ArbeitsagenturAPIError,
    ArbeitsagenturAuthError,
    ArbeitsagenturClient,
    ArbeitsagenturConnectionError,
    ArbeitsagenturNotFoundError,
    ArbeitsagenturRateLimitError,
    ArbeitsagenturTimeoutError,
)

pytestmark = pytest.mark.asyncio


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


async def test_badetailedjob_from_api_dict_edge_cases():
    """Verify BADetailedJob parses complex and sparse structures without errors."""
    # 1. Empty dict
    detailed_empty = BADetailedJob.from_api_dict({})
    assert detailed_empty.ref_nr == ""
    assert detailed_empty.title == ""
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
