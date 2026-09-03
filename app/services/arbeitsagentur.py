"""Bundesagentur für Arbeit (Arbeitsagentur) REST API client."""

import asyncio
import logging
import urllib.parse
from typing import Any

import httpx

from app.schemas.job import (
    BADetailedJob,
    BAJobListing,
    BASearchResponse,
    JobSearchParams,
)

logger = logging.getLogger(__name__)

# Official Arbeitsagentur Jobsuche API endpoints
DEFAULT_BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
DEFAULT_SEARCH_PATH = "/pc/v6/jobs"
DEFAULT_DETAILS_PATH = "/pc/v4/jobdetails/{code}"
DEFAULT_API_KEY = "jobboerse-jobsuche"
DEFAULT_TIMEOUT = 12.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.5


class ArbeitsagenturAPIError(Exception):
    """Base exception for all Arbeitsagentur API errors."""

    def __init__(
        self, message: str, status_code: int | None = None, response_text: str | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class ArbeitsagenturConnectionError(ArbeitsagenturAPIError):
    """Raised when network connection fails after all retries."""


class ArbeitsagenturTimeoutError(ArbeitsagenturAPIError):
    """Raised when an API request times out after all retries."""


class ArbeitsagenturRateLimitError(ArbeitsagenturAPIError):
    """Raised when HTTP 429 Too Many Requests is received and retries are exhausted."""


class ArbeitsagenturNotFoundError(ArbeitsagenturAPIError):
    """Raised when the requested job posting (HTTP 404) is not found."""


class ArbeitsagenturAuthError(ArbeitsagenturAPIError):
    """Raised when API Key authentication fails (HTTP 401 / 403)."""


class ArbeitsagenturClient:
    """Async client for querying the Bundesagentur für Arbeit (BA) Jobsuche REST API.

    Supports:
    - Search queries with 'was', 'wo', 'umkreis', 'arbeitszeit', pagination.
    - Retrieval of detailed job postings.
    - Automatic retries with exponential backoff for transient failures (429, 502, 503, 504, timeouts).
    - Custom AsyncClient injection for testing and mock environments.
    """

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        search_path: str = DEFAULT_SEARCH_PATH,
        details_path: str = DEFAULT_DETAILS_PATH,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.search_path = search_path
        self.details_path = details_path
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._external_client = client
        self._internal_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Returns the active httpx.AsyncClient instance."""
        if self._external_client is not None:
            return self._external_client
        if self._internal_client is None or self._internal_client.is_closed:
            self._internal_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers=self._get_default_headers(),
            )
        return self._internal_client

    def _get_default_headers(self) -> dict[str, str]:
        """Build standard headers required by the BA API."""
        return {
            "X-API-Key": self.api_key,
            "User-Agent": "Jobvis/0.1.0 (+https://github.com/metimol/Jobvis)",
            "Accept": "application/json",
        }

    async def __aenter__(self) -> "ArbeitsagenturClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close internal HTTP client session."""
        if self._internal_client and not self._internal_client.is_closed:
            await self._internal_client.aclose()
            self._internal_client = None

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Execute HTTP request with exponential backoff and error handling."""
        req_headers = self._get_default_headers()
        if headers:
            req_headers.update(headers)

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=req_headers,
                )

                # Success
                if response.status_code == 200:
                    return response

                # 404 Not Found (no retry)
                if response.status_code == 404:
                    raise ArbeitsagenturNotFoundError(
                        f"Resource not found at {url}: {response.text}",
                        status_code=404,
                        response_text=response.text,
                    )

                # 401 / 403 Authentication Error (no retry)
                if response.status_code in (401, 403):
                    raise ArbeitsagenturAuthError(
                        f"Authentication failed ({response.status_code}) for {url}: {response.text}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )

                # 429 Rate Limited or 5xx Transient Server Error -> Retry
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < self.max_retries:
                        sleep_seconds = self.backoff_factor * (2**attempt)
                        logger.warning(
                            f"Arbeitsagentur API returned {response.status_code} on attempt {attempt + 1}. Retrying in {sleep_seconds:.2f}s..."
                        )
                        await asyncio.sleep(sleep_seconds)
                        continue
                    if response.status_code == 429:
                        raise ArbeitsagenturRateLimitError(
                            f"Rate limit exceeded (429) after {self.max_retries} retries.",
                            status_code=429,
                            response_text=response.text,
                        )
                    raise ArbeitsagenturAPIError(
                        f"Server returned status {response.status_code} after {self.max_retries} retries: {response.text}",
                        status_code=response.status_code,
                        response_text=response.text,
                    )

                # Other 4xx Client Errors -> Do not retry
                raise ArbeitsagenturAPIError(
                    f"HTTP error {response.status_code} for {url}: {response.text}",
                    status_code=response.status_code,
                    response_text=response.text,
                )

            except (httpx.ConnectError, httpx.NetworkError) as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_seconds = self.backoff_factor * (2**attempt)
                    logger.warning(
                        f"Connection error on attempt {attempt + 1}: {e}. Retrying in {sleep_seconds:.2f}s..."
                    )
                    await asyncio.sleep(sleep_seconds)
                else:
                    raise ArbeitsagenturConnectionError(
                        f"Failed to connect to Arbeitsagentur API after {self.max_retries} retries: {e}"
                    ) from e

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_seconds = self.backoff_factor * (2**attempt)
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}: {e}. Retrying in {sleep_seconds:.2f}s..."
                    )
                    await asyncio.sleep(sleep_seconds)
                else:
                    raise ArbeitsagenturTimeoutError(
                        f"Request timed out after {self.max_retries} retries: {e}"
                    ) from e

        if last_error:
            raise ArbeitsagenturAPIError(f"Request failed: {last_error}") from last_error
        raise ArbeitsagenturAPIError(f"Request to {url} failed.")

    async def search_jobs(
        self,
        query: str = "",
        location: str = "",
        radius_km: int = 25,
        arbeitszeit: str = "",
        page: int = 1,
        size: int = 25,
        **kwargs: Any,
    ) -> list[BAJobListing]:
        """Search Bundesagentur für Arbeit job listings matching the provided criteria.

        Args:
            query: Occupation, keyword, or search query ('was')
            location: City, region, or postal code ('wo')
            radius_km: Search radius in km ('umkreis')
            arbeitszeit: Working time filter ('vz', 'tz', 'mj' or combinations)
            page: Page number (1-indexed)
            size: Results per page (1-100)
            **kwargs: Extra parameters (e.g. angebotsart, veroeffentlichtseit)

        Returns:
            List of BAJobListing models.
        """
        response_model = await self.search_jobs_response(
            JobSearchParams(
                was=query or kwargs.get("was"),
                wo=location or kwargs.get("wo"),
                umkreis=radius_km if radius_km is not None else kwargs.get("umkreis", 25),
                arbeitszeit=arbeitszeit or kwargs.get("arbeitszeit"),
                page=page or kwargs.get("page", 1),
                size=size or kwargs.get("size", 25),
                angebotsart=kwargs.get("angebotsart", 1),
                veroeffentlichtseit=kwargs.get("veroeffentlichtseit"),
            )
        )
        return response_model.stellenangebote

    async def search_jobs_response(
        self,
        params: JobSearchParams | dict[str, Any],
    ) -> BASearchResponse:
        """Execute a search query returning the full BASearchResponse envelope."""
        if isinstance(params, dict):
            search_params = JobSearchParams(**params)
        else:
            search_params = params

        query_dict = search_params.to_query_params()
        url = f"{self.base_url}{self.search_path}"

        response = await self._request("GET", url, params=query_dict)
        try:
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to decode JSON response from {url}: {e}")
            raise ArbeitsagenturAPIError(
                f"Invalid JSON response from BA API: {response.text}"
            ) from e

        stellenangebote_raw = data.get("ergebnisliste") or data.get("stellenangebote") or []
        items: list[BAJobListing] = []
        for raw_item in stellenangebote_raw:
            if isinstance(raw_item, dict):
                items.append(BAJobListing.from_api_dict(raw_item))

        max_ergebnisse = data.get("maxErgebnisse") or len(items)

        return BASearchResponse(
            stellenangebote=items,
            max_ergebnisse=int(max_ergebnisse),
            page=search_params.page,
            size=search_params.size,
            raw_data=data,
        )

    async def get_job_details(self, ref_nr: str) -> BADetailedJob | None:
        """Fetch detailed information for a specific job posting code/refnr.

        Args:
            ref_nr: The job reference code (e.g. '10000-1198765432-S' or hash code)

        Returns:
            BADetailedJob instance if found, or None if the job is not found (404).
        """
        if not ref_nr or not ref_nr.strip():
            return None

        # Clean and encode ref_nr for URL path safety
        safe_code = urllib.parse.quote(ref_nr.strip(), safe="")
        path = self.details_path.format(code=safe_code)
        url = f"{self.base_url}{path}"

        try:
            response = await self._request("GET", url)
            data = response.json()
            return BADetailedJob.from_api_dict(data)
        except ArbeitsagenturNotFoundError:
            logger.info(f"Job posting {ref_nr} not found (404).")
            return None
        except Exception as e:
            logger.error(f"Error fetching job details for {ref_nr}: {e}")
            raise
