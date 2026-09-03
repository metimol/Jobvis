"""Job vacancy, matching, and Arbeitsagentur API integration schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobStatusLiteral = Literal["new", "viewed", "saved", "dismissed"]


class WorkingTimeEnum(str, Enum):
    """Working time models supported by BA API."""

    VOLLZEIT = "vz"
    TEILZEIT = "tz"
    MINIJOB = "mj"
    ALL = "all"
    HOMEOFFICE = "ho"


class GermanLevelEnum(str, Enum):
    """CEFR German language proficiency levels."""

    A2 = "A2"
    B1 = "B1"
    B2 = "B2"
    C1 = "C1"


class JobMatchStatusEnum(str, Enum):
    """Status of matched jobs for a user."""

    NEW = "new"
    VIEWED = "viewed"
    SAVED = "saved"
    DISMISSED = "dismissed"


class JobResponse(BaseModel):
    """Job vacancy details stored in database."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    ref_nr: str
    canonical_hash: str
    title: str
    employer: str | None = None
    location: str | None = None
    working_time: str | None = None
    description: str | None = None
    external_url: str | None = None
    published_date: datetime | None = None
    created_at: datetime | None = None


class MatchedJobResponse(BaseModel):
    """Job vacancy matched to a user profile with score and rationales."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: str
    job_id: str
    score: float
    match_reasons: list[dict[str, Any]] = Field(default_factory=list)
    status: JobStatusLiteral
    created_at: datetime | None = None
    job: JobResponse | None = None


class MatchStatusUpdate(BaseModel):
    """Payload to update match status (viewed, saved, dismissed)."""

    status: JobStatusLiteral


class JobSearchParams(BaseModel):
    """Query parameters for searching jobs on Bundesagentur für Arbeit API."""

    model_config = ConfigDict(extra="ignore")

    was: str | None = Field(default=None, description="Search term, job title, or keywords")
    wo: str | None = Field(default=None, description="Location, city, or postal code")
    umkreis: int | None = Field(
        default=25, description="Search radius in kilometers (e.g. 0, 10, 25, 50, 100, 200)"
    )
    arbeitszeit: str | None = Field(
        default=None, description="Working time model: vz, tz, mj or combinations"
    )
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    size: int = Field(default=25, ge=1, le=100, description="Page size (1-100)")
    angebotsart: int | None = Field(
        default=1, description="1 for standard employment, 4 for apprenticeship"
    )
    veroeffentlichtseit: int | None = Field(
        default=None, description="Published within past N days"
    )

    def to_query_params(self) -> dict[str, Any]:
        """Convert model fields to BA API query parameters."""
        params: dict[str, Any] = {
            "page": self.page,
            "size": self.size,
        }
        if self.was:
            params["was"] = self.was
        if self.wo:
            params["wo"] = self.wo
        if self.umkreis is not None:
            params["umkreis"] = self.umkreis
        if self.arbeitszeit:
            params["arbeitszeit"] = self.arbeitszeit
        if self.angebotsart is not None:
            params["angebotsart"] = self.angebotsart
        if self.veroeffentlichtseit is not None:
            params["veroeffentlichtseit"] = self.veroeffentlichtseit
        return params


class BAJobListing(BaseModel):
    """Canonical representation of a Bundesagentur für Arbeit job search result item."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ref_nr: str = Field(description="Unique reference number or hash ID from Arbeitsagentur")
    title: str = Field(description="Job title / Beruf")
    employer: str | None = Field(default=None, description="Employer name")
    location: str | None = Field(
        default=None, description="Formatted location (e.g. '10115 Berlin')"
    )
    working_time: str | None = Field(
        default=None, description="Working time model (e.g. 'vz', 'tz', 'Vollzeit')"
    )
    description: str | None = Field(default=None, description="Job description or teaser")
    external_url: str | None = Field(
        default=None, description="URL to the job posting on Jobboerse or external site"
    )
    published_date: datetime | str | None = Field(
        default=None, description="Publication or entry date"
    )
    canonical_hash: str | None = Field(
        default=None, description="Computed SHA-256 deduplication fingerprint"
    )
    score: float | None = Field(
        default=None, description="AI matching score (0.0 to 1.0 or 0 to 100)"
    )
    match_reasons: list[Any] | dict[str, Any] | None = Field(
        default=None, description="AI rationale for matching"
    )
    raw_data: dict[str, Any] | None = Field(
        default=None, description="Original raw payload from API"
    )

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "BAJobListing":
        """Factory method to parse heterogeneous BA API response structures."""
        # 1. Reference number
        ref_nr = (
            data.get("referenznummer")
            or data.get("refnr")
            or data.get("hashId")
            or data.get("ref_nr")
            or data.get("id")
            or ""
        )

        # 2. Title
        title = (
            data.get("stellenangebotsTitel")
            or data.get("titel")
            or data.get("beruf")
            or data.get("title")
            or "Unbenanntes Stellenangebot"
        )

        # 3. Employer
        employer = data.get("arbeitgeber") or data.get("employer") or data.get("firma")

        # 4. Location parsing
        location_raw = (
            data.get("stellenlokationen")
            or data.get("arbeitsort")
            or data.get("location")
            or data.get("ort")
        )
        location_str: str | None = "Unbekannter Ort"
        if isinstance(location_raw, list) and location_raw:
            loc = location_raw[0]
            plz = loc.get("postleitzahl", "") or loc.get("plz", "")
            ort = loc.get("ort", "")
            region = loc.get("region", "")
            parts = [p for p in [plz, ort or region] if p]
            location_str = " ".join(parts) if parts else "Unbekannter Ort"
        elif isinstance(location_raw, dict):
            plz = location_raw.get("plz", "").strip()
            ort = location_raw.get("ort", "").strip()
            region = location_raw.get("region", "").strip()
            parts = [p for p in [plz, ort or region] if p]
            location_str = " ".join(parts) if parts else "Unbekannter Ort"
        elif isinstance(location_raw, str):
            location_str = location_raw.strip() or "Unbekannter Ort"

        # 5. Working time
        working_time = (
            data.get("arbeitszeitmodell") or data.get("arbeitszeit") or data.get("working_time")
        )

        # 6. Description / Teaser
        description = (
            data.get("stellenbeschreibung")
            or data.get("beschreibung")
            or data.get("description")
            or data.get("kurzbeschreibung")
        )

        # 7. External URL
        external_url = data.get("externeUrl") or data.get("external_url") or data.get("url")
        if not external_url and ref_nr:
            # Fallback to standard BA Jobsuche portal link format
            external_url = f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{ref_nr}"

        # 8. Published date
        published_date = (
            data.get("aktuelleVeroeffentlichungsdatum")
            or data.get("eintrittsdatum")
            or data.get("modifikationsTimestamp")
            or data.get("published_date")
        )

        # 9. Canonical hash (if already present)
        canonical_hash = data.get("canonical_hash")

        return cls(
            ref_nr=str(ref_nr),
            title=str(title).strip(),
            employer=str(employer).strip() if employer else None,
            location=location_str,
            working_time=str(working_time).strip() if working_time else None,
            description=str(description).strip() if description else None,
            external_url=str(external_url).strip() if external_url else None,
            published_date=published_date,
            canonical_hash=str(canonical_hash) if canonical_hash else None,
            raw_data=data,
        )


class BADetailedJob(BaseModel):
    """Full details of a job posting from Arbeitsagentur `/pc/v4/jobdetails/{code}`."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ref_nr: str = Field(description="Unique reference number")
    title: str = Field(description="Job title")
    employer: str | None = Field(default=None, description="Employer name")
    description: str | None = Field(default=None, description="Full job description")
    tasks: list[str] = Field(default_factory=list, description="List of duties and activities")
    requirements: list[str] = Field(
        default_factory=list, description="Requirements and qualifications"
    )
    locations: list[dict[str, Any]] = Field(
        default_factory=list, description="Structured workplace locations"
    )
    location_str: str | None = Field(
        default=None, description="Primary workplace location as formatted text"
    )
    working_time: str | None = Field(default=None, description="Working time / hours model")
    remuneration: str | None = Field(default=None, description="Salary or tariff remuneration")
    contract_duration: str | None = Field(
        default=None, description="Permanent or temporary contract"
    )
    entry_date: str | None = Field(default=None, description="Earliest entry / start date")
    contact: dict[str, Any] | None = Field(default=None, description="Contact information")
    external_url: str | None = Field(default=None, description="Application or portal URL")
    raw_data: dict[str, Any] | None = Field(default=None, description="Raw API response")

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> "BADetailedJob":
        """Factory method to parse BA detailed job payload."""
        ref_nr = str(data.get("refnr") or data.get("ref_nr") or data.get("hashId") or "")
        title = str(data.get("titel") or data.get("beruf") or data.get("title") or "")
        employer = data.get("arbeitgeber") or data.get("employer") or data.get("firma")
        description = (
            data.get("stellenbeschreibung") or data.get("beschreibung") or data.get("description")
        )

        # Tasks / Activities
        tasks_raw = data.get("taetigkeiten") or data.get("tasks") or []
        if isinstance(tasks_raw, str):
            tasks = [t.strip() for t in tasks_raw.split("\n") if t.strip()]
        elif isinstance(tasks_raw, list):
            tasks = [str(t).strip() for t in tasks_raw if t]
        else:
            tasks = []

        # Requirements
        reqs_raw = data.get("anforderungen") or data.get("requirements") or []
        if isinstance(reqs_raw, str):
            requirements = [r.strip() for r in reqs_raw.split("\n") if r.strip()]
        elif isinstance(reqs_raw, list):
            requirements = [str(r).strip() for r in reqs_raw if r]
        else:
            requirements = []

        # Locations
        locations_raw = data.get("arbeitsorte") or data.get("locations") or []
        locations: list[dict[str, Any]] = []
        location_str: str | None = None
        if isinstance(locations_raw, list) and locations_raw:
            for loc in locations_raw:
                if isinstance(loc, dict):
                    locations.append(loc)
            if locations:
                first = locations[0]
                plz = first.get("plz", "")
                ort = first.get("ort", "")
                strasse = first.get("strasse", "")
                parts = [p for p in [strasse, plz, ort] if p]
                location_str = ", ".join(parts) if parts else None
        elif isinstance(data.get("arbeitsort"), dict):
            loc_dict = data["arbeitsort"]
            locations.append(loc_dict)
            plz = loc_dict.get("plz", "")
            ort = loc_dict.get("ort", "")
            location_str = f"{plz} {ort}".strip() or None
        elif isinstance(data.get("arbeitsort"), str):
            location_str = data["arbeitsort"]

        return cls(
            ref_nr=ref_nr,
            title=title,
            employer=str(employer).strip() if employer else None,
            description=str(description).strip() if description else None,
            tasks=tasks,
            requirements=requirements,
            locations=locations,
            location_str=location_str,
            working_time=str(data.get("arbeitszeit") or data.get("arbeitszeitmodell") or "")
            or None,
            remuneration=str(data.get("verguetung") or data.get("gehalt") or "") or None,
            contract_duration=str(data.get("befristung") or "") or None,
            entry_date=str(data.get("eintrittsdatum") or "") or None,
            contact=data.get("kontakt") if isinstance(data.get("kontakt"), dict) else None,
            external_url=str(data.get("externeUrl") or data.get("url") or "") or None,
            raw_data=data,
        )


class BASearchResponse(BaseModel):
    """Parsed response envelope for BA search requests."""

    model_config = ConfigDict(extra="ignore")

    stellenangebote: list[BAJobListing] = Field(default_factory=list)
    max_ergebnisse: int = Field(
        default=0, description="Total matching listings found on Arbeitsagentur"
    )
    page: int = Field(default=1)
    size: int = Field(default=25)
    raw_data: dict[str, Any] | None = None


class DuplicateRecord(BaseModel):
    """Detailed record of a detected duplicate job."""

    model_config = ConfigDict(extra="ignore")

    ref_nr: str
    title: str
    employer: str | None = None
    canonical_hash: str
    reason: str = Field(description="e.g. 'intra_batch_duplicate', 'historically_seen'")
    matching_ref_nr: str | None = Field(
        default=None, description="Ref number of the original listing"
    )


class DeduplicationResult(BaseModel):
    """Summary and details of deduplication operation."""

    model_config = ConfigDict(extra="ignore")

    total_incoming: int
    unique_jobs: list[BAJobListing]
    duplicates_removed: int
    duplicate_records: list[DuplicateRecord] = Field(default_factory=list)
    unique_hashes: list[str] = Field(default_factory=list)
