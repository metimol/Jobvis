"""3-tier Job Deduplication Engine for Bundesagentur für Arbeit listings."""

import hashlib
import logging
import re
import unicodedata
from typing import Any

from app.schemas.job import (
    BAJobListing,
    DeduplicationResult,
    DuplicateRecord,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def normalize_umlauts(text: str) -> str:
    """Normalize German umlauts and special characters to standard ASCII representations."""
    if not text:
        return ""
    # Precompose decomposed NFD characters (e.g. Ba\u0308cker -> Bäcker)
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "Ä": "Ae",
        "Ö": "Oe",
        "Ü": "Ue",
        "ß": "ss",
        "ẞ": "ss",  # Uppercase German sharp S (U+1E9E)
    }
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    # Remove any lingering combining diacritics
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Regex Patterns for Normalization
# ---------------------------------------------------------------------------

# German & English gender markers in job titles:
# e.g., (m/w/d), (w/m/d), (d/m/w), (m/w/x), (gn), (all genders), (m/w/div.), [m/w/d], / m/w/d, etc.
GENDER_MARKER_PATTERNS = [
    # Bracketed / parenthesized gender markers: (m/w/d), [w/m/x], (gn), (m, w, d), (m/w/divers), (all genders), etc.
    re.compile(
        r"[\(\[\{]\s*(?:m|w|d|f|x|gn|divers|div\.?|all\s+genders?)(?:[/\-–—,_\|\\]\s*(?:m|w|d|f|x|gn|divers|div\.?|all\s+genders?))*\s*[\)\]\}]",
        re.IGNORECASE,
    ),
    # Delimited by slash, hyphen, pipe, colon: e.g. " / m/w/d", " - w/m/div.", " | gn", " : m/w/d"
    re.compile(
        r"(?:^|\s+)[\-–—/_\|:]\s*(?:m/w/d|w/m/d|d/m/w|m/w/x|w/m/x|m/f/d|f/m/d|m/w/gn|gn|all\s+genders?|m/w/div(?:ers|\.)?|w/m/div(?:ers|\.)?|d/w/m)(?:\s+|$)",
        re.IGNORECASE,
    ),
    # Gendered suffixes like *in, :in, _in, /in, /-in, -in, *innen, :innen, _innen, /-innen
    re.compile(r"[\*:\_/–—\-]+in(?:nen)?\b", re.IGNORECASE),
    re.compile(r"[\*:\_/–—\-]+frau(?:en)?\b", re.IGNORECASE),
    re.compile(r"[\*:\_/–—\-]+mann\b", re.IGNORECASE),
    # Parenthetical gender suffixes: (in), (innen)
    re.compile(r"\(\s*in(?:nen)?\s*\)", re.IGNORECASE),
    # Standalone trailing / boundary gender marker
    re.compile(
        r"\b(?:m/w/d|w/m/d|d/m/w|m/w/x|w/m/x|m/f/d|f/m/d|m/w/gn|all\s+genders?|m/w/div(?:ers|\.)?|w/m/div(?:ers|\.)?)\b",
        re.IGNORECASE,
    ),
]

# Legal entity suffixes to strip from employer names
# e.g., GmbH & Co. KG aA, AG, UG (haftungsbeschränkt), e.V., e.K., SE, Inc., LLC, Ltd.
LEGAL_FORM_PATTERNS = [
    # Complex multi-word legal forms first (GmbH & Co. KG aA, GmbH & Co. KGaA, etc.)
    re.compile(r"\b(?:gmbh|ag|se)\s*&\s*co\.?\s*(?:kg\s*a\.?a\.?|kgaa)\b", re.IGNORECASE),
    re.compile(r"\b(?:gmbh|ag|se)\s*&\s*co\.?\s*kg\b", re.IGNORECASE),
    re.compile(r"\bgmbh\s*&\s*co\b", re.IGNORECASE),
    re.compile(r"\bgmbh\s*&\s*cie\.?\b", re.IGNORECASE),
    re.compile(r"\b(?:g?ug)\s*\(\s*haftungsbeschr[äa]nkt\s*\)", re.IGNORECASE),
    re.compile(r"\b(?:g?ug)\s*\(haftungsbeschraenkt\)", re.IGNORECASE),
    re.compile(r"\bpartg\s*mbb\b", re.IGNORECASE),
    re.compile(r"\bpartg\b", re.IGNORECASE),
    # Registered merchant variants: e.K., e.Kfm., e.Kfr., eingetragener Kaufmann / Kauffrau
    re.compile(r"\be\.\s*k(?:fm|fr)?\.?\b", re.IGNORECASE),
    re.compile(r"\be\.k(?:fm|fr)?\b", re.IGNORECASE),
    re.compile(r"\beingetragene?[rn]?\s+kauf(?:mann|frau)\b", re.IGNORECASE),
    # Registered associations: e.V., e. V., eingetragener Verein (requires dots or end-of-string eV)
    re.compile(r"\beingetragener\s+verein\b", re.IGNORECASE),
    re.compile(r"\be\.\s*v\.?\b", re.IGNORECASE),
    re.compile(r"\be\.v\b", re.IGNORECASE),
    re.compile(r"(?:\s+e\.?\s*v\.?)+$", re.IGNORECASE),
    # Standard single-word legal forms (including gGmbH, gUG, gAG, etc.)
    re.compile(r"\bg?gmbh\b", re.IGNORECASE),
    re.compile(r"\bg?kgaa\b", re.IGNORECASE),
    re.compile(r"\bg?ug\b", re.IGNORECASE),
    re.compile(r"\bohg\b", re.IGNORECASE),
    re.compile(r"\bgbr\b", re.IGNORECASE),
    re.compile(r"\bkg\b", re.IGNORECASE),
    re.compile(r"\bg?ag\b", re.IGNORECASE),
    re.compile(r"\bse\b", re.IGNORECASE),
    re.compile(r"\binc\.?\b", re.IGNORECASE),
    re.compile(r"\bllc\.?\b", re.IGNORECASE),
    re.compile(r"\bltd\.?\b", re.IGNORECASE),
    re.compile(r"\bcorp\.?\b", re.IGNORECASE),
    re.compile(r"\bcorporation\b", re.IGNORECASE),
    # Suffix-only forms (e.g., 'Fintech Holding' -> 'fintech', while preserving 'Holding Solutions')
    re.compile(r"\bholding\s*$", re.IGNORECASE),
]

# HTML tags cleaner
HTML_TAGS_PATTERN = re.compile(r"<[^>]+>")


class JobDeduplicator:
    """3-tier Canonical Job Normalization & Deduplication Engine."""

    @classmethod
    def normalize_title(cls, title: str | None) -> str:
        """Normalize job title by removing gender markers, casing, punctuation, and extra whitespace.

        Examples:
            'Softwareentwickler Python (m/w/d)' -> 'softwareentwickler python'
            'Senior Frontend Developer (gn) / Berlin' -> 'senior frontend developer berlin'
            'Data Scientist (w/m/div.)' -> 'data scientist'
            'Fullstack Entwickler*in' -> 'fullstack entwickler'
        """
        if not title:
            return ""

        text = title.strip()

        # 1. Apply gender marker cleaner patterns
        for pattern in GENDER_MARKER_PATTERNS:
            text = pattern.sub(" ", text)

        # 2. Normalize umlauts and lowercase
        text = normalize_umlauts(text).lower()

        # 3. Replace non-alphanumeric characters (except spaces) with space
        text = re.sub(r"[^\w\s]", " ", text)

        # 4. Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def normalize_employer(cls, employer: str | None) -> str:
        """Normalize employer name by resolving dotted abbreviations and removing legal forms.

        Examples:
            'Siemens AG' -> 'siemens'
            'Musterfirma GmbH & Co. KG' -> 'musterfirma'
            'N.O.C. Solutions UG (haftungsbeschränkt)' -> 'noc solutions'
            'Deutsche Bahn AG' -> 'deutsche bahn'
            'A.B.C. Consulting Ltd.' -> 'abc consulting'
        """
        if not employer:
            return ""

        text = employer.strip()

        # 1. Normalize umlauts and lowercase
        text = normalize_umlauts(text).lower()

        # 2. Remove legal forms before stripping acronym dots
        for pattern in LEGAL_FORM_PATTERNS:
            text = pattern.sub(" ", text)

        # 3. Handle dotted single-letter abbreviations: 'n.o.c.' -> 'noc', 'i.b.m.' -> 'ibm'
        text = re.sub(r"\b([a-zA-Z])\.", r"\1", text)

        # 4. Clean non-alphanumeric characters and collapse spaces
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def normalize_location(cls, location: str | None) -> str:
        """Normalize location by stripping postal codes, umlauts, and punctuation.

        Examples:
            '10115 Berlin' -> 'berlin'
            '80331 München, Bayern' -> 'muenchen bayern'
            'Frankfurt am Main' -> 'frankfurt am main'
        """
        if not location:
            return ""

        text = location.strip()

        # 1. Remove 5-digit German postal codes (e.g. 10115, 80331)
        text = re.sub(r"\b\d{5}\b", " ", text)

        # 2. Normalize umlauts and lowercase
        text = normalize_umlauts(text).lower()

        # 3. Clean punctuation and collapse spaces
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def normalize_description(cls, description: str | None, max_chars: int = 600) -> str:
        """Strip HTML tags and normalize description text prefix for fingerprinting."""
        if not description:
            return ""

        text = description.strip()

        # 1. Strip HTML tags
        text = HTML_TAGS_PATTERN.sub(" ", text)

        # 2. Normalize umlauts and lowercase
        text = normalize_umlauts(text).lower()

        # 3. Clean punctuation and collapse spaces
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:max_chars]

    @classmethod
    def compute_canonical_hash(
        cls,
        title: str,
        employer: str | None = None,
        location: str | None = None,
        description: str | None = None,
    ) -> str:
        """Generate a deterministic 64-character SHA-256 composite fingerprint.

        Fingerprint schema:
            SHA-256(norm_title | norm_employer | norm_location | desc_subhash)
        """
        norm_title = cls.normalize_title(title)
        norm_emp = cls.normalize_employer(employer)
        norm_loc = cls.normalize_location(location)
        norm_desc = cls.normalize_description(description)

        desc_subhash = (
            hashlib.sha256(norm_desc.encode("utf-8")).hexdigest()[:16] if norm_desc else ""
        )

        composite_string = f"{norm_title}|{norm_emp}|{norm_loc}|{desc_subhash}"
        return hashlib.sha256(composite_string.encode("utf-8")).hexdigest()

    @classmethod
    def filter_duplicates(
        cls,
        incoming_jobs: list[BAJobListing | dict[str, Any]],
        seen_hashes: set[str] | None = None,
        seen_ref_nrs: set[str] | None = None,
    ) -> list[BAJobListing]:
        """Filter out duplicate job postings from an incoming batch.

        Args:
            incoming_jobs: List of BAJobListing instances or raw API dicts.
            seen_hashes: Optional set of SHA-256 hashes previously seen/stored in DB.
            seen_ref_nrs: Optional set of reference numbers previously seen/stored in DB.

        Returns:
            List of unique BAJobListing objects with canonical_hash populated.
        """
        result = cls.deduplicate_with_report(
            incoming_jobs=incoming_jobs,
            seen_hashes=seen_hashes,
            seen_ref_nrs=seen_ref_nrs,
        )
        return result.unique_jobs

    @classmethod
    def deduplicate_with_report(
        cls,
        incoming_jobs: list[BAJobListing | dict[str, Any]],
        seen_hashes: set[str] | None = None,
        seen_ref_nrs: set[str] | None = None,
    ) -> DeduplicationResult:
        """Perform 3-tier deduplication and produce a detailed audit report.

        Tiers:
        1. Exact reference number match against historical seen ref numbers.
        2. Exact canonical hash match against historical seen hashes.
        3. Intra-batch duplicate elimination (ref_nr and canonical composite hash).
        """
        historical_hashes: set[str] = set(seen_hashes) if seen_hashes else set()
        historical_refs: set[str] = set(seen_ref_nrs) if seen_ref_nrs else set()

        batch_hashes: set[str] = set()
        batch_refs: set[str] = set()
        hash_to_primary_ref: dict[str, str] = {}

        unique_jobs: list[BAJobListing] = []
        duplicate_records: list[DuplicateRecord] = []

        for item in incoming_jobs:
            if isinstance(item, dict):
                job = BAJobListing.from_api_dict(item)
            elif isinstance(item, BAJobListing):
                job = item
            else:
                continue

            ref_nr = job.ref_nr.strip()
            title = job.title or ""
            employer = job.employer or ""
            location = job.location or ""
            description = job.description or ""

            # Compute canonical SHA-256 composite fingerprint
            c_hash = cls.compute_canonical_hash(
                title=title,
                employer=employer,
                location=location,
                description=description,
            )
            job.canonical_hash = c_hash

            # Check 1: Historical Reference Number duplicate
            if ref_nr and ref_nr in historical_refs:
                duplicate_records.append(
                    DuplicateRecord(
                        ref_nr=ref_nr,
                        title=title,
                        employer=employer,
                        canonical_hash=c_hash,
                        reason="historically_seen_ref_nr",
                        matching_ref_nr=ref_nr,
                    )
                )
                continue

            # Check 2: Historical Canonical Hash duplicate
            if c_hash in historical_hashes:
                duplicate_records.append(
                    DuplicateRecord(
                        ref_nr=ref_nr,
                        title=title,
                        employer=employer,
                        canonical_hash=c_hash,
                        reason="historically_seen_canonical_hash",
                        matching_ref_nr=None,
                    )
                )
                continue

            # Check 3: Intra-batch Reference Number duplicate
            if ref_nr and ref_nr in batch_refs:
                duplicate_records.append(
                    DuplicateRecord(
                        ref_nr=ref_nr,
                        title=title,
                        employer=employer,
                        canonical_hash=c_hash,
                        reason="intra_batch_duplicate_ref_nr",
                        matching_ref_nr=ref_nr,
                    )
                )
                continue

            # Check 4: Intra-batch Canonical Hash duplicate
            if c_hash in batch_hashes:
                duplicate_records.append(
                    DuplicateRecord(
                        ref_nr=ref_nr,
                        title=title,
                        employer=employer,
                        canonical_hash=c_hash,
                        reason="intra_batch_duplicate_canonical_hash",
                        matching_ref_nr=hash_to_primary_ref.get(c_hash),
                    )
                )
                continue

            # It is a unique job!
            if ref_nr:
                batch_refs.add(ref_nr)
            batch_hashes.add(c_hash)
            hash_to_primary_ref[c_hash] = ref_nr
            unique_jobs.append(job)

        return DeduplicationResult(
            total_incoming=len(incoming_jobs),
            unique_jobs=unique_jobs,
            duplicates_removed=len(duplicate_records),
            duplicate_records=duplicate_records,
            unique_hashes=list(batch_hashes),
        )
