"""Automated tests for 3-tier Job Deduplication Engine."""

import pytest

from app.schemas.job import BAJobListing, DeduplicationResult
from app.services.deduplicator import JobDeduplicator, normalize_umlauts


def test_normalize_umlauts():
    """Verify German umlauts are normalized to standard ASCII."""
    assert normalize_umlauts("München") == "Muenchen"
    assert normalize_umlauts("Köln") == "Koeln"
    assert normalize_umlauts("Nürnberg") == "Nuernberg"
    assert normalize_umlauts("Straße") == "Strasse"
    assert normalize_umlauts("Ärzte") == "Aerzte"
    assert normalize_umlauts("Österreich") == "Oesterreich"
    assert normalize_umlauts("Überlingen") == "Ueberlingen"
    assert normalize_umlauts("") == ""
    assert normalize_umlauts(None) == ""


@pytest.mark.parametrize(
    "raw_title, expected_clean",
    [
        ("Python Developer (m/w/d)", "python developer"),
        ("Senior Frontend Engineer (w/m/d)", "senior frontend engineer"),
        ("Backend Entwickler (d/m/w)", "backend entwickler"),
        ("Data Scientist (m/w/x)", "data scientist"),
        ("Pflegefachkraft (gn)", "pflegefachkraft"),
        ("IT-Consultant (all genders)", "it consultant"),
        ("DevOps Specialist (m/w/div.)", "devops specialist"),
        ("Cloud Architect [m/w/d]", "cloud architect"),
        ("Fullstack Entwickler / m/w/d", "fullstack entwickler"),
        ("Software Architect - m/w/d", "software architect"),
        ("Product Owner | gn", "product owner"),
        ("Projektleiter (m, w, d)", "projektleiter"),
        ("Systemadministrator:in", "systemadministrator"),
        ("Softwareentwickler*in", "softwareentwickler"),
        ("Sachbearbeiter_in", "sachbearbeiter"),
        ("Buchhalter/in", "buchhalter"),
        ("Marketing Manager (m/w/divers)", "marketing manager"),
        ("Junior Recruiter (w/m/div)", "junior recruiter"),
        ("QA Tester (f/m/d)", "qa tester"),
        ("Scrum Master (m/f/d)", "scrum master"),
        ("Werkstudent (m/w/d) - Softwareentwicklung / KI", "werkstudent softwareentwicklung ki"),
        ("🚀 Cloud Security Engineer (m/w/d) 🔥", "cloud security engineer"),
        ("", ""),
        (None, ""),
    ],
)
def test_title_gender_normalization(raw_title, expected_clean):
    """Verify title normalizer strips diverse gender markers and cleans punctuation."""
    assert JobDeduplicator.normalize_title(raw_title) == expected_clean


@pytest.mark.parametrize(
    "raw_employer, expected_clean",
    [
        ("Siemens AG", "siemens"),
        ("Musterfirma GmbH", "musterfirma"),
        ("Automotive Parts GmbH & Co. KG", "automotive parts"),
        ("Future Tech SE & Co. KGaA", "future tech"),
        ("Startup Ventures UG (haftungsbeschränkt)", "startup ventures"),
        ("Deutscher Alpenverein e.V.", "deutscher alpenverein"),
        ("Gemeinnütziger Verein e. V.", "gemeinnuetziger verein"),
        ("N.O.C. Solutions", "noc solutions"),
        ("I.B.M. Deutschland", "ibm deutschland"),
        ("S.A.P. Global", "sap global"),
        ("A.B.C. Consulting Ltd.", "abc consulting"),
        ("Acme International Inc.", "acme international"),
        ("Global Logistics LLC", "global logistics"),
        ("Fintech Holding", "fintech"),
        ("", ""),
        (None, ""),
    ],
)
def test_employer_normalization(raw_employer, expected_clean):
    """Verify employer normalizer cleans legal forms and resolves dotted abbreviations."""
    assert JobDeduplicator.normalize_employer(raw_employer) == expected_clean


@pytest.mark.parametrize(
    "raw_location, expected_clean",
    [
        ("10115 Berlin", "berlin"),
        ("80331 München, Bayern", "muenchen bayern"),
        ("50667 Köln", "koeln"),
        ("Frankfurt am Main", "frankfurt am main"),
        ("70173 Stuttgart (Baden-Württemberg)", "stuttgart baden wuerttemberg"),
        ("", ""),
        (None, ""),
    ],
)
def test_location_normalization(raw_location, expected_clean):
    """Verify location normalizer strips 5-digit postal codes and normalizes umlauts."""
    assert JobDeduplicator.normalize_location(raw_location) == expected_clean


def test_description_normalization():
    """Verify description normalizer strips HTML and collapses whitespace."""
    raw_html = "<p>Wir suchen <strong>Python Entwickler</strong> ab sofort.<br>Ihre Aufgaben:<br>• Coding</p>"
    clean = JobDeduplicator.normalize_description(raw_html)
    assert "<p>" not in clean
    assert "<strong>" not in clean
    assert "<br>" not in clean
    assert "wir suchen python entwickler ab sofort ihre aufgaben coding" in clean
    assert JobDeduplicator.normalize_description(None) == ""
    assert JobDeduplicator.normalize_description("") == ""


def test_canonical_hash_equivalence():
    """Verify identical jobs with varying gender markers and legal forms produce the SAME canonical hash."""
    job_var_1 = {
        "title": "Senior Python Backend Engineer (m/w/d)",
        "employer": "Tech Corp GmbH & Co. KG",
        "location": "10117 Berlin",
        "description": "<p>Build scalable services with Python</p>",
    }
    job_var_2 = {
        "title": "Senior Python Backend Engineer [gn]",
        "employer": "Tech Corp GmbH",
        "location": "Berlin",
        "description": "Build scalable services with Python",
    }
    job_var_3 = {
        "title": "Senior Python Backend Engineer / w/m/div.",
        "employer": "Tech Corp",
        "location": "10117 Berlin",
        "description": "Build scalable services with Python",
    }

    hash1 = JobDeduplicator.compute_canonical_hash(**job_var_1)
    hash2 = JobDeduplicator.compute_canonical_hash(**job_var_2)
    hash3 = JobDeduplicator.compute_canonical_hash(**job_var_3)

    assert hash1 == hash2 == hash3
    assert len(hash1) == 64  # SHA-256 hex string


def test_canonical_hash_distinctiveness():
    """Verify distinct job postings produce DIFFERENT canonical hashes."""
    hash_dev = JobDeduplicator.compute_canonical_hash(
        title="Python Developer",
        employer="Company A",
        location="Berlin",
    )
    hash_pm = JobDeduplicator.compute_canonical_hash(
        title="Product Manager",
        employer="Company A",
        location="Berlin",
    )
    hash_diff_loc = JobDeduplicator.compute_canonical_hash(
        title="Python Developer",
        employer="Company A",
        location="Hamburg",
    )
    hash_diff_emp = JobDeduplicator.compute_canonical_hash(
        title="Python Developer",
        employer="Company B",
        location="Berlin",
    )

    assert hash_dev != hash_pm
    assert hash_dev != hash_diff_loc
    assert hash_dev != hash_diff_emp


def test_intra_batch_deduplication():
    """Verify intra-batch deduplication filters out repeated postings in the same batch."""
    batch = [
        BAJobListing(
            ref_nr="REF-001",
            title="DevOps Engineer (m/w/d)",
            employer="Cloud Sys AG",
            location="10115 Berlin",
        ),
        BAJobListing(
            ref_nr="REF-002",
            title="DevOps Engineer (gn)",
            employer="Cloud Sys GmbH",
            location="Berlin",
        ),
        BAJobListing(
            ref_nr="REF-003",
            title="Frontend Developer (w/m/d)",
            employer="Cloud Sys AG",
            location="10115 Berlin",
        ),
        BAJobListing(
            ref_nr="REF-001",  # Same ref_nr
            title="DevOps Engineer (m/w/d)",
            employer="Cloud Sys AG",
            location="10115 Berlin",
        ),
    ]

    unique_jobs = JobDeduplicator.filter_duplicates(batch)

    assert len(unique_jobs) == 2
    assert unique_jobs[0].ref_nr == "REF-001"
    assert unique_jobs[1].ref_nr == "REF-003"
    assert unique_jobs[0].canonical_hash is not None
    assert unique_jobs[1].canonical_hash is not None


def test_historical_deduplication():
    """Verify historical deduplication filters out jobs matching seen hashes and seen ref_nrs."""
    existing_hash = JobDeduplicator.compute_canonical_hash(
        title="Data Analyst",
        employer="Data Corp GmbH",
        location="München",
    )
    seen_hashes = {existing_hash}
    seen_ref_nrs = {"REF-HISTORICAL-999"}

    incoming = [
        # Match by canonical hash (different ref_nr, but same normalized attributes)
        BAJobListing(
            ref_nr="REF-NEW-100",
            title="Data Analyst (m/w/d)",
            employer="Data Corp",
            location="80331 München",
        ),
        # Match by historical ref_nr
        BAJobListing(
            ref_nr="REF-HISTORICAL-999",
            title="Completely New Title",
            employer="Other Corp",
            location="Köln",
        ),
        # Genuinely new job
        BAJobListing(
            ref_nr="REF-GENUINE-200",
            title="Junior Data Engineer (gn)",
            employer="Data Corp",
            location="80331 München",
        ),
    ]

    report: DeduplicationResult = JobDeduplicator.deduplicate_with_report(
        incoming_jobs=incoming,
        seen_hashes=seen_hashes,
        seen_ref_nrs=seen_ref_nrs,
    )

    assert report.total_incoming == 3
    assert report.duplicates_removed == 2
    assert len(report.unique_jobs) == 1
    assert report.unique_jobs[0].ref_nr == "REF-GENUINE-200"

    reasons = [r.reason for r in report.duplicate_records]
    assert "historically_seen_canonical_hash" in reasons
    assert "historically_seen_ref_nr" in reasons


def test_deduplicate_raw_dicts():
    """Verify deduplicator accepts raw API dicts and parses them into unique BAJobListings."""
    raw_data = [
        {
            "refnr": "10000-1111",
            "titel": "Python Entwickler (m/w/d)",
            "arbeitgeber": "Alpha Corp GmbH",
            "arbeitsort": {"plz": "10117", "ort": "Berlin"},
        },
        {
            "refnr": "10000-2222",
            "titel": "Python Entwickler (gn)",
            "arbeitgeber": "Alpha Corp",
            "arbeitsort": "Berlin",
        },
        {
            "refnr": "10000-3333",
            "titel": "Java Entwickler",
            "arbeitgeber": "Beta AG",
            "arbeitsort": "Frankfurt",
        },
    ]

    unique_jobs = JobDeduplicator.filter_duplicates(raw_data)
    assert len(unique_jobs) == 2
    assert unique_jobs[0].ref_nr == "10000-1111"
    assert unique_jobs[1].ref_nr == "10000-3333"


def test_deduplicate_empty_list():
    """Verify deduplicating empty list produces valid empty result."""
    assert JobDeduplicator.filter_duplicates([]) == []
    report = JobDeduplicator.deduplicate_with_report([])
    assert report.total_incoming == 0
    assert report.duplicates_removed == 0
    assert len(report.unique_jobs) == 0
