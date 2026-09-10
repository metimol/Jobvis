"""Tests for multi-tier job deduplication, fuzzy matching, and cross-sector deduplication."""

import time

import httpx
import pytest
import respx

from app.schemas.job import BAJobListing, DeduplicationResult
from app.services.arbeitsagentur import (
    DEFAULT_BASE_URL,
    ArbeitsagenturAuthError,
    ArbeitsagenturClient,
)
from app.services.deduplicator import JobDeduplicator, normalize_umlauts

# --- Core Deduplicator Tests ---


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


# --- Adversarial Deduplicator & Normalization Tests ---

# ===========================================================================
# 1. Title Normalization & Gender Markers
# ===========================================================================

ADVERSARIAL_TITLE_TEST_CASES = [
    # Bracketed variations
    ("Senior Python Developer (m/w/d)", "senior python developer"),
    ("Frontend Engineer [w/m/d]", "frontend engineer"),
    ("Backend Architect {d/m/w}", "backend architect"),
    ("Data Scientist (m/w/x)", "data scientist"),
    ("Cloud Specialist (w/m/x)", "cloud specialist"),
    ("DevOps Engineer (m/f/d)", "devops engineer"),
    ("Platform Engineer (f/m/d)", "platform engineer"),
    ("Site Reliability Engineer (gn)", "site reliability engineer"),
    ("Security Analyst (all genders)", "security analyst"),
    ("Lead Architect (m/w/divers)", "lead architect"),
    ("Product Owner (w/m/div.)", "product owner"),
    ("Scrum Master (m/w/div)", "scrum master"),
    ("QA Engineer (m, w, d)", "qa engineer"),
    ("UX Designer (m,w,d)", "ux designer"),
    ("Mobile Developer (d/w/m)", "mobile developer"),
    # Delimited markers
    ("Fullstack Developer / m/w/d", "fullstack developer"),
    ("Software Engineer - w/m/d", "software engineer"),
    ("Database Admin | gn", "database admin"),
    ("Solutions Architect : m/w/div.", "solutions architect"),
    ("Engineering Manager — all genders", "engineering manager"),
    # Suffix markers
    ("Softwareentwickler*in", "softwareentwickler"),
    ("Systemadministrator:in", "systemadministrator"),
    ("Sachbearbeiter_in", "sachbearbeiter"),
    ("Buchhalter/in", "buchhalter"),
    ("Kaufmann/-frau fuer Bueromanagement", "kaufmann fuer bueromanagement"),
    ("Berater*innen", "berater"),
    ("Entwickler:innen", "entwickler"),
    ("Experten_innen", "experten"),
    # Unicode, Emojis, and Punctuation
    ("🚀 Cloud Security Architect (m/w/d) 🔥", "cloud security architect"),
    ("⚡ Lead AI Engineer [gn] 🤖", "lead ai engineer"),
    ("C# / .NET Core Developer (m/w/d)", "c net core developer"),
    ("Senior C++ Engineer (w/m/d)", "senior c engineer"),
    ("Node.js & React.js Developer (gn)", "node js react js developer"),
    ("R&D Specialist (m/w/d)", "r d specialist"),
    ("   Senior   Backend    Engineer    (m/w/d)   ", "senior backend engineer"),
    ("Software Developer\n\t(w/m/d)\r\n", "software developer"),
]


@pytest.mark.parametrize("raw_title, expected", ADVERSARIAL_TITLE_TEST_CASES)
def test_adversarial_title_normalization(raw_title, expected):
    clean = JobDeduplicator.normalize_title(raw_title)
    assert (
        clean == expected
    ), f"Failed for title '{raw_title}': got '{clean}', expected '{expected}'"


def test_standard_title_umlaut_handling():
    """Verify standard NFC German umlaut normalization in titles."""
    t1 = JobDeduplicator.normalize_title("Geschäftsführer (m/w/d)")
    assert t1 == "geschaeftsfuehrer"
    t2 = JobDeduplicator.normalize_title("Köchin / Koch (m/w/d)")
    assert t2 == "koechin koch"


# ===========================================================================
# 2. Employer Normalization & German Legal Forms
# ===========================================================================

ADVERSARIAL_EMPLOYER_TEST_CASES = [
    ("Siemens AG", "siemens"),
    ("Robert Bosch GmbH", "robert bosch"),
    ("Automotive Parts GmbH & Co. KG", "automotive parts"),
    ("Future Tech SE & Co. KGaA", "future tech"),
    ("Precision Tools GmbH & Cie.", "precision tools"),
    ("NextGen UG (haftungsbeschränkt)", "nextgen"),
    ("NextGen UG (haftungsbeschraenkt)", "nextgen"),
    ("Advokaten PartG mbB", "advokaten"),
    ("Steuerberater PartG", "steuerberater"),
    ("Deutscher Alpenverein e.V.", "deutscher alpenverein"),
    ("Sportverein 1860 e. V.", "sportverein 1860"),
    ("Kulturförderung eV", "kulturfoerderung"),
    ("Mueller KGaA", "mueller"),
    ("Schmidt OHG", "schmidt"),
    ("Baecker GbR", "baecker"),
    ("Logistics KG", "logistics"),
    ("Global Solutions SE", "global solutions"),
    ("Acme International Inc.", "acme international"),
    ("Pacific LLC", "pacific"),
    ("British Trade Ltd.", "british trade"),
    ("Global Corp.", "global"),
    ("Enterprise Corporation", "enterprise"),
    ("I.B.M. Deutschland", "ibm deutschland"),
    ("N.O.C. Systems", "noc systems"),
    ("S.A.P. Global", "sap global"),
    ("A.B.C. Consulting Ltd.", "abc consulting"),
    ("M.A.N. Truck & Bus", "man truck bus"),
]


@pytest.mark.parametrize("raw_employer, expected", ADVERSARIAL_EMPLOYER_TEST_CASES)
def test_adversarial_employer_normalization(raw_employer, expected):
    clean = JobDeduplicator.normalize_employer(raw_employer)
    assert (
        clean == expected
    ), f"Failed for employer '{raw_employer}': got '{clean}', expected '{expected}'"


# ===========================================================================
# 3. Location Normalization & Fuzzy Boundaries
# ===========================================================================

ADVERSARIAL_LOCATION_TEST_CASES = [
    ("10115 Berlin", "berlin"),
    ("D-10115 Berlin", "d berlin"),
    ("80331 München, Bayern", "muenchen bayern"),
    ("50667 Köln", "koeln"),
    ("60311 Frankfurt am Main", "frankfurt am main"),
    ("70173 Stuttgart (Baden-Württemberg)", "stuttgart baden wuerttemberg"),
    ("90403 Nürnberg", "nuernberg"),
    ("01067 Dresden, Sachsen", "dresden sachsen"),
    ("Berlin - Mitte", "berlin mitte"),
    ("Hamburg / Altona", "hamburg altona"),
]


@pytest.mark.parametrize("raw_loc, expected", ADVERSARIAL_LOCATION_TEST_CASES)
def test_adversarial_location_normalization(raw_loc, expected):
    clean = JobDeduplicator.normalize_location(raw_loc)
    assert (
        clean == expected
    ), f"Failed for location '{raw_loc}': got '{clean}', expected '{expected}'"


# ===========================================================================
# 4. True Duplicates Equivalence vs False Positives Distinction
# ===========================================================================


def test_true_duplicates_produce_identical_canonical_hashes():
    """Verify that realistic variations of the SAME job produce identical hashes."""
    variations = [
        {
            "title": "Senior Cloud Security Engineer (m/w/d)",
            "employer": "CyberGuard Solutions GmbH & Co. KG",
            "location": "10117 Berlin",
            "description": "<p>Lead our cloud security team with AWS and Kubernetes.</p>",
        },
        {
            "title": "Senior Cloud Security Engineer [gn]",
            "employer": "CyberGuard Solutions GmbH",
            "location": "Berlin",
            "description": "Lead our cloud security team with AWS and Kubernetes.",
        },
        {
            "title": "Senior Cloud Security Engineer / w/m/div.",
            "employer": "CyberGuard Solutions",
            "location": "10117 Berlin",
            "description": "<div>Lead our cloud security team with AWS and Kubernetes.</div>",
        },
        {
            "title": "🚀 Senior Cloud Security Engineer (all genders) 🔥",
            "employer": "CyberGuard Solutions",
            "location": "Berlin",
            "description": "Lead our cloud security team with AWS and Kubernetes.",
        },
    ]

    hashes = [JobDeduplicator.compute_canonical_hash(**var) for var in variations]
    assert len(set(hashes)) == 1, f"Expected all hashes to be identical, but got {set(hashes)}"


def test_distinct_jobs_never_collapse_false_positives():
    """Verify that distinct jobs at different locations, different roles, or employers NEVER collapse."""
    base_job = {
        "title": "Senior Python Developer (m/w/d)",
        "employer": "TechNova AG",
        "location": "Berlin",
        "description": "Backend development with FastAPI and PostgreSQL.",
    }
    base_hash = JobDeduplicator.compute_canonical_hash(**base_job)

    # 1. Different seniority / role
    job_junior = dict(base_job, title="Junior Python Developer (m/w/d)")
    assert JobDeduplicator.compute_canonical_hash(**job_junior) != base_hash

    # 2. Different tech stack / role
    job_frontend = dict(base_job, title="Senior Frontend Developer (m/w/d)")
    assert JobDeduplicator.compute_canonical_hash(**job_frontend) != base_hash

    # 3. Different branch location (Berlin vs Munich vs Hamburg)
    job_munich = dict(base_job, location="München")
    job_hamburg = dict(base_job, location="Hamburg")
    assert JobDeduplicator.compute_canonical_hash(**job_munich) != base_hash
    assert JobDeduplicator.compute_canonical_hash(**job_hamburg) != base_hash
    assert JobDeduplicator.compute_canonical_hash(
        **job_munich
    ) != JobDeduplicator.compute_canonical_hash(**job_hamburg)

    # 4. Different employer in the same city
    job_diff_emp = dict(base_job, employer="Bosch GmbH")
    assert JobDeduplicator.compute_canonical_hash(**job_diff_emp) != base_hash

    # 5. Frankfurt am Main vs Frankfurt (Oder)
    job_ffm = dict(base_job, location="Frankfurt am Main")
    job_ffo = dict(base_job, location="Frankfurt Oder")
    assert JobDeduplicator.compute_canonical_hash(
        **job_ffm
    ) != JobDeduplicator.compute_canonical_hash(**job_ffo)


# ===========================================================================
# 5. Stress Testing Large Batches (1,000 jobs) with Performance Benchmarks
# ===========================================================================


def test_large_batch_stress_deduplication():
    """Stress-test deduplication with 1,000 mixed listings (duplicates, variations, historical, distinct)."""
    start_time = time.perf_counter()

    num_distinct_jobs = 100
    incoming_batch: list[BAJobListing] = []
    historical_hashes: set[str] = set()
    historical_refs: set[str] = set()

    for i in range(50):
        ref = f"HIST-REF-{i:04d}"
        historical_refs.add(ref)
        h = JobDeduplicator.compute_canonical_hash(
            title=f"Historical Job Position {i}",
            employer=f"Historical Company {i} GmbH",
            location="Berlin",
        )
        historical_hashes.add(h)

    for i in range(num_distinct_jobs):
        incoming_batch.append(
            BAJobListing(
                ref_nr=f"PRIMARY-{i:04d}",
                title=f"Software Engineer {i} (m/w/d)",
                employer=f"Company {i} GmbH & Co. KG",
                location=f"1011{i%10} Berlin",
                description=f"Description for job {i}",
            )
        )

    for i in range(300):
        target_idx = i % num_distinct_jobs
        incoming_batch.append(
            BAJobListing(
                ref_nr=f"VAR-{i:04d}",
                title=f"Software Engineer {target_idx} [gn]",
                employer=f"Company {target_idx} GmbH",
                location="Berlin",
                description=f"<p>Description for job {target_idx}</p>",
            )
        )

    for i in range(200):
        target_idx = i % num_distinct_jobs
        incoming_batch.append(
            BAJobListing(
                ref_nr=f"PRIMARY-{target_idx:04d}",
                title=f"Software Engineer {target_idx} (m/w/d)",
                employer=f"Company {target_idx} GmbH & Co. KG",
                location=f"1011{target_idx%10} Berlin",
            )
        )

    for i in range(200):
        if i % 2 == 0:
            incoming_batch.append(
                BAJobListing(
                    ref_nr=f"HIST-REF-{(i//2)%50:04d}",
                    title="New Title But Historical Ref",
                    employer="Some Corp",
                )
            )
        else:
            hist_idx = (i // 2) % 50
            incoming_batch.append(
                BAJobListing(
                    ref_nr=f"NEW-REF-FOR-HIST-{i:04d}",
                    title=f"Historical Job Position {hist_idx} (w/m/d)",
                    employer=f"Historical Company {hist_idx}",
                    location="Berlin",
                )
            )

    for i in range(200):
        group_idx = i % 20
        incoming_batch.append(
            BAJobListing(
                ref_nr=f"SPARSE-{i:04d}",
                title=f"Sparse Position {group_idx}",
                employer=None,
                location=None,
                description=None,
            )
        )

    report: DeduplicationResult = JobDeduplicator.deduplicate_with_report(
        incoming_jobs=incoming_batch,
        seen_hashes=historical_hashes,
        seen_ref_nrs=historical_refs,
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert report.total_incoming == 1000
    assert len(report.unique_jobs) == 120
    assert report.duplicates_removed == 880
    assert len(report.duplicate_records) == 880
    assert elapsed_ms < 1500, f"Deduplication took {elapsed_ms:.2f}ms (expected < 1500ms)"


# ===========================================================================
# 6. Arbeitsagentur REST API Client Robustness & Error Recovery
# ===========================================================================


@pytest.mark.asyncio
@respx.mock
async def test_arbeitsagentur_client_query_param_encoding():
    """Verify complex search queries with symbols, spaces, and umlauts are correctly encoded."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(200, json={"stellenangebote": [], "maxErgebnisse": 0})
    )

    async with ArbeitsagenturClient() as client:
        await client.search_jobs(
            query="C++ / C# Developer & KI",
            location="München - Zentrum",
            radius_km=100,
            arbeitszeit="vz,tz",
            page=3,
            size=50,
        )

    assert route.called
    req = route.calls.last.request
    assert req.url.params["was"] == "C++ / C# Developer & KI"
    assert req.url.params["wo"] == "München - Zentrum"
    assert req.url.params["umkreis"] == "100"
    assert req.url.params["arbeitszeit"] == "vz,tz"
    assert req.url.params["page"] == "3"
    assert req.url.params["size"] == "50"


@pytest.mark.asyncio
@respx.mock
async def test_arbeitsagentur_client_resilience_to_malformed_json_and_empty():
    """Verify client handles empty or corrupt JSON payloads without unhandled crashes."""
    respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(
            200, text="<html><body>502 Bad Gateway from Cloudflare</body></html>"
        )
    )

    async with ArbeitsagenturClient() as client:
        with pytest.raises(Exception):
            await client.search_jobs(query="Developer")


@pytest.mark.asyncio
@respx.mock
async def test_arbeitsagentur_client_handles_transient_502_504_retries():
    """Verify client retries on transient 502, 503, 504 errors and recovers on 200."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs")
    route.side_effect = [
        httpx.Response(502, text="Bad Gateway"),
        httpx.Response(504, text="Gateway Timeout"),
        httpx.Response(200, json={"stellenangebote": [{"refnr": "12345", "titel": "Recov Job"}]}),
    ]

    async with ArbeitsagenturClient(max_retries=3, backoff_factor=0.01) as client:
        jobs = await client.search_jobs(query="Developer")

    assert len(jobs) == 1
    assert jobs[0].ref_nr == "12345"
    assert route.call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_arbeitsagentur_client_handles_immediate_auth_failure():
    """Verify 403 Forbidden raises ArbeitsagenturAuthError without retry loops."""
    route = respx.get(f"{DEFAULT_BASE_URL}/pc/v6/jobs").mock(
        return_value=httpx.Response(403, text="Forbidden: Invalid API Key")
    )

    async with ArbeitsagenturClient(max_retries=3) as client:
        with pytest.raises(ArbeitsagenturAuthError) as exc_info:
            await client.search_jobs(query="Developer")
        assert exc_info.value.status_code == 403

    assert route.call_count == 1


# ===========================================================================
# 7. Challenger 1 Normalization Edge Case Regressions
# ===========================================================================


def test_capital_sharp_s_canonical_hash_equivalence():
    """Verify capital sharp S (ẞ U+1E9E) matches standard ß/ss and produces canonical hash equivalence."""
    t1 = JobDeduplicator.normalize_title("GROẞKUNDENBETREUER (M/W/D)")
    t2 = JobDeduplicator.normalize_title("Großkundenbetreuer (m/w/d)")
    t3 = JobDeduplicator.normalize_title("Grosskundenbetreuer (m/w/d)")
    assert t1 == "grosskundenbetreuer"
    assert t2 == "grosskundenbetreuer"
    assert t3 == "grosskundenbetreuer"

    h1 = JobDeduplicator.compute_canonical_hash(
        "GROẞKUNDENBETREUER (M/W/D)", "Bank AG", "Frankfurt"
    )
    h2 = JobDeduplicator.compute_canonical_hash(
        "Großkundenbetreuer (m/w/d)", "Bank AG", "Frankfurt"
    )
    h3 = JobDeduplicator.compute_canonical_hash(
        "Grosskundenbetreuer (m/w/d)", "Bank AG", "Frankfurt"
    )
    assert h1 == h2 == h3


def test_decomposed_nfd_unicode_umlaut_transliteration():
    """Verify decomposed NFD unicode (e.g. Ba\u0308cker) produces identical normalized title and hash as NFC."""
    decomposed_title = "Ba\u0308cker (m/w/d)"  # 'ä' as 'a' + U+0308
    precomposed_title = "Bäcker (m/w/d)"

    norm_decomposed = JobDeduplicator.normalize_title(decomposed_title)
    norm_precomposed = JobDeduplicator.normalize_title(precomposed_title)

    assert norm_decomposed == "baecker"
    assert norm_precomposed == "baecker"

    h_decomposed = JobDeduplicator.compute_canonical_hash(
        decomposed_title, "Handwerk GmbH", "München"
    )
    h_precomposed = JobDeduplicator.compute_canonical_hash(
        precomposed_title, "Handwerk GmbH", "München"
    )
    assert h_decomposed == h_precomposed


def test_unanchored_legal_form_and_brand_preservation():
    """Verify brand names with 'EV', 'Holding', etc. are NOT erroneously collapsed."""
    emp_ev = JobDeduplicator.normalize_employer("EV Solutions")
    emp_holding = JobDeduplicator.normalize_employer("Holding Solutions")
    emp_plain = JobDeduplicator.normalize_employer("Solutions GmbH")

    assert emp_ev == "ev solutions"
    assert emp_holding == "holding solutions"
    assert emp_plain == "solutions"

    h_ev = JobDeduplicator.compute_canonical_hash("Python Developer", "EV Solutions", "Berlin")
    h_holding = JobDeduplicator.compute_canonical_hash(
        "Python Developer", "Holding Solutions", "Berlin"
    )
    h_plain = JobDeduplicator.compute_canonical_hash("Python Developer", "Solutions GmbH", "Berlin")

    assert h_ev != h_holding
    assert h_ev != h_plain
    assert h_holding != h_plain


def test_additional_german_legal_suffixes():
    """Verify stripping of complex and modern German legal forms (gGmbH, e.K., gUG, GmbH & Co. KG aA)."""
    assert JobDeduplicator.normalize_employer("Musterfirma gGmbH") == "musterfirma"
    assert JobDeduplicator.normalize_employer("Schreiner e.K.") == "schreiner"
    assert JobDeduplicator.normalize_employer("Kaufmann e. Kfm.") == "kaufmann"
    assert JobDeduplicator.normalize_employer("Sozialwerk gUG") == "sozialwerk"
    assert JobDeduplicator.normalize_employer("Firma GmbH & Co. KG aA") == "firma"
    assert JobDeduplicator.normalize_employer("Klinikum gAG") == "klinikum"


def test_parenthetical_and_hyphenated_gender_markers():
    """Verify (in), /-in, and /-innen are cleanly stripped without trailing 'in' artifacts."""
    assert JobDeduplicator.normalize_title("Projektleiter(in)") == "projektleiter"
    assert JobDeduplicator.normalize_title("Entwickler/-in") == "entwickler"
    assert JobDeduplicator.normalize_title("Mitarbeiter/-innen") == "mitarbeiter"
    assert JobDeduplicator.normalize_title("Senior Entwickler(in) (m/w/d)") == "senior entwickler"
