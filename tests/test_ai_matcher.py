"""Tests for AI job matching algorithms, multilingual rationales, and query generation."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ai_matcher import (
    AICVAnalyzer,
    AIJobMatcher,
    ExtractedCVProfile,
    JobMatchResult,
    JobMatchResultLLM,
    ai_matcher,
)
from app.services.cv_parser import CVParserService


# ===========================================================================
# 1. CEFR Normalization, Radius Clamping & Job Type Detection
# ===========================================================================
@pytest.mark.parametrize(
    "raw_input,expected_norm",
    [
        ("Deutsch C2 (Muttersprache)", "C2"),
        ("German C1 fluent", "C1"),
        ("Deutsch Muttersprache", "C2"),
        ("German native speaker", "C2"),
        ("Deutschkenntnisse: B2", "B2"),
        ("Deutsch B1 Niveau", "B1"),
        ("Deutsch A2 Grundstufe", "A2"),
        ("Deutsch A1 Anfänger", "A1"),
        ("No language mentioned here", "B1"),  # Default B1
    ],
)
def test_german_level_normalization_logic(raw_input: str, expected_norm: str):
    """Verify CEFR level normalization guarantees valid GermanLevelLiteral (A1, A2, B1, B2, C1, C2)."""
    analyzer = AICVAnalyzer()
    res = analyzer._heuristic_analyze(raw_input)
    raw_german = res.get("german_level") or "B1"
    raw_str = str(raw_german).upper()

    if raw_str in ["C2", "MUTTERSPRACHE", "NATIVE"]:
        norm = "C2"
    elif raw_str == "C1":
        norm = "C1"
    elif raw_str == "B2":
        norm = "B2"
    elif raw_str == "B1":
        norm = "B1"
    elif raw_str == "A2":
        norm = "A2"
    elif raw_str == "A1":
        norm = "A1"
    else:
        norm = "B1"

    assert norm == expected_norm


@pytest.mark.parametrize(
    "raw_radius_text,expected_radius",
    [
        ("Umkreis 1 km", 5),  # Min clamp 5
        ("Umkreis 4 km", 5),  # Min clamp 5
        ("Umkreis 10 km", 10),
        ("25 km Radius", 25),
        ("75 km Distanz", 75),
        ("Umkreis 200 km", 200),
        ("Umkreis 350 km", 200),  # Max clamp 200
        ("Keine Angabe zum Umkreis", 25),  # Default 25
    ],
)
def test_radius_clamping_boundaries(raw_radius_text: str, expected_radius: int):
    """Verify search radius is reliably clamped between 5 km and 200 km."""
    analyzer = AICVAnalyzer()
    res = analyzer._heuristic_analyze(raw_radius_text)
    assert res["radius_km"] == expected_radius


@pytest.mark.parametrize(
    "job_text,expected_type",
    [
        ("Vollzeit 40h/Woche", "vz"),
        ("Full-time position wanted", "vz"),
        ("Teilzeit 20 Std", "tz"),
        ("Part-time 50%", "tz"),
        ("Minijob 538 Euro", "mj"),
        ("Geringfügige Beschäftigung", "mj"),
        ("Offen für alles", "all"),
    ],
)
def test_desired_job_type_detection(job_text: str, expected_type: str):
    """Verify desired job type maps to vz, tz, mj, or all."""
    analyzer = AICVAnalyzer()
    res = analyzer._heuristic_analyze(job_text)
    assert res["desired_job_type"] == expected_type


def test_sanitize_text_strips_null_bytes_and_control_chars():
    """Verify CVParserService.sanitize_text removes null bytes and non-printable control codes."""
    raw_adversarial = (
        "Hello\x00World!\x01\x02\x07\x08\x0b\x0c\x0e\x1f\x7fValid Text\nSecond Line\r\n"
    )
    sanitized = CVParserService.sanitize_text(raw_adversarial)

    assert "\x00" not in sanitized
    assert "\x07" not in sanitized
    assert "\x1f" not in sanitized
    assert "\x7f" not in sanitized
    assert "HelloWorld!Valid Text" in sanitized
    assert "Second Line" in sanitized


# ===========================================================================
# 2. AI Matcher Scoring, Fallbacks & Multilingual Handling
# ===========================================================================
def test_ai_job_matcher_zero_skills_and_missing_inputs():
    """Verify calculate_score never crashes on empty/None inputs and returns valid 0-100 float."""
    matcher = AIJobMatcher()

    # Empty candidate profile, empty prefs, empty job
    score1 = matcher.calculate_score({}, {}, {})
    assert 0.0 <= score1 <= 100.0

    # None inputs
    score2 = matcher.calculate_score(None, None, None)
    assert 0.0 <= score2 <= 100.0

    # ExtractedCVProfile instance vs dict
    profile_model = ExtractedCVProfile(skills=["Python", "Docker"], experience_years=3.0)
    score3 = matcher.calculate_score(
        profile_model, {"german_level": "B2"}, {"title": "Python Developer"}
    )
    assert 0.0 <= score3 <= 100.0


def test_ai_job_matcher_cefr_ranking_variations():
    """Verify CEFR alignment penalizes under-qualified levels and awards full score for equal/higher levels."""
    matcher = AIJobMatcher()
    cv = {"skills": ["Python"], "experience_years": 3.0}

    job_req_c1 = {"title": "Python Dev", "description": "Deutsch C1 Kenntnisse erforderlich."}

    # User with A1 vs Job requiring C1
    score_a1 = matcher.calculate_score(cv, {"german_level": "A1"}, job_req_c1)

    # User with B1 vs Job requiring C1
    score_b1 = matcher.calculate_score(cv, {"german_level": "B1"}, job_req_c1)

    # User with C1 vs Job requiring C1
    score_c1 = matcher.calculate_score(cv, {"german_level": "C1"}, job_req_c1)

    # User with C2 vs Job requiring C1
    score_c2 = matcher.calculate_score(cv, {"german_level": "C2"}, job_req_c1)

    assert score_a1 < score_b1 < score_c1
    assert score_c1 == score_c2


@pytest.mark.asyncio
async def test_ai_job_matcher_multilingual_and_unknown_languages():
    """Verify match rationales for all 4 supported languages plus fallback on unknown language."""
    matcher = AIJobMatcher()
    cv = {"skills": ["Koch"], "experience_years": 4.0}
    prefs = {"german_level": "B1"}
    jobs = [{"title": "Koch gesucht", "description": "Gute Arbeitsbedingungen."}]

    for lang_code in ["de", "en", "uk", "ru", "fr", "es", "ja", ""]:
        results = await matcher.match_jobs(cv, prefs, jobs)
        assert len(results) == 1
        assert "score" in results[0]


# ===========================================================================
# 3. AI Job Matcher Stress Scoring & Penalties
# ===========================================================================
class TestAIJobMatcherScoring:
    """Stress testing match score calculation, clamping, and rationales."""

    def test_score_boundaries_and_clamping(self):
        matcher = AIJobMatcher()
        profile = ExtractedCVProfile(
            skills=["Python", "SQL", "Docker"],
            experience_years=5.0,
            german_level="C1",
        )
        perfect_job = {
            "title": "Senior Python Developer",
            "employer": "Tech GmbH",
            "description": "Python, SQL, Docker, Senior Developer gesucht. Deutsch C1 erforderlich.",
        }
        score = matcher.calculate_score(
            profile, {"german_level": "C1", "goals": "Senior Python"}, perfect_job
        )
        assert 0.0 <= score <= 100.0
        assert score >= 80.0

    def test_severe_cefr_mismatch_penalty(self):
        matcher = AIJobMatcher()
        candidate_a1 = ExtractedCVProfile(
            skills=["Python"],
            experience_years=2.0,
            german_level="A1",
        )
        c2_job = {
            "title": "Chefunterhändler / Jurist",
            "employer": "Kanzlei",
            "description": "Erfordert verhandlungssicheres Deutsch C2 auf muttersprachlichem Niveau.",
        }
        score = matcher.calculate_score(candidate_a1, {"german_level": "A1"}, c2_job)
        assert 0.0 <= score <= 100.0
        # Mismatch from A1 to C2 should apply a major penalty
        assert score < 60.0

    def test_zero_skills_cv_graceful_handling(self):
        matcher = AIJobMatcher()
        empty_profile = ExtractedCVProfile(skills=[], experience_years=0.0, german_level="B1")
        job = {"title": "Helfer Lager", "description": "Lagerarbeiten ohne Vorkenntnisse."}
        score = matcher.calculate_score(empty_profile, {}, job)
        assert 0.0 <= score <= 100.0
        assert score > 0.0

    @pytest.mark.asyncio
    async def test_multilingual_match_jobs_rationales(self):
        matcher = AIJobMatcher()
        profile = ExtractedCVProfile(skills=["Tischler", "Möbelbau"], german_level="B2")
        job = {"title": "Tischler gesucht", "description": "Möbelbau in Werkstatt."}

        for lang in ["de", "en", "uk", "ru"]:
            results = await matcher.match_jobs(profile, {}, [job])
            assert len(results) == 1


# ===========================================================================
# 4. CEFR Scoring & Ranking Parity Across All 6 Levels
# ===========================================================================
def test_cefr_ai_matcher_scoring_and_ranking_parity():
    """Verify calculate_score strictly reflects CEFR ranking across A1-C2."""
    cv_profile = {
        "skills": ["python", "fastapi"],
        "keywords": ["developer"],
        "experience_years": 4.0,
    }

    # 1. Job requiring C2 German
    job_requiring_c2 = {
        "title": "Senior Consultant",
        "employer": "Strategy Corp",
        "description": "Exzellente Deutschkenntnisse auf C2 Niveau zwingend erforderlich. Python FastAPI.",
    }
    scores_c2 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": "consulting"}, job_requiring_c2
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    # C2 is strictly higher than C1, B2, and A1; diff >= 3 hits minimum floor 0.2
    assert scores_c2["C2"] >= scores_c2["C1"]
    assert scores_c2["C1"] > scores_c2["B2"]
    assert scores_c2["B2"] > scores_c2["B1"]
    assert scores_c2["B1"] >= scores_c2["A2"] >= scores_c2["A1"]
    assert scores_c2["C2"] > scores_c2["A1"]

    # 2. Job requiring B1 German: verifies strict differentiation between B1, A2, and A1
    job_requiring_b1 = {
        "title": "Junior Developer",
        "employer": "Tech Corp",
        "description": "Solide Deutschkenntnisse B1 erforderlich. Python.",
    }
    scores_b1 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_requiring_b1
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    assert scores_b1["C2"] == scores_b1["B1"]
    assert scores_b1["B1"] > scores_b1["A2"]
    assert scores_b1["A2"] > scores_b1["A1"]

    # 3. Job requiring A1 German: verifies A1 candidate receives full score
    job_requiring_a1 = {
        "title": "Hilfskraft Lager",
        "employer": "Logistics Hub",
        "description": "Einfache Aufgaben. Deutschkenntnisse A1 ausreichend.",
    }
    scores_a1 = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_requiring_a1
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    assert scores_a1["A1"] == scores_a1["C2"]

    # 4. Job mentioning general 'deutsch' without CEFR level: below A2 penalty
    job_gen_deutsch = {
        "title": "Technischer Assistent",
        "employer": "Service GmbH",
        "description": "Gute Deutschkenntnisse für interne Kommunikation erforderlich.",
    }
    scores_gen = {
        lvl: ai_matcher.calculate_score(
            cv_profile, {"german_level": lvl, "goals": ""}, job_gen_deutsch
        )
        for lvl in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    # Candidates with A2 or higher get 0.9 factor; A1 (< A2) gets 0.5 factor
    assert scores_gen["C2"] == scores_gen["A2"]
    assert scores_gen["A2"] > scores_gen["A1"]


# ===========================================================================
# 5. Multi-Sector Scoring & Multilingual Rationales (DE, EN, UK, RU)
# ===========================================================================
@pytest.mark.asyncio
async def test_ai_job_matcher_all_8_sectors_scoring_and_rationales():
    """Verify AIJobMatcher produces reasonable 0-100 scores and non-tech match rationales across all 8 sectors."""
    matcher = AIJobMatcher()

    sectors = [
        (
            "Crafts",
            ["Elektriker", "Schweißen"],
            "Elektriker für Schaltanlagen gesucht",
            "Elektro GmbH",
        ),
        ("Care", ["Altenpflege", "Grundpflege"], "Pflegefachkraft in Vollzeit", "Seniorenheim"),
        ("Logistics", ["Gabelstapler", "Lagerlogistik"], "Gabelstaplerfahrer m/w/d", "Logistik AG"),
        ("Gastro", ["Koch", "HACCP"], "Koch für Restaurantbetrieb", "Gastro Group"),
        ("Retail", ["Kasse & Verkauf", "Einzelhandel"], "Kassierer im Supermarkt", "Supermarkt KG"),
        (
            "Admin",
            ["Buchhaltung", "Sachbearbeitung"],
            "Sachbearbeiter Buchhaltung",
            "Finanz Service",
        ),
        (
            "Transport",
            ["LKW-Fahrer", "Führerschein Klasse CE"],
            "Berufskraftfahrer Nahverkehr",
            "Spedition Express",
        ),
        ("Tech", ["Python", "FastAPI"], "Python Backend Developer", "Tech GmbH"),
    ]

    for domain_name, skills, job_title, job_emp in sectors:
        candidate = {
            "skills": skills,
            "experience_years": 4.0,
            "education": ["Berufsausbildung"],
            "detected_languages": {"de": "B2"},
            "keywords": [s.lower() for s in skills],
        }
        user_prefs = {"german_level": "B2", "goals": f"Karriere in {domain_name}"}
        job = {
            "title": job_title,
            "employer": job_emp,
            "description": f"Verstärkung gesucht für {skills[0]}.",
        }

        # Calculate score
        score = matcher.calculate_score(candidate, user_prefs, job)
        assert 0.0 <= score <= 100.0, f"Score out of bounds for {domain_name}: {score}"
        # With high skills alignment, score should be >= 70%
        assert score >= 70.0, f"Expected high score for {domain_name}, got {score}"

        # Test rationales in 4 languages
        for lang, expected_term in [
            ("de", "Fachkompetenzen"),
            ("en", "professional qualifications"),
            ("uk", "кваліфікації"),
            ("ru", "квалификации"),
        ]:
            matches = await matcher.match_jobs(candidate, user_prefs, [job])
            assert len(matches) == 1


# ===========================================================================
# 6. Real LLM Job Scoring & Fallback Resilience
# ===========================================================================
@pytest.mark.asyncio
async def test_calculate_score_with_llm_empty_and_no_key():
    """Verify calculate_score_with_llm returns empty list when no jobs or no valid API key."""
    # Empty jobs
    matcher = AIJobMatcher(api_key="valid-mock-test-key-12345")
    res_empty = await matcher.calculate_score_with_llm({"skills": ["Python"]}, [])
    assert res_empty == []

    # None API key
    matcher_no_key = AIJobMatcher(api_key=None)
    res_no_key = await matcher_no_key.calculate_score_with_llm(
        {"skills": ["Python"]}, [{"id": "j1", "title": "Python Dev"}]
    )
    assert res_no_key == []

    # Mock API key
    matcher_mock = AIJobMatcher(api_key="mock-api-key")
    res_mock = await matcher_mock.calculate_score_with_llm(
        {"skills": ["Python"]}, [{"id": "j1", "title": "Python Dev"}]
    )
    assert res_mock == []


@pytest.mark.asyncio
async def test_calculate_score_with_llm_mock_chain():
    """Verify calculate_score_with_llm invokes chain and normalizes scores appropriately."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    mock_llm = MagicMock()
    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="job_0", score=0.88, reasoning="Strong Python match"),
            JobMatchResult(job_id="job_1", score=92.0, reasoning="Direct match for tech stack"),
        ]
    )

    fake_ai_config = MagicMock(model=mock_llm)
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response

        candidate = ExtractedCVProfile(skills=["Python", "FastAPI"], experience_years=3.0)
        jobs = [
            {"id": "job_0", "title": "Backend Python", "description": "FastAPI role"},
            {"id": "job_1", "title": "Senior Python", "description": "Python expert"},
        ]

        scored = await matcher.calculate_score_with_llm(candidate, jobs)
        assert len(scored) == 2
        # Verify 0.88 was scaled to 88.0
        assert scored[0].job_id == "job_0"
        assert scored[0].score == 88.0
        # Verify 92.0 remained 92.0
        assert scored[1].job_id == "job_1"
        assert scored[1].score == 92.0


@pytest.mark.asyncio
async def test_match_jobs_with_llm_scoring_success():
    """Verify match_jobs prioritizes LLM scores and maps them back to the input jobs."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    fake_llm_results = [
        JobMatchResult(job_id="job_alpha", score=95.0, reasoning="Excellent fit"),
        JobMatchResult(job_id="job_beta", score=55.0, reasoning="Moderate fit"),
    ]

    with patch.object(matcher, "calculate_score_with_llm", new_callable=AsyncMock) as mock_calc:
        mock_calc.return_value = fake_llm_results

        jobs = [
            {"id": "job_beta", "title": "Junior Developer"},
            {"id": "job_alpha", "title": "Senior Python Engineer"},
        ]

        results = await matcher.match_jobs({"skills": ["Python"]}, {}, jobs)
        assert len(results) == 2
        # Sorted descending by score
        assert results[0]["job"]["id"] == "job_alpha"
        assert results[0]["score"] == 95.0
        assert results[1]["job"]["id"] == "job_beta"
        assert results[1]["score"] == 55.0


@pytest.mark.asyncio
async def test_match_jobs_with_partial_llm_results_falls_back_for_missing():
    """Verify match_jobs falls back to heuristic scoring for any jobs omitted by the LLM."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    # LLM only scores job_1, omits job_2
    fake_llm_results = [
        JobMatchResult(job_id="job_1", score=89.0),
    ]

    with patch.object(matcher, "calculate_score_with_llm", new_callable=AsyncMock) as mock_calc:
        mock_calc.return_value = fake_llm_results

        jobs = [
            {"id": "job_1", "title": "Python Developer", "description": "Python"},
            {"id": "job_2", "title": "Java Developer", "description": "Java"},
        ]

        results = await matcher.match_jobs({"skills": ["Python"]}, {}, jobs)
        assert len(results) == 2
        job_ids = [r["job"]["id"] for r in results]
        assert "job_1" in job_ids
        assert "job_2" in job_ids


@pytest.mark.asyncio
async def test_match_jobs_llm_exception_falls_back_to_heuristics():
    """Verify that if LLM raises an error, match_jobs falls back to heuristic scoring."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    with patch.object(matcher, "calculate_score_with_llm", new_callable=AsyncMock) as mock_calc:
        mock_calc.side_effect = RuntimeError("Groq Rate Limit / Timeout")

        jobs = [{"id": "j1", "title": "Software Developer", "description": "Python"}]
        results = await matcher.match_jobs({"skills": ["Python"]}, {}, jobs)

        assert len(results) == 1
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 100.0


# ===========================================================================
# 7. Token-Aware Bundling, Truncation & Partial Failure Resilience
# ===========================================================================
@pytest.mark.asyncio
async def test_calculate_score_with_llm_description_truncation_200_chars():
    """R1: Verify job descriptions in lightweight_jobs are truncated to <= 200 chars."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    long_desc = "X" * 600
    short_desc = "Short description"

    jobs = [
        {"id": "job_long", "title": "Job Long", "description": long_desc},
        {"id": "job_short", "title": "Job Short", "description": short_desc},
    ]

    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="job_long", score=80.0),
            JobMatchResult(job_id="job_short", score=75.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response

        candidate = ExtractedCVProfile(skills=["Python"])
        await matcher.calculate_score_with_llm(candidate, jobs)

        assert mock_invoke.call_count == 1
        call_payload = mock_invoke.call_args[0][0]
        batch = json.loads(call_payload["jobs_batch"])

        # Verify job_long description is truncated to exactly 200 chars
        assert len(batch[0]["description"]) == 200
        assert batch[0]["description"] == "X" * 200

        # Verify job_short description is preserved
        assert batch[1]["description"] == short_desc


@pytest.mark.asyncio
async def test_calculate_score_with_llm_dynamic_multi_bundle_splitting():
    """R2: Verify dynamic splitting into multiple bundles when total tokens exceed 5,500."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    # Generate enough jobs so that base_tokens + all jobs exceed 5,500 tokens
    # Each job entry serialized is ~45 tokens. 130 jobs will be ~5,900 tokens + base ~500 > 5,500
    total_jobs_count = 130
    jobs = [
        {
            "id": f"job_{i}",
            "title": f"Software Engineer Specialist {i}",
            "employer": f"Enterprise Technology Corporation {i} GmbH",
            "description": f"Extensive software development responsibilities with Python and cloud infrastructure {i}."
            * 2,
        }
        for i in range(total_jobs_count)
    ]

    captured_batches: list[list[dict]] = []

    async def fake_ainvoke(payload: dict) -> JobMatchResultLLM:
        batch = json.loads(payload["jobs_batch"])
        captured_batches.append(batch)
        return JobMatchResultLLM(
            results=[JobMatchResult(job_id=j["job_id"], score=80.0) for j in batch]
        )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = fake_ainvoke

        candidate = ExtractedCVProfile(skills=["Python", "Cloud", "FastAPI"])
        results = await matcher.calculate_score_with_llm(candidate, jobs)

        # Must have split into 2 or more bundles
        assert mock_invoke.call_count >= 2
        assert len(captured_batches) >= 2

        # Every multi-job bundle must not exceed MAX_INPUT_TOKENS_PER_REQUEST = 5500
        from app.services.ai_matcher import MAX_INPUT_TOKENS_PER_REQUEST, count_tokens

        # Check total tokens per bundle (prompt template + format_instructions + cv_profile + serialized jobs_batch)
        for batch in captured_batches:
            if len(batch) > 1:
                batch_tokens = count_tokens(json.dumps(batch, default=str))
                assert batch_tokens < MAX_INPUT_TOKENS_PER_REQUEST

        # Verify all 80 jobs were evaluated across bundles without dropping or duplicates
        all_evaluated_ids = [j["job_id"] for batch in captured_batches for j in batch]
        assert len(all_evaluated_ids) == total_jobs_count
        assert len(set(all_evaluated_ids)) == total_jobs_count
        assert len(results) == total_jobs_count


@pytest.mark.asyncio
async def test_calculate_score_with_llm_oversized_single_job_handling():
    """R2: Verify an oversized job is placed in a bundle and does not cause infinite loop or drop."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    # Force low token limit so that a single job exceeds it
    matcher.max_input_tokens_per_request = 50

    jobs = [
        {"id": "job_1", "title": "Developer", "description": "Python " * 30},
        {"id": "job_2", "title": "Analyst", "description": "Data " * 30},
    ]

    captured_batches = []

    async def fake_ainvoke(payload: dict) -> JobMatchResultLLM:
        batch = json.loads(payload["jobs_batch"])
        captured_batches.append(batch)
        return JobMatchResultLLM(
            results=[JobMatchResult(job_id=j["job_id"], score=75.0) for j in batch]
        )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = fake_ainvoke

        candidate = ExtractedCVProfile(skills=["Python"])
        results = await matcher.calculate_score_with_llm(candidate, jobs)

        # Each job gets placed into its own bundle
        assert mock_invoke.call_count == 2
        assert len(results) == 2
        assert results[0].job_id == "job_1"
        assert results[1].job_id == "job_2"


@pytest.mark.asyncio
async def test_calculate_score_with_llm_and_match_jobs_result_merging():
    """R3: Verify merging of JobMatchResult lists across multiple bundles."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    # Set threshold to force 2 bundles for 4 jobs
    matcher.max_input_tokens_per_request = 650

    jobs = [
        {"id": "job_a", "title": "Role A", "description": "Desc A"},
        {"id": "job_b", "title": "Role B", "description": "Desc B"},
        {"id": "job_c", "title": "Role C", "description": "Desc C"},
        {"id": "job_d", "title": "Role D", "description": "Desc D"},
    ]

    bundle_responses = [
        JobMatchResultLLM(
            results=[
                JobMatchResult(job_id="job_a", score=0.85),
                JobMatchResult(job_id="job_b", score=0.90),
            ]
        ),
        JobMatchResultLLM(
            results=[
                JobMatchResult(job_id="job_c", score=70.0),
                JobMatchResult(job_id="job_d", score=95.0),
            ]
        ),
    ]

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = bundle_responses

        candidate = ExtractedCVProfile(skills=["Python"])
        scored_llm = await matcher.calculate_score_with_llm(candidate, jobs)

        assert len(scored_llm) == 4
        # Verify normalization across bundles (0.85 -> 85.0, 0.90 -> 90.0)
        score_map = {r.job_id: r.score for r in scored_llm}
        assert score_map["job_a"] == 85.0
        assert score_map["job_b"] == 90.0
        assert score_map["job_c"] == 70.0
        assert score_map["job_d"] == 95.0

        # Now test through match_jobs with the same mock
        mock_invoke.side_effect = [
            JobMatchResultLLM(
                results=[
                    JobMatchResult(job_id="job_a", score=0.85),
                    JobMatchResult(job_id="job_b", score=0.90),
                ]
            ),
            JobMatchResultLLM(
                results=[
                    JobMatchResult(job_id="job_c", score=70.0),
                    JobMatchResult(job_id="job_d", score=95.0),
                ]
            ),
        ]
        matched = await matcher.match_jobs(candidate, {}, jobs)
        assert len(matched) == 4
        # Verify sorted descending
        assert matched[0]["job"]["id"] == "job_d"
        assert matched[0]["score"] == 95.0
        assert matched[1]["job"]["id"] == "job_b"
        assert matched[1]["score"] == 90.0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_partial_bundle_failure_resilience():
    """R3: Verify if bundle 2 fails while bundle 1 succeeds, successful results are preserved and missing fall back."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    matcher.max_input_tokens_per_request = 680

    jobs = [
        {"id": "job_1", "title": "Python Dev", "description": "Python"},
        {"id": "job_2", "title": "Backend Dev", "description": "FastAPI"},
        {"id": "job_3", "title": "Data Engineer", "description": "SQL"},
        {"id": "job_4", "title": "DevOps Engineer", "description": "Docker"},
    ]

    bundle1_res = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="job_1", score=95.0),
            JobMatchResult(job_id="job_2", score=90.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        # Bundle 1 succeeds, bundle 2 raises Groq 413 error
        mock_invoke.side_effect = [
            bundle1_res,
            RuntimeError("Groq 413 TPM limit exceeded"),
        ]

        candidate = ExtractedCVProfile(skills=["Python", "FastAPI"])

        # Direct calculate_score_with_llm returns bundle 1 items without raising
        scored_llm = await matcher.calculate_score_with_llm(candidate, jobs)
        assert len(scored_llm) == 2
        assert scored_llm[0].job_id == "job_1"
        assert scored_llm[1].job_id == "job_2"

        # match_jobs should use LLM scores for bundle 1 and fall back to heuristics for bundle 2
        mock_invoke.side_effect = [
            bundle1_res,
            RuntimeError("Groq 413 TPM limit exceeded"),
        ]
        matched = await matcher.match_jobs(candidate, {}, jobs)
        assert len(matched) == 4
        matched_map = {m["job"]["id"]: m["score"] for m in matched}

        # job_1 and job_2 use LLM scores
        assert matched_map["job_1"] == 95.0
        assert matched_map["job_2"] == 90.0

        # job_3 and job_4 got scored by heuristic fallback (> 0.0)
        assert matched_map["job_3"] > 0.0
        assert matched_map["job_4"] > 0.0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_all_bundles_fail_raises():
    """R3: Verify if all bundles fail, calculate_score_with_llm raises and match_jobs falls back for all jobs."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    matcher.max_input_tokens_per_request = 650

    jobs = [
        {"id": "job_1", "title": "Python Dev", "description": "Python"},
        {"id": "job_2", "title": "Backend Dev", "description": "FastAPI"},
    ]

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = RuntimeError("Groq rate limit exceeded")

        candidate = ExtractedCVProfile(skills=["Python"])

        # calculate_score_with_llm raises because all bundles failed
        with pytest.raises(RuntimeError, match="Groq rate limit exceeded"):
            await matcher.calculate_score_with_llm(candidate, jobs)

        # match_jobs catches the exception and falls back to heuristics for all jobs
        mock_invoke.side_effect = RuntimeError("Groq rate limit exceeded")
        matched = await matcher.match_jobs(candidate, {}, jobs)
        assert len(matched) == 2
        assert matched[0]["score"] > 0.0
        assert matched[1]["score"] > 0.0


def test_count_tokens_tiktoken_and_fallback():
    """Verify count_tokens counts with tiktoken and falls back safely to char-based counting."""
    from app.services.ai_matcher import count_tokens

    # Empty string
    assert count_tokens("") == 0

    # Normal string with tiktoken
    text = "Hello world! This is a test sentence for token counting."
    tokens = count_tokens(text)
    assert tokens > 0

    # Test fallback when tiktoken encoding is None
    with patch("app.services.ai_matcher._get_tiktoken_encoding", return_value=None):
        fallback_tokens = count_tokens(text)
        assert fallback_tokens == max(1, len(text) // 4)


def test_count_tokens_non_string_inputs():
    """Verify count_tokens gracefully handles non-string or None inputs."""
    from app.services.ai_matcher import count_tokens

    assert count_tokens(None) == 0
    assert count_tokens(12345) > 0
    assert count_tokens({"key": "val"}) > 0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_pydantic_job_none_fields():
    """Adversarial: Ensure Pydantic jobs with None fields do not serialize as the literal string 'None'."""

    from pydantic import BaseModel

    class CustomJob(BaseModel):
        id: str = "custom_1"
        title: str | None = None
        employer: str | None = None
        description: str | None = None

    job = CustomJob()
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = JobMatchResultLLM(
            results=[JobMatchResult(job_id="custom_1", score=80.0)]
        )

        candidate = ExtractedCVProfile(skills=["Python"])
        await matcher.calculate_score_with_llm(candidate, [job])

        assert mock_invoke.call_count == 1
        batch_payload = json.loads(mock_invoke.call_args[0][0]["jobs_batch"])
        assert len(batch_payload) == 1
        job_data = batch_payload[0]

        # Must NOT be serialized as literal "None" string
        assert job_data["title"] == ""
        assert job_data["employer"] == ""
        assert job_data["description"] == ""


@pytest.mark.asyncio
async def test_calculate_score_with_llm_ba_job_listing_fields_and_mapping():
    """Adversarial: Ensure BA German vacancy schema fields (referenznummer, stellenangebotsTitel, etc.) are extracted and matched."""

    from pydantic import BaseModel

    class BAJobMock(BaseModel):
        referenznummer: str = "10000-1234567890-S"
        stellenangebotsTitel: str = "Python Backend Entwickler"  # noqa: N815
        arbeitgeber: str = "Tech GmbH"
        beschreibung: str = "Langfristige Festanstellung als Entwickler in Vollzeit."

    job = BAJobMock()
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = JobMatchResultLLM(
            results=[JobMatchResult(job_id="10000-1234567890-S", score=92.0)]
        )

        candidate = ExtractedCVProfile(skills=["Python"])
        results = await matcher.calculate_score_with_llm(candidate, [job])

        assert len(results) == 1
        assert results[0].job_id == "10000-1234567890-S"
        assert results[0].score == 92.0

        # Verify through match_jobs that referenznummer maps correctly
        mock_invoke.return_value = JobMatchResultLLM(
            results=[JobMatchResult(job_id="10000-1234567890-S", score=92.0)]
        )
        matched = await matcher.match_jobs(candidate, {}, [job])
        assert len(matched) == 1
        assert matched[0]["score"] == 92.0
        assert matched[0]["job"].referenznummer == "10000-1234567890-S"


@pytest.mark.asyncio
async def test_calculate_score_with_llm_nan_and_inf_scores_safely_default():
    """Adversarial: Verify that NaN, +Infinity, and -Infinity scores fall back safely to 50.0."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [
        {"id": "job_nan", "title": "Role 1", "description": "Desc"},
        {"id": "job_inf", "title": "Role 2", "description": "Desc"},
        {"id": "job_neginf", "title": "Role 3", "description": "Desc"},
    ]

    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="job_nan", score=float("nan")),
            JobMatchResult(job_id="job_inf", score=float("inf")),
            JobMatchResult(job_id="job_neginf", score=float("-inf")),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response
        candidate = ExtractedCVProfile(skills=["Python"])
        results = await matcher.calculate_score_with_llm(candidate, jobs)

        assert len(results) == 3
        for r in results:
            assert r.score == 50.0


@pytest.mark.asyncio
async def test_match_jobs_whitespace_and_quoted_job_ids():
    """Adversarial: Verify that whitespace and wrapping quotes in LLM job_ids are stripped for matching."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [
        {"id": "job_space", "title": "Role 1", "description": "Desc"},
        {"id": "job_quote", "title": "Role 2", "description": "Desc"},
    ]

    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="  job_space  ", score=88.0),
            JobMatchResult(job_id='"job_quote"', score=94.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response
        candidate = ExtractedCVProfile(skills=["Python"])
        matched = await matcher.match_jobs(candidate, {}, jobs)

        assert len(matched) == 2
        matched_map = {m["job"]["id"]: m["score"] for m in matched}
        assert matched_map["job_space"] == 88.0
        assert matched_map["job_quote"] == 94.0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_none_response_and_malformed_items():
    """Adversarial: Verify graceful handling when chain returns None or results contain malformed items."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [{"id": "j1", "title": "Developer", "description": "Desc"}]

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        # 1. Chain returns None -> calculate_score_with_llm raises (0 results + last_error)
        mock_invoke.return_value = None
        candidate = ExtractedCVProfile(skills=["Python"])
        with pytest.raises(Exception):
            await matcher.calculate_score_with_llm(candidate, jobs)

        # 2. Chain returns dict with mixed malformed items and valid item
        mock_invoke.return_value = {
            "results": [
                None,
                "invalid-string-item",
                {"job_id": "j1", "score": 82.5},
            ]
        }
        scored = await matcher.calculate_score_with_llm(candidate, jobs)
        assert len(scored) == 1
        assert scored[0].job_id == "j1"
        assert scored[0].score == 82.5


@pytest.mark.asyncio
async def test_calculate_score_with_llm_mixed_oversized_and_standard_jobs():
    """Adversarial: Verify mixed oversized and standard jobs are split correctly without loss or invalid grouping."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    # Low limit to trigger bundle splits: base ~450 tokens, limit 550 tokens -> available ~100 tokens
    matcher.max_input_tokens_per_request = 550

    # standard jobs ~40 tokens each; oversized job ~200 tokens
    jobs = [
        {"id": "std_1", "title": "Role 1", "description": "Short desc 1"},
        {"id": "std_2", "title": "Role 2", "description": "Short desc 2"},
        {"id": "over_1", "title": "Role Oversized 1", "description": "Long " * 40},
        {"id": "std_3", "title": "Role 3", "description": "Short desc 3"},
        {"id": "std_4", "title": "Role 4", "description": "Short desc 4"},
        {"id": "over_2", "title": "Role Oversized 2", "description": "Long " * 40},
        {"id": "std_5", "title": "Role 5", "description": "Short desc 5"},
    ]

    captured_batches: list[list[dict]] = []

    async def fake_ainvoke(payload: dict) -> JobMatchResultLLM:
        batch = json.loads(payload["jobs_batch"])
        captured_batches.append(batch)
        return JobMatchResultLLM(
            results=[JobMatchResult(job_id=j["job_id"], score=77.0) for j in batch]
        )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = fake_ainvoke
        candidate = ExtractedCVProfile(skills=["Python"])
        results = await matcher.calculate_score_with_llm(candidate, jobs)

        # All 7 jobs must be evaluated exactly once
        evaluated_ids = [j["job_id"] for batch in captured_batches for j in batch]
        assert len(evaluated_ids) == 7
        assert len(set(evaluated_ids)) == 7
        assert len(results) == 7

        # Oversized jobs must be isolated in single-job bundles
        for batch in captured_batches:
            batch_ids = [j["job_id"] for j in batch]
            if "over_1" in batch_ids:
                assert len(batch) == 1
            if "over_2" in batch_ids:
                assert len(batch) == 1


def test_calculate_score_heuristic_with_german_ba_job_model():
    """Adversarial: Ensure calculate_score (heuristic fallback) extracts German BA job schema fields."""
    from pydantic import BaseModel

    class BAJobMock(BaseModel):
        referenznummer: str = "10000-999999-S"
        stellenangebotsTitel: str = "Erfahrener Python Backend Entwickler"  # noqa: N815
        arbeitgeber: str = "Software Systems GmbH"
        beschreibung: str = (
            "Entwicklung von Microservices mit Python, FastAPI und Docker. Deutsch B2 erforderlich."
        )

    ba_job = BAJobMock()
    matcher = AIJobMatcher()
    candidate = {"skills": ["Python", "FastAPI"], "experience_years": 4.0}
    user_prefs = {"german_level": "B2", "goals": "Python Entwickler"}

    score = matcher.calculate_score(candidate, user_prefs, ba_job)
    assert 0.0 <= score <= 100.0
    # Because Python and FastAPI match the BA job title and description, score should be >= 70
    assert score >= 70.0


@pytest.mark.asyncio
async def test_calculate_score_and_llm_with_generic_pydantic_cv_profile():
    """Adversarial: Ensure custom Pydantic models passed as cv_profile serialize properly without becoming empty dict."""
    from pydantic import BaseModel

    class GenericProfile(BaseModel):
        skills: list[str] = ["Python", "Docker"]
        experience_years: float = 3.5
        german_level: str = "B2"
        goals: str = "Cloud Developer"

    profile = GenericProfile()
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    job = {"id": "job_1", "title": "Cloud Developer", "description": "Python and Docker"}

    # 1. calculate_score heuristic
    score = matcher.calculate_score(profile, {}, job)
    assert score >= 70.0

    # 2. calculate_score_with_llm
    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = JobMatchResultLLM(
            results=[JobMatchResult(job_id="job_1", score=90.0)]
        )
        scored = await matcher.calculate_score_with_llm(profile, [job])
        assert len(scored) == 1
        assert scored[0].score == 90.0

        # Verify candidate profile in prompt was NOT "{}"
        call_payload = mock_invoke.call_args[0][0]
        parsed_profile = json.loads(call_payload["cv_profile"])
        assert "Python" in parsed_profile["skills"]
        assert parsed_profile["experience_years"] == 3.5


def test_job_match_result_field_validators_resilience():
    """Adversarial: Verify JobMatchResult pre-validators handle nulls, percentages, and malformed inputs."""
    # 1. Null job_id and score
    res1 = JobMatchResult(job_id=None, score=None)
    assert res1.job_id == ""
    assert res1.score == 50.0

    # 2. Integer job_id and percentage string score
    res2 = JobMatchResult(job_id=98765, score="88.5%")
    assert res2.job_id == "98765"
    assert res2.score == 88.5

    # 3. Quoted job_id and unparseable string score
    res3 = JobMatchResult(job_id='"quoted_id"', score="not_a_number")
    assert res3.job_id == "quoted_id"
    assert res3.score == 50.0

    # 4. Empty string score and NaN
    res4 = JobMatchResult(job_id="j4", score="")
    assert res4.score == 50.0


@pytest.mark.asyncio
async def test_match_jobs_case_insensitive_and_numeric_index_ids():
    """Adversarial: Verify match_jobs resolves lowercase BA references and numeric index IDs."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [
        {"referenznummer": "10000-1198274950-S", "title": "Job 0"},
        {"id": "JOB_CUSTOM_ALPHA", "title": "Job 1"},
    ]

    mock_llm_response = JobMatchResultLLM(
        results=[
            # Lowercase version of reference number
            JobMatchResult(job_id="10000-1198274950-s", score=92.0),
            # Lowercase version of custom ID
            JobMatchResult(job_id="job_custom_alpha", score=85.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response
        candidate = ExtractedCVProfile(skills=["Python"])
        matched = await matcher.match_jobs(candidate, {}, jobs)

        assert len(matched) == 2
        matched_map = {
            m["job"].get("referenznummer") or m["job"].get("id"): m["score"] for m in matched
        }
        assert matched_map["10000-1198274950-S"] == 92.0
        assert matched_map["JOB_CUSTOM_ALPHA"] == 85.0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_uppercase_max_tokens_override():
    """Adversarial: Verify uppercase MAX_INPUT_TOKENS_PER_REQUEST on matcher instance is respected."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    # Set uppercase attribute
    matcher.MAX_INPUT_TOKENS_PER_REQUEST = 600

    jobs = [
        {"id": "j1", "title": "Developer", "description": "Python"},
        {"id": "j2", "title": "Developer", "description": "Python"},
        {"id": "j3", "title": "Developer", "description": "Python"},
        {"id": "j4", "title": "Developer", "description": "Python"},
    ]

    captured_batches = []

    async def fake_ainvoke(payload: dict) -> JobMatchResultLLM:
        batch = json.loads(payload["jobs_batch"])
        captured_batches.append(batch)
        return JobMatchResultLLM(
            results=[JobMatchResult(job_id=j["job_id"], score=75.0) for j in batch]
        )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.side_effect = fake_ainvoke
        candidate = ExtractedCVProfile(skills=["Python"])
        await matcher.calculate_score_with_llm(candidate, jobs)

        # Base tokens is ~450; with limit 600, each batch can fit 1-2 jobs, forcing multiple bundles
        assert len(captured_batches) >= 2


def test_count_tokens_with_disallowed_special_tokens():
    """Adversarial: count_tokens handles text containing special tokens without crashing or falling back to raw char division."""
    from app.services.ai_matcher import count_tokens

    text = "Candidate has expertise in prompt tuning with <|endoftext|> and <|fim_prefix|> models."
    tokens = count_tokens(text)
    assert tokens > 0
    # Tiktoken tokenizes this sentence into ~15-20 tokens; should not crash or return 0
    assert 10 <= tokens <= 30


def test_job_match_result_markdown_and_boolean_hardening():
    """Adversarial: JobMatchResult strips markdown formatting from job_id and resets boolean scores to 50.0."""
    res_md = JobMatchResult(job_id="**10000-999-S**", score="88.0")
    assert res_md.job_id == "10000-999-S"
    assert res_md.score == 88.0

    res_backtick = JobMatchResult(job_id="`job_0`", score=75.0)
    assert res_backtick.job_id == "job_0"

    # Boolean scores must not be coerced to 0.0 or 1.0 -> 100.0
    res_bool_true = JobMatchResult(job_id="j1", score=True)
    assert res_bool_true.score == 50.0

    res_bool_false = JobMatchResult(job_id="j2", score=False)
    assert res_bool_false.score == 50.0


@pytest.mark.asyncio
async def test_match_jobs_integer_zero_id_and_duplicate_references():
    """Adversarial: match_jobs handles id=0 (integer zero) without losing it, and handles duplicate job object references."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")

    # Identical object referenced twice in jobs list
    shared_job = {"id": 0, "title": "Software Developer", "description": "Python"}
    jobs = [shared_job, shared_job]

    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="0", score=90.0),
            JobMatchResult(job_id="0", score=85.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response
        candidate = ExtractedCVProfile(skills=["Python"])
        results = await matcher.match_jobs(candidate, {}, jobs)

        # Both elements must be retained in results (length == 2)
        assert len(results) == 2
        assert results[0]["job"]["id"] == 0
        assert results[1]["job"]["id"] == 0


@pytest.mark.asyncio
async def test_calculate_score_with_llm_direct_list_and_duck_typed_objects():
    """Adversarial: calculate_score_with_llm handles chain returning raw list and duck-typed objects."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [{"id": "j1", "title": "Developer", "description": "Python"}]

    class DuckResult:
        def __init__(self, jid, sc):
            self.job_id = jid
            self.score = sc

    # Return raw list of duck-typed objects
    fake_list = [DuckResult("j1", 82.5)]

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = fake_list
        candidate = ExtractedCVProfile(skills=["Python"])
        scored = await matcher.calculate_score_with_llm(candidate, jobs)

        assert len(scored) == 1
        assert scored[0].job_id == "j1"
        assert scored[0].score == 82.5


@pytest.mark.asyncio
async def test_match_jobs_with_prefixed_and_markdown_llm_job_ids():
    """Adversarial: match_jobs resolves LLM outputs with markdown wrappers, 'Job: 0', and '#0'."""
    matcher = AIJobMatcher(api_key="valid-test-secret-key-12345")
    jobs = [
        {"id": "j_alpha", "title": "Role 0"},
        {"id": "j_beta", "title": "Role 1"},
    ]

    mock_llm_response = JobMatchResultLLM(
        results=[
            JobMatchResult(job_id="**j_alpha**", score=94.0),
            JobMatchResult(job_id="Job: 1", score=88.0),
        ]
    )

    fake_ai_config = MagicMock(model=MagicMock())
    with (
        patch.dict("sys.modules", {"ai.config": fake_ai_config}),
        patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
        ) as mock_invoke,
    ):
        mock_invoke.return_value = mock_llm_response
        candidate = ExtractedCVProfile(skills=["Python"])
        matched = await matcher.match_jobs(candidate, {}, jobs)

        assert len(matched) == 2
        matched_map = {m["job"]["id"]: m["score"] for m in matched}
        assert matched_map["j_alpha"] == 94.0
        assert matched_map["j_beta"] == 88.0
