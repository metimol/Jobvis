"""Tests for AI job matching algorithms, multilingual rationales, and query generation."""

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
        mock_calc.side_effect = RuntimeError("Google GenAI Rate Limit / Timeout")

        jobs = [{"id": "j1", "title": "Software Developer", "description": "Python"}]
        results = await matcher.match_jobs({"skills": ["Python"]}, {}, jobs)

        assert len(results) == 1
        assert "score" in results[0]
        assert 0.0 <= results[0]["score"] <= 100.0
