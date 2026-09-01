"""AI Job Matching and CV Analysis Service using LangChain Google GenAI and Heuristics."""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

# CEFR Language Ranking for comparison
CEFR_LEVELS = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6,
}


class ExtractedCVProfile(BaseModel):
    """Structured candidate profile extracted from CV text."""

    skills: list[str] = Field(default_factory=list)
    experience_years: float = Field(default=0.0)
    education: list[str | dict[str, Any]] = Field(default_factory=list)
    detected_languages: dict[str, str] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None


class JobMatchResult(BaseModel):
    """Result of AI match scoring for a candidate against a job vacancy."""

    job: Any
    score: float
    match_reason: str
    factors: dict[str, float] = Field(default_factory=dict)


class AICVAnalyzer:
    """Extracts structured skills, experience, education, and language levels from CVs."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY

    async def analyze_cv(self, cv_text: str) -> dict[str, Any]:
        """Analyze CV text using Google GenAI or robust heuristic fallback."""
        if not cv_text or not cv_text.strip():
            return {
                "skills": [],
                "experience_years": 0.0,
                "education": [],
                "detected_languages": {},
                "keywords": [],
            }

        # If API key configured, attempt LangChain Google GenAI structured extraction
        if self.api_key and not self.api_key.startswith("mock-") and len(self.api_key) > 10:
            try:
                from langchain_core.output_parsers import PydanticOutputParser
                from langchain_core.prompts import PromptTemplate
                from langchain_google_genai import ChatGoogleGenerativeAI

                parser = PydanticOutputParser(pydantic_object=ExtractedCVProfile)
                prompt = PromptTemplate(
                    template="Extract candidate CV details into JSON format.\n{format_instructions}\n\nCV Text:\n{cv_text}\n",
                    input_variables=["cv_text"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    google_api_key=self.api_key,
                    temperature=0.1,
                )
                chain = prompt | llm | parser
                res: ExtractedCVProfile = await chain.ainvoke({"cv_text": cv_text[:8000]})
                return res.model_dump()
            except Exception as e:
                logger.warning(
                    "Google GenAI extraction failed (%s), falling back to heuristics.", e
                )

        # Robust Heuristic Fallback Analysis
        return self._heuristic_analyze(cv_text)

    def _heuristic_analyze(self, cv_text: str) -> dict[str, Any]:
        """Deterministic heuristic extractor supporting DE, EN, UK, RU CVs."""
        text_lower = cv_text.lower()
        skills: list[str] = []

        # Technical & Engineering Skills
        tech_catalog = {
            "python": "Python",
            "fastapi": "FastAPI",
            "django": "Django",
            "flask": "Flask",
            "react": "React",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "node": "Node.js",
            "docker": "Docker",
            "kubernetes": "Kubernetes",
            "postgresql": "PostgreSQL",
            "sql": "SQL",
            "mysql": "MySQL",
            "aws": "AWS",
            "azure": "Azure",
            "git": "Git",
            "c++": "C++",
            "c#": "C#",
            "java": "Java",
            "sps": "SPS-Programmierung",
            "sps-programmierung": "SPS-Programmierung",
            "schaltanlagenbau": "Schaltanlagenbau",
            "industrieautomation": "Industrieautomation",
            "elektroniker": "Elektroniker",
            "elektrotechnik": "Elektrotechnik",
            "grundpflege": "Grundpflege",
            "altenpflege": "Altenpflege",
            "medikamentenverabreichung": "Medikamentenverabreichung",
            "wundversorgung": "Wundversorgung",
            "buchhaltung": "Buchhaltung",
            "rechnungswesen": "Rechnungswesen",
            "kundenbetreuung": "Kundenbetreuung",
            "projektmanagement": "Projektmanagement",
            "lagerlogistik": "Lagerlogistik",
            "staplerschein": "Gabelstapler",
        }

        for keyword, canonical in tech_catalog.items():
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                if canonical not in skills:
                    skills.append(canonical)

        # Experience calculation
        experience_years = 0.0
        exp_match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(?:jahre?|years?|yrs?|р[оі]к(?:ів)?|лет|года)\s*(?:berufserfahrung|erfahrung|experience|досв[іі]ду|опыта)?",
            text_lower,
        )
        if exp_match:
            try:
                val = float(exp_match.group(1).replace(",", "."))
                experience_years = min(val, 50.0)
            except ValueError:
                experience_years = 3.0
        elif "senior" in text_lower or "lead" in text_lower:
            experience_years = 5.0
        elif "junior" in text_lower or "werkstudent" in text_lower:
            experience_years = 1.0
        elif skills:
            experience_years = 3.0

        # Education
        education = []
        if "master" in text_lower or "m.sc." in text_lower or "magister" in text_lower:
            education.append("Master")
        elif "bachelor" in text_lower or "b.sc." in text_lower:
            education.append("Bachelor")
        elif "promotion" in text_lower or "dr." in text_lower or "phd" in text_lower:
            education.append("Promotion / PhD")
        elif (
            "ausbildung" in text_lower
            or "berufsausbildung" in text_lower
            or "geselle" in text_lower
        ):
            education.append("Berufsausbildung")
        elif "abitur" in text_lower or "matura" in text_lower:
            education.append("Abitur / Fachhochschulreife")
        else:
            education.append("Berufsausbildung")

        # CEFR Language Detection
        detected_languages: dict[str, str] = {}

        # German level
        if re.search(
            r"\b(?:deutsch|german|німецька|немецкий)\b.*?\b(c2|c1|b2|b1|a2|a1)\b", text_lower
        ):
            m = re.search(
                r"\b(?:deutsch|german|німецька|немецкий)\b.*?\b(c2|c1|b2|b1|a2|a1)\b", text_lower
            )
            detected_languages["de"] = m.group(1).upper()
        elif "c1" in text_lower:
            detected_languages["de"] = "C1"
        elif "b2" in text_lower:
            detected_languages["de"] = "B2"
        elif "a2" in text_lower:
            detected_languages["de"] = "A2"
        else:
            detected_languages["de"] = "B1"

        # English level
        if re.search(
            r"\b(?:englisch|english|англійська|английский)\b.*?\b(c2|c1|b2|b1|a2|a1)\b", text_lower
        ):
            m = re.search(
                r"\b(?:englisch|english|англійська|английский)\b.*?\b(c2|c1|b2|b1|a2|a1)\b",
                text_lower,
            )
            detected_languages["en"] = m.group(1).upper()
        elif "fließend englisch" in text_lower or "fluent english" in text_lower:
            detected_languages["en"] = "C1"
        else:
            detected_languages["en"] = "B2"

        # Keywords
        keywords = [s.lower() for s in skills]

        return {
            "skills": skills,
            "experience_years": experience_years,
            "education": education,
            "detected_languages": detected_languages,
            "keywords": keywords,
        }


class AIJobMatcher:
    """Multi-factor AI job matching scoring and multilingual rationale generator."""

    def calculate_score(
        self,
        cv_profile: dict[str, Any] | ExtractedCVProfile,
        user_prefs: dict[str, Any] | Any,
        job: dict[str, Any] | Any,
    ) -> float:
        """Calculate a composite 0-100 match score across 4 weighted factors:

        - Skills alignment: 40%
        - Experience alignment: 25%
        - German CEFR language alignment: 20%
        - Career goals alignment: 15%
        """
        if isinstance(cv_profile, ExtractedCVProfile):
            profile_dict = cv_profile.model_dump()
        elif isinstance(cv_profile, dict):
            profile_dict = cv_profile
        else:
            profile_dict = {}

        # Extract user preferences
        if isinstance(user_prefs, dict):
            user_german = user_prefs.get("german_level", "B1")
            user_goals = user_prefs.get("goals", "")
        else:
            user_german = getattr(user_prefs, "german_level", "B1") or "B1"
            user_goals = getattr(user_prefs, "goals", "") or ""

        # Extract job text
        if isinstance(job, dict):
            job_title = job.get("title") or job.get("titel") or ""
            job_desc = job.get("description") or job.get("beschreibung") or ""
            job_emp = job.get("employer") or job.get("firma") or ""
        else:
            job_title = getattr(job, "title", "") or ""
            job_desc = getattr(job, "description", "") or ""
            job_emp = getattr(job, "employer", "") or ""

        full_job_text = f"{job_title} {job_emp} {job_desc}".lower()

        # 1. Skills factor (40%)
        candidate_skills = profile_dict.get("skills", [])
        candidate_keywords = profile_dict.get("keywords", [])
        all_skills = set(
            [s.lower() for s in candidate_skills] + [k.lower() for k in candidate_keywords]
        )

        if all_skills:
            matched_skills = [s for s in all_skills if s in full_job_text]
            skill_score = min(1.0, 0.4 + (len(matched_skills) / max(1, len(all_skills))) * 0.6)
            if not matched_skills:
                skill_score = 0.5
        else:
            skill_score = 0.7

        # 2. Experience factor (25%)
        candidate_exp = profile_dict.get("experience_years", 0.0)
        exp_score = 0.75
        if "senior" in full_job_text or "lead" in full_job_text or "leiter" in full_job_text:
            if candidate_exp >= 5.0:
                exp_score = 0.95
            elif candidate_exp >= 3.0:
                exp_score = 0.75
            else:
                exp_score = 0.5
        elif "junior" in full_job_text or "trainee" in full_job_text:
            if candidate_exp <= 3.0:
                exp_score = 0.95
            else:
                exp_score = 0.8
        else:
            exp_score = 0.85 if candidate_exp >= 2.0 else 0.7

        # 3. German Language CEFR factor (20%)
        user_rank = CEFR_LEVELS.get(user_german.upper(), 3)
        german_score = 0.9

        # Check if job requires specific CEFR level
        req_match = re.search(r"\b(?:deutsch|german)\b.*?\b(c2|c1|b2|b1|a2|a1)\b", full_job_text)
        if not req_match:
            req_match = re.search(
                r"\b(c2|c1|b2|b1|a2|a1)\s*(?:deutsch|kenntnisse|niveau)?\b", full_job_text
            )

        if req_match:
            required_level = req_match.group(1).upper()
            required_rank = CEFR_LEVELS.get(required_level, 3)
            if user_rank < required_rank:
                diff = required_rank - user_rank
                german_score = max(0.2, 0.9 - (diff * 0.25))
            else:
                german_score = 1.0
        elif "deutsch" in full_job_text:
            if user_rank < 2:  # Below A2
                german_score = 0.5
            else:
                german_score = 0.9

        # 4. Career Goals factor (15%)
        goals_score = 0.75
        if user_goals and user_goals.strip():
            goal_words = [w.lower() for w in re.findall(r"\w+", user_goals) if len(w) > 3]
            if goal_words:
                matched_goals = [w for w in goal_words if w in full_job_text]
                if matched_goals:
                    goals_score = min(1.0, 0.6 + (len(matched_goals) / len(goal_words)) * 0.4)
                else:
                    goals_score = 0.5

        # Weighted calculation
        weighted = (
            (0.40 * skill_score) + (0.25 * exp_score) + (0.20 * german_score) + (0.15 * goals_score)
        )
        return round(weighted * 100, 1)

    async def match_jobs(
        self,
        cv_profile: dict[str, Any] | ExtractedCVProfile,
        user_prefs: dict[str, Any] | Any,
        jobs: list[Any],
        lang: str = "de",
    ) -> list[dict[str, Any]]:
        """Score candidate profile against list of vacancies and generate localized explanations."""
        results = []
        for job in jobs:
            score = self.calculate_score(cv_profile, user_prefs, job)
            reasons = {
                "en": f"Strong alignment in technical skills and professional background ({score}% match).",
                "de": f"Hohe Übereinstimmung mit Ihren Fachkompetenzen und Ihrer Berufserfahrung ({score}% Übereinstimmung).",
                "uk": f"Висока відповідність кваліфікації та професійного досвіду ({score}% збіг).",
                "ru": f"Высокое соответствие квалификации и профессионального опыта ({score}% совпадение).",
            }
            reason = reasons.get(lang, reasons["en"])
            results.append(
                {
                    "job": job,
                    "score": score,
                    "match_reason": reason,
                    "factors": {
                        "skills": 0.40,
                        "experience": 0.25,
                        "german_level": 0.20,
                        "goals_alignment": 0.15,
                    },
                }
            )

        return sorted(results, key=lambda x: x["score"], reverse=True)


# Global default instances
cv_analyzer = AICVAnalyzer()
ai_matcher = AIJobMatcher()
