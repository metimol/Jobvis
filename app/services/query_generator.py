"""Bundesagentur für Arbeit (BA) Search Query Generation Service.

Translates natural language user goals and CV profile into optimal,
targeted search parameters ('was', 'wo', 'arbeitszeit', 'angebotsart')
for the Bundesagentur für Arbeit Jobsuche API using LangChain Gemini LLM
with resilient heuristic fallback.
"""

import asyncio
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.config import settings

logger = logging.getLogger(__name__)

# Valid BA API working time options
VALID_ARBEITSZEIT_VALUES = {"vz", "tz", "mj", "ho"}

ARBEITSZEIT_MAP: dict[str, str] = {
    "vz": "vz",
    "vollzeit": "vz",
    "fulltime": "vz",
    "full-time": "vz",
    "tz": "tz",
    "teilzeit": "tz",
    "parttime": "tz",
    "part-time": "tz",
    "mj": "mj",
    "minijob": "mj",
    "mini-job": "mj",
    "ho": "ho",
    "homeoffice": "ho",
    "home-office": "ho",
    "remote": "ho",
}

# Common translations for frequent job goal terms into German BA keywords
TERM_TRANSLATIONS: dict[str, str] = {
    # English
    "retail": "Einzelhandel",
    "marketing": "Marketing",
    "sales": "Vertrieb",
    "accounting": "Buchhaltung",
    "developer": "Entwickler",
    "software engineer": "Softwareentwickler",
    "nurse": "Pflegefachkraft",
    "electrician": "Elektriker",
    "driver": "Fahrer",
    "warehouse": "Lagerlogistik",
    "logistics": "Logistik",
    "customer service": "Kundenservice",
    "receptionist": "Empfangskraft",
    "cook": "Koch",
    "waiter": "Kellner",
    "waitress": "Kellnerin",
    "cleaner": "Reinigungskraft",
    "craftsman": "Handwerker",
    "carpenter": "Tischler",
    "mechanic": "Kfz-Mechatroniker",
    "painter": "Maler",
    "plumber": "Anlagenmechaniker SHK",
    # Ukrainian & Russian
    "водій": "Fahrer",
    "водитель": "Fahrer",
    "кухар": "Koch",
    "повар": "Koch",
    "електрик": "Elektriker",
    "электрик": "Elektriker",
    "прибиральник": "Reinigungskraft",
    "прибиральниця": "Reinigungskraft",
    "уборщица": "Reinigungskraft",
    "уборщик": "Reinigungskraft",
    "продавець": "Einzelhandel",
    "продавец": "Einzelhandel",
    "офіціант": "Kellner",
    "официант": "Kellner",
    "медсестра": "Pflegefachkraft",
    "склад": "Lagerlogistik",
    "комірник": "Lagerlogistik",
    "кладовщик": "Lagerlogistik",
    "розробник": "Entwickler",
    "разработчик": "Entwickler",
    "програміст": "Softwareentwickler",
    "программист": "Softwareentwickler",
    "бухгалтер": "Buchhalter",
    "вихователь": "Erzieher",
    "воспитатель": "Erzieher",
    "будівельник": "Handwerker",
    "строитель": "Handwerker",
    "механік": "Kfz-Mechatroniker",
    "механик": "Kfz-Mechatroniker",
    "маляр": "Maler",
    "столяр": "Tischler",
    "сантехнік": "Anlagenmechaniker SHK",
    "сантехник": "Anlagenmechaniker SHK",
}

STOPWORDS = {
    "i",
    "want",
    "a",
    "an",
    "the",
    "in",
    "and",
    "or",
    "for",
    "to",
    "at",
    "looking",
    "job",
    "jobs",
    "position",
    "role",
    "work",
    "seeking",
    "need",
    "ich",
    "suche",
    "eine",
    "einen",
    "ein",
    "als",
    "im",
    "raum",
    "stelle",
    "stellen",
    "arbeit",
    "beruf",
    "möchte",
    "gern",
    "gerne",
    "bereich",
    "mit",
    "minijob",
    "vollzeit",
    "teilzeit",
    "ausbildung",
    "full-time",
    "part-time",
    "remote",
    "homeoffice",
    "duales",
    "studium",
    "lehrstelle",
    "azubi",
    "я",
    "хочу",
    "шукаю",
    "ищу",
    "роботу",
    "работу",
    "в",
    "у",
    "на",
    "для",
    "та",
    "і",
}

_QUERY_GEN_SENTINEL = object()


def _clean_str(val: Any) -> str | None:
    """Normalize string and convert empty/null-like strings to None."""
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in {
        "none",
        "null",
        "nil",
        "n/a",
        "undefined",
        "not specified",
        "unspecified",
        "keine",
        "",
    }:
        return None
    return s


def _clean_arbeitszeit(val: Any) -> str | None:
    """Clean and map arbeitszeit input to valid BA values ('vz', 'tz', 'mj', 'ho')."""
    cleaned = _clean_str(val)
    if not cleaned:
        return None
    return ARBEITSZEIT_MAP.get(
        cleaned.lower(), cleaned if cleaned in VALID_ARBEITSZEIT_VALUES else None
    )


class BAQueryParams(BaseModel):
    """Targeted search parameters for the Bundesagentur für Arbeit Jobsuche API."""

    model_config = ConfigDict(extra="ignore")

    was: str | None = Field(
        default=None,
        description="Search term, job title, or German keywords for BA Jobsuche (e.g. 'Einzelhandel Marketing', 'Python Entwickler')",
    )
    wo: str | None = Field(
        default=None,
        description="City, region, or postal code (e.g. 'Berlin', 'München') or None if not specified",
    )
    arbeitszeit: str | None = Field(
        default=None,
        description="Working time model: 'vz' (Vollzeit), 'tz' (Teilzeit), 'mj' (Minijob), 'ho' (Homeoffice), or combinations, or None",
    )
    angebotsart: int | None = Field(
        default=1,
        description="Type of offer: 1 for standard employment (Arbeit), 4 for apprenticeship / training (Ausbildung)",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for ArbeitsagenturClient search_jobs parameters."""
        return {
            "was": self.was,
            "wo": self.wo,
            "arbeitszeit": self.arbeitszeit,
            "angebotsart": self.angebotsart,
        }

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


def _normalize_cv_profile(cv: Any) -> dict[str, Any]:
    """Normalize CV profile from dict, CVAnalysis, or ExtractedCVProfile instance."""
    if not cv:
        return {"skills": [], "experience_years": 0.0, "keywords": [], "city": None}
    if isinstance(cv, dict):
        return {
            "skills": cv.get("skills") or [],
            "experience_years": float(cv.get("experience_years") or 0.0),
            "keywords": cv.get("keywords") or [],
            "city": cv.get("city"),
        }
    return {
        "skills": getattr(cv, "skills", []) or [],
        "experience_years": float(getattr(cv, "experience_years", 0.0) or 0.0),
        "keywords": getattr(cv, "keywords", []) or [],
        "city": getattr(cv, "city", None),
    }


def _normalize_user_prefs(prefs: Any) -> dict[str, Any]:
    """Normalize user preferences from dict or Profile instance."""
    if not prefs:
        return {"location": "", "desired_job_type": "all", "radius_km": 25, "goals": ""}
    if isinstance(prefs, dict):
        return {
            "location": prefs.get("location") or "",
            "desired_job_type": prefs.get("desired_job_type") or "all",
            "radius_km": int(prefs.get("radius_km") or 25),
            "goals": prefs.get("goals") or "",
        }
    return {
        "location": getattr(prefs, "location", "") or "",
        "desired_job_type": getattr(prefs, "desired_job_type", "all") or "all",
        "radius_km": int(getattr(prefs, "radius_km", 25) or 25),
        "goals": getattr(prefs, "goals", "") or "",
    }


def extract_heuristic_query(
    goals: str | None,
    cv_dict: dict[str, Any],
    prefs_dict: dict[str, Any],
) -> BAQueryParams:
    """Robust heuristic extraction for BA API search parameters when goals are missing or LLM is offline."""
    goals_text = (goals or prefs_dict.get("goals") or "").strip()
    goals_lower = goals_text.lower()

    # 1. Working time (arbeitszeit)
    arbeitszeit: str | None = None
    if re.search(
        r"\b(minijob|mini-job|geringf[uü]gig|aushilfe|520|538|мініджоб|миниджоб)\b", goals_lower
    ):
        arbeitszeit = "mj"
    elif re.search(
        r"\b(teilzeit|part-time|part\s*time|halbtags|неповна\s*зайнятість|неполная\s*занятость)\b",
        goals_lower,
    ):
        arbeitszeit = "tz"
    elif re.search(
        r"\b(vollzeit|full-time|full\s*time|ganztags|повна\s*зайнятість|полная\s*занятость)\b",
        goals_lower,
    ):
        arbeitszeit = "vz"
    elif re.search(
        r"\b(homeoffice|home-office|remote|telearbeit|дистанційн\w*|дистанционн\w*)\b", goals_lower
    ):
        arbeitszeit = "ho"
    else:
        desired = prefs_dict.get("desired_job_type")
        if desired in VALID_ARBEITSZEIT_VALUES:
            arbeitszeit = desired

    # 2. Offer type (angebotsart: 1 = regular job, 4 = Ausbildung/apprenticeship, 2 = Selbstständigkeit)
    angebotsart: int = 1
    if re.search(
        r"\b(ausbildung|lehrstelle|duales\s*studium|apprenticeship|azubi|trainee|навчання|стажування|стажировка)\b",
        goals_lower,
    ):
        angebotsart = 4
    elif re.search(r"\b(selbstst[aä]ndig|freiberuflich|freelance|фріланс|фриланс)\b", goals_lower):
        angebotsart = 2

    # 3. Location (wo)
    wo: str | None = None
    # Check if location mentioned in goals via patterns like "in Berlin", "im Raum München", "у Берліні"
    loc_match = re.search(
        r"\b(?:in|im\s+raum|around|near|im|nach|у|в)\s+([A-ZÄÖÜА-ЯІЇЄ][a-zäöüßа-яіїє]+(?:\s+[A-ZÄÖÜА-ЯІЇЄ][a-zäöüßа-яіїє]+)?)",
        goals_text,
    )
    if loc_match:
        cand_loc = loc_match.group(1).strip()
        if cand_loc.lower() not in {
            "der",
            "die",
            "das",
            "ein",
            "eine",
            "a",
            "an",
            "the",
            "retail",
            "marketing",
        }:
            wo = cand_loc
    if not wo and arbeitszeit != "ho":
        wo = prefs_dict.get("location") or cv_dict.get("city") or None

    # 4. Search keywords (was)
    was: str | None = None
    if goals_text:
        # Check domain translations first
        translated_tokens = []
        for term, de_term in TERM_TRANSLATIONS.items():
            if re.search(r"\b" + re.escape(term) + r"\b", goals_lower):
                translated_tokens.append(de_term)

        if translated_tokens:
            was = " ".join(dict.fromkeys(translated_tokens))
        else:
            # Tokenize and filter stopwords (supports unicode/Cyrillic words)
            tokens = re.findall(r"[\w\-]+", goals_text)
            filtered = [t for t in tokens if t.lower() not in STOPWORDS and len(t) > 2]
            if filtered:
                was = " ".join(filtered[:3])

    # Fallback to CV skills or keywords if was could not be determined from goals
    if not was:
        skills = cv_dict.get("skills", [])
        keywords = cv_dict.get("keywords", [])
        if skills:
            was = " ".join(skills[:2])
        elif keywords:
            was = " ".join(keywords[:2])

    return BAQueryParams(
        was=was,
        wo=wo,
        arbeitszeit=arbeitszeit,
        angebotsart=angebotsart,
    )


QUERY_GEN_PROMPT = (
    "You are an expert recruitment and job search query optimization specialist for the German "
    "Bundesagentur für Arbeit (BA) Jobsuche platform.\n"
    "Given the candidate's natural language career goals and CV profile, generate optimal, targeted search "
    "parameters ('was', 'wo', 'arbeitszeit', 'angebotsart') for the BA Jobsuche API.\n\n"
    "BA API Parameter Rules:\n"
    "1. 'was' (Beruf / Suchbegriff): Concise German job title or targeted keywords (e.g. 'Einzelhandel Marketing', "
    "'Python Entwickler', 'Buchhalter'). Translate multilingual goals into German standard professional terms. "
    "Exclude filler words like 'job', 'suche', 'looking for'.\n"
    "2. 'wo' (Arbeitsort): City or region name if specified in candidate goals or profile, otherwise null.\n"
    "3. 'arbeitszeit': Working time filter: 'vz' (Vollzeit), 'tz' (Teilzeit), 'mj' (Minijob), 'ho' (Homeoffice), "
    "or null if flexible/unspecified.\n"
    "4. 'angebotsart': Integer 1 for regular employment (standard), 4 for apprenticeship / training (Ausbildung / duales Studium).\n\n"
    "Candidate Goals:\n"
    "{goals}\n\n"
    "Candidate Profile Summary:\n"
    "- Skills: {skills}\n"
    "- Experience: {experience_years} years\n"
    "- Candidate Location: {location}\n"
    "- Desired Job Type: {job_type}\n\n"
    "{format_instructions}\n"
)


async def generate_search_query(
    goals: str | None = None,
    cv_profile: dict[str, Any] | Any | None = None,
    user_prefs: dict[str, Any] | Any | None = None,
    llm: Any | None = None,
    api_key: Any = _QUERY_GEN_SENTINEL,
    timeout_seconds: float = 8.0,
) -> BAQueryParams:
    """Generate optimal Arbeitsagentur API search parameters from natural language goals and CV profile.

    Uses LangChain Gemini LLM structured output when available, and gracefully falls back to
    resilient heuristic parameter extraction if LLM is unavailable, offline, or times out.

    Args:
        goals: User's natural language goal description (e.g., 'I want a minijob in retail and marketing').
        cv_profile: CV analysis data (dict, CVAnalysis model, or None).
        user_prefs: User profile preferences (dict, Profile model, or None).
        llm: Optional LangChain ChatModel or Runnable. If provided, used directly without network lookup.
        api_key: Optional Google GenAI API key. If not provided, reads from settings.GOOGLE_API_KEY.
                 Explicitly passing None disables API key lookup and uses heuristics.
        timeout_seconds: Maximum seconds to wait for LLM invocation before falling back to heuristics.

    Returns:
        BAQueryParams with 'was', 'wo', 'arbeitszeit', and 'angebotsart'.
    """
    cv_dict = _normalize_cv_profile(cv_profile)
    prefs_dict = _normalize_user_prefs(user_prefs)

    raw_goals = goals if goals is not None else prefs_dict.get("goals", "")
    goals_text = (raw_goals or "").strip()

    # Pre-compute heuristic parameters (used for empty goals or LLM fallback)
    heuristic_params = extract_heuristic_query(goals_text, cv_dict, prefs_dict)

    # Seamless handling of empty or missing goals: return heuristic params without invoking LLM
    if not goals_text:
        logger.debug("Goals empty or missing; using normalized profile/CV heuristic parameters.")
        return heuristic_params

    # Determine active LLM
    active_llm = llm
    if api_key is _QUERY_GEN_SENTINEL:
        effective_api_key = settings.GOOGLE_API_KEY
    else:
        effective_api_key = api_key

    if active_llm is None and effective_api_key:
        str_key = str(effective_api_key)
        if not str_key.startswith("mock-") and len(str_key) > 10:
            try:
                from ai.config import model as configured_model

                active_llm = configured_model
            except Exception as e:
                logger.warning("Failed to load Gemini model from ai.config: %s", e)
                active_llm = None

    if active_llm is None:
        logger.debug("No active LLM available; returning heuristic BA search parameters.")
        return heuristic_params

    try:
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import PromptTemplate

        parser = PydanticOutputParser(pydantic_object=BAQueryParams)
        prompt = PromptTemplate(
            template=QUERY_GEN_PROMPT,
            input_variables=["goals", "skills", "experience_years", "location", "job_type"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | active_llm | parser

        prompt_input = {
            "goals": goals_text,
            "skills": ", ".join(cv_dict.get("skills", [])) or "None listed",
            "experience_years": cv_dict.get("experience_years", 0.0),
            "location": prefs_dict.get("location") or cv_dict.get("city") or "Not specified",
            "job_type": prefs_dict.get("desired_job_type") or "Not specified",
        }

        res: BAQueryParams = await asyncio.wait_for(
            chain.ainvoke(prompt_input),
            timeout=timeout_seconds,
        )

        # Normalize outputs and merge with fallbacks for missing/malformed fields
        raw_was = _clean_str(res.was)
        raw_wo = _clean_str(res.wo)
        clean_arbeitszeit = _clean_arbeitszeit(res.arbeitszeit) or heuristic_params.arbeitszeit
        clean_angebotsart = (
            res.angebotsart if res.angebotsart in {1, 2, 4} else heuristic_params.angebotsart
        )

        was_val = raw_was or heuristic_params.was

        # Intelligent location fallback:
        # If candidate requested remote / homeoffice and gave no location, respect None across Germany
        if raw_wo:
            wo_val = raw_wo
        elif clean_arbeitszeit == "ho":
            # For remote, only set location if user explicitly stated a location in goals
            loc_in_goals = bool(
                re.search(r"\b(?:in|im\s+raum|around|near)\s+([A-ZÄÖÜ][a-zäöüß]+)", goals_text)
            )
            wo_val = heuristic_params.wo if loc_in_goals else None
        else:
            wo_val = heuristic_params.wo

        return BAQueryParams(
            was=was_val,
            wo=wo_val,
            arbeitszeit=clean_arbeitszeit,
            angebotsart=clean_angebotsart or 1,
        )

    except Exception as exc:
        logger.warning(
            "LangChain Gemini query generation failed or timed out (%s); using heuristic fallback.",
            exc,
        )
        return heuristic_params


# Convenience alias
generate_ba_query = generate_search_query
