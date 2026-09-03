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

# Multi-industry Skills & Vocations Taxonomy across DE, EN, UK, RU
SKILLS_CATALOG: dict[str, str] = {
    # 1. Tech & IT
    "python": "Python",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "react": "React",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "node": "Node.js",
    "nodejs": "Node.js",
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
    "devops": "DevOps",
    "linux": "Linux",
    "frontend": "Frontend Development",
    "backend": "Backend Development",
    "fullstack": "Fullstack Development",
    "програмування": "Software Development",
    "программирование": "Software Development",
    "розробка": "Software Development",
    "разработка": "Software Development",
    # 2. Crafts & Trades / Handwerk
    "sps": "SPS-Programmierung",
    "sps-programmierung": "SPS-Programmierung",
    "schaltanlagenbau": "Schaltanlagenbau",
    "industrieautomation": "Industrieautomation",
    "elektroniker": "Elektroniker",
    "elektrotechnik": "Elektrotechnik",
    "elektriker": "Elektriker",
    "electrician": "Elektriker",
    "електрик": "Elektriker",
    "электрик": "Elektriker",
    "tischler": "Tischler",
    "schreiner": "Tischler",
    "carpenter": "Tischler",
    "тесляр": "Tischler",
    "столяр": "Tischler",
    "плотник": "Tischler",
    "maler": "Maler",
    "lackierer": "Maler & Lackierer",
    "painter": "Maler",
    "маляр": "Maler",
    "sanitär": "Sanitär- und Klimatechnik (SHK)",
    "klempner": "Sanitär- und Klimatechnik (SHK)",
    "plumber": "Sanitär- und Klimatechnik (SHK)",
    "сантехнік": "Sanitär- und Klimatechnik (SHK)",
    "сантехник": "Sanitär- und Klimatechnik (SHK)",
    "heizungsbau": "Sanitär- und Klimatechnik (SHK)",
    "schweißen": "Schweißen",
    "schweisser": "Schweißen",
    "welder": "Schweißen",
    "welding": "Schweißen",
    "зварювальник": "Schweißen",
    "зварювання": "Schweißen",
    "сварщик": "Schweißen",
    "сварка": "Schweißen",
    "bau": "Bauhandwerk",
    "bauarbeiter": "Bauhandwerk",
    "construction": "Bauhandwerk",
    "будівництво": "Bauhandwerk",
    "строительство": "Bauhandwerk",
    "schlosser": "Schlosser",
    "locksmith": "Schlosser",
    "слюсар": "Schlosser",
    "слесарь": "Schlosser",
    # 3. Healthcare & Nursing / Pflege
    "grundpflege": "Grundpflege",
    "behandlungspflege": "Behandlungspflege",
    "altenpflege": "Altenpflege",
    "krankenpflege": "Krankenpflege",
    "krankenpfleger": "Krankenpflege",
    "krankenschwester": "Krankenpflege",
    "pflegefachkraft": "Pflegefachkraft",
    "pflege": "Pflege & Betreuung",
    "caregiver": "Pflege & Betreuung",
    "nursing": "Krankenpflege",
    "nurse": "Krankenpflege",
    "медсестра": "Krankenpflege",
    "догляд": "Pflege & Betreuung",
    "сиделка": "Pflege & Betreuung",
    "уход": "Pflege & Betreuung",
    "medikamentenverabreichung": "Medikamentenverabreichung",
    "wundversorgung": "Wundversorgung",
    "pflegedokumentation": "Pflegedokumentation",
    "medizin": "Medizinische Versorgung",
    "betreuung": "Betreuung",
    "heilerziehungspflege": "Heilerziehungspflege",
    # 4. Logistics & Warehouse / Lager
    "lagerlogistik": "Lagerlogistik",
    "lager": "Lagerlogistik",
    "lagerist": "Lagerist",
    "warehouse": "Lagerlogistik",
    "склад": "Lagerlogistik",
    "staplerschein": "Gabelstapler",
    "gabelstapler": "Gabelstapler",
    "staplerfahrer": "Gabelstapler",
    "forklift": "Gabelstapler",
    "навантажувач": "Gabelstapler",
    "погрузчик": "Gabelstapler",
    "kommissionierung": "Kommissionierung",
    "kommissionierer": "Kommissionierung",
    "order picking": "Kommissionierung",
    "picker": "Kommissionierung",
    "комплектування": "Kommissionierung",
    "комплектовка": "Kommissionierung",
    "wareneingang": "Wareneingang",
    "warenausgang": "Warenausgang",
    "versand": "Versand & Logistik",
    "shipping": "Versand & Logistik",
    "verpackung": "Verpackung",
    "packaging": "Verpackung",
    "пакування": "Verpackung",
    "упаковка": "Verpackung",
    # 5. Gastronomy & Hospitality / Gastro
    "koch": "Koch",
    "köchin": "Koch",
    "chef": "Koch",
    "cook": "Koch",
    "кухар": "Koch",
    "повар": "Koch",
    "beikoch": "Beikoch",
    "küchenhilfe": "Küchenhilfe",
    "kitchen helper": "Küchenhilfe",
    "kellner": "Kellner / Service",
    "kellnerin": "Kellner / Service",
    "waiter": "Kellner / Service",
    "waitress": "Kellner / Service",
    "servicekraft": "Kellner / Service",
    "офіціант": "Kellner / Service",
    "официант": "Kellner / Service",
    "barista": "Barista",
    "бариста": "Barista",
    "küche": "Küche & Gastronomie",
    "gastronomie": "Gastronomie",
    "catering": "Catering",
    "haccp": "HACCP",
    "reinigungskraft": "Reinigungskraft",
    "cleaner": "Reinigungskraft",
    "прибиральник": "Reinigungskraft",
    "уборщик": "Reinigungskraft",
    # 6. Retail & Sales / Handel
    "kassierer": "Kasse & Verkauf",
    "kassiererin": "Kasse & Verkauf",
    "kasse": "Kasse & Verkauf",
    "kassieren": "Kasse & Verkauf",
    "cashier": "Kasse & Verkauf",
    "касир": "Kasse & Verkauf",
    "кассир": "Kasse & Verkauf",
    "verkäufer": "Verkauf & Einzelhandel",
    "verkäuferin": "Verkauf & Einzelhandel",
    "sales": "Verkauf & Einzelhandel",
    "einzelhandel": "Einzelhandel",
    "retail": "Einzelhandel",
    "продавець": "Verkauf & Einzelhandel",
    "продавец": "Verkauf & Einzelhandel",
    "kundenberatung": "Kundenberatung",
    "customer service": "Kundenberatung",
    "warenverräumung": "Warenverräumung",
    # 7. Office & Administration / Admin
    "bürokaufmann": "Büroorganisation",
    "bürokauffrau": "Büroorganisation",
    "bürokommunikation": "Büroorganisation",
    "sachbearbeiter": "Sachbearbeitung",
    "sachbearbeitung": "Sachbearbeitung",
    "clerk": "Sachbearbeitung",
    "адміністратор": "Administration & Büro",
    "администратор": "Administration & Büro",
    "buchhaltung": "Buchhaltung",
    "rechnungswesen": "Rechnungswesen",
    "accounting": "Buchhaltung",
    "bookkeeping": "Buchhaltung",
    "бухгалтерія": "Buchhaltung",
    "бухгалтер": "Buchhaltung",
    "бухгалтерия": "Buchhaltung",
    "empfang": "Empfang & Rezeption",
    "rezeption": "Empfang & Rezeption",
    "reception": "Empfang & Rezeption",
    "sekretariat": "Sekretariat",
    "kundenbetreuung": "Kundenbetreuung",
    "projektmanagement": "Projektmanagement",
    "project management": "Projektmanagement",
    # 8. Transport & Driving / Fahrer
    "lkw-fahrer": "LKW-Fahrer",
    "berufskraftfahrer": "Berufskraftfahrer",
    "truck driver": "LKW-Fahrer",
    "водій": "Fahrer & Transport",
    "водитель": "Fahrer & Transport",
    "kurier": "Kurier & Zusteller",
    "courier": "Kurier & Zusteller",
    "zusteller": "Kurier & Zusteller",
    "delivery": "Kurier & Zusteller",
    "кур'єр": "Kurier & Zusteller",
    "курьер": "Kurier & Zusteller",
    "auslieferungsfahrer": "Auslieferungsfahrer",
    "fahrer": "Fahrer & Transport",
    "driver": "Fahrer & Transport",
    "führerschein b": "Führerschein Klasse B",
    "führerschein c": "Führerschein Klasse C",
    "führerschein ce": "Führerschein Klasse CE",
}

# Major German Cities for Location / City Extraction
GERMAN_CITIES: dict[str, str] = {
    "frankfurt am main": "Frankfurt am Main",
    "frankfurt/main": "Frankfurt am Main",
    "frankfurt": "Frankfurt am Main",
    "berlin": "Berlin",
    "hamburg": "Hamburg",
    "münchen": "München",
    "munchen": "München",
    "munich": "München",
    "köln": "Köln",
    "koln": "Köln",
    "cologne": "Köln",
    "stuttgart": "Stuttgart",
    "düsseldorf": "Düsseldorf",
    "dusseldorf": "Düsseldorf",
    "leipzig": "Leipzig",
    "dortmund": "Dortmund",
    "essen": "Essen",
    "bremen": "Bremen",
    "dresden": "Dresden",
    "hannover": "Hannover",
    "hanover": "Hannover",
    "nürnberg": "Nürnberg",
    "nurnberg": "Nürnberg",
    "nuremberg": "Nürnberg",
    "duisburg": "Duisburg",
    "bochum": "Bochum",
    "wuppertal": "Wuppertal",
    "bielefeld": "Bielefeld",
    "bonn": "Bonn",
    "münster": "Münster",
    "munster": "Münster",
    "karlsruhe": "Karlsruhe",
    "mannheim": "Mannheim",
    "augsburg": "Augsburg",
    "wiesbaden": "Wiesbaden",
    "gelsenkirchen": "Gelsenkirchen",
    "mönchengladbach": "Mönchengladbach",
    "braunschweig": "Braunschweig",
    "chemnitz": "Chemnitz",
    "kiel": "Kiel",
    "aachen": "Aachen",
    "halle": "Halle",
    "magdeburg": "Magdeburg",
    "freiburg": "Freiburg",
    "krefeld": "Krefeld",
    "mainz": "Mainz",
    "lübeck": "Lübeck",
    "lubeck": "Lübeck",
    "erfurt": "Erfurt",
    "oberhausen": "Oberhausen",
    "rostock": "Rostock",
    "kassel": "Kassel",
    "hagen": "Hagen",
    "potsdam": "Potsdam",
    "saarbrücken": "Saarbrücken",
    "saarbrucken": "Saarbrücken",
    "hamm": "Hamm",
    "ludwigshafen": "Ludwigshafen",
    "mülheim": "Mülheim",
    "oldenburg": "Oldenburg",
    "osnabrück": "Osnabrück",
    "osnabruck": "Osnabrück",
    "leverkusen": "Leverkusen",
    "heidelberg": "Heidelberg",
    "darmstadt": "Darmstadt",
    "regensburg": "Regensburg",
    "ingolstadt": "Ingolstadt",
    "würzburg": "Würzburg",
    "wurzburg": "Würzburg",
    "wolfsburg": "Wolfsburg",
    "ulm": "Ulm",
    "heilbronn": "Heilbronn",
    "pforzheim": "Pforzheim",
    "göttingen": "Göttingen",
    "gottingen": "Göttingen",
    "bottrop": "Bottrop",
    "recklinghausen": "Recklinghausen",
    "reutlingen": "Reutlingen",
    "koblenz": "Koblenz",
    "bremerhaven": "Bremerhaven",
    "bergisch gladbach": "Bergisch Gladbach",
    "remscheid": "Remscheid",
    "jena": "Jena",
    "erlangen": "Erlangen",
    "trier": "Trier",
    "salzgitter": "Salzgitter",
    "cottbus": "Cottbus",
    "hildesheim": "Hildesheim",
}


class ExtractedCVProfile(BaseModel):
    """Structured candidate profile extracted from CV text."""

    skills: list[str] = Field(default_factory=list)
    experience_years: float = Field(default=0.0)
    education: list[str | dict[str, Any]] = Field(default_factory=list)
    detected_languages: dict[str, str] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    german_level: str = "B1"
    city: str | None = None
    radius_km: int = 25
    desired_job_type: str = "all"
    goals: str | None = None


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
                "summary": None,
                "german_level": "B1",
                "city": None,
                "radius_km": 25,
                "desired_job_type": "all",
                "goals": None,
            }

        # If API key configured, attempt LangChain Google GenAI structured extraction
        if self.api_key and not self.api_key.startswith("mock-") and len(self.api_key) > 10:
            try:
                from langchain_core.output_parsers import PydanticOutputParser
                from langchain_core.prompts import PromptTemplate

                from ai.config import model as llm

                parser = PydanticOutputParser(pydantic_object=ExtractedCVProfile)
                prompt = PromptTemplate(
                    template=(
                        "You are an expert AI recruiter for Jobvis and the German Jobcenter.\n"
                        "Extract structured candidate profile details and job preferences from the CV text into JSON format conforming to the schema.\n\n"
                        "Extraction Guidelines:\n"
                        "1. Skills & Experience: Extract professional competencies and total years of experience.\n"
                        "2. German Proficiency (german_level): Detect CEFR level (A1, A2, B1, B2, C1, C2, or Native -> C2/C1). Defaults to B1.\n"
                        "3. Location / City (city): Extract candidate's city/town of residence or target work location (e.g. 'Berlin', 'München', 'Hamburg', 'Köln').\n"
                        "4. Search Radius (radius_km): Commute radius in km (default 25, 5-200 km).\n"
                        "5. Desired Job Type (desired_job_type): Preferred employment type: 'vz' (full-time), 'tz' (part-time), 'mj' (minijob), or 'all'.\n"
                        "6. Career Goals (goals): Candidate's career goals, target role, or professional objective.\n\n"
                        "{format_instructions}\n\n"
                        "CV Text:\n"
                        "{cv_text}\n"
                    ),
                    input_variables=["cv_text"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                chain = prompt | llm | parser
                res: ExtractedCVProfile = await chain.ainvoke({"cv_text": cv_text[:8000]})
                logger.info(
                    "Google GenAI CV extraction succeeded: %d skills, %.1f exp yrs, %s languages, city=%s, radius=%d, german_level=%s",
                    len(res.skills),
                    res.experience_years,
                    list(res.detected_languages.keys()),
                    res.city,
                    res.radius_km,
                    res.german_level,
                )
                logger.debug("Extracted CV Profile details: %s", res.model_dump())
                return res.model_dump()
            except Exception as e:
                logger.warning(
                    "Google GenAI extraction failed (%s), falling back to heuristics.", e
                )

        # Robust Heuristic Fallback Analysis
        result = self._heuristic_analyze(cv_text)
        logger.info(
            "Heuristic CV analysis succeeded: %d skills, %.1f exp yrs, %s languages, city=%s, radius=%d, german_level=%s",
            len(result.get("skills", [])),
            result.get("experience_years", 0.0),
            list(result.get("detected_languages", {}).keys()),
            result.get("city"),
            result.get("radius_km", 25),
            result.get("german_level", "B1"),
        )
        logger.debug("Heuristic CV analysis details: %s", result)
        return result

    def _heuristic_analyze(self, cv_text: str) -> dict[str, Any]:
        """Deterministic heuristic extractor supporting DE, EN, UK, RU CVs across multiple industries."""
        text_lower = cv_text.lower()
        skills: list[str] = []

        for keyword, canonical in SKILLS_CATALOG.items():
            pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
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
        german_level = "B1"
        de_match = re.search(
            r"\b(?:deutsch|german|німецька|немецкий)\b.*?\b(c2|c1|b2|b1|a2|a1|muttersprache|native|рідна|родной)\b",
            text_lower,
        )
        if not de_match:
            de_match = re.search(
                r"\b(c2|c1|b2|b1|a2|a1)\s*(?:in\s+)?(?:deutsch|german|kenntnisse|niveau)\b",
                text_lower,
            )
        if not de_match:
            de_match = re.search(
                r"\b(?:deutschkenntnisse|sprachkenntnisse|sprachen)\b.*?\b(c2|c1|b2|b1|a2|a1)\b",
                text_lower,
            )

        if de_match:
            matched_val = de_match.group(1).lower()
            if matched_val in ["muttersprache", "native", "рідна", "родной", "c2"]:
                german_level = "C2"
            elif matched_val in ["c1", "b2", "b1", "a2", "a1"]:
                german_level = matched_val.upper()
        elif "c1" in text_lower:
            german_level = "C1"
        elif "b2" in text_lower:
            german_level = "B2"
        elif "a2" in text_lower:
            german_level = "A2"
        elif "a1" in text_lower:
            german_level = "A1"
        else:
            german_level = "B1"

        detected_languages["de"] = german_level

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

        # Ukrainian / Russian detection
        if re.search(r"[\u0400-\u04FF]", cv_text):
            if any(k in text_lower for k in ["україн", "ukrain"]):
                detected_languages["uk"] = "C2"
            if any(k in text_lower for k in ["русск", "russian"]):
                detected_languages["ru"] = "C2"

        # City detection
        city: str | None = None

        # 1. Look for explicit location / residence labels
        loc_label_match = re.search(
            r"\b(?:wohnort|standort|ort|stadt|location|city|residence|місто|город)\s*[:\-–]\s*([A-Za-zÄÖÜäöüß\s\-/]+?)(?=[,\n\r;\.]|$)",
            cv_text,
            flags=re.IGNORECASE,
        )
        if loc_label_match:
            candidate_loc = loc_label_match.group(1).strip()
            clean_loc = re.sub(r"^\d{5}\s*", "", candidate_loc).strip()
            if clean_loc.lower() in GERMAN_CITIES:
                city = GERMAN_CITIES[clean_loc.lower()]
            elif (
                clean_loc
                and len(clean_loc) < 40
                and not any(
                    w in clean_loc.lower()
                    for w in ["deutsch", "jahr", "erfahrung", "straße", "str."]
                )
            ):
                city = clean_loc.title()

        # 2. Look for German postal code + City (e.g. "10115 Berlin" or "D-80331 München")
        if not city:
            plz_match = re.search(
                r"\b(?:D-)?\d{5}\s+([A-ZÄÖÜ][a-zäöüß]+(?:[\s\-][A-ZÄÖÜ][a-zäöüß]+)*)\b",
                cv_text,
            )
            if plz_match:
                candidate_city = plz_match.group(1).strip()
                if candidate_city.lower() in GERMAN_CITIES:
                    city = GERMAN_CITIES[candidate_city.lower()]
                elif candidate_city:
                    city = candidate_city.title()

        # 3. Look for "<City>, Deutschland / Germany"
        if not city:
            country_match = re.search(
                r"\b([A-ZÄÖÜ][a-zäöüß]+(?:[\s\-][A-ZÄÖÜ][a-zäöüß]+)*),\s*(?:Deutschland|Germany|DE)\b",
                cv_text,
                flags=re.IGNORECASE,
            )
            if country_match:
                candidate_city = country_match.group(1).strip()
                if candidate_city.lower() in GERMAN_CITIES:
                    city = GERMAN_CITIES[candidate_city.lower()]
                elif candidate_city:
                    city = candidate_city.title()

        # 4. Look for "in <City>", "im Raum <City>", "Region <City>"
        if not city:
            in_city_match = re.search(
                r"\b(?:in|nach|im\s+raum|großraum|region|near)\s+([A-ZÄÖÜ][a-zäöüß]+(?:[\s\-][A-ZÄÖÜ][a-zäöüß]+)*)\b",
                cv_text,
                flags=re.IGNORECASE,
            )
            if in_city_match:
                candidate_city = in_city_match.group(1).strip().lower()
                if candidate_city in GERMAN_CITIES:
                    city = GERMAN_CITIES[candidate_city]

        # 5. Fallback: Search all known German cities in CV text (sorted by length descending)
        if not city:
            for k in sorted(GERMAN_CITIES.keys(), key=len, reverse=True):
                if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", text_lower):
                    city = GERMAN_CITIES[k]
                    break

        # Radius extraction (default 25 km, clamped between 5 and 200)
        radius_km = 25
        radius_match = re.search(
            r"\b(?:umkreis|radius|distanz|distance|радіус|радиус)\b[^\d\n\r]*?(\d+)\s*(?:km|kilometer)?\b",
            text_lower,
        )
        if not radius_match:
            radius_match = re.search(
                r"\b(\d+)\s*(?:km|kilometer)\s*(?:umkreis|radius|distanz|distance)\b",
                text_lower,
            )
        if radius_match:
            try:
                val = int(radius_match.group(1))
                radius_km = max(5, min(val, 200))
            except ValueError:
                radius_km = 25

        # Desired Job Type
        desired_job_type = "all"
        if re.search(
            r"\b(?:vollzeit\w*|full-time|full\s*time|40\s*(?:h|std)|повна\s*зайнятість|полная\s*занятость)\b",
            text_lower,
        ):
            desired_job_type = "vz"
        elif re.search(
            r"\b(?:teilzeit\w*|part-time|part\s*time|неповна\s*зайнятість|неполная\s*занятость|\d+-\d+\s*std/woche)\b",
            text_lower,
        ):
            desired_job_type = "tz"
        elif re.search(
            r"\b(?:minijob\w*|geringfügig\w*|520\s*€|538\s*€|450\s*€|mini-job|миниджоб)\b",
            text_lower,
        ):
            desired_job_type = "mj"

        # Career Goals
        goals: str | None = None
        goals_match = re.search(
            r"\b(?:karriereziel|berufliches\s+ziel|berufsziel|zielsetzung|ziel|ziele|objective|career\s+goal|target\s+role|gesuchte\s+position|angestrebte\s+position|meta|мета|цель)\s*[:\-–]?\s*([^\n\r]+)",
            cv_text,
            flags=re.IGNORECASE,
        )
        if not goals_match:
            goals_match = re.search(
                r"\b(?:präferenzen|präferenz|preferences|preference)\s*[:\-–]?\s*([^\n\r]+)",
                cv_text,
                flags=re.IGNORECASE,
            )
        if goals_match:
            raw_goal = goals_match.group(1).strip()
            clean_goal = re.sub(r"^[\s:\-–.]+|[\s.]+$", "", raw_goal).strip()
            if clean_goal:
                goals = clean_goal
        elif skills and city:
            goals = f"{skills[0]} in {city}"
        elif skills:
            goals = f"{skills[0]}"

        summary = goals

        # Keywords
        keywords = [s.lower() for s in skills]

        return {
            "skills": skills,
            "experience_years": experience_years,
            "education": education,
            "detected_languages": detected_languages,
            "keywords": keywords,
            "summary": summary,
            "german_level": german_level,
            "city": city,
            "radius_km": radius_km,
            "desired_job_type": desired_job_type,
            "goals": goals,
        }


class AIJobMatcher:
    """Multi-factor AI job matching scoring and multilingual rationale generator."""

    # TODO: It should be real job analyzer, not just algorythm

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
        final_score = round(weighted * 100, 1)
        logger.debug(
            "Job Match scoring [%s | %s]: skills=%.2f, exp=%.2f, german=%.2f, goals=%.2f -> composite=%.1f",
            job_title,
            job_emp,
            skill_score,
            exp_score,
            german_score,
            goals_score,
            final_score,
        )
        return final_score

    async def match_jobs(
        self,
        cv_profile: dict[str, Any] | ExtractedCVProfile,
        user_prefs: dict[str, Any] | Any,
        jobs: list[Any],
        lang: str = "de",
    ) -> list[dict[str, Any]]:
        """Score candidate profile against list of vacancies and generate localized explanations."""
        candidate_skills = (
            cv_profile.skills
            if isinstance(cv_profile, ExtractedCVProfile)
            else (cv_profile.get("skills", []) if isinstance(cv_profile, dict) else [])
        )
        logger.info(
            "Matching %d jobs against candidate profile (%d skills)",
            len(jobs),
            len(candidate_skills),
        )
        results = []
        for job in jobs:
            score = self.calculate_score(cv_profile, user_prefs, job)
            reasons = {
                "en": f"High alignment with your professional qualifications and work experience ({score}% match).",
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
