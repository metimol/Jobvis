"""Multilingual internationalization service for EN, DE, UK, RU translations."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).parent.parent / "locales"


class I18nService:
    """Translation manager supporting German, English, Ukrainian, and Russian."""

    SUPPORTED_LANGS = ["en", "de", "uk", "ru"]
    DEFAULT_LANG = "de"
    _cached_dictionaries: dict[str, dict[str, str]] | None = None

    @classmethod
    def _load_all_locales(cls) -> dict[str, dict[str, str]]:
        """Load and cache translation dictionaries from JSON files."""
        if cls._cached_dictionaries is not None:
            return cls._cached_dictionaries

        dictionaries = {}
        for lang in cls.SUPPORTED_LANGS:
            file_path = LOCALES_DIR / f"{lang}.json"
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        dictionaries[lang] = json.load(f)
                except Exception as e:
                    logger.error("Failed to load locale %s: %s", lang, e)
                    dictionaries[lang] = {}
            else:
                dictionaries[lang] = {}

        cls._cached_dictionaries = dictionaries
        return dictionaries

    @classmethod
    def get_dictionary(cls, lang: str | None = None) -> dict[str, str]:
        """Return the complete dictionary for the requested language with fallback to German."""
        locales = cls._load_all_locales()
        norm_lang = (lang or cls.DEFAULT_LANG).lower().strip()
        if locales.get(norm_lang):
            return locales[norm_lang]
        return locales.get(cls.DEFAULT_LANG, {})

    @classmethod
    def translate(cls, key: str, lang: str | None = None) -> str:
        """Translate a single key for the given language."""
        dict_lang = cls.get_dictionary(lang)
        if key in dict_lang:
            return dict_lang[key]
        dict_de = cls.get_dictionary(cls.DEFAULT_LANG)
        return dict_de.get(key, key)


# Global instance
i18n_service = I18nService
