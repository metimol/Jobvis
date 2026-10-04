"""Sanitization helpers for untrusted text received from external APIs (e.g. BA Jobsuche).

Everything coming from third-party APIs is treated as hostile input:
- HTML / script markup is stripped (defense-in-depth against stored XSS).
- Control, format (zero-width, bidi-override) and private-use characters are removed
  (prevents hidden-text tricks, e.g. invisible prompt-injection payloads for the LLM matcher).
- Whitespace is normalized and the length is capped (protects DB columns and LLM context).
"""

import re
import unicodedata
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

DEFAULT_MAX_LENGTH = 20_000
MAX_URL_LENGTH = 1024
MAX_REF_NR_LENGTH = 100

# Tags whose *content* must be discarded entirely, not only the markup.
_DROP_CONTENT_TAGS = frozenset(
    {"script", "style", "iframe", "object", "embed", "noscript", "template", "svg", "math", "head"}
)
# Tags that represent a line break in plain text.
_BLOCK_TAGS = frozenset(
    {"br", "p", "div", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
)
# Characters kept even though they belong to the "C*" unicode categories.
_ALLOWED_CONTROL_CHARS = frozenset({"\n", "\t"})
# Unicode categories to strip: control, format, surrogate, private use, unassigned.
_STRIPPED_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn"})

_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_ANY_WS_RE = re.compile(r"\s+")
_TAG_HINT_RE = re.compile(r"<[a-zA-Z/!]|&[#a-zA-Z0-9]+;")
_REF_NR_RE = re.compile(r"^[A-Za-z0-9._~+/=-]+$")


class _HTMLTextExtractor(HTMLParser):
    """Collects visible text from an HTML fragment, dropping dangerous elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_CONTENT_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _DROP_CONTENT_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS and tag != "li":
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _strip_html_once(value: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:
        # Malformed markup: fall back to removing anything that looks like a tag.
        return re.sub(r"<[^>]*>", "", value)
    return parser.get_text()


def _strip_html(value: str, max_passes: int = 3) -> str:
    # Repeat so entity-encoded markup (e.g. "&lt;script&gt;") cannot survive as a decoded tag.
    for _ in range(max_passes):
        if not _TAG_HINT_RE.search(value):
            break
        value = _strip_html_once(value)
    return value


def _strip_invisible_chars(value: str) -> str:
    return "".join(
        ch
        for ch in value
        if ch in _ALLOWED_CONTROL_CHARS or unicodedata.category(ch) not in _STRIPPED_CATEGORIES
    )


def sanitize_text(
    value: Any,
    *,
    multiline: bool = False,
    max_length: int | None = DEFAULT_MAX_LENGTH,
) -> str | None:
    """Convert an untrusted value into clean plain text.

    Args:
        value: Raw value from an external source.
        multiline: Preserve line breaks (descriptions). Otherwise collapse to one line.
        max_length: Maximum resulting length (``None`` disables truncation).

    Returns:
        Sanitized text, or ``None`` if nothing meaningful remains.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _strip_html(text)
    text = _strip_invisible_chars(text)

    if multiline:
        lines = [_HORIZONTAL_WS_RE.sub(" ", line).strip() for line in text.split("\n")]
        text = _MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()
    else:
        text = _ANY_WS_RE.sub(" ", text).strip()

    if max_length is not None and len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"

    return text or None


def sanitize_url(value: Any, max_length: int = MAX_URL_LENGTH) -> str | None:
    """Return the URL only if it is a well-formed absolute http(s) URL, else ``None``."""
    text = sanitize_text(value, max_length=None)
    if not text or len(text) > max_length or " " in text:
        return None
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return None
    return text


def sanitize_ref_nr(value: Any) -> str:
    """Return a BA reference number restricted to a safe character set, or ``""``."""
    text = sanitize_text(value, max_length=None)
    if not text or len(text) > MAX_REF_NR_LENGTH or not _REF_NR_RE.match(text):
        return ""
    return text


def sanitize_structure(value: Any, *, max_depth: int = 5) -> Any:
    """Recursively sanitize string leaves of nested dicts/lists (e.g. contact / address data)."""
    if max_depth <= 0:
        return None
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, dict):
        return {
            sanitize_text(k, max_length=200) or "": sanitize_structure(v, max_depth=max_depth - 1)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_structure(v, max_depth=max_depth - 1) for v in value]
    if value is None or isinstance(value, bool | int | float):
        return value
    return sanitize_text(value)
