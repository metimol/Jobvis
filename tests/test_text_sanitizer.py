"""Tests for sanitization of untrusted text received from the Arbeitsagentur API."""

import pytest

from app.schemas.job import BADetailedJob, BAJobListing
from app.utils.text_sanitizer import (
    sanitize_ref_nr,
    sanitize_structure,
    sanitize_text,
    sanitize_url,
)


# ===========================================================================
# 1. sanitize_text
# ===========================================================================
@pytest.mark.parametrize("value", [None, "", "   ", "<p></p>", "\u200b\u200b"])
def test_sanitize_text_empty_values_return_none(value):
    """Verify values with no meaningful content collapse to None."""
    assert sanitize_text(value) is None


def test_sanitize_text_strips_dangerous_markup_and_content():
    """Adversarial: script/style content is removed entirely, other tags keep their text."""
    raw = "<script>alert(1)</script><style>p{}</style><b>Hello</b> <i>World</i>"
    assert sanitize_text(raw) == "Hello World"


def test_sanitize_text_entity_encoded_markup_cannot_survive():
    """Adversarial: entity-encoded tags must not decode into live markup."""
    result = sanitize_text("&lt;script&gt;alert(1)&lt;/script&gt;Text")
    assert result is not None
    assert "<script" not in result


def test_sanitize_text_removes_invisible_characters():
    """Adversarial: zero-width, bidi-override and control characters are stripped."""
    assert sanitize_text("Py\u200bthon\u202e Dev\x00eloper") == "Python Developer"


def test_sanitize_text_multiline_preserves_structure():
    """Verify multiline mode keeps line breaks / list items and collapses excess blank lines."""
    raw = "<p>Intro</p><ul><li>One</li><li>Two</li></ul>\n\n\n\nEnd"
    result = sanitize_text(raw, multiline=True)
    assert result == "Intro\n\n- One\n- Two\n\nEnd"


def test_sanitize_text_single_line_collapses_whitespace():
    """Verify default mode collapses all whitespace into single spaces."""
    assert sanitize_text("  a\n\tb\r\n  c  ") == "a b c"


def test_sanitize_text_truncates_with_ellipsis():
    """Verify max_length caps output and marks truncation."""
    result = sanitize_text("x" * 50, max_length=10)
    assert result == "x" * 9 + "…"
    assert len(result) == 10


# ===========================================================================
# 2. sanitize_url / sanitize_ref_nr / sanitize_structure
# ===========================================================================
@pytest.mark.parametrize(
    "url",
    ["https://www.arbeitsagentur.de/jobsuche/jobdetail/1", "http://example.com/a?b=c"],
)
def test_sanitize_url_accepts_http_urls(url):
    assert sanitize_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:text/html,<script>alert(1)</script>",
        "ftp://example.com/file",
        "//example.com/no-scheme",
        "https://exa mple.com",
        "https://" + "a" * 2000 + ".com",
        None,
    ],
)
def test_sanitize_url_rejects_unsafe_urls(url):
    """Adversarial: non-http(s), relative, malformed or oversized URLs are rejected."""
    assert sanitize_url(url) is None


@pytest.mark.parametrize("ref_nr", ["10000-1198765432-S", "abc_DEF.1~2+3/4="])
def test_sanitize_ref_nr_accepts_safe_values(ref_nr):
    assert sanitize_ref_nr(ref_nr) == ref_nr


@pytest.mark.parametrize(
    "ref_nr", ["10000/MÜNCHEN SPEC#1", "../../etc/passwd?x", "a" * 101, "", None]
)
def test_sanitize_ref_nr_rejects_unsafe_values(ref_nr):
    """Adversarial: ref_nr is used in URL paths, so anything outside the safe charset is dropped."""
    assert sanitize_ref_nr(ref_nr) == ""


def test_sanitize_structure_cleans_nested_leaves():
    """Verify nested dict/list string leaves are sanitized while primitives are preserved."""
    raw = {
        "<b>name</b>": "<script>x</script>Jane",
        "phones": ["<i>123</i>", 42, None, True],
        "nested": {"city": "Ber\u200blin"},
    }
    assert sanitize_structure(raw) == {
        "name": "Jane",
        "phones": ["123", 42, None, True],
        "nested": {"city": "Berlin"},
    }


def test_sanitize_structure_depth_limit():
    """Verify deeply nested payloads are cut off instead of recursing indefinitely."""
    assert sanitize_structure({"a": {"b": "c"}}, max_depth=2) == {"a": {"b": None}}


# ===========================================================================
# 3. Schema integration (BA payload parsing)
# ===========================================================================
def test_ba_job_listing_sanitizes_api_payload():
    """Verify BAJobListing.from_api_dict applies sanitization to untrusted fields."""
    listing = BAJobListing.from_api_dict(
        {
            "refnr": "10000-1-S",
            "titel": "<b>Dev</b>\u200b<script>alert(1)</script>",
            "arbeitgeber": "<i>ACME</i>",
            "stellenangebotsBeschreibung": "<p>Line 1</p><p>Line 2</p>",
            "externeUrl": "javascript:alert(1)",
        }
    )
    assert listing.ref_nr == "10000-1-S"
    assert listing.title == "Dev"
    assert listing.employer == "ACME"
    assert listing.description == "Line 1\n\nLine 2"
    assert listing.external_url is None or listing.external_url.startswith("https://")


def test_ba_detailed_job_sanitizes_api_payload():
    """Verify BADetailedJob.from_api_dict sanitizes description, contact and rejects bad URLs."""
    detail = BADetailedJob.from_api_dict(
        {
            "refnr": "10000-1-S",
            "titel": "Dev",
            "stellenangebotsBeschreibung": "<div>Hello<br>World</div><script>x</script>",
            "kontakt": {"name": "<b>Jane</b>"},
            "externeUrl": "data:text/html,boom",
        }
    )
    assert detail.description == "Hello\nWorld"
    assert detail.contact == {"name": "Jane"}
    assert detail.external_url is None
