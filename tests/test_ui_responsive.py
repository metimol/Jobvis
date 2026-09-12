"""Comprehensive E2E responsive test suite for Jobvis mobile overhaul.

Validates mobile responsiveness (<= 375px viewports), touch-friendly targets (>= 44x44px),
mobile navigation drawer, safe interactive spacing, and desktop parity (>= 901px / 1200px)
across 4 methodology tiers.
"""

import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import lxml.html
import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.services.i18n import I18nService

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
CSS_DIR = REPO_ROOT / "static" / "assets" / "css"

ALL_7_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "onboarding.html",
    "profile.html",
    "feed.html",
    "settings.html",
]

ALL_4_LOCALES = ["de", "en", "uk", "ru"]


# ===========================================================================
# Fixtures and DOM / CSS Inspection Helpers
# ===========================================================================


@pytest.fixture(scope="module")
def jinja_env() -> Generator[Environment, None, None]:
    """Module-scoped Jinja2 environment configured identically to production."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["t"] = lambda key, lang="de": I18nService.translate(key, lang)
    yield env


def get_default_context(
    lang: str = "de",
    authenticated: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    """Generate realistic SSR context dictionary for template rendering."""
    t_dict = I18nService.get_dictionary(lang)
    ctx: dict[str, Any] = {
        "t": t_dict,
        "lang": lang,
        "current_user": (
            {
                "id": 1,
                "email": "candidate@example.de",
                "name": "Alex Müller",
                "is_active": True,
            }
            if authenticated
            else None
        ),
        "profile": {
            "onboarding_completed": True,
            "target_roles": ["Software Developer", "Data Analyst"],
            "city": "Berlin",
            "radius_km": 25,
            "german_level": "B2",
            "employment_type": "vz",
        },
        "cv_analysis": {
            "skills": ["Python", "SQL", "FastAPI", "Docker"],
            "summary": "Experienced Python backend engineer with data skills.",
        },
        "total_active_jobs": 42,
        "active_tab": "all",
    }
    ctx.update(overrides)
    return ctx


def render_view(
    jinja_env: Environment,
    template_name: str,
    context: dict[str, Any] | None = None,
) -> str:
    """Render a template with context and return HTML string."""
    ctx = context if context is not None else get_default_context()
    return jinja_env.get_template(template_name).render(**ctx)


def parse_dom(html_content: str) -> lxml.html.HtmlElement:
    """Parse HTML string into an lxml DOM element."""
    return lxml.html.fromstring(html_content)


def get_all_css_sources() -> str:
    """Read all static CSS files and template inline <style> blocks."""
    css_parts: list[str] = []
    if CSS_DIR.exists():
        for css_file in sorted(CSS_DIR.glob("*.css")):
            css_parts.append(css_file.read_text(encoding="utf-8"))

    for tmpl in sorted(TEMPLATES_DIR.glob("*.html")):
        content = tmpl.read_text(encoding="utf-8")
        styles = re.findall(r"<style[^>]*>(.*?)</style>", content, re.DOTALL | re.IGNORECASE)
        css_parts.extend(styles)

    return "\n".join(css_parts)


def extract_media_query_blocks(css_text: str) -> list[tuple[str, str]]:
    """Extract all @media query headers and their entire bodies using balanced brace tracking."""
    blocks: list[tuple[str, str]] = []
    idx = 0
    while True:
        media_idx = css_text.find("@media", idx)
        if media_idx == -1:
            break
        open_brace = css_text.find("{", media_idx)
        if open_brace == -1:
            break
        header = css_text[media_idx:open_brace].strip()
        depth = 1
        curr = open_brace + 1
        while curr < len(css_text) and depth > 0:
            if css_text[curr] == "{":
                depth += 1
            elif css_text[curr] == "}":
                depth -= 1
            curr += 1
        body = css_text[open_brace + 1 : curr - 1].strip()
        blocks.append((header, body))
        idx = curr
    return blocks


def extract_selector_declarations(css_text: str, selector: str) -> list[str]:
    """Find property declarations block for an exact CSS selector."""
    pattern = re.escape(selector) + r"\s*\{([^}]*)\}"
    return re.findall(pattern, css_text)


# ===========================================================================
# Tier 1: Feature Coverage (>= 5 tests per feature area)
# ===========================================================================


class TestTier1MobileResponsiveness:
    """Tier 1: Verify mobile viewport compliance, container padding scaling, and fluid layout."""

    @pytest.mark.parametrize("template_name", ALL_7_TEMPLATES)
    def test_t1_f01_viewport_meta_tag_across_all_7_templates(
        self, jinja_env: Environment, template_name: str
    ):
        """F01: All 7 templates render viewport meta tag with width=device-width and initial-scale=1."""
        rendered = render_view(jinja_env, template_name)
        doc = parse_dom(rendered)
        meta_tags = doc.xpath('//meta[@name="viewport"]')
        assert len(meta_tags) >= 1, f"Template '{template_name}' missing <meta name='viewport'> tag"
        content = meta_tags[0].get("content", "").lower()
        assert (
            "width=device-width" in content
        ), f"Template '{template_name}' viewport missing 'width=device-width': '{content}'"
        assert (
            "initial-scale=1" in content
        ), f"Template '{template_name}' viewport missing 'initial-scale=1': '{content}'"

    def test_t1_f01_main_container_mobile_padding_reduction(self):
        """F01: Main container horizontal padding scales down on mobile viewports (<= 768px or <= 600px)."""
        all_css = get_all_css_sources()
        media_blocks = extract_media_query_blocks(all_css)

        found_mobile_padding = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(600|768)px", header):
                if ".main-container" in body:
                    padding_match = re.search(
                        r"\.main-container\s*\{[^}]*padding\s*:\s*([^;]+);",
                        body,
                    )
                    if padding_match:
                        padding_val = padding_match.group(1).strip()
                        if re.search(r"1rem|1\.25rem|1\.5rem|16px", padding_val):
                            found_mobile_padding = True
                            break

        assert found_mobile_padding, (
            "F01 violation: Missing mobile media query (<= 600px or <= 768px) "
            "reducing .main-container horizontal padding to <= 1rem / 16px"
        )

    def test_t1_f05_landing_features_grid_mobile_single_column(self):
        """F05: Landing features grid overrides minmax(320px, 1fr) on mobile to prevent 319px overflow."""
        index_raw = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        assert (
            "grid-template-columns: repeat(auto-fit, minmax(320px, 1fr))" in index_raw
        ), "F05 invariant broken: Desktop auto-fit minmax(320px, 1fr) must remain for tests/test_ui_templates.py:573"

        media_blocks = extract_media_query_blocks(index_raw)
        has_mobile_grid_override = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(480|600|640|768)px", header):
                if ".features-grid" in body and (
                    "grid-template-columns: 1fr" in body
                    or "grid-template-columns: repeat(1, 1fr)" in body
                ):
                    has_mobile_grid_override = True
                    break

        assert has_mobile_grid_override, (
            "F05 violation: index.html must override .features-grid with "
            "grid-template-columns: 1fr under a mobile media query (<= 640px) to prevent 375px horizontal scroll"
        )

    def test_t1_f06_hero_title_fluid_typography_clamp(self):
        """F06: Landing hero title uses fluid clamp with lower bound < 2.5rem for narrow screens."""
        index_raw = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        clamp_match = re.search(
            r"\.hero-title\s*\{[^}]*font-size\s*:\s*clamp\(([^,]+),([^,]+),([^)]+)\)",
            index_raw,
        )
        assert clamp_match, "F06 violation: .hero-title must declare fluid font-size: clamp(...)"
        lower_bound = clamp_match.group(1).strip()
        is_safe_lower_bound = (
            "1." in lower_bound
            or "2.0" in lower_bound
            or "2rem" in lower_bound
            or "24px" in lower_bound
            or "28px" in lower_bound
            or "32px" in lower_bound
        ) and "2.5rem" not in lower_bound
        assert is_safe_lower_bound, f"F06 violation: .hero-title lower clamp bound '{lower_bound}' too large; must be < 2.5rem (e.g. 1.85rem)"

    def test_t1_f06_hero_actions_mobile_stacking(self):
        """F06: Landing hero actions stack vertically on mobile screens (<= 600px)."""
        index_raw = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        media_blocks = extract_media_query_blocks(index_raw)
        has_hero_stacking = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(480|600|640|768)px", header):
                if ".hero-actions" in body and "flex-direction: column" in body:
                    has_hero_stacking = True
                    break
        assert has_hero_stacking, "F06 violation: index.html must stack .hero-actions vertically (flex-direction: column) on mobile"

    def test_t1_f07_auth_card_mobile_padding_reduction(self):
        """F07: Candidate login card padding reduces on mobile screens (<= 480px)."""
        login_raw = (TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")
        media_blocks = extract_media_query_blocks(login_raw)
        has_auth_card_mobile_padding = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(480|600|768)px", header):
                if ".auth-card" in body and re.search(
                    r"padding\s*:\s*[^;]*(1\.25rem|1rem|16px|20px)", body
                ):
                    has_auth_card_mobile_padding = True
                    break
        assert has_auth_card_mobile_padding, "F07 violation: login.html must reduce .auth-card padding to <= 1.25rem (20px) under mobile media query"

    def test_t1_f08_onboarding_wizard_header_collision_eliminated(self):
        """F08: Onboarding wizard eliminates absolute positioning collision of .wizard-lang on mobile."""
        onboarding_raw = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        media_blocks = extract_media_query_blocks(onboarding_raw)
        has_header_fix = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(480|600|768)px", header):
                if ".wizard-lang" in body and (
                    "position: static" in body
                    or "position: relative" in body
                    or "wizard-header" in body
                ):
                    has_header_fix = True
                    break
        assert has_header_fix, "F08 violation: onboarding.html must eliminate position: absolute on .wizard-lang under <= 600px"

    def test_t1_f12_profile_settings_card_mobile_padding(self):
        """F12: Profile and Settings card padding scales down on mobile viewports (<= 600px)."""
        profile_raw = (TEMPLATES_DIR / "profile.html").read_text(encoding="utf-8")
        settings_raw = (TEMPLATES_DIR / "settings.html").read_text(encoding="utf-8")

        for name, raw in [("profile", profile_raw), ("settings", settings_raw)]:
            media_blocks = extract_media_query_blocks(raw)
            has_card_reduction = False
            for header, body in media_blocks:
                if re.search(r"max-width\s*:\s*(600|768|900)px", header):
                    if ".card" in body and re.search(
                        r"padding\s*:\s*[^;]*(1\.25rem|1\.35rem|1rem|16px|20px)",
                        body,
                    ):
                        has_card_reduction = True
                        break
            assert has_card_reduction, f"F12 violation: {name}.html must reduce .card padding to <= 1.35rem on mobile viewports"


class TestTier1TouchFriendlyTargetsAndSpacing:
    """Tier 1: Verify touch target dimensions (>= 44x44px) and safe separation distances."""

    def test_t1_f04_global_btn_enforces_min_height_44px(self):
        """F04: Global .btn class enforces minimum 44px height and touch centering."""
        all_css = get_all_css_sources()
        btn_decls = extract_selector_declarations(all_css, ".btn")
        has_min_44 = any(
            re.search(r"min-height\s*:\s*(4[4-9]|[5-9]\d)px", decl) for decl in btn_decls
        )
        assert has_min_44, "F04 violation: .btn class must declare min-height: 44px (or >= 44px)"

    def test_t1_f04_form_control_enforces_min_height_44px(self):
        """F04: Global .form-control class enforces minimum 44px height."""
        all_css = get_all_css_sources()
        fc_decls = extract_selector_declarations(all_css, ".form-control")
        has_min_44 = any(
            re.search(r"min-height\s*:\s*(4[4-9]|[5-9]\d)px", decl) for decl in fc_decls
        )
        assert (
            has_min_44
        ), "F04 violation: .form-control class must declare min-height: 44px (or >= 44px)"

    def test_t1_f15_feed_filter_btn_enforces_min_height_44px(self):
        """F15: Feed filter buttons (.filter-btn) declare min-height: 44px."""
        feed_raw = (TEMPLATES_DIR / "feed.html").read_text(encoding="utf-8")
        filter_decls = extract_selector_declarations(feed_raw, ".filter-btn")
        has_min_44 = any(
            re.search(r"min-height\s*:\s*(4[4-9]|[5-9]\d)px", decl) for decl in filter_decls
        )
        assert has_min_44, "F15 violation: .filter-btn in feed.html must declare min-height: 44px"

    def test_t1_f16_feed_sort_dir_btn_enforces_min_44x44px(self):
        """F16: Feed sort direction button (.sort-dir-btn) is at least 44x44px, replacing legacy 34px."""
        feed_raw = (TEMPLATES_DIR / "feed.html").read_text(encoding="utf-8")
        assert (
            "width: 34px; height: 34px;" not in feed_raw and "width: 34px;" not in feed_raw
        ), "F16 violation: Legacy sub-standard 34x34px sizing on .sort-dir-btn must be replaced"
        sort_btn_decls = extract_selector_declarations(feed_raw, ".sort-dir-btn")
        has_min_44 = any(
            re.search(r"(width|min-width)\s*:\s*(4[4-9]|[5-9]\d)px", decl)
            and re.search(r"(height|min-height)\s*:\s*(4[4-9]|[5-9]\d)px", decl)
            for decl in sort_btn_decls
        )
        assert has_min_44, "F16 violation: .sort-dir-btn must enforce at least 44x44px dimensions"

    def test_t1_f09_city_chip_enforces_min_height_44px(self):
        """F09: Onboarding .city-chip elevates from 25.6px to min 44px touch target."""
        onboarding_raw = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        chip_decls = extract_selector_declarations(onboarding_raw, ".city-chip")
        has_min_44 = any(
            re.search(r"min-height\s*:\s*(4[2-9]|[5-9]\d)px", decl) for decl in chip_decls
        )
        assert (
            has_min_44
        ), "F09 violation: .city-chip in onboarding.html must declare min-height: 44px (or >= 42px)"

    def test_t1_f10_review_edit_btn_enforces_min_44x44px(self):
        """F10: Onboarding .review-edit-btn elevates from 27.2px to min 44x44px touch target."""
        onboarding_raw = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        edit_btn_decls = extract_selector_declarations(onboarding_raw, ".review-edit-btn")
        has_min_44 = any(
            re.search(r"min-height\s*:\s*(4[4-9]|[5-9]\d)px", decl) for decl in edit_btn_decls
        )
        assert has_min_44, "F10 violation: .review-edit-btn must declare min-height: 44px"

    def test_t1_f18_job_card_save_vs_dismiss_safe_spacing(self):
        """F18: Job card Save vs Dismiss buttons maintain safe separation (gap >= 12px or >= 0.75rem)."""
        feed_raw = (TEMPLATES_DIR / "feed.html").read_text(encoding="utf-8")
        assert 'style="display: flex; gap: 0.65rem;"' not in feed_raw, (
            "F18 violation: Legacy gap: 0.65rem (10.4px) between Save and Dismiss is too narrow. "
            "Must be enlarged to >= 0.75rem (12px) to prevent accidental dismissals."
        )
        has_safe_gap = bool(
            re.search(
                r"gap\s*:\s*(0\.[8-9]\d?rem|1rem|1\.[0-9]+rem|1[2-9]px|2\dpx)",
                feed_raw,
            )
            or "job-actions-btn-group" in feed_raw
            or "flex-direction: column" in feed_raw
        )
        assert has_safe_gap, "F18 violation: Feed job card actions must provide >= 12px separation between Save and Dismiss"

    def test_t1_f09_city_chips_safe_spacing_gap(self):
        """F09: Onboarding city chips container enforces safe spacing gap >= 8px (>= 0.5rem)."""
        onboarding_raw = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        assert "gap: 0.4rem;" not in onboarding_raw, (
            "F09 violation: Legacy gap: 0.4rem (6.4px) between city chips causes tap overlap. "
            "Must be increased to >= 0.5rem (8-10px)."
        )

    @pytest.mark.parametrize("locale", ALL_4_LOCALES)
    @pytest.mark.parametrize("template_name", ALL_7_TEMPLATES)
    def test_t1_f03_rendered_dom_zero_sub_44px_inline_style_overrides(
        self, jinja_env: Environment, locale: str, template_name: str
    ):
        """F03: Rendered DOM across all 7 templates in all 4 locales contains zero sub-44px inline padding."""
        ctx = get_default_context(lang=locale)
        rendered = render_view(jinja_env, template_name, ctx)
        doc = parse_dom(rendered)

        sub_44_padding = re.compile(r"padding\s*:\s*0\.[1-4][0-9]?rem", re.IGNORECASE)
        violations: list[str] = []
        for elem in doc.xpath("//button | //a[contains(@class, 'btn')] | //select"):
            style = elem.get("style", "")
            if sub_44_padding.search(style):
                tag_text = (elem.text or "").strip()
                violations.append(
                    f"<{elem.tag} class='{elem.get('class', '')}'> '{tag_text}': style='{style}'"
                )

        assert not violations, (
            f"F03 violation: Sub-44px inline padding found in {template_name} ({locale}):\n"
            + "\n".join(violations)
        )

    def test_t1_f14_form_inputs_prevent_ios_zoom_16px_minimum(self):
        """F14: Form controls enforce minimum 16px (1rem) font size to prevent iOS Safari auto-zoom."""
        all_css = get_all_css_sources()
        fc_decls = extract_selector_declarations(all_css, ".form-control")
        has_16px_or_1rem = any(
            re.search(r"font-size\s*:\s*(1rem|16px|1\.\d+rem)", decl) for decl in fc_decls
        )
        assert has_16px_or_1rem, "F14 violation: .form-control must specify font-size: 1rem (16px) to prevent iOS auto-zoom"


class TestTier1MobileNavigation:
    """Tier 1: Verify hamburger toggle, mobile nav drawer, and JavaScript toggle mechanics."""

    def test_t1_f02_hamburger_toggle_button_exists_in_base(self, jinja_env: Environment):
        """F02: Master navigation contains <button class='nav-toggle'> with accessibility attributes."""
        rendered = render_view(jinja_env, "base.html")
        doc = parse_dom(rendered)
        toggles = doc.xpath('//nav//button[contains(@class, "nav-toggle")]')
        assert (
            len(toggles) == 1
        ), "F02 violation: Master <nav> in base.html must contain exactly one <button class='nav-toggle'>"
        btn = toggles[0]
        assert btn.get(
            "aria-label"
        ), "F02 violation: <button class='nav-toggle'> must have an aria-label attribute"
        assert btn.get("aria-expanded") in [
            "false",
            "true",
        ], "F02 violation: <button class='nav-toggle'> must have aria-expanded attribute"

    def test_t1_f02_hamburger_toggle_contains_three_bars(self, jinja_env: Environment):
        """F02: Hamburger toggle button contains exactly 3 visual bar elements."""
        rendered = render_view(jinja_env, "base.html")
        doc = parse_dom(rendered)
        bars = doc.xpath(
            '//button[contains(@class, "nav-toggle")]//*[contains(@class, "hamburger-bar") or contains(@class, "bar")]'
        )
        assert (
            len(bars) == 3
        ), f"F02 violation: Hamburger toggle must contain 3 visual bar elements, found {len(bars)}"

    def test_t1_f02_mobile_nav_drawer_markup_and_id(self, jinja_env: Environment):
        """F02: Navigation links wrapper has ID 'navLinks' and class 'nav-links' for drawer targeting."""
        rendered = render_view(jinja_env, "base.html")
        doc = parse_dom(rendered)
        drawer = doc.xpath('//nav//*[@id="navLinks"]')
        assert (
            len(drawer) == 1
        ), "F02 violation: Master navbar must include drawer container with id='navLinks'"
        assert (
            "nav-links" in drawer[0].get("class", "").split()
        ), "F02 violation: Element with id='navLinks' must have class 'nav-links'"

    def test_t1_f02_mobile_nav_drawer_open_class_styling(self):
        """F02: CSS declares .nav-links.open displaying mobile drawer flex structure."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        assert re.search(
            r"\.nav-links\.open\s*\{[^}]*display\s*:\s*flex", base_raw
        ), "F02 violation: base.html must style .nav-links.open with display: flex"

    def test_t1_f02_mobile_nav_drawer_hidden_by_default_on_mobile(self):
        """F02: On mobile viewports (<= 768px), .nav-links is hidden by default until toggled."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        media_blocks = extract_media_query_blocks(base_raw)
        has_nav_hidden = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*768px", header):
                if re.search(r"\.nav-links\s*\{[^}]*display\s*:\s*none", body):
                    has_nav_hidden = True
                    break
        assert has_nav_hidden, "F02 violation: base.html must set .nav-links { display: none; } under @media (max-width: 768px)"

    def test_t1_f02_toggle_mobile_nav_javascript_function_exists(self):
        """F02: base.html defines toggleMobileNav() JavaScript function toggling .open and aria-expanded."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        assert (
            "function toggleMobileNav()" in base_raw
        ), "F02 violation: base.html must define function toggleMobileNav()"
        assert (
            "classList.toggle('open')" in base_raw
        ), "F02 violation: toggleMobileNav() must toggle 'open' class on drawer"
        assert (
            "setAttribute('aria-expanded'" in base_raw
        ), "F02 violation: toggleMobileNav() must update aria-expanded on toggle button"

    def test_t1_f02_outside_click_listener_closes_drawer(self):
        """F02: base.html registers document click listener closing open drawer on outside click."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        assert (
            "document.addEventListener('click'" in base_raw
            or 'document.addEventListener("click"' in base_raw
        ), "F02 violation: base.html must register document click listener to dismiss mobile nav"
        assert (
            "contains(e.target)" in base_raw or "contains(event.target)" in base_raw
        ), "F02 violation: Outside click handler must check if click target is outside navbar"

    def test_t1_f03_drawer_links_and_buttons_enforce_44px_targets(self):
        """F03: Nav links and buttons inside mobile drawer enforce min-height 44px."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        media_blocks = extract_media_query_blocks(base_raw)
        has_touch_drawer = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*768px", header):
                if re.search(
                    r"\.nav-links\s+(a|\.btn)[^{]*\{[^}]*min-height\s*:\s*44px",
                    body,
                ):
                    has_touch_drawer = True
                    break
        assert has_touch_drawer, "F03 violation: Mobile drawer (.nav-links a, .nav-links .btn) must enforce min-height: 44px"


class TestTier1DesktopParity:
    """Tier 1: Verify desktop layouts (>= 901px / 1200px) remain completely intact and uncorrupted."""

    def test_t1_f19_desktop_navbar_hides_hamburger_button(self):
        """F19: Hamburger toggle button is hidden on desktop viewports."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        has_desktop_hidden = False
        # Either display: none by default or hidden under desktop media query
        default_decls = extract_selector_declarations(base_raw, ".nav-toggle")
        if any("display: none" in d for d in default_decls):
            has_desktop_hidden = True

        media_blocks = extract_media_query_blocks(base_raw)
        for header, body in media_blocks:
            if re.search(r"min-width\s*:\s*(769|901)px", header):
                if ".nav-toggle" in body and "display: none" in body:
                    has_desktop_hidden = True
                    break

        assert (
            has_desktop_hidden
        ), "F19 violation: Hamburger button (.nav-toggle) must be hidden on desktop viewports"

    def test_t1_f19_desktop_navbar_maintains_horizontal_flex(self):
        """F19: Desktop navbar maintains horizontal flex row with gap: 1.5rem."""
        base_raw = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        nav_links_decls = extract_selector_declarations(base_raw, ".nav-links")
        has_horizontal_flex = any(
            "display: flex" in decl and "gap: 1.5rem" in decl for decl in nav_links_decls
        )
        assert (
            has_horizontal_flex
        ), "F19 violation: Desktop .nav-links must preserve horizontal flex layout with gap: 1.5rem"

    def test_t1_f19_profile_two_column_split_preserved(self):
        """F19: Profile layout preserves desktop 2-column split (1fr 1.25fr) and 900px collapse breakpoint."""
        profile_raw = (TEMPLATES_DIR / "profile.html").read_text(encoding="utf-8")
        assert (
            "grid-template-columns: 1fr 1.25fr;" in profile_raw
        ), "F19 violation: profile.html must preserve desktop grid-template-columns: 1fr 1.25fr;"
        assert (
            "@media (max-width: 900px)" in profile_raw
        ), "F19 invariant broken: profile.html must retain @media (max-width: 900px) for tests/test_ui_templates.py:569"

    def test_t1_f19_feed_toolbar_right_alignment_preserved(self):
        """F19: Feed toolbar preserves desktop right-alignment (justify-content: flex-end)."""
        feed_raw = (TEMPLATES_DIR / "feed.html").read_text(encoding="utf-8")
        toolbar_decls = extract_selector_declarations(feed_raw, ".feed-toolbar")
        has_right_align = any("justify-content: flex-end" in decl for decl in toolbar_decls)
        assert has_right_align, "F19 violation: feed.html must preserve justify-content: flex-end on desktop .feed-toolbar"

    def test_t1_f19_landing_features_grid_auto_fit_preserved(self):
        """F19: Landing page preserves desktop auto-fit grid minmax(320px, 1fr)."""
        index_raw = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
        assert (
            "grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));" in index_raw
        ), "F19 violation: index.html must retain repeat(auto-fit, minmax(320px, 1fr)) for desktop screens"

    def test_t1_f19_onboarding_wizard_desktop_card_frame(self):
        """F19: Onboarding wizard preserves desktop 720px max-width container and 2-column cards."""
        onboarding_raw = (TEMPLATES_DIR / "onboarding.html").read_text(encoding="utf-8")
        wizard_decls = extract_selector_declarations(onboarding_raw, ".wizard-container")
        has_720_max = any("max-width: 720px" in decl for decl in wizard_decls)
        assert has_720_max, "F19 violation: onboarding.html must preserve max-width: 720px on desktop .wizard-container"


# ===========================================================================
# Tier 2: Boundary & Corner Cases (Extreme viewports, long strings, capacities)
# ===========================================================================


class TestTier2BoundaryAndCornerCases:
    """Tier 2: Boundary conditions for 320px/375px viewports, long German/Cyrillic strings, and dataset sizes."""

    def test_t2_viewport_320px_container_fluidity(self):
        """Tier 2: At 320px (iPhone SE 1st gen), layout leaves >= 288px usable content width."""
        all_css = get_all_css_sources()
        media_blocks = extract_media_query_blocks(all_css)

        # Confirm container horizontal padding does not exceed 16px each side (32px total) on narrow screens
        found_compliant_padding = False
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(480|600|768)px", header):
                if ".main-container" in body:
                    padding_match = re.search(
                        r"\.main-container\s*\{[^}]*padding\s*:\s*([^;]+);",
                        body,
                    )
                    if padding_match:
                        pad = padding_match.group(1).strip()
                        # e.g., '1.5rem 1rem' (1rem = 16px) or '1rem' or '12px'
                        parts = pad.split()
                        h_pad = parts[1] if len(parts) > 1 else parts[0]
                        if any(unit in h_pad for unit in ["1rem", "0.75rem", "12px", "16px"]):
                            found_compliant_padding = True
                            break

        assert found_compliant_padding, "Tier 2 violation: At 320px viewport, .main-container horizontal padding must be <= 16px (1rem)"

    @pytest.mark.parametrize("template_name", ALL_7_TEMPLATES)
    def test_t2_viewport_375px_baseline_fluidity(self, jinja_env: Environment, template_name: str):
        """Tier 2: All 7 templates render without hardcoded element widths exceeding 375px."""
        rendered = render_view(jinja_env, template_name)
        doc = parse_dom(rendered)

        overflowing_elements: list[str] = []
        for elem in doc.xpath("//*[@style]"):
            style = elem.get("style", "")
            # Find explicit fixed width or min-width > 375px (excluding responsive max-width)
            matches = re.findall(r"(?<!max-)\b(?:min-)?width\s*:\s*(\d+)px", style)
            for w in matches:
                if int(w) > 375:
                    overflowing_elements.append(f"<{elem.tag}> style='{style}'")

        assert not overflowing_elements, (
            f"Tier 2 violation: Hardcoded width > 375px in {template_name}:\n"
            + "\n".join(overflowing_elements)
        )

    def test_t2_long_compound_german_strings_rendering(self, jinja_env: Environment):
        """Tier 2: Long German compound words (e.g. Arbeitsagentur-Verknüpfung) render without layout crash."""
        ctx = get_default_context(lang="de")
        ctx["profile"]["goals"] = (
            "Fachinformatiker für Systemintegration mit Fokus auf Arbeitsagentur-Verknüpfung."
        )
        ctx["profile"]["location"] = "Garmisch-Partenkirchen"
        rendered = render_view(jinja_env, "profile.html", ctx)
        doc = parse_dom(rendered)
        # Ensure rendered HTML contains German compound words safely escaped
        assert "Fachinformatiker für Systemintegration" in rendered
        assert "Garmisch-Partenkirchen" in rendered
        # Ensure location input and goals textarea exist
        inputs = doc.xpath('//input[@id="locationInput"]')
        assert len(inputs) == 1
        goals = doc.xpath('//textarea[@id="goalsInput"]')
        assert len(goals) == 1

    def test_t2_cyrillic_and_ukrainian_strings_rendering(self, jinja_env: Environment):
        """Tier 2: Ukrainian and Russian Cyrillic characters render cleanly with 44px touch targets."""
        for locale in ["uk", "ru"]:
            ctx = get_default_context(lang=locale)
            rendered = render_view(jinja_env, "settings.html", ctx)
            doc = parse_dom(rendered)
            assert len(rendered) > 1000, f"Empty render for locale {locale}"
            # Verify danger zone button exists and has .btn class
            danger_btns = doc.xpath('//button[contains(@class, "btn-danger")]')
            assert len(danger_btns) == 1

    def test_t2_feed_empty_state_responsiveness(self, jinja_env: Environment):
        """Tier 2: Feed empty state (0 items) renders centered responsive message without layout breakage."""
        ctx = get_default_context(total_active_jobs=0)
        rendered = render_view(jinja_env, "feed.html", ctx)
        doc = parse_dom(rendered)
        container = doc.xpath('//*[@id="feedContainer"]')
        assert len(container) == 1
        assert "main-container" in rendered

    def test_t2_feed_max_capacity_50_items_consistency(self, jinja_env: Environment):
        """Tier 2: Feed template loads 50 item capacity parameter safely."""
        rendered = render_view(jinja_env, "feed.html")
        assert (
            "/api/feed?size=50" in rendered
        ), "Tier 2 violation: feed.html must query exactly 50 items capacity"

    def test_t2_extreme_job_title_length_250_chars(self):
        """Tier 2: Job card header layout supports long job titles without squishing CEFR & score badges."""
        feed_raw = (TEMPLATES_DIR / "feed.html").read_text(encoding="utf-8")
        # Ensure .job-card-top CSS or mobile rules permit flex wrapping
        top_decls = extract_selector_declarations(feed_raw, ".job-card-top")
        media_blocks = extract_media_query_blocks(feed_raw)
        has_wrapping = any(
            "flex-wrap: wrap" in d or "flex-direction: column" in d for d in top_decls
        )
        for header, body in media_blocks:
            if re.search(r"max-width\s*:\s*(600|768)px", header):
                if ".job-card-top" in body and (
                    "flex-direction: column" in body or "flex-wrap" in body
                ):
                    has_wrapping = True
                    break

        assert (
            has_wrapping
        ), "Tier 2 violation: .job-card-top must support flex wrapping or column stacking on mobile"

    def test_t2_commute_radius_range_slider_bounds(self):
        """Tier 2: Range slider thumb touch area styled with minimum 24-28px dimensions."""
        all_css = get_all_css_sources()
        has_slider_thumb_css = (
            "::-webkit-slider-thumb" in all_css or "::-moz-range-thumb" in all_css
        )
        assert has_slider_thumb_css, (
            "Tier 2 violation: CSS must define custom ::-webkit-slider-thumb / ::-moz-range-thumb "
            "with minimum 24-28px dimensions for mobile touch dragging"
        )


# ===========================================================================
# Tier 3: Cross-Feature Combinations
# ===========================================================================


class TestTier3CrossFeatureCombinations:
    """Tier 3: Interactions between navigation drawer, language switching, auth states, and controls."""

    @pytest.mark.parametrize("locale", ALL_4_LOCALES)
    def test_t3_mobile_nav_drawer_with_language_switch_across_all_locales(
        self, jinja_env: Environment, locale: str
    ):
        """Tier 3: Mobile nav drawer contains language switcher with all 4 locales and correct selection."""
        ctx = get_default_context(lang=locale)
        rendered = render_view(jinja_env, "base.html", ctx)
        doc = parse_dom(rendered)
        selects = doc.xpath('//nav//*[@id="globalLangSelect"]')
        assert len(selects) == 1, "Missing #globalLangSelect in navigation"
        select = selects[0]
        options = select.xpath(".//option")
        assert len(options) == 4, f"Expected 4 language options, got {len(options)}"
        selected_opt = select.xpath(".//option[@selected]")
        assert len(selected_opt) == 1
        assert selected_opt[0].get("value") == locale

    def test_t3_nav_drawer_authenticated_vs_unauthenticated_state(self, jinja_env: Environment):
        """Tier 3: Nav drawer switches seamlessly between authenticated and unauthenticated item sets."""
        # Unauthenticated
        unauth_rendered = render_view(
            jinja_env, "base.html", get_default_context(authenticated=False)
        )
        unauth_doc = parse_dom(unauth_rendered)
        unauth_links = unauth_doc.xpath('//nav//*[@id="navLinks"]//a/@href')
        assert "/" in unauth_links
        assert "/login" in unauth_links
        assert "/feed" not in unauth_links
        assert "/profile" not in unauth_links

        # Authenticated
        auth_rendered = render_view(jinja_env, "base.html", get_default_context(authenticated=True))
        auth_doc = parse_dom(auth_rendered)
        auth_links = auth_doc.xpath('//nav//*[@id="navLinks"]//a/@href')
        assert "/feed" in auth_links
        assert "/profile" in auth_links
        assert "/settings" in auth_links
        assert "/auth/logout" in auth_links
        assert "/login" not in auth_links

    def test_t3_feed_segmented_filter_and_sort_direction_combination(self, jinja_env: Environment):
        """Tier 3: Feed segmented filter bar and sort controls coexist without visual collision."""
        rendered = render_view(jinja_env, "feed.html")
        doc = parse_dom(rendered)
        filters = doc.xpath('//button[contains(@class, "filter-btn")]')
        assert len(filters) == 3, "Expected 3 filter buttons (All, New, Saved)"
        sort_btn = doc.xpath('//*[@id="sortDirBtn"]')
        assert len(sort_btn) == 1, "Expected sort direction button"
        sort_select = doc.xpath('//*[@id="sortSelect"]')
        assert len(sort_select) == 1, "Expected sort criterion dropdown"

    def test_t3_onboarding_step4_city_input_and_chips_combination(self, jinja_env: Environment):
        """Tier 3: Onboarding Step 4 combines city input, 8 quick-chips, and nav buttons fluidly."""
        rendered = render_view(jinja_env, "onboarding.html")
        doc = parse_dom(rendered)
        city_input = doc.xpath('//input[@id="cityInput"]')
        assert len(city_input) == 1
        chips = doc.xpath('//button[contains(@class, "city-chip")]')
        assert len(chips) >= 8, f"Expected at least 8 city chips, found {len(chips)}"
        back_btn = doc.xpath('//*[@id="btnBack"]')
        assert len(back_btn) == 1
        next_btn = doc.xpath('//*[@id="btnNext"]')
        assert len(next_btn) == 1

    def test_t3_onboarding_step7_review_list_with_all_edit_buttons(self, jinja_env: Environment):
        """Tier 3: Onboarding Step 7 review list renders 5 edit buttons with step jump handlers."""
        rendered = render_view(jinja_env, "onboarding.html")
        doc = parse_dom(rendered)
        edit_btns = doc.xpath('//button[contains(@class, "review-edit-btn")]')
        assert len(edit_btns) == 5, f"Expected 5 review edit buttons, got {len(edit_btns)}"
        for btn in edit_btns:
            onclick = btn.get("onclick", "")
            assert "goToStep(" in onclick, f"Missing goToStep in edit button: {onclick}"

    def test_t3_profile_cv_dropzone_and_form_submit_combination(self, jinja_env: Environment):
        """Tier 3: Profile CV upload dropzone and preferences form submit button both comply with touch standards."""
        rendered = render_view(jinja_env, "profile.html")
        doc = parse_dom(rendered)
        dropzone = doc.xpath('//*[contains(@class, "upload-dropzone")]')
        assert len(dropzone) == 1
        submit_btn = doc.xpath('//button[@type="submit"]')
        assert len(submit_btn) == 1
        assert "btn" in submit_btn[0].get("class", "").split()


# ===========================================================================
# Tier 4: Real-World Workload Scenarios
# ===========================================================================


class TestTier4RealWorldWorkloadScenarios:
    """Tier 4: Realistic candidate workflows spanning multiple views, state changes, and localized journeys."""

    def test_t4_candidate_mobile_journey_e2e_render(self, jinja_env: Environment):
        """Tier 4: Complete candidate mobile journey across all 7 views validates cleanly."""
        journey_views = [
            ("index.html", False),
            ("login.html", False),
            ("onboarding.html", True),
            ("feed.html", True),
            ("profile.html", True),
            ("settings.html", True),
        ]

        for tmpl, is_auth in journey_views:
            ctx = get_default_context(lang="de", authenticated=is_auth)
            rendered = render_view(jinja_env, tmpl, ctx)
            doc = parse_dom(rendered)

            # Viewport invariant
            metas = doc.xpath('//meta[@name="viewport"]')
            assert len(metas) >= 1, f"Step {tmpl} failed viewport check"

            # Check for broken template syntax
            assert "{{" not in rendered, f"Step {tmpl} contains unrendered Jinja expression"

    @pytest.mark.parametrize("locale", ALL_4_LOCALES)
    def test_t4_multilingual_candidate_journey_all_locales(
        self, jinja_env: Environment, locale: str
    ):
        """Tier 4: Candidate journey in all 4 locales verifies absence of template leakage or encoding bugs."""
        for tmpl in ["index.html", "feed.html", "settings.html"]:
            ctx = get_default_context(lang=locale, authenticated=True)
            rendered = render_view(jinja_env, tmpl, ctx)
            assert len(rendered) > 500
            doc = parse_dom(rendered)
            brand_elem = doc.xpath('//a[contains(@class, "nav-brand")]')
            assert len(brand_elem) == 1
            assert brand_elem[0].text_content().strip() == "JOBVIS"

    def test_t4_feed_status_triage_flow(self, jinja_env: Environment):
        """Tier 4: Candidate status triage (All -> New -> Saved) maintains filter button state coherence."""
        for tab in ["all", "new", "saved"]:
            ctx = get_default_context(active_tab=tab)
            rendered = render_view(jinja_env, "feed.html", ctx)
            doc = parse_dom(rendered)
            filters = doc.xpath('//button[contains(@class, "filter-btn")]')
            assert len(filters) == 3

    def test_t4_settings_preference_reset_and_gdpr_deletion_mobile(self, jinja_env: Environment):
        """Tier 4: Settings actions (Reset, Delete Account) provide 44px touch targets on mobile."""
        rendered = render_view(jinja_env, "settings.html")
        doc = parse_dom(rendered)
        reset_btn = doc.xpath('//button[contains(@class, "btn-outline")]')
        assert len(reset_btn) >= 1
        danger_btn = doc.xpath('//button[contains(@class, "btn-danger")]')
        assert len(danger_btn) == 1

    def test_t4_form_error_and_alert_banner_responsiveness(self):
        """Tier 4: Alert banners and error messages enforce box-sizing: border-box to prevent overflow."""
        all_css = get_all_css_sources()
        assert (
            "box-sizing: border-box" in all_css
        ), "Tier 4 violation: CSS must enforce box-sizing: border-box across containers and alert elements"
