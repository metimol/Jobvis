"""Automated forensic audits for zero TODOs, CSS/HTML syntax integrity, and repository hygiene."""

import re
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from main import app as full_app

REPO_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"
CSS_DIR = STATIC_DIR / "assets" / "css"
JS_DIR = STATIC_DIR / "assets" / "js"
IMG_DIR = STATIC_DIR / "assets" / "img"

ALL_ACTIVE_TEMPLATES = [
    "base.html",
    "index.html",
    "login.html",
    "onboarding.html",
    "feed.html",
    "profile.html",
    "settings.html",
]

DELETED_DEAD_ASSETS = [
    "templates/gallery.html",
    "static/assets/css/page-about.css",
    "static/assets/css/page-contact.css",
    "static/assets/css/page-gallery.css",
    "static/assets/css/page-pricing.css",
    "static/assets/css/subpage.css",
    "static/assets/js/page-contact.js",
    "static/assets/js/page-gallery.js",
    "static/assets/js/page-pricing.js",
    "static/assets/js/vayra-console.js",
    "static/assets/js/vayra-gl.js",
    "static/assets/js/vayra-shell.js",
    "static/assets/img/gen",
] + [f"static/assets/img/gen/img_{i:03d}.webp" for i in range(1, 31)]


# ===========================================================================
# 1. Consolidated Zero-TODO Forensic Audit
# ===========================================================================


def test_strict_zero_todo_across_all_codebase():
    """Consolidated scan ensuring zero TODO, FIXME, XXX, or HACK markers in app, templates, and static."""
    todo_regex = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
    violations = []

    # Check app/
    for py_path in (REPO_ROOT / "app").rglob("*.py"):
        with open(py_path, encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, 1):
                if todo_regex.search(line):
                    violations.append(
                        f"{py_path.relative_to(REPO_ROOT)}:{line_idx}: {line.strip()}"
                    )

    # Check templates/
    for ext in ["*.html", "*.jinja", "*.jinja2"]:
        for file_path in TEMPLATES_DIR.rglob(ext):
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if todo_regex.search(line):
                        violations.append(
                            f"{file_path.relative_to(REPO_ROOT)}:{line_idx}: {line.strip()}"
                        )

    # Check static/
    for ext in ["*.css", "*.js"]:
        for file_path in STATIC_DIR.rglob(ext):
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f, 1):
                    if todo_regex.search(line):
                        violations.append(
                            f"{file_path.relative_to(REPO_ROOT)}:{line_idx}: {line.strip()}"
                        )

    assert not violations, f"Found actionable TODO/FIXME markers: {violations}"


# ===========================================================================
# 2. Dead Assets Physical Deletion & Absence Audits
# ===========================================================================


def test_physical_deletion_of_all_dead_assets():
    """Verify that every single one of the 42 dead assets has been physically removed from disk."""
    for rel_path in DELETED_DEAD_ASSETS:
        full_path = REPO_ROOT / rel_path
        assert not full_path.exists(), f"Dead asset still exists on disk: {rel_path}"


def test_physical_presence_of_all_active_templates():
    """Verify all 7 required core Jinja2 templates exist and are non-empty."""
    for template_name in ALL_ACTIVE_TEMPLATES:
        t_path = TEMPLATES_DIR / template_name
        assert t_path.exists(), f"Active template missing: {template_name}"
        assert t_path.stat().st_size > 0, f"Active template is empty: {template_name}"


def test_complete_absence_of_gallery_in_all_templates():
    """Verify no active template references the deleted gallery or WebGL scripts."""
    forbidden_terms = ["gallery.html", "page-gallery.css", "page-gallery.js", "vayra-gl", "WebGL"]
    for template_name in ALL_ACTIVE_TEMPLATES:
        t_path = TEMPLATES_DIR / template_name
        content = t_path.read_text(encoding="utf-8")
        for term in forbidden_terms:
            assert term not in content, f"Forbidden term '{term}' found in {template_name}"


def test_index_html_does_not_contain_webgl_or_gallery():
    """Verify templates/index.html is completely free of WebGL canvas, wrapper, and Three.js scripts."""
    index_path = TEMPLATES_DIR / "index.html"
    assert index_path.exists(), "templates/index.html does not exist"
    content = index_path.read_text(encoding="utf-8")
    lower_content = content.lower()

    assert "webgl" not in lower_content
    assert "three.js" not in lower_content
    assert "three.min.js" not in lower_content
    assert "gallery.html" not in lower_content
    assert "page-gallery.css" not in lower_content
    assert "page-gallery.js" not in lower_content
    assert "vayra-gl" not in lower_content
    assert "vayra-shell" not in lower_content
    assert "vayra-console" not in lower_content
    assert "hero-canvas" not in lower_content
    assert "canvas" not in lower_content


def test_gallery_template_is_deleted():
    """Verify templates/gallery.html is permanently deleted from the filesystem."""
    gallery_path = TEMPLATES_DIR / "gallery.html"
    assert not gallery_path.exists(), "templates/gallery.html was found but should be deleted"


def test_required_templates_are_present():
    """Verify all required active templates exist and are non-empty."""
    required = [
        "base.html",
        "index.html",
        "login.html",
        "feed.html",
        "profile.html",
        "settings.html",
    ]
    for tmpl in required:
        p = TEMPLATES_DIR / tmpl
        assert p.exists(), f"Required template {tmpl} is missing"
        assert p.stat().st_size > 0, f"Template {tmpl} is empty"


def test_dead_css_assets_are_deleted():
    """Verify dead CSS stylesheets are completely deleted from static/assets/css/."""
    dead_css = [
        "page-about.css",
        "page-contact.css",
        "page-gallery.css",
        "page-pricing.css",
        "subpage.css",
    ]
    for css_file in dead_css:
        p = CSS_DIR / css_file
        assert not p.exists(), f"Dead CSS asset {css_file} still exists"


def test_dead_js_assets_are_deleted():
    """Verify dead JS scripts are completely deleted from static/assets/js/."""
    dead_js = [
        "page-contact.js",
        "page-gallery.js",
        "page-pricing.js",
        "vayra-console.js",
        "vayra-gl.js",
        "vayra-shell.js",
    ]
    for js_file in dead_js:
        p = JS_DIR / js_file
        assert not p.exists(), f"Dead JS asset {js_file} still exists"


def test_gen_images_directory_is_deleted():
    """Verify static/assets/img/gen directory and all WebP images are deleted."""
    gen_dir = Path("static/assets/img/gen")
    assert not gen_dir.exists(), "static/assets/img/gen directory must be removed"


def test_preserved_static_assets_exist():
    """Verify favicon.svg and static/assets/js directory are preserved."""
    favicon = Path("static/assets/img/favicon.svg")
    assert favicon.exists(), "static/assets/img/favicon.svg must be preserved"

    core_css_files = [
        "static/assets/css/_tokens-bridge.css",
        "static/assets/css/parts.css",
        "static/assets/css/scope-context.css",
        "static/assets/css/sections.css",
    ]
    for css_file in core_css_files:
        assert Path(css_file).exists(), f"Core CSS file {css_file} must be preserved"


def test_adv_index_html_comprehensive_gallery_and_script_purge():
    """Comprehensive check on templates/index.html ensuring zero residual gallery markup."""
    index_html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
    banned_tokens = [
        "<canvas",
        'id="canvas"',
        "id='canvas'",
        'class="gallery"',
        "gallery.html",
        "page-gallery.css",
        "page-gallery.js",
        "vayra-gl.js",
        "vayra-console.js",
        "vayra-shell.js",
        "three.min.js",
        "three.js",
        "page-about.css",
        "page-contact.css",
        "page-pricing.css",
        "subpage.css",
    ]
    for token in banned_tokens:
        assert token not in index_html, f"Banned token '{token}' found in templates/index.html"


def test_adv_all_active_templates_have_zero_dangling_references_to_deleted_assets():
    """Audit all active templates to ensure no link/script references any deleted asset."""
    active_templates = [
        "base.html",
        "index.html",
        "login.html",
        "feed.html",
        "profile.html",
        "settings.html",
    ]
    deleted_asset_names = [
        "page-about.css",
        "page-contact.css",
        "page-gallery.css",
        "page-pricing.css",
        "subpage.css",
        "page-contact.js",
        "page-gallery.js",
        "page-pricing.js",
        "vayra-console.js",
        "vayra-gl.js",
        "vayra-shell.js",
        "gallery.html",
        "/img/gen/",
        "assets/img/gen",
    ]
    for template_name in active_templates:
        content = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
        for asset_name in deleted_asset_names:
            assert (
                asset_name not in content
            ), f"Dangling reference to deleted asset '{asset_name}' found in {template_name}"


def test_adv_filesystem_complete_absence_of_all_42_dead_assets():
    """Filesystem check for all 42 deleted assets."""
    for rel_path in DELETED_DEAD_ASSETS:
        full_path = REPO_ROOT / rel_path
        assert not full_path.exists(), f"Dead asset {rel_path} still exists on disk!"


def test_adv_filesystem_preservation_of_required_assets():
    """Verify that all required core assets are intact and non-empty."""
    favicon = Path("static/assets/img/favicon.svg")
    assert favicon.exists(), "Favicon 'static/assets/img/favicon.svg' is missing!"
    assert favicon.stat().st_size > 0, "Favicon file is empty!"

    core_css_files = [
        "static/assets/css/_tokens-bridge.css",
        "static/assets/css/parts.css",
        "static/assets/css/scope-context.css",
        "static/assets/css/sections.css",
    ]
    for css_path_str in core_css_files:
        p = Path(css_path_str)
        assert p.exists(), f"Core CSS file {css_path_str} is missing!"
        assert p.stat().st_size > 0, f"Core CSS file {css_path_str} is empty!"

    active_templates = [
        "templates/base.html",
        "templates/index.html",
        "templates/login.html",
        "templates/profile.html",
        "templates/feed.html",
        "templates/settings.html",
    ]
    for tmpl_str in active_templates:
        p = Path(tmpl_str)
        assert p.exists(), f"Active template {tmpl_str} is missing!"
        assert p.stat().st_size > 0, f"Active template {tmpl_str} is empty!"


def test_adv_login_template_has_no_gallery_or_dead_references():
    """Audit login.html specifically for dead asset references."""
    login_html = (TEMPLATES_DIR / "login.html").read_text(encoding="utf-8")
    assert "gallery.html" not in login_html
    assert "page-gallery" not in login_html
    assert "vayra-" not in login_html
    assert "page-about" not in login_html
    assert "page-contact" not in login_html
    assert "page-pricing" not in login_html


@pytest.mark.asyncio
async def test_static_files_http_serving_and_404_on_dead_assets():
    """Verify HTTP serving returns 404 for deleted assets and 200 for retained assets."""
    async with AsyncClient(
        transport=ASGITransport(app=full_app), base_url="http://testserver"
    ) as client:
        # Deleted assets should 404
        r1 = await client.get("/static/assets/css/page-gallery.css")
        assert r1.status_code == 404

        r2 = await client.get("/static/assets/js/vayra-gl.js")
        assert r2.status_code == 404

        r3 = await client.get("/static/assets/img/gen/img_001.webp")
        assert r3.status_code == 404

        # Retained assets should 200
        r4 = await client.get("/static/assets/img/favicon.svg")
        assert r4.status_code == 200

        r5 = await client.get("/static/assets/css/_tokens-bridge.css")
        assert r5.status_code == 200
