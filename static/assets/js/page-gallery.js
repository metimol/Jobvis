/* ============================================================================
 * Vayra — console behaviour for gallery.html
 *
 * Page specific: the shared engine (vayra-shell.js) is used by all four subpages, so
 * nothing is added to it from here.
 *
 * What this file makes real: the filter tabs actually filter the list, the selected
 * render's metadata actually appears in the viewer, and the prompt is actually copied.
 *
 * All of the data lives in the markup, in the data attributes on each row. You can edit
 * or delete rows in the HTML without opening this file.
 * ========================================================================= */
(function () {
  'use strict';

  var list = document.getElementById('vy-log-list');
  var viewer = document.querySelector('.vy-out__viewer');
  if (!list || !viewer) return;

  var rows = Array.prototype.slice.call(list.querySelectorAll('.vy-reel-cell'));
  var filters = Array.prototype.slice.call(document.querySelectorAll('.vy-groups button[data-filter]'));
  var countEl = document.getElementById('vy-log-count');

  var out = {
    image: document.getElementById('vy-out-image'),
    title: document.getElementById('vy-out-title'),
    kind: document.getElementById('vy-out-kind'),
    model: document.getElementById('vy-out-model'),
    steps: document.getElementById('vy-out-steps'),
    seed: document.getElementById('vy-out-seed'),
    size: document.getElementById('vy-out-size'),
    prompt: document.getElementById('vy-out-prompt'),
    forkDt: document.getElementById('vy-out-fork-dt'),
    fork: document.getElementById('vy-out-fork'),
    copy: document.getElementById('vy-out-copy')
  };

  var canHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  var isNarrow = window.matchMedia('(max-width: 900px)');
  var current = null;

  /* ── Viewer update ───────────────────────────────────────────────────────
     The image is copied straight from the row. Replace a row's image in the HTML and the
     viewer follows, which is why the viewer has no image slot of its own. */
  function paint(row) {
    if (!row || row === current) return;
    current = row;

    var d = row.dataset;
    var thumb = row.querySelector('.vy-out__thumb img');
    var src = thumb ? thumb.getAttribute('src') : '';

    rows.forEach(function (r) { r.setAttribute('aria-pressed', r === row ? 'true' : 'false'); });

    if (out.title) out.title.textContent = d.title || '';
    if (out.kind) out.kind.textContent = d.kind || '';
    if (out.model) out.model.textContent = d.model || '';
    if (out.steps) out.steps.textContent = d.steps || '';
    if (out.seed) out.seed.textContent = d.seed || '';
    if (out.size) out.size.textContent = d.size || '';
    if (out.prompt) out.prompt.textContent = d.renderPrompt || '';

    /* The fork row only appears on renders that came from another one; no empty rows */
    var fork = d.fork || '';
    if (out.forkDt && out.fork) {
      out.forkDt.hidden = !fork;
      out.fork.hidden = !fork;
      if (fork) out.fork.textContent = 'from ' + fork;
    }

    /* The ratio is the same number as the size, so portrait renders open tall and a 4x
       upscale opens wide */
    if (d.ratio) viewer.style.setProperty('--vy-out-ratio', d.ratio);

    if (out.image && src && out.image.getAttribute('src') !== src) {
      viewer.setAttribute('data-swapping', '1');
      var done = function () { viewer.removeAttribute('data-swapping'); };
      out.image.onload = done;
      out.image.onerror = done;                 /* Keep the panel usable if the image fails to load */
      out.image.setAttribute('src', src);
      out.image.setAttribute('alt', thumb ? (thumb.getAttribute('alt') || '') : '');
      if (out.image.complete) done();
    }
  }

  /* On narrow screens the viewer sits above the list, so a click may land off screen.
     Scrolling is owned by Lenis here: window.scrollTo is undone on the next frame. */
  function revealViewer() {
    if (!isNarrow.matches) return;
    var box = viewer.getBoundingClientRect();
    if (box.bottom > 120) return;               /* Already visible: leave it alone */
    var lenis = window.__lenis || window._lenis;
    if (lenis && typeof lenis.scrollTo === 'function') lenis.scrollTo(viewer, { offset: -100 });
    else window.scrollTo(0, window.pageYOffset + box.top - 100);
  }

  /* Hover selection is only accepted when the pointer actually moved.
     While scrolling, rows pass under a stationary cursor and fire a stream of enter
     events; left alone, the viewer keeps changing although the visitor chose nothing.
     The test is the coordinate, not a flag. Two flag based versions failed:
       · mouseenter plus "was there a pointermove just before" loses the first hover,
         because on a real move the enter event arrives before the move event.
       · "clear the flag on scroll" is undone by the synthetic pointermove the browser
         sends after scrolling to refresh hover state.
     An enter event caused by scrolling carries the same coordinates as the last one.
     That is the one signal that does not lie. */
  var lastX = null, lastY = null;
  if (canHover) {
    window.addEventListener('pointermove', function (e) { lastX = e.clientX; lastY = e.clientY; }, { passive: true });
  }
  function pointerActuallyMoved(e) {
    var moved = (lastX === null) || e.clientX !== lastX || e.clientY !== lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    return moved;
  }

  rows.forEach(function (row) {
    row.addEventListener('click', function () { paint(row); revealViewer(); });
    row.addEventListener('focus', function () { paint(row); });
    if (canHover) {
      row.addEventListener('pointerenter', function (e) {
        if (e.pointerType === 'mouse' && pointerActuallyMoved(e)) paint(row);
      });
    }
  });

  /* ── Filter ──────────────────────────────────────────────────────────────
     The tabs really filter. Numbers are not reassigned: the number on a row is the id of
     that render, not its position on screen. */
  function applyFilter(path) {
    var shown = 0;
    var first = null;

    rows.forEach(function (row) {
      var match = path === 'all' || row.dataset.path === path;
      row.hidden = !match;
      if (match) {
        shown++;
        if (!first) first = row;
      }
    });

    filters.forEach(function (b) {
      b.setAttribute('aria-pressed', b.dataset.filter === path ? 'true' : 'false');
    });

    if (countEl) countEl.textContent = shown + ' / ' + rows.length;

    /* If the render being viewed was filtered out, move to the first one still listed */
    if (first && (!current || current.hidden)) paint(first);

    if (window.ScrollTrigger) window.ScrollTrigger.refresh();
  }

  filters.forEach(function (b) {
    b.addEventListener('click', function () { applyFilter(b.dataset.filter); });
  });

  /* ── Copy prompt (same behaviour as index) ─────────────────────────────── */
  if (out.copy) {
    out.copy.addEventListener('click', function () {
      if (!out.prompt) return;
      var text = out.prompt.textContent.trim();

      function done() {
        out.copy.setAttribute('data-copied', '1');
        out.copy.textContent = 'Copied';
        setTimeout(function () {
          out.copy.removeAttribute('data-copied');
          out.copy.textContent = 'Copy prompt';
        }, 1600);
      }

      function fallback() {
        /* The clipboard API only exists in a secure context, and this has to work from a file:// preview too */
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) { out.copy.textContent = 'Copy failed'; }
        document.body.removeChild(ta);
      }

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else {
        fallback();
      }
    });
  }

  /* The first row is marked as pressed in the markup, so the list and the viewer agree
     even before this script runs. This call is only here to fill in the current selection. */
  if (rows.length) {
    var initial = rows.filter(function (r) { return r.getAttribute('aria-pressed') === 'true'; })[0] || rows[0];
    paint(initial);
  }
})();
