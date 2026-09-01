/* ============================================================================
 * Vayra — the working parts of the product console on index
 *
 *   gallery  → filters that filter and a prompt that copies
 *   metrics  → a multi series panel with a series toggle and a cursor readout
 *   pricing  → a calculator whose slider and stepper update the recommended plan live
 *
 * Every number here is sample data. To make it yours, edit the SAMPLE_ constants and the
 * plan table below; nothing else in this file needs to change.
 * ============================================================================ */
(function () {
  'use strict';

  var reduced = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
    || document.documentElement.getAttribute('data-vy-motion') === 'off'
    || (document.body && document.body.getAttribute('data-vy-motion') === 'off');
  var hasGsap = typeof window.gsap !== 'undefined';

  /* ══════════════════════════════════════════════════════════════════════════
   * 1. Gallery — filter and copy prompt
   * ══════════════════════════════════════════════════════════════════════════ */
  function initGallery() {
    var grid = document.getElementById('vy-gallery-grid');
    var countEl = document.getElementById('vy-gallery-count');
    if (!grid) return;

    var cards = Array.prototype.slice.call(grid.querySelectorAll('.vy-work'));
    var buttons = Array.prototype.slice.call(document.querySelectorAll('.vy-filter'));

    function apply(filter) {
      var shown = 0;
      cards.forEach(function (card) {
        var match = filter === 'all' || card.getAttribute('data-path') === filter;
        card.hidden = !match;
        if (match) shown++;
      });
      buttons.forEach(function (b) {
        b.setAttribute('aria-pressed', b.getAttribute('data-filter') === filter ? 'true' : 'false');
      });
      if (countEl) countEl.textContent = shown + ' / ' + cards.length;
      if (hasGsap && window.ScrollTrigger) ScrollTrigger.refresh();
    }

    buttons.forEach(function (b) {
      b.addEventListener('click', function () { apply(b.getAttribute('data-filter')); });
    });

    /* Copy the prompt: a console should let you take the value out */
    Array.prototype.forEach.call(grid.querySelectorAll('.vy-copy-prompt'), function (btn) {
      btn.addEventListener('click', function () {
        var field = btn.parentElement.querySelector('.vy-prompt-field');
        if (!field) return;
        var text = field.textContent.trim();

        function done() {
          btn.setAttribute('data-copied', '1');
          btn.textContent = 'Copied';
          setTimeout(function () { btn.removeAttribute('data-copied'); btn.textContent = 'Copy prompt'; }, 1600);
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, fallback);
        } else {
          fallback();
        }

        function fallback() {
          /* The clipboard API only exists in a secure context, and this has to work from a file:// preview too */
          var ta = document.createElement('textarea');
          ta.value = text;
          ta.setAttribute('readonly', '');
          ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand('copy'); done(); } catch (e) { btn.textContent = 'Copy failed'; }
          document.body.removeChild(ta);
        }
      });
    });

    apply('all');
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 2. Metrics — multi series SVG panel
   * The palette has no hues to spend on series, so they are separated by lightness and
   * dash pattern instead (the CSS keys off the series index).
   * ══════════════════════════════════════════════════════════════════════════ */
  var SAMPLE_SERIES_NAMES = ['Photoreal', 'Product', 'Illustration', 'Upscale'];

  /* 24 hours at 30 minute intervals, 49 points. Generated deterministically, so a refresh
     shows the same graph. Sample data. */
  function buildSampleSeries() {
    var out = [];
    var shapes = [
      { base: 8.4, amp: 3.6, w1: 0.42, w2: 0.17, ph: 0.0 },
      { base: 6.9, amp: 2.4, w1: 0.31, w2: 0.23, ph: 1.7 },
      { base: 3.6, amp: 1.5, w1: 0.26, w2: 0.44, ph: 3.1 },
      { base: 2.1, amp: 0.9, w1: 0.19, w2: 0.37, ph: 4.6 }
    ];
    shapes.forEach(function (s) {
      var pts = [];
      for (var i = 0; i <= 48; i++) {
        var v = s.base
          + Math.sin(i * s.w1 + s.ph) * s.amp
          + Math.sin(i * s.w2 + s.ph * 1.9) * (s.amp * 0.38)
          + Math.sin(i * 0.91 + s.ph * 2.7) * (s.amp * 0.12);
        pts.push(Math.max(0.2, Math.round(v * 100) / 100));
      }
      out.push(pts);
    });
    return out;
  }

  var X_LABELS = ['16:00', '22:30', '05:00', '11:30'];
  var Y_MAX = 15;
  var PLOT = { l: 96, r: 1184, t: 20, b: 360 };

  function initChart() {
    var host = document.getElementById('vy-chart');
    if (!host) return;
    var gGrid = document.getElementById('vy-chart-grid');
    var gAxis = document.getElementById('vy-chart-axis');
    var gSeries = document.getElementById('vy-chart-series');
    var gCursor = document.getElementById('vy-chart-cursor');
    var readout = document.getElementById('vy-chart-readout');
    if (!gGrid || !gAxis || !gSeries) return;

    var data = buildSampleSeries();
    var n = data[0].length;
    var NS = 'http://www.w3.org/2000/svg';

    function x(i) { return PLOT.l + (i / (n - 1)) * (PLOT.r - PLOT.l); }
    function y(v) { return PLOT.b - (v / Y_MAX) * (PLOT.b - PLOT.t); }

    /* Grid and axes */
    for (var v = 0; v <= Y_MAX; v += 3) {
      var line = document.createElementNS(NS, 'line');
      line.setAttribute('x1', PLOT.l); line.setAttribute('x2', PLOT.r);
      line.setAttribute('y1', y(v)); line.setAttribute('y2', y(v));
      gGrid.appendChild(line);

      var lbl = document.createElementNS(NS, 'text');
      lbl.setAttribute('x', PLOT.l - 18);
      lbl.setAttribute('y', y(v) + 5);
      lbl.setAttribute('text-anchor', 'end');
      lbl.textContent = String(v);
      gAxis.appendChild(lbl);
    }

    X_LABELS.forEach(function (t, i) {
      var lbl = document.createElementNS(NS, 'text');
      var px = PLOT.l + (i / (X_LABELS.length - 1)) * (PLOT.r - PLOT.l);
      lbl.setAttribute('x', px);
      lbl.setAttribute('y', PLOT.b + 32);
      lbl.setAttribute('text-anchor', i === 0 ? 'start' : (i === X_LABELS.length - 1 ? 'end' : 'middle'));
      lbl.textContent = t;
      gAxis.appendChild(lbl);
    });

    /* Series, drawn as a smooth curve */
    function pathFor(points) {
      var d = 'M ' + x(0).toFixed(1) + ' ' + y(points[0]).toFixed(1);
      for (var i = 0; i < points.length - 1; i++) {
        var x0 = x(i), y0 = y(points[i]);
        var x1 = x(i + 1), y1 = y(points[i + 1]);
        var cx = (x0 + x1) / 2;
        d += ' C ' + cx.toFixed(1) + ' ' + y0.toFixed(1) + ', ' + cx.toFixed(1) + ' ' + y1.toFixed(1) +
             ', ' + x1.toFixed(1) + ' ' + y1.toFixed(1);
      }
      return d;
    }

    var paths = [];
    data.forEach(function (points, si) {
      var g = document.createElementNS(NS, 'g');
      g.setAttribute('data-off', '0');
      var p = document.createElementNS(NS, 'path');
      p.setAttribute('d', pathFor(points));
      p.setAttribute('data-series', String(si));
      g.appendChild(p);
      gSeries.appendChild(g);
      paths.push({ g: g, p: p });

      /* Entrance: the lines draw themselves. With reduced motion they are simply there. */
      if (!reduced && hasGsap && window.ScrollTrigger) {
        var len = p.getTotalLength ? p.getTotalLength() : 0;
        if (len) {
          gsap.set(p, { strokeDasharray: len, strokeDashoffset: len });
          gsap.to(p, {
            strokeDashoffset: 0, duration: 1.4, ease: 'expo.out', delay: si * 0.09,
            scrollTrigger: { trigger: host, start: 'top 82%', once: true },
            onComplete: function () { p.style.strokeDasharray = ''; p.style.strokeDashoffset = ''; }
          });
        }
      }
    });

    /* Legend toggle */
    Array.prototype.forEach.call(document.querySelectorAll('#vy-chart-legend button'), function (btn) {
      btn.addEventListener('click', function () {
        var si = parseInt(btn.getAttribute('data-series'), 10);
        var on = btn.getAttribute('aria-pressed') === 'true';
        btn.setAttribute('aria-pressed', on ? 'false' : 'true');
        paths[si].g.setAttribute('data-off', on ? '1' : '0');
      });
    });

    /* Cursor readout: this is what makes the panel something you read rather than look at */
    var svg = host.querySelector('svg');
    function readAt(clientX) {
      var r = svg.getBoundingClientRect();
      var vx = ((clientX - r.left) / r.width) * 1200;
      var i = Math.round(((vx - PLOT.l) / (PLOT.r - PLOT.l)) * (n - 1));
      i = Math.max(0, Math.min(n - 1, i));

      if (gCursor) {
        gCursor.setAttribute('data-on', '1');
        gCursor.querySelector('line').setAttribute('x1', x(i));
        gCursor.querySelector('line').setAttribute('x2', x(i));
      }

      if (readout) {
        var parts = [];
        data.forEach(function (pts, si) {
          if (paths[si].g.getAttribute('data-off') === '1') return;
          parts.push(SAMPLE_SERIES_NAMES[si] + ' ' + pts[i].toFixed(1));
        });
        var hh = (16 + Math.floor(i / 2)) % 24;
        var mm = (i % 2) ? '30' : '00';
        readout.textContent = (hh < 10 ? '0' : '') + hh + ':' + mm + '  ·  ' + (parts.join('   ') || 'no series');
      }
    }

    svg.addEventListener('pointermove', function (e) { readAt(e.clientX); });
    svg.addEventListener('pointerleave', function () {
      if (gCursor) gCursor.setAttribute('data-on', '0');
      if (readout) readout.textContent = '—';
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 3. Count up for the statistics under the chart
   * ══════════════════════════════════════════════════════════════════════════ */
  function initCountUp() {
    var els = Array.prototype.slice.call(document.querySelectorAll('[data-countup]'));
    if (!els.length) return;

    function run(el) {
      if (el.dataset.vyCounted === '1') return;
      el.dataset.vyCounted = '1';
      var end = parseFloat(el.getAttribute('data-countup'));
      var dec = parseInt(el.getAttribute('data-decimals') || '0', 10);
      var suffix = el.querySelector('small');
      var tail = suffix ? suffix.outerHTML : '';

      if (reduced) { el.innerHTML = end.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + tail; return; }

      var start = performance.now();
      var dur = 1500;
      (function step(now) {
        var t = Math.min(1, (now - start) / dur);
        var eased = 1 - Math.pow(1 - t, 4);        /* Exponential ease out */
        var v = end * eased;
        el.innerHTML = v.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) + tail;
        if (t < 1) requestAnimationFrame(step);
      })(start);
    }

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        run(e.target);
        io.unobserve(e.target);
      });
    }, { threshold: 0.35 });

    els.forEach(function (el) { io.observe(el); });

    setTimeout(function () {                       /* Backstop: the numbers have to appear even if the observer never fires */
      /* Decide from the number, not from the string. Matching on '0' and '0%' misses an item
         whose suffix differs (0 followed by a unit), and the result is a panel where three of
         four figures animate and one sits at zero. */
      els.forEach(function (el) { if (parseFloat(el.textContent) === 0) run(el); });
    }, 3000);
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 4. Pricing calculator (it really calculates)
   * ══════════════════════════════════════════════════════════════════════════ */
  var SIZES = [
    { px: 1024, mul: 1 },
    { px: 2048, mul: 2 },
    { px: 3072, mul: 3 },
    { px: 4096, mul: 4 }
  ];

  var PLANS = [
    { id: 'starter', name: 'STARTER', credits: 200,   maxPx: 1024, cost: 0,  cta: 'Start free' },
    { id: 'studio',  name: 'STUDIO',  credits: 4000,  maxPx: 4096, cost: 24, cta: 'Choose Studio' },
    { id: 'scale',   name: 'SCALE',   credits: 20000, maxPx: 4096, cost: 96, cta: 'Choose Scale' }
  ];

  function initCalculator() {
    var renders = document.getElementById('vy-calc-renders');
    var size = document.getElementById('vy-calc-size');
    if (!renders || !size) return;

    var rendersOut = document.getElementById('vy-calc-renders-out');
    var sizeOut = document.getElementById('vy-calc-size-out');
    var upOut = document.getElementById('vy-calc-up-out');
    var upMinus = document.getElementById('vy-calc-up-minus');
    var upPlus = document.getElementById('vy-calc-up-plus');
    var totalEl = document.getElementById('vy-calc-total');
    var formulaEl = document.getElementById('vy-calc-formula');
    var planEl = document.getElementById('vy-calc-plan');
    var whyEl = document.getElementById('vy-calc-why');
    var costEl = document.getElementById('vy-calc-cost');
    var ctaEl = document.getElementById('vy-calc-cta');

    var upPer10 = 0;
    var lastPlanId = null;

    function nf(v) { return v.toLocaleString('en-US'); }

    function update() {
      var r = parseInt(renders.value, 10);
      var s = SIZES[parseInt(size.value, 10)];
      var upscales = Math.round(r * (upPer10 / 10));
      var credits = r * s.mul + upscales * 2;

      if (rendersOut) rendersOut.textContent = nf(r);
      if (sizeOut) sizeOut.textContent = s.px + ' px';
      if (upOut) upOut.textContent = String(upPer10);
      if (upMinus) upMinus.disabled = upPer10 <= 0;
      if (upPlus) upPlus.disabled = upPer10 >= 10;
      if (totalEl) totalEl.textContent = nf(credits);
      if (formulaEl) {
        formulaEl.textContent = '(' + nf(r) + ' renders × ' + s.mul + ') + (' + nf(upscales) + ' upscales × 2)';
      }

      /* The recommendation is the first plan that satisfies both the credits and the resolution cap */
      var pick = null;
      for (var i = 0; i < PLANS.length; i++) {
        if (credits <= PLANS[i].credits && s.px <= PLANS[i].maxPx) { pick = PLANS[i]; break; }
      }

      var why;
      if (!pick) {
        pick = PLANS[PLANS.length - 1];
        why = nf(credits) + ' credits is past the top plan. Scale covers ' + nf(pick.credits) +
              '; the rest is billed as overage at the same rate.';
      } else if (pick.id === 'starter') {
        why = nf(credits) + ' credits fits inside the free tier. You would not gain anything by paying yet.';
      } else if (s.px > 1024 && credits <= PLANS[0].credits) {
        why = 'The credit count is small, but ' + s.px + ' px output is not on the free tier.';
      } else {
        var prev = PLANS[PLANS.indexOf(pick) - 1];
        why = nf(credits) + ' credits is over ' + prev.name.toLowerCase() + "'s " + nf(prev.credits) +
              '. ' + pick.name.charAt(0) + pick.name.slice(1).toLowerCase() + ' has ' + nf(pick.credits) + '.';
      }

      if (whyEl) whyEl.textContent = why;
      if (costEl) costEl.innerHTML = '<b>$' + pick.cost + '</b> / mo';
      if (ctaEl) ctaEl.textContent = pick.cta;

      if (planEl && pick.name !== planEl.textContent) {
        planEl.textContent = pick.name;
        /* A small pop, so a changed recommendation is felt as well as read */
        if (hasGsap && !reduced && lastPlanId !== null) {
          gsap.fromTo(planEl, { scale: 0.94, y: 6 }, { scale: 1, y: 0, duration: 0.5, ease: 'back.out(1.5)' });
        }
      }
      lastPlanId = pick.id;

      /* Reflect the current recommendation on the tier card */
      Array.prototype.forEach.call(document.querySelectorAll('.vy-tier'), function (card) {
        card.setAttribute('data-recommended', card.getAttribute('data-tier') === pick.id ? '1' : '0');
      });
    }

    renders.addEventListener('input', update);
    size.addEventListener('input', update);
    if (upMinus) upMinus.addEventListener('click', function () { upPer10 = Math.max(0, upPer10 - 1); update(); });
    if (upPlus) upPlus.addEventListener('click', function () { upPer10 = Math.min(10, upPer10 + 1); update(); });

    update();
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 5. faq — grid-template-rows: 0fr ↔ 1fr
   * ══════════════════════════════════════════════════════════════════════════ */
  function initFaq() {
    var root = document.getElementById('vy-faq');
    if (!root) return;

    Array.prototype.forEach.call(root.querySelectorAll('.vy-faq__q'), function (btn) {
      btn.addEventListener('click', function () {
        var panel = document.getElementById(btn.getAttribute('aria-controls'));
        if (!panel) return;
        var open = btn.getAttribute('aria-expanded') === 'true';
        btn.setAttribute('aria-expanded', open ? 'false' : 'true');
        panel.setAttribute('data-open', open ? '0' : '1');

        if (!open && hasGsap && !reduced) {
          gsap.fromTo(btn, { x: -6 }, { x: 0, duration: 0.45, ease: 'back.out(1.4)' });
        }
        if (hasGsap && window.ScrollTrigger) {
          setTimeout(function () { ScrollTrigger.refresh(); }, 420);
        }
      });
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 6. Capability — the three columns reveal line by line (no library needed)
   * ══════════════════════════════════════════════════════════════════════════ */
  function initCapabilityReveal() {
    var section = document.getElementById('capability');
    if (!section || reduced) return;
    /* The global reveal already applies here; this only redistributes the per item delay so the
     three columns read as lines rather than as blocks */
    Array.prototype.forEach.call(section.querySelectorAll('.vy-cap__item'), function (item, i) {
      item.style.setProperty('--i', String(2 + i * 1.4));
    });
  }

  function boot() {
    initGallery();
    initChart();
    initCountUp();
    initCalculator();
    initFaq();
    initCapabilityReveal();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
