/* ============================================================================
 * Vayra — calculator and accordion for pricing.html
 *
 * Page specific: neither the shared engine (vayra-shell.js) nor the index console
 * (vayra-console.js) is extended from here, because several pages use both.
 *
 * ── Relationship to the calculator on index (the important contract) ─────────
 * It uses the same formula:
 *
 *     credits = renders × sizeMultiplier + round(renders × upscalesPer10 / 10) × upscaleCost
 *     recommendation = the first plan that satisfies both the credits and the resolution cap
 *
 * This page adds three inputs (batch size, licence, API tier). Those three do not change
 * the credit count; they only raise the minimum plan. So while they sit at their defaults,
 * the same inputs produce the same credits, the same plan, the same price and the same
 * sentence as on index. Had they been multipliers, the two screens would quote different
 * numbers for the same inputs, which is a defect a buyer finds immediately.
 *
 * ── The constants live in the markup, not in this file ───────────────────────
 * Prices, credits, resolution caps, batch limits, licence and API tier are read from the
 * data attributes on the tier cards in pricing.html; the size multipliers and the upscale
 * cost come from the calculator element. Edit the cards in the HTML and the calculator
 * follows, without opening any script.
 * ========================================================================= */
(function () {
  'use strict';

  var hasGsap = typeof window.gsap !== 'undefined';
  var reduced = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
    || document.documentElement.getAttribute('data-vy-motion') === 'off'
    || (document.body && document.body.getAttribute('data-vy-motion') === 'off');

  /* ══════════════════════════════════════════════════════════════════════════
   * 1. Accordion — comparison groups and the billing FAQ
   *    Same behaviour as the FAQ on index: aria-expanded drives a data attribute and CSS
   *    opens the row from 0fr to 1fr. No new motion is introduced, down to the same
   *    easing on the opening nudge.
   * ══════════════════════════════════════════════════════════════════════════ */
  function initAccordion(root) {
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
      });
    });
  }

  initAccordion(document.getElementById('vy-price-compare'));
  initAccordion(document.getElementById('vy-price-faq'));

  /* ══════════════════════════════════════════════════════════════════════════
   * 2. Calculator
   * ══════════════════════════════════════════════════════════════════════════ */
  var root = document.getElementById('vy-sizer');
  if (!root) return;

  var renders = document.getElementById('vy-sizer-renders');
  var size = document.getElementById('vy-sizer-size');
  if (!renders || !size) return;

  /* ── Reading the constants out of the markup ─────────────────────────────
     Each parser has its own fallback, so a missing attribute does not kill the calculator. */
  function parseSizes(raw) {
    var out = [];
    (raw || '').split(',').forEach(function (pair) {
      var bits = pair.split(':');
      var px = parseInt(bits[0], 10);
      var mul = parseFloat(bits[1]);
      if (px > 0 && mul > 0) out.push({ px: px, mul: mul });
    });
    return out.length ? out : [{ px: 1024, mul: 1 }];
  }

  var SIZES = parseSizes(root.getAttribute('data-sizes'));
  var UP_COST = parseFloat(root.getAttribute('data-upscale-credits')) || 2;

  var BATCH_STEPS = (root.getAttribute('data-batch-steps') || '1')
    .split(',')
    .map(function (v) { return parseInt(v, 10); })
    .filter(function (v) { return v > 0; });
  if (!BATCH_STEPS.length) BATCH_STEPS = [1];

  var LICENSE_RANK = { personal: 0, commercial: 1 };
  var API_RANK = { none: 0, shared: 1, project: 2 };

  /* A plan is a tier card. Their DOM order is trusted as the order from low to high. */
  var PLANS = Array.prototype.map.call(
    document.querySelectorAll('.vy-price-plans .vy-tier[data-tier]'),
    function (card) {
      var d = card.dataset;
      var nameEl = card.querySelector('.vy-tier__name');
      return {
        el: card,
        id: d.tier,
        label: nameEl ? nameEl.textContent.trim() : d.tier,
        credits: parseInt(d.credits, 10) || 0,
        maxPx: parseInt(d.maxPx, 10) || 0,
        cost: parseFloat(d.cost) || 0,
        cta: d.cta || 'Choose',
        batch: parseInt(d.batch, 10) || 1,
        license: d.license || 'personal',
        api: d.api || 'none'
      };
    }
  );
  if (!PLANS.length) return;

  /* ── Elements ─────────────────────────────────────────────────────────── */
  var rendersOut = document.getElementById('vy-sizer-renders-out');
  var sizeOut = document.getElementById('vy-sizer-size-out');
  var upOut = document.getElementById('vy-sizer-up-out');
  var upMinus = document.getElementById('vy-sizer-up-minus');
  var upPlus = document.getElementById('vy-sizer-up-plus');
  var batchOut = document.getElementById('vy-sizer-batch-out');
  var batchMinus = document.getElementById('vy-sizer-batch-minus');
  var batchPlus = document.getElementById('vy-sizer-batch-plus');
  var totalEl = document.getElementById('vy-sizer-total');
  var formulaEl = document.getElementById('vy-sizer-formula');
  var planEl = document.getElementById('vy-sizer-plan');
  var whyEl = document.getElementById('vy-sizer-why');
  var costEl = document.getElementById('vy-sizer-cost');
  var ctaEl = document.getElementById('vy-sizer-cta');

  /* ── State. The starting values also come from the markup ── */
  var upPer10 = upOut ? (parseInt(upOut.textContent, 10) || 0) : 0;
  var batchIdx = batchOut ? BATCH_STEPS.indexOf(parseInt(batchOut.textContent, 10)) : -1;
  if (batchIdx < 0) batchIdx = 0;

  var state = { license: 'personal', api: 'none' };
  var lastPlanId = null;

  function nf(v) { return v.toLocaleString('en-US'); }

  /* Index of the lowest plan that satisfies a condition, or the top plan if none do */
  function lowestPlanIndex(test) {
    for (var i = 0; i < PLANS.length; i++) { if (test(PLANS[i])) return i; }
    return PLANS.length - 1;
  }

  function update() {
    var r = parseInt(renders.value, 10);
    var sIdx = parseInt(size.value, 10);
    var s = SIZES[sIdx] || SIZES[0];
    var batch = BATCH_STEPS[batchIdx];

    var upscales = Math.round(r * (upPer10 / 10));
    var credits = r * s.mul + upscales * UP_COST;   /* Same formula as index */

    if (rendersOut) rendersOut.textContent = nf(r);
    if (sizeOut) sizeOut.textContent = s.px + ' px';
    if (upOut) upOut.textContent = String(upPer10);
    if (upMinus) upMinus.disabled = upPer10 <= 0;
    if (upPlus) upPlus.disabled = upPer10 >= 10;
    if (batchOut) batchOut.textContent = nf(batch);
    if (batchMinus) batchMinus.disabled = batchIdx <= 0;
    if (batchPlus) batchPlus.disabled = batchIdx >= BATCH_STEPS.length - 1;
    if (totalEl) totalEl.textContent = nf(credits);
    if (formulaEl) {
      formulaEl.textContent = '(' + nf(r) + ' renders × ' + s.mul + ') + (' +
        nf(upscales) + ' upscales × ' + UP_COST + ')';
    }

    /* ── Floors: the limits that credits cannot buy ───────────────────── */
    var floorIdx = 0;
    var floorLabel = '';
    function raise(idx, label) {
      if (idx > floorIdx) { floorIdx = idx; floorLabel = label; }
    }
    raise(lowestPlanIndex(function (p) { return p.batch >= batch; }),
          'Batches of ' + nf(batch));
    raise(lowestPlanIndex(function (p) {
            return LICENSE_RANK[p.license] >= LICENSE_RANK[state.license];
          }), 'A commercial license');
    raise(lowestPlanIndex(function (p) {
            return API_RANK[p.api] >= API_RANK[state.api];
          }), state.api === 'project' ? 'A key per project' : 'API access');

    /* ── Recommendation ────────────────────────────────────────────────────
       With the floors at zero this loop is literally the loop on index. */
    function firstFit(from) {
      for (var i = from; i < PLANS.length; i++) {
        if (credits <= PLANS[i].credits && s.px <= PLANS[i].maxPx) return i;
      }
      return -1;
    }

    var creditIdx = firstFit(0);
    var pickIdx = firstFit(floorIdx);
    var over = pickIdx < 0;
    if (over) pickIdx = PLANS.length - 1;

    var pick = PLANS[pickIdx];
    var creditIdxSafe = creditIdx < 0 ? PLANS.length - 1 : creditIdx;

    var why;
    if (over) {
      why = nf(credits) + ' credits is past the top plan. ' + pick.label + ' covers ' +
            nf(pick.credits) + '; the rest is billed as overage at the same rate.';
    } else if (floorIdx > creditIdxSafe) {
      /* When a limit rather than the credit count raised the plan, say which limit first */
      why = floorLabel + ' starts on ' + pick.label + '. ' + nf(credits) +
            ' credits sits inside its ' + nf(pick.credits) + '.';
    } else if (pickIdx === 0) {
      why = nf(credits) + ' credits fits inside the free tier. You would not gain anything by paying yet.';
    } else if (s.px > PLANS[0].maxPx && credits <= PLANS[0].credits) {
      why = 'The credit count is small, but ' + s.px + ' px output is not on the free tier.';
    } else {
      var prev = PLANS[pickIdx - 1];
      why = nf(credits) + ' credits is over ' + prev.label.toLowerCase() + "'s " +
            nf(prev.credits) + '. ' + pick.label + ' has ' + nf(pick.credits) + '.';
    }

    if (whyEl) whyEl.textContent = why;
    if (costEl) costEl.innerHTML = '<b>$' + pick.cost + '</b> / mo';
    if (ctaEl) ctaEl.textContent = pick.cta;

    if (planEl && planEl.textContent.trim().toUpperCase() !== pick.label.toUpperCase()) {
      planEl.textContent = pick.label;
      /* A small pop, so a changed result is felt as well as read (same values as index) */
      if (hasGsap && !reduced && lastPlanId !== null) {
        gsap.fromTo(planEl, { scale: 0.94, y: 6 }, { scale: 1, y: 0, duration: 0.5, ease: 'back.out(1.5)' });
      }
    }
    lastPlanId = pick.id;

    /* Reflect the current recommendation on the tier card */
    PLANS.forEach(function (p) {
      p.el.setAttribute('data-recommended', p.id === pick.id ? '1' : '0');
    });
  }

  /* ── Wiring ───────────────────────────────────────────────────────────── */
  renders.addEventListener('input', update);
  size.addEventListener('input', update);

  if (upMinus) upMinus.addEventListener('click', function () { upPer10 = Math.max(0, upPer10 - 1); update(); });
  if (upPlus) upPlus.addEventListener('click', function () { upPer10 = Math.min(10, upPer10 + 1); update(); });

  if (batchMinus) batchMinus.addEventListener('click', function () { batchIdx = Math.max(0, batchIdx - 1); update(); });
  if (batchPlus) batchPlus.addEventListener('click', function () { batchIdx = Math.min(BATCH_STEPS.length - 1, batchIdx + 1); update(); });

  Array.prototype.forEach.call(root.querySelectorAll('.vy-sizer__seg'), function (group) {
    var key = group.getAttribute('data-seg');
    if (!key) return;
    var btns = Array.prototype.slice.call(group.querySelectorAll('button[data-value]'));
    if (!btns.length) return;

    var active = btns.filter(function (b) { return b.getAttribute('aria-pressed') === 'true'; })[0] || btns[0];
    state[key] = active.getAttribute('data-value');

    btns.forEach(function (b) {
      b.addEventListener('click', function () {
        btns.forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
        state[key] = b.getAttribute('data-value');
        update();
      });
    });
  });

  update();
})();
