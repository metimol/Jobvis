/* ============================================================================
 * Vayra — page shell, shared by all five pages
 *
 * What is in here: the entry gate, the page transition curtain, the custom cursor, the
 * single scroll driver, the nav, the scroll driven colour shift, the reveal masks, the
 * marquee ticker and the footer letter roll.
 *
 * The scroll driver is created here and nowhere else. A second instance makes the page
 * scroll twice.
 * ============================================================================ */
(function () {
  'use strict';

  var reduced = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
    || document.documentElement.getAttribute('data-vy-motion') === 'off'
    || (document.body && document.body.getAttribute('data-vy-motion') === 'off');
  var hasGsap = typeof window.gsap !== 'undefined';
  if (hasGsap && window.ScrollTrigger) gsap.registerPlugin(ScrollTrigger);

  var EASE = 'expo.out';            /* The signature easing */
  var DUR_HERO = 1.2;
  var DUR_SECTION = 0.8;
  var DUR_MICRO = 0.3;

  /* ══════════════════════════════════════════════════════════════════════════
   * 0. Scroll driver: one owner, plus a velocity signal
   * Velocity is sampled per animation frame. A scroll event listener is not used.
   * ══════════════════════════════════════════════════════════════════════════ */
  var scrollVel = 0;
  var lastScroll = window.scrollY;

  (function initScrollDriver() {
    if (reduced || typeof window.Lenis === 'undefined') return;   /* Native scrolling when Lenis is absent or motion is reduced */
    var lenis = new window.Lenis({ duration: 1.05, smoothWheel: true, touchMultiplier: 1.4 });
    /* Two global names for one instance, because different components look for different names */
    window.__lenis = lenis;
    window._lenis = lenis;

    if (hasGsap && window.ScrollTrigger) {
      lenis.on('scroll', ScrollTrigger.update);
      gsap.ticker.add(function (t) { lenis.raf(t * 1000); });
      gsap.ticker.lagSmoothing(0);
    } else {
      (function raf(t) { lenis.raf(t); requestAnimationFrame(raf); })(0);
    }

    /* Once Lenis owns scrolling, the browser's own anchor jump is undone: the next Lenis
       frame rewinds to its own target. The nav, the hero quick links and the footer are all
       in page anchors, so leaving this alone kills navigation on the whole page.
       For the same reason the CSS smooth scroll behaviour is switched off here: it fights
       with Lenis rather than adding to it. */
    document.documentElement.style.scrollBehavior = 'auto';

    document.addEventListener('click', function (e) {
      var a = e.target && e.target.closest ? e.target.closest('a[href^="#"]') : null;
      if (!a) return;
      var id = a.getAttribute('href');
      if (!id || id === '#') return;
      var target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      lenis.scrollTo(target, { offset: 0, duration: 1.1 });
      history.pushState(null, '', id);
      /* Move focus as well. For a keyboard user, a link that only scrolls has no destination. */
      target.setAttribute('tabindex', '-1');
      target.focus({ preventScroll: true });
    });
  })();

  (function sampleVelocity() {
    function tick() {
      var y = (window.__lenis && typeof window.__lenis.scroll === 'number') ? window.__lenis.scroll : window.scrollY;
      var d = y - lastScroll;
      lastScroll = y;
      scrollVel += (d - scrollVel) * 0.28;
      if (Math.abs(scrollVel) < 0.01) scrollVel = 0;
      window.__vyScrollVel = scrollVel;
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  })();

  /* ══════════════════════════════════════════════════════════════════════════
   * 1. Pixel curtain: the page transition (and a one time entrance)
   * The grid, origin, step and peak below are locked; they define the wipe.
   * ══════════════════════════════════════════════════════════════════════════ */
  var CURTAIN_CFG = {
    rows: 9, cols: 17,
    cellBg: 'var(--pt-bg, #0A0A0B)',
    origin: 'center',
    step: 0.022,
    peak: 1.03,
    /* Deliberately close to linear. With an ease out curve all 221 cells finish inside the
       first fifteen percent of the duration and the staggered wave collapses into one flash:
       a grid dissolve needs a near linear curve to read.
       The curtain is shortened through its depth token in the bridge, not through the curve. */
    out: { duration: 0.3, ease: 'power1' },
    /* The covering direction when leaving is the opening played backwards, on the same curve.
       It is slightly longer than the reveal, because covering is the moment the visitor reads
       as "something happened"; under about 0.3s it is indistinguishable from the click itself.
       The navigation itself is capped separately. */
    inn: { duration: 0.42, ease: 'power1' }
  };

  function PixelCurtain(host) {
    this.host = host; this.layer = null; this.cells = [];
  }

  PixelCurtain.prototype.build = function () {
    if (!this.host || this.layer) return this;
    var layer = document.createElement('div');
    layer.className = 'page-transition-pixels';
    layer.setAttribute('aria-hidden', 'true');
    layer.style.cssText = 'position:absolute;inset:0;display:grid;opacity:0;pointer-events:none;' +
      'grid-template-columns:repeat(' + CURTAIN_CFG.cols + ',1fr);' +
      'grid-template-rows:repeat(' + CURTAIN_CFG.rows + ',1fr);';
    var total = CURTAIN_CFG.rows * CURTAIN_CFG.cols;
    for (var i = 0; i < total; i++) {
      var cell = document.createElement('div');
      cell.style.background = CURTAIN_CFG.cellBg;
      cell.style.willChange = 'transform, opacity';
      layer.appendChild(cell);
      this.cells.push(cell);
    }
    this.host.appendChild(layer);
    this.layer = layer;
    return this;
  };

  PixelCurtain.prototype.spread = function () {
    return { grid: [CURTAIN_CFG.rows, CURTAIN_CFG.cols], from: CURTAIN_CFG.origin, each: CURTAIN_CFG.step };
  };

  PixelCurtain.prototype.prime = function () {
    if (!this.layer) return;
    this.layer.style.opacity = '1';
    if (hasGsap) gsap.set(this.cells, { scale: CURTAIN_CFG.peak, opacity: 1, transformOrigin: '50% 50%' });
  };

  PixelCurtain.prototype.open = function () {
    var self = this;
    if (!this.layer) return Promise.resolve();
    if (!hasGsap) { this.layer.style.opacity = '0'; return Promise.resolve(); }
    return new Promise(function (done) {
      gsap.to(self.cells, {
        duration: CURTAIN_CFG.out.duration,
        ease: CURTAIN_CFG.out.ease,
        scale: 0, opacity: 0,
        transformOrigin: '50% 50%',
        stagger: self.spread(),
        overwrite: 'auto',
        onComplete: function () { self.layer.style.opacity = '0'; done(); }
      });
    });
  };

  /* Leaving: the cells fill the screen from the centre outwards */
  PixelCurtain.prototype.close = function () {
    var self = this;
    if (!this.layer) return Promise.resolve();
    this.layer.style.opacity = '1';
    if (!hasGsap) return Promise.resolve();
    return new Promise(function (done) {
      gsap.fromTo(
        self.cells,
        { scale: 0, opacity: 0, transformOrigin: '50% 50%' },
        {
          duration: CURTAIN_CFG.inn.duration,
          ease: CURTAIN_CFG.inn.ease,
          scale: CURTAIN_CFG.peak, opacity: 1,
          transformOrigin: '50% 50%',
          stagger: self.spread(),
          overwrite: 'auto',
          onComplete: done
        }
      );
    });
  };

  /* Kill whatever is running and clear unconditionally. This is where the fallbacks and the
       back button restore both end up. */
  PixelCurtain.prototype.freeze = function () {
    if (!this.layer) return;
    if (hasGsap) gsap.killTweensOf(this.cells);
    this.layer.style.opacity = '0';
  };

  /* ══════════════════════════════════════════════════════════════════════════
   * 1-b. Page transition: cover on the way out, clear on arrival.
   *
   * The worst possible failure here is a curtain that never clears, leaving the visitor on
   * a black screen. So there are three ways it opens: the normal path, a timeout in this
   * file, and a timer in the inline script in each subpage head, which covers the case where
   * this file never loaded at all.
   * ══════════════════════════════════════════════════════════════════════════ */
  var HANDOFF_KEY = 'vy:page-transition';   /* The key the inline script in each subpage head reads */
  var LEAVE_HARD_NAV = 1200;                /* Navigate at this point whether or not the cover finished */
  var ARRIVE_FAILSAFE = 2500;               /* Open at this point whether or not the normal path ran */
  var departing = false;
  var arriveTimer = 0;

  function handoffSet() {
    try { sessionStorage.setItem(HANDOFF_KEY, '1'); } catch (err) { /* Private mode: only the handoff is lost */ }
  }
  function handoffClear() {
    try { sessionStorage.removeItem(HANDOFF_KEY); } catch (err) { /* noop */ }
  }

  /* Clear the curtain and hand the page over. Safe to call more than once. */
  function releaseCurtain(shell, curtain) {
    if (arriveTimer) { clearTimeout(arriveTimer); arriveTimer = 0; }
    document.documentElement.classList.remove('vy-arriving');
    if (curtain) curtain.freeze();
    if (shell) shell.classList.remove('vy-is-live');
  }

  function armArriveFailsafe(shell, curtain) {
    if (arriveTimer) clearTimeout(arriveTimer);
    arriveTimer = setTimeout(function () {
      if (departing) return;               /* Already leaving, so being covered is correct */
      releaseCurtain(shell, curtain);
    }, ARRIVE_FAILSAFE);
  }

  /* Treat "/" and "/index.html" as the same document. Without this, opening the site at the
     server root turns the home link into a full reload of the page you are already on. */
  function normalizePath(p) {
    var s = p || '/';
    try { s = decodeURI(s); } catch (err) { /* Compare the raw value if it will not decode */ }
    if (s.charAt(s.length - 1) === '/') s += 'index.html';
    return s.toLowerCase();
  }
  function isSameDocument(pathname) {
    return normalizePath(pathname) === normalizePath(window.location.pathname);
  }

  /* In page movement, using the same rule as the anchor handler above (Lenis owns scrolling) */
  function scrollToHash(hash) {
    var target = null;
    try { target = hash ? document.querySelector(hash) : null; } catch (err) { target = null; }
    if (!target) {
      if (window.__lenis) window.__lenis.scrollTo(0, { duration: 1.1 });
      else window.scrollTo(0, 0);
      return;
    }
    if (window.__lenis) window.__lenis.scrollTo(target, { offset: 0, duration: 1.1 });
    else window.scrollTo(0, target.getBoundingClientRect().top + window.scrollY);
    history.pushState(null, '', hash);
    target.setAttribute('tabindex', '-1');
    target.focus({ preventScroll: true });
  }

  function depart(href, shell, curtain) {
    if (departing) return;
    departing = true;
    if (arriveTimer) { clearTimeout(arriveTimer); arriveTimer = 0; }

    var gone = false;
    function go() { if (gone) return; gone = true; window.location.href = href; }

    /* Reduced motion, no GSAP or no curtain: navigate immediately, and leave no flag behind */
    if (reduced || !hasGsap || !shell || !curtain) { departing = false; go(); return; }

    handoffSet();
    if (window.__lenis && typeof window.__lenis.stop === 'function') window.__lenis.stop();
    shell.classList.add('vy-is-live');
    setTimeout(go, LEAVE_HARD_NAV);
    curtain.close().then(go);
  }

  /* This does not touch the anchor handler in the scroll driver. The two handlers cover
     different sets: that one takes href values starting with a hash, this one takes document
     paths ending in .html. */
  function initPageLinks(shell, curtain) {
    document.addEventListener('click', function (e) {
      if (e.defaultPrevented) return;
      if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;   /* New tab or modifier click */

      var a = e.target && e.target.closest ? e.target.closest('a[href]') : null;
      if (!a) return;
      if (a.target && a.target !== '_self') return;                                     /* target="_blank" */
      if (a.hasAttribute('download')) return;

      var raw = a.getAttribute('href');
      if (!raw || raw.charAt(0) === '#') return;                                        /* An anchor belongs to the other handler */

      var dest;
      try { dest = new URL(raw, window.location.href); } catch (err) { return; }
      if (dest.origin !== window.location.origin) return;                               /* External domain */
      if (!/\.html$/i.test(dest.pathname)) return;                                      /* Documents only */

      /* The nav and footer markup is identical on all five pages, so the home link arrives as
         index.html with an anchor. When that is the current document it means scroll, not
         navigate; reloading index from index would be a regression. */
      if (isSameDocument(dest.pathname)) {
        e.preventDefault();
        scrollToHash(dest.hash);
        return;
      }

      e.preventDefault();
      depart(dest.href, shell, curtain);
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 2. Entry gate (the counter step is locked)
   * ══════════════════════════════════════════════════════════════════════════ */
  function initGate(onDone) {
    var el = document.getElementById('vy-probe');
    var ld = document.getElementById('vy-door');
    if (!el || !ld) { onDone(); return; }
    if (reduced) { ld.classList.add('vy-set'); onDone(); return; }

    var n = 0;
    var iv = setInterval(function () {
      n += Math.ceil((100 - n) / 7) + 1;          /* Locked: counter step */
      if (n >= 100) {
        n = 100;
        clearInterval(iv);
        el.textContent = '100';
        setTimeout(function () { ld.classList.add('vy-set'); onDone(); }, 400);
      } else {
        el.textContent = n;
      }
    }, 42);                                       /* Locked: about 26 ticks of 42ms, so roughly a 1.5s ceiling */
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 3. Hero load choreography: the masked elements arrive in sequence
   * ══════════════════════════════════════════════════════════════════════════ */
  function heroChoreography() {
    if (!hasGsap || reduced) return;
    var targets = document.querySelectorAll(
      '.vy-banner-band .vy-mega-copy > div, .vy-banner-band .vy-mini-tag > div,' +
      '.vy-banner-band .vy-crafts-stack li > div, .vy-banner-band .vy-banner-fast-paths > div,' +
      '.vy-banner-band .vy-edge-top-rule > div > div, .vy-banner-band .vy-outro-span > div'
    );
    if (!targets.length) return;
    gsap.set(targets, { yPercent: 110 });
    gsap.to(targets, { yPercent: 0, duration: DUR_HERO, ease: EASE, stagger: 0.035 });

    var console_ = document.querySelector('.vy-low-ui-frame');
    if (console_) {
      gsap.set(console_, { autoAlpha: 0, y: 18 });
      gsap.to(console_, { autoAlpha: 1, y: 0, duration: DUR_SECTION, ease: EASE, delay: 0.42 });
    }
  }

  /* Hero pointer parallax. The overlay UI moves very slightly against the GL plane, which is
     what separates the two layers. */
  function heroOverlayParallax() {
    if (reduced || window.matchMedia('(hover: none)').matches) return;
    var ui = document.querySelector('.vy-hud-ui');
    var hero = document.querySelector('.vy-banner-band');
    if (!ui || !hero) return;
    var tx = 0, ty = 0, cx = 0, cy = 0, raf = 0;

    hero.addEventListener('mousemove', function (e) {
      var r = hero.getBoundingClientRect();
      tx = ((e.clientX - r.left) / r.width - 0.5) * -12;
      ty = ((e.clientY - r.top) / r.height - 0.5) * -8;
      if (!raf) raf = requestAnimationFrame(loop);
    }, { passive: true });

    function loop() {
      raf = 0;
      cx += (tx - cx) * 0.06;
      cy += (ty - cy) * 0.06;
      ui.style.transform = 'translate3d(' + cx.toFixed(2) + 'px,' + cy.toFixed(2) + 'px,0)';
      if (Math.abs(tx - cx) > 0.05 || Math.abs(ty - cy) > 0.05) raf = requestAnimationFrame(loop);
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 4. Custom cursor, and the switch that turns it off
   * ══════════════════════════════════════════════════════════════════════════ */
  function initCursor() {
    var dot = document.getElementById('vy-probe-bead');
    var ring = document.getElementById('vy-probe-halo');
    if (!dot || !ring) return;

    var mx = 0, my = 0, rx = 0, ry = 0;
    var magnetTarget = null, magRect = null;
    var magnetStrength = 0.22;                    /* Locked: how strongly the ring is pulled toward a target */
    var leaveTimer = null, enterRaf = 0, cursorRaf = 0, lastWake = 0;
    var EXIT_MS = 140;

    function wake() {
      lastWake = performance.now();
      if (!cursorRaf) cursorRaf = requestAnimationFrame(loop);
    }

    function showLabel(label, side) {
      if (leaveTimer) { clearTimeout(leaveTimer); leaveTimer = null; }
      dot.classList.add('vy-is-path');
      ring.classList.remove('vy-is-away', 'vy-is-path');
      ring.setAttribute('data-vy-probe-cap', label);
      ring.setAttribute('data-vy-probe-side', side);
      if (enterRaf) cancelAnimationFrame(enterRaf);
      enterRaf = requestAnimationFrame(function () {
        enterRaf = 0;
        ring.classList.add('vy-is-path');
        wake();
      });
    }

    function hideLabel() {
      if (enterRaf) { cancelAnimationFrame(enterRaf); enterRaf = 0; }
      ring.classList.remove('vy-is-path');
      ring.classList.add('vy-is-away');
      wake();
      if (leaveTimer) clearTimeout(leaveTimer);
      leaveTimer = setTimeout(function () {
        ring.classList.remove('vy-is-away');
        dot.classList.remove('vy-is-path');
        ring.setAttribute('data-vy-probe-cap', 'EXPLORE');
        leaveTimer = null;
        wake();
      }, EXIT_MS);
    }

    document.addEventListener('mousemove', function (e) { mx = e.clientX; my = e.clientY; wake(); }, { passive: true });

    function loop() {
      cursorRaf = 0;
      var targetX = (!magnetTarget || !magRect) ? mx : mx + ((magRect.left + magRect.width / 2) - mx) * magnetStrength;
      var targetY = (!magnetTarget || !magRect) ? my : my + ((magRect.top + magRect.height / 2) - my) * magnetStrength;

      rx += (targetX - rx) * 0.18;                /* Locked: ring following interpolation */
      ry += (targetY - ry) * 0.18;

      dot.style.transform = 'translate(' + targetX + 'px,' + targetY + 'px)';
      ring.style.transform = 'translate(' + rx + 'px,' + ry + 'px)';

      var settled = Math.abs(ry - targetY) < 0.08 && Math.abs(rx - targetX) < 0.08;
      if (!settled || performance.now() - lastWake < 260) cursorRaf = requestAnimationFrame(loop);
    }

    var LINKS = 'a, button, input, .vy-shot-chip, [data-vy-probe]';
    function target(node) { return (!node || typeof node.closest !== 'function') ? null : node.closest(LINKS); }

    document.addEventListener('pointerover', function (ev) {
      var el = target(ev.target);
      if (!el || target(ev.relatedTarget) === el) return;
      var magnetize = el.offsetWidth < 300;
      showLabel(el.dataset.vyProbe || 'EXPLORE', el.dataset.vyProbeSide || 'right');
      magnetTarget = magnetize ? el : null;
      magRect = magnetize ? el.getBoundingClientRect() : null;
      wake();
    });

    document.addEventListener('pointerout', function (ev) {
      if (!target(ev.target) || target(ev.relatedTarget)) return;
      hideLabel();
      magnetTarget = null; magRect = null;
      wake();
    });

    /* On and off switch for the cursor. The choice is remembered. */
    var toggle = document.getElementById('vy-cursor-toggle');
    var state = document.getElementById('vy-cursor-state');
    function apply(off) {
      document.body.setAttribute('data-vy-cursor', off ? 'off' : 'on');
      if (toggle) toggle.setAttribute('aria-pressed', off ? 'true' : 'false');
      if (state) state.textContent = off ? 'Off' : 'On';
    }
    var saved = null;
    try { saved = localStorage.getItem('vy:cursor'); } catch (e) { /* Private mode: carry on with the default */ }
    /* Priority: a remembered choice, then the markup, then on.
       apply() writes the same attribute it reads, so without the markup fallback a
       data-vy-cursor="off" written into the HTML would be erased on load. */
    apply(saved ? saved === 'off'
                : document.documentElement.getAttribute('data-vy-cursor') === 'off'
                  || document.body.getAttribute('data-vy-cursor') === 'off');
    if (toggle) {
      toggle.addEventListener('click', function () {
        var off = document.body.getAttribute('data-vy-cursor') !== 'off';
        apply(off);
        try { localStorage.setItem('vy:cursor', off ? 'off' : 'on'); } catch (e) { /* Ignore */ }
      });
    }
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 5. Nav: reveal on scroll, full screen menu, hover preview
   * ══════════════════════════════════════════════════════════════════════════ */
  var NAV_THUMB_IDLE_SCALE = 1.3;                 /* Locked: the thumbnail entrance settles back from 1.3 to 1 */
  var NAV_TITLE_SHIFT = 104;                      /* Locked: the offset of the two stacked title layers, in percent */
  var NAV_REVEAL_AT = 300;                        /* CFG.scrollNav.revealAt */

  function initNav() {
    var container = document.getElementById('vy-topbar_glide_frame');
    var dropdown = document.getElementById('vy-topbar-glide-flyout');
    var btn = document.getElementById('vy-topbar-glide-panel-btn');
    if (!container || !dropdown || !btn) return;

    var rows = Array.prototype.slice.call(dropdown.querySelectorAll('.vy-hdr-reel-strip'));
    var items = Array.prototype.slice.call(dropdown.querySelectorAll('.vy-hdr-flyout__cell'));
    var open = false;
    var tl = null;

    /* ── Hover timeline (already on the signature easing) ── */
    if (hasGsap && !reduced) {
      rows.forEach(function (row) {
        var thumbs = Array.prototype.slice.call(row.querySelectorAll('.vy-hdr-reel-strip__chip'));
        var images = Array.prototype.slice.call(row.querySelectorAll('.vy-hdr-reel-strip__chip img'));
        var track = row.querySelector('.vy-hdr-reel-strip__head-groove');
        var primary = row.querySelector('.vy-is-prime');
        var accent = row.querySelector('.vy-is-hot');
        if (!thumbs.length || !track || !primary || !accent) return;

        gsap.set(thumbs, {
          clipPath: function (_, t) { return t.classList.contains('vy-is-start') ? 'inset(0% 0% 0% 100%)' : 'inset(0% 100% 0% 0%)'; },
          autoAlpha: 0
        });
        if (images.length) gsap.set(images, { scale: NAV_THUMB_IDLE_SCALE });
        gsap.set(primary, { yPercent: 0 });
        gsap.set(accent, { yPercent: NAV_TITLE_SHIFT });

        var hover = gsap.timeline({ paused: true, defaults: { overwrite: 'auto' } });
        hover.to(thumbs, { clipPath: 'inset(0% 0% 0% 0%)', autoAlpha: 1, ease: EASE, duration: 0.42, stagger: 0.025 }, 0);
        if (images.length) hover.to(images, { scale: 1, ease: EASE, duration: 0.46, stagger: 0.025 }, 0.02);
        hover.to(primary, { yPercent: -NAV_TITLE_SHIFT, ease: EASE, duration: 0.46 }, 0.02)
             .to(accent, { yPercent: 0, ease: EASE, duration: 0.46 }, 0.02);

        function enter() { row.classList.add('vy-is-peeked'); hover.play(); }
        function leave() { row.classList.remove('vy-is-peeked'); hover.reverse(); }
        row.addEventListener('mouseenter', enter);
        row.addEventListener('mouseleave', leave);
        row.addEventListener('focus', enter);
        row.addEventListener('blur', leave);
      });
    }

    /* ── Open and close timeline ──
       The container morph plays backwards on close, so it needs a symmetric curve; the
       contents only ever arrive, so they use the one way curve. Both are in the same family. */
    function openWidth() { return window.innerWidth; }
    function openHeight() { return window.innerHeight; }
    function collapsedTop() { return window.innerWidth <= 767 ? 16 : 30; }

    function buildTl() {
      if (!hasGsap) return null;
      var t = gsap.timeline({
        paused: true,
        onStart: function () {
          dropdown.style.pointerEvents = 'auto';
          container.classList.add('vy-is-panel-open');
          btn.setAttribute('aria-expanded', 'true');
          btn.setAttribute('data-vy-probe', 'CLOSE');
          btn.setAttribute('aria-label', 'Close menu');
          dropdown.setAttribute('aria-hidden', 'false');
        },
        onReverseComplete: function () {
          dropdown.style.pointerEvents = 'none';
          container.classList.remove('vy-is-panel-open');
          btn.setAttribute('aria-expanded', 'false');
          btn.setAttribute('data-vy-probe', 'OPEN');
          btn.setAttribute('aria-label', 'Open menu');
          dropdown.setAttribute('aria-hidden', 'true');
        }
      });

      t.to(container, { height: 2, top: collapsedTop, borderRadius: 0, ease: 'expo.inOut', duration: 0.2 }, 0)
       .to(container, { width: openWidth, ease: 'expo.inOut', duration: 0.25 }, 0.1)
       .to(container, { height: openHeight, top: 0, borderRadius: 0, ease: 'expo.inOut', duration: 0.35 }, 0.3)
       .to(dropdown, { y: 0, opacity: 1, filter: 'blur(0px)', clipPath: 'inset(0% 0% 0% 0%)', ease: EASE, duration: DUR_MICRO }, 0.35)
       .to(items, { opacity: 1, y: 0, filter: 'blur(0px)', ease: EASE, stagger: 0.02, duration: 0.35 }, 0.4);
      return t;
    }

    function setOpen(next) {
      open = next;
      if (!hasGsap || reduced) {
        /* Fallback with no GSAP or with reduced motion: switch state without animating. The menu
     has to open either way. */
        container.classList.toggle('vy-is-panel-open', open);
        container.style.height = open ? '100vh' : '';
        container.style.width = open ? '100vw' : '';
        container.style.top = open ? '0px' : '';
        dropdown.style.opacity = open ? '1' : '0';
        dropdown.style.transform = 'none';
        dropdown.style.pointerEvents = open ? 'auto' : 'none';
        items.forEach(function (it) { it.style.opacity = open ? '1' : '0'; it.style.transform = 'none'; });
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        dropdown.setAttribute('aria-hidden', open ? 'false' : 'true');
        return;
      }
      if (!tl) tl = buildTl();
      if (open) tl.play(); else tl.reverse();
    }

    btn.addEventListener('click', function () { setOpen(!open); });

    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && open) { setOpen(false); btn.focus(); }
    });

    items.forEach(function (a) {
      a.addEventListener('click', function () { if (open) setOpen(false); });
    });

    window.addEventListener('resize', function () {
      if (open && tl) { tl.pause(); tl.invalidate(); tl.progress(1); }
    }, { passive: true });

    /* ── Reveal on scroll ── */
    var shown = false;
    function sync() {
      var y = (window.__lenis && typeof window.__lenis.scroll === 'number') ? window.__lenis.scroll : window.scrollY;
      /* The subpages keep the nav visible at all times.
         index has a full screen hero, so revealing the nav on scroll is intended there. A
         subpage has a shorter hero and a short page, and would otherwise arrive with no
         visible way to reach another page. The hidden state is also transparent and click
         through, so it looks present while not being clickable.
         The branch is on the subpage body class alone: neither the reveal threshold nor the
         behaviour on index changes. */
      var want = y > NAV_REVEAL_AT || window.innerWidth <= 767
        || document.body.classList.contains('vy-leafview');
      if (want === shown) return;
      shown = want;
      container.classList.toggle('vy-hdr-arrive', want);
      container.classList.toggle('vy-hdr-depart', !want);
      if (want) container.removeAttribute('inert');
      else if (!open) container.setAttribute('inert', '');
      requestAnimationFrame(sync);
    }

    if (hasGsap && window.ScrollTrigger) {
      ScrollTrigger.create({ start: NAV_REVEAL_AT + ' top', end: 'max', onUpdate: sync, onToggle: sync });
    }
    (function poll() { sync(); requestAnimationFrame(poll); })();
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 6. Scroll driven colour shift, at one boundary only (hero into problem)
   * The default is open; this arms the closed state. If arming fails the content stays.
   * ══════════════════════════════════════════════════════════════════════════ */
  function initDarkShift() {
    var mask = document.getElementById('vy-night-hull-veil');
    var overlay = document.getElementById('vy-dw-black-hud');
    var hero = document.querySelector('.vy-banner-band');
    if (!mask || !hero) return;
    if (reduced || !hasGsap || !window.ScrollTrigger || window.innerWidth <= 768) return;

    gsap.set(mask, { clipPath: 'inset(100% 0 0 0)' });     /* This is the first close, which is the arming */

    ScrollTrigger.create({
      trigger: hero,
      start: 'bottom bottom',
      end: 'bottom top',
      scrub: true,
      invalidateOnRefresh: true,
      onUpdate: function (self) {
        var p = self.progress;
        mask.style.clipPath = 'inset(' + ((1 - p) * 100).toFixed(2) + '% 0 0 0)';
        if (overlay) overlay.style.opacity = String((1 - p) * 0.5);
      },
      onLeave: function () { mask.style.clipPath = 'inset(0% 0 0 0)'; if (overlay) overlay.style.opacity = '0'; }
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 7. Reveal masks: the features component and the global rise class
   * Do not hide these with clip-path. A clipped element reports zero intersection, so the
   * style that hides it also removes the signal that would reveal it.
   * ══════════════════════════════════════════════════════════════════════════ */
  function initReveal() {
    var inners = Array.prototype.slice.call(document.querySelectorAll('.vy-rise-mid'));

    if (hasGsap && window.ScrollTrigger && !reduced && inners.length) {
      inners.forEach(function (el) {
        var wrap = el.closest('.vy-rise-case') || el.parentElement;
        gsap.set(el, { yPercent: 130 });
        /* Synchronised with the signature easing and the section duration. A mask reveal only
           plays one way, so changing the curve changes the speed but not the behaviour. */
        gsap.to(el, {
          yPercent: 0, ease: EASE, duration: DUR_SECTION,
          scrollTrigger: { trigger: wrap, start: 'top 88%', once: true }
        });
      });
    } else {
      inners.forEach(function (el) { el.style.transform = 'none'; });
    }

    /* The clipped media reveal in the features section (the third phase of that component) */
    var media = document.querySelector('.vy-axis-film-pod');
    if (media && hasGsap && window.ScrollTrigger && !reduced) {
      gsap.set(media, { clipPath: 'inset(0% 0% 100% 0%)' });
      gsap.to(media, {
        clipPath: 'inset(0% 0% 0% 0%)', ease: EASE, duration: DUR_HERO,
        scrollTrigger: { trigger: media, start: 'top 85%', once: true }
      });
    }

    /* The global rise class: visible by default, armed here */
    var rises = Array.prototype.slice.call(document.querySelectorAll('.vy-rise'));
    if (!rises.length || reduced) return;
    document.body.classList.add('vy-armed');

    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (!e.isIntersecting) return;
        e.target.classList.add('is-in');
        io.unobserve(e.target);
      });
    }, { rootMargin: '0px 0px -12% 0px', threshold: 0.01 });

    rises.forEach(function (el) { io.observe(el); });

    /* Backstop: the document has to be readable even if the observer never fires */
    setTimeout(function () {
      if (!document.querySelector('.vy-rise.is-in')) {
        rises.forEach(function (el) { el.classList.add('is-in'); });
      }
    }, 2600);
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 8. Marquee ticker (speed follows the scroll; direction comes from a data attribute)
   * ══════════════════════════════════════════════════════════════════════════ */
  function initMarquee() {
    var inners = Array.prototype.slice.call(document.querySelectorAll('.vy-js-glide-crawl'));
    if (!inners.length) return;

    var BASE_SPEED = 1;                            /* Locked: px per tick baseline */

    var tracks = inners.map(function (inner) {
      var dirAttr = parseFloat(inner.getAttribute('data-vy-belt-dir'));
      var first = inner.querySelector('.vy-crawl-kit');
      if (!first) return null;

      /* To have something to wrap to, the set has to be wider than its container, so it is
     duplicated until it is */
      var need = Math.max(2, Math.ceil((inner.parentElement.offsetWidth * 2) / Math.max(1, first.offsetWidth)) + 1);
      for (var i = inner.querySelectorAll('.vy-crawl-kit').length; i < need; i++) {
        inner.appendChild(first.cloneNode(true));
      }

      return {
        inner: inner,
        setWidth: first.offsetWidth,
        xPos: 0,
        dir: (isFinite(dirAttr) && dirAttr !== 0) ? Math.sign(dirAttr) : -1
      };
    }).filter(Boolean);

    if (reduced) return;

    (function tick() {
      var vel = window.__vyScrollVel || 0;
      tracks.forEach(function (t) {
        if (!t.setWidth) { t.setWidth = t.inner.querySelector('.vy-crawl-kit').offsetWidth; return; }
        /* Scroll speed is added to the marquee speed */
        t.xPos += t.dir * (BASE_SPEED + Math.abs(vel) * 0.42);
        if (Math.abs(t.xPos) >= t.setWidth) t.xPos += t.setWidth * -t.dir;   /* Locked: wrap threshold */
        t.inner.style.transform = 'translate3d(' + t.xPos.toFixed(2) + 'px,0,0)';
      });
      requestAnimationFrame(tick);
    })();
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 9. Footer letter roll (two stacked copies with a per letter delay)
   * ══════════════════════════════════════════════════════════════════════════ */
  function initFooterRoll() {
    var links = document.querySelectorAll('[data-vy-roll]');
    Array.prototype.forEach.call(links, function (a) {
      var text = a.textContent.trim();
      a.textContent = '';
      text.split('').forEach(function (ch, i) {
        var flow = document.createElement('span');
        flow.className = 'vy-flow';
        flow.style.setProperty('--i', String(i));
        var base = document.createElement('span');
        base.textContent = ch === ' ' ? ' ' : ch;
        var echo = document.createElement('span');
        echo.className = 'vy-echo';
        echo.setAttribute('aria-hidden', 'true');
        echo.textContent = base.textContent;
        flow.appendChild(base);
        flow.appendChild(echo);
        a.appendChild(flow);
      });
      a.setAttribute('aria-label', text);
    });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 10. Call to action form: a static slot with no backend. Add your action here.
   * ══════════════════════════════════════════════════════════════════════════ */
  function initCtaForm() {
    var form = document.getElementById('vy-cta-form');
    var email = document.getElementById('vy-cta-email');
    var note = document.getElementById('vy-cta-note');
    if (!form || !email || !note) return;

    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var v = email.value.trim();
      if (!v || !/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(v)) {
        note.textContent = 'That address does not look right.';
        email.focus();
        return;
      }
      note.textContent = 'Nothing was sent — this form has no endpoint yet. Add your own before launch.';
      form.reset();
    });

    email.addEventListener('input', function () { if (note.textContent) note.textContent = ''; });
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * boot
   * ══════════════════════════════════════════════════════════════════════════ */
  function boot() {
    initCursor();
    initNav();
    initReveal();
    initMarquee();
    initFooterRoll();
    initCtaForm();
    initDarkShift();
    heroOverlayParallax();

    var shell = document.querySelector('.vy-js-view-swap-rig');
    var curtain = (shell && !reduced) ? new PixelCurtain(shell).build() : null;

    /* The handoff flag is read by the inline script in each subpage head, which covers the
       first frames with CSS before this file loads. All that happens here is clearing it: left
       set, the next direct visit would show a cover with nothing behind it. Session storage is
       per tab, so closing the tab clears it anyway. */
    handoffClear();

    initPageLinks(shell, curtain);

    initGate(function () {
      if (curtain) {
        shell.classList.add('vy-is-live');
        curtain.prime();
        /* Immediately after priming, the cell grid covers the screen, and that is when the CSS
           cover is removed. In the other order the arriving page is visible for one frame. */
        document.documentElement.classList.remove('vy-arriving');
        armArriveFailsafe(shell, curtain);
        requestAnimationFrame(function () {
          curtain.open().then(function () {
            releaseCurtain(shell, curtain);
            heroChoreography();
          });
        });
      } else {
        document.documentElement.classList.remove('vy-arriving');
        heroChoreography();
      }
      document.dispatchEvent(new CustomEvent('vy:gate-open'));
      if (hasGsap && window.ScrollTrigger) ScrollTrigger.refresh();
    });

    /* Returning through the back button: the curtain that covered the exit is restored with
       the page. Not clearing it here leaves the visitor on a black screen. */
    window.addEventListener('pageshow', function (e) {
      if (!e.persisted) return;
      departing = false;
      handoffClear();
      releaseCurtain(shell, curtain);
      if (window.__lenis && typeof window.__lenis.start === 'function') window.__lenis.start();
      if (hasGsap && window.ScrollTrigger) ScrollTrigger.refresh();
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
