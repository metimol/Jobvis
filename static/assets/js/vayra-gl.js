/* ============================================================================
 * Vayra — WebGL layer
 *
 *   1. Hero: the signature layered parallax. Cover fit in the shader, pointer parallax,
 *      crossfade between slides.
 *   2. Gallery: a GL overlay on the archive cards. Scroll driven warp plus a lens bulge
 *      that follows the cursor.
 *
 * Locked: the shader uniforms and the cover fit maths. Changing them changes the
 * behaviour of the transition, not just how it looks. Adjust durations in
 * _tokens-bridge.css instead.
 *
 * three is loaded once and each component owns its own canvas.
 *
 * Textures: while an image slot is still a placeholder there is no texture to load.
 *   - The hero falls back to a gradient built from the palette, so the signature motion
 *     is visible while you are still choosing images. Point the slide at a real file and
 *     it becomes a photograph with no code change.
 *   - The gallery keeps its DOM fallback instead, because a warp over flat colour is
 *     invisible. It switches itself on as soon as one real texture loads.
 * ============================================================================ */
(function () {
  'use strict';

  var reduced = (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)
    || document.documentElement.getAttribute('data-vy-motion') === 'off'
    || (document.body && document.body.getAttribute('data-vy-motion') === 'off');
  var hasGsap = typeof window.gsap !== 'undefined';

  /* ── Diagnostics ─────────────────────────────────────────────────────────────
     Both GL paths switch themselves off for legitimate reasons (no texture, no WebGL,
     small screen, reduced motion). Looking at the screen alone you cannot tell "wired
     but dormant" from "not implemented", so each path reports its state and its reason
     on window.__vyGL. */
  var glStatus = window.__vyGL = { hero: 'pending', archive: 'pending' };
  function say(path, state) { glStatus[path] = state; }

  /* The hero has several success paths, so its state is confirmed from the result
     (is there a canvas on screen) rather than declared up front. */
  setTimeout(function () {
    if (glStatus.hero !== 'pending') return;
    glStatus.hero = document.querySelector('#vy-webgl-frame canvas')
      ? 'live'
      : (reduced ? 'skipped:reduced-motion' : 'dom-fallback');
  }, 4000);

  function whenThree(fn) {
    if (window.THREE) { fn(); return; }
    document.addEventListener('vy:three-ready', fn, { once: true });
    /* If three never arrives, give up quietly: the DOM fallback is already carrying the screen */
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * Placeholder texture: a vertical ramp built from the palette.
   * It is not a generated image; it is the GL equivalent of the flat panel used for
   * placeholder images elsewhere.
   * ══════════════════════════════════════════════════════════════════════════ */
  /* The lightness range is wide enough to separate the plane from the page background.
     Too close and the hero reads as an empty black screen, which hides the very motion
     this section exists to show. The direction alternates per slide so a transition reads
     as a change of value. None of this is used once real images are in place. */
  var RAMPS = [
    [0x12, 0x38], [0x30, 0x14], [0x16, 0x40], [0x36, 0x16], [0x14, 0x2C]
  ];

  function placeholderTexture(index) {
    var H = 256, W = 2;
    var pair = RAMPS[index % RAMPS.length];
    var data = new Uint8Array(W * H * 4);
    for (var y = 0; y < H; y++) {
      var t = y / (H - 1);
      var v = Math.round(pair[0] + (pair[1] - pair[0]) * t);
      for (var x = 0; x < W; x++) {
        var o = (y * W + x) * 4;
        data[o] = v; data[o + 1] = v; data[o + 2] = v + 1; data[o + 3] = 255;
      }
    }
    var tex = new THREE.DataTexture(data, W, H, THREE.RGBAFormat);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    tex.userData.placeholder = true;
    return tex;
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 1. Hero — WebGL manager and slideshow
   * ══════════════════════════════════════════════════════════════════════════ */
  function WebGLManager(containerId, urls) {
    var host = document.getElementById(containerId);
    this.container = host;
    this.imageUrls = urls;
    this.activeTextureIndex = 0;
    this.renderPaused = false;
    this.textures = [];
    this.textureResolutions = {};

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    this.textureLoader = new THREE.TextureLoader();

    var size = this.getRenderSize();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(size.width, size.height);
    this.container.appendChild(this.renderer.domElement);

    this.mouse = new THREE.Vector2(0, 0);
    this.targetMouse = new THREE.Vector2(0, 0);

    this.initShader();
    this.loadTextures();

    var self = this;
    this._resizeRaf = 0;
    window.addEventListener('resize', function () {
      if (self._resizeRaf) return;
      self._resizeRaf = requestAnimationFrame(function () { self._resizeRaf = 0; self.onResize(); });
    }, { passive: true });
    window.addEventListener('mousemove', function (e) {
      self.targetMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      self.targetMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
    }, { passive: true });
  }

  WebGLManager.prototype.getRenderSize = function () {
    var box = this.container;
    if (!box) return { width: window.innerWidth, height: window.innerHeight };
    return {
      width: Math.max(1, Math.round(box.clientWidth || window.innerWidth)),
      height: Math.max(1, Math.round(box.clientHeight || window.innerHeight))
    };
  };

  WebGLManager.prototype.initShader = function () {
    var size = this.getRenderSize();

    /* ── Hero transition shader (independent implementation) ──
       Clip space pass through, per frame aspect fill, and a simplex noise warp across the
       crossfade. The fill is computed per slot: sharing one computation squashes the image
       when a desktop and a mobile asset swap mid transition.
       simplexNoise2D is the public Ashima Arts / Stefan Gustavson algorithm (MIT) and is
       kept verbatim — see LICENSES.md. Everything else is written from the behaviour spec. */
    var vertexShader = [
      'varying vec2 vFrameUv;',
      'void main(){ vFrameUv = uv; gl_Position = vec4(position, 1.0); }'
    ].join('\n');

    var fragmentShader = [
      'uniform sampler2D uFrameA;',
      'uniform sampler2D uFrameB;',
      'uniform float uBlend;',
      'uniform float uTravel;',
      'uniform float uReady;',
      'uniform vec2 uViewport;',
      'uniform vec2 uFrameASize;',
      'uniform vec2 uFrameBSize;',
      'uniform vec2 uPointer;',
      'varying vec2 vFrameUv;',
      'vec3 gradPermute(vec3 x){ return mod((x * 34.0 + 1.0) * x, 289.0); }',
      'float simplexNoise2D(vec2 point){',
      '  const vec4 skew = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);',
      '  vec2 cellOrigin = floor(point + dot(point, skew.yy));',
      '  vec2 localPos = point - cellOrigin + dot(cellOrigin, skew.xx);',
      '  vec2 secondCorner = (localPos.x > localPos.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);',
      '  vec4 offsets = localPos.xyxy + skew.xxzz;',
      '  offsets.xy -= secondCorner;',
      '  cellOrigin = mod(cellOrigin, 289.0);',
      '  vec3 perm = gradPermute(gradPermute(cellOrigin.y + vec3(0.0, secondCorner.y, 1.0)) + cellOrigin.x + vec3(0.0, secondCorner.x, 1.0));',
      '  vec3 falloff = max(0.5 - vec3(dot(localPos, localPos), dot(offsets.xy, offsets.xy), dot(offsets.zw, offsets.zw)), 0.0);',
      '  falloff = falloff * falloff; falloff = falloff * falloff;',
      '  vec3 gradX = 2.0 * fract(perm * skew.www) - 1.0;',
      '  vec3 gradAbs = abs(gradX) - 0.5;',
      '  vec3 gradRound = floor(gradX + 0.5);',
      '  vec3 gradFinal = gradX - gradRound;',
      '  falloff *= 1.79284291400159 - 0.85373472095314 * (gradFinal * gradFinal + gradAbs * gradAbs);',
      '  vec3 grad;',
      '  grad.x = gradFinal.x * localPos.x + gradAbs.x * localPos.y;',
      '  grad.yz = gradFinal.yz * offsets.xz + gradAbs.yz * offsets.yw;',
      '  return 130.0 * dot(falloff, grad);',
      '}',
      'vec2 fillFrame(vec2 uv, vec2 frameSize){',
      '  float viewAspect = uViewport.x / uViewport.y;',
      '  float frameAspect = frameSize.x / frameSize.y;',
      '  vec2 axisScale = viewAspect > frameAspect',
      '    ? vec2(1.0, frameAspect / viewAspect)',
      '    : vec2(viewAspect / frameAspect, 1.0);',
      '  return (uv - 0.5) * axisScale + 0.5;',
      '}',
      'float travelWeight(float t){ return t * (1.0 - t); }',
      'vec2 layerUv(vec2 uv, float phase, float dir, float smear){',
      '  vec2 toward = uv + (vec2(0.5) - uv) * (phase * 0.14);',
      '  return toward + vec2(0.0, dir * phase * 0.28) + smear;',
      '}',
      'void main(){',
      '  if (uReady < 0.5) { gl_FragColor = vec4(0.0); return; }',
      '  float p = uBlend;',
      '  vec2 lens = uPointer * 0.014;',
      '  vec2 uvA = (fillFrame(vFrameUv, uFrameASize) - 0.5) * 0.95 + 0.5 - lens;',
      '  vec2 uvB = (fillFrame(vFrameUv, uFrameBSize) - 0.5) * 0.95 + 0.5 - lens;',
      '  float bell = travelWeight(p);',
      '  float grain = (p > 0.001 && p < 0.999)',
      '    ? simplexNoise2D((uvA + uvB) * 1.5 + p * 2.0)',
      '    : 0.0;',
      '  float smear = grain * bell * 0.32;',
      '  float slip = bell * (grain + 1.0) * 0.042;',
      '  vec4 frameA = texture2D(uFrameA, layerUv(uvA, p, uTravel, smear) + vec2(0.0, slip));',
      '  vec4 frameB = texture2D(uFrameB, layerUv(uvB, 1.0 - p, -uTravel, smear) - vec2(0.0, slip));',
      '  gl_FragColor = mix(frameA, frameB, smoothstep(0.0, 1.0, p));',
      '}'
    ].join('\n');

    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uFrameA: { value: null }, uFrameB: { value: null },
        uBlend: { value: 0 }, uTravel: { value: 1 }, uReady: { value: 0 },
        uViewport: { value: new THREE.Vector2(size.width, size.height) },
        uFrameASize: { value: new THREE.Vector2(1920, 1080) },
        uFrameBSize: { value: new THREE.Vector2(1920, 1080) },
        uPointer: { value: new THREE.Vector2(0, 0) }
      },
      vertexShader: vertexShader,
      fragmentShader: fragmentShader
    });

    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.scene.add(this.mesh);
  };

  WebGLManager.prototype.rememberTextureResolution = function (index, w, h) {
    if (!w || !h) return;
    this.textureResolutions[index] = { width: w, height: h };
    if (index === this.activeTextureIndex) {
      this.applyTextureResolution(index, null, 1);
      this.applyTextureResolution(index, null, 2);
    }
  };

  WebGLManager.prototype.applyTextureResolution = function (index, texture, slot) {
    var name = slot === 2 ? 'uFrameBSize' : 'uFrameASize';
    var known = this.textureResolutions[index];
    if (known) { this.material.uniforms[name].value.set(known.width, known.height); return; }
    if (texture && texture.image && texture.image.width) {
      this.material.uniforms[name].value.set(texture.image.width, texture.image.height);
    }
  };

  WebGLManager.prototype.loadTextures = function () {
    this.loadTextureAt(0);
    if (this.imageUrls.length > 1) {
      var self = this;
      requestAnimationFrame(function () { self.loadTextureAt(1); });
    }
  };

  WebGLManager.prototype.loadTextureAt = function (index) {
    var self = this;
    if (this.textures[index]) return Promise.resolve(this.textures[index]);

    var url = this.imageUrls[index];

    /* No asset yet: the palette ramp texture stands in.
       Point the slide at a real path and the loader branch below takes over, no code change. */
    if (!url) {
      var ph = placeholderTexture(index);
      this.textures[index] = ph;
      this.rememberTextureResolution(index, 1600, 1000);
      if (index === 0) {
        this.material.uniforms.uFrameA.value = ph;
        this.material.uniforms.uFrameB.value = ph;
        this.material.uniforms.uReady.value = 1;
      }
      return Promise.resolve(ph);
    }

    return new Promise(function (resolve) {
      self.textureLoader.load(url, function (texture) {
        texture.generateMipmaps = true;
        texture.minFilter = THREE.LinearMipmapLinearFilter;
        self.textures[index] = texture;
        if (texture.image) self.rememberTextureResolution(index, texture.image.width, texture.image.height);
        if (index === 0) {
          self.material.uniforms.uFrameA.value = texture;
          self.material.uniforms.uFrameB.value = texture;
          self.material.uniforms.uReady.value = 1;
          self.applyTextureResolution(index, texture, 1);
          self.applyTextureResolution(index, texture, 2);
        }
        resolve(texture);
      }, undefined, function () {
        /* There is a path but it failed to load. Fall back to flat colour rather than to black. */
        var fb = placeholderTexture(index);
        self.textures[index] = fb;
        if (index === 0) {
          self.material.uniforms.uFrameA.value = fb;
          self.material.uniforms.uFrameB.value = fb;
          self.material.uniforms.uReady.value = 1;
        }
        resolve(fb);
      });
    });
  };

  WebGLManager.prototype.onResize = function () {
    var size = this.getRenderSize();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(size.width, size.height);
    this.material.uniforms.uViewport.value.set(size.width, size.height);
  };

  WebGLManager.prototype.render = function () {
    /* Stop drawing once the hero leaves the screen */
    var y = (window.__lenis && typeof window.__lenis.scroll === 'number') ? window.__lenis.scroll : window.scrollY;
    if (y > window.innerHeight * 1.1) {
      if (!this.renderPaused) { this.renderPaused = true; this.container.style.visibility = 'hidden'; }
      return;
    }
    if (this.renderPaused) { this.renderPaused = false; this.container.style.visibility = ''; }

    this.mouse.lerp(this.targetMouse, 0.05);      /* Locked: parallax damping */
    this.material.uniforms.uPointer.value.copy(this.mouse);
    this.renderer.render(this.scene, this.camera);
  };

  WebGLManager.prototype.transition = function (from, to, direction, onComplete) {
    var self = this;
    var token = {};
    this._token = token;

    Promise.all([this.loadTextureAt(from), this.loadTextureAt(to)]).then(function (pair) {
      if (self._token !== token) return;
      if (!pair[0] || !pair[1]) { if (onComplete) onComplete(); return; }

      self.material.uniforms.uFrameA.value = pair[0];
      self.material.uniforms.uFrameB.value = pair[1];
      self.applyTextureResolution(from, pair[0], 1);
      self.applyTextureResolution(to, pair[1], 2);
      self.material.uniforms.uTravel.value = direction;
      self.material.uniforms.uBlend.value = 0;
      self.material.uniforms.uReady.value = 1;
      self.activeTextureIndex = from;

      function finish() {
        self.material.uniforms.uFrameA.value = pair[1];
        self.material.uniforms.uBlend.value = 0;
        self.activeTextureIndex = to;
        self.applyTextureResolution(to, pair[1], 1);
        self.applyTextureResolution(to, pair[1], 2);
        if (onComplete) onComplete();
      }

      if (!hasGsap || reduced) { self.material.uniforms.uBlend.value = 1; finish(); return; }

      /* Deliberately not the signature easing. With an ease out curve the progress value
         passes 0.5 within the first fifteen percent of the duration, and since the warp and
         the crossfade both peak at 0.5 the whole transition collapses into a single frame.
         This is a case where the curve is part of the behaviour. Only the duration was
         brought in line with the hero duration token. */
      gsap.to(self.material.uniforms.uBlend, {
        value: 1, duration: 1.2, ease: 'expo.inOut', onComplete: finish
      });
    });
  };

  /* ── Slideshow ── */
  function Slideshow(webgl, total) {
    this.current = 0;
    this.total = total;
    this.webgl = webgl;
    this.animating = false;
    this.pending = null;
  }

  Slideshow.prototype._commit = function (prev, direction) {
    var self = this;
    var thumbs = document.querySelectorAll('.vy-shot-chip');
    Array.prototype.forEach.call(thumbs, function (t, i) { t.classList.toggle('active', i === self.current); });

    var cur = document.querySelector('.vy-now-shot');
    if (cur) cur.textContent = String(self.current + 1).padStart(2, '0');
    var title = document.querySelector('.vy-shot-head');
    if (title && window.__vySlides && window.__vySlides[self.current]) {
      title.textContent = window.__vySlides[self.current].title;
    }
    updateDragLines(self.current, self.total);
    updateProgress(self.current, self.total);

    this.webgl.transition(prev, this.current, direction, function () {
      self.animating = false;
      if (self.pending) {
        var q = self.pending; self.pending = null;
        setTimeout(function () { q.type === 'goto' ? self.goTo(q.index) : self.navigate(q.direction); }, 50);
      }
    });
  };

  Slideshow.prototype.goTo = function (index) {
    if (this.animating) { this.pending = { type: 'goto', index: index }; return false; }
    if (index === this.current) return false;
    var prev = this.current;
    var dir = index > prev ? 1 : -1;
    this.current = index;
    this.animating = true;
    this._commit(prev, dir);
    return true;
  };

  Slideshow.prototype.navigate = function (dir) {
    if (this.animating) { this.pending = { type: 'nav', direction: dir }; return false; }
    var prev = this.current;
    this.current = dir === 1
      ? (prev < this.total - 1 ? prev + 1 : 0)
      : (prev > 0 ? prev - 1 : this.total - 1);
    this.animating = true;
    this._commit(prev, dir);
    return true;
  };

  var dragLines = [];
  function buildDragLines() {
    var host = document.querySelector('.vy-pull-cue .vy-rules-frame');
    if (!host) return;
    host.innerHTML = '';
    dragLines = [];
    for (var i = 0; i < 60; i++) {                /* Locked: 60 lines is the density of this component */
      var line = document.createElement('div');
      line.className = 'vy-pull-rule';
      host.appendChild(line);
      dragLines.push(line);
    }
  }

  function updateDragLines(current, total) {
    if (!dragLines.length) return;
    var center = ((current + 0.5) / total) * dragLines.length;
    dragLines.forEach(function (line, i) {
      var d = Math.abs(i - center);
      var f = Math.max(0, 1 - d / 8);
      line.style.height = (12 + f * 32) + 'px';
      line.style.backgroundColor = f > 0.05 ? 'var(--vy-text-primary)' : 'var(--vy-line)';
      line.style.opacity = String(0.3 + f * 0.7);
    });
  }

  function updateProgress(current, total) {
    var lead = document.querySelector('.vy-meter-brief');
    if (lead) lead.style.width = (((current + 1) / total) * 100) + '%';
  }

  function initHero() {
    var container = document.getElementById('vy-webgl-frame');
    if (!container || !window.THREE) return;

    var slideEls = Array.prototype.slice.call(document.querySelectorAll('.vy-shots .vy-shot__pic'));
    if (!slideEls.length) return;

    var slides = slideEls.map(function (el) {
      return {
        url: el.getAttribute('data-vy-pic') || '',
        thumb: el.getAttribute('data-vy-chip') || '',
        title: el.getAttribute('data-vy-head') || ''
      };
    });
    window.__vySlides = slides;

    var webgl;
    try {
      webgl = new WebGLManager('vy-webgl-frame', slides.map(function (s) { return s.url; }));
    } catch (err) {
      return;   /* No WebGL: the hero still works from the overlay UI alone, and nothing is logged as an error */
    }

    var show = new Slideshow(webgl, slides.length);

    /* Thumbnail strip */
    var strip = document.querySelector('.vy-shot-chips');
    if (strip) {
      strip.innerHTML = '';
      slides.forEach(function (s, i) {
        var thumb = document.createElement('button');
        thumb.type = 'button';
        thumb.className = 'vy-shot-chip' + (i === 0 ? ' active' : '');
        thumb.setAttribute('aria-label', 'Show frame ' + (i + 1) + ': ' + s.title);
        if (s.thumb) thumb.style.backgroundImage = 'url("' + s.thumb + '")';
        thumb.addEventListener('click', function () { stopAuto(); show.goTo(i); startAuto(); });
        strip.appendChild(thumb);
      });
    }

    var totalEl = document.querySelector('.vy-sum-shots');
    if (totalEl) totalEl.textContent = String(slides.length).padStart(2, '0');

    buildDragLines();
    updateDragLines(0, slides.length);
    updateProgress(0, slides.length);

    var prevBtn = document.querySelector('.vy-prev-shot');
    var nextBtn = document.querySelector('.vy-next-shot');
    if (prevBtn) prevBtn.addEventListener('click', function () { stopAuto(); show.navigate(-1); startAuto(); });
    if (nextBtn) nextBtn.addEventListener('click', function () { stopAuto(); show.navigate(1); startAuto(); });

    /* Keyboard, only while the hero is on screen */
    document.addEventListener('keydown', function (e) {
      if (window.scrollY > window.innerHeight * 0.5) return;
      if (e.key === 'ArrowRight') show.navigate(1);
      else if (e.key === 'ArrowLeft') show.navigate(-1);
    });

    /* Touch swipe, only inside the hero (below it the page has to scroll) */
    var touchX = 0;
    document.addEventListener('touchstart', function (e) { touchX = e.touches[0].clientX; }, { passive: true });
    document.addEventListener('touchend', function (e) {
      var d = touchX - e.changedTouches[0].clientX;
      if (Math.abs(d) > 50 && window.scrollY < window.innerHeight) show.navigate(d > 0 ? 1 : -1);
    }, { passive: true });

    /* Auto advance */
    var auto = null;
    function startAuto() { if (reduced) return; stopAuto(); auto = setInterval(function () { show.navigate(1); }, 5200); }
    function stopAuto() { if (auto) { clearInterval(auto); auto = null; } }

    var thumbsArea = document.querySelector('.vy-chips-frame');
    if (thumbsArea) {
      thumbsArea.addEventListener('mouseenter', stopAuto);
      thumbsArea.addEventListener('mouseleave', startAuto);
    }
    document.addEventListener('visibilitychange', function () { document.hidden ? stopAuto() : startAuto(); });

    document.addEventListener('vy:gate-open', startAuto, { once: true });
    setTimeout(startAuto, 3200);                  /* The slides advance even if the gate event is missed */

    (function frame() { webgl.render(); requestAnimationFrame(frame); })();
  }

  /* ══════════════════════════════════════════════════════════════════════════
   * 2. Gallery — GL overlay on the cards
   * A warp driven by scroll velocity plus a bulge that follows the pointer. Only the card
   * rectangles are drawn.
   * ══════════════════════════════════════════════════════════════════════════ */
  function initArchiveGL() {
    var canvas = document.getElementById('vy-gl-vault');
    if (!canvas) return say('archive', 'skipped:no-canvas');
    if (!window.THREE) return say('archive', 'skipped:no-three');
    if (reduced) return say('archive', 'skipped:reduced-motion');
    if (window.innerWidth <= 820) return say('archive', 'skipped:mobile');  /* Mobile keeps the DOM fallback */

    var items = Array.prototype.slice.call(document.querySelectorAll('.vy-vault .vy-gl-pic'));
    /* Only cards with a real texture are handed to GL. If every card is still a placeholder,
       switching GL on shows flat colour with an invisible warp, and the DOM fallback is the
       more honest result.
       The test is the status attribute, not the presence of a src: a pending slot still has
       the placeholder image as its src, so testing the path would turn GL on with no assets. */
    var live = items.filter(function (el) {
      return el.getAttribute('data-status') !== 'pending' && !!(el.currentSrc || el.getAttribute('src'));
    });
    if (!live.length) return say('archive', 'dormant:all-slots-pending');

    var W = window.innerWidth, H = window.innerHeight;
    var renderer, scene, camera, geometry, loader;
    try {
      renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    } catch (err) { return say('archive', 'skipped:no-webgl'); }

    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(W, H);
    scene = new THREE.Scene();
    camera = new THREE.OrthographicCamera(W / -2, W / 2, H / 2, H / -2, 1, 1000);
    camera.position.z = 10;
    geometry = new THREE.PlaneGeometry(1, 1, 24, 24);
    loader = new THREE.TextureLoader();

    /* ── Locked: the shader itself ──
       The vertex stage warps in Z with scroll speed; the fragment stage drifts vertical
       slices and adds the pointer bulge. Colour is always a single sample, so no channel
       separation can occur, which is what keeps this consistent with the palette. */
    var vertexShader = [
      'precision highp float;',
      '#define PI 3.1415926535897932384626433832795',
      'uniform float uStrength;',
      'uniform vec2 uViewportSizes;',
      'uniform vec2 uVelocityDir;',
      'varying vec2 vUv;',
      'void main(){',
      '  vec3 newPosition = position;',
      '  vec4 mvPosition = modelViewMatrix * vec4(newPosition, 1.0);',
      '  float waveY = -sin(mvPosition.y / uViewportSizes.y * PI + PI / 2.0) * abs(uVelocityDir.y);',
      '  float waveX =  sin(mvPosition.x / uViewportSizes.x * PI + PI / 2.0) * abs(uVelocityDir.x);',
      '  float wave  = (waveY + waveX) * uStrength;',
      '  mvPosition.x -= wave * 0.03;',            /* Restrained: one third of the warp amplitude */
      '  vUv = uv;',
      '  gl_Position = projectionMatrix * mvPosition;',
      '}'
    ].join('\n');

    var fragmentShader = [
      'precision highp float;',
      'uniform sampler2D uTexture;',
      'uniform float uStrength;',
      'uniform float uOpacity;',
      'uniform vec2 uCoverScale;',
      'uniform vec2 uVelocityDir;',
      'uniform vec2 uLocalMouse;',
      'uniform float uHover;',
      'uniform float uDriftMul;',
      'varying vec2 vUv;',
      'float sliceSeed(float c){ return fract(sin(c * 12.9898) * 43758.5453); }',
      'void main(){',
      '  vec2 coverUv = (vUv - 0.5) * uCoverScale + 0.5;',
      '  float slices = 180.0;',
      '  float col    = floor(vUv.x * slices);',
      '  float phase  = sliceSeed(col) - 0.5;',
      '  float dir    = uVelocityDir.y >= 0.0 ? 1.0 : -1.0;',
      '  float drift  = phase * uStrength * uDriftMul * dir;',
      '  vec2 uv = coverUv + vec2(0.0, drift);',
      /* The area around the cursor bulges like a lens */
      '  vec2 d = uv - uLocalMouse;',
      '  float r = length(d);',
      '  float bulge = smoothstep(0.42, 0.0, r) * uHover * 0.06;',
      '  uv -= d * bulge;',
      '  gl_FragColor = vec4(texture2D(uTexture, uv).rgb, uOpacity);',
      '}'
    ].join('\n');
    /* ── Locked region ends ── */

    function Tile(el) {
      var self = this;
      this.el = el;
      this.imgAspect = 1;
      this.hover = 0;
      this.targetHover = 0;
      this.localMouse = new THREE.Vector2(0.5, 0.5);
      this.tex = loader.load(el.currentSrc || el.getAttribute('src'), function (t) {
        if (t.image) self.imgAspect = t.image.width / t.image.height;
      });
      this.tex.minFilter = THREE.LinearFilter;
      this.tex.generateMipmaps = false;
      this.u = {
        uTexture: { value: this.tex }, uStrength: { value: 0 }, uOpacity: { value: 1 },
        uCoverScale: { value: new THREE.Vector2(1, 1) },
        uVelocityDir: { value: new THREE.Vector2(0, 1) },
        uLocalMouse: { value: this.localMouse },
        uHover: { value: 0 }, uDriftMul: { value: 0.25 },
        uViewportSizes: { value: new THREE.Vector2(1, 1) }
      };
      this.mesh = new THREE.Mesh(geometry, new THREE.ShaderMaterial({
        uniforms: this.u, vertexShader: vertexShader, fragmentShader: fragmentShader, transparent: true
      }));
      scene.add(this.mesh);

      el.addEventListener('mouseenter', function () { self.targetHover = 1; });
      el.addEventListener('mouseleave', function () { self.targetHover = 0; });
      el.addEventListener('mousemove', function (e) {
        var r = el.getBoundingClientRect();
        self.localMouse.set((e.clientX - r.left) / r.width, 1 - (e.clientY - r.top) / r.height);
      }, { passive: true });
    }

    Tile.prototype.update = function (strength, vdir) {
      var r = this.el.getBoundingClientRect();
      if (r.width === 0 || r.bottom < -150 || r.top > H + 150) { this.mesh.visible = false; return; }
      if (this.el.checkVisibility && !this.el.checkVisibility({ checkOpacity: false, checkVisibilityCSS: true })) {
        this.mesh.visible = false; return;
      }
      this.mesh.visible = true;
      this.mesh.scale.set(r.width, r.height, 1);
      this.mesh.position.x = r.left - W / 2 + r.width / 2;
      this.mesh.position.y = -r.top + H / 2 - r.height / 2;

      /* Cover fit correction. Locked. */
      var planeAspect = r.width / r.height, sx = 1, sy = 1;
      if (this.imgAspect > planeAspect) sx = planeAspect / this.imgAspect;
      else sy = this.imgAspect / planeAspect;
      this.u.uCoverScale.value.set(sx, sy);
      this.u.uViewportSizes.value.set(r.width, r.height);
      this.u.uStrength.value = strength;
      this.u.uVelocityDir.value.copy(vdir);
      this.hover += (this.targetHover - this.hover) * 0.12;
      this.u.uHover.value = this.hover;
    };

    var tiles = live.map(function (el) { return new Tile(el); });
    document.body.classList.add('vy-vault-gl-live');
    say('archive', 'live:' + live.length + '/' + items.length);

    var strength = 0, target = 0;
    var vdir = new THREE.Vector2(0, 1);

    (function frame() {
      var v = window.__vyScrollVel || 0;
      target = Math.min(Math.abs(v) * 0.006, 0.34);   /* Locked: the intensity ceiling, again one third */
      if (Math.abs(v) > 0.01) vdir.set(0, v >= 0 ? 1 : -1);

      strength += (target - strength) * 0.16;
      target *= 0.72;
      if (target < 0.0015 && strength < 0.004) { strength = 0; target = 0; }

      for (var i = 0; i < tiles.length; i++) tiles[i].update(strength, vdir);
      renderer.render(scene, camera);
      requestAnimationFrame(frame);
    })();

    window.addEventListener('resize', function () {
      W = window.innerWidth; H = window.innerHeight;
      renderer.setSize(W, H);
      camera.left = W / -2; camera.right = W / 2; camera.top = H / 2; camera.bottom = H / -2;
      camera.updateProjectionMatrix();
      document.body.classList[W <= 820 ? 'remove' : 'add']('vy-vault-gl-live');
    }, { passive: true });
  }

  whenThree(function () {
    initHero();
    initArchiveGL();
  });
})();
