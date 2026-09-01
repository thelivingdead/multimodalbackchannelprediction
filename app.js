(function () {
  const THRESHOLD = 16.3538328545911;
  const MIN_FRAMES = 5;
  const MAX_FRAMES = 50;
  const PLAY_RATE = 8;
  const SG = [
    -0.08391608391608413, 0.02097902097902111, 0.10256410256410285,
    0.1608391608391613, 0.19580419580419636, 0.20745920745920804,
    0.19580419580419636, 0.1608391608391613, 0.10256410256410287,
    0.020979020979021053, -0.0839160839160841,
  ];

  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const mobile = () => window.matchMedia("(max-width: 860px)").matches;
  function setText(sel, value) {
    const el = $(sel);
    if (el) el.textContent = value;
  }

  function finite(arr) {
    const out = [];
    for (let i = 0; i < arr.length; i++) if (Number.isFinite(arr[i])) out.push(arr[i]);
    return out;
  }
  function savgol(x) {
    const k = SG.length;
    const h = (k - 1) / 2;
    const n = x.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < k; j++) {
        let t = i + j - h;
        if (t < 0) t = 0;
        else if (t >= n) t = n - 1;
        s += SG[j] * x[t];
      }
      out[i] = s;
    }
    return out;
  }
  function sign(v) { return v > 0 ? 1 : v < 0 ? -1 : 0; }
  function ptp(arr) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < lo) lo = arr[i];
      if (arr[i] > hi) hi = arr[i];
    }
    return hi - lo;
  }
  function ruleScore(xRaw) {
    const x = finite(xRaw);
    if (x.length < 11) return { score: 0, sm: x };
    const sm = savgol(x);
    const d = [];
    for (let i = 1; i < sm.length; i++) d.push(sm[i] - sm[i - 1]);
    const turns = [];
    for (let i = 1; i < d.length; i++) if (sign(d[i]) - sign(d[i - 1]) !== 0) turns.push(i);
    let best = 0;
    for (let i = 0; i < turns.length; i++) {
      for (let j = i + 1; j < turns.length; j++) {
        const span = turns[j] - turns[i];
        if (span < MIN_FRAMES || span > MAX_FRAMES) continue;
        const amp = Math.abs(sm[turns[j]] - sm[turns[i]]);
        if (amp > best) best = amp;
      }
    }
    if (best === 0) best = ptp(sm);
    return { score: best, sm };
  }
  function svgSize(svg) {
    const vb = svg.viewBox && svg.viewBox.baseVal;
    if (vb && vb.width) return { w: vb.width, h: vb.height };
    return { w: svg.clientWidth || 640, h: svg.clientHeight || 96 };
  }
  function polyline(values, w, h, pad) {
    const n = values.length;
    if (!n) return { d: "", lo: 0, hi: 1 };
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
    const span = hi - lo || 1;
    let d = "";
    const step = n > 900 ? 2 : 1;
    for (let i = 0; i < n; i += step) {
      const x = pad + (i / (n - 1)) * (w - pad * 2);
      const y = pad + (1 - (values[i] - lo) / span) * (h - pad * 2);
      d += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1) + " ";
    }
    return { d, lo, hi };
  }
  function fmt(n, d) { return Number(n).toFixed(d); }
  function labelName(v) { return Number(v) === 1 ? "CLEAR NOD" : "UNCLEAR"; }
  function predName(v) { return Number(v) === 1 ? "NOD" : "NEUTRAL"; }
  function pad4(n) { return String(n).padStart(4, "0"); }

  const state = {
    data: null,
    preds: null,
    id: "gold_016",
    t: 0,
    playing: false,
    last: 0,
    tracesDirty: true,
    visible: true,
    layers: { raw: true, smoothed: true, threshold: true },
    parallax: { x: 0, y: 0 },
    scrub: false,
  };

  function currentWindow() {
    if (!state.data || !state.data.windows || !state.data.windows.length) return null;
    return state.data.windows.find((w) => w.sample_id === state.id) || state.data.windows[0];
  }
  function scored(win) {
    if (!win._rule) win._rule = ruleScore(win.x);
    return win._rule;
  }
  function means(win) {
    if (win._m) return win._m;
    win._m = {
      x: win.x.reduce((a, b) => a + b, 0) / win.n,
      y: win.y.reduce((a, b) => a + b, 0) / win.n,
      z: win.z.reduce((a, b) => a + b, 0) / win.n,
    };
    return win._m;
  }
  function setPlayhead(el, i, n) {
    if (!el) return;
    el.style.left = ((i / Math.max(1, n - 1)) * 100).toFixed(3) + "%";
  }

  function setupScroll() {
    if (reduced || !window.gsap || !window.ScrollTrigger) return;
    window.gsap.registerPlugin(window.ScrollTrigger);
    const path = $("#scroll-path");
    const len = Number(path && path.dataset.len);
    if (path && len) {
      window.gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
      window.gsap.to(path, {
        strokeDashoffset: 0,
        ease: "none",
        scrollTrigger: {
          trigger: "#signal",
          start: "top 75%",
          end: "center 35%",
          scrub: 0.65,
        },
      });
    }
    window.gsap.from("#finding .stat b", {
      opacity: 0,
      y: 30,
      duration: 1.2,
      stagger: 0.16,
      ease: "power2.out",
      scrollTrigger: { trigger: "#finding", start: "top 72%" },
    });
  }

  function paintStatic(win) {
    const open = $("#open-trace");
    if (open) {
      const { w, h } = svgSize(open);
      const path = open.querySelector("path");
      if (path) path.setAttribute("d", polyline(win.x, w, h, 8).d);
    }

    const scroll = $("#scroll-path");
    if (scroll) {
      const sw = 1000, sh = 180;
      scroll.setAttribute("d", polyline(win.x, sw, sh, 12).d);
      try {
        const len = scroll.getTotalLength();
        scroll.dataset.len = String(len);
        if (!reduced) {
          scroll.style.strokeDasharray = String(len);
          scroll.style.strokeDashoffset = String(len);
        } else {
          scroll.style.strokeDasharray = "none";
          scroll.style.strokeDashoffset = "0";
        }
      } catch (e) {
        scroll.dataset.len = "0";
      }
    }
    const band = $("#gold-band");
    const mark = (win.gold_marks || []).find((m) => m.label === 1) || (win.gold_marks || [])[0];
    if (band && mark && win.n) {
      const dur = win.n / 25;
      band.style.left = (mark.start_s / dur) * 100 + "%";
      band.style.width = Math.max(0.4, ((mark.end_s - mark.start_s) / dur) * 100) + "%";
    } else if (band) {
      band.style.width = "0";
    }

    const { score, sm } = scored(win);
    const ampEl = $("#r-amp");
    const decEl = $("#r-dec");
    if (ampEl) ampEl.textContent = fmt(score, 2) + "°";
    const nod = score >= THRESHOLD;
    if (decEl) {
      decEl.textContent = nod ? "NOD" : "NEUTRAL";
      decEl.className = nod ? "hit" : "";
    }
    const svg = $("#rule-svg");
    if (!svg) {
      state.tracesDirty = false;
      return;
    }
    const rs = svgSize(svg);
    const show = state.layers;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < win.x.length; i++) {
      const v = win.x[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    for (let i = 0; i < sm.length; i++) {
      if (sm[i] < lo) lo = sm[i];
      if (sm[i] > hi) hi = sm[i];
    }
    if (show.threshold) {
      lo = Math.min(lo, THRESHOLD);
      hi = Math.max(hi, THRESHOLD);
    }
    const span = hi - lo || 1;
    const yOf = (v) => 10 + (1 - (v - lo) / span) * (rs.h - 20);
    const redo = (values) => {
      let d = "";
      const step = values.length > 900 ? 2 : 1;
      for (let i = 0; i < values.length; i += step) {
        const x = 10 + (i / (values.length - 1)) * (rs.w - 20);
        d += (i === 0 ? "M" : "L") + x.toFixed(1) + " " + yOf(values[i]).toFixed(1) + " ";
      }
      return d;
    };
    const raw = svg.querySelector(".raw");
    const smoothed = svg.querySelector(".smoothed");
    const thr = svg.querySelector(".thr");
    if (raw) raw.setAttribute("d", show.raw ? redo(win.x) : "");
    if (smoothed) smoothed.setAttribute("d", show.smoothed ? redo(sm) : "");
    if (thr) {
      if (show.threshold) {
        const y = yOf(THRESHOLD);
        thr.setAttribute("y1", y); thr.setAttribute("y2", y);
        thr.setAttribute("x1", 10); thr.setAttribute("x2", rs.w - 10);
      } else {
        thr.setAttribute("y1", -20); thr.setAttribute("y2", -20);
      }
    }
    state.tracesDirty = false;
  }

  function inGold(win, tSec) {
    const marks = win.gold_marks || [];
    for (let i = 0; i < marks.length; i++) {
      if (tSec >= marks[i].start_s && tSec <= marks[i].end_s) {
        return marks[i].label === 1 ? "NOD" : "UNCLEAR";
      }
    }
    return "";
  }

  function renderMotion() {
    if (!state.data) return;
    const win = currentWindow();
    if (!win) return;
    if (state.tracesDirty) paintStatic(win);
    const i = Math.min(Math.floor(state.t), win.n - 1);
    const m = means(win);
    const pitch = win.x[i] - m.x;
    const yaw = win.y[i] - m.y;
    const roll = win.z[i] - m.z;
    let rx = Math.max(-18, Math.min(18, pitch * 0.35));
    let ry = Math.max(-22, Math.min(22, yaw * 0.45));
    let rz = Math.max(-16, Math.min(16, roll * 0.5));
    if (!reduced && !mobile()) {
      rx += state.parallax.y * 4;
      ry += state.parallax.x * 6;
    }
    const squash = 1 - Math.abs(rx) / 90;
    const g = $("#head-rot");
    if (g) g.setAttribute("transform", `translate(${ry.toFixed(2)} ${rx.toFixed(2)}) rotate(${rz.toFixed(2)}) scale(1 ${squash.toFixed(3)})`);
    setText("#v-pitch", fmt(win.x[i], 1) + "°");
    setText("#v-yaw", fmt(win.y[i], 1) + "°");
    setText("#v-roll", fmt(win.z[i], 1) + "°");
    setPlayhead($("#open-playhead"), i, win.n);
    setPlayhead($("#rule-playhead"), i, win.n);
    setText("#r-frame", pad4(i) + " / " + pad4(win.n));
    const tSec = i / (win.fps || 25);
    setText("#open-event", inGold(win, tSec));
    setText("#open-cap", win.sample_id + " · TEST · frame " + pad4(i) + " / " + pad4(win.n));
  }

  function renderExplorer() {
    if (!state.preds || !state.preds.test) return;
    const p = state.preds.test.find((r) => r.sample_id === state.id);
    const win = currentWindow();
    if (!win) return;
    const rule = scored(win);
    setText("#ex-window", String(state.id).replace(/^gold_/i, "GOLD_"));
    setText("#ex-label", p ? labelName(p.label) : "—");
    setText("#ex-split", "TEST");
    setText("#ex-fps", String(win.fps || 25));
    setText("#m-rule", predName(rule.score >= THRESHOLD ? 1 : 0));
    setText("#m-rule-src", "Interactive");
    setText("#m-cnn", p ? predName(p.cnn_pred) : "—");
    setText("#m-ft", p ? predName(p.ft_pred) : "—");
    $$(".win-list button").forEach((b) => {
      if (b.dataset.id === state.id) b.setAttribute("aria-current", "true");
      else b.removeAttribute("aria-current");
    });
  }

  function updateScrollDraw() {
    if (!reduced && window.ScrollTrigger) return;
    const path = $("#scroll-path");
    if (!path || reduced) return;
    const len = Number(path.dataset.len || 0);
    if (!len) return;
    const sec = $("#signal");
    const rect = sec.getBoundingClientRect();
    const vh = window.innerHeight || 1;
    const start = vh * 0.75;
    const end = vh * 0.2;
    let p = (start - rect.top) / (start - end);
    p = Math.max(0, Math.min(1, p));
    path.style.strokeDashoffset = String(len * (1 - p));
  }

  function updatePipe() {
    lightSteps("#method", "#pipe .step");
  }
  function lightSteps(secId, itemSel) {
    const items = $$(itemSel);
    if (!items.length) return;
    const sec = $(secId);
    if (!sec) return;
    const rect = sec.getBoundingClientRect();
    const vh = window.innerHeight || 1;
    let p = (vh * 0.65 - rect.top) / (rect.height || 1);
    p = Math.max(0, Math.min(0.999, p));
    const idx = Math.floor(p * items.length);
    items.forEach((el, i) => el.classList.toggle("on", i <= idx));
  }

  function tick(now) {
    try {
      if (state.data) {
        const win = currentWindow();
        if (win && state.playing && !state.scrub) {
          if (!state.last) state.last = now;
          const dt = Math.min(0.05, (now - state.last) / 1000);
          state.t += dt * (win.fps || 25) * PLAY_RATE;
          if (state.t >= win.n) state.t = 0;
          state.last = now;
        } else {
          state.last = 0;
        }
        const frame = Math.floor(state.t);
        if (state.tracesDirty || frame !== state._frame) {
          state._frame = frame;
          renderMotion();
        }
      }
    } catch (err) {
      console.error(err);
    }
    requestAnimationFrame(tick);
  }

  function bindScrub(el) {
    if (!el) return;
    let pid = null;
    const seek = (clientX) => {
      if (!state.data) return;
      const win = currentWindow();
      if (!win || !win.n) return;
      const r = el.getBoundingClientRect();
      const p = (clientX - r.left) / Math.max(1, r.width);
      state.t = Math.max(0, Math.min(win.n - 1, p * (win.n - 1)));
      state.last = 0;
      state._frame = -1;
      renderMotion();
    };
    const onMove = (e) => {
      if (pid == null || e.pointerId !== pid) return;
      if (e.cancelable) e.preventDefault();
      seek(e.clientX);
    };
    const onUp = (e) => {
      if (pid == null || (e && e.pointerId !== pid)) return;
      pid = null;
      state.scrub = false;
      document.removeEventListener("pointermove", onMove, true);
      document.removeEventListener("pointerup", onUp, true);
      document.removeEventListener("pointercancel", onUp, true);
    };
    el.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      if (!state.data) return;
      e.preventDefault();
      pid = e.pointerId;
      state.scrub = true;
      state.last = 0;
      seek(e.clientX);
      // No setPointerCapture: it fired an early cancel/up that cleared scrub mid-drag.
      document.addEventListener("pointermove", onMove, true);
      document.addEventListener("pointerup", onUp, true);
      document.addEventListener("pointercancel", onUp, true);
    });
  }

  function fillList(preds) {
    const box = $("#win-list");
    if (!box || !preds || !preds.test) return;
    box.innerHTML = "";
    preds.test.forEach((p) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.id = p.sample_id;
      b.textContent = p.sample_id.replace("gold_", "");
      b.title = labelName(p.label);
      box.appendChild(b);
    });
  }

  function setPlaying(on) {
    state.playing = !!on;
    state.scrub = false;
    state.last = 0;
    state._frame = -1;
    $$("[data-act='play']").forEach((b) => b.setAttribute("aria-pressed", on ? "true" : "false"));
    $$("[data-act='pause']").forEach((b) => b.setAttribute("aria-pressed", on ? "false" : "true"));
    if (state.data) renderMotion();
  }

  function bindTransport() {
    $$("[data-act='play']").forEach((b) => {
      b.addEventListener("click", (e) => {
        e.preventDefault();
        setPlaying(true);
      });
    });
    $$("[data-act='pause']").forEach((b) => {
      b.addEventListener("click", (e) => {
        e.preventDefault();
        setPlaying(false);
      });
    });
    $$("[data-act='restart']").forEach((b) => {
      b.addEventListener("click", (e) => {
        e.preventDefault();
        state.t = 0;
        state._frame = -1;
        state.last = 0;
        renderMotion();
      });
    });
    $$(".legend button").forEach((b) => {
      b.addEventListener("click", () => {
        const k = b.dataset.layer;
        state.layers[k] = !state.layers[k];
        b.classList.toggle("on", state.layers[k]);
        state.tracesDirty = true;
        if (state.data) renderMotion();
      });
    });
    bindScrub($("#living .trace-box"));
    bindScrub($("#inspect-wave"));
  }

  function bindWindows() {
    $$(".win-list button").forEach((b) => {
      b.addEventListener("click", () => {
        state.id = b.dataset.id;
        state.t = 0;
        state.tracesDirty = true;
        renderExplorer();
        renderMotion();
        updateScrollDraw();
      });
    });
  }

  function bindPage() {
    window.addEventListener("resize", () => {
      state.tracesDirty = true;
    }, { passive: true });
    window.addEventListener("scroll", () => {
      updateScrollDraw();
      updatePipe();
    }, { passive: true });
    if (!reduced) {
      window.addEventListener("mousemove", (e) => {
        const nx = (e.clientX / window.innerWidth) * 2 - 1;
        const ny = (e.clientY / window.innerHeight) * 2 - 1;
        state.parallax.x = nx * 0.55;
        state.parallax.y = ny * 0.55;
      }, { passive: true });
    }
    const opening = $("#opening");
    if (opening && "IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        state.visible = entries.some((en) => en.isIntersecting);
      }, { threshold: 0.05 }).observe(opening);
    }
  }

  function themeMode() {
    try {
      return localStorage.getItem("mbp-theme") || "system";
    } catch (e) {
      return "system";
    }
  }
  function applyTheme(mode) {
    if (mode !== "light" && mode !== "dark") mode = "system";
    if (mode === "light" || mode === "dark") {
      document.documentElement.setAttribute("data-theme", mode);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    try { localStorage.setItem("mbp-theme", mode); } catch (e) {}
  }
  $$(".theme button").forEach((b) => {
    b.addEventListener("click", () => applyTheme(b.dataset.theme));
  });
  applyTheme(themeMode());
  bindTransport();
  bindPage();
  requestAnimationFrame(tick);

  Promise.all([
    fetch("site-data/windows.json").then((r) => {
      if (!r.ok) throw new Error("windows.json " + r.status);
      return r.json();
    }),
    fetch("site-data/predictions.json").then((r) => {
      if (!r.ok) throw new Error("predictions.json " + r.status);
      return r.json();
    }),
  ]).then(([data, preds]) => {
    state.data = data;
    state.preds = preds;
    fillList(preds);
    bindWindows();
    renderExplorer();
    renderMotion();
    setPlaying(false);
    const fallback = $("#head-fallback");
    if (fallback) fallback.hidden = false;
    try { setupScroll(); } catch (e) { console.error(e); }
    updatePipe();
  }).catch((err) => {
    console.error(err);
  });
})();
