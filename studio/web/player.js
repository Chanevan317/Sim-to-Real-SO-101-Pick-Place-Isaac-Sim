// Data Studio episode player: camera videos kept in sync, a frame slider, playback speed,
// camera choice and joint plots.
// The host page provides the elements and the CSS (.videos, .vbox, .plot, ...) and the colour
// variables --muted, --state, --action and --cursor.

class EpisodePlayer {
  static CAM_LABELS = { external_D455: "Scene camera", ego: "Wrist camera" };

  // el: {videos, slider, frameLabel, playBtn, speed, plots, camMode?} (DOM elements). Without a
  // camMode select, all cameras are shown side by side.
  constructor(el, { onError } = {}) {
    this.el = el;
    this.onError = onError || (() => {});
    this.ep = null;
    this.joints = null;
    this.videos = [];
    this.panels = [];
    this.lastCursor = -1;
    this.camMode = el.camMode ? EpisodePlayer.load("camMode") || "both" : "both";
    el.playBtn.onclick = () => this.togglePlay();
    el.slider.oninput = (e) => this.seek(Number(e.target.value));
    el.speed.onchange = (e) => { for (const v of this.videos) v.playbackRate = Number(e.target.value); };
    if (el.camMode) el.camMode.onchange = (e) => this.setCam(e.target.value);
    window.addEventListener("resize", () => this.drawPlots());
    const loop = () => { this.tick(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  static load(k) { try { return localStorage.getItem(k); } catch { return null; } }
  static save(k, v) { try { localStorage.setItem(k, v); } catch {} }
  static css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  get frame() { return Number(this.el.slider.value); }

  // ep: an episode row from /api/episodes ({name, frames, fps, cameras, ...})
  async show(ep) {
    this.ep = ep;
    this.el.slider.max = ep.frames - 1;
    this.el.slider.value = 0;
    this.renderCamOptions();
    this.buildVideos();
    this.joints = null;
    this.el.plots.innerHTML = "";
    try {
      const r = await fetch(`/api/joints/${ep.name}`);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      if (this.ep && this.ep.name === ep.name) { this.joints = j; this.buildPlots(); }
    } catch (err) { this.onError("Could not load the joints: " + err.message); }
  }

  clear() {
    for (const v of this.videos) { v.pause(); v.removeAttribute("src"); v.load(); }
    this.videos = [];
    this.ep = null;
    this.joints = null;
    this.el.videos.innerHTML = "";
    this.el.plots.innerHTML = "";
  }

  renderCamOptions() {
    if (!this.el.camMode) return;
    const cams = this.ep.cameras;
    const opts = cams.map((c) => [c, EpisodePlayer.CAM_LABELS[c] || c]);
    if (cams.length > 1) opts.push(["both", "Both cameras"]);
    if (!opts.some(([v]) => v === this.camMode)) this.camMode = cams.length > 1 ? "both" : cams[0];
    this.el.camMode.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
    this.el.camMode.value = this.camMode;
  }

  setCam(mode) {
    if (!this.ep || !this.el.camMode || ![...this.el.camMode.options].some((o) => o.value === mode)) return;
    this.camMode = mode;
    this.el.camMode.value = mode;
    EpisodePlayer.save("camMode", mode);
    const f = this.frame;
    this.buildVideos();
    this.videos[0].addEventListener("loadeddata", () => this.seek(f), { once: true });
  }

  buildVideos() {
    for (const v of this.videos) { v.pause(); v.removeAttribute("src"); v.load(); }
    const box = this.el.videos;
    box.innerHTML = "";
    const order = (c) => (c === "external_D455" ? 0 : c === "ego" ? 1 : 2);  // scene camera first
    const cams = this.camMode === "both" ? [...this.ep.cameras].sort((a, b) => order(a) - order(b)) : [this.camMode];
    box.classList.toggle("both", cams.length > 1);
    this.videos = cams.map((cam) => {
      const wrap = document.createElement("div");
      wrap.className = "vbox";
      wrap.innerHTML = `<div class="wait">Preparing video…</div><span class="cap">${EpisodePlayer.CAM_LABELS[cam] || cam}</span>`;
      const v = document.createElement("video");
      v.muted = true; v.playsInline = true; v.preload = "auto";
      v.playbackRate = Number(this.el.speed.value);
      v.addEventListener("loadeddata", () => wrap.querySelector(".wait")?.remove(), { once: true });
      v.addEventListener("error", () => { const w = wrap.querySelector(".wait"); if (w) w.textContent = "Could not load this video."; });
      v.src = `/video/${this.ep.name}/${cam}.mp4`;
      wrap.prepend(v);
      box.appendChild(wrap);
      return v;
    });
    this.videos[0].addEventListener("ended", () => { this.el.playBtn.textContent = "Play"; for (const v of this.videos.slice(1)) v.pause(); });
    this.el.playBtn.textContent = "Play";
  }

  frameOf(t) { return Math.max(0, Math.min(this.ep.frames - 1, Math.floor(t * this.ep.fps + 1e-3))); }

  seek(f) {
    if (!this.ep || !this.videos.length) return;
    f = Math.max(0, Math.min(this.ep.frames - 1, f));
    for (const v of this.videos) { v.pause(); v.currentTime = (f + 0.5) / this.ep.fps; }
    this.el.playBtn.textContent = "Play";
  }

  step(n) { this.seek(this.frame + n); }

  togglePlay() {
    if (!this.videos.length) return;
    const m = this.videos[0];
    if (m.paused) {
      if (m.ended || this.frameOf(m.currentTime) >= this.ep.frames - 1) for (const v of this.videos) v.currentTime = 0;
      for (const v of this.videos) v.play().catch(() => {});
      this.el.playBtn.textContent = "Pause";
    } else {
      for (const v of this.videos) v.pause();
      this.el.playBtn.textContent = "Play";
    }
  }

  tick() {
    if (!this.ep || !this.videos.length) return;
    const t = this.videos[0].currentTime, f = this.frameOf(t);
    for (const v of this.videos.slice(1)) if (Math.abs(v.currentTime - t) > 1.5 / this.ep.fps) v.currentTime = t;
    if (document.activeElement !== this.el.slider) this.el.slider.value = f;
    this.el.frameLabel.textContent = `frame ${f} / ${this.ep.frames - 1} · ${(f / this.ep.fps).toFixed(2)} s`;
    this.drawCursors(f);
  }

  // Keys shared by both pages: Space, ←/→ (Shift: 10 frames), 1/2/3 cameras. True if handled.
  handleKey(ev) {
    const step = ev.shiftKey ? 10 : 1;
    const keys = {
      " ": () => this.togglePlay(),
      ArrowLeft: () => this.step(-step), ArrowRight: () => this.step(step),
      1: () => this.setCam("external_D455"), 2: () => this.setCam("ego"), 3: () => this.setCam("both"),
    };
    if (!keys[ev.key]) return false;
    ev.preventDefault();
    keys[ev.key]();
    return true;
  }

  // ---------- joint plots ----------
  buildPlots() {
    const box = this.el.plots;
    box.innerHTML = "";
    this.panels = [];
    this.joints.names.forEach((name, j) => {
      const p = document.createElement("div");
      p.className = "plot" + (name === "gripper" ? " gripper" : "");
      p.innerHTML = `<div class="pname"><span>${name}</span><span class="val"></span></div><div class="stack"><canvas></canvas><canvas></canvas></div>`;
      box.appendChild(p);
      const [base, over] = p.querySelectorAll("canvas");
      over.addEventListener("click", (ev) => {
        const r = over.getBoundingClientRect();
        this.seek(Math.round(((ev.clientX - r.left) / r.width) * (this.ep.frames - 1)));
      });
      this.panels.push({ j, base, over, val: p.querySelector(".val") });
    });
    this.drawPlots();
  }

  static sizeCanvas(c) {
    const dpr = window.devicePixelRatio || 1, r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width * dpr));
    c.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [ctx, r.width, r.height];
  }

  drawPlots() {
    if (!this.joints) return;
    const css = EpisodePlayer.css;
    for (const p of this.panels) {
      const s = this.joints.state[p.j], a = this.joints.action[p.j], n = s.length;
      let lo = Math.min(...s, ...a), hi = Math.max(...s, ...a);
      if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
      const padY = (hi - lo) * 0.08;
      lo -= padY; hi += padY;
      const [ctx, w, h] = EpisodePlayer.sizeCanvas(p.base);
      EpisodePlayer.sizeCanvas(p.over);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = css("--muted");
      ctx.font = "10px system-ui";
      ctx.fillText(hi.toFixed(0), 2, 10);
      ctx.fillText(lo.toFixed(0), 2, h - 3);
      const line = (vals, color, dash) => {
        ctx.beginPath(); ctx.setLineDash(dash); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        vals.forEach((v, i) => { const x = (i / Math.max(n - 1, 1)) * w, y = h - ((v - lo) / (hi - lo)) * h; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke(); ctx.setLineDash([]);
      };
      line(s, css("--state"), []);
      line(a, css("--action"), [5, 3]);
    }
    this.drawCursors(this.frame, true);
  }

  drawCursors(f, force = false) {
    if (!this.joints || (!force && f === this.lastCursor)) return;
    this.lastCursor = f;
    for (const p of this.panels) {
      const ctx = p.over.getContext("2d"), r = p.over.getBoundingClientRect(), n = this.joints.state[p.j].length;
      ctx.clearRect(0, 0, r.width, r.height);
      const x = (f / Math.max(n - 1, 1)) * r.width;
      ctx.strokeStyle = EpisodePlayer.css("--cursor"); ctx.globalAlpha = 0.6; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, r.height); ctx.stroke(); ctx.globalAlpha = 1;
      const fi = Math.min(f, n - 1);
      p.val.textContent = `state ${this.joints.state[p.j][fi].toFixed(1)} · action ${this.joints.action[p.j][fi].toFixed(1)}`;
    }
  }
}
