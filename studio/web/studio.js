// SO-Arm Data Studio front-end. Talks to studio.py; the episode player comes from /static/player.js.
const $ = (id) => document.getElementById(id);
const STEPS = [
  ["campaign", "Campaign"], ["check", "Check"], ["record", "Record"],
  ["review", "Review"], ["convert", "Convert"], ["send", "Send"], ["train", "Train"],
];
const REASONS = ["Dropped the cube", "Wrong cube", "No release before saving", "Cube not on the plate",
  "Hit the plate or a wall", "Long pause or hesitation", "Other"];
const LAYOUT_NAMES = { camera_side: "cubes near camera", far_side: "cubes far side", plate_middle: "plate in the middle" };

let S = null;              // last /api/studio snapshot
let confirmed = false;     // a campaign was opened since this window opened
let page = null;           // current page id
let episodes = [];         // campaign episodes (review page)
let reviewTab = "todo";
let current = null;        // episode shown in the review
let selectedCampaign = null;
let motorTimer = null;
let prevRecordRunning = false;
let prevConvertRunning = false;
let prevSendRunning = false;
let sendMethod = "ssh";     // chosen tab on the Send page
let transferInfo = null;   // /api/transfer: dataset size, croc, last SSH target

const player = new EpisodePlayer({
  videos: $("videos"), slider: $("slider"), frameLabel: $("frameLabel"), playBtn: $("playBtn"),
  speed: $("speed"), plots: $("plots"),  // no camera choice: both cameras, always the same size
}, { onError: (m) => toast(m) });

// ---------- helpers ----------
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg, ms = 3500) {
  const t = $("toast"); t.textContent = msg; t.style.display = "block";
  clearTimeout(toast.t); toast.t = setTimeout(() => (t.style.display = "none"), ms);
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pad = (n) => String(n).padStart(4, "0");
const pretty = (folder) => folder.replace(/^raw_teleop_data_/, "");
const dot = (color) => (color ? `<span class="dot" style="background:var(--${color})"></span>` : "");
function canStartRound() {
  const c = S && S.counts;
  const armOk = (motorState && motorState.passed) || (S && S.motor_test_passed);
  return !!(c && armOk && !c.pending && !c.flagged && S.next_round && !S.round && !recordRunning() && !convertRunning());
}
const lastEvent = () => (S && S.history && S.history.length ? S.history[S.history.length - 1].event : null);
const recordRunning = () => !!(S && S.record && S.record.status === "running");
const convertRunning = () => !!(S && S.convert && S.convert.status === "running");
const sendRunning = () => !!(S && S.send && S.send.status === "running");
const hasDataset = () => !!(S && S.campaign && S.campaign.dataset && S.dataset_exists && !convertRunning());
function kv(rows) { return rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join(""); }

// ---------- steps ----------
function stepInfo(id) {
  if (!confirmed) return { open: id === "campaign", done: false };
  const c = S.counts, camp = S.campaign;
  const ready = c && camp;
  switch (id) {
    case "campaign": return { open: true, done: !!(S.folder && camp) };
    case "check": return { open: !!ready, done: S.motor_test_passed };
    case "record": return { open: !!ready, done: false };
    case "review": return { open: !!(ready && c.episodes), done: !!(ready && c.episodes && !c.pending && !c.flagged), badge: c && c.pending };
    case "convert": return { open: !!(ready && c.approved >= camp.target && !c.pending && !c.flagged), done: hasDataset() };
    case "send": return { open: !!ready && (hasDataset() || sendRunning()), done: !!(S.send && S.send.status === "finished") };
    default: return { open: false, done: false };
  }
}

function renderSteps() {
  $("steps").innerHTML = STEPS.map(([id, label], i) => {
    const s = stepInfo(id);
    const cls = ["step", s.open ? "" : "locked", s.done ? "done" : "", page === id ? "current" : ""].join(" ");
    return `<button class="${cls}" data-step="${id}" ${s.open ? "" : "disabled"}><span class="n">${s.done ? "✓" : i + 1}</span>` +
      `<span class="t">${label}</span>${s.badge ? `<span class="b">${s.badge}</span>` : ""}</button>`;
  }).join("");
  $("steps").querySelectorAll("[data-step]").forEach((b) => (b.onclick = () => go(b.dataset.step)));
}

function renderTop() {
  const parts = [];
  $("switchBtn").hidden = !confirmed || page === "campaign";
  if (!confirmed) { $("who").innerHTML = ""; return; }
  if (S.folder && S.campaign && S.counts) parts.push(`Campaign <b>${esc(pretty(S.folder))}</b>: ${S.counts.approved} / ${S.campaign.target} approved`);
  $("who").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

function go(id) {
  if (!stepInfo(id).open) return;
  if (page === "check" && id !== "check") stopMotorTest(false);
  if (page === "review" && id !== "review") player.clear(), (current = null);
  page = id;
  history.replaceState(null, "", `#${id}`);
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("show", p.id === `page-${id}`));
  renderSteps();
  enter(id);
}

function autoPage() {
  const c = S.counts, camp = S.campaign;
  if (!confirmed || !S.folder || !camp) return "campaign";
  if (recordRunning() || S.round) return "record";
  if (convertRunning()) return "convert";
  if (sendRunning()) return "send";
  if (c.pending || c.flagged) return "review";
  if (c.approved >= camp.target) return "convert";
  return "check";
}

// ---------- refresh loop ----------
async function refresh() {
  try { S = await api("/api/studio"); } catch (e) { toast("Data Studio is not responding: " + e.message); return; }
  renderTop();
  if (!page) {
    const wanted = location.hash.slice(1);
    go(STEPS.some(([id]) => id === wanted) && stepInfo(wanted).open ? wanted : autoPage());
    return;
  }
  if (prevRecordRunning && !recordRunning()) {
    prevRecordRunning = false;
    if (S.round) {
      toast(`Round paused: ${S.round.saved} of ${S.round.size} saved. Continue when you are ready.`, 6000);
      if (page !== "record") { go("record"); return; }
    } else if (lastEvent() === "round_dropped") {
      toast("Round dropped.", 5000);
      go(autoPage()); return;
    } else {
      toast("Round complete. Review the episodes now.", 6000);
      if (S.counts && S.counts.episodes) { go("review"); return; }
    }
  }
  if (prevConvertRunning && !convertRunning()) toast(S.convert.status === "finished" ? "Conversion finished." : "Conversion stopped with an error.", 6000);
  if (prevSendRunning && !sendRunning()) toast(S.send.status === "finished" ? "Dataset sent." : S.send.status === "stopped" ? "Sending cancelled." : "Sending failed.", 6000);
  prevSendRunning = sendRunning();
  prevRecordRunning = recordRunning();
  prevConvertRunning = convertRunning();
  renderSteps();
  if (page === "record") renderRecord();
  if (page === "convert") renderConvert();
  if (page === "send") renderSend();
  if (page === "check") renderRoundInfo();
}

function enter(id) {
  if (id === "campaign") enterCampaign();
  if (id === "check") enterCheck();
  if (id === "record") renderRecord();
  if (id === "review") loadEpisodes(true);
  if (id === "convert") enterConvert();
  if (id === "send") enterSend();
}

$("switchBtn").onclick = () => go("campaign");

// ---------- 2. campaign ----------
function fillRoundSelect(sel, value) {
  sel.innerHTML = S.config.round_sizes.map((n) => `<option value="${n}">${n}</option>`).join("");
  sel.value = value || S.config.round_sizes[0];
}
function enterCampaign() {
  const rows = S.campaigns;
  $("campaignList").innerHTML = rows.length ? rows.map((c) => {
    const goal = c.target ? `${c.episodes} recorded · goal ${c.target}` : `${c.episodes} recorded · no goal set yet`;
    const active = c.folder === (selectedCampaign || S.folder);
    return `<div class="item ${active ? "active" : ""}" data-f="${c.folder}"><span class="name">${esc(pretty(c.folder))}</span>` +
      `<span class="muted small">${c.folder === S.folder ? "last used" : ""}</span><span class="sub">${goal}</span></div>`;
  }).join("") : `<p class="muted" style="padding:12px">No campaign yet: create one on the right.</p>`;
  $("campaignList").querySelectorAll("[data-f]").forEach((el) => (el.onclick = () => { selectedCampaign = el.dataset.f; enterCampaign(); }));
  const sel = rows.find((c) => c.folder === (selectedCampaign || S.folder));
  $("campaignEdit").hidden = !sel;
  if (sel) {
    $("campaignEditTitle").textContent = pretty(sel.folder);
    $("editTarget").value = sel.target || S.config.default_target;
    fillRoundSelect($("editRound"), sel.round_size);
  }
  if (!$("newTarget").value) $("newTarget").value = S.config.default_target;
  if (!$("newRound").options.length) fillRoundSelect($("newRound"));
}
$("openCampaignBtn").onclick = async () => {
  const folder = selectedCampaign || S.folder;
  try {
    await api("/api/campaign", { folder, target: Number($("editTarget").value), round_size: Number($("editRound").value) });
    S = await api("/api/studio");
    selectedCampaign = null;
    confirmed = true;
    go(autoPage());
  } catch (e) { toast(e.message); }
};
$("newCampaignBtn").onclick = async () => {
  try {
    await api("/api/campaign", { new_name: $("newName").value, target: Number($("newTarget").value), round_size: Number($("newRound").value) });
    S = await api("/api/studio");
    $("newName").value = "";
    confirmed = true;
    go(autoPage());
  } catch (e) { toast(e.message); }
};

// ---------- 3. check ----------
let checksOk = false;
async function enterCheck() {
  renderRoundInfo();
  await loadChecks();
  renderMotors(null);
}
async function loadChecks() {
  $("checks").innerHTML = `<p class="muted">Checking…</p>`;
  try {
    const { checks } = await api("/api/checks");
    checksOk = checks.every((c) => c.ok);
    $("checks").innerHTML = checks.map((c) => `<div class="check ${c.ok ? "ok" : "bad"}"><span class="ic">${c.ok ? "✓" : "✕"}</span>` +
      `<span>${esc(c.label)}</span><span class="detail">${esc(c.detail)}</span>${c.ok ? "" : `<span class="fix">${esc(c.fix)}</span>`}</div>`).join("");
  } catch (e) { checksOk = false; $("checks").innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  updateStartButton();
}
$("recheckBtn").onclick = loadChecks;

let motorState = null;
function renderMotors(st) {
  motorState = st;
  const motors = st ? st.motors : ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"].map((name) => ({ name }));
  $("motors").innerHTML = motors.map((m) => {
    const ok = m.responding && m.moved;
    const dead = st && st.running && m.responding === false;
    const pos = m.deg == null ? null : ((m.deg + 180) / 360) * 100;
    return `<div class="motor ${ok ? "ok" : dead ? "dead" : ""}"><span>${m.name.replace("_", " ")}</span>` +
      `<span class="gauge">${pos == null ? "" : `<span style="left:calc(${pos}% - 1px)"></span>`}</span>` +
      `<span class="val">${m.deg == null ? "–" : m.deg.toFixed(0) + "°"}</span><span class="ic">${ok ? "✓" : dead ? "✕" : "·"}</span></div>`;
  }).join("");
  if (st) {
    const dead = st.motors.filter((m) => st.running && !m.responding).map((m) => m.name);
    const still = st.motors.filter((m) => m.responding && !m.moved).map((m) => m.name.replace("_", " "));
    $("motorMsg").textContent = st.error ? st.error
      : st.no_answer ? "No motor answers. Check that the leader arm's power supply is plugged in and switched on, and the cable from the small board to the arm."
      : st.passed ? "All six joints work."
      : dead.length && st.motors.some((m) => m.deg != null) ? `No answer from: ${dead.join(", ")}. Check the cables.`
      : still.length ? `Move: ${still.join(", ")}` : "Reading the leader arm…";
  } else $("motorMsg").textContent = S && S.motor_test_passed ? "Test passed. Continue to recording." : "";
  $("motorStartBtn").textContent = st && st.running ? "Stop test" : "Start test";
  updateStartButton();
}
async function pollMotors() {
  try {
    const st = await api("/api/motors");
    renderMotors(st);
    if (!st.running) { clearInterval(motorTimer); motorTimer = null; }
  } catch (e) { clearInterval(motorTimer); motorTimer = null; }
}
async function stopMotorTest(render = true) {
  clearInterval(motorTimer); motorTimer = null;
  try { await api("/api/motors/stop", {}); } catch {}
  if (render) { S = await api("/api/studio"); renderMotors(null); renderSteps(); }
}
$("motorStartBtn").onclick = async () => {
  if (motorState && motorState.running) return stopMotorTest(true);
  try {
    await api("/api/motors/start", {});
    clearInterval(motorTimer);
    motorTimer = setInterval(pollMotors, 200);
  } catch (e) { toast(e.message); }
};
function renderRoundInfo() {
  const c = S.counts, camp = S.campaign;
  if (!c || !camp) return;
  let msg;
  if (recordRunning()) msg = "A round is running.";
  else if (S.round) msg = `A round is paused (${S.round.size - S.round.saved} episode(s) left): continue it on the Record step.`;
  else if (c.pending || c.flagged) msg = `Review the last round first (${c.pending} episode(s) waiting).`;
  else if (!S.next_round) msg = "All episodes are recorded and approved: go to Convert.";
  else msg = `Next round: <b>${S.next_round}</b> episode(s). ${c.approved} of ${camp.target} approved so far.`;
  $("roundInfo").innerHTML = msg;
  $("startRoundBtn").textContent = S.round ? "Go to the paused round" : S.next_round ? `Continue to recording (${S.next_round} episodes)` : "Continue to recording";
  updateStartButton();
}
function updateStartButton() {
  const c = S && S.counts;
  const armOk = (motorState && motorState.passed) || (S && S.motor_test_passed);
  if (S && S.round && !recordRunning()) { $("startRoundBtn").disabled = false; return; }
  $("startRoundBtn").disabled = !(checksOk && armOk && c && !c.pending && !c.flagged && S.next_round && !recordRunning() && !convertRunning());
}
$("startRoundBtn").onclick = async () => {
  // Only moves on: the simulator opens when the operator presses Start on the Record page.
  await stopMotorTest(false);  // ends the arm test; the server remembers that it passed
  S = await api("/api/studio");
  go("record");
};

// ---------- 4. record ----------
const PAUSE_REASONS = { paused: "You paused the round.", "no movement": "Closed after the leader arm did not move for a while.",
  "simulator closed": "The simulator window was closed.", error: "The simulator stopped with an error." };
function renderRecord() {
  const job = S.record, running = recordRunning(), round = S.round;
  const info = (job && job.info) || {};
  const phases = { starting: ["Starting the simulator…", "warn", "This takes about a minute."],
    recording: ["Recording", "ok", "Use the leader arm. The simulator window shows the target."],
    saving: ["Saving the last episodes…", "warn", "Almost done."], finished: ["Closing…", "warn", ""] };
  let title, cls, sub;
  if (running) [title, cls, sub] = job.stopping ? ["Pausing…", "warn", "Closing the simulator. Saved episodes are kept."] : (phases[info.phase] || phases.starting);
  else if (!round && canStartRound()) [title, cls, sub] = ["Ready to record", "ok",
    "Read the rules on the right and look at the home pose. Put the leader arm in its home pose, then press Start."];
  else if (round) [title, cls, sub] = ["Round paused", round.paused === "error" ? "bad" : "warn", PAUSE_REASONS[round.paused] || ""];
  else if (job && lastEvent() === "round_dropped") [title, cls, sub] = ["Round dropped", "warn", "Start a new round from the Check step."];
  else if (job) [title, cls, sub] = ["Round complete", "ok", "Go to Review."];
  else [title, cls, sub] = ["No round running", "", "Start one from the Check step."];
  $("recState").textContent = title; $("recState").className = "state " + cls; $("recSub").textContent = sub;
  const readyNow = !running && !round && canStartRound();
  const saved = round ? round.saved : readyNow ? 0 : info.saved || 0;
  const size = round ? round.size : readyNow ? S.next_round : info.max || 0;
  $("recSaved").textContent = saved; $("recMax").textContent = size;
  $("recBar").style.width = size ? `${Math.min(100, (100 * saved) / size)}%` : "0";
  const rows = [];
  if (running && info.target) rows.push(["Move this cube", `${dot(info.target)}<b>${info.target.toUpperCase()}</b>`]);
  if (running && info.layout) rows.push(["Scene", esc(LAYOUT_NAMES[info.layout] || info.layout)]);
  if (running && info.index != null) rows.push(["Episode", pad(info.index)]);
  if (running && info.discarded) rows.push(["Thrown away (X)", info.discarded]);
  if (S.counts && S.campaign) rows.push(["Campaign", `${S.counts.approved} approved of ${S.campaign.target}`]);
  $("recNow").innerHTML = kv(rows);
  $("dropRoundBtn").hidden = !round || (running && job.stopping);
  $("stopRoundBtn").hidden = !running;
  $("stopRoundBtn").disabled = !running || job.stopping;
  $("pauseHint").hidden = !running;
  $("continueRoundBtn").hidden = running || !round;
  const ready = !running && !round && canStartRound();
  $("readyStartBtn").hidden = !ready;
  if (ready) $("readyStartBtn").textContent = `Start round (${S.next_round} episodes)`;
  if (round) $("continueRoundBtn").textContent = `Continue round (${round.size - round.saved} left)`;
  const showLog = !running && job && (job.status === "failed" || (round && round.paused === "error"));
  $("recLog").hidden = !showLog;
  if (showLog) $("recLog").textContent = job.log.join("\n");
}
$("stopRoundBtn").onclick = async () => {
  if (!confirm("Pause the round now?\nThe simulator closes. Saved episodes are kept and you can continue later; the try in progress is dropped.")) return;
  try { await api("/api/record/pause", {}); toast("Pausing… the simulator closes in a moment."); } catch (e) { toast(e.message); }
};
$("readyStartBtn").onclick = async () => {
  $("readyStartBtn").disabled = true;
  try {
    await api("/api/record/start", {});
    S = await api("/api/studio");
    prevRecordRunning = true;
    renderRecord();
  } catch (e) { toast(e.message, 6000); }
  $("readyStartBtn").disabled = false;
};
$("dropRoundBtn").onclick = async () => {
  const saved = S.round ? S.round.saved : 0;
  if (!saved) {
    if (confirm("Drop this round?" + (recordRunning() ? "\nThe simulator closes; the try in progress is dropped." : ""))) dropRound(true);
    return;
  }
  $("dropText").textContent = `${saved} episode(s) were already saved in this round. Keep them for review, or throw them away?` +
    (recordRunning() ? " The simulator closes; the try in progress is dropped." : "");
  $("dropModal").hidden = false;
};
async function dropRound(keep) {
  $("dropModal").hidden = true;
  try {
    await api("/api/record/drop", { keep });
    toast(keep ? "Round dropped. Saved episodes are kept for review." : "Round dropped. Its episodes were moved to the trash.", 5000);
  } catch (e) { toast(e.message); }
}
$("dropKeepBtn").onclick = () => dropRound(true);
$("dropTrashBtn").onclick = () => dropRound(false);
$("dropCancelBtn").onclick = () => ($("dropModal").hidden = true);
$("continueRoundBtn").onclick = async () => {
  $("continueRoundBtn").disabled = true;
  try {
    await api("/api/record/continue", {});
    S = await api("/api/studio");
    prevRecordRunning = true;
    renderRecord();
  } catch (e) { toast(e.message, 6000); }
  $("continueRoundBtn").disabled = false;
};

// ---------- 5. review ----------
const isTodo = (e) => e.status === "ok" && !e.approved && !e.flagged;
async function loadEpisodes(selectFirst = false) {
  try { episodes = (await api("/api/episodes")).episodes; } catch (e) { toast(e.message); return; }
  renderReviewList();
  if (selectFirst || !current || !episodes.some((e) => e.name === current.name)) {
    const first = visibleEpisodes().find((e) => e.status === "ok") || episodes.find(isTodo);
    if (first) select(first.name); else showReviewEmpty();
  } else {
    current = episodes.find((e) => e.name === current.name);
    renderDecision();
  }
}
function visibleEpisodes() {
  const f = { todo: isTodo, approved: (e) => e.approved, flagged: (e) => e.flagged, all: () => true }[reviewTab];
  return episodes.filter(f);
}
function renderReviewList() {
  const n = { todo: episodes.filter(isTodo).length, approved: episodes.filter((e) => e.approved).length,
    flagged: episodes.filter((e) => e.flagged).length, all: episodes.length };
  const labels = { todo: "To do", approved: "Approved", flagged: "Flagged", all: "All" };
  $("reviewTabs").innerHTML = Object.keys(labels).map((k) => `<button data-tab="${k}" class="${reviewTab === k ? "on" : ""}">${labels[k]} ${n[k]}</button>`).join("");
  $("reviewTabs").querySelectorAll("[data-tab]").forEach((b) => (b.onclick = () => { reviewTab = b.dataset.tab; renderReviewList(); }));
  const rows = visibleEpisodes();
  $("reviewList").innerHTML = rows.length ? rows.map((e) => {
    if (e.status !== "ok") return `<div class="item"><span class="name">Episode ${pad(e.index)}</span><span class="sub">${e.status === "compressing" ? "still saving…" : "unreadable file"}</span></div>`;
    const tag = e.approved ? `<span class="tag approved">✓</span>` : e.flagged ? `<span class="tag flagged">✕</span>`
      : e.warnings && e.warnings.length ? `<span class="tag warn">!</span>` : "";
    return `<div class="item ${current && current.name === e.name ? "active" : ""}" data-n="${e.name}"><span class="name">Episode ${pad(e.index)}</span>${tag}` +
      `<span class="sub">${dot(e.target)}${esc(e.target || "")} · ${esc(LAYOUT_NAMES[e.layout] || e.layout || "")} · ${e.seconds.toFixed(1)} s${e.flagged && e.reason ? " · " + esc(e.reason) : ""}</span></div>`;
  }).join("") : `<p class="muted" style="padding:12px">${reviewTab === "todo" ? "Everything is reviewed." : "Nothing here."}</p>`;
  $("reviewList").querySelectorAll("[data-n]").forEach((el) => (el.onclick = () => select(el.dataset.n)));
  $("reviewProgress").textContent = n.todo ? `${n.todo} left to review` : n.flagged ? `${n.flagged} flagged will go to the trash` : "All reviewed";
  $("finishReviewBtn").disabled = !!S.round || !!n.todo || !(n.flagged || (S.counts && S.counts.pending === 0 && episodes.length));
  if (S.round) $("reviewProgress").textContent = "Finish the round first (it is paused).";
}
function showReviewEmpty() {
  current = null; player.clear();
  $("reviewPlayer").hidden = true; $("reviewEmpty").hidden = false;
  $("reviewEmpty").textContent = episodes.some(isTodo) ? "Choose an episode on the left." : "Nothing left to review. Press Finish review.";
}
async function select(name) {
  const e = episodes.find((x) => x.name === name);
  if (!e || e.status !== "ok") return;
  current = e;
  $("reviewEmpty").hidden = true; $("reviewPlayer").hidden = false;
  $("epTitle").textContent = `Episode ${pad(e.index)}`;
  $("epTarget").innerHTML = e.target ? `${dot(e.target)}Target: ${e.target} cube` : "";
  $("epMeta").textContent = [LAYOUT_NAMES[e.layout] || e.layout, `${e.seconds.toFixed(1)} s`, e.operator ? `recorded by ${e.operator}` : ""].filter(Boolean).join(" · ");
  $("epWarnings").innerHTML = (e.warnings || []).map((w) => `<span>! ${esc(w)}</span>`).join("");
  $("reasons").hidden = true;
  renderDecision(); renderReviewList();
  await player.show(e);
}
function renderDecision() {
  if (!current) return;
  const d = $("epDecision");
  d.className = "decision " + (current.approved ? "approved" : current.flagged ? "flagged" : "");
  d.textContent = current.approved ? "Approved" : current.flagged ? `Flagged: ${current.reason || "bad"}` : "Not reviewed yet";
}
async function decide(decision, reason = "") {
  if (!current) return;
  try {
    await api("/api/review", { name: current.name, decision, reason });
    Object.assign(current, { approved: decision === "approve", flagged: decision === "flag", reason });
    $("reasons").hidden = true;
    renderDecision(); renderReviewList();
    const next = episodes.find((e) => isTodo(e) && e.name !== current.name && e.index > current.index) || episodes.find(isTodo);
    if (next) setTimeout(() => select(next.name), 250); else { toast("All episodes reviewed. Press Finish review."); renderReviewList(); }
  } catch (e) { toast(e.message); }
}
$("approveBtn").onclick = () => decide("approve");
$("flagBtn").onclick = () => { $("reasons").hidden = !$("reasons").hidden; };
$("reasons").insertAdjacentHTML("beforeend", REASONS.map((r) => `<button data-r="${esc(r)}">${esc(r)}</button>`).join(""));
$("reasons").querySelectorAll("[data-r]").forEach((b) => (b.onclick = () => decide("flag", b.dataset.r)));
$("finishReviewBtn").onclick = async () => {
  const flagged = episodes.filter((e) => e.flagged).length;
  if (flagged && !confirm(`Move ${flagged} flagged episode(s) to the trash and finish the review?`)) return;
  try {
    const res = await api("/api/review/finish", {});
    toast(res.moved.length ? `${res.moved.length} episode(s) moved to the trash.` : "Review finished.");
    S = await api("/api/studio");
    player.clear(); current = null;
    go(autoPage());
  } catch (e) { toast(e.message); }
};
function neighbour(step) {
  const rows = visibleEpisodes().filter((e) => e.status === "ok");
  const i = rows.findIndex((e) => current && e.name === current.name);
  return rows[i + step] || null;
}

// ---------- 6. convert ----------
function enterConvert() {
  if (!$("datasetName").value || !convertRunning()) $("datasetName").value = S.dataset_name || "";
  renderConvert();
}
function renderConvert() {
  const c = S.counts, camp = S.campaign;
  if (!c || !camp) return;
  $("convSummary").innerHTML = kv([["Approved episodes", `${c.approved} / ${camp.target}`], ["Total length", `${c.minutes} min`],
    ["Recordings", esc(S.folder)]]);
  const colors = ["blue", "red"], layouts = Object.keys(LAYOUT_NAMES);
  $("balance").innerHTML = `<tr><th></th>${colors.map((k) => `<th>${dot(k)}${k}</th>`).join("")}</tr>` +
    layouts.map((l) => `<tr><th>${LAYOUT_NAMES[l]}</th>${colors.map((k) => `<td>${c.balance[`${l}/${k}`] || 0}</td>`).join("")}</tr>`).join("");
  const job = S.convert, running = convertRunning();
  const taken = S.dataset_exists && $("datasetName").value.trim() === S.dataset_name;
  $("convertBtn").disabled = running || taken || !stepInfo("convert").open;
  $("convertBtn").textContent = taken ? "Already converted" : "Convert";
  $("datasetName").disabled = running;
  $("convProgress").hidden = !job;
  if (job) {
    const info = job.info || {}, total = info.total || c.approved, done = info.done || 0;
    const states = { running: [info.phase === "converting" ? "Converting" : "Starting…", "warn"], finished: ["Dataset ready", "ok"],
      failed: ["Conversion failed", "bad"], stopped: ["Stopped", "warn"] };
    const [title, cls] = states[job.status] || states.running;
    $("convState").textContent = title; $("convState").className = "state " + cls;
    $("convSub").textContent = job.status === "finished" ? `lerobot/${S.campaign.dataset || ""}` : running ? `${done} / ${total} episodes` : info.error || "";
    $("convBar").style.width = `${total ? Math.min(100, (100 * (job.status === "finished" ? total : done)) / total) : 0}%`;
    $("convLog").hidden = job.status !== "failed";  // the encoder output only matters when something went wrong
    $("convLog").textContent = job.log.join("\n");
  }
  $("toSendBtn").hidden = !hasDataset();
}
$("toSendBtn").onclick = () => go("send");
$("datasetName").oninput = () => renderConvert();
$("convertBtn").onclick = async () => {
  try {
    await api("/api/convert/start", { name: $("datasetName").value.trim() });
    S = await api("/api/studio");
    prevConvertRunning = true;
    renderConvert();
  } catch (e) { toast(e.message, 6000); }
};

// ---------- 7. send ----------
async function loadTransferInfo() {
  try { transferInfo = await api("/api/transfer"); } catch (e) { toast(e.message); return; }
  renderSend();
}
async function enterSend() {
  if (S.send) sendMethod = S.send.method;
  transferInfo = null;
  renderSend();
  await loadTransferInfo();
  if (!$("sshHost").value) $("sshHost").value = transferInfo ? transferInfo.ssh_host : "";
  if (!$("sshDir").value) $("sshDir").value = transferInfo ? transferInfo.ssh_dir : "";
}
function renderSend() {
  const t = transferInfo, job = S.send, running = sendRunning();
  const ds = t && t.dataset;
  $("sendSummary").innerHTML = kv([["Dataset", `lerobot/${esc((ds && ds.name) || (S.campaign && S.campaign.dataset) || "")}`],
    ["Size", ds ? `${ds.gb} GB in ${ds.files} files` : "…"]]);
  $("sendTabs").querySelectorAll("[data-m]").forEach((b) => { b.classList.toggle("on", b.dataset.m === sendMethod); b.disabled = running; });
  $("method-ssh").hidden = sendMethod !== "ssh";
  $("method-croc").hidden = sendMethod !== "croc";
  ["sshHost", "sshDir", "sshTestBtn"].forEach((id) => ($(id).disabled = running));
  $("sshSendBtn").disabled = running || !hasDataset();
  const croc = t && t.croc, installing = croc && croc.install && croc.install.status === "running";
  $("crocStatus").className = "check " + (croc ? (croc.path ? "ok" : "bad") : "");
  $("crocStatus").innerHTML = !croc ? `<span class="ic">·</span><span>Looking for croc…</span>`
    : croc.path ? `<span class="ic">✓</span><span>croc ${esc(croc.version || "")} is installed</span>`
    : `<span class="ic">✕</span><span>croc is not installed on this computer</span>`;
  $("crocInstallBtn").hidden = !croc || (!!croc.path && !installing);
  $("crocInstallBtn").disabled = installing;
  $("crocInstallBtn").textContent = installing ? "Installing…" : "Install croc";
  $("crocMsg").textContent = croc && croc.install ? croc.install.message : "";
  $("crocSendBtn").disabled = running || !hasDataset() || !(croc && croc.path);
  if (installing) setTimeout(loadTransferInfo, 1500);

  $("sendProgress").hidden = !job;
  if (!job) return;
  const info = job.info || {};
  const waiting = running && info.phase === "waiting", connecting = running && info.phase === "connecting";
  const states = { finished: ["Sent", "ok"], failed: ["Sending failed", "bad"], stopped: ["Cancelled", "warn"] };
  const [title, cls] = running ? (waiting ? ["Waiting for the training computer…", "warn"] : connecting ? ["Connecting…", "warn"] : ["Sending", "warn"])
    : states[job.status] || ["", ""];
  $("sendState").textContent = title; $("sendState").className = "state " + cls;
  const where = job.method === "ssh" ? `to ${job.target}` : "with croc";
  $("sendSub").textContent = running ? (info.percent != null ? `${info.percent}%${info.speed ? " · " + info.speed : ""} · ${where}` : where)
    : job.status === "finished" ? (job.method === "ssh" ? `Now in ${job.target}` : "The training computer has the dataset.")
    : info.error || "";
  $("sendBar").style.width = `${job.status === "finished" ? 100 : info.percent || 0}%`;
  $("crocCode").hidden = !(job.method === "croc" && running);
  if (job.method === "croc") { $("crocCmd").textContent = job.receive; $("crocSecret").textContent = job.code; }
  $("cancelSendBtn").hidden = !running;
  $("cancelSendBtn").disabled = job.stopping;
  $("sendLog").hidden = job.status !== "failed";
  $("sendLog").textContent = job.log.join("\n");
}
$("sendTabs").querySelectorAll("[data-m]").forEach((b) => (b.onclick = () => { sendMethod = b.dataset.m; renderSend(); }));
$("sshTestBtn").onclick = async () => {
  $("sshTestBtn").disabled = true;
  $("sshMsg").className = "small muted"; $("sshMsg").textContent = "Connecting…";
  try {
    const r = await api("/api/transfer/ssh_test", { host: $("sshHost").value, dir: $("sshDir").value });
    $("sshMsg").className = "small " + (r.ok ? "ok-text" : "bad-text"); $("sshMsg").textContent = r.message;
  } catch (e) { $("sshMsg").className = "small bad-text"; $("sshMsg").textContent = e.message; }
  $("sshTestBtn").disabled = false;
};
async function startSend(body) {
  try {
    await api("/api/transfer/send", body);
    S = await api("/api/studio");
    prevSendRunning = true;
    renderSend();
  } catch (e) { toast(e.message, 6000); }
}
$("sshSendBtn").onclick = () => startSend({ method: "ssh", host: $("sshHost").value, dir: $("sshDir").value });
$("crocSendBtn").onclick = () => startSend({ method: "croc" });
$("crocInstallBtn").onclick = async () => {
  try { await api("/api/transfer/install_croc", {}); await loadTransferInfo(); } catch (e) { toast(e.message); }
};
$("cancelSendBtn").onclick = async () => {
  if (!confirm("Cancel sending? You can send again later.")) return;
  try { await api("/api/transfer/cancel", {}); } catch (e) { toast(e.message); }
};
$("copyCmdBtn").onclick = async () => {
  try { await navigator.clipboard.writeText($("crocCmd").textContent); toast("Copied."); }
  catch { toast("Select the text and copy it by hand."); }
};

// ---------- keys ----------
document.addEventListener("keydown", (ev) => {
  if (page !== "review" || !current) return;
  const tag = document.activeElement.tagName;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(tag) && document.activeElement.type !== "range") return;
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  if (player.handleKey(ev)) return;
  const keys = {
    a: () => decide("approve"), A: () => decide("approve"),
    f: () => { $("reasons").hidden = !$("reasons").hidden; }, F: () => { $("reasons").hidden = !$("reasons").hidden; },
    ArrowUp: () => { const e = neighbour(-1); if (e) select(e.name); },
    ArrowDown: () => { const e = neighbour(1); if (e) select(e.name); },
  };
  if (keys[ev.key]) { ev.preventDefault(); keys[ev.key](); }
});

// ---------- start ----------
refresh();
setInterval(refresh, 1500);
setInterval(() => { if (page === "review") loadEpisodes(); }, 4000);
setInterval(() => api("/api/heartbeat", {}).catch(() => {}), 10000);
api("/api/heartbeat", {}).catch(() => {});
