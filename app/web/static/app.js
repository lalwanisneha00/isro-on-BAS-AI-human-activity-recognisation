/* Dashboard shell: mission clock plus camera health polling. */

const clockEl = document.getElementById("clock");
const feedStateEl = document.getElementById("feed-state");
const fpsEl = document.getElementById("fps-value");
const detailEl = document.getElementById("panel-detail");
const pillEl = document.getElementById("live-pill");
const pillLabelEl = document.getElementById("live-label");
const videoEl = document.getElementById("video");
const poseStateEl = document.getElementById("pose-state");
const landmarkCountEl = document.getElementById("landmark-count");
const latencyEl = document.getElementById("latency");

const POSE_STATES = {
  tracking: { label: "Locked", cls: "ok" },
  searching: { label: "Searching", cls: "wait" },
  unavailable: { label: "Offline", cls: "bad" },
};

function tickClock() {
  const now = new Date();
  clockEl.textContent = now.toTimeString().slice(0, 8) + " IST";
}

const FEED_STATES = {
  live: { label: "Live", pill: "online", pillLabel: "LIVE" },
  reconnecting: { label: "Reconnecting", pill: "offline", pillLabel: "NO SIGNAL" },
  starting: { label: "Starting", pill: "", pillLabel: "STANDBY" },
};

async function pollStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    const data = await response.json();

    const state = FEED_STATES[data.camera_status] || FEED_STATES.starting;
    feedStateEl.textContent = state.label;
    pillEl.className = "live-pill " + state.pill;
    pillLabelEl.textContent = state.pillLabel;
    fpsEl.textContent = data.fps ? data.fps.toFixed(1) + " fps" : "--.- fps";

    const tracked = data.pose_status === "tracking";
    poseStateEl.textContent = tracked ? "Locked" : "Searching";
    poseStateEl.className = "meta-value tag-" + (tracked ? "ok" : "wait");
    landmarkCountEl.textContent = tracked ? "33" : "0";
    latencyEl.textContent = data.latency_ms != null ? data.latency_ms.toFixed(0) : "--";

    // While the camera is healthy, the pose detail is the more useful message.
    renderMode(data);
    applyView(data);
    detailEl.textContent =
      (data.mode === "demo" || data.camera_status === "live")
        ? data.pose_detail : data.detail;
  } catch (err) {
    // Server briefly unreachable: show it in the UI, never break the page.
    feedStateEl.textContent = "Offline";
    pillEl.className = "live-pill offline";
    pillLabelEl.textContent = "NO LINK";
    detailEl.textContent = "Dashboard cannot reach the local server";
  }
}

// If the MJPEG stream ever drops, reattach it instead of leaving a broken image.
videoEl.addEventListener("error", () => {
  setTimeout(() => { videoEl.src = "/video?t=" + Date.now(); }, 1500);
});


/* ------------------------------------------------ normalisation telemetry -- */

const normCanvas = document.getElementById("norm-canvas");
const normCtx = normCanvas.getContext("2d");
const bufferFillEl = document.getElementById("buffer-fill");
const bufferTagEl = document.getElementById("buffer-tag");
const windowStateEl = document.getElementById("window-state");

const T = {
  torso: document.getElementById("t-torso"),
  hipx: document.getElementById("t-hipx"),
  hipy: document.getElementById("t-hipy"),
  spine: document.getElementById("t-spine"),
  shx: document.getElementById("t-shx"),
  shy: document.getElementById("t-shy"),
  conf: document.getElementById("t-conf"),
  span: document.getElementById("t-span"),
};

// Same 35 bone segments MediaPipe uses, so the mini-view matches the overlay.
const BONES = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[27,29],[27,31],[29,31],
  [24,26],[26,28],[28,30],[28,32],[30,32],
];

function drawNormalised(points) {
  const w = normCanvas.width, h = normCanvas.height;
  normCtx.clearRect(0, 0, w, h);

  // Reference frame: origin at the hips, one torso length upward.
  // Fixed scale, chosen so a whole body from head to feet fits the canvas:
  // the point of this view is that the figure never changes size, so it must
  // not be auto-fitted to the pose.
  const originX = w / 2, originY = h * 0.60, unit = h * 0.24;

  normCtx.strokeStyle = "rgba(0,217,255,0.14)";
  normCtx.lineWidth = 1;
  normCtx.beginPath();
  normCtx.moveTo(0, originY); normCtx.lineTo(w, originY);
  normCtx.moveTo(originX, 0); normCtx.lineTo(originX, h);
  normCtx.stroke();

  // The "one torso" ring the shoulders should sit on at all times.
  normCtx.setLineDash([3, 4]);
  normCtx.beginPath();
  normCtx.arc(originX, originY, unit, 0, Math.PI * 2);
  normCtx.stroke();
  normCtx.setLineDash([]);

  if (!points) {
    normCtx.fillStyle = "#4E5C75";
    normCtx.font = "10px 'Cascadia Mono', monospace";
    normCtx.textAlign = "center";
    normCtx.fillText("AWAITING CREW", originX, originY);
    return;
  }

  const px = (p) => originX + p[0] * unit;
  const py = (p) => originY + p[1] * unit;

  normCtx.strokeStyle = "#00D9FF";
  normCtx.lineWidth = 1.6;
  normCtx.beginPath();
  for (const [a, b] of BONES) {
    if (!points[a] || !points[b]) continue;
    normCtx.moveTo(px(points[a]), py(points[a]));
    normCtx.lineTo(px(points[b]), py(points[b]));
  }
  normCtx.stroke();

  normCtx.fillStyle = "#E6EDF7";
  for (const p of points) {
    normCtx.beginPath();
    normCtx.arc(px(p), py(p), 1.8, 0, Math.PI * 2);
    normCtx.fill();
  }

  // Hip origin marker, the fixed point everything is measured from.
  normCtx.fillStyle = "#00D9FF";
  normCtx.beginPath();
  normCtx.arc(originX, originY, 3.5, 0, Math.PI * 2);
  normCtx.fill();
}

const blank = () => { for (const k in T) if (k !== "span") T[k].textContent = "--"; };

async function pollTelemetry() {
  try {
    const data = await (await fetch("/api/telemetry", { cache: "no-store" })).json();

    bufferTagEl.textContent = data.buffer_count + "/" + data.buffer_size;
    bufferFillEl.style.width =
      (100 * data.buffer_count / data.buffer_size).toFixed(0) + "%";
    T.span.textContent = data.buffer_span.toFixed(2) + " s";
    windowStateEl.textContent = data.buffer_ready ? "READY" : "FILLING";

    if (!data.has_pose) {
      blank();
      drawNormalised(null);
      return;
    }

    T.torso.textContent = data.torso_length.toFixed(4);
    T.hipx.textContent = data.hip_x.toFixed(4);
    T.hipy.textContent = data.hip_y.toFixed(4);
    T.spine.textContent = data.spine_angle.toFixed(1) + "\u00B0";
    T.shx.textContent = data.norm_shoulder_x.toFixed(4);
    T.shy.textContent = data.norm_shoulder_y.toFixed(4);
    T.conf.textContent = data.confidence.toFixed(3);

    drawNormalised(data.norm_points);
  } catch (err) {
    blank();
    drawNormalised(null);
  }
}


/* ------------------------------------------------------ monitored subject -- */
/* v1 follows exactly one person. The panel shows who is locked, what they are
   doing, and - when other people are in frame - says plainly that they are
   being ignored on purpose. */

const actPanelEl = document.getElementById("activity-panel");
const lockPillEl = document.getElementById("lock-pill");
const lockLabelEl = document.getElementById("lock-label");
const subjectEmptyEl = document.getElementById("subject-empty");
const subjectBodyEl = document.getElementById("subject-body");
const subjectActivityEl = document.getElementById("subject-activity");
const postureChipEl = document.getElementById("posture-chip");
const postureTextEl = document.getElementById("posture-text");
const subjectConfEl = document.getElementById("subject-conf");
const subjectConfFillEl = document.getElementById("subject-conf-fill");
const subjectNameEl = document.getElementById("subject-name");
const subjectHeldEl = document.getElementById("subject-held");
const subjectNoteEl = document.getElementById("subject-note");
const scoreListEl = document.getElementById("score-list");
const scoreTitleEl = document.getElementById("score-title");

/* The task list lives in config.py and is served to the page, so adding an
   activity there updates the score panel, the timeline and the chart without
   any change here. Until the first response arrives these are empty. */
let ACTIVITIES = [];
let ACTIVITY_COLOURS = {};

const scoreRows = {};

function buildScoreRows() {
  scoreListEl.innerHTML = "";
  for (const key of Object.keys(scoreRows)) delete scoreRows[key];

  for (const name of ACTIVITIES) {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML =
      '<span class="score-name"></span><span class="score-value">0.00</span>';
    row.querySelector(".score-name").textContent = name;

    const track = document.createElement("div");
    track.className = "score-track";
    track.innerHTML = '<div class="score-fill"></div>';
    row.appendChild(track);

    scoreListEl.appendChild(row);
    scoreRows[name] = {
      root: row,
      value: row.querySelector(".score-value"),
      fill: row.querySelector(".score-fill"),
    };
  }
}

function renderScores(scores, activity) {
  // Only the activities actually in play are worth showing: a list of twenty
  // rows reading 0.00 tells the reader nothing.
  const shown = ACTIVITIES.filter(
    (name) => ((scores && scores[name]) || 0) > 0.005 || name === activity);

  for (const name of ACTIVITIES) {
    const row = scoreRows[name];
    const value = (scores && scores[name]) || 0;
    row.value.textContent = value.toFixed(2);
    row.fill.style.width = Math.min(100, value * 100).toFixed(0) + "%";
    row.fill.style.background = ACTIVITY_COLOURS[name] || "#6E7686";
    row.root.classList.toggle("is-top", name === activity);
    row.root.hidden = !shown.includes(name);
  }
}

function renderSubject(data) {
  const locked = !!data.locked;
  const visible = !!data.visible;
  const activity = data.activity || "Acquiring";

  // Three honest states: following someone, holding a lock through a brief
  // disappearance, or nobody to follow.
  lockPillEl.className = "lock-pill"
    + (locked && visible ? " locked" : locked ? " holding" : "");
  lockLabelEl.textContent = locked && visible ? "SUBJECT LOCKED"
    : locked ? "HOLDING LOCK" : "NO SUBJECT";

  subjectEmptyEl.hidden = locked;
  subjectBodyEl.hidden = !locked;
  actPanelEl.classList.toggle("alert", !!data.is_anomaly);

  if (!locked) {
    postureChipEl.hidden = true;
    renderScores(null, null);
    return;
  }

  // Posture rides alongside the activity rather than competing with it, so
  // the panel can say "Seated" and "Laptop Work" at the same time.
  const posture = data.posture;
  const knownPosture = posture && posture.posture
    && posture.posture !== "Unknown"
    && !(posture.basis || "").includes("cannot be determined");
  postureChipEl.hidden = !knownPosture;
  if (knownPosture) {
    postureTextEl.textContent =
      posture.posture + "  " + Math.round(posture.confidence * 100) + "%";
    postureChipEl.classList.toggle("uncertain", !posture.seated);
    postureChipEl.title = posture.basis || "";
  }

  const colour = ACTIVITY_COLOURS[activity] || "#6E7686";
  subjectActivityEl.textContent = activity;
  subjectActivityEl.style.color = data.is_anomaly ? "#fab219" : colour;

  const confidence = data.confidence || 0;
  subjectConfEl.textContent =
    confidence > 0 ? (confidence * 100).toFixed(0) + "%" : "--";
  subjectConfFillEl.style.width = (confidence * 100).toFixed(0) + "%";
  subjectConfFillEl.style.background = data.is_anomaly ? "#fab219" : colour;

  subjectNameEl.textContent = data.subject_name || "CM-1";
  subjectHeldEl.textContent = visible
    ? "locked " + formatClock(data.lock_seconds || 0)
    : "briefly out of view";

  // Saying this out loud turns a single tracked person from something that
  // looks like a limit into something that looks like a decision.
  const ignored = data.ignored || 0;
  subjectNoteEl.hidden = ignored === 0;
  if (ignored) {
    subjectNoteEl.textContent =
      ignored + (ignored > 1 ? " other people are" : " other person is")
      + " in frame and is not being tracked. v1 monitors one crew member,"
      + " chosen as the one closest to the camera.";
  }

  renderScores(data.scores, activity);
}

async function pollActivity() {
  try {
    const data = await (await fetch("/api/activity", { cache: "no-store" })).json();
    renderSubject(data);
    renderObjects(data.objects);
    renderAlert(data);
  } catch (err) {
    lockLabelEl.textContent = "LINK LOST";
    lockPillEl.className = "lock-pill holding";
  }
}

/* ---------------------------------------------------------- activity log -- */

const logBodyEl = document.getElementById("log-body");
const logStatusEl = document.getElementById("log-status");
const logPathEl = document.getElementById("log-path");


function formatClock(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

let lastTopEntry = null;

async function pollLog() {
  try {
    const data = await (await fetch("/api/log", { cache: "no-store" })).json();
    ensureActivities(data.order, data.colours);
    const events = data.events || [];

    logStatusEl.textContent =
      events.length === 1 ? "1 entry" : events.length + " entries";

    if (!events.length) {
      logBodyEl.innerHTML =
        '<tr class="empty-row"><td colspan="5">' +
        "No activity segments recorded yet</td></tr>";
    } else {
      const topKey = events[0].timestamp + events[0].activity;
      const isNew = lastTopEntry !== null && topKey !== lastTopEntry;
      lastTopEntry = topKey;

      logBodyEl.innerHTML = events.map((e, i) => {
        const colour = ACTIVITY_COLOURS[e.activity] || "#7C8AA5";
        const fresh = isNew && i === 0 ? " class=\"new-row\"" : "";
        const conf = e.confidence > 0
          ? (e.confidence * 100).toFixed(0) + "%" : "\u2014";
        return `<tr${fresh}>` +
          `<td class="cell-time">${e.clock}</td>` +
          `<td class="cell-crew mono">${e.crew || "\u2014"}</td>` +
          `<td class="cell-activity" style="--row-colour:${colour}">${e.activity}</td>` +
          `<td class="cell-duration num">${formatClock(e.duration)}</td>` +
          `<td class="cell-conf num">${conf}</td></tr>`;
      }).join("");
    }

    if (data.timeline) renderTimeline(data.timeline);
    renderChart(data.totals || {});

    const status = data.status || {};
    if (status.storage_ok === false) {
      logPathEl.textContent = status.detail;
      logPathEl.classList.add("fault");
    } else {
      logPathEl.classList.remove("fault");
      const open = status.open_segments || [];
      logPathEl.textContent =
        "Writing to logs/activity_log.csv and logs/activity_log.db" +
        (open.length
          ? "  \u2014  open: " + open.map(
              (o) => o.crew + " " + o.activity + " " + formatClock(o.seconds)
            ).join(", ")
          : "");
    }
  } catch (err) {
    logStatusEl.textContent = "link lost";
  }
}


/* ------------------------------------------------- timeline and chart ----- */

const timelineEl = document.getElementById("timeline");
const timelineAxisEl = document.getElementById("timeline-axis");
const timelineRangeEl = document.getElementById("timeline-range");
const chartEl = document.getElementById("chart");
const chartTotalEl = document.getElementById("chart-total");
const tooltipEl = document.getElementById("tooltip");

// Fixed activity order, shared with the server. Never sorted by value: a bar
// keeps its position and colour so the reader is not re-learning the chart.
const lanes = {};
const bars = {};

function buildTimelineAndChart() {
  timelineEl.innerHTML = "";
  chartEl.innerHTML = "";
  for (const key of Object.keys(lanes)) delete lanes[key];
  for (const key of Object.keys(bars)) delete bars[key];

  for (const name of ACTIVITIES) {
  const colour = ACTIVITY_COLOURS[name];

  const lane = document.createElement("div");
  lane.className = "lane";
  lane.style.setProperty("--lane-colour", colour);
  lane.innerHTML =
    '<div class="lane-name"><span class="lane-swatch"></span><span></span></div>' +
    '<div class="lane-track"></div>';
  lane.querySelector(".lane-name span:last-child").textContent = name;
  timelineEl.appendChild(lane);
  lanes[name] = { root: lane, track: lane.querySelector(".lane-track") };

  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML =
    '<div class="bar-head"><span class="bar-name"></span>' +
    '<span class="bar-value">0s</span></div>' +
    '<div class="bar-track"><div class="bar-fill"></div></div>';
  row.querySelector(".bar-name").textContent = name;
  row.querySelector(".bar-fill").style.setProperty("--bar-colour", colour);
  chartEl.appendChild(row);
    bars[name] = {
      root: row,
      value: row.querySelector(".bar-value"),
      fill: row.querySelector(".bar-fill"),
    };
  }
}

/* Rebuild everything the first time the server tells us the task list, and
   again if it ever changes. */
function ensureActivities(order, colours) {
  if (!order || !order.length) return;
  const signature = order.join("|");
  if (signature === ensureActivities.signature) return;
  ensureActivities.signature = signature;

  ACTIVITIES = order;
  ACTIVITY_COLOURS = colours || {};
  buildScoreRows();
  buildTimelineAndChart();
}

function showTooltip(event, activity, detail) {
  tooltipEl.style.setProperty("--tip-colour", ACTIVITY_COLOURS[activity] || "#6E7686");
  tooltipEl.innerHTML =
    '<div class="tip-activity"><span class="tip-swatch"></span>' +
    '<span class="tip-name"></span></div><div class="tip-detail"></div>';
  tooltipEl.querySelector(".tip-name").textContent = activity;
  tooltipEl.querySelector(".tip-detail").textContent = detail;
  tooltipEl.hidden = false;
  moveTooltip(event);
}

function moveTooltip(event) {
  const pad = 14;
  const rect = tooltipEl.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tooltipEl.style.left = x + "px";
  tooltipEl.style.top = y + "px";
}

const hideTooltip = () => { tooltipEl.hidden = true; };

function renderTimeline(timeline) {
  const blocks = timeline.blocks || [];
  const grouped = {};
  for (const name of ACTIVITIES) grouped[name] = [];
  for (const b of blocks) {
    if (grouped[b.activity]) grouped[b.activity].push(b);
  }

  for (const name of ACTIVITIES) {
    const lane = lanes[name];
    const items = grouped[name];
    lane.root.classList.toggle("active", items.length > 0);

    lane.track.innerHTML = "";
    for (const b of items) {
      const el = document.createElement("div");
      el.className = "block" + (b.open ? " live" : "");
      el.style.left = (b.offset * 100).toFixed(3) + "%";
      el.style.width = Math.max(0.35, b.width * 100).toFixed(3) + "%";
      const who = b.crew ? b.crew + "  \u00B7  " : "";
      el.addEventListener("mouseenter", (ev) => showTooltip(
        ev, name,
        `${who}${b.clock}  \u00B7  ${formatClock(b.duration)}` +
        `${b.open ? "  \u00B7  in progress" : ""}`));
      el.addEventListener("mousemove", moveTooltip);
      el.addEventListener("mouseleave", hideTooltip);
      lane.track.appendChild(el);
    }
  }

  const minutes = Math.round(timeline.window_seconds / 60);
  timelineRangeEl.textContent = `Last ${minutes} minutes`;

  const end = new Date(timeline.now * 1000);
  const mid = new Date((timeline.now - timeline.window_seconds / 2) * 1000);
  const begin = new Date(timeline.start * 1000);
  const hhmm = (d) => d.toTimeString().slice(0, 5);
  timelineAxisEl.innerHTML =
    `<span>${hhmm(begin)}</span><span>${hhmm(mid)}</span><span>NOW ${hhmm(end)}</span>`;
}

function renderChart(totals) {
  let grand = 0, peak = 0;
  for (const name of ACTIVITIES) {
    const v = totals[name] || 0;
    grand += v;
    if (v > peak) peak = v;
  }

  chartTotalEl.textContent = formatClock(grand);

  for (const name of ACTIVITIES) {
    const v = totals[name] || 0;
    const bar = bars[name];
    bar.root.classList.toggle("has-data", v > 0);
    bar.value.textContent = v > 0 ? formatClock(v) : "\u2014";
    bar.fill.style.width = peak > 0 ? Math.max(0, (v / peak) * 100).toFixed(1) + "%" : "0%";

    if (v > 0) {
      const share = grand > 0 ? ((v / grand) * 100).toFixed(0) : "0";
      bar.root.onmouseenter = (ev) => showTooltip(
        ev, name, `${formatClock(v)}  \u00B7  ${share}% of session`);
      bar.root.onmousemove = moveTooltip;
      bar.root.onmouseleave = hideTooltip;
    } else {
      bar.root.onmouseenter = null;
      bar.root.onmousemove = null;
      bar.root.onmouseleave = null;
    }
  }
}

/* --------------------------------------------- anomaly alert & demo mode -- */

const alertBannerEl = document.getElementById("alert-banner");
const alertDetailEl = document.getElementById("alert-detail");
const alertTimerEl = document.getElementById("alert-timer");
const modeLiveEl = document.getElementById("mode-live");
const modeDemoEl = document.getElementById("mode-demo");
const demoBadgeEl = document.getElementById("demo-badge");
const demoProgressEl = document.getElementById("demo-progress");

function renderAlert(data) {
  const active = !!data.is_anomaly;
  alertBannerEl.hidden = !active;
  document.body.classList.toggle("anomaly", active);

  if (active) {
    const seconds = Math.max(data.still_seconds || 0, data.duration || 0);
    alertTimerEl.textContent = Math.floor(seconds) + "s";
    alertDetailEl.textContent =
      "Crew member has shown no motion for " + formatClock(seconds) +
      " \u2014 confirm crew status";
  }
}

async function applyMode(mode) {
  modeLiveEl.disabled = modeDemoEl.disabled = true;
  try {
    await fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    // Reattach the stream so the switch is visible immediately.
    videoEl.src = "/video?t=" + Date.now();
  } catch (err) {
    /* The next status poll will show the real state. */
  } finally {
    modeLiveEl.disabled = modeDemoEl.disabled = false;
  }
}

modeLiveEl.addEventListener("click", () => applyMode("live"));
modeDemoEl.addEventListener("click", () => applyMode("demo"));

function renderMode(data) {
  const demo = data.mode === "demo";
  modeLiveEl.classList.toggle("is-on", !demo);
  modeDemoEl.classList.toggle("is-on", demo);
  demoBadgeEl.hidden = !demo;

  if (demo) {
    demoProgressEl.textContent =
      formatClock(data.demo_elapsed || 0) + " / " + formatClock(data.demo_duration || 0);
  }
  if (!data.demo_available) {
    modeDemoEl.title = data.demo_detail || "Demo clip unavailable";
  }
}

/* ---------------------------------------------------------------- start --- */
/* Every poller is started here, at the end, so each one is guaranteed that
   the elements and lookup tables it writes into already exist. */

tickClock();
drawNormalised(null);

pollStatus();
pollTelemetry();
pollActivity();
pollLog();

setInterval(tickClock, 1000);
setInterval(pollStatus, 1000);
setInterval(pollTelemetry, 150);
setInterval(pollActivity, 250);
setInterval(pollLog, 900);

/* ------------------------------------------ overlay, privacy, downloads --- */
/* Presentation controls. None of these touch classification or logging: the
   same frames are analysed and the same rows written whatever is toggled. */

const btnSkeleton = document.getElementById("btn-skeleton");
const btnObjects = document.getElementById("btn-objects");
const btnPrivacy = document.getElementById("btn-privacy");
const videoFrameEl = document.querySelector(".video-frame");

const objectsEl = document.getElementById("objects");
const objectsEmptyEl = document.getElementById("objects-empty");
const objectStatusEl = document.getElementById("object-status");

const downloadScopeEl = document.getElementById("download-scope");

async function setView(payload, button) {
  if (button) button.classList.add("is-busy");
  try {
    const data = await (await fetch("/api/view", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })).json();
    applyView(data);
  } catch (err) {
    /* The next status poll will show the true state. */
  } finally {
    if (button) button.classList.remove("is-busy");
  }
}

function applyView(data) {
  const privacy = data.video_mode === "privacy";
  btnSkeleton.classList.toggle("is-on", !!data.show_skeleton);
  btnObjects.classList.toggle("is-on", !!data.show_objects);
  btnPrivacy.classList.toggle("is-on", privacy);
  if (videoFrameEl) videoFrameEl.classList.toggle("privacy", privacy);

  // Object boxes cannot be drawn over a suppressed video.
  btnObjects.disabled = privacy;

  const objects = data.objects || {};
  if (!objects.enabled) {
    objectStatusEl.textContent = "off";
  } else if (!objects.available) {
    objectStatusEl.textContent = "unavailable";
  } else {
    objectStatusEl.textContent =
      objects.model + "  " + Math.round(objects.inference_ms) + " ms";
  }
}

btnSkeleton.addEventListener("click", () => setView(
  { show_skeleton: !btnSkeleton.classList.contains("is-on") }, btnSkeleton));
btnObjects.addEventListener("click", () => setView(
  { show_objects: !btnObjects.classList.contains("is-on") }, btnObjects));
btnPrivacy.addEventListener("click", () => setView(
  { video_mode: btnPrivacy.classList.contains("is-on") ? "normal" : "privacy" },
  btnPrivacy));

function renderObjects(detections) {
  const items = detections || [];
  objectsEmptyEl.hidden = items.length > 0;

  for (const el of Array.from(objectsEl.querySelectorAll(".object-row"))) {
    el.remove();
  }
  // In-hand objects first: those are the ones changing the classification.
  const ordered = items.slice().sort(
    (a, b) => (b.in_hand ? 1 : 0) - (a.in_hand ? 1 : 0));

  for (const item of ordered) {
    const row = document.createElement("div");
    row.className = "object-row" + (item.in_hand ? " in-hand" : "");
    row.innerHTML =
      '<span class="object-dot"></span>' +
      '<span class="object-name"></span>' +
      '<span class="object-held"></span>' +
      '<span class="object-conf"></span>';
    row.querySelector(".object-name").textContent = item.name;
    row.querySelector(".object-held").textContent = item.in_hand ? "IN HAND" : "";
    row.querySelector(".object-conf").textContent =
      (item.confidence * 100).toFixed(0) + "%";
    objectsEl.appendChild(row);
  }
}

function download(kind) {
  const scope = downloadScopeEl ? downloadScopeEl.value : "session";
  // A plain navigation, so the browser saves the file the server names.
  window.location.href = "/api/download?kind=" + kind + "&scope=" + scope;
}

document.getElementById("btn-csv")
  .addEventListener("click", () => download("csv"));
document.getElementById("btn-json")
  .addEventListener("click", () => download("json"));
document.getElementById("btn-report")
  .addEventListener("click", () => download("report"));
