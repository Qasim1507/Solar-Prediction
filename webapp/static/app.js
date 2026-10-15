// Polls the Flask job runner and renders forecast / verification views.
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const FRAME_LABELS = {
  "himawari_prev2.png": "t − 20 min",
  "himawari_prev1.png": "t − 10 min",
  "himawari_current.png": "t (issue)",
};
const FRAME_ORDER = ["himawari_prev2.png", "himawari_prev1.png", "himawari_current.png"];

let cursor = 0;          // lines of the current job already shown
let jobKey = null;       // kind + start time of the job being displayed
let lastState = null;
let frameMtimes = {};
let verifyTimer = null;

// The Vercel build (webapp/export_static.py) is a read-only snapshot: the API
// responses are saved under /data, and there is no job runner to poll.
const STATIC = window.STATIC_SITE === true;
const runId = (t) => t.replace(/[-:]/g, "").replace(" ", "_");  // app.run_id_for
const API = {
  forecast: () => (STATIC ? "/data/forecast.json" : "/api/forecast"),
  frames: () => (STATIC ? "/data/frames.json" : "/api/frames"),
  verification: (t) => STATIC
    ? `/data/verification/${t ? runId(t) : "latest"}.json`
    : (t ? `/api/verification?forecast_time=${encodeURIComponent(t)}` : "/api/verification"),
};

// Explainer text for the ⓘ tooltips, kept in one place.
const TIPS = {
  ghi: "Global Horizontal Irradiance: sunlight power reaching flat ground, in watts per square metre. ~1000 W/m² is bright midday sun.",
  ci: "90% interval: the model expects the real value to fall inside this range about 9 times out of 10.",
  kt: "Clear-sky index: the fraction of a cloudless sky's sunlight that gets through. 1.0 = clear, ~0.3 = heavy cloud.",
  clearsky: "Sunlight a perfectly cloudless sky would give over that hour, from solar geometry (pvlib).",
  method: "v3 ensemble is the deep-learning model. Smart persistence is a simple baseline that assumes today's cloudiness stays the same.",
  sp: "Smart persistence: a naive baseline that keeps the current clear-sky index unchanged. A useful model should beat it.",
  challenger: "Published alongside the model so the two can be compared fairly on the same hours.",
  abs_error: "How far the forecast was from the reported value, in W/m². Lower is better.",
  in_ci: "Whether the reported value fell inside the forecast's 90% range.",
  actual: "Sunlight reported for that hour by the ERA5 reanalysis (smoothed; not a ground sensor).",
  image_ok: "The latest satellite frame downloaded and looks valid (not blank or corrupted).",
  lookback: "Whether the 24 h weather history runs right up to the forecast time. Stale data makes the forecast less reliable.",
  hours: "The model was trained on 08:00–17:00 SGT only. Forecasts outside those hours should be treated with caution.",
};
const tip = (key) => `<span class="tip" tabindex="0" data-tip="${esc(TIPS[key])}" aria-label="${esc(TIPS[key])}">ⓘ</span>`;

// One floating box for every tooltip, so tables that scroll don't clip it.
const tipBox = document.createElement("div");
tipBox.className = "tipbox";
tipBox.hidden = true;
document.body.appendChild(tipBox);
function showTip(el) {
  tipBox.textContent = el.dataset.tip;
  tipBox.hidden = false;
  const r = el.getBoundingClientRect();
  const w = tipBox.offsetWidth;
  tipBox.style.left = `${Math.max(8, Math.min(r.left + r.width / 2 - w / 2, innerWidth - w - 8))}px`;
  const below = r.bottom + 8 + tipBox.offsetHeight < innerHeight;
  tipBox.style.top = `${below ? r.bottom + 8 : r.top - tipBox.offsetHeight - 8}px`;
}
const hideTip = () => (tipBox.hidden = true);
for (const ev of ["mouseover", "focusin"]) {
  document.addEventListener(ev, (e) => { const t = e.target.closest?.(".tip"); if (t) showTip(t); });
}
for (const ev of ["mouseout", "focusout"]) {
  document.addEventListener(ev, (e) => { if (e.target.closest?.(".tip")) hideTip(); });
}
addEventListener("scroll", hideTip, true);

// ── admin key ─────────────────────────────────────────────────────────────
// /?key=… stores the token for this browser, then drops it from the URL.
let adminKey = null;
try {
  const params = new URLSearchParams(location.search);
  if (params.has("key")) {
    localStorage.setItem("adminKey", params.get("key"));
    params.delete("key");
    const q = params.toString();
    history.replaceState(null, "", location.pathname + (q ? `?${q}` : "") + location.hash);
  }
  adminKey = localStorage.getItem("adminKey");
} catch { /* storage blocked: visitor mode */ }
const authHeaders = () => (adminKey ? { "X-Admin-Token": adminKey } : {});

async function loadConfig() {
  let admin = false;
  if (STATIC) {
    try {
      const m = await (await fetch("/data/meta.json")).json();
      $("job-status").textContent = `updated ${m.exported_sgt} SGT`;
    } catch {}
    return;
  }
  try { admin = (await (await fetch("/api/config", { headers: authHeaders() })).json()).admin; } catch {}
  document.querySelector(".actions").hidden = !admin;
  document.querySelectorAll("[data-log]").forEach((d) => (d.open = admin));
  if (admin) {
    $("log-predict").textContent = "Press “Run prediction” to fetch live data and forecast.";
    $("log-verify").textContent = "Press “Run verification” to score the latest forecast.";
  }
}

// ── intro ───────────────────────────────────────────────────────────────────
function setHero(show) {
  $("hero").hidden = !show;
  $("hero-show").hidden = show;
  try { localStorage.setItem("heroHidden", show ? "" : "1"); } catch {}
}
$("hero-hide").addEventListener("click", () => setHero(false));
$("hero-show").addEventListener("click", () => setHero(true));
try { if (localStorage.getItem("heroHidden")) setHero(false); } catch {}

// ── tabs ────────────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ── jobs ────────────────────────────────────────────────────────────────────
async function startJob(kind) {
  showTab(kind);
  const r = await fetch(`/api/${kind}`, { method: "POST", headers: authHeaders() });
  if (r.status === 409 || r.status === 403) {
    const j = await r.json();
    $("job-status").textContent = j.error;
    return;
  }
  poll();
}
document.querySelectorAll(".actions button").forEach((b) => b.addEventListener("click", () => startJob(b.dataset.kind)));

async function poll() {
  let j;
  try {
    j = await (await fetch(`/api/job?since=${cursor}`)).json();
  } catch {
    $("job-status").textContent = "server not reachable";
    return;
  }
  if (j.kind) {
    const key = `${j.kind}@${j.started}`;
    const log = $(`log-${j.kind}`);
    if (key !== jobKey) {
      // a new job: restart the log from line 0
      jobKey = key;
      cursor = 0;
      log.textContent = "";
      if (j.state === "running") log.closest("details").open = true;
      return poll();
    }
    if (j.lines.length) {
      const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
      log.textContent += (log.textContent ? "\n" : "") + j.lines.join("\n");
      if (atBottom) log.scrollTop = log.scrollHeight;
    }
    cursor = j.next;

    const running = j.state === "running";
    document.querySelectorAll(".actions button").forEach((b) => (b.disabled = running));
    $("job-status").textContent = running ? `${j.kind} running…` : `${j.kind} ${j.state} (exit ${j.returncode})`;
    $("job-status").className = `job ${j.state}`;

    if (running && j.kind === "predict") loadFrames(j.started);
    if (lastState === "running" && !running) {
      if (j.kind === "predict") { loadFrames(j.started); loadForecast(); }
      loadVerification(j.kind === "verify" ? null : $("ftime").value || null);
    }
    lastState = j.state;
  }
}
if (!STATIC) setInterval(poll, 1000);

// ── prediction view ─────────────────────────────────────────────────────────
async function loadFrames(jobStarted) {
  const frames = await (await fetch(API.frames())).json();
  const box = $("frames");
  if (!box.children.length) {
    box.innerHTML = FRAME_ORDER.map((n) => `
      <figure id="fr-${n.split(".")[0]}">
        <div class="img-slot"><span class="muted">waiting…</span></div>
        <figcaption>${FRAME_LABELS[n]}</figcaption>
      </figure>`).join("");
  }
  for (const f of frames) {
    const fig = $(`fr-${f.name.split(".")[0]}`);
    const slot = fig.querySelector(".img-slot");
    if (!f.mtime) { slot.innerHTML = `<span class="muted">not downloaded</span>`; continue; }
    const fresh = jobStarted && f.mtime >= jobStarted;
    if (frameMtimes[f.name] !== f.mtime) {
      frameMtimes[f.name] = f.mtime;
      slot.innerHTML = `<img src="${f.url}?v=${f.mtime}" alt="${FRAME_LABELS[f.name]}">`;
    }
    fig.classList.toggle("fresh", !!fresh);
    fig.classList.toggle("stale", !!jobStarted && !fresh);
    const when = new Date(f.mtime * 1000).toLocaleTimeString();
    fig.querySelector("figcaption").innerHTML = `${FRAME_LABELS[f.name]} <span class="muted">· saved ${when}</span>`;
  }
}

async function loadForecast() {
  const { forecast: fc, weather, collected } = await (await fetch(API.forecast())).json();
  if (weather) {
    const items = [
      ["Temperature", weather.avg_temperature_c, "°C"],
      ["Humidity", weather.avg_humidity_pct, "%"],
      ["Rainfall", weather.avg_rainfall_mm, "mm"],
      ["Wind", weather.avg_wind_speed_kmh, "km/h"],
    ];
    $("weather").innerHTML = items.map(([k, v, u]) =>
      `<div class="stat"><span class="k">${k}</span><span class="v">${fmt(v)}<small> ${u}</small></span></div>`).join("");
    $("collected").textContent = collected ? `collected ${collected}` : "";
  }
  if (!fc) return;
  $("latest").textContent = `latest forecast issued ${fc.forecast_time_sgt} SGT`;
  $("fc-time").textContent = `issued ${fc.forecast_time_sgt} SGT · seeds ${fc.seeds.join(", ")}`;
  $("forecast").innerHTML = `
    <table>
      <thead><tr><th>Horizon</th><th>Target (SGT)</th><th>GHI W/m² ${tip("ghi")}</th><th>90% interval ${tip("ci")}</th><th>k_t ${tip("kt")}</th><th>Clear-sky ${tip("clearsky")}</th><th>Method ${tip("method")}</th></tr></thead>
      <tbody>${fc.forecasts.map((f) => `
        <tr class="${f.method === "smart_persistence" ? "challenger" : ""}">
          <td>${esc(f.horizon)}</td><td>${esc(f.time_sgt.slice(11))}</td>
          <td class="num ghi">${fmt(f.ghi_forecast_wm2)}</td>
          <td class="num ci">[${fmt(f.ghi_lower_90)} – ${fmt(f.ghi_upper_90)}]</td>
          <td class="num">${fmt(f.kt_forecast, 3)}</td>
          <td class="num">${fmt(f.clearsky_hour_mean_wm2, 0)}</td>
          <td>${f.method === "smart_persistence" ? "smart persistence" : "v3 ensemble"}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
  const d = fc.diagnostics || {};
  const badge = (ok, good, bad, key) => `<span class="badge ${ok ? "ok" : "bad"}">${ok ? good : bad} ${tip(key)}</span>`;
  $("diagnostics").innerHTML = `<span class="k">Input checks</span>` + [
    badge(d.image_ok, "satellite image OK", "satellite image suspect", "image_ok"),
    badge(d.lookback_fresh, `weather history fresh (ends ${esc(d.lookback_ends || "")})`, `weather history STALE (ends ${esc(d.lookback_ends || "")})`, "lookback"),
    badge(!d.outside_training_hours, "inside training hours", "outside training hours", "hours"),
  ].join("");
}

// ── verification view ───────────────────────────────────────────────────────
function imgBlock(img, alt) {
  if (!img || img.status === "pending") return `<div class="img-slot"><span class="muted">hour not reached yet</span></div>`;
  if (img.status === "fetching") return `<div class="img-slot"><span class="spinner"></span><span class="muted">downloading scan…</span></div>`;
  if (img.status === "missing") return `<div class="img-slot"><span class="muted">no scan in NOAA archive</span></div>`;
  const cap = img.scan_utc ? `<figcaption class="muted">scan ${esc(img.scan_utc)} UTC</figcaption>` : "";
  return `<figure><div class="img-slot"><img src="${img.url}" alt="${esc(alt)}"></div>${cap}</figure>`;
}

function column(c) {
  const v = c.v3 || {};
  const verified = c.status === "verified";
  const statusText = { verified: "verified", pending: "pending — hour not reached", unverified: "not verified yet" }[c.status];
  let body = `
    <div class="pair">
      <div><span class="k">Forecast</span><span class="v ghi">${fmt(v.forecast)}</span></div>
      <div><span class="k">Actual ${tip("actual")}</span><span class="v">${verified ? fmt(v.actual) : "—"}</span></div>
    </div>
    <div class="line"><span class="k">90% interval ${tip("ci")}</span><span class="ci">${v.lower != null ? `[${fmt(v.lower)} – ${fmt(v.upper)}]` : "—"}</span></div>`;
  if (verified) {
    body += `
      <div class="line"><span class="k">Abs error ${tip("abs_error")}</span><span>${fmt(v.abs_error)} W/m²</span></div>
      <div class="line"><span class="k">k_t forecast / actual ${tip("kt")}</span><span>${fmt(v.kt_forecast, 3)} / ${fmt(v.kt_actual, 3)}</span></div>
      <div class="line"><span class="k">In 90% interval ${tip("in_ci")}</span><span class="badge ${v.in_ci ? "ok" : "bad"}">${v.in_ci ? "yes" : "no"}</span></div>
      <div class="line ref"><span class="k">Smart persistence ref ${tip("sp")}</span><span>${fmt(v.sp_forecast)} · err ${fmt(v.sp_abs_error)}</span></div>`;
  }
  if (c.challenger) {
    const s = c.challenger;
    body += `
      <div class="challenger-box">
        <div class="k">A/B challenger · smart persistence ${tip("challenger")}</div>
        <div class="line"><span class="k">Forecast</span><span>${fmt(s.forecast)} ${s.lower != null ? `<span class="ci">[${fmt(s.lower)} – ${fmt(s.upper)}]</span>` : ""}</span></div>
        ${s.actual != null ? `<div class="line"><span class="k">Abs error</span><span>${fmt(s.abs_error)} W/m² <span class="badge ${s.in_ci ? "ok" : "bad"}">${s.in_ci ? "in CI" : "outside CI"}</span></span></div>` : ""}
      </div>`;
  }
  return `
    <article class="card col ${c.status}">
      <header><h2>${esc(c.horizon)}</h2><span class="muted">${esc(c.target_time)} SGT</span></header>
      <div class="status ${c.status}">${statusText}</div>
      ${imgBlock(c.image, `${c.horizon} satellite`)}
      ${body}
    </article>`;
}

async function loadVerification(ftime) {
  clearTimeout(verifyTimer);
  const r = await fetch(API.verification(ftime));
  if (!r.ok) { $("columns").innerHTML = `<div class="card muted">No forecast to verify yet.</div>`; return; }
  const d = await r.json();

  const sel = $("ftime");
  sel.innerHTML = d.times.map((t) => `<option ${t === d.forecast_time ? "selected" : ""}>${esc(t)}</option>`).join("");
  $("verify-note").textContent = d.has_forecast_json ? "" : "No saved forecast file for this run; intervals unavailable.";
  $("columns").innerHTML = d.columns.map(column).join("");
  $("issue-src").textContent = d.issue_image.source || "";
  $("issue-img").innerHTML = imgBlock(d.issue_image, "issue-time satellite");

  const waiting = [d.issue_image, ...d.columns.map((c) => c.image)].some((i) => i && i.status === "fetching");
  if (waiting) verifyTimer = setTimeout(() => loadVerification(d.forecast_time), 2000);
}
$("ftime").addEventListener("change", (e) => loadVerification(e.target.value));

// ── initial load ────────────────────────────────────────────────────────────
loadConfig();
loadForecast();
loadFrames(null);
// /?forecast_time=2026-09-30%2010:00#verify opens a past day straight on the Verify tab
loadVerification(new URLSearchParams(location.search).get("forecast_time"));
if (["#verify", "#about"].includes(location.hash)) showTab(location.hash.slice(1));
if (!STATIC) poll();
