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
  backtest: () => (STATIC ? "/data/backtest.json" : "/api/backtest"),
  btRuns: () => (STATIC ? "/data/backtest/runs.json" : "/api/backtest/runs"),
  btDay: (d) => (STATIC ? `/data/backtest/days/${d}.json` : `/api/backtest/day?date=${d}`),
  btRun: (t) => (STATIC ? `/data/backtest/runs/${runId(t)}.json`
    : `/api/backtest/run?forecast_time=${encodeURIComponent(t)}`),
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
  skill: "How much smaller the model's error is than the persistence baseline's. Positive means the model is better.",
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

// ── backtest ────────────────────────────────────────────────────────────────
const SERIES = [
  { key: "mae", name: "v3 model", color: "var(--s-model)" },
  { key: "sp_mae", name: "Smart persistence", color: "var(--s-sp)" },
];

function scoreTable(rows, label) {
  if (!rows || !rows.length) return `<span class="muted">No data.</span>`;
  const head = `<tr><th>${label}</th><th>n</th><th>Model MAE ${tip("abs_error")}</th>
    <th>Persistence MAE ${tip("sp")}</th><th>Skill ${tip("skill")}</th>
    <th>Bias</th><th>In 90% range ${tip("ci")}</th></tr>`;
  const body = rows.map((r) => `<tr><td>${esc(r.key)}</td><td class="num">${r.n}</td>
    <td class="num ghi">${fmt(r.mae)}</td><td class="num">${fmt(r.sp_mae)}</td>
    <td class="num ${r.skill > 0 ? "good" : "worse"}">${r.skill > 0 ? "+" : ""}${fmt(r.skill)}%</td>
    <td class="num">${r.bias > 0 ? "+" : ""}${fmt(r.bias)}</td><td class="num">${fmt(r.picp, 0)}%</td></tr>`).join("");
  return `<table>${head}${body}</table>`;
}

function dailyChart(el, rows, onPick) {
  const W = Math.max(320, el.clientWidth), H = 260;
  const m = { t: 12, r: W < 600 ? 12 : 120, b: 26, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const max = Math.max(...rows.flatMap((r) => SERIES.map((s) => r[s.key])));
  const step = max > 300 ? 100 : 50;
  const top = Math.ceil(max / step) * step;
  const x = (i) => m.l + (rows.length < 2 ? iw / 2 : (i / (rows.length - 1)) * iw);
  const y = (v) => m.t + ih - (v / top) * ih;

  const ticks = [];
  for (let v = 0; v <= top; v += step) ticks.push(v);
  const grid = ticks.map((v) => `<line x1="${m.l}" x2="${m.l + iw}" y1="${y(v)}" y2="${y(v)}"/>`).join("");
  const yl = ticks.map((v) => `<text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("");
  // x labels: the 1st of each month
  const xl = rows.map((r, i) => (r.key.endsWith("-01") || i === 0)
    ? `<text x="${x(i)}" y="${H - 6}" text-anchor="start">${new Date(r.key + "T00:00").toLocaleDateString("en-GB", { day: "numeric", month: "short" })}</text>` : "").join("");

  const lines = SERIES.map((s) => `<path class="series" stroke="${s.color}" d="${rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(r[s.key]).toFixed(1)}`).join("")}"/>`).join("");
  // direct labels at the right end, nudged apart if they collide
  const last = rows.length - 1;
  const ends = SERIES.map((s) => ({ s, y: y(rows[last][s.key]) })).sort((a, b) => a.y - b.y);
  if (ends.length === 2 && ends[1].y - ends[0].y < 14) { const mid = (ends[0].y + ends[1].y) / 2; ends[0].y = mid - 7; ends[1].y = mid + 7; }
  const labels = m.r < 100 ? "" : ends.map((e) => `<text class="label" x="${x(last) + 8}" y="${e.y + 4}">${esc(e.s.name)}</text>`).join("");

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Daily mean absolute error, v3 model vs smart persistence">
    <g class="grid">${grid}</g><g class="axis">${yl}${xl}</g>${lines}${labels}
    <line class="hair" y1="${m.t}" y2="${m.t + ih}" visibility="hidden"/>
    ${SERIES.map((s) => `<circle class="dot" r="4" fill="${s.color}" visibility="hidden"/>`).join("")}
    <rect x="${m.l}" y="${m.t}" width="${iw}" height="${ih}" fill="transparent" tabindex="0"/></svg>
    <div class="chart-tip" hidden></div>`;

  const svg = el.querySelector("svg"), hit = svg.querySelector("rect"), hair = svg.querySelector(".hair");
  const dots = svg.querySelectorAll(".dot"), tipEl = el.querySelector(".chart-tip");
  let cur = last;
  const show = (i) => {
    cur = Math.max(0, Math.min(last, i));
    const r = rows[cur], px = x(cur);
    hair.setAttribute("x1", px); hair.setAttribute("x2", px); hair.setAttribute("visibility", "visible");
    SERIES.forEach((s, k) => { dots[k].setAttribute("cx", px); dots[k].setAttribute("cy", y(r[s.key])); dots[k].setAttribute("visibility", "visible"); });
    tipEl.replaceChildren();
    const d = document.createElement("div"); d.className = "d";
    d.textContent = `${r.key} · ${r.n} forecast hours`; tipEl.append(d);
    SERIES.forEach((s) => {
      const row = document.createElement("div"); row.className = "r";
      const key = document.createElement("i"); key.style.background = s.color;
      const v = document.createElement("b"); v.textContent = fmt(r[s.key]);
      const n = document.createElement("span"); n.textContent = s.name;
      row.append(key, v, n); tipEl.append(row);
    });
    tipEl.hidden = false;
    const scale = svg.getBoundingClientRect().width / W;
    const left = px * scale + 12;
    tipEl.style.left = `${left + tipEl.offsetWidth > el.clientWidth ? px * scale - tipEl.offsetWidth - 12 : left}px`;
    tipEl.style.top = `${m.t * scale}px`;
  };
  const hide = () => { tipEl.hidden = true; hair.setAttribute("visibility", "hidden"); dots.forEach((c) => c.setAttribute("visibility", "hidden")); };
  hit.addEventListener("pointermove", (e) => {
    const b = svg.getBoundingClientRect();
    const sx = ((e.clientX - b.left) / b.width) * W;
    show(Math.round(((sx - m.l) / iw) * last));
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("click", () => onPick && onPick(rows[cur].key));
  hit.addEventListener("focus", () => show(cur));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { show(cur - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(cur + 1); e.preventDefault(); }
    if (e.key === "Enter" && onPick) onPick(rows[cur].key);
  });
}

async function loadBacktest() {
  const r = await fetch(API.backtest());
  if (!r.ok) { $("tab-backtest").querySelectorAll(".card").forEach((c, i) => { if (i) c.hidden = true; }); return; }
  const d = await r.json();
  const o = d.overall;
  $("bt-period").textContent = `${d.start} → ${d.end} · ${d.forecasts} forecasts over ${d.days} days`;
  $("bt-headline").innerHTML = [
    ["Model error (MAE)", `${fmt(o.mae)} W/m²`],
    ["Persistence error (MAE)", `${fmt(o.sp_mae)} W/m²`],
    ["Skill vs persistence", `${o.skill > 0 ? "+" : ""}${fmt(o.skill)}%`],
    ["Inside 90% range", `${fmt(o.picp, 0)}%`],
  ].map(([k, v]) => `<div class="stat"><span class="k">${k}</span><span class="v">${v}</span></div>`).join("");
  $("bt-legend").innerHTML = SERIES.map((s) => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join("");
  // Drawn at the real pixel width so text stays 11px; redrawn on resize.
  // The tab may be hidden at load (width 0), so draw when it is first shown too.
  const draw = () => { if ($("bt-chart").clientWidth) dailyChart($("bt-chart"), d.daily, pickDay); };
  draw();
  let rz; addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(draw, 150); });
  document.querySelector('.tab[data-tab="backtest"]').addEventListener("click", () => requestAnimationFrame(draw));
  $("bt-horizons").innerHTML = scoreTable(d.horizons, "Horizon");
  $("bt-months").innerHTML = scoreTable(d.months, "Month");
  $("bt-hours").innerHTML = scoreTable(d.hours, "Issued at");
}

// ── backtest: one day, and one forecast in full ─────────────────────────────
let btTimes = [];          // every backtested issue time, oldest first
let btDays = [];           // the days they fall on, oldest first
let btDay = null;          // the day on screen
let btCurrent = null;      // the forecast opened in full

function markStrip(ftime) {
  const issue = ftime.slice(11, 16);
  const targets = [1, 2, 3].map((h) => `${String(+issue.slice(0, 2) + h).padStart(2, "0")}:00`);
  document.querySelectorAll("#bt-strip figure").forEach((f) => {
    const t = f.dataset.time;
    f.classList.toggle("issue", t === issue);
    f.classList.toggle("target", targets.includes(t));
    f.querySelector(".role").textContent = t === issue ? "issued" : targets.includes(t) ? `t+${targets.indexOf(t) + 1}h` : "";
  });
  document.querySelectorAll("#bt-day-table tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.ftime === ftime));
}

function dayTable(d) {
  const cell = (z) => z ? `<td class="num">${fmt(z.forecast, 0)} / ${fmt(z.actual, 0)}
      <span class="e">err ${fmt(z.abs_error, 0)} · <span class="${z.in_ci ? "ok" : "bad"}">${z.in_ci ? "in range" : "outside"}</span></span></td>` : "<td>—</td>";
  const rows = d.runs.map((r) => `<tr data-ftime="${esc(r.forecast_time)}" tabindex="0">
      <td class="num">${esc(r.forecast_time.slice(11))}</td>
      ${HORIZONS_UI.map((h) => cell(r.horizons[h])).join("")}
      <td class="num ghi">${fmt(r.mae)}</td><td class="num">${fmt(r.sp_mae)}</td></tr>`).join("");
  return `<table class="day-table"><thead><tr><th>Issued</th><th>t+1h</th><th>t+2h</th><th>t+3h</th>
    <th>Model error ${tip("abs_error")}</th><th>Persistence error ${tip("sp")}</th></tr></thead><tbody>${rows}</tbody></table>`;
}
const HORIZONS_UI = ["t+1h", "t+2h", "t+3h"];

async function loadBacktestDay(day, ftime) {
  btDay = day;
  const i = btDays.indexOf(day);
  $("bt-day").value = day;
  $("bt-prev").disabled = i <= 0;
  $("bt-next").disabled = i >= btDays.length - 1;
  const r = await fetch(API.btDay(day));
  if (btDay !== day) return;                      // a newer pick overtook this one
  if (!r.ok) { $("bt-strip").innerHTML = `<span class="muted">No backtest for ${esc(day)}.</span>`; return; }
  const d = await r.json();
  const m = d.runs.filter((x) => x.mae != null);
  $("bt-day-note").textContent = m.length
    ? `${m.length} forecasts · mean error ${fmt(m.reduce((a, x) => a + x.mae, 0) / m.length)} W/m² (persistence ${fmt(m.reduce((a, x) => a + (x.sp_mae || 0), 0) / m.length)})` : "";
  $("bt-strip").innerHTML = d.hours.map((h) => `<figure data-time="${h.time}">
      <div class="t"><span>${h.time}</span></div>
      <div class="img-slot">${h.image.status === "ready" ? `<img src="${h.image.url}" alt="scan at ${h.time} SGT" loading="lazy">` : `<span class="muted">no scan</span>`}</div>
      <div class="a">${fmt(h.actual, 0)} <span class="muted">W/m²</span></div>
      <div class="role"></div></figure>`).join("");
  $("bt-day-table").innerHTML = dayTable(d);
  $("bt-hour").innerHTML = d.runs.map((x) => `<option value="${esc(x.forecast_time)}">${esc(x.forecast_time.slice(11))}</option>`).join("");
  document.querySelectorAll("#bt-day-table tbody tr").forEach((tr) => {
    const open = () => loadBacktestRun(tr.dataset.ftime);
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
  });
  const pick = d.runs.some((x) => x.forecast_time === ftime) ? ftime : d.runs[0].forecast_time;
  loadBacktestRun(pick);
}

async function loadBacktestRun(ftime) {
  btCurrent = ftime;
  $("bt-hour").value = ftime;
  markStrip(ftime);
  const r = await fetch(API.btRun(ftime));
  if (btCurrent !== ftime) return;
  if (!r.ok) { $("bt-columns").innerHTML = `<div class="card muted">No saved forecast for ${esc(ftime)}.</div>`; return; }
  const d = await r.json();
  const v3 = d.columns.map((c) => c.v3).filter((v) => v && v.abs_error != null);
  $("bt-run-note").textContent = v3.length
    ? `mean error ${fmt(v3.reduce((a, v) => a + v.abs_error, 0) / v3.length)} W/m² over ${v3.length} hours` : "";
  $("bt-issue").textContent = `issued ${ftime} SGT`;
  $("bt-inputs").innerHTML = d.inputs.map((f) => `<div><div class="k">${esc(f.label)}</div>${imgBlock(f, f.label + " satellite frame")}</div>`).join("");
  $("bt-columns").innerHTML = d.columns.map(column).join("");
}

function pickDay(day) {
  loadBacktestDay(day);
  $("bt-run-bar").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadBacktestRuns() {
  const r = await fetch(API.btRuns());
  if (!r.ok) { $("bt-run-bar").hidden = true; return; }
  btTimes = (await r.json()).times.slice().sort();
  if (!btTimes.length) return;
  btDays = [...new Set(btTimes.map((t) => t.slice(0, 10)))];
  $("bt-day").innerHTML = btDays.slice().reverse().map((d) => `<option>${d}</option>`).join("");
  const want = new URLSearchParams(location.search).get("backtest") || "";
  const day = btDays.includes(want.slice(0, 10)) ? want.slice(0, 10) : btDays[btDays.length - 1];
  loadBacktestDay(day, want);
}
$("bt-day").addEventListener("change", (e) => loadBacktestDay(e.target.value));
$("bt-hour").addEventListener("change", (e) => loadBacktestRun(e.target.value));
$("bt-prev").addEventListener("click", () => loadBacktestDay(btDays[btDays.indexOf(btDay) - 1]));
$("bt-next").addEventListener("click", () => loadBacktestDay(btDays[btDays.indexOf(btDay) + 1]));

// ── initial load ────────────────────────────────────────────────────────────
loadConfig();
loadForecast();
loadFrames(null);
loadBacktest();
loadBacktestRuns();
// /?forecast_time=2026-09-30%2010:00#verify opens a past day straight on the Verify tab
loadVerification(new URLSearchParams(location.search).get("forecast_time"));
if (["#verify", "#backtest", "#about"].includes(location.hash)) showTab(location.hash.slice(1));
if (!STATIC) poll();
