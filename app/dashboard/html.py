"""
app/dashboard/html.py — Single-file dashboard rendered as HTML.
"""


def render_dashboard() -> str:
    return DASHBOARD_HTML


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EV Scout — Sports AI Analytics</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  :root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --border: #1e2d45;
    --text: #e2e8f0;
    --muted: #64748b;
    --green: #10b981;
    --green-dim: #064e3b;
    --amber: #f59e0b;
    --amber-dim: #451a03;
    --red: #ef4444;
    --red-dim: #450a0a;
    --purple: #8b5cf6;
    --purple-dim: #2e1065;
    --blue: #3b82f6;
    --accent: #06b6d4;
    --mono: 'Space Mono', monospace;
    --sans: 'Space Grotesk', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Grid scan line texture ── */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: repeating-linear-gradient(
      0deg, transparent, transparent 40px, rgba(6,182,212,0.015) 40px, rgba(6,182,212,0.015) 41px
    );
    pointer-events: none;
    z-index: 0;
  }

  /* ── Header ── */
  header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10,14,26,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-mark {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--accent), var(--purple));
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
  }

  .logo-text {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
    color: var(--accent);
  }

  .logo-sub {
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
  }

  .live-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--green);
    background: var(--green-dim);
    padding: 4px 10px;
    border-radius: 100px;
    border: 1px solid var(--green);
  }

  .live-dot {
    width: 6px; height: 6px;
    background: var(--green);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }

  .scan-timer {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
  }

  #scan-btn {
    background: var(--accent);
    color: var(--bg);
    border: none;
    padding: 6px 16px;
    border-radius: 6px;
    font-family: var(--sans);
    font-weight: 600;
    font-size: 0.8rem;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  #scan-btn:hover { opacity: 0.85; }
  #scan-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* ── Layout ── */
  main {
    position: relative;
    z-index: 1;
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
  }

  /* ── Stat Cards ── */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
  }

  .stat-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, var(--accent));
  }

  .stat-card.green::after { --accent-color: var(--green); }
  .stat-card.amber::after { --accent-color: var(--amber); }
  .stat-card.red::after { --accent-color: var(--red); }
  .stat-card.purple::after { --accent-color: var(--purple); }

  .stat-label {
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }

  .stat-value {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
  }

  .stat-card.green .stat-value { color: var(--green); }
  .stat-card.amber .stat-value { color: var(--amber); }
  .stat-card.red .stat-value { color: var(--red); }
  .stat-card.purple .stat-value { color: var(--purple); }

  .stat-sub {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
  }

  /* ── Section header ── */
  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
  }

  .section-title {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    font-family: var(--mono);
  }

  .section-meta {
    font-size: 0.7rem;
    color: var(--muted);
    font-family: var(--mono);
  }

  /* ── Two-column layout ── */
  .two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
  }

  @media (max-width: 900px) {
    .two-col { grid-template-columns: 1fr; }
  }

  /* ── Picks table ── */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
  }

  .panel-header {
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .panel-title {
    font-size: 0.8rem;
    font-family: var(--mono);
    font-weight: 700;
    letter-spacing: 0.05em;
  }

  .panel-body {
    padding: 0;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
  }

  th {
    text-align: left;
    padding: 0.6rem 1rem;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    font-weight: 500;
  }

  td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(30,45,69,0.5);
    vertical-align: middle;
  }

  tr:last-child td { border-bottom: none; }

  tr:hover td { background: var(--surface2); }

  .ev-badge {
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    display: inline-block;
  }

  .ev-positive { background: var(--green-dim); color: var(--green); }
  .ev-low { background: var(--amber-dim); color: var(--amber); }

  .conf-bar-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .conf-bar-bg {
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    min-width: 60px;
  }

  .conf-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--blue), var(--accent));
  }

  .conf-pct {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text);
    min-width: 34px;
    text-align: right;
  }

  .risk-pill {
    font-size: 0.65rem;
    padding: 2px 6px;
    border-radius: 100px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  .risk-low { background: var(--green-dim); color: var(--green); border: 1px solid var(--green); }
  .risk-medium { background: var(--amber-dim); color: var(--amber); border: 1px solid var(--amber); }
  .risk-high { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); }

  .market-tag {
    font-size: 0.6rem;
    padding: 1px 5px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* ── Parlay cards ── */
  .parlay-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }

  .parlay-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: border-color 0.2s;
  }

  .parlay-card:hover { border-color: var(--purple); }

  .parlay-header {
    background: linear-gradient(135deg, var(--purple-dim), var(--surface));
    padding: 1rem 1.25rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .parlay-title {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--purple);
  }

  .parlay-odds {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
  }

  .parlay-legs {
    padding: 0.75rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .parlay-leg {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.77rem;
  }

  .parlay-leg-name {
    color: var(--text);
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-right: 8px;
  }

  .parlay-leg-odds {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    white-space: nowrap;
  }

  .parlay-footer {
    padding: 0.75rem 1.25rem;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 1rem;
  }

  .parlay-stat { font-size: 0.72rem; }
  .parlay-stat span { color: var(--muted); }

  /* ── Model accuracy panel ── */
  .model-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
  }

  .model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }

  .model-metric {
    text-align: center;
  }

  .model-metric-val {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent);
  }

  .model-metric-label {
    font-size: 0.65rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
  }

  /* ── Empty state ── */
  .empty {
    padding: 3rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
  }

  .empty-icon { font-size: 2rem; margin-bottom: 0.5rem; }

  /* ── Toast ── */
  #toast {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    background: var(--surface2);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 0.75rem 1.25rem;
    font-size: 0.8rem;
    z-index: 200;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.3s;
  }

  #toast.show {
    opacity: 1;
    transform: translateY(0);
  }

  /* ── Loading skeleton ── */
  .skeleton {
    background: linear-gradient(90deg, var(--surface) 25%, var(--surface2) 50%, var(--surface) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
    height: 16px;
  }
  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .tab-row {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
  }

  .tab {
    padding: 0.6rem 1.25rem;
    font-size: 0.78rem;
    font-family: var(--mono);
    cursor: pointer;
    color: var(--muted);
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    background: none;
    border-top: none;
    border-left: none;
    border-right: none;
  }

  .tab.active {
    color: var(--accent);
    border-bottom-color: var(--accent);
  }

  .tab:hover:not(.active) { color: var(--text); }

  .tab-pane { display: none; }
  .tab-pane.active { display: block; }

  .sport-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-mark">⚡</div>
    <div>
      <div class="logo-text">EV SCOUT</div>
      <div class="logo-sub">AI Sports Analytics</div>
    </div>
  </div>
  <div class="header-right">
    <span class="scan-timer">Next scan: <span id="countdown">5:00</span></span>
    <div class="live-badge"><div class="live-dot"></div> LIVE</div>
    <button id="scan-btn" onclick="triggerScan()">▶ Scan Now</button>
  </div>
</header>

<main>

  <!-- Stat cards -->
  <div class="stat-grid" id="stat-grid">
    <div class="stat-card green">
      <div class="stat-label">Today's ROI</div>
      <div class="stat-value" id="stat-roi">—</div>
      <div class="stat-sub" id="stat-roi-sub">Loading…</div>
    </div>
    <div class="stat-card amber">
      <div class="stat-label">Win Rate</div>
      <div class="stat-value" id="stat-winrate">—</div>
      <div class="stat-sub" id="stat-wr-sub">Loading…</div>
    </div>
    <div class="stat-card purple">
      <div class="stat-label">High EV Picks</div>
      <div class="stat-value" id="stat-picks">—</div>
      <div class="stat-sub">Above threshold today</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Model Accuracy</div>
      <div class="stat-value" style="color:var(--accent)" id="stat-acc">—</div>
      <div class="stat-sub" id="stat-model-name">Loading…</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Markets Scanned</div>
      <div class="stat-value" style="color:var(--text)" id="stat-markets">—</div>
      <div class="stat-sub">This session</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total P&L</div>
      <div class="stat-value" id="stat-pl" style="color:var(--text)">—</div>
      <div class="stat-sub" id="stat-pl-sub">Loading…</div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tab-row">
    <button class="tab active" onclick="switchTab('picks')">🎯 Top Picks</button>
    <button class="tab" onclick="switchTab('parlays')">🎰 Parlays</button>
    <button class="tab" onclick="switchTab('history')">📊 History</button>
    <button class="tab" onclick="switchTab('model')">🤖 Model</button>
  </div>

  <!-- PICKS TAB -->
  <div class="tab-pane active" id="tab-picks">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">🎯 TODAY'S TOP PICKS</span>
        <span class="section-meta" id="picks-meta">Loading…</span>
      </div>
      <div class="panel-body">
        <table>
          <thead>
            <tr>
              <th>Event / Selection</th>
              <th>Market</th>
              <th>Odds</th>
              <th>Confidence</th>
              <th>EV</th>
              <th>Kelly</th>
              <th>Risk</th>
            </tr>
          </thead>
          <tbody id="picks-tbody">
            <tr><td colspan="7"><div class="empty"><div class="empty-icon">⏳</div>Loading picks…</div></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- PARLAYS TAB -->
  <div class="tab-pane" id="tab-parlays">
    <div class="parlay-grid" id="parlay-grid">
      <div class="empty"><div class="empty-icon">⏳</div>Loading parlays…</div>
    </div>
  </div>

  <!-- HISTORY TAB -->
  <div class="tab-pane" id="tab-history">
    <div class="panel">
      <div class="panel-header">
        <span class="panel-title">📊 PREDICTION HISTORY</span>
        <span class="section-meta" id="history-meta"></span>
      </div>
      <div class="panel-body">
        <table>
          <thead>
            <tr>
              <th>Event</th>
              <th>Selection</th>
              <th>Sport</th>
              <th>Market</th>
              <th>EV</th>
              <th>Conf</th>
              <th>Alert</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody id="history-tbody">
            <tr><td colspan="8"><div class="empty"><div class="empty-icon">⏳</div>Loading…</div></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- MODEL TAB -->
  <div class="tab-pane" id="tab-model">
    <div class="model-panel" id="model-panel">
      <div class="section-header">
        <span class="section-title">🤖 MODEL PERFORMANCE</span>
        <span class="section-meta" id="model-meta">Loading…</span>
      </div>
      <div class="model-grid" id="model-grid">
        <div class="skeleton" style="height:80px"></div>
        <div class="skeleton" style="height:80px"></div>
        <div class="skeleton" style="height:80px"></div>
        <div class="skeleton" style="height:80px"></div>
      </div>
    </div>
  </div>

</main>

<div id="toast"></div>

<script>
const API = '';
let scanInterval = 300;
let countdown = scanInterval;
let sessionMarkets = 0;

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
  if (name === 'history') loadHistory();
  if (name === 'model') loadModel();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(msg, color = 'var(--accent)') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.borderColor = color;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3500);
}

// ── Formatting helpers ────────────────────────────────────────────────────────
function fmtEV(ev) {
  const sign = ev >= 0 ? '+' : '';
  return sign + (ev * 100).toFixed(1) + '%';
}

function evBadge(ev) {
  const cls = ev >= 0.05 ? 'ev-positive' : 'ev-low';
  return `<span class="ev-badge ${cls}">${fmtEV(ev)}</span>`;
}

function riskPill(risk) {
  return `<span class="risk-pill risk-${risk}">${risk}</span>`;
}

function confBar(conf) {
  const pct = Math.round(conf * 100);
  const w = pct;
  return `<div class="conf-bar-wrap">
    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${w}%"></div></div>
    <span class="conf-pct">${pct}%</span>
  </div>`;
}

function sportColor(sport) {
  const map = {soccer:'#10b981', basketball:'#f59e0b', american_football:'#ef4444',
    baseball:'#3b82f6', hockey:'#8b5cf6', tennis:'#06b6d4', prediction:'#ec4899'};
  return map[sport] || '#64748b';
}

function relTime(iso) {
  if (!iso) return '—';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return mins + 'm ago';
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return hrs + 'h ago';
  return Math.floor(hrs / 24) + 'd ago';
}

// ── Load picks ────────────────────────────────────────────────────────────────
async function loadPicks() {
  try {
    const res = await fetch(API + '/api/picks/today?limit=30&min_ev=0');
    const data = await res.json();
    const tbody = document.getElementById('picks-tbody');
    document.getElementById('picks-meta').textContent = data.length + ' picks found';

    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="7"><div class="empty"><div class="empty-icon">🔍</div>No picks yet — trigger a scan to start.</div></td></tr>';
      return;
    }

    document.getElementById('stat-picks').textContent = data.filter(p => p.expected_value >= 0.05).length;

    tbody.innerHTML = data.map(p => `
      <tr>
        <td>
          <div style="font-weight:600;font-size:0.82rem;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.event}</div>
          <div style="color:var(--accent);font-size:0.75rem;margin-top:2px">${p.selection}</div>
        </td>
        <td><span class="market-tag">${(p.market_type||'').replace(/_/g,' ')}</span></td>
        <td>
          <span style="font-family:var(--mono);font-size:0.82rem">${p.decimal_odds?.toFixed(2) || '—'}</span>
          <div style="color:var(--muted);font-size:0.68rem">${p.american_odds > 0 ? '+' : ''}${p.american_odds || ''}</div>
        </td>
        <td>${confBar(p.confidence)}</td>
        <td>${evBadge(p.expected_value)}</td>
        <td><span style="font-family:var(--mono);font-size:0.75rem;color:var(--amber)">${(p.kelly_fraction * 100).toFixed(1)}%</span></td>
        <td>${riskPill(p.risk_level || 'medium')}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.error('picks error', e);
  }
}

// ── Load parlays ──────────────────────────────────────────────────────────────
async function loadParlays() {
  try {
    const res = await fetch(API + '/api/parlays/today?limit=12');
    const data = await res.json();
    const grid = document.getElementById('parlay-grid');

    if (!data.length) {
      grid.innerHTML = '<div class="empty" style="grid-column:1/-1"><div class="empty-icon">🎰</div>No parlays constructed yet.</div>';
      return;
    }

    grid.innerHTML = data.map(p => `
      <div class="parlay-card">
        <div class="parlay-header">
          <span class="parlay-title">🎰 ${p.num_legs}-LEG PARLAY</span>
          <span class="parlay-odds">${p.combined_odds?.toFixed(2)}x</span>
        </div>
        <div class="parlay-legs">
          ${(p.legs || []).map(leg => `
            <div class="parlay-leg">
              <span class="parlay-leg-name">${leg.selection} <span style="color:var(--muted)">(${leg.event})</span></span>
              <span class="parlay-leg-odds">${leg.decimal_odds?.toFixed(2)}</span>
            </div>
          `).join('')}
        </div>
        <div class="parlay-footer">
          <div class="parlay-stat"><span>EV </span>${fmtEV(p.combined_ev)}</div>
          <div class="parlay-stat"><span>Conf </span>${Math.round(p.combined_confidence * 100)}%</div>
          <div class="parlay-stat">${riskPill(p.risk_level || 'medium')}</div>
        </div>
      </div>
    `).join('');
  } catch(e) {
    console.error('parlays error', e);
  }
}

// ── Load history ──────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const res = await fetch(API + '/api/predictions?page=1&page_size=50');
    const data = await res.json();
    const tbody = document.getElementById('history-tbody');
    document.getElementById('history-meta').textContent = data.length + ' records';

    if (!data.length) {
      tbody.innerHTML = '<tr><td colspan="8"><div class="empty">No history yet.</div></td></tr>';
      return;
    }

    tbody.innerHTML = data.map(p => `
      <tr>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:0.78rem">${p.event}</td>
        <td style="font-size:0.78rem;color:var(--accent)">${p.selection}</td>
        <td><span class="sport-dot" style="background:${sportColor(p.sport)}"></span>${p.sport}</td>
        <td><span class="market-tag">${(p.market_type||'').replace(/_/g,' ')}</span></td>
        <td>${evBadge(p.expected_value)}</td>
        <td>${Math.round(p.confidence * 100)}%</td>
        <td>${p.alert_sent ? '✅' : '—'}</td>
        <td style="color:var(--muted);font-size:0.72rem">${relTime(p.created_at)}</td>
      </tr>
    `).join('');
  } catch(e) {
    console.error('history error', e);
  }
}

// ── Load model stats ──────────────────────────────────────────────────────────
async function loadModel() {
  try {
    const res = await fetch(API + '/api/stats/model');
    const data = await res.json();
    const grid = document.getElementById('model-grid');
    document.getElementById('model-meta').textContent = data.model ? `${data.model} — trained ${relTime(data.trained_at)}` : '';

    if (data.message) {
      grid.innerHTML = `<div class="empty" style="grid-column:1/-1">${data.message}</div>`;
      return;
    }

    const metrics = [
      { label: 'Accuracy', val: data.accuracy ? (data.accuracy * 100).toFixed(1) + '%' : '—' },
      { label: 'ROC-AUC', val: data.roc_auc?.toFixed(4) || '—' },
      { label: 'Brier Score', val: data.brier_score?.toFixed(4) || '—' },
      { label: 'Log Loss', val: data.log_loss?.toFixed(4) || '—' },
      { label: 'Train Samples', val: data.train_samples?.toLocaleString() || '—' },
      { label: 'Version', val: data.version || '—' },
    ];

    grid.innerHTML = metrics.map(m => `
      <div class="model-metric">
        <div class="model-metric-val">${m.val}</div>
        <div class="model-metric-label">${m.label}</div>
      </div>
    `).join('');

    // Update header stat
    if (data.accuracy) {
      document.getElementById('stat-acc').textContent = (data.accuracy * 100).toFixed(1) + '%';
      document.getElementById('stat-model-name').textContent = data.model || 'ensemble';
    }
  } catch(e) {
    console.error('model error', e);
  }
}

// ── Load ROI stats ────────────────────────────────────────────────────────────
async function loadROI() {
  try {
    const res = await fetch(API + '/api/stats/roi');
    const data = await res.json();
    const roiPct = (data.roi * 100).toFixed(1) + '%';
    document.getElementById('stat-roi').textContent = roiPct;
    document.getElementById('stat-roi').style.color = data.roi >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('stat-roi-sub').textContent = `${data.wins}W / ${data.losses}L`;
    document.getElementById('stat-winrate').textContent = (data.win_rate * 100).toFixed(1) + '%';
    document.getElementById('stat-wr-sub').textContent = `${data.total} resolved bets`;
    const pl = data.total_pl;
    document.getElementById('stat-pl').textContent = (pl >= 0 ? '+' : '') + pl.toFixed(2);
    document.getElementById('stat-pl').style.color = pl >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('stat-pl-sub').textContent = 'Total units P&L';
  } catch(e) {
    console.error('ROI error', e);
  }
}

// ── Scan trigger ──────────────────────────────────────────────────────────────
async function triggerScan() {
  const btn = document.getElementById('scan-btn');
  btn.disabled = true;
  btn.textContent = '⏳ Scanning…';
  showToast('🔍 Scan started…', 'var(--accent)');
  try {
    const res = await fetch(API + '/api/scan', { method: 'POST' });
    const data = await res.json();
    sessionMarkets += data.markets || 0;
    document.getElementById('stat-markets').textContent = sessionMarkets.toLocaleString();
    showToast(`✅ Scan done: ${data.markets} markets, ${data.high_ev_picks} high-EV picks`, 'var(--green)');
    await refreshAll();
    countdown = scanInterval;
  } catch(e) {
    showToast('❌ Scan failed: ' + e.message, 'var(--red)');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Scan Now';
  }
}

// ── Countdown timer ───────────────────────────────────────────────────────────
function startCountdown() {
  setInterval(() => {
    countdown--;
    if (countdown <= 0) {
      countdown = scanInterval;
      refreshAll();
    }
    const m = Math.floor(countdown / 60);
    const s = countdown % 60;
    document.getElementById('countdown').textContent = m + ':' + String(s).padStart(2, '0');
  }, 1000);
}

// ── Full refresh ──────────────────────────────────────────────────────────────
async function refreshAll() {
  await Promise.all([loadPicks(), loadParlays(), loadROI(), loadModel()]);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  refreshAll();
  startCountdown();
  // Auto-refresh every 60s
  setInterval(refreshAll, 60000);
});
</script>
</body>
</html>"""
