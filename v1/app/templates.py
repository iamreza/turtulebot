# Course: Autonomous Robotics
# Supervisor: Prof. Dr.-Ing. Reinhard Gerndt
# Semester: Sommer Semester
# Group: 7
#
# Team:
# - Reza Babaee, 70498082
# - Hamid Safisamghabadi, 70497663
# - Emad Mohammadi, 70494663
# - Azarjan Gharibian

# ============================================================
# HTML
# ============================================================

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TurtleBot Card Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500;600&family=Geist:wght@400;500;600;700&display=swap" rel="stylesheet">

<!--
  ============================================================
  TurtleBot Card Detector  ·  drop-in Flask template
  ============================================================
  Element IDs your backend / JS should populate:
    #status              text   — current state line
    #base-health-state   text   — "ALIVE" / "DEAD"
    #base-health         text   — meta line (odom age, count, subs)
    #battery-level       text   — "82%"  (sets the bar fill via data-pct on #battery)
    #battery             elem   — set data-pct="0..100" to drive the gauge
    #battery-state       text   — "charging" | "discharging" | "low"
    #cpu-level           text   — "34%" (drive bar via #cpu-bar style --pct: 34)
    #ram-level           text   — "58%" (drive bar via #ram-bar style --pct: 58)
    #result              text   — rotated card report
    #last-report         text   — last report line
    #missing-references  group  — 5 .num-chip children (1..5)
    #missing-checks      group  — 5 .num-chip children (1..5)
    #card-1-ref ... #card-5-ref     — inner <img src="..."> when captured
    #card-1-check ... #card-5-check — inner <img src="..."> when captured
    #card-1-status .. #card-5-status — toggle class: pending | verified | flagged
  Slot frame colors:
    .card-slot                — no scan yet (neutral)
    .card-slot.verified       — scanned, NOT rotated  (green frame)
    .card-slot.flagged        — scanned, rotated card (red frame)
    #camera-feed         img    — set src to your MJPEG feed
    #mask-feed           img    — set src to your debug mask feed
    #btn-start #btn-stop #btn-clear  — your existing handlers
  Helpers used by demo (you can keep or remove):
    .num-chip.done           — mark a 1..5 chip as captured
    .card-slot.flagged       — highlight the rotated card slot
  Turn off the demo: set window.DEMO = false  (or delete the demo <script>).
-->

<style>
:root {
  --bg: #0a0b0d;
  --panel: #131519;
  --panel-2: #181b20;
  --panel-3: #1d2026;
  --border: #25282e;
  --border-strong: #353941;
  --text: #e7e9ec;
  --muted: #8a9098;
  --dim: #5a606a;
  --accent: #6ee7b7;
  --accent-soft: #8af0c8;
  --accent-dim: #3a7d5e;
  --accent-bg: rgba(110, 231, 183, 0.08);
  --danger: #ef5a5a;
  --danger-bg: rgba(239, 90, 90, 0.10);
  --warning: #e6c372;
  --battery-low: #ef5a5a;
  --battery-mid: #e6c372;
  --battery-ok: #6ee7b7;
}
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  background:
    radial-gradient(1200px 700px at 50% -100px, rgba(110,231,183,0.04), transparent 60%),
    var(--bg);
  color: var(--text);
  font-family: 'Geist', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.5;
  letter-spacing: -0.005em;
  -webkit-font-smoothing: antialiased;
  min-height: 100vh;
}
.mono, code { font-family: 'Geist Mono', ui-monospace, monospace; }

/* ---------------- Layout ---------------- */
.shell {
  max-width: 1480px;
  margin: 0 auto;
  padding: 28px 32px 64px;
}

/* ---------------- Header ---------------- */
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 14px; }
.brand-mark {
  width: 40px; height: 40px;
  position: relative; flex-shrink: 0;
}
.brand-mark span {
  position: absolute; width: 22px; height: 30px;
  border: 1.5px solid var(--accent);
  border-radius: 4px;
  top: 5px; left: 9px;
  background: rgba(110,231,183,0.04);
}
.brand-mark span:nth-child(1) { transform: rotate(-10deg) translateX(-3px); opacity: .5; }
.brand-mark span:nth-child(2) { transform: rotate(10deg) translateX(3px); background: var(--bg); }
.brand-name {
  font-weight: 600; font-size: 16px; letter-spacing: -0.005em;
  line-height: 1.2;
}
.brand-sub {
  color: var(--muted); font-size: 11px;
  font-family: 'Geist Mono', monospace;
  letter-spacing: 0.10em;
  margin-top: 3px;
}
.header-right {
  display: flex; align-items: center; gap: 12px;
  font-family: 'Geist Mono', monospace;
}
.crumb {
  color: var(--dim);
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding-right: 12px;
  border-right: 1px solid var(--border);
}
.health-pill {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 8px 14px; border-radius: 100px;
  background: var(--accent-bg);
  border: 1px solid var(--accent-dim);
  font-size: 12px;
}
.health-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 0 rgba(110,231,183,0.6);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(110,231,183,0.6); }
  100% { box-shadow: 0 0 0 8px rgba(110,231,183,0); }
}
.health-label { color: var(--accent); font-weight: 600; }
.health-meta { color: var(--muted); }

/* ---------------- System metrics (CPU / RAM) ---------------- */
.metric-pill {
  display: inline-flex; align-items: center; gap: 10px;
  padding: 7px 12px;
  border-radius: 100px;
  background: var(--panel);
  border: 1px solid var(--border-strong);
  font-family: 'Geist Mono', monospace;
  font-size: 12px;
}
.metric-label {
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dim);
}
.metric-bar {
  position: relative;
  width: 46px; height: 4px;
  border-radius: 100px;
  background: var(--panel-3);
  overflow: hidden;
}
.metric-bar-fill {
  position: absolute; left: 0; top: 0; bottom: 0;
  width: calc(var(--pct, 0) * 1%);
  background: var(--accent);
  border-radius: 100px;
  transition: width .4s ease, background .3s;
}
.metric-bar-fill[style*="--pct: 7"],
.metric-bar-fill[style*="--pct: 8"],
.metric-bar-fill[style*="--pct: 9"],
.metric-bar-fill.warn { background: var(--warning); }
.metric-bar-fill.crit { background: var(--danger); }
.metric-value {
  color: var(--text); font-weight: 600;
  min-width: 36px; text-align: right;
}

/* ---------------- Battery (shares .metric-pill base) ---------------- */
#battery .battery-fill { background: var(--battery-ok); }
#battery[data-state="mid"] .battery-fill { background: var(--battery-mid); }
#battery[data-state="low"] .battery-fill {
  background: var(--battery-low);
  animation: low-blink 1.2s infinite;
}
#battery[data-state="low"] .metric-value { color: var(--battery-low); }
#battery[data-state="charging"] .battery-fill {
  background: linear-gradient(90deg, var(--battery-ok), var(--accent-soft));
  background-size: 200% 100%;
  animation: charging-flow 1.8s linear infinite;
}
@keyframes low-blink { 50% { opacity: 0.4; } }
@keyframes charging-flow { 0% { background-position: 100% 0; } 100% { background-position: -100% 0; } }

/* ---------------- Description ---------------- */
.desc {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 18px;
  margin-bottom: 22px;
  font-size: 13px;
  color: var(--muted);
  line-height: 1.7;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 0 16px;
}
.desc-label {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dim);
  padding-top: 3px;
}
.desc strong { color: var(--text); font-weight: 600; }
.desc .rule {
  grid-column: 2;
  color: var(--dim);
  font-size: 12px;
  margin-top: 4px;
  font-family: 'Geist Mono', monospace;
  letter-spacing: 0.01em;
}

/* ---------------- Command grid ---------------- */
.command {
  display: grid;
  grid-template-columns: 1.5fr 1fr 1fr;
  gap: 14px;
  margin-bottom: 28px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
}
.panel-head {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 16px;
}
.panel-title {
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
}
.panel-num {
  font-family: 'Geist Mono', monospace;
  color: var(--dim); font-size: 11px;
  letter-spacing: 0.08em;
}

/* ---------------- Buttons ---------------- */
.actions { display: flex; gap: 10px; align-items: stretch; }
button { font-family: inherit; cursor: pointer; border: none; padding: 0; }
.btn {
  display: inline-flex; align-items: center; justify-content: center; gap: 8px;
  padding: 12px 18px; font-size: 14px; font-weight: 500;
  border-radius: 8px;
  letter-spacing: 0.01em;
  transition: background .15s ease, transform .15s ease, box-shadow .15s ease;
}
.btn-primary {
  background: var(--accent);
  color: #0a1410;
  flex: 1.4;
  font-weight: 600;
  font-size: 14px;
  padding: 13px 18px;
  box-shadow: 0 1px 0 0 rgba(255,255,255,0.15) inset, 0 8px 24px -10px rgba(110,231,183,0.5);
}
.btn-primary:hover { background: var(--accent-soft); transform: translateY(-1px); }
.btn-primary:active { transform: translateY(0); }
.btn-secondary {
  background: var(--panel-3);
  color: var(--text);
  border: 1px solid var(--border-strong);
  flex: 1;
}
.btn-secondary:hover { background: #252a31; }
.btn-danger {
  background: var(--danger-bg);
  border: 1px solid rgba(239,90,90,0.35);
  color: var(--danger);
  flex: 1;
}
.btn-danger:hover { background: rgba(239,90,90,0.18); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }
.btn-icon { width: 13px; height: 13px; }

/* ---------------- Run meta tiles ---------------- */
.run-meta {
  margin-top: auto;
  padding-top: 16px;
  border-top: 1px dashed var(--border);
  display: grid;
  grid-template-columns: 1.2fr 1fr 1fr;
  gap: 14px;
}
.meta-tile { display: grid; gap: 5px; }
.meta-tile-label {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dim);
}
.meta-tile-value {
  font-family: 'Geist Mono', monospace;
  font-size: 15px;
  font-weight: 500;
  color: var(--text);
  display: inline-flex; align-items: center; gap: 8px;
  font-variant-numeric: tabular-nums;
}
.run-state-running { color: var(--accent); }
.run-state-complete { color: var(--accent); }
.run-state-stopped { color: var(--warning); }
.workflow-indicator {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--dim);
  flex-shrink: 0;
  transition: background .2s;
}
.workflow-indicator.running {
  background: var(--accent);
  animation: pulse 1.2s infinite;
}

/* ---------------- Fields ---------------- */
.field { margin-bottom: 14px; }
.field:last-child { margin-bottom: 0; }
.field-label {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 8px;
}
.field-value {
  font-size: 13px;
  font-family: 'Geist Mono', monospace;
  color: var(--text);
  word-break: break-word;
}
.field-value.empty { color: var(--dim); }
.field-value.accent { color: var(--accent); }

.progress-numbers { display: flex; gap: 6px; flex-wrap: wrap; }
.num-chip {
  width: 28px; height: 28px;
  display: grid; place-items: center;
  border-radius: 6px;
  background: var(--panel-3);
  border: 1px solid var(--border);
  font-family: 'Geist Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  transition: all .25s ease;
}
.num-chip.done {
  background: var(--accent-bg);
  border-color: var(--accent-dim);
  color: var(--accent);
}

/* ---------------- Phase timeline (horizontal) ---------------- */
.phase-list {
  list-style: none;
  margin: 4px 0 16px; padding: 0;
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0;
  position: relative;
}
.phase-list::before {
  content: '';
  position: absolute;
  left: 10%; right: 10%;
  top: 5px;
  height: 1px;
  background: var(--border);
}
.phase {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  position: relative;
  z-index: 1;
}
.phase-dot {
  width: 11px; height: 11px;
  border-radius: 50%;
  background: var(--panel-3);
  border: 1.5px solid var(--border-strong);
  transition: all .25s ease;
}
.phase-text {
  font-size: 10.5px;
  color: var(--muted);
  letter-spacing: 0;
  line-height: 1.25;
  transition: color .25s;
  text-align: center;
  padding: 0 4px;
  max-width: 100%;
  overflow-wrap: break-word;
}
.phase.done .phase-dot {
  background: var(--accent);
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-bg);
}
.phase.done .phase-text { color: var(--text); }
.phase.active .phase-dot {
  background: var(--accent);
  border-color: var(--accent);
  box-shadow: 0 0 0 4px var(--accent-bg);
  animation: phase-pulse 1.5s ease-in-out infinite;
}
.phase.active .phase-text { color: var(--text); font-weight: 500; }
@keyframes phase-pulse {
  0%, 100% { box-shadow: 0 0 0 4px var(--accent-bg); }
  50% { box-shadow: 0 0 0 7px rgba(110,231,183,0.04); }
}
.phase-result {
  margin-top: auto;
  padding-top: 14px;
  border-top: 1px dashed var(--border);
  display: grid;
  gap: 6px;
}
.phase-result-label {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--dim);
}
.phase-result-value {
  font-size: 14px;
  font-weight: 500;
  color: var(--accent);
  letter-spacing: -0.005em;
}
.phase-result-value.empty { color: var(--dim); font-weight: 400; }

/* ---------------- Sections ---------------- */
.section-head {
  display: flex; align-items: center;
  margin: 36px 0 14px;
  gap: 14px;
}
.section-title {
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  white-space: nowrap;
}
.section-line { height: 1px; flex: 1; background: var(--border); }
.section-meta {
  color: var(--dim); font-size: 11px;
  font-family: 'Geist Mono', monospace;
  letter-spacing: 0.06em;
  white-space: nowrap;
}

/* ---------------- Card matrix ---------------- */
.card-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 14px;
}
.card-slot {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  transition: border-color .3s ease, box-shadow .3s ease, transform .3s ease;
}
.card-slot.verified {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px var(--accent-bg), 0 10px 24px -14px rgba(110,231,183,0.4);
}
.card-slot.flagged {
  border-color: var(--danger);
  box-shadow: 0 0 0 3px var(--danger-bg), 0 12px 30px -12px rgba(239,90,90,0.45);
}
.card-slot-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid var(--border);
}
.card-slot-name { font-weight: 600; font-size: 13px; }
.card-slot-idx {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.1em;
  color: var(--dim);
  padding: 3px 7px;
  background: var(--panel-3);
  border-radius: 4px;
  border: 1px solid var(--border);
}
.card-pair { padding: 12px; display: grid; gap: 10px; }
.card-image-wrap { position: relative; }
.card-image-label {
  position: absolute; top: 7px; left: 7px; z-index: 2;
  font-family: 'Geist Mono', monospace;
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  background: rgba(10,12,14,0.78);
  padding: 3px 7px; border-radius: 4px;
  backdrop-filter: blur(4px);
}
.card-image {
  aspect-ratio: 1/1;
  background: var(--panel-2);
  border: 1px dashed var(--border-strong);
  border-radius: 8px;
  display: grid; place-items: center;
  color: var(--dim);
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  position: relative;
  overflow: hidden;
  transition: all .35s ease;
}
.card-image.filled {
  border-style: solid;
  border-color: var(--border-strong);
  background: var(--panel-2);
  animation: slot-fill .5s ease;
}
@keyframes slot-fill {
  0% { transform: scale(0.96); opacity: 0; }
  100% { transform: scale(1); opacity: 1; }
}
.card-image img {
  width: 100%; height: 100%; object-fit: cover; display: block;
}
.card-image .placeholder-fill {
  position: absolute; inset: 0;
  display: grid; place-items: center;
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  color: rgba(255,255,255,0.55);
}
.card-status {
  padding: 10px 14px;
  border-top: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--dim); transition: background .2s; }
.card-status.pending { color: var(--muted); }
.card-status.pending .status-dot { background: var(--dim); }
.card-status.captured { color: var(--accent); }
.card-status.captured .status-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.card-status.verified { color: var(--accent); }
.card-status.verified .status-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
.card-status.flagged { color: var(--danger); }
.card-status.flagged .status-dot { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

/* ---------------- Vision pane ---------------- */
.vision { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.vision-tile {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
}
.vision-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
}
.vision-name {
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text);
  font-weight: 500;
  display: flex; align-items: center; gap: 9px;
}
.live-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--danger);
  animation: live-pulse 1.5s infinite;
}
@keyframes live-pulse {
  0% { box-shadow: 0 0 0 0 rgba(239,90,90,0.6); }
  100% { box-shadow: 0 0 0 8px rgba(239,90,90,0); }
}
.vision-body {
  aspect-ratio: var(--feed-aspect, 4/3);
  background: var(--panel-2);
  display: grid; place-items: center;
  position: relative;
  overflow: hidden;
}
.vision-body img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  background: #050608;
}
.vision-placeholder {
  color: var(--dim);
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-align: center;
}

/* ---------------- Team strip ---------------- */
.team {
  margin-bottom: 22px;
  padding: 20px 24px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  display: grid;
  grid-template-columns: auto 1fr;
  align-items: center;
  gap: 28px;
}
.team-meta {
  border-right: 1px solid var(--border);
  padding-right: 28px;
  display: grid;
  gap: 12px;
}
.meta-row {
  display: grid;
  grid-template-columns: 90px 1fr;
  align-items: baseline;
  gap: 12px;
}
.meta-label {
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--dim);
}
.meta-value {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  letter-spacing: -0.005em;
}
.team-list {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 18px;
}
.member { display: flex; align-items: center; gap: 12px; }
.avatar {
  width: 40px; height: 40px;
  border-radius: 50%;
  display: grid; place-items: center;
  flex-shrink: 0;
  font-family: 'Geist Mono', monospace;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.04em;
  color: var(--text);
  position: relative;
  border: 1px solid var(--border-strong);
  overflow: hidden;
  background: var(--panel-3);
}
.avatar img {
  position: absolute; inset: 0;
  width: 100%; height: 100%;
  object-fit: cover;
}
.avatar::after {
  content: '';
  position: absolute; inset: -3px;
  border-radius: 50%;
  border: 1px solid transparent;
  pointer-events: none;
}
.member-name {
  font-size: 13px;
  font-weight: 500;
  line-height: 1.25;
  letter-spacing: -0.005em;
  white-space: nowrap;
}
.member-id {
  font-family: 'Geist Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--dim);
  margin-top: 3px;
}
@media (max-width: 1100px) {
  .team { grid-template-columns: 1fr; gap: 18px; }
  .team-meta { border-right: none; border-bottom: 1px solid var(--border); padding: 0 0 16px 0; }
  .team-list { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 640px) {
  .team-list { grid-template-columns: 1fr; }
}

.foot {
  margin-top: 28px;
  font-family: 'Geist Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--dim);
  display: flex; justify-content: space-between;
  padding-top: 18px;
  border-top: 1px solid var(--border);
}

@media (max-width: 1100px) {
  .command { grid-template-columns: 1fr; }
  .card-grid { grid-template-columns: repeat(3, 1fr); }
  .vision { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .shell { padding: 18px; }
  .card-grid { grid-template-columns: repeat(2, 1fr); }
  .actions { flex-direction: column; }
}
</style>
</head>
<body>
<div class="shell">

  <!-- HEADER -->
  <header class="header">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true">
        <span></span><span></span>
      </div>
      <div>
        <div class="brand-name">TurtleBot Card Detector</div>
        <div class="brand-sub">AUTO ONE-SCAN · v8</div>
      </div>
    </div>
    <div class="header-right">
      <span class="crumb">Robot Console</span>
      <div class="metric-pill" title="CPU usage">
        <span class="metric-label">CPU</span>
        <span class="metric-bar"><span class="metric-bar-fill" id="cpu-bar" style="--pct: 34;"></span></span>
        <span class="metric-value" id="cpu-level">34%</span>
      </div>
      <div class="metric-pill" title="RAM usage">
        <span class="metric-label">RAM</span>
        <span class="metric-bar"><span class="metric-bar-fill" id="ram-bar" style="--pct: 58;"></span></span>
        <span class="metric-value" id="ram-level">58%</span>
      </div>
      <div class="metric-pill" id="battery" data-state="ok" title="Battery level">
        <span class="metric-label">BAT</span>
        <span class="metric-bar"><span class="metric-bar-fill battery-fill" id="battery-bar" style="--pct: 82;"></span></span>
        <span class="metric-value" id="battery-level">82%</span>
      </div>
      <div class="health-pill" id="health-pill" title="Base health">
        <span class="health-dot"></span>
        <span class="health-label" id="base-health-state">ALIVE</span>
        <span class="health-meta" id="base-health">odom 0.05s · 114 · /cmd_vel 1</span>
      </div>
    </div>
  </header>

  <!-- TEAM -->
  <div class="team">
    <div class="team-meta">
      <div class="meta-row">
        <div class="meta-label">Course</div>
        <div class="meta-value">Autonomous Robotics</div>
      </div>
      <div class="meta-row">
        <div class="meta-label">Supervisor</div>
        <div class="meta-value">Prof. Dr.-Ing. Reinhard Gerndt</div>
      </div>
      <div class="meta-row">
        <div class="meta-label">Group</div>
        <div class="meta-value">Group 7</div>
      </div>
    </div>
    <div class="team-list">
      <div class="member">
        <div class="avatar" style="background: linear-gradient(135deg, oklch(0.42 0.10 165), oklch(0.30 0.06 165));">
          <!-- replace with: <img src="/static/team/reza.jpg" alt=""> -->
          <span>RB</span>
        </div>
        <div>
          <div class="member-name">Reza Babaee</div>
          <div class="member-id">70498082</div>
        </div>
      </div>
      <div class="member">
        <div class="avatar" style="background: linear-gradient(135deg, oklch(0.42 0.10 240), oklch(0.30 0.06 240));">
          <span>HS</span>
        </div>
        <div>
          <div class="member-name">Hamid Safisamghabadi</div>
          <div class="member-id">70497663</div>
        </div>
      </div>
      <div class="member">
        <div class="avatar" style="background: linear-gradient(135deg, oklch(0.42 0.10 60), oklch(0.30 0.06 60));">
          <span>EM</span>
        </div>
        <div>
          <div class="member-name">Emad Mohammadi</div>
          <div class="member-id">70494663</div>
        </div>
      </div>
      <div class="member">
        <div class="avatar" style="background: linear-gradient(135deg, oklch(0.42 0.10 320), oklch(0.30 0.06 320));">
          <span>AG</span>
        </div>
        <div>
          <div class="member-name">Azarjan Gharibian</div>
          <div class="member-id">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- DESCRIPTION -->
  <div class="desc">
    <span class="desc-label">Routine</span>
    <span><strong>Start</strong> calibrates a full five-card frame, scans baseline, turns away 180°, waits 15&nbsp;s for the card flip, returns to the original facing yaw, re-aligns the five-card frame, scans check, and reports the rotated card.</span>
    <span class="rule">v8 — look-away is relaxed; return is vision-driven. Yaw only gets the robot close; sweep / quality decides the final frame.</span>
  </div>

  <!-- COMMAND PANELS -->
  <section class="command">

    <!-- CONTROL -->
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Control</span>
        <span class="panel-num">01</span>
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-start" type="button">
          <svg class="btn-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 3L13 8L4 13V3Z" fill="currentColor"/></svg>
          Start
        </button>
        <button class="btn btn-danger" id="btn-stop" type="button">Stop + Report</button>
        <button class="btn btn-secondary" id="btn-clear" type="button">Clear</button>
      </div>
      <div class="run-meta">
        <div class="meta-tile">
          <span class="meta-tile-label">State</span>
          <span class="meta-tile-value" id="run-state">
            <span class="workflow-indicator" id="workflow-indicator"></span>
            <span id="run-state-text">Idle</span>
          </span>
        </div>
        <div class="meta-tile">
          <span class="meta-tile-label">Elapsed</span>
          <span class="meta-tile-value" id="run-elapsed">00:00</span>
        </div>
        <div class="meta-tile">
          <span class="meta-tile-label">Runs</span>
          <span class="meta-tile-value" id="run-count">0</span>
        </div>
      </div>
      <!-- legacy hidden field kept so backend setters do not error -->
      <span id="status" style="display:none;">READY — Click Start to run the magic trick</span>
    </div>

    <!-- PROGRESS -->
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Progress</span>
        <span class="panel-num">02</span>
      </div>
      <div class="field">
        <div class="field-label">Missing References</div>
        <div class="progress-numbers" id="missing-references">
          <span class="num-chip">1</span><span class="num-chip">2</span><span class="num-chip">3</span><span class="num-chip">4</span><span class="num-chip">5</span>
        </div>
      </div>
      <div class="field">
        <div class="field-label">Missing Checks</div>
        <div class="progress-numbers" id="missing-checks">
          <span class="num-chip">1</span><span class="num-chip">2</span><span class="num-chip">3</span><span class="num-chip">4</span><span class="num-chip">5</span>
        </div>
      </div>
    </div>

    <!-- PHASES -->
    <div class="panel">
      <div class="panel-head">
        <span class="panel-title">Phase</span>
        <span class="panel-num">03</span>
      </div>
      <ol class="phase-list" id="phase-list">
        <li class="phase" data-phase="calibrate">
          <span class="phase-dot"></span>
          <span class="phase-text">Calibrate</span>
        </li>
        <li class="phase" data-phase="baseline">
          <span class="phase-dot"></span>
          <span class="phase-text">Capture</span>
        </li>
        <li class="phase" data-phase="away">
          <span class="phase-dot"></span>
          <span class="phase-text">Look away</span>
        </li>
        <li class="phase" data-phase="wait">
          <span class="phase-dot"></span>
          <span class="phase-text">Wait</span>
        </li>
        <li class="phase" data-phase="return">
          <span class="phase-dot"></span>
          <span class="phase-text">Return</span>
        </li>
        <li class="phase" data-phase="check">
          <span class="phase-dot"></span>
          <span class="phase-text">Check</span>
        </li>
      </ol>
      <div class="phase-result" id="phase-result">
        <span class="phase-result-label">Result</span>
        <span class="phase-result-value empty" id="result">No rotated card detected yet.</span>
      </div>
      <span id="last-report" style="display:none;">—</span>
    </div>

  </section>

  <!-- CARD MATRIX -->
  <!-- VISION -->
  <div class="section-head">
    <span class="section-title">Vision Feeds</span>
    <div class="section-line"></div>
    <span class="section-meta">RGB + binary mask</span>
  </div>

  <div class="vision">
    <div class="vision-tile">
      <div class="vision-head">
        <span class="vision-name"><span class="live-dot"></span>Camera Feed</span>
        <span class="section-meta">/video_feed</span>
      </div>
      <div class="vision-body">
        <img id="camera-feed" alt="" style="display:none;">
        <div class="vision-placeholder" id="camera-placeholder">CAMERA FEED · awaiting stream</div>
      </div>
    </div>
    <div class="vision-tile">
      <div class="vision-head">
        <span class="vision-name"><span class="live-dot"></span>Card Mask Debug</span>
        <span class="section-meta">/mask_feed</span>
      </div>
      <div class="vision-body">
        <img id="mask-feed" alt="" style="display:none;">
        <div class="vision-placeholder" id="mask-placeholder">MASK DEBUG · awaiting stream</div>
      </div>
    </div>
  </div>

  <!-- CARD MATRIX -->
  <div class="section-head">
    <span class="section-title">Slot Matrix</span>
    <span class="section-meta">/ each scanned card lands in its own slot</span>
    <div class="section-line"></div>
    <span class="section-meta" id="slot-count">0 / 5 captured</span>
  </div>

  <div class="card-grid" id="card-grid"><!-- generated --></div>

  <!-- TEAM (moved to top) -->

  <div class="foot">
    <span>TurtleBot · ROS · Flask</span>
    <span>Auto One-Scan v8</span>
  </div>

</div>

<!-- ============================================================ -->
<!-- Card-slot generator (static markup; keep this in production) -->
<!-- ============================================================ -->
<script>
(function () {
  const grid = document.getElementById('card-grid');
  // Seed with one of each state so all three frame colors are visible at load.
  // States: 'pending' (neutral), 'verified' (green), 'flagged' (red rotated)
  const SAMPLE = [
    { state: 'pending' },
    { state: 'pending' },
    { state: 'pending' },
    { state: 'pending' },
    { state: 'pending' }
  ];
  let html = '';
  for (let i = 1; i <= 5; i++) {
    const s = SAMPLE[i-1];
    const filled = s.state === 'verified' || s.state === 'flagged';
    const slotClass = s.state === 'verified' ? ' verified'
                    : s.state === 'flagged'  ? ' flagged'  : '';
    const statusClass = s.state === 'verified' ? 'verified'
                      : s.state === 'flagged'  ? 'flagged'  : 'pending';
    const statusLabel = s.state === 'verified' ? 'Verified'
                      : s.state === 'flagged'  ? 'Rotated'  : 'Pending';
    const gradient = filled
      ? `background: linear-gradient(135deg, oklch(0.26 0.005 240), oklch(0.36 0.008 240)); border-style: solid;`
      : '';
    const refInner   = filled ? `<div class="placeholder-fill">REF · 0${i}</div>` : `<span>no ref yet</span>`;
    const checkInner = filled ? `<div class="placeholder-fill">CHK · 0${i}</div>` : `<span>no check yet</span>`;
    const filledCls  = filled ? ' filled' : '';
    html += `
      <div class="card-slot${slotClass}" id="card-slot-${i}" data-idx="${i}">
        <div class="card-slot-head">
          <span class="card-slot-name">Card ${i}</span>
          <span class="card-slot-idx">SLOT 0${i}</span>
        </div>
        <div class="card-pair">
          <div class="card-image-wrap">
            <span class="card-image-label">Reference</span>
            <div class="card-image${filledCls}" id="card-${i}-ref" style="${gradient}">${refInner}</div>
          </div>
          <div class="card-image-wrap">
            <span class="card-image-label">Check</span>
            <div class="card-image${filledCls}" id="card-${i}-check" style="${gradient}">${checkInner}</div>
          </div>
        </div>
        <div class="card-status ${statusClass}" id="card-${i}-status">
          <span class="status-dot"></span><span>${statusLabel}</span>
        </div>
      </div>`;
  }
  grid.innerHTML = html;
})();
</script>

<!-- ============================================================ -->
<!-- Backend wiring · uses the existing Flask endpoints            -->
<!-- ============================================================ -->
<script>
(function () {
  const $ = id => document.getElementById(id);
  let runStart = 0;
  let runTickTimer = null;
  let runCount = 0;
  let lastWorkflowRunning = false;

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value == null || value === '' ? '—' : value;
  }

  function setIndicator(on) {
    const el = $('workflow-indicator');
    if (el) el.classList.toggle('running', !!on);
  }

  function setState(label, cls) {
    const text = $('run-state-text');
    if (!text) return;
    const holder = text.parentElement;
    text.textContent = label;
    holder.classList.remove('run-state-running', 'run-state-complete', 'run-state-stopped');
    if (cls) holder.classList.add(cls);
  }

  function fmtClock(ms) {
    const s = Math.floor(ms / 1000);
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  function startClock() {
    if (!runStart) runStart = performance.now();
    if (runTickTimer) clearInterval(runTickTimer);
    runTickTimer = setInterval(() => {
      setText('run-elapsed', fmtClock(performance.now() - runStart));
    }, 250);
  }

  function stopClock() {
    if (runTickTimer) clearInterval(runTickTimer);
    runTickTimer = null;
  }

  function resetClock() {
    stopClock();
    runStart = 0;
    setText('run-elapsed', '00:00');
  }

  function setMetric(name, pct) {
    const bar = $(`${name}-bar`);
    if (!bar) return;
    if (pct == null || Number.isNaN(Number(pct))) {
      bar.style.setProperty('--pct', 0);
      bar.classList.remove('warn', 'crit');
      setText(`${name}-level`, 'N/A');
      return;
    }
    pct = Math.max(0, Math.min(100, Math.round(Number(pct))));
    bar.style.setProperty('--pct', pct);
    bar.classList.toggle('warn', pct >= 70 && pct < 90);
    bar.classList.toggle('crit', pct >= 90);
    setText(`${name}-level`, `${pct}%`);
  }

  function setBattery(pct, state) {
    const el = $('battery');
    if (!el) return;
    if (pct == null || Number.isNaN(Number(pct))) {
      el.dataset.state = 'unknown';
      const bar = $('battery-bar');
      if (bar) bar.style.setProperty('--pct', 0);
      setText('battery-level', 'N/A');
      return;
    }
    pct = Math.max(0, Math.min(100, Math.round(Number(pct))));
    el.dataset.state = state || (pct <= 15 ? 'low' : pct <= 35 ? 'mid' : 'ok');
    const bar = $('battery-bar');
    if (bar) bar.style.setProperty('--pct', pct);
    setText('battery-level', `${pct}%`);
  }

  function setPhase(phase, status) {
    const order = ['calibrate', 'baseline', 'away', 'wait', 'return', 'check'];
    let current = String(phase || '').toLowerCase();
    const s = String(status || '').toUpperCase();

    if (current === 'starting') current = 'calibrate';
    if (!order.includes(current)) {
      if (s.includes('CALIBRATE') || s.includes('STARTING')) current = 'calibrate';
      else if (s.includes('BASELINE') || s.includes('SAVE')) current = 'baseline';
      else if (s.includes('LOOK_AWAY') || s.includes('AWAY')) current = 'away';
      else if (s.includes('WAIT')) current = 'wait';
      else if (s.includes('LOOK_BACK') || s.includes('RETURN')) current = 'return';
      else if (s.includes('CHECK')) current = 'check';
      else current = null;
    }

    const idx = order.indexOf(current);
    document.querySelectorAll('.phase').forEach((el, i) => {
      el.classList.remove('done', 'active');
      if (idx === -1) return;
      if (i < idx) el.classList.add('done');
      if (i === idx) el.classList.add('active');
    });

    if (current === 'done' || s.includes('DONE') || s.includes('COMPLETE')) {
      document.querySelectorAll('.phase').forEach(el => {
        el.classList.remove('active');
        el.classList.add('done');
      });
    }
  }

  function clearSlots() {
    for (let i = 1; i <= 5; i++) {
      setSlotImage(i, 'ref', null);
      setSlotImage(i, 'check', null);
      const slot = $(`card-slot-${i}`);
      if (slot) slot.classList.remove('verified', 'flagged');
      setSlotStatus(i, 'pending');
    }
    updateSlotCount();
  }

  function setSlotImage(slotId, type, src) {
    const el = $(`card-${slotId}-${type}`);
    if (!el) return;
    if (!src) {
      el.classList.remove('filled');
      el.removeAttribute('style');
      el.innerHTML = `<span>no ${type} yet</span>`;
      return;
    }
    const path = src.startsWith('/') ? src : `/${src}`;
    el.classList.add('filled');
    el.removeAttribute('style');
    const old = el.querySelector('img');
    if (old && old.getAttribute('src') === path) return;
    el.innerHTML = `<img src="${path}" alt="card ${slotId} ${type}">`;
  }

  function setSlotStatus(slotId, state) {
    const status = $(`card-${slotId}-status`);
    if (!status) return;
    const label = state === 'flagged' ? 'Rotated' : state === 'verified' ? 'Verified' : 'Pending';
    status.className = `card-status ${state}`;
    status.innerHTML = `<span class="status-dot"></span><span>${label}</span>`;
  }

  function updateSlotCount() {
    const count = document.querySelectorAll('.card-status.verified, .card-status.flagged').length;
    setText('slot-count', `${count} / 5 captured`);
  }

  function updateChips(groupId, missing) {
    const group = $(groupId);
    if (!group) return;
    const missingSet = new Set((missing || []).map(Number));
    Array.from(group.children).forEach((chip, index) => {
      chip.classList.toggle('done', !missingSet.has(index + 1));
    });
  }

  function updateSlots(slots) {
    if (!Array.isArray(slots)) return;
    slots.forEach(slot => {
      const id = Number(slot.slot_id);
      const slotEl = $(`card-slot-${id}`);
      if (!slotEl) return;
      setSlotImage(id, 'ref', slot.reference_image);
      setSlotImage(id, 'check', slot.check_image);
      slotEl.classList.remove('verified', 'flagged');
      if (slot.is_rotated) {
        slotEl.classList.add('flagged');
        setSlotStatus(id, 'flagged');
      } else if (slot.check_orientation === 'normal') {
        slotEl.classList.add('verified');
        setSlotStatus(id, 'verified');
      } else {
        setSlotStatus(id, 'pending');
      }
    });
    updateSlotCount();
  }

  function inferRunning(status) {
    const s = String(status || '').toUpperCase();
    return s.includes('WORKFLOW_') || s.includes('AUTO_') || s.includes('MOTION_') || s.includes('WAIT');
  }

  async function refreshStatus() {
    try {
      const res = await fetch('/status');
      const data = await res.json();
      const status = data.status || 'READY';
      const running = Boolean(data.workflow_active) || inferRunning(status);

      setText('status', status);
      setText('base-health-state', data.base_alive ? 'ALIVE' : 'DEAD');
      setText('base-health', `${data.base_health || 'unknown'} · /cmd_vel ${data.cmd_vel_subscribers ?? '?'}`);
      setText('result', data.result || 'No rotated card detected yet.');
      setText('last-report', data.last_report || '—');
      setMetric('cpu', data.system?.cpu_percent);
      setMetric('ram', data.system?.ram_percent);
      setBattery(data.system?.battery?.percent, data.system?.battery?.state);

      const result = $('result');
      if (result) {
        const empty = !data.result || data.result === 'No rotated card detected yet.';
        result.classList.toggle('empty', empty);
        result.classList.toggle('accent', !empty);
      }

      updateChips('missing-references', data.missing_refs || []);
      updateChips('missing-checks', data.missing_checks || []);
      updateSlots(data.slots || []);
      setPhase(data.ui_phase, status);
      setIndicator(running);

      if (running) {
        setState('Running', 'run-state-running');
        startClock();
      } else if (status.includes('STOP')) {
        setState('Stopped', 'run-state-stopped');
        stopClock();
      } else if (status.includes('DONE') || status.includes('COMPLETE')) {
        setState('Complete', 'run-state-complete');
        stopClock();
      } else {
        setState('Idle');
        if (!lastWorkflowRunning) resetClock();
      }
      lastWorkflowRunning = running;
    } catch (err) {
      setText('status', `STATUS FETCH ERROR: ${err}`);
      setState('Disconnected', 'run-state-stopped');
      setIndicator(false);
    }
  }

  async function doAction(name) {
    try {
      if (name === 'start') {
        runCount += 1;
        setText('run-count', String(runCount));
        resetClock();
      }
      const res = await fetch(`/action/${name}`, { method: 'POST' });
      const data = await res.json();
      if (!data.ok) setText('status', `ERROR: ${data.message || 'unknown'}`);
      await refreshStatus();
    } catch (err) {
      setText('status', `ACTION ERROR: ${err}`);
    }
  }

  function initFeeds() {
    const syncFeedAspect = (img) => {
      const body = img.closest('.vision-body');
      if (!body || !img.naturalWidth || !img.naturalHeight) return;
      body.style.setProperty('--feed-aspect', `${img.naturalWidth} / ${img.naturalHeight}`);
    };

    const camera = $('camera-feed');
    const cameraPlaceholder = $('camera-placeholder');
    if (camera) {
      camera.src = '/video_feed';
      camera.style.display = 'block';
      camera.onload = () => {
        syncFeedAspect(camera);
        if (cameraPlaceholder) cameraPlaceholder.style.display = 'none';
      };
    }
    const maskPlaceholder = $('mask-placeholder');
    const mask = $('mask-feed');
    if (mask) {
      mask.src = '/mask_feed';
      mask.style.display = 'block';
      mask.onload = () => {
        syncFeedAspect(mask);
        if (maskPlaceholder) maskPlaceholder.style.display = 'none';
      };
    }
  }

  $('btn-start').addEventListener('click', () => doAction('start'));
  $('btn-stop').addEventListener('click', () => doAction('stop_robot'));
  $('btn-clear').addEventListener('click', async () => {
    await doAction('clear_all');
    clearSlots();
    resetClock();
  });

  setBattery(null, 'unknown');
  setMetric('cpu', null);
  setMetric('ram', null);
  initFeeds();
  refreshStatus();
  setInterval(refreshStatus, 500);
})();
</script>
</body>
</html>

"""
