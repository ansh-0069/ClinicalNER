import re

with open("src/api/app.py", "r", encoding="utf-8") as f:
    content = f.read()

# I will replace the <style>...</style> blocks and some chart defaults in the templates.

def replace_style(template_name, new_style):
    global content
    # Find start of template
    t_start = content.find(template_name + ' = r"""<!DOCTYPE html>')
    if t_start == -1: return
    s_start = content.find("<style>", t_start)
    s_end = content.find("</style>", s_start) + len("</style>")
    
    if s_start != -1 and s_end != -1:
        content = content[:s_start] + new_style + content[s_end:]

# 1. DASHBOARD_TEMPLATE Styles
dashboard_style = """<style>
:root {
  --bg:       #F8FAFC;
  --bg2:      #FFFFFF;
  --bg3:      #F1F5F9;
  --border:   #E2E8F0;
  --border2:  #CBD5E1;
  --teal:     #0F766E;
  --teal-light: rgba(15,118,110,0.1);
  --amber:    #D97706;
  --amber-light: rgba(217,119,6,0.1);
  --red:      #E11D48;
  --red-light: rgba(225,29,72,0.1);
  --blue:     #2563EB;
  --blue-light: rgba(37,99,235,0.1);
  --txt:      #0F172A;
  --txt2:     #334155;
  --txt3:     #64748B;
  --shadow:   0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);
  --shadow-hover: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
  --card-bg:  #FFFFFF;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Instrument Sans', sans-serif;
  background: var(--bg);
  color: var(--txt);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Animated grid background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(15,118,110,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,118,110,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

body::after {
  content: '';
  position: fixed;
  top: -20%;
  right: -10%;
  width: 50vw;
  height: 60vh;
  background: radial-gradient(ellipse, rgba(37,99,235,0.04) 0%, transparent 65%);
  pointer-events: none;
  z-index: 0;
}

/* ── Layout ── */
.layout { display: flex; min-height: 100vh; position: relative; z-index: 1; }

/* ── Sidebar ── */
.sidebar {
  width: 240px;
  flex-shrink: 0;
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(12px);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 28px 0;
  position: sticky;
  top: 0;
  height: 100vh;
  z-index: 50;
}

.sidebar-logo {
  padding: 0 24px 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.logo-mark {
  font-family: 'Syne', sans-serif;
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--teal);
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo-dot {
  width: 8px; height: 8px;
  background: var(--teal);
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(15,118,110,0.4);
  animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%,100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px rgba(15,118,110,0.4); }
  50%      { opacity: 0.6; transform: scale(0.8); box-shadow: 0 0 4px rgba(15,118,110,0.2); }
}

.logo-sub {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  color: var(--txt3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 6px;
  font-weight: 600;
}

.nav-section {
  padding: 0 12px;
  flex: 1;
}

.nav-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--txt3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 0 16px;
  margin-bottom: 8px;
  margin-top: 20px;
  font-weight: 600;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
  color: var(--txt2);
  transition: all 0.2s ease;
  margin-bottom: 4px;
  text-decoration: none;
}
.nav-item:hover { background: var(--bg3); color: var(--teal); transform: translateX(2px); }
.nav-item.active { background: var(--teal-light); color: var(--teal); border-left: 3px solid var(--teal); }
.nav-icon { width: 18px; stroke-width: 2px; }

.sidebar-footer {
  padding: 20px 24px 0;
  border-top: 1px solid var(--border);
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--teal);
  background: var(--teal-light);
  padding: 8px 14px;
  border-radius: 20px;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.status-dot {
  width: 6px; height: 6px;
  background: var(--teal);
  border-radius: 50%;
  animation: pulse-dot 2s ease-in-out infinite;
}

/* ── Main content ── */
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }

/* ── Topbar ── */
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 36px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.8);
  backdrop-filter: blur(12px);
  position: sticky;
  top: 0;
  z-index: 20;
}

.topbar-title {
  font-family: 'Syne', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: var(--txt);
}

.topbar-path {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--txt3);
  letter-spacing: 0.08em;
  font-weight: 500;
  margin-top: 4px;
}

.topbar-right { display: flex; align-items: center; gap: 12px; }

.tag {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.1em;
  padding: 6px 14px;
  border-radius: 20px;
  text-transform: uppercase;
}

.tag-teal { background: var(--teal-light); color: var(--teal); border: 1px solid rgba(15,118,110,0.2); }
.tag-amber { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }

/* ── Content ── */
.content { padding: 32px 36px; flex: 1; }

/* ── Section header ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 800;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--txt2);
}

.section-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Stat cards ── */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 36px;
}

.stat-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
  opacity: 0;
  transform: translateY(16px);
  animation: card-in 0.5s ease forwards;
}

.stat-card:nth-child(1) { animation-delay: 0.05s; }
.stat-card:nth-child(2) { animation-delay: 0.12s; }
.stat-card:nth-child(3) { animation-delay: 0.19s; }
.stat-card:nth-child(4) { animation-delay: 0.26s; }

@keyframes card-in {
  to { opacity: 1; transform: translateY(0); }
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: linear-gradient(90deg, var(--teal), var(--blue));
  opacity: 0;
  transition: opacity 0.3s;
}

.stat-card:hover { border-color: var(--border2); transform: translateY(-4px); box-shadow: var(--shadow-hover); }
.stat-card:hover::before { opacity: 1; }

.stat-label {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt3);
  margin-bottom: 12px;
}

.stat-value {
  font-family: 'Syne', sans-serif;
  font-size: 36px;
  font-weight: 800;
  color: var(--txt);
  line-height: 1;
  letter-spacing: -1px;
}

.stat-value.teal  { color: var(--teal); }
.stat-value.amber { color: var(--amber); }
.stat-value.blue  { color: var(--blue); }

.stat-delta {
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  font-weight: 500;
  color: var(--txt3);
  margin-top: 10px;
}

.stat-icon {
  position: absolute;
  top: 24px; right: 24px;
  font-size: 24px;
  opacity: 0.1;
  filter: grayscale(100%);
}

/* ── Charts grid ── */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 36px;
}

.chart-wide { grid-column: span 2; }

.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow);
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.35s;
  transition: box-shadow 0.3s ease;
}

.chart-card:hover { box-shadow: var(--shadow-hover); }

.chart-card:nth-child(2) { animation-delay: 0.42s; }
.chart-card:nth-child(3) { animation-delay: 0.49s; }
.chart-card:nth-child(4) { animation-delay: 0.56s; }

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.chart-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--txt);
}

.chart-badge {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  color: var(--txt3);
  letter-spacing: 0.1em;
  background: var(--bg3);
  padding: 4px 10px;
  border-radius: 6px;
}

canvas { max-height: 240px; }

/* ── Audit table ── */
.audit-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  box-shadow: var(--shadow);
  overflow: hidden;
  margin-bottom: 24px;
  opacity: 0;
  animation: card-in 0.5s ease forwards;
  animation-delay: 0.6s;
}

.audit-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.audit-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--txt);
}

table { width: 100%; border-collapse: collapse; }

thead th {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt3);
  padding: 14px 24px;
  text-align: left;
  border-bottom: 1px solid var(--border);
  background: var(--bg2);
}

tbody td {
  padding: 16px 24px;
  font-size: 13px;
  font-weight: 500;
  color: var(--txt2);
  border-bottom: 1px solid var(--border);
}

tbody tr:last-child td { border-bottom: none; }
tbody tr { transition: background 0.15s; }
tbody tr:hover td { background: var(--bg3); color: var(--txt); }

/* ── Event badges ── */
.ev {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 4px 10px;
  border-radius: 20px;
}

.ev-teal   { background: var(--teal-light); color: var(--teal); border: 1px solid rgba(15,118,110,0.2); }
.ev-amber  { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }
.ev-red    { background: var(--red-light); color: var(--red); border: 1px solid rgba(225,29,72,0.2); }
.ev-blue   { background: var(--blue-light); color: var(--blue); border: 1px solid rgba(37,99,235,0.2); }

.ev-dot { width: 4px; height: 4px; border-radius: 50%; background: currentColor; }

/* ── Count col ── */
.count {
  font-family: 'DM Mono', monospace;
  font-size: 14px;
  font-weight: 600;
  color: var(--txt);
}

.ts {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  font-weight: 500;
  color: var(--txt3);
}

/* ── Loading shimmer ── */
.shimmer {
  background: linear-gradient(90deg, var(--bg3) 25%, #FFFFFF 50%, var(--bg3) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 6px;
  height: 24px;
  width: 90px;
}
@keyframes shimmer { to { background-position: -200% 0; } }

/* ── Error ── */
.error-banner {
  background: var(--red-light);
  border: 1px solid rgba(225,29,72,0.2);
  border-radius: 12px;
  padding: 16px 20px;
  font-family: 'Instrument Sans', sans-serif;
  font-weight: 500;
  font-size: 14px;
  color: var(--red);
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}

@media (max-width: 1000px) {
  .sidebar { display: none; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .charts-grid { grid-template-columns: 1fr; }
  .chart-wide { grid-column: span 1; }
}
</style>"""

replace_style("DASHBOARD_TEMPLATE", dashboard_style)

# 2. REPORT_TEMPLATE
report_style = """<style>
:root {
  --bg:       #F8FAFC;
  --bg2:      #FFFFFF;
  --bg3:      #F1F5F9;
  --border:   #E2E8F0;
  --teal:     #0F766E;
  --teal-light: rgba(15,118,110,0.1);
  --amber:    #D97706;
  --amber-light: rgba(217,119,6,0.1);
  --red:      #E11D48;
  --txt:      #0F172A;
  --txt2:     #334155;
  --txt3:     #64748B;
  --shadow:   0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);
  --card-bg:  #FFFFFF;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Instrument Sans', sans-serif;
  background: var(--bg);
  color: var(--txt);
  min-height: 100vh;
  padding: 0;
}
body::before {
  content: '';
  position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(15,118,110,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,118,110,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none; z-index: 0;
}
.page { position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 40px; }

.breadcrumb {
  display: flex; align-items: center; gap: 8px;
  font-family: 'DM Mono', monospace; font-size: 11px; font-weight: 500;
  color: var(--txt3); margin-bottom: 32px; letter-spacing: 0.08em;
}
.breadcrumb a { color: var(--teal); text-decoration: none; font-weight: 600;}
.breadcrumb a:hover { text-decoration: underline; }
.breadcrumb-sep { opacity: 0.4; }

.report-header {
  display: flex; align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 32px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--border);
}

.report-title {
  font-family: 'Syne', sans-serif;
  font-size: 28px; font-weight: 800;
  letter-spacing: -0.5px;
  color: var(--txt);
}

.report-title span { color: var(--teal); }

.report-meta {
  display: flex; gap: 24px; margin-top: 12px; flex-wrap: wrap;
}

.meta-item {
  font-family: 'DM Mono', monospace;
  font-size: 11px; letter-spacing: 0.1em;
  color: var(--txt3); text-transform: uppercase; font-weight: 600;
}
.meta-item strong { color: var(--txt); font-weight: 700; margin-left: 8px; }

.header-right { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-start; }

.badge {
  font-family: 'DM Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  padding: 6px 14px; border-radius: 20px; text-transform: uppercase;
}
.badge-teal  { background: var(--teal-light); color: var(--teal);  border: 1px solid rgba(15,118,110,0.2); }
.badge-amber { background: var(--amber-light); color: var(--amber); border: 1px solid rgba(217,119,6,0.2); }

/* Diff panels */
.diff-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 32px;
}

.diff-panel {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: var(--shadow);
}

.panel-head {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}

.panel-head-title {
  font-family: 'DM Mono', monospace;
  font-size: 10px; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
  display: flex; align-items: center; gap: 10px;
}

.panel-dot {
  width: 8px; height: 8px; border-radius: 50%;
}

.panel-orig  .panel-dot { background: var(--amber); }
.panel-masked .panel-dot { background: var(--teal);  }
.panel-orig  .panel-head { border-bottom-color: rgba(217,119,6,0.2); background: var(--amber-light); }
.panel-masked .panel-head { border-bottom-color: rgba(15,118,110,0.2); background: var(--teal-light); }
.panel-orig  .panel-head-title { color: var(--amber); }
.panel-masked .panel-head-title { color: var(--teal);  }

.panel-body {
  padding: 24px;
  font-family: 'DM Mono', monospace;
  font-size: 13px; line-height: 1.8;
  white-space: pre-wrap; word-break: break-word;
  max-height: 500px; overflow-y: auto;
  color: var(--txt2);
}

.panel-body::-webkit-scrollbar { width: 6px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.masked-token {
  background: var(--teal-light);
  color: var(--teal);
  border: 1px solid rgba(15,118,110,0.3);
  border-radius: 6px;
  padding: 2px 6px;
  font-weight: 600;
  font-size: 12px;
}

/* Stats row */
.stats-row {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;
  margin-bottom: 0;
}

.mini-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
  box-shadow: var(--shadow);
}
.mini-label {
  font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--txt3); margin-bottom: 8px;
}
.mini-value {
  font-family: 'Syne', sans-serif; font-size: 26px; font-weight: 800;
  color: var(--teal);
}
.mini-value.white { color: var(--txt); }
.mini-value.amber { color: var(--amber); }

@media (max-width: 800px) {
  .diff-grid { grid-template-columns: 1fr; }
  .stats-row { grid-template-columns: repeat(2, 1fr); }
}
</style>"""

replace_style("REPORT_TEMPLATE", report_style)

# 3. API_EXPLORER_TEMPLATE
api_style = """<style>
:root{--bg:#F8FAFC;--bg2:#FFFFFF;--bg3:#F1F5F9;--border:#E2E8F0;--border2:#CBD5E1;--teal:#0F766E;--amber:#D97706;--red:#E11D48;--blue:#2563EB;--txt:#0F172A;--txt2:#334155;--txt3:#64748B; --shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025); --shadow-hover: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(15,118,110,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(15,118,110,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:240px;flex-shrink:0;background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:19px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 10px rgba(15,118,110,.4);animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;color:var(--txt3);font-weight:600;letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;margin-top:20px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--txt2);transition:all .2s;margin-bottom:4px;text-decoration:none;}
.nav-item:hover{background:var(--bg3);color:var(--teal);transform:translateX(2px);}
.nav-item.active{background:rgba(15,118,110,0.1);color:var(--teal);border-left:3px solid var(--teal);}
.nav-icon{width:18px;stroke-width:2px;opacity:.8;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--teal);background:rgba(15,118,110,0.1);padding:8px 14px;border-radius:20px;letter-spacing:.05em;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 36px;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);}
.topbar-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.topbar-path{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);letter-spacing:.08em;margin-top:4px;}
.content{padding:32px 36px;flex:1;}
.tag{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;padding:6px 14px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}
.tag-post{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}
.tag-get{background:rgba(37,99,235,0.1);color:var(--blue);border:1px solid rgba(37,99,235,.2);}

/* Endpoint cards */
.endpoint{background:var(--bg2);border:1px solid var(--border);border-radius:16px;margin-bottom:20px;overflow:hidden;box-shadow:var(--shadow);transition:box-shadow .3s;}
.endpoint:hover{box-shadow:var(--shadow-hover);}
.ep-header{display:flex;align-items:center;gap:16px;padding:20px 24px;cursor:pointer;transition:background .15s;}
.ep-header:hover{background:var(--bg3);}
.ep-method{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;padding:6px 12px;border-radius:8px;letter-spacing:.08em;}
.ep-method.post{background:rgba(15,118,110,0.1);color:var(--teal);}
.ep-method.get{background:rgba(37,99,235,0.1);color:var(--blue);}
.ep-path{font-family:'DM Mono',monospace;font-size:14px;font-weight:600;color:var(--txt);}
.ep-desc{font-size:13px;font-weight:500;color:var(--txt3);margin-left:auto;}
.ep-body{border-top:1px solid var(--border);padding:24px;display:none;background:var(--bg2);}
.ep-body.open{display:block;}

label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);display:block;margin-bottom:8px;}
textarea,input[type=text]{width:100%;font-weight:500;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:16px 18px;color:var(--txt);font-family:'DM Mono',monospace;font-size:13px;resize:vertical;outline:none;transition:border-color .2s;box-shadow:inset 0 1px 2px rgba(0,0,0,0.02);}
textarea:focus,input[type=text]:focus{border-color:var(--teal);background:#FFF;}
textarea{min-height:120px;}

.btn{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;padding:12px 24px;border-radius:8px;border:none;cursor:pointer;transition:all .2s;margin-top:16px;}
.btn-primary{background:var(--teal);color:#FFF;box-shadow:0 4px 6px -1px rgba(15,118,110,0.2);}
.btn-primary:hover{background:#0D9488;transform:translateY(-1px);box-shadow:0 6px 8px -1px rgba(15,118,110,0.3);}

.response-box{margin-top:20px;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:16px;display:none;}
.response-box.visible{display:block;}
.response-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--txt3);margin-bottom:12px;}
.response-content{font-family:'DM Mono',monospace;font-size:12px;font-weight:500;color:var(--teal);white-space:pre-wrap;word-break:break-all;max-height:360px;overflow-y:auto;}
.response-content::-webkit-scrollbar{width:6px;}
.response-content::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px;}
.status-ok{color:var(--teal);}
.status-err{color:var(--red);}

.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:24px;}
.section-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
</style>"""

replace_style("API_EXPLORER_TEMPLATE", api_style)


# 4. SYSTEM_STATUS_TEMPLATE
sys_style = """<style>
:root{--bg:#F8FAFC;--bg2:#FFFFFF;--bg3:#F1F5F9;--border:#E2E8F0;--border2:#CBD5E1;--teal:#0F766E;--amber:#D97706;--red:#E11D48;--blue:#2563EB;--txt:#0F172A;--txt2:#334155;--txt3:#64748B;--shadow:0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.025);--shadow-hover:0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Instrument Sans',sans-serif;background:var(--bg);color:var(--txt);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(15,118,110,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(15,118,110,0.03) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.layout{display:flex;min-height:100vh;position:relative;z-index:1;}
.sidebar{width:240px;flex-shrink:0;background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:28px 0;position:sticky;top:0;height:100vh;}
.sidebar-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);margin-bottom:20px;}
.logo-mark{font-family:'Syne',sans-serif;font-size:19px;font-weight:800;letter-spacing:-0.5px;color:var(--teal);display:flex;align-items:center;gap:10px;}
.logo-dot{width:8px;height:8px;background:var(--teal);border-radius:50%;box-shadow:0 0 10px rgba(15,118,110,.4);animation:pulse-dot 2s ease-in-out infinite;}
@keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.6;transform:scale(.8);}}
.logo-sub{font-family:'DM Mono',monospace;font-size:9px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;margin-top:6px;}
.nav-section{padding:0 12px;flex:1;}
.nav-label{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:var(--txt3);letter-spacing:.12em;text-transform:uppercase;padding:0 16px;margin-bottom:8px;margin-top:20px;}
.nav-item{display:flex;align-items:center;gap:12px;padding:10px 16px;border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--txt2);transition:all .2s;margin-bottom:4px;text-decoration:none;}
.nav-item:hover{background:var(--bg3);color:var(--teal);transform:translateX(2px);}
.nav-item.active{background:rgba(15,118,110,0.1);color:var(--teal);border-left:3px solid var(--teal);}
.nav-icon{width:18px;stroke-width:2px;opacity:.8;}
.sidebar-footer{padding:20px 24px 0;border-top:1px solid var(--border);}
.status-badge{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:var(--teal);background:rgba(15,118,110,0.1);padding:8px 14px;border-radius:20px;letter-spacing:.05em;}
.status-dot{width:6px;height:6px;background:var(--teal);border-radius:50%;animation:pulse-dot 2s ease-in-out infinite;}
.main{flex:1;display:flex;flex-direction:column;min-width:0;}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:20px 36px;border-bottom:1px solid var(--border);background:rgba(255,255,255,0.8);backdrop-filter:blur(12px);}
.topbar-title{font-family:'Syne',sans-serif;font-size:18px;font-weight:700;}
.topbar-path{font-family:'DM Mono',monospace;font-size:11px;font-weight:500;color:var(--txt3);letter-spacing:.08em;margin-top:4px;}
.content{padding:32px 36px;flex:1;}
.section-header{display:flex;align-items:baseline;gap:16px;margin-bottom:24px;}
.section-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;letter-spacing:.18em;text-transform:uppercase;color:var(--txt3);}
.section-line{flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
.tag{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.1em;padding:6px 14px;border-radius:20px;text-transform:uppercase;}
.tag-teal{background:rgba(15,118,110,0.1);color:var(--teal);border:1px solid rgba(15,118,110,.2);}

/* Status cards */
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:36px;}
.status-card{background:var(--bg2);border:1px solid var(--border);border-radius:16px;padding:24px;box-shadow:var(--shadow);opacity:0;transform:translateY(16px);animation:fadein .5s cubic-bezier(0.16, 1, 0.3, 1) forwards;}
.status-card:nth-child(2){animation-delay:.08s;}
.status-card:nth-child(3){animation-delay:.16s;}
@keyframes fadein{to{opacity:1;transform:translateY(0);}}
.status-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}
.status-card-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--txt);}
.status-indicator{display:flex;align-items:center;gap:8px;font-family:'DM Mono',monospace;font-size:11px;font-weight:600;}
.ind-dot{width:8px;height:8px;border-radius:50%;}
.ind-ok{background:var(--teal);box-shadow:0 0 8px rgba(15,118,110,0.5);}
.ind-warn{background:var(--amber);box-shadow:0 0 8px rgba(217,119,6,0.5);}
.ind-err{background:var(--red);box-shadow:0 0 8px rgba(225,29,72,0.5);}
.ok{color:var(--teal);}
.warn{color:var(--amber);}
.err{color:var(--red);}

/* Check rows */
.check-list{display:flex;flex-direction:column;gap:10px;}
.check-row{display:flex;align-items:center;justify-content:space-between;padding:12px 16px;background:var(--bg3);border-radius:10px;border:1px solid var(--border);}
.check-label{font-size:13px;font-weight:600;color:var(--txt2);}
.check-value{font-family:'DM Mono',monospace;font-size:12px;font-weight:600;}

/* Route table */
.route-table{background:var(--bg2);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);}
.route-header{display:flex;align-items:center;justify-content:space-between;padding:20px 24px;border-bottom:1px solid var(--border);background:var(--bg3);}
.route-title{font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:var(--txt);}
table{width:100%;border-collapse:collapse;}
thead th{font-family:'DM Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;color:var(--txt3);padding:14px 24px;text-align:left;border-bottom:1px solid var(--border);background:var(--bg2);}
tbody td{padding:16px 24px;font-size:13px;font-weight:500;color:var(--txt2);border-bottom:1px solid var(--border);}
tbody tr:last-child td{border-bottom:none;}
tbody tr{transition:background .2s;}
tbody tr:hover td{background:rgba(15,118,110,0.02);color:var(--txt);}
.method{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;letter-spacing:0.05em;}
.method-get{background:rgba(37,99,235,0.1);color:var(--blue);}
.method-post{background:rgba(15,118,110,0.1);color:var(--teal);}
.test-btn{font-family:'DM Mono',monospace;font-size:11px;font-weight:600;letter-spacing:.08em;padding:6px 14px;border-radius:8px;border:1px solid var(--border2);background:#FFF;color:var(--txt2);cursor:pointer;transition:all .2s;text-decoration:none;box-shadow:0 1px 2px rgba(0,0,0,0.05);}
.test-btn:hover{border-color:var(--teal);color:var(--teal);transform:translateY(-1px);box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);}
</style>"""

replace_style("SYSTEM_STATUS_TEMPLATE", sys_style)


# Next, let's fix the Chart settings in DASHBOARD_TEMPLATE
# we will just do a simple string replace for the Chart defaults.
js_old = "Chart.defaults.color = '#7A9AB8';"
js_new = "Chart.defaults.color = '#64748B'; /* Slate 500 */ Chart.defaults.font.weight = '500';"
content = content.replace(js_old, js_new)

palette_old = "const PALETTE = ['#00E5B4','#4F8EF7','#FFB547','#FF5C6A','#B794F4','#76E4F7','#FBB6CE','#9AE6B4','#F6AD55'];"
palette_new = "const PALETTE = ['#0F766E','#2563EB','#D97706','#E11D48','#7C3AED','#0EA5E9','#DB2777','#10B981','#F59E0B'];"
content = content.replace(palette_old, palette_new)

grid_old = "const GRID = { color: 'rgba(0,229,180,0.05)', drawBorder: false };"
grid_new = "const GRID = { color: '#F1F5F9', drawBorder: false };"
content = content.replace(grid_old, grid_new)

# Bar chart settings
bar_old = "backgroundColor: 'rgba(0,229,180,0.15)',"
bar_new = "backgroundColor: 'rgba(15,118,110,0.15)', borderColor: '#0F766E', borderWidth: 1.5, borderRadius: 6,"
content = content.replace(bar_old, bar_new)

bar_line_2_old = "borderColor: '#00E5B4',"
content = content.replace(bar_line_2_old, "")

bar_line_3_old = "borderWidth: 1,"
content = content.replace(bar_line_3_old, "")

bar_line_4_old = "borderRadius: 4,"
content = content.replace(bar_line_4_old, "")

hover_old = "hoverBackgroundColor: 'rgba(0,229,180,0.28)',"
hover_new = "hoverBackgroundColor: 'rgba(15,118,110,0.25)',"
content = content.replace(hover_old, hover_new)

with open("src/api/app.py", "w", encoding="utf-8") as f:
    f.write(content)
print("UI templates updated.")
