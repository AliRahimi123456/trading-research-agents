import html as html_lib
import json

import pandas as pd

from src.config import WS, RESULTS_DIR, REVIEWS_DIR
from src.registry import REGISTRY, _decisions

HOLDOUT_COLS = ["cagr", "sharpe", "sortino", "max_dd", "ann_turnover"]

NUMERIC_COLS = ["dev_cagr", "dev_sharpe", "dev_sortino", "dev_max_dd", "dev_turnover",
                "val_cagr", "val_sharpe", "val_max_dd", "dev_cagr_20bps"]


def _fmt(val) -> str:
    if pd.isna(val):
        return ""
    if isinstance(val, float):
        return f"{val:.4f}"
    return html_lib.escape(str(val))


def _registry_table_html() -> str:
    if not REGISTRY.exists():
        return "<p class='empty'>No runs recorded yet.</p>"
    df = pd.read_csv(REGISTRY)
    if df.empty:
        return "<p class='empty'>No runs recorded yet.</p>"

    cols = ["version", "run", "status", "params"] + NUMERIC_COLS + ["error"]
    cols = [c for c in cols if c in df.columns]

    head = "".join(f"<th class='{'num' if c in NUMERIC_COLS else ''}'>{c}</th>" for c in cols)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            if c == "status":
                pill_class = "pill-ok" if row[c] == "ok" else "pill-error"
                cells.append(f"<td><span class='pill {pill_class}'>{row[c]}</span></td>")
            elif c in NUMERIC_COLS:
                cells.append(f"<td class='num'>{_fmt(row[c])}</td>")
            else:
                cells.append(f"<td>{_fmt(row[c])}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _decisions_html() -> str:
    decisions = _decisions()
    if not decisions:
        return "<p class='empty'>No decisions recorded yet.</p>"
    cards = []
    for d in decisions:
        cards.append(f"""
        <div class="decision-card">
          <div class="decision-head">
            <span class="tag">{d['version']}</span>
            <span class="arrow">&rarr;</span>
            <span class="champion-pill">{d['champion']}</span>
          </div>
          <p class="rationale">{html_lib.escape(d['rationale'])}</p>
        </div>""")
    return "".join(cards)


def _reviews_html() -> str:
    files = sorted(REVIEWS_DIR.glob("*.md"))
    if not files:
        return "<p class='empty'>No reviews yet.</p>"
    sections = []
    for f in files:
        sections.append(f"<div class='review-card'><h3>{f.stem}</h3><pre>{html_lib.escape(f.read_text())}</pre></div>")
    return "".join(sections)


def _frozen_html():
    frozen_path = WS / "strategies" / "frozen.json"
    if not frozen_path.exists():
        return None
    frozen = json.loads(frozen_path.read_text())
    params_str = ", ".join(f"{k}={v}" for k, v in frozen["params"].items())
    return f"""
    <div class="frozen-card">
      <div class="frozen-head">
        <span class="frozen-label">Frozen champion</span>
        <span class="frozen-version">{frozen['version']}</span>
      </div>
      <p class="frozen-params">{html_lib.escape(params_str)}</p>
      <p class="rationale">{html_lib.escape(frozen['rationale'])}</p>
    </div>"""


def _holdout_html() -> str:
    results_path = WS / "results" / "holdout.json"
    if not results_path.exists():
        return "<p class='empty'>Holdout not yet unlocked.</p>"
    final = json.loads(results_path.read_text())

    from src.benchmarks import benchmark_table
    bench = benchmark_table("holdout")

    rows_html = []
    for split in ["dev", "val", "holdout"]:
        m = final[split]
        cells = "".join(f"<td class='num'>{m[c]:.4f}</td>" for c in HOLDOUT_COLS)
        row_class = " class='hl'" if split == "holdout" else ""
        rows_html.append(f"<tr{row_class}><td>frozen strategy &mdash; {split}</td>{cells}</tr>")
    for name, row in bench[HOLDOUT_COLS].iterrows():
        cells = "".join(f"<td class='num'>{row[c]:.4f}</td>" for c in HOLDOUT_COLS)
        rows_html.append(f"<tr><td>{name} &mdash; holdout</td>{cells}</tr>")

    head = "<th>strategy / period</th>" + "".join(f"<th class='num'>{c}</th>" for c in HOLDOUT_COLS)
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"


def _report_html():
    report_path = WS / "report.md"
    if not report_path.exists():
        return None
    return f"<pre>{html_lib.escape(report_path.read_text())}</pre>"


def _images_html() -> str:
    files = sorted(RESULTS_DIR.glob("*.png"))
    if not files:
        return "<p class='empty'>No equity curves yet.</p>"
    figs = "".join(
        f"<figure><img src='results/{f.name}' loading='lazy'><figcaption>{f.stem}</figcaption></figure>"
        for f in files
    )
    return f"<div class='equity-grid'>{figs}</div>"


CSS = """
:root{
  --bg:#f5f6f8; --surface:#ffffff; --surface-2:#eef0f3; --border:#dde1e7;
  --text:#1a1d23; --text-muted:#5c6472; --accent:#0e7c74; --accent-soft:#dcf0ee;
  --ok:#1e7a34; --ok-bg:#e6f4ea; --err:#a3231a; --err-bg:#fbe9e7;
  --mono:'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  --sans:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0f1216; --surface:#161a20; --surface-2:#1c2129; --border:#2b313b;
    --text:#e7e9ed; --text-muted:#9aa2af; --accent:#3fbdb2; --accent-soft:#173330;
    --ok:#4ade80; --ok-bg:#0f2b1a; --err:#f87171; --err-bg:#2e1414;
  }
}
*{box-sizing:border-box;}
body{
  background:var(--bg); color:var(--text); font-family:var(--sans);
  max-width:1100px; margin:0 auto; padding:2.5rem 1.5rem 4rem; line-height:1.55;
}
h1{ font-size:1.7rem; margin:0 0 0.3rem; }
.subtitle{ color:var(--text-muted); margin:0 0 2rem; font-size:0.95rem; }
h2{
  font-size:1.05rem; text-transform:uppercase; letter-spacing:0.04em;
  color:var(--text-muted); border-bottom:1px solid var(--border);
  padding-bottom:0.5rem; margin:2.5rem 0 1rem;
}
.champion-banner{
  display:inline-flex; align-items:center; gap:0.6rem; background:var(--accent-soft);
  border:1px solid var(--accent); border-radius:8px; padding:0.5rem 1rem;
  font-weight:600; color:var(--accent); margin-bottom:1.5rem;
}
pre{
  background:var(--surface-2); border:1px solid var(--border); border-radius:8px;
  padding:1rem 1.2rem; overflow-x:auto; white-space:pre-wrap; font-family:var(--mono);
  font-size:0.85rem; color:var(--text);
}
.empty{ color:var(--text-muted); font-style:italic; }
.table-wrap{ overflow-x:auto; border:1px solid var(--border); border-radius:8px; }
table{ border-collapse:collapse; width:100%; font-size:0.82rem; background:var(--surface); }
th,td{ padding:0.5rem 0.7rem; border-bottom:1px solid var(--border); text-align:left; white-space:nowrap; }
th{ background:var(--surface-2); font-family:var(--mono); font-weight:600; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.03em; color:var(--text-muted); }
td.num, th.num{ text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }
tr:last-child td{ border-bottom:none; }
.pill{ display:inline-block; padding:0.15rem 0.55rem; border-radius:999px; font-size:0.72rem; font-weight:600; }
.pill-ok{ background:var(--ok-bg); color:var(--ok); }
.pill-error{ background:var(--err-bg); color:var(--err); }
.decision-card{
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:1rem 1.2rem; margin-bottom:0.8rem;
}
.decision-head{ display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem; }
.tag{ font-family:var(--mono); font-size:0.85rem; color:var(--text-muted); }
.arrow{ color:var(--text-muted); }
.champion-pill{ background:var(--accent-soft); color:var(--accent); padding:0.15rem 0.6rem; border-radius:6px; font-weight:600; font-size:0.85rem; }
.rationale{ margin:0; color:var(--text-muted); font-size:0.9rem; }
.review-card{
  background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:1rem 1.2rem; margin-bottom:1rem;
}
.review-card h3{ margin:0 0 0.6rem; font-family:var(--mono); font-size:0.9rem; color:var(--accent); }
.equity-grid{ display:grid; grid-template-columns:repeat(auto-fill, minmax(240px, 1fr)); gap:1rem; }
.equity-grid figure{ margin:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.6rem; }
.equity-grid img{ width:100%; display:block; border-radius:4px; }
.equity-grid figcaption{ font-family:var(--mono); font-size:0.75rem; color:var(--text-muted); text-align:center; margin-top:0.4rem; }
.frozen-card{
  background:var(--surface); border:2px solid var(--accent); border-radius:10px;
  padding:1.2rem 1.4rem; margin-bottom:1rem;
}
.frozen-head{ display:flex; align-items:baseline; gap:0.7rem; margin-bottom:0.5rem; }
.frozen-label{ text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em; color:var(--text-muted); }
.frozen-version{ font-family:var(--mono); font-size:1.3rem; font-weight:700; color:var(--accent); }
.frozen-params{ font-family:var(--mono); font-size:0.82rem; color:var(--text-muted); margin:0 0 0.6rem; }
tr.hl td{ background:var(--accent-soft); font-weight:600; }
"""


def build_dashboard():
    decisions = _decisions()
    current_champion = decisions[-1]["champion"] if decisions else "none yet"
    benchmarks_text = (WS / "BENCHMARKS.md").read_text() if (WS / "BENCHMARKS.md").exists() else "not yet computed"
    rule_text = (WS / "SELECTION_RULE.md").read_text() if (WS / "SELECTION_RULE.md").exists() else "not yet written"

    frozen_html = _frozen_html()
    frozen_section = f"<h2>Frozen Champion</h2>{frozen_html}" if frozen_html else ""
    holdout_section = f"<h2>Holdout Performance</h2>{_holdout_html()}" if frozen_html else ""
    report_html = _report_html()
    report_section = f"<h2>Audit Report</h2>{report_html}" if report_html else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Dashboard</title>
<style>{CSS}</style></head>
<body>
<h1>Trading Research Dashboard</h1>
<p class="subtitle">Live view of the deterministic research trail: registry, decisions, reviews, equity curves.</p>
<div class="champion-banner">&#9733; Current champion: {current_champion}</div>
{frozen_section}
{holdout_section}
<h2>Selection Rule</h2><pre>{html_lib.escape(rule_text)}</pre>
<h2>Fixed Benchmarks</h2><pre>{html_lib.escape(benchmarks_text)}</pre>
<h2>Experiment Registry</h2>{_registry_table_html()}
<h2>Decisions</h2>{_decisions_html()}
<h2>Critic Reviews</h2>{_reviews_html()}
<h2>Equity Curves</h2>{_images_html()}
{report_section}
</body></html>"""

    out = WS / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    path = build_dashboard()
    print(f"dashboard written to {path}")
