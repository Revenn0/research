#!/usr/bin/env python3
"""Gera painel HTML + JSON de status do batch LM 100M (TokenBlend vs baseline)."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS_JSON = ROOT / "lm_100m_status.json"
DASHBOARD_HTML = ROOT / "lm_100m_dashboard.html"
TRACKER_MD = ROOT / "lm_100m_tracker.md"
LOG_PATH = ROOT / "experiments_log_lm.jsonl"
BATCH_LOG = ROOT / "batch_lm_100m.log"
RESULTS_DIR = ROOT / "results" / "lm_100m"
LIVE_PROGRESS = RESULTS_DIR / "_live_progress.json"
STEPS_PER_SEED = 1000
SEEDS_TOTAL = 3
SEC_PER_STEP = 18.5  # calibrado CPU ~100M char LM

EXPERIMENTS = [
    {
        "id": "exp-lm-100m-baseline",
        "label": "global-attn (baseline)",
        "mixing": "global-attn",
        "params_m": 99.53,
        "kind": "BASELINE",
    },
    {
        "id": "exp-lm-100m-tokenblend",
        "label": "token_blend k=3",
        "mixing": "token_blend",
        "params_m": 74.76,
        "kind": "NOVEL",
    },
    {
        "id": "exp-lm-100m-tokenblend-ms",
        "label": "token_blend_ms 3+5",
        "mixing": "token_blend_ms",
        "params_m": 74.81,
        "kind": "NOVEL",
    },
]

MIXING_TO_ID = {e["mixing"]: e["id"] for e in EXPERIMENTS}


def load_completed() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not LOG_PATH.exists():
        return done
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("id", "").startswith("exp-lm-100m"):
            done[e["id"]] = e
    return done


def load_seed_results() -> dict[str, list[dict]]:
    by_exp: dict[str, list[dict]] = {}
    if not RESULTS_DIR.exists():
        return by_exp
    for p in sorted(RESULTS_DIR.glob("*.json")):
        if p.name == "_live_progress.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        eid = data.get("experiment_id") or p.stem.rsplit("_seed", 1)[0]
        by_exp.setdefault(eid, []).append(data)
    for eid in by_exp:
        by_exp[eid].sort(key=lambda x: x.get("seed", 0))
    return by_exp


def load_live_progress() -> dict | None:
    if not LIVE_PROGRESS.exists():
        return None
    try:
        return json.loads(LIVE_PROGRESS.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def parse_etime_sec(etime: str) -> float:
    if "-" in etime:
        d, t = etime.split("-", 1)
        hh, mm, ss = (t.split(":") + ["0"])[:3]
        return int(d) * 86400 + int(hh) * 3600 + int(mm) * 60 + int(ss)
    if etime.count(":") == 2:
        hh, mm, ss = etime.split(":")
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    if etime.count(":") == 1:
        mm, ss = etime.split(":")
        return int(mm) * 60 + int(ss)
    return float(etime)


def detect_running() -> dict | None:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if "train_lm_exp.py" not in line or "grep" in line:
            continue
        m_seed = re.search(r"--seed\s+(\d+)", line)
        m_mix = re.search(r"--mixing\s+(\S+)", line)
        m_tag = re.search(r"--tag\s+(\S+)", line)
        seed = int(m_seed.group(1)) if m_seed else None
        mixing = m_mix.group(1) if m_mix else None
        exp_id = m_tag.group(1) if m_tag else MIXING_TO_ID.get(mixing or "", None)
        parts = line.split()
        pid = parts[1] if len(parts) > 1 else None
        etime_sec = 0.0
        if pid:
            try:
                et = subprocess.check_output(["ps", "-p", pid, "-o", "etime="], text=True).strip()
                etime_sec = parse_etime_sec(et)
            except (subprocess.CalledProcessError, ValueError):
                pass
        return {
            "seed": seed,
            "mixing": mixing,
            "experiment_id": exp_id,
            "pid": pid,
            "etime_sec": etime_sec,
        }
    return None


def detect_current_exp_id(completed: dict[str, dict]) -> str | None:
    running = detect_running()
    if running and running.get("experiment_id"):
        return running["experiment_id"]
    for meta in EXPERIMENTS:
        if meta["id"] not in completed:
            return meta["id"]
    return EXPERIMENTS[-1]["id"]


def seed_index(seed: int | None) -> int:
    if seed is None:
        return 0
    return max(0, min(SEEDS_TOTAL - 1, seed - 1000))


def compute_progress_pct(
    status: str,
    seeds_done: int,
    current_seed: int | None,
    current_step: int,
    total_steps: int = STEPS_PER_SEED,
) -> float:
    if status == "done":
        return 100.0
    if status == "pending":
        return 0.0
    si = seed_index(current_seed)
    step_frac = min(1.0, current_step / max(1, total_steps))
    return round(100.0 * (seeds_done + si + step_frac) / SEEDS_TOTAL, 1)


def fmt_loss(x: float | None) -> str:
    return f"{x:.4f}" if x is not None else "—"


def fmt_delta_loss(x: float | None) -> str:
    """Delta positivo = melhor que baseline (menor loss)."""
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.4f}"


def estimate_eta_hours(global_pct: float, batch_start_utc: datetime | None) -> str:
    if global_pct <= 0 or batch_start_utc is None:
        return "calculando…"
    elapsed = (datetime.now(timezone.utc) - batch_start_utc).total_seconds()
    if elapsed <= 0:
        return "calculando…"
    total_sec = elapsed / (global_pct / 100.0)
    remaining_h = max(0, (total_sec - elapsed) / 3600)
    return f"~{remaining_h:.1f} h restantes"


def batch_start_time() -> datetime | None:
    if not BATCH_LOG.exists():
        return None
    try:
        stat = BATCH_LOG.stat()
        return datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    except OSError:
        return None


def build_status() -> dict:
    completed = load_completed()
    seed_results = load_seed_results()
    running = detect_running()
    current_id = detect_current_exp_id(completed)
    live = load_live_progress()

    baseline_loss = None
    if "exp-lm-100m-baseline" in completed:
        baseline_loss = completed["exp-lm-100m-baseline"]["result"]["best_val_loss_mean"]

    current_step = 0
    current_step_pct = 0.0
    live_val_loss = None
    if live and running and live.get("seed") == running.get("seed"):
        current_step = int(live.get("step", 0))
        current_step_pct = float(live.get("step_pct", 0))
        live_val_loss = live.get("best_val_loss")
    elif running:
        current_step = min(STEPS_PER_SEED, int(running.get("etime_sec", 0) / SEC_PER_STEP))
        current_step_pct = round(100.0 * current_step / STEPS_PER_SEED, 1)

    rows = []
    done_count = 0
    for meta in EXPERIMENTS:
        eid = meta["id"]
        entry = completed.get(eid)
        seeds = seed_results.get(eid, [])

        if entry:
            status = "done"
            done_count += 1
            r = entry["result"]
            loss_mean = r["best_val_loss_mean"]
            loss_std = r["best_val_loss_std"]
            per_seed = [
                {
                    "seed": s["seed"],
                    "best_val_loss": s["best_val_loss"],
                    "wall_time_s": s["wall_time_s"],
                }
                for s in r.get("per_seed", [])
            ]
            prog_pct = 100.0
            step_info = f"1000/1000 × {SEEDS_TOTAL} seeds"
        elif eid == current_id:
            status = "running"
            loss_mean = loss_std = None
            if seeds:
                vals = [s["best_val_loss"] for s in seeds]
                loss_mean = sum(vals) / len(vals)
            per_seed = [
                {
                    "seed": s.get("seed"),
                    "best_val_loss": s.get("best_val_loss"),
                    "wall_time_s": s.get("wall_time_s"),
                }
                for s in seeds
            ]
            cs = running.get("seed") if running else None
            prog_pct = compute_progress_pct("running", len(per_seed), cs, current_step)
            step_info = f"seed {cs}: step {current_step}/{STEPS_PER_SEED} ({current_step_pct}%)"
            if live_val_loss is not None and eid == current_id:
                loss_mean = live_val_loss
        else:
            status = "pending"
            loss_mean = loss_std = None
            per_seed = []
            prog_pct = 0.0
            step_info = "aguardando"

        delta = (baseline_loss - loss_mean) if (loss_mean is not None and baseline_loss is not None) else None

        rows.append({
            **meta,
            "status": status,
            "progress_pct": prog_pct,
            "step_info": step_info,
            "best_val_loss_mean": loss_mean,
            "best_val_loss_std": loss_std,
            "delta_vs_baseline": delta,
            "per_seed": per_seed,
            "n_seeds_done": len(per_seed),
            "n_seeds_total": SEEDS_TOTAL,
        })

    global_pct = round(sum(r["progress_pct"] for r in rows) / len(rows), 1)
    batch_running = subprocess.run(
        ["pgrep", "-f", "atlas_batch_lm_100m"], capture_output=True, text=True
    ).returncode == 0

    start = batch_start_time()
    eta = estimate_eta_hours(global_pct, start)

    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "tinyshakespeare (char-level)",
        "params_m_baseline": 99.53,
        "batch_running": batch_running,
        "global_progress_pct": global_pct,
        "progress": {"done": done_count, "total": len(EXPERIMENTS)},
        "baseline_loss": baseline_loss,
        "current_experiment_id": current_id,
        "current_seed": running.get("seed") if running else None,
        "current_step": current_step,
        "current_step_pct": current_step_pct,
        "live_val_loss": live_val_loss,
        "eta_note": f"~5 h/seed CPU · refresh a cada 10s (fallback 60s) · {eta}",
        "experiments": rows,
    }


HTML_SHELL = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="60"/>
  <title>ATLAS LM 100M — TokenBlend vs Baseline</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #0f1117; color: #e6edf3; }
    body { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .sub { color: #8b949e; margin-bottom: 16px; }
    .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; min-width: 140px; }
    .card b { display: block; font-size: 1.3rem; }
    .card small { color: #8b949e; font-size: 0.75rem; }
    .global-bar { height: 14px; background: #21262d; border-radius: 7px; overflow: hidden; margin: 12px 0 20px; }
    .global-bar > div { height: 100%; background: linear-gradient(90deg,#238636,#3fb950); transition: width .5s; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { border-bottom: 1px solid #30363d; padding: 8px 6px; text-align: left; vertical-align: middle; }
    th { color: #8b949e; }
    tr.running { background: #1a2e1a; }
    tr.done td:first-child + td { text-decoration: line-through; color: #8b949e; }
    .pbar { width: 100px; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 6px; }
    .pbar > div { height: 100%; background: #388bfd; transition: width .5s; }
    .pbar.done > div { background: #238636; }
    .badge { padding: 2px 7px; border-radius: 10px; font-size: 0.7rem; text-transform: uppercase; }
    .badge.done { background: #238636; }
    .badge.running { background: #9e6a03; }
    .badge.pending { background: #30363d; color: #8b949e; }
    .badge.novel { background: #1f3d5c; color: #79c0ff; font-size: 0.65rem; margin-left: 4px; }
    .badge.baseline { background: #30363d; color: #8b949e; font-size: 0.65rem; margin-left: 4px; }
    #clock { color: #3fb950; font-weight: 600; }
    .err { color: #f85149; }
    .better { color: #3fb950; }
    .worse { color: #f85149; }
  </style>
</head>
<body>
  <h1>ATLAS — LM 100M TokenBlend vs Baseline</h1>
  <p class="sub">tinyshakespeare char-level · ~100M params · <span id="clock">carregando…</span></p>
  <div class="cards" id="cards"></div>
  <div class="global-bar"><div id="global-fill" style="width:0%"></div></div>
  <p id="global-label" style="margin-top:-12px;color:#8b949e;font-size:0.85rem"></p>
  <table>
    <thead><tr>
      <th>#</th><th>ID</th><th>Mixer</th><th>Params</th><th>Progresso</th><th>Status</th>
      <th>Val Loss</th><th>Δ base</th><th>Detalhe</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <p class="sub" id="footer"></p>
  <script>
    function loss(x) { return (x == null) ? '—' : Number(x).toFixed(4); }
    function delta(x) {
      if (x == null) return '—';
      const cls = x >= 0 ? 'better' : 'worse';
      const sign = x >= 0 ? '+' : '';
      return `<span class="${cls}">${sign}${Number(x).toFixed(4)}</span>`;
    }
    function render(d) {
      document.getElementById('clock').textContent = 'atualizado ' + d.updated_utc.replace('T',' ').slice(0,19) + ' UTC';
      const cur = d.current_experiment_id || '—';
      document.getElementById('cards').innerHTML = `
        <div class="card">Global<b>${d.global_progress_pct}%</b></div>
        <div class="card">Concluídos<b>${d.progress.done}/${d.progress.total}</b></div>
        <div class="card">Rodando<b>${cur}</b><small>step ${d.current_step||0}/${1000} · seed ${d.current_seed||'—'}</small></div>
        <div class="card">Batch<b>${d.batch_running?'ATIVO':'PARADO'}</b></div>`;
      document.getElementById('global-fill').style.width = d.global_progress_pct + '%';
      document.getElementById('global-label').textContent = `Progresso global do batch: ${d.global_progress_pct}%`;
      document.getElementById('footer').textContent = d.eta_note + ' · métrica: menor val_loss = melhor · Δ positivo = melhor que baseline';
      const tb = document.getElementById('tbody');
      tb.innerHTML = d.experiments.map((e,i) => {
        const cls = e.status;
        const pcls = e.status==='done'?'pbar done':'pbar';
        const kindBadge = e.kind==='NOVEL' ? '<span class="badge novel">NOVEL</span>' : (e.kind==='BASELINE' ? '<span class="badge baseline">BASELINE</span>' : '');
        const vl = e.best_val_loss_mean!=null ? loss(e.best_val_loss_mean) : (d.live_val_loss!=null && e.status==='running' ? loss(d.live_val_loss)+' (live)' : '—');
        const seeds = e.per_seed.map(s => `s${s.seed}:${Number(s.best_val_loss).toFixed(3)}`).join(', ');
        return `<tr class="${cls}">
          <td>${i+1}</td><td><code>${e.id}</code>${kindBadge}</td><td>${e.label}</td>
          <td>${e.params_m}M</td>
          <td><span class="${pcls}"><div style="width:${e.progress_pct}%"></div></span><strong>${e.progress_pct}%</strong></td>
          <td><span class="badge ${cls}">${cls}</span></td>
          <td><strong>${vl}</strong></td>
          <td>${delta(e.delta_vs_baseline)}</td>
          <td><small>${e.step_info}${seeds?', '+seeds:''}</small></td></tr>`;
      }).join('');
    }
    async function refresh() {
      try {
        const r = await fetch('lm_100m_status.json?ts=' + Date.now());
        if (!r.ok) throw new Error(r.status);
        render(await r.json());
      } catch (e) {
        document.getElementById('clock').innerHTML = '<span class="err">erro ao carregar JSON — sirva com: python3 -m http.server 8766</span>';
      }
    }
    refresh();
    setInterval(refresh, 10000);
  </script>
</body>
</html>"""


def render_md(status: dict) -> str:
    lines = [
        "# ATLAS — Tracker LM 100M (TokenBlend vs Baseline)",
        "",
        f"**Atualizado:** {status['updated_utc']}",
        f"**Progresso global:** {status['global_progress_pct']}%",
        f"**Concluídos:** {status['progress']['done']}/{status['progress']['total']}",
        f"**Rodando:** {status['current_experiment_id'] or '—'} seed {status['current_seed'] or '—'} step {status['current_step']}/{STEPS_PER_SEED}",
        "",
        "| # | ID | Mixer | % | Status | Val Loss | Δ base |",
        "|---|-----|-------|---|--------|----------|--------|",
    ]
    for i, e in enumerate(status["experiments"], 1):
        vl = fmt_loss(e["best_val_loss_mean"])
        lines.append(
            f"| {i} | {e['id']} | {e['label']} | **{e['progress_pct']}%** | {e['status']} | {vl} | {fmt_delta_loss(e['delta_vs_baseline'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    status = build_status()
    STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DASHBOARD_HTML.write_text(HTML_SHELL, encoding="utf-8")
    TRACKER_MD.write_text(render_md(status), encoding="utf-8")
    print(
        f"global={status['global_progress_pct']}% done={status['progress']['done']}/{status['progress']['total']} "
        f"step={status['current_step']} exp={status['current_experiment_id']}"
    )


if __name__ == "__main__":
    main()
