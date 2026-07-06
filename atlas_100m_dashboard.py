#!/usr/bin/env python3
"""Gera painel desktop + JSON de status do batch 100M (atualização contínua)."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATUS_JSON = ROOT / "promissora_100m_status.json"
DASHBOARD_HTML = ROOT / "promissora_100m_dashboard.html"
TRACKER_MD = ROOT / "promissora_100m_tracker.md"
LOG_PATH = ROOT / "experiments_log.jsonl"
BATCH_LOG = ROOT / "batch_100m.log"
RESULTS_DIR = ROOT / "results" / "100m"
LIVE_PROGRESS = RESULTS_DIR / "_live_progress.json"
PARAMS_M = 107.51
STEPS_PER_SEED = 1000
SEEDS_TOTAL = 3
SEC_PER_STEP = 2.4  # calibrado CPU 107M

EXPERIMENTS = [
    {"id": "exp-100m-001", "orig": "exp-007", "label": "warmup+cosine", "kind": "RECOMB"},
    {"id": "exp-100m-002", "orig": "exp-011", "label": "warmup+cosine", "kind": "RECOMB"},
    {"id": "exp-100m-003", "orig": "exp-016", "label": "local_blend", "kind": "NOVEL"},
    {"id": "exp-100m-004", "orig": "exp-018", "label": "local_blend", "kind": "NOVEL"},
    {"id": "exp-100m-005", "orig": "exp-021", "label": "local_blend_k5", "kind": "NOVEL"},
    {"id": "exp-100m-006", "orig": "exp-022", "label": "laplacian_blend", "kind": "NOVEL"},
    {"id": "exp-100m-007", "orig": "exp-024", "label": "blend+warmup+cosine", "kind": "RECOMB"},
    {"id": "exp-100m-008", "orig": "exp-025", "label": "blend+label_smooth", "kind": "RECOMB"},
    {"id": "exp-100m-009", "orig": "exp-026", "label": "local_blend_k5", "kind": "NOVEL"},
    {"id": "exp-100m-010", "orig": "exp-027", "label": "blend+LS (ex-campeão)", "kind": "RECOMB"},
    {"id": "exp-100m-011", "orig": "exp-028", "label": "laplacian_blend", "kind": "NOVEL"},
    {"id": "exp-100m-012", "orig": "exp-032", "label": "k5+label_smooth", "kind": "RECOMB"},
    {"id": "exp-100m-baseline", "orig": "—", "label": "baseline (referência)", "kind": "BASELINE"},
]


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
        if e.get("id", "").startswith("exp-100m"):
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


def detect_running() -> dict | None:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.splitlines():
        if "train_baseline.py" not in line or "width_mult" not in line or "hidden_dim 2048" not in line:
            continue
        if "grep" in line:
            continue
        m_seed = re.search(r"--seed\s+(\d+)", line)
        seed = int(m_seed.group(1)) if m_seed else None
        parts = line.split()
        pid = parts[1] if len(parts) > 1 else None
        etime_sec = 0.0
        if pid:
            try:
                et = subprocess.check_output(["ps", "-p", pid, "-o", "etime="], text=True).strip()
                if "-" in et:
                    d, t = et.split("-", 1)
                    hh, mm, ss = (t.split(":") + ["0"])[:3]
                    etime_sec = int(d) * 86400 + int(hh) * 3600 + int(mm) * 60 + int(ss)
                elif et.count(":") == 2:
                    hh, mm, ss = et.split(":")
                    etime_sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
                elif et.count(":") == 1:
                    mm, ss = et.split(":")
                    etime_sec = int(mm) * 60 + int(ss)
                else:
                    etime_sec = int(et)
            except (subprocess.CalledProcessError, ValueError):
                pass
        return {"seed": seed, "pid": pid, "etime_sec": etime_sec}
    return None


def detect_current_exp_id() -> str | None:
    if not BATCH_LOG.exists():
        return None
    text = BATCH_LOG.read_text(encoding="utf-8")
    headers = re.findall(r"=== (exp-100m-[^\s|]+)", text)
    if not headers:
        return None
    completed = load_completed()
    for hid in reversed(headers):
        if hid not in completed:
            return hid
    return headers[-1] if headers else None


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
    # running
    si = seed_index(current_seed)
    step_frac = min(1.0, current_step / max(1, total_steps))
    return round(100.0 * (si + step_frac) / SEEDS_TOTAL, 1)


def fmt_pct_acc(x: float | None) -> str:
    return f"{x * 100:.2f}%" if x is not None else "—"


def fmt_delta(x: float | None) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x * 100:.2f} pp"


def build_status() -> dict:
    completed = load_completed()
    seed_results = load_seed_results()
    running = detect_running()
    current_id = detect_current_exp_id()
    live = load_live_progress()

    baseline_acc = None
    if "exp-100m-baseline" in completed:
        baseline_acc = completed["exp-100m-baseline"]["result"]["best_val_acc_mean"]

    current_step = 0
    current_step_pct = 0.0
    live_val_acc = None
    if live and running and live.get("seed") == running.get("seed"):
        current_step = int(live.get("step", 0))
        current_step_pct = float(live.get("step_pct", 0))
        live_val_acc = live.get("best_val_acc")
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
            acc_mean = r["best_val_acc_mean"]
            acc_std = r["best_val_acc_std"]
            per_seed = [
                {"seed": s["seed"], "best_val_acc": s["best_val_acc"], "wall_time_s": s["wall_time_s"]}
                for s in r.get("per_seed", [])
            ]
            prog_pct = 100.0
            step_info = "1000/1000 × 3 seeds"
        elif eid == current_id:
            status = "running"
            acc_mean = acc_std = None
            if seeds:
                accs = [s["best_val_acc"] for s in seeds]
                acc_mean = sum(accs) / len(accs)
            per_seed = [
                {"seed": s.get("seed"), "best_val_acc": s.get("best_val_acc"), "wall_time_s": s.get("wall_time_s")}
                for s in seeds
            ]
            cs = running.get("seed") if running else None
            prog_pct = compute_progress_pct("running", len(per_seed), cs, current_step)
            step_info = f"seed {cs}: step {current_step}/{STEPS_PER_SEED} ({current_step_pct}%)"
            if live_val_acc is not None and eid == current_id:
                acc_mean = live_val_acc
        else:
            status = "pending"
            acc_mean = acc_std = None
            per_seed = []
            prog_pct = 0.0
            step_info = "aguardando"

        delta = (acc_mean - baseline_acc) if (acc_mean is not None and baseline_acc is not None) else None

        rows.append({
            **meta,
            "status": status,
            "progress_pct": prog_pct,
            "step_info": step_info,
            "best_val_acc_mean": acc_mean,
            "best_val_acc_std": acc_std,
            "delta_vs_baseline": delta,
            "per_seed": per_seed,
            "n_seeds_done": len(per_seed),
            "n_seeds_total": SEEDS_TOTAL,
        })

    global_pct = round(sum(r["progress_pct"] for r in rows) / len(rows), 1)

    batch_running = subprocess.run(
        ["pgrep", "-f", "atlas_batch_100m.sh"], capture_output=True, text=True
    ).returncode == 0

    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "params_m": PARAMS_M,
        "batch_running": batch_running,
        "global_progress_pct": global_pct,
        "progress": {"done": done_count, "total": len(EXPERIMENTS)},
        "baseline_acc": baseline_acc,
        "current_experiment_id": current_id,
        "current_seed": running.get("seed") if running else None,
        "current_step": current_step,
        "current_step_pct": current_step_pct,
        "live_val_acc": live_val_acc,
        "eta_note": "~35–40 min/seed · refresh automático a cada 5s",
        "experiments": rows,
    }


HTML_SHELL = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <title>ATLAS 100M — 12 Promissoras</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #0f1117; color: #e6edf3; }
    body { max-width: 1280px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 1.5rem; margin-bottom: 4px; }
    .sub { color: #8b949e; margin-bottom: 16px; }
    .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; min-width: 140px; }
    .card b { display: block; font-size: 1.3rem; }
    .global-bar { height: 14px; background: #21262d; border-radius: 7px; overflow: hidden; margin: 12px 0 20px; }
    .global-bar > div { height: 100%; background: linear-gradient(90deg,#238636,#3fb950); transition: width .5s; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th, td { border-bottom: 1px solid #30363d; padding: 8px 6px; text-align: left; vertical-align: middle; }
    th { color: #8b949e; }
    tr.running { background: #1a2e1a; }
    .pbar { width: 100px; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 6px; }
    .pbar > div { height: 100%; background: #388bfd; transition: width .5s; }
    .pbar.done > div { background: #238636; }
    .badge { padding: 2px 7px; border-radius: 10px; font-size: 0.7rem; text-transform: uppercase; }
    .badge.done { background: #238636; }
    .badge.running { background: #9e6a03; }
    .badge.pending { background: #30363d; color: #8b949e; }
    #clock { color: #3fb950; font-weight: 600; }
    .err { color: #f85149; }
  </style>
</head>
<body>
  <h1>ATLAS — 12 Promissoras @ ~100M</h1>
  <p class="sub">FashionMNIST · 107.5M params · <span id="clock">carregando…</span></p>
  <div class="cards" id="cards"></div>
  <div class="global-bar"><div id="global-fill" style="width:0%"></div></div>
  <p id="global-label" style="margin-top:-12px;color:#8b949e;font-size:0.85rem"></p>
  <table>
    <thead><tr>
      <th>#</th><th>ID</th><th>Mecanismo</th><th>Progresso</th><th>Status</th>
      <th>Val Acc</th><th>Δ base</th><th>Detalhe</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>
  <p class="sub" id="footer"></p>
  <script>
    function pct(x) { return (x == null) ? '—' : (x*100).toFixed(2)+'%'; }
    function delta(x) {
      if (x == null) return '—';
      return (x>=0?'+':'') + (x*100).toFixed(2) + ' pp';
    }
    function render(d) {
      document.getElementById('clock').textContent = 'atualizado ' + d.updated_utc.replace('T',' ').slice(0,19) + ' UTC';
      document.getElementById('cards').innerHTML = `
        <div class="card">Global<b>${d.global_progress_pct}%</b></div>
        <div class="card">Concluídos<b>${d.progress.done}/${d.progress.total}</b></div>
        <div class="card">Rodando<b>${d.current_experiment_id||'—'}</b>step ${d.current_step||0}/${1000}</div>
        <div class="card">Batch<b>${d.batch_running?'ATIVO':'PARADO'}</b></div>`;
      document.getElementById('global-fill').style.width = d.global_progress_pct + '%';
      document.getElementById('global-label').textContent = `Progresso global do batch: ${d.global_progress_pct}%`;
      document.getElementById('footer').textContent = d.eta_note;
      const tb = document.getElementById('tbody');
      tb.innerHTML = d.experiments.map((e,i) => {
        const cls = e.status;
        const pcls = e.status==='done'?'pbar done':'pbar';
        const acc = e.best_val_acc_mean!=null ? pct(e.best_val_acc_mean) : (d.live_val_acc!=null && e.status==='running' ? pct(d.live_val_acc)+' (live)' : '—');
        const seeds = e.per_seed.map(s => `s${s.seed}:${(s.best_val_acc*100).toFixed(1)}%`).join(', ');
        return `<tr class="${cls}">
          <td>${i+1}</td><td><code>${e.id}</code></td><td>${e.label}</td>
          <td><span class="${pcls}"><div style="width:${e.progress_pct}%"></div></span><strong>${e.progress_pct}%</strong></td>
          <td><span class="badge ${cls}">${cls}</span></td>
          <td><strong>${acc}</strong></td>
          <td>${delta(e.delta_vs_baseline)}</td>
          <td><small>${e.step_info}${seeds?', '+seeds:''}</small></td></tr>`;
      }).join('');
    }
    async function refresh() {
      try {
        const r = await fetch('promissora_100m_status.json?ts=' + Date.now());
        if (!r.ok) throw new Error(r.status);
        render(await r.json());
      } catch (e) {
        document.getElementById('clock').innerHTML = '<span class="err">erro ao carregar JSON — use http://localhost:8765/promissora_100m_dashboard.html</span>';
      }
    }
    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>"""


def render_md(status: dict) -> str:
    lines = [
        "# ATLAS — Tracker 100M (12 Promissoras)",
        "",
        f"**Atualizado:** {status['updated_utc']}",
        f"**Progresso global:** {status['global_progress_pct']}%",
        f"**Concluídos:** {status['progress']['done']}/{status['progress']['total']}",
        f"**Rodando:** {status['current_experiment_id'] or '—'} seed {status['current_seed'] or '—'} step {status['current_step']}/{STEPS_PER_SEED}",
        "",
        "| # | ID | Mecanismo | % | Status | Val Acc | Δ base |",
        "|---|-----|-----------|---|--------|---------|--------|",
    ]
    for i, e in enumerate(status["experiments"], 1):
        acc = fmt_pct_acc(e["best_val_acc_mean"])
        lines.append(
            f"| {i} | {e['id']} | {e['label']} | **{e['progress_pct']}%** | {e['status']} | {acc} | {fmt_delta(e['delta_vs_baseline'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    status = build_status()
    STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not DASHBOARD_HTML.exists() or "setInterval(refresh" not in DASHBOARD_HTML.read_text(encoding="utf-8"):
        DASHBOARD_HTML.write_text(HTML_SHELL, encoding="utf-8")
    TRACKER_MD.write_text(render_md(status), encoding="utf-8")
    print(f"global={status['global_progress_pct']}% done={status['progress']['done']}/{status['progress']['total']} step={status['current_step']}")


if __name__ == "__main__":
    main()
