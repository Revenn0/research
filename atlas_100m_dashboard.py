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
PARAMS_M = 107.51

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
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        eid = data.get("experiment_id") or p.stem.rsplit("_seed", 1)[0]
        by_exp.setdefault(eid, []).append(data)
    for eid in by_exp:
        by_exp[eid].sort(key=lambda x: x.get("seed", 0))
    return by_exp


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
        m = re.search(r"--seed\s+(\d+)", line)
        seed = int(m.group(1)) if m else None
        # tempo CPU acumulado
        parts = line.split()
        cpu_min = 0.0
        if len(parts) > 9:
            t = parts[9]
            if ":" in t:
                mm, ss = t.split(":")
                cpu_min = int(mm) + int(ss) / 60
        return {"seed": seed, "cpu_min": round(cpu_min, 1), "pid": parts[1] if len(parts) > 1 else None}
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


def fmt_pct(x: float | None) -> str:
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

    baseline_acc = None
    if "exp-100m-baseline" in completed:
        baseline_acc = completed["exp-100m-baseline"]["result"]["best_val_acc_mean"]

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
        elif eid == current_id and running:
            status = "running"
            acc_mean = acc_std = None
            if seeds:
                accs = [s["best_val_acc"] for s in seeds]
                acc_mean = sum(accs) / len(accs)
            per_seed = [
                {
                    "seed": s.get("seed"),
                    "best_val_acc": s.get("best_val_acc"),
                    "wall_time_s": s.get("wall_time_s"),
                }
                for s in seeds
            ]
        elif eid == current_id:
            status = "running"
            acc_mean = acc_std = None
            per_seed = []
        else:
            status = "pending"
            acc_mean = acc_std = None
            per_seed = []

        delta = (acc_mean - baseline_acc) if (acc_mean is not None and baseline_acc is not None) else None

        rows.append({
            **meta,
            "status": status,
            "best_val_acc_mean": acc_mean,
            "best_val_acc_std": acc_std,
            "delta_vs_baseline": delta,
            "per_seed": per_seed,
            "n_seeds_done": len(per_seed),
            "n_seeds_total": 3,
        })

    batch_running = subprocess.run(
        ["pgrep", "-f", "atlas_batch_100m.sh"], capture_output=True, text=True
    ).returncode == 0

    return {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "params_m": PARAMS_M,
        "batch_running": batch_running,
        "progress": {"done": done_count, "total": len(EXPERIMENTS)},
        "baseline_acc": baseline_acc,
        "current_experiment_id": current_id,
        "current_seed": running.get("seed") if running else None,
        "current_cpu_min": running.get("cpu_min") if running else None,
        "eta_note": "~35–40 min/seed, ~2 h/config, ~22–26 h total (CPU)",
        "experiments": rows,
    }


def render_html(status: dict) -> str:
    data_json = json.dumps(status, ensure_ascii=False)
    done = status["progress"]["done"]
    total = status["progress"]["total"]
    pct = int(100 * done / total) if total else 0
    running_label = status["current_experiment_id"] or "—"
    seed_label = status["current_seed"] or "—"

    rows_html = []
    for i, e in enumerate(status["experiments"], 1):
        st = e["status"]
        badge_cls = {"done": "done", "running": "running", "pending": "pending"}[st]
        acc = fmt_pct(e["best_val_acc_mean"]) if e["best_val_acc_mean"] is not None else "—"
        std = f"± {e['best_val_acc_std'] * 100:.2f}" if e.get("best_val_acc_std") else ""
        delta = fmt_delta(e["delta_vs_baseline"])
        seeds_txt = f"{e['n_seeds_done']}/{e['n_seeds_total']}"
        if e["per_seed"]:
            seed_detail = ", ".join(f"s{s['seed']}:{s['best_val_acc']*100:.1f}%" for s in e["per_seed"])
        else:
            seed_detail = ""
        rows_html.append(
            f"<tr class='{badge_cls}'>"
            f"<td>{i}</td><td><code>{e['id']}</code></td><td>{e['orig']}</td>"
            f"<td>{e['label']}</td><td><span class='badge {badge_cls}'>{st}</span></td>"
            f"<td><strong>{acc}</strong> <small>{std}</small></td>"
            f"<td>{delta}</td><td>{seeds_txt}</td>"
            f"<td><small>{seed_detail}</small></td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta http-equiv="refresh" content="30"/>
  <title>ATLAS 100M — 12 Promissoras</title>
  <style>
    :root {{ font-family: system-ui, sans-serif; background: #0f1117; color: #e6edf3; }}
    body {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
    .sub {{ color: #8b949e; margin-bottom: 20px; }}
    .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 18px; min-width: 160px; }}
    .card b {{ display: block; font-size: 1.4rem; }}
    .bar {{ height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; margin: 8px 0 20px; }}
    .bar > div {{ height: 100%; background: #238636; width: {pct}%; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
    th, td {{ border-bottom: 1px solid #30363d; padding: 10px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #8b949e; font-weight: 600; }}
    tr.running {{ background: #1f2a1f; }}
    tr.done {{ background: #161b22; }}
    .badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; text-transform: uppercase; }}
    .badge.done {{ background: #238636; }}
    .badge.running {{ background: #9e6a03; color: #fff; }}
    .badge.pending {{ background: #30363d; color: #8b949e; }}
    code {{ font-size: 0.8rem; }}
    footer {{ margin-top: 20px; color: #8b949e; font-size: 0.8rem; }}
  </style>
</head>
<body>
  <h1>ATLAS — 12 Promissoras @ ~100M</h1>
  <p class="sub">FashionMNIST · {PARAMS_M:.1f}M parâmetros · atualiza a cada 30s</p>
  <div class="cards">
    <div class="card">Progresso<b>{done}/{total}</b></div>
    <div class="card">Rodando agora<b>{running_label}</b>seed {seed_label}</div>
    <div class="card">Baseline<b>{fmt_pct(status['baseline_acc'])}</b></div>
    <div class="card">Batch ativo<b>{'SIM' if status['batch_running'] else 'NÃO'}</b></div>
  </div>
  <div class="bar"><div></div></div>
  <table>
    <thead><tr>
      <th>#</th><th>ID</th><th>Origem</th><th>Mecanismo</th><th>Status</th>
      <th>Val Acc</th><th>Δ baseline</th><th>Seeds</th><th>Detalhe seeds</th>
    </tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
  <footer>Última atualização UTC: {status['updated_utc']} · {status['eta_note']}</footer>
  <script>const STATUS = {data_json};</script>
</body>
</html>"""


def render_md(status: dict) -> str:
    lines = [
        "# ATLAS — Tracker 100M (12 Promissoras)",
        "",
        f"**Atualizado:** {status['updated_utc']}",
        f"**Parâmetros:** {PARAMS_M:.1f}M",
        f"**Progresso:** {status['progress']['done']}/{status['progress']['total']}",
        f"**Batch rodando:** {'sim' if status['batch_running'] else 'não'}",
        f"**Experimento atual:** {status['current_experiment_id'] or '—'} (seed {status['current_seed'] or '—'})",
        "",
        "| # | ID | Origem | Mecanismo | Status | Val Acc | Δ base | Seeds |",
        "|---|-----|--------|-----------|--------|---------|--------|-------|",
    ]
    for i, e in enumerate(status["experiments"], 1):
        acc = fmt_pct(e["best_val_acc_mean"])
        delta = fmt_delta(e["delta_vs_baseline"])
        seeds = f"{e['n_seeds_done']}/{e['n_seeds_total']}"
        lines.append(
            f"| {i} | {e['id']} | {e['orig']} | {e['label']} | {e['status']} | {acc} | {delta} | {seeds} |"
        )
    if status["baseline_acc"] is not None:
        lines.extend(["", f"**Baseline @100M:** {fmt_pct(status['baseline_acc'])}"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    status = build_status()
    STATUS_JSON.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DASHBOARD_HTML.write_text(render_html(status), encoding="utf-8")
    TRACKER_MD.write_text(render_md(status), encoding="utf-8")
    print(f"Dashboard: {DASHBOARD_HTML}")
    print(f"Status:    {STATUS_JSON}")
    print(f"Progress:  {status['progress']['done']}/{status['progress']['total']}")


if __name__ == "__main__":
    main()
