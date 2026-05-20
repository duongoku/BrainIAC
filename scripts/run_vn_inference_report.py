#!/usr/bin/env python3
"""Run VN AD/CN inference for linear probing and few-shot seed sweeps."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


REPO_ROOT = Path("/home2/duongoku/BrainIAC_ADCN/BrainIAC")
CHECKPOINT_DIRS = [
    REPO_ROOT / "src/results/vn_checkpoints",
    Path("/mnt/data_lab513/duongoku/brainiac_checkpoints"),
]
RESULT_DIR = REPO_ROOT / "src/results/vn_inference_reports"
TEST_CSV = REPO_ROOT / "src/data/csvs/vn_test.csv"
ROOT_DIR = REPO_ROOT / "src/data/images"
SIMCLR_CKPT = REPO_ROOT / "src/checkpoints/BrainIAC.ckpt"

LINEAR_RE = re.compile(
    r"vn_linear_probing_model-epoch-epoch=(?P<epoch>\d+)-val_auc=(?P<val_auc>[0-9.]+)\.ckpt$"
)
FEWSHOT_RE = re.compile(
    r"vn_fewshot_k_(?P<k>[15])(?:_seedplus_(?P<seedplus>\d+))?"
    r"_model-epoch-epoch=(?P<epoch>\d+)-val_auc=(?P<val_auc>[0-9.]+)\.ckpt$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VN AD/CN inference and create per-method HTML report."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint-dir", action="append", type=Path, default=[])
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--root-dir", type=Path, default=ROOT_DIR)
    parser.add_argument("--simclr-ckpt", type=Path, default=SIMCLR_CKPT)
    parser.add_argument("--methods", nargs="*", choices=["linear", "k1", "k5"], default=["linear", "k1", "k5"])
    parser.add_argument("--include-legacy-fewshot", action="store_true")
    parser.add_argument("--linear-checkpoint", type=Path, default=None)
    parser.add_argument("--k1-checkpoint", type=Path, default=None)
    parser.add_argument("--k5-checkpoint", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def add_src_to_path(repo_root: Path) -> None:
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def checkpoint_files(checkpoint_dirs: Iterable[Path]) -> Iterable[Path]:
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.exists():
            continue
        yield from checkpoint_dir.glob("*.ckpt")


def parse_checkpoint(path: Path, include_legacy_fewshot: bool) -> Optional[dict]:
    name = path.name
    linear = LINEAR_RE.match(name)
    if linear:
        return {
            "method": "linear",
            "method_label": "Linear probing",
            "run_id": "linear",
            "k": None,
            "seedplus": None,
            "epoch": int(linear.group("epoch")),
            "val_auc": float(linear.group("val_auc")),
            "checkpoint_path": str(path),
            "mtime": path.stat().st_mtime,
        }

    fewshot = FEWSHOT_RE.match(name)
    if not fewshot:
        return None

    k = int(fewshot.group("k"))
    seedplus_text = fewshot.group("seedplus")
    if seedplus_text is None and not include_legacy_fewshot:
        return None
    seedplus = int(seedplus_text) if seedplus_text is not None else None
    method = f"k{k}"
    seed_label = f"seedplus_{seedplus}" if seedplus is not None else "legacy"
    return {
        "method": method,
        "method_label": f"Few-shot k={k}",
        "run_id": f"{method}_{seed_label}",
        "k": k,
        "seedplus": seedplus,
        "epoch": int(fewshot.group("epoch")),
        "val_auc": float(fewshot.group("val_auc")),
        "checkpoint_path": str(path),
        "mtime": path.stat().st_mtime,
    }


def select_best_checkpoints(
    checkpoint_dirs: Iterable[Path],
    methods: set,
    include_legacy_fewshot: bool,
) -> List[dict]:
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for path in checkpoint_files(checkpoint_dirs):
        parsed = parse_checkpoint(path, include_legacy_fewshot=include_legacy_fewshot)
        if parsed is None or parsed["method"] not in methods:
            continue
        key = (parsed["method"], parsed["run_id"])
        groups.setdefault(key, []).append(parsed)

    selected = []
    for group in groups.values():
        selected.append(max(group, key=lambda item: (item["val_auc"], item["mtime"])))
    return sorted(selected, key=lambda item: (item["method"], item["seedplus"] if item["seedplus"] is not None else -1))


def manual_target(method: str, checkpoint_path: Path) -> dict:
    labels = {
        "linear": ("Linear probing", None),
        "k1": ("Few-shot k=1", 1),
        "k5": ("Few-shot k=5", 5),
    }
    method_label, k = labels[method]
    return {
        "method": method,
        "method_label": method_label,
        "run_id": f"{method}_manual",
        "k": k,
        "seedplus": None,
        "epoch": None,
        "val_auc": None,
        "checkpoint_path": str(checkpoint_path),
        "mtime": checkpoint_path.stat().st_mtime if checkpoint_path.exists() else 0,
    }


def append_manual_targets(selected: List[dict], args: argparse.Namespace) -> List[dict]:
    manual_paths = {
        "linear": args.linear_checkpoint,
        "k1": args.k1_checkpoint,
        "k5": args.k5_checkpoint,
    }
    existing_methods = {item["method"] for item in selected}
    for method, path in manual_paths.items():
        if path is None or method in existing_methods:
            continue
        selected.append(manual_target(method, path.resolve()))
    return sorted(selected, key=lambda item: (item["method"], item["seedplus"] if item["seedplus"] is not None else -1))


def metric_value(metrics: dict, name: str) -> Optional[float]:
    value = metrics.get(name)
    return None if value is None else float(value)


def summarize(rows: List[dict]) -> List[dict]:
    summaries = []
    for method in ["linear", "k1", "k5"]:
        method_rows = [row for row in rows if row["method"] == method and "error" not in row]
        if not method_rows:
            continue
        summary = {
            "method": method,
            "method_label": method_rows[0]["method_label"],
            "n_runs": len(method_rows),
        }
        for metric in ["auc", "f1_score", "accuracy", "precision", "recall"]:
            values = [float(row[metric]) for row in method_rows if row.get(metric) is not None]
            if values:
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
                summary[f"{metric}_min"] = min(values)
                summary[f"{metric}_max"] = max(values)
            else:
                summary[f"{metric}_mean"] = None
                summary[f"{metric}_std"] = None
                summary[f"{metric}_min"] = None
                summary[f"{metric}_max"] = None
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_html(path: Path, rows: List[dict], summaries: List[dict], generated_at: str, test_csv: Path) -> None:
    payload = json.dumps({"rows": rows, "summaries": summaries}, indent=2)
    escaped_payload = html.escape(payload)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VN AD/CN Inference Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    h1 {{ margin: 0 0 4px; font-size: 24px; }}
    .meta {{ color: #52606d; margin-bottom: 20px; }}
    .controls {{ display: flex; gap: 8px; margin: 16px 0; flex-wrap: wrap; }}
    button {{ border: 1px solid #bcccdc; background: #fff; padding: 8px 10px; cursor: pointer; border-radius: 4px; }}
    button.active {{ background: #102a43; color: #fff; border-color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; cursor: pointer; position: sticky; top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; }}
    .metric {{ font-size: 22px; font-weight: 700; }}
    .sub {{ color: #52606d; font-size: 13px; }}
    .bar {{ height: 8px; background: #d9e2ec; border-radius: 4px; overflow: hidden; margin-top: 6px; }}
    .bar span {{ display: block; height: 8px; background: #0b7285; }}
    code {{ font-size: 12px; }}
  </style>
</head>
<body>
  <h1>VN AD/CN Inference Report</h1>
  <div class="meta">Generated {html.escape(generated_at)}. Test CSV: <code>{html.escape(str(test_csv))}</code></div>
  <div class="controls" id="filters"></div>
  <div class="cards" id="summary"></div>
  <h2>Runs</h2>
  <table id="runs"></table>
  <script type="application/json" id="payload">{escaped_payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    let active = 'all';
    let sortKey = 'method';
    let sortAsc = true;
    const methods = [['all', 'All'], ['linear', 'Linear probing'], ['k1', 'Few-shot k=1'], ['k5', 'Few-shot k=5']];
    function number(v) {{ return typeof v === 'number' && Number.isFinite(v); }}
    function cell(v) {{ return number(v) ? v.toFixed(4) : (v ?? ''); }}
    function renderFilters() {{
      const el = document.getElementById('filters');
      el.innerHTML = methods.map(([key, label]) => `<button class="${{active === key ? 'active' : ''}}" onclick="active='${{key}}'; render();">${{label}}</button>`).join('');
    }}
    function renderSummary() {{
      const summaries = payload.summaries.filter(s => active === 'all' || s.method === active);
      document.getElementById('summary').innerHTML = summaries.map(s => `
        <div class="card">
          <div><strong>${{s.method_label}}</strong></div>
          <div class="sub">runs: ${{s.n_runs}}</div>
          <div class="metric">AUC ${{cell(s.auc_mean)}}</div>
          <div class="sub">std ${{cell(s.auc_std)}} | min ${{cell(s.auc_min)}} | max ${{cell(s.auc_max)}}</div>
          <div class="bar"><span style="width:${{Math.max(0, Math.min(100, (s.auc_mean || 0) * 100))}}%"></span></div>
          <div class="metric">F1 ${{cell(s.f1_score_mean)}}</div>
          <div class="sub">std ${{cell(s.f1_score_std)}} | min ${{cell(s.f1_score_min)}} | max ${{cell(s.f1_score_max)}}</div>
        </div>`).join('');
    }}
    function renderTable() {{
      const keys = ['method_label', 'run_id', 'seedplus', 'epoch', 'val_auc', 'auc', 'f1_score', 'accuracy', 'precision', 'recall', 'output_csv_path', 'checkpoint_path'];
      let rows = payload.rows.filter(r => active === 'all' || r.method === active);
      rows.sort((a, b) => {{
        const av = a[sortKey], bv = b[sortKey];
        if (number(av) && number(bv)) return sortAsc ? av - bv : bv - av;
        return sortAsc ? String(av ?? '').localeCompare(String(bv ?? '')) : String(bv ?? '').localeCompare(String(av ?? ''));
      }});
      document.getElementById('runs').innerHTML =
        `<thead><tr>${{keys.map(k => `<th onclick="sortKey='${{k}}'; sortAsc=!sortAsc; renderTable();">${{k}}</th>`).join('')}}</tr></thead>` +
        `<tbody>${{rows.map(r => `<tr>${{keys.map(k => `<td>${{cell(r[k])}}</td>`).join('')}}</tr>`).join('')}}</tbody>`;
    }}
    function render() {{ renderFilters(); renderSummary(); renderTable(); }}
    render();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")


def run_one(target: dict, args: argparse.Namespace) -> dict:
    from test_inference_finetune import (
        calculate_metrics,
        create_test_dataset,
        load_model,
        run_inference,
        save_predictions,
    )

    output_csv = args.result_dir / "predictions" / f"{target['run_id']}.csv"
    row = dict(target)
    row["output_csv_path"] = str(output_csv)

    model = load_model(
        checkpoint_path=target["checkpoint_path"],
        simclr_ckpt_path=str(args.simclr_ckpt),
        task_type="classification",
        image_type="single",
        num_classes=1,
    )
    dataset, collate_fn = create_test_dataset(
        csv_path=str(args.test_csv),
        root_dir=str(args.root_dir),
        image_type="single",
        image_size=(96, 96, 96),
        dataset_name=f"vn_{target['method']}",
    )
    raw_outputs, predictions, class_predictions, labels = run_inference(
        model=model,
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=1,
        task_type="classification",
    )
    save_predictions(
        csv_path=str(args.test_csv),
        raw_outputs=raw_outputs,
        predictions=predictions,
        class_predictions=class_predictions,
        output_path=str(output_csv),
        task_type="classification",
    )
    metrics = calculate_metrics(
        y_true=labels,
        raw_outputs=raw_outputs,
        predictions=predictions,
        class_predictions=class_predictions,
        task_type="classification",
        dataset_name=target["run_id"],
    )
    for key in ["auc", "f1_score", "accuracy", "precision", "recall", "n_samples"]:
        row[key] = metric_value(metrics, key) if key != "n_samples" else metrics.get(key)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.result_dir = args.result_dir.resolve()
    args.test_csv = args.test_csv.resolve()
    args.root_dir = args.root_dir.resolve()
    args.simclr_ckpt = args.simclr_ckpt.resolve()
    add_src_to_path(args.repo_root)

    checkpoint_dirs = args.checkpoint_dir or CHECKPOINT_DIRS
    selected = select_best_checkpoints(
        checkpoint_dirs=checkpoint_dirs,
        methods=set(args.methods),
        include_legacy_fewshot=args.include_legacy_fewshot,
    )
    selected = append_manual_targets(selected, args)
    if not selected:
        print("No matching checkpoints found.", file=sys.stderr)
        return 2

    print(f"selected_checkpoints={len(selected)}")
    for item in selected:
        val_auc = "NA" if item["val_auc"] is None else f"{item['val_auc']:.4f}"
        print(f"{item['method_label']} {item['run_id']} val_auc={val_auc} {item['checkpoint_path']}")
    missing = sorted(set(args.methods) - {item["method"] for item in selected})
    if missing:
        print(f"missing_methods={','.join(missing)}")

    if args.dry_run:
        return 0

    rows = []
    for target in selected:
        print(f"Running inference: {target['run_id']}")
        try:
            rows.append(run_one(target, args))
        except Exception as exc:
            row = dict(target)
            row["error"] = str(exc)
            rows.append(row)
            print(f"ERROR {target['run_id']}: {exc}", file=sys.stderr)

    summaries = summarize(rows)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.result_dir / "vn_inference_run_metrics.csv", rows)
    write_csv(args.result_dir / "vn_inference_method_summary.csv", summaries)
    (args.result_dir / "vn_inference_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": generated_at,
                "test_csv": str(args.test_csv),
                "root_dir": str(args.root_dir),
                "simclr_ckpt": str(args.simclr_ckpt),
                "rows": rows,
                "summaries": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    html_path = args.result_dir / "vn_inference_report.html"
    write_html(html_path, rows, summaries, generated_at, args.test_csv)
    print(f"report_html={html_path}")
    print(f"metrics_csv={args.result_dir / 'vn_inference_run_metrics.csv'}")
    print(f"summary_csv={args.result_dir / 'vn_inference_method_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
