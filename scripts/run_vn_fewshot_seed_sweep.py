#!/usr/bin/env python3
"""Generate per-seed configs and run VN AD/CN few-shot training."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Union

import yaml


DEFAULT_REPO_ROOT = Path("/home2/duongoku/BrainIAC_ADCN/BrainIAC")
DEFAULT_CSV_GLOB = "src/data/csvs/vn_train_k_*_SEEDPLUS_*.csv"
DEFAULT_TRAIN_SCRIPT = "src/train_lightning_mci.py"
DEFAULT_CONFIG_DIR = "src/generated_configs/vn_fewshot_seed_sweep"
DEFAULT_LOG_DIR = "src/results/vn_fewshot_seed_sweep_logs"
CSV_PATTERN = re.compile(r"vn_train_k_(?P<k>\d+)_SEEDPLUS_(?P<seedplus>\d+)\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VN few-shot seed sweep by patching config fields per training CSV."
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--csv-glob", default=DEFAULT_CSV_GLOB)
    parser.add_argument("--train-script", default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--k", nargs="*", type=int, choices=[1, 5], default=[1, 5])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing-configs", action="store_true")
    return parser.parse_args()


def resolve(repo_root: Path, path: Union[str, Path]) -> Path:
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


def discover_csvs(repo_root: Path, csv_glob: str, allowed_k: set) -> List[Tuple[int, int, Path]]:
    paths = sorted(glob.glob(str(resolve(repo_root, csv_glob))))
    runs: List[Tuple[int, int, Path]] = []
    for raw_path in paths:
        path = Path(raw_path)
        match = CSV_PATTERN.match(path.name)
        if not match:
            continue
        k = int(match.group("k"))
        seedplus = int(match.group("seedplus"))
        if k in allowed_k:
            runs.append((k, seedplus, path))
    return sorted(runs, key=lambda item: (item[0], item[1], str(item[2])))


def load_base_config(repo_root: Path, k: int) -> dict:
    config_path = repo_root / "src" / f"config_vn_fewshot_k_{k}.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_config(base_config: dict, csv_path: Path, k: int, seedplus: int) -> tuple[dict, str]:
    run_name = f"vn_fewshot_k_{k}_seedplus_{seedplus}"
    config = dict(base_config)
    config["data"] = dict(base_config["data"])
    config["logger"] = dict(base_config["logger"])
    config["data"]["csv_file"] = str(csv_path)
    config["logger"]["save_name"] = (
        f"{run_name}_model-epoch-{{epoch:02d}}-{{val_auc:.2f}}"
    )
    config["logger"]["run_name_mci"] = run_name
    return config, run_name


def write_yaml(path: Path, config: dict, skip_existing: bool) -> None:
    if skip_existing and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def append_manifest(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_training(
    repo_root: Path,
    python_exe: str,
    train_script: Path,
    config_path: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [python_exe, str(train_script), "--config", str(config_path)]
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"command: {' '.join(command)}\n")
        log_handle.flush()
        process = subprocess.run(
            command,
            cwd=repo_root,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return process.returncode


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    train_script = resolve(repo_root, args.train_script)
    config_dir = resolve(repo_root, args.config_dir)
    log_dir = resolve(repo_root, args.log_dir)
    manifest_path = log_dir / "manifest.csv"

    runs = discover_csvs(repo_root, args.csv_glob, set(args.k))
    runs = runs[args.start_index :]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]

    if not runs:
        print("No matching few-shot CSV files found.", file=sys.stderr)
        return 2

    print(f"repo_root={repo_root}")
    print(f"train_script={train_script}")
    print(f"runs={len(runs)}")

    for index, (k, seedplus, csv_path) in enumerate(runs, start=args.start_index):
        base_config = load_base_config(repo_root, k)
        config, run_name = build_config(base_config, csv_path, k, seedplus)
        config_path = config_dir / f"{run_name}.yml"
        log_path = log_dir / f"{run_name}.log"
        write_yaml(config_path, config, args.skip_existing_configs)

        started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        print(f"[{index}] {run_name}")
        print(f"  csv={csv_path}")
        print(f"  config={config_path}")
        print(f"  log={log_path}")

        if args.dry_run:
            return_code = 0
            status = "dry_run"
        else:
            return_code = run_training(
                repo_root=repo_root,
                python_exe=args.python_exe,
                train_script=train_script,
                config_path=config_path,
                log_path=log_path,
            )
            status = "ok" if return_code == 0 else "failed"

        finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
        append_manifest(
            manifest_path,
            {
                "index": index,
                "k": k,
                "seedplus": seedplus,
                "csv_file": str(csv_path),
                "config_file": str(config_path),
                "run_name_mci": run_name,
                "save_name": config["logger"]["save_name"],
                "log_file": str(log_path),
                "status": status,
                "return_code": return_code,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
            },
        )

        if return_code != 0:
            print(f"Stopped after failed run: {run_name}", file=sys.stderr)
            return return_code

    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
