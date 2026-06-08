#!/usr/bin/env python3
"""Hyperband-style successive-halving search for the local ALO T8 pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


BASELINE_JI = 0.7833741836842695

RESULT_FIELDS = [
    "candidate_id",
    "stage",
    "stage_name",
    "aperture_diam",
    "detector_radial_shift_mm",
    "layout_idxs",
    "t8_poses",
    "fwhm_mean",
    "sensitivity_total",
    "sensitivity_mean",
    "asci_pct",
    "n_ppdf_files",
    "n_prop_files",
    "n_asci_files",
    "JI",
    "baseline_JI",
    "improvement_vs_baseline_pct",
    "run_name",
    "run_dir",
    "status",
    "error_message",
    "command",
]


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    aperture_diam: float
    detector_radial_shift_mm: float


@dataclass(frozen=True)
class Stage:
    stage: int
    stage_name: str
    layout_idxs: str
    t8_poses: str
    keep: int | None


STAGES = [
    Stage(
        stage=1,
        stage_name="cheap_layout1_t8_4",
        layout_idxs="1",
        t8_poses="0,2,4,6",
        keep=None,
    ),
    Stage(
        stage=2,
        stage_name="medium_layout01_t8_6",
        layout_idxs="0,1",
        t8_poses="0,1,2,4,5,6",
        keep=None,
    ),
    Stage(
        stage=3,
        stage_name="final_layout01_t8_8",
        layout_idxs="0,1",
        t8_poses="0,1,2,3,4,5,6,7",
        keep=None,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a staged Hyperband-style search over aperture diameter and detector radial shift."
    )
    parser.add_argument("--n-initial", type=int, default=30)
    parser.add_argument("--keep-stage1", type=int, default=15)
    parser.add_argument("--keep-stage2", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diam-min", type=float, default=0.4)
    parser.add_argument("--diam-max", type=float, default=0.7)
    parser.add_argument("--shift-min", type=float, default=-20.0)
    parser.add_argument("--shift-max", type=float, default=20.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/hyperband"))
    parser.add_argument("--cpus", type=int, default=None, help="Optional --cpus value forwarded to run_pipeline.py.")
    parser.add_argument("--resume", action="store_true", help="Reuse successful prior stage results from out-dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print and record planned commands without running pipeline jobs.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_initial < 1:
        raise ValueError("--n-initial must be >= 1")
    if args.keep_stage1 < 1:
        raise ValueError("--keep-stage1 must be >= 1")
    if args.keep_stage2 < 1:
        raise ValueError("--keep-stage2 must be >= 1")
    if args.keep_stage1 > args.n_initial:
        raise ValueError("--keep-stage1 cannot exceed --n-initial")
    if args.keep_stage2 > args.keep_stage1:
        raise ValueError("--keep-stage2 cannot exceed --keep-stage1")
    if args.diam_max <= args.diam_min:
        raise ValueError("--diam-max must be greater than --diam-min")
    if args.shift_max <= args.shift_min:
        raise ValueError("--shift-max must be greater than --shift-min")


def configured_stages(args: argparse.Namespace) -> list[Stage]:
    return [
        Stage(
            stage=STAGES[0].stage,
            stage_name=STAGES[0].stage_name,
            layout_idxs=STAGES[0].layout_idxs,
            t8_poses=STAGES[0].t8_poses,
            keep=args.keep_stage1,
        ),
        Stage(
            stage=STAGES[1].stage,
            stage_name=STAGES[1].stage_name,
            layout_idxs=STAGES[1].layout_idxs,
            t8_poses=STAGES[1].t8_poses,
            keep=args.keep_stage2,
        ),
        STAGES[2],
    ]


def latin_hypercube_candidates(
    *,
    n: int,
    seed: int,
    diam_min: float,
    diam_max: float,
    shift_min: float,
    shift_max: float,
) -> list[Candidate]:
    rng = random.Random(seed)
    diam_values = [
        diam_min + ((idx + rng.random()) / n) * (diam_max - diam_min)
        for idx in range(n)
    ]
    shift_values = [
        shift_min + ((idx + rng.random()) / n) * (shift_max - shift_min)
        for idx in range(n)
    ]
    rng.shuffle(diam_values)
    rng.shuffle(shift_values)
    return [
        Candidate(
            candidate_id=f"c{idx:03d}",
            aperture_diam=diam_values[idx],
            detector_radial_shift_mm=shift_values[idx],
        )
        for idx in range(n)
    ]


def sanitize_float_for_name(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def build_run_name(candidate: Candidate, stage: Stage) -> str:
    diam = sanitize_float_for_name(candidate.aperture_diam)
    shift = sanitize_float_for_name(candidate.detector_radial_shift_mm)
    return f"hb_{candidate.candidate_id}_s{stage.stage}_d{diam}_shift{shift}"


def build_command(
    *,
    repo_dir: Path,
    eval_runs_dir: Path,
    candidate: Candidate,
    stage: Stage,
    cpus: int | None,
) -> tuple[list[str], str, Path]:
    run_name = build_run_name(candidate, stage)
    run_dir = eval_runs_dir / run_name
    cmd = [
        sys.executable,
        str(repo_dir / "run_pipeline.py"),
        "--runs-dir",
        str(eval_runs_dir),
        "--run-name",
        run_name,
        "--layout-idxs",
        stage.layout_idxs,
        "--t8-poses",
        stage.t8_poses,
        "--aperture-diam",
        f"{candidate.aperture_diam:.12g}",
        "--detector-radial-shift-mm",
        f"{candidate.detector_radial_shift_mm:.12g}",
        "--skip-recon",
    ]
    if cpus is not None:
        cmd.extend(["--cpus", str(cpus)])
    return cmd, run_name, run_dir


def empty_row(
    *,
    candidate: Candidate,
    stage: Stage,
    run_name: str,
    run_dir: Path,
    command: list[str],
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "stage": stage.stage,
        "stage_name": stage.stage_name,
        "aperture_diam": candidate.aperture_diam,
        "detector_radial_shift_mm": candidate.detector_radial_shift_mm,
        "layout_idxs": stage.layout_idxs,
        "t8_poses": stage.t8_poses,
        "fwhm_mean": math.nan,
        "sensitivity_total": math.nan,
        "sensitivity_mean": math.nan,
        "asci_pct": math.nan,
        "n_ppdf_files": 0,
        "n_prop_files": 0,
        "n_asci_files": 0,
        "JI": 0.0,
        "baseline_JI": BASELINE_JI,
        "improvement_vs_baseline_pct": -100.0,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": status,
        "error_message": error_message,
        "command": shlex.join(command),
    }


def parse_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_ji_metrics(csv_path: Path) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in JI metrics CSV: {csv_path}")
    return rows[-1]


def row_from_metrics(
    *,
    candidate: Candidate,
    stage: Stage,
    run_name: str,
    run_dir: Path,
    command: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    ji = parse_float(metrics.get("JI"), default=0.0)
    if not math.isfinite(ji):
        ji = 0.0
    improvement = (ji - BASELINE_JI) / BASELINE_JI * 100.0
    return {
        "candidate_id": candidate.candidate_id,
        "stage": stage.stage,
        "stage_name": stage.stage_name,
        "aperture_diam": candidate.aperture_diam,
        "detector_radial_shift_mm": candidate.detector_radial_shift_mm,
        "layout_idxs": stage.layout_idxs,
        "t8_poses": stage.t8_poses,
        "fwhm_mean": parse_float(metrics.get("fwhm_mean")),
        "sensitivity_total": parse_float(metrics.get("sensitivity_total")),
        "sensitivity_mean": parse_float(metrics.get("sensitivity_mean")),
        "asci_pct": parse_float(metrics.get("asci_pct")),
        "n_ppdf_files": parse_int(metrics.get("n_ppdf_files")),
        "n_prop_files": parse_int(metrics.get("n_prop_files")),
        "n_asci_files": parse_int(metrics.get("n_asci_files")),
        "JI": ji,
        "baseline_JI": BASELINE_JI,
        "improvement_vs_baseline_pct": improvement,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "status": "ok",
        "error_message": "",
        "command": shlex.join(command),
    }


def ensure_result_files(out_dir: Path, *, dry_run: bool) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    eval_runs_dir = out_dir / "evals"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)
    prefix = "hyperband_dry_run" if dry_run else "hyperband"
    return (
        out_dir / f"{prefix}_results.csv",
        out_dir / f"{prefix}_results.jsonl",
        out_dir / f"{prefix}_state.json",
        logs_dir,
    )


def assert_output_files_clean(
    *,
    csv_path: Path,
    jsonl_path: Path,
    state_path: Path,
    resume: bool,
) -> None:
    if resume:
        return
    existing = [path for path in [csv_path, jsonl_path, state_path] if path.exists()]
    if existing:
        raise FileExistsError(
            "Output directory already contains Hyperband results. "
            "Use --resume or choose a new --out-dir."
        )


def append_result(csv_path: Path, jsonl_path: Path, row: dict[str, Any]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe({field: row.get(field, "") for field in RESULT_FIELDS})) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_resume_results(csv_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not csv_path.exists():
        return {}
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (str(row["candidate_id"]), int(row["stage"]))
            except (KeyError, ValueError):
                continue
            if row.get("status") == "ok":
                rows_by_key[key] = row
    return rows_by_key


def resume_row_matches(row: dict[str, Any], candidate: Candidate, stage: Stage) -> bool:
    return (
        row.get("status") == "ok"
        and row.get("layout_idxs") == stage.layout_idxs
        and row.get("t8_poses") == stage.t8_poses
        and abs(parse_float(row.get("aperture_diam")) - candidate.aperture_diam) <= 1e-9
        and abs(parse_float(row.get("detector_radial_shift_mm")) - candidate.detector_radial_shift_mm) <= 1e-9
    )


def save_state(
    state_path: Path,
    *,
    args: argparse.Namespace,
    candidates: list[Candidate],
    stage: Stage | None,
    rows: list[dict[str, Any]],
    current_candidate_ids: list[str],
    completed: bool,
) -> None:
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed": completed,
        "baseline_JI": BASELINE_JI,
        "args": {
            "n_initial": args.n_initial,
            "keep_stage1": args.keep_stage1,
            "keep_stage2": args.keep_stage2,
            "seed": args.seed,
            "diam_min": args.diam_min,
            "diam_max": args.diam_max,
            "shift_min": args.shift_min,
            "shift_max": args.shift_max,
            "out_dir": str(args.out_dir),
            "cpus": args.cpus,
            "resume": args.resume,
            "dry_run": args.dry_run,
        },
        "active_stage": None if stage is None else {
            "stage": stage.stage,
            "stage_name": stage.stage_name,
            "layout_idxs": stage.layout_idxs,
            "t8_poses": stage.t8_poses,
            "keep": stage.keep,
        },
        "candidates": [candidate.__dict__ for candidate in candidates],
        "current_candidate_ids": current_candidate_ids,
        "results_count": len(rows),
        "latest_results": rows[-20:],
    }
    state_path.write_text(json.dumps(json_safe(state), indent=2) + "\n", encoding="utf-8")


def log_tail(path: Path, max_lines: int = 20) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def evaluate_candidate(
    *,
    repo_dir: Path,
    out_dir: Path,
    logs_dir: Path,
    candidate: Candidate,
    stage: Stage,
    cpus: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    eval_runs_dir = out_dir / "evals"
    cmd, run_name, run_dir = build_command(
        repo_dir=repo_dir,
        eval_runs_dir=eval_runs_dir,
        candidate=candidate,
        stage=stage,
        cpus=cpus,
    )
    print(f"[stage {stage.stage}] {candidate.candidate_id}: {shlex.join(cmd)}")

    if dry_run:
        return empty_row(
            candidate=candidate,
            stage=stage,
            run_name=run_name,
            run_dir=run_dir,
            command=cmd,
            status="dry_run",
            error_message="dry-run: command not executed",
        )

    log_path = logs_dir / f"{run_name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"cmd: {shlex.join(cmd)}\n\n")
        completed = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        return empty_row(
            candidate=candidate,
            stage=stage,
            run_name=run_name,
            run_dir=run_dir,
            command=cmd,
            status="failed",
            error_message=f"run_pipeline.py exited {completed.returncode}; log tail:\n{log_tail(log_path)}",
        )

    ji_csv = run_dir / "results" / "ji_metrics.csv"
    try:
        metrics = read_ji_metrics(ji_csv)
    except Exception as exc:
        return empty_row(
            candidate=candidate,
            stage=stage,
            run_name=run_name,
            run_dir=run_dir,
            command=cmd,
            status="failed",
            error_message=f"failed to read {ji_csv}: {exc}",
        )

    return row_from_metrics(
        candidate=candidate,
        stage=stage,
        run_name=run_name,
        run_dir=run_dir,
        command=cmd,
        metrics=metrics,
    )


def ji_for_sort(row: dict[str, Any]) -> float:
    if row.get("status") not in {"ok", "dry_run"}:
        return -math.inf
    ji = parse_float(row.get("JI"), default=0.0)
    return ji if math.isfinite(ji) else -math.inf


def promote_candidates(
    *,
    stage_results: list[dict[str, Any]],
    candidates_by_id: dict[str, Candidate],
    keep: int,
) -> list[Candidate]:
    ranked = sorted(
        [row for row in stage_results if row.get("status") in {"ok", "dry_run"}],
        key=ji_for_sort,
        reverse=True,
    )
    return [candidates_by_id[str(row["candidate_id"])] for row in ranked[:keep]]


def print_stage3_summary(stage3_rows: list[dict[str, Any]]) -> None:
    ranked = sorted(stage3_rows, key=ji_for_sort, reverse=True)
    if not ranked:
        print("\nNo Stage 3 candidates were evaluated.")
        return

    print("\nTop Stage 3 candidates:")
    print("candidate_id  aperture_diam  detector_shift_mm  JI        FWHM      sens_mean  ASCI      improvement_pct")
    for row in ranked:
        print(
            f"{row['candidate_id']:>12}  "
            f"{parse_float(row['aperture_diam']):13.6f}  "
            f"{parse_float(row['detector_radial_shift_mm']):17.6f}  "
            f"{parse_float(row['JI'], 0.0):8.6f}  "
            f"{parse_float(row['fwhm_mean']):8.6f}  "
            f"{parse_float(row['sensitivity_mean']):9.6f}  "
            f"{parse_float(row['asci_pct']):8.4f}  "
            f"{parse_float(row['improvement_vs_baseline_pct']):15.3f}"
        )

    best = ranked[0]
    best_ji = parse_float(best.get("JI"), default=0.0)
    print("\nBest Stage 3 candidate:")
    print(f"  candidate_id: {best['candidate_id']}")
    print(f"  aperture_diam: {parse_float(best['aperture_diam']):.8f}")
    print(f"  detector_radial_shift_mm: {parse_float(best['detector_radial_shift_mm']):.8f}")
    print(f"  final JI: {best_ji:.12f}")
    print(f"  baseline JI: {BASELINE_JI:.12f}")
    print(f"  improvement_vs_baseline_pct: {parse_float(best['improvement_vs_baseline_pct']):.3f}")
    print(f"  run_dir: {best['run_dir']}")
    if best_ji > BASELINE_JI:
        print("Best candidate beats baseline.")
    else:
        print("No candidate beat baseline.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"hyperband_optimize.py: error: {exc}", file=sys.stderr)
        return 2

    repo_dir = Path(__file__).resolve().parent
    out_dir = (repo_dir / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    args.out_dir = out_dir
    prefix = "hyperband_dry_run" if args.dry_run else "hyperband"
    assert_output_files_clean(
        csv_path=out_dir / f"{prefix}_results.csv",
        jsonl_path=out_dir / f"{prefix}_results.jsonl",
        state_path=out_dir / f"{prefix}_state.json",
        resume=args.resume,
    )
    csv_path, jsonl_path, state_path, logs_dir = ensure_result_files(out_dir, dry_run=args.dry_run)

    stages = configured_stages(args)
    candidates = latin_hypercube_candidates(
        n=args.n_initial,
        seed=args.seed,
        diam_min=args.diam_min,
        diam_max=args.diam_max,
        shift_min=args.shift_min,
        shift_max=args.shift_max,
    )
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    resume_rows = load_resume_results(csv_path) if args.resume else {}

    all_rows: list[dict[str, Any]] = []
    current_candidates = candidates
    stage3_rows: list[dict[str, Any]] = []

    print(f"Hyperband output directory: {out_dir}")
    print(f"Baseline JI: {BASELINE_JI:.12f}")

    for stage in stages:
        print(
            f"\nStage {stage.stage}: {stage.stage_name} | "
            f"layouts={stage.layout_idxs} | t8_poses={stage.t8_poses} | "
            f"candidates={len(current_candidates)}"
        )
        stage_results: list[dict[str, Any]] = []
        for candidate in current_candidates:
            resume_key = (candidate.candidate_id, stage.stage)
            if resume_key in resume_rows and resume_row_matches(resume_rows[resume_key], candidate, stage):
                row = resume_rows[resume_key]
                print(f"[stage {stage.stage}] {candidate.candidate_id}: reusing successful result from {row.get('run_dir')}")
            else:
                row = evaluate_candidate(
                    repo_dir=repo_dir,
                    out_dir=out_dir,
                    logs_dir=logs_dir,
                    candidate=candidate,
                    stage=stage,
                    cpus=args.cpus,
                    dry_run=args.dry_run,
                )
                append_result(csv_path, jsonl_path, row)

            stage_results.append(row)
            all_rows.append(row)
            save_state(
                state_path,
                args=args,
                candidates=candidates,
                stage=stage,
                rows=all_rows,
                current_candidate_ids=[candidate.candidate_id for candidate in current_candidates],
                completed=False,
            )

        if stage.keep is not None:
            current_candidates = promote_candidates(
                stage_results=stage_results,
                candidates_by_id=candidates_by_id,
                keep=stage.keep,
            )
            print(
                f"Promoted {len(current_candidates)} candidates: "
                f"{', '.join(candidate.candidate_id for candidate in current_candidates)}"
            )
        else:
            stage3_rows = stage_results

    save_state(
        state_path,
        args=args,
        candidates=candidates,
        stage=None,
        rows=all_rows,
        current_candidate_ids=[],
        completed=True,
    )

    if args.dry_run:
        print("\nDry run complete. No scientific winner selected because no pipeline jobs were executed.")
    else:
        print_stage3_summary(stage3_rows)
    print(f"\nResults CSV: {csv_path}")
    print(f"Results JSONL: {jsonl_path}")
    print(f"State JSON: {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
