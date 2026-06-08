#!/usr/bin/env python3
"""Bayesian optimization with a cheap Hyperband-style gate for ALO T8 runs.

The BO surrogate is trained only on successful full-fidelity JI values.
Cheap evaluations are used only for promotion decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


RESULT_FIELDS = [
    "candidate_id",
    "source",
    "iteration",
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
    "JI",
    "predicted_full_JI",
    "posterior_std",
    "acquisition_value",
    "best_full_JI_before",
    "promoted_to_full",
    "promotion_reason",
    "run_name",
    "run_dir",
    "status",
    "command",
    "error_message",
    "started_at",
    "finished_at",
    "elapsed_sec",
]

METRIC_FIELDS = ["fwhm_mean", "sensitivity_total", "sensitivity_mean", "asci_pct", "JI"]

SUCCESS_STATUSES = {"success"}
COMPLETED_STATUSES = {"success", "failed", "skipped_existing"}

CHEAP_STAGE = None
FULL_STAGE = None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    aperture_diam: float
    detector_radial_shift_mm: float
    source: str
    iteration: int


@dataclass(frozen=True)
class Stage:
    stage: str
    stage_name: str
    layout_idxs: str
    t8_poses: str


@dataclass
class EvalResult:
    candidate_id: str
    source: str
    iteration: int
    stage: str
    stage_name: str
    aperture_diam: float
    detector_radial_shift_mm: float
    layout_idxs: str
    t8_poses: str
    fwhm_mean: float
    sensitivity_total: float
    sensitivity_mean: float
    asci_pct: float
    JI: float
    predicted_full_JI: float
    posterior_std: float
    acquisition_value: float
    best_full_JI_before: float
    promoted_to_full: bool | None
    promotion_reason: str
    run_name: str
    run_dir: str
    status: str
    command: str
    error_message: str
    started_at: str
    finished_at: str
    elapsed_sec: float

    def to_csv_row(self) -> dict[str, str]:
        row: dict[str, str] = {}
        for key, value in asdict(self).items():
            if isinstance(value, bool):
                row[key] = "true" if value else "false"
            elif value is None:
                row[key] = ""
            elif isinstance(value, float):
                row[key] = "" if not math.isfinite(value) else repr(value)
            else:
                row[key] = str(value)
        return row

    def to_json_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
            else:
                row[key] = value
        return row


CHEAP_STAGE = Stage(
    stage="cheap",
    stage_name="cheap_layout1_t8_4",
    layout_idxs="1",
    t8_poses="0,2,4,6",
)

FULL_STAGE = Stage(
    stage="full",
    stage_name="full_layout01_t8_8",
    layout_idxs="0,1",
    t8_poses="0,1,2,3,4,5,6,7",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run BO with full-fidelity warm start and cheap T8 promotion gate."
    )
    parser.add_argument("--out-dir", default="runs/bo_hyperband")
    parser.add_argument("--n-initial", type=int, default=25)
    parser.add_argument("--bo-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diam-min", type=float, default=0.4)
    parser.add_argument("--diam-max", type=float, default=0.7)
    parser.add_argument("--shift-min", type=float, default=-20.0)
    parser.add_argument("--shift-max", type=float, default=20.0)
    parser.add_argument("--cheap-promote-quantile", type=float, default=0.75)
    parser.add_argument("--min-cheap-history-for-gate", type=int, default=5)
    parser.add_argument("--mc-samples", type=int, default=256)
    parser.add_argument("--num-restarts", type=int, default=20)
    parser.add_argument("--raw-samples", type=int, default=512)
    parser.add_argument("--duplicate-tol", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-pipeline", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run exactly one full-stage baseline candidate and exit before LHS/BO.",
    )
    parser.add_argument("--baseline-aperture-diam", type=float, default=0.4)
    parser.add_argument("--baseline-detector-radial-shift-mm", type=float, default=0.0)
    parser.add_argument(
        "--skip-recon",
        action="store_true",
        help="Forward --skip-recon to run_pipeline.py so projection, MLEM, and NPZ viewing are skipped.",
    )
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--pose-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-pipeline", default=str(repo_dir / "run_pipeline.py"))
    parser.add_argument("--min-acq-value", type=float, default=None)
    parser.add_argument("--self-test-bo", action="store_true")
    args = parser.parse_args(argv)

    if args.n_initial < 0:
        parser.error("--n-initial must be non-negative")
    if args.bo_iters < 0:
        parser.error("--bo-iters must be non-negative")
    if args.diam_max <= args.diam_min:
        parser.error("--diam-max must be greater than --diam-min")
    if args.shift_max <= args.shift_min:
        parser.error("--shift-max must be greater than --shift-min")
    if not 0.0 <= args.cheap_promote_quantile <= 1.0:
        parser.error("--cheap-promote-quantile must be in [0, 1]")
    if args.min_cheap_history_for_gate < 0:
        parser.error("--min-cheap-history-for-gate must be non-negative")
    if args.mc_samples <= 0 or args.num_restarts <= 0 or args.raw_samples <= 0:
        parser.error("--mc-samples, --num-restarts, and --raw-samples must be positive")
    if args.duplicate_tol < 0.0:
        parser.error("--duplicate-tol must be non-negative")
    return args


def repo_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_dir() / candidate).resolve()


def result_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "results_csv": out_dir / "bo_hyperband_results.csv",
        "results_jsonl": out_dir / "bo_hyperband_results.jsonl",
        "state_json": out_dir / "bo_hyperband_state.json",
        "training_csv": out_dir / "bo_training_history.csv",
        "evals_dir": out_dir / "evals",
        "logs_dir": out_dir / "logs",
    }


def ensure_dirs(args: argparse.Namespace) -> dict[str, Path]:
    out_dir = resolve_path(args.out_dir)
    paths = result_paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths["evals_dir"].mkdir(parents=True, exist_ok=True)
    paths["logs_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def latin_hypercube_candidates(args: argparse.Namespace) -> list[Candidate]:
    rng = random.Random(args.seed)
    n = args.n_initial
    diam_values = [
        args.diam_min + (i + rng.random()) / n * (args.diam_max - args.diam_min)
        for i in range(n)
    ]
    shift_values = [
        args.shift_min + (i + rng.random()) / n * (args.shift_max - args.shift_min)
        for i in range(n)
    ]
    rng.shuffle(diam_values)
    rng.shuffle(shift_values)
    return [
        Candidate(
            candidate_id=f"lhs_{idx:03d}",
            aperture_diam=diam_values[idx],
            detector_radial_shift_mm=shift_values[idx],
            source="lhs",
            iteration=idx,
        )
        for idx in range(n)
    ]


def normalize_X(points: Iterable[Sequence[float]], args: argparse.Namespace) -> list[list[float]]:
    normalized: list[list[float]] = []
    for aperture_diam, detector_shift in points:
        normalized.append(
            [
                (float(aperture_diam) - args.diam_min) / (args.diam_max - args.diam_min),
                (float(detector_shift) - args.shift_min) / (args.shift_max - args.shift_min),
            ]
        )
    return normalized


def unnormalize_X(points: Iterable[Sequence[float]], args: argparse.Namespace) -> list[list[float]]:
    physical: list[list[float]] = []
    for diam_norm, shift_norm in points:
        diam_clamped = min(1.0, max(0.0, float(diam_norm)))
        shift_clamped = min(1.0, max(0.0, float(shift_norm)))
        physical.append(
            [
                args.diam_min + diam_clamped * (args.diam_max - args.diam_min),
                args.shift_min + shift_clamped * (args.shift_max - args.shift_min),
            ]
        )
    return physical


def load_results_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_result_csv(path: Path, result: EvalResult) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result.to_csv_row())


def append_result_jsonl(path: Path, result: EvalResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_json_row(), sort_keys=True) + "\n")


def append_training_history(path: Path, result: EvalResult) -> None:
    if not is_training_result(result.to_csv_row()):
        return
    append_result_csv(path, result)


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, str) and not value.strip():
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_training_result(row: dict[str, str]) -> bool:
    return (
        row.get("stage") == "full"
        and row.get("status") in SUCCESS_STATUSES
        and math.isfinite(safe_float(row.get("JI")))
        and math.isfinite(safe_float(row.get("aperture_diam")))
        and math.isfinite(safe_float(row.get("detector_radial_shift_mm")))
    )


def get_successful_full_history(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if is_training_result(row)]


def get_successful_cheap_history(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("stage") == "cheap"
        and row.get("status") in SUCCESS_STATUSES
        and math.isfinite(safe_float(row.get("JI")))
    ]


def best_full_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    full_rows = get_successful_full_history(rows)
    if not full_rows:
        return None
    return max(full_rows, key=lambda row: safe_float(row.get("JI")))


def get_candidate_params_from_rows(rows: list[dict[str, str]]) -> list[list[float]]:
    params: list[list[float]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        diam = safe_float(row.get("aperture_diam"))
        shift = safe_float(row.get("detector_radial_shift_mm"))
        if not (math.isfinite(diam) and math.isfinite(shift)):
            continue
        key = (repr(diam), repr(shift))
        if key in seen:
            continue
        seen.add(key)
        params.append([diam, shift])
    return params


def latest_result_for(
    rows: list[dict[str, str]],
    candidate_id: str,
    stage_name: str,
) -> dict[str, str] | None:
    for row in reversed(rows):
        if row.get("candidate_id") == candidate_id and row.get("stage_name") == stage_name:
            return row
    return None


def completed_existing_result(
    rows: list[dict[str, str]],
    candidate_id: str,
    stage_name: str,
    dry_run: bool,
) -> dict[str, str] | None:
    row = latest_result_for(rows, candidate_id, stage_name)
    if row is None:
        return None
    status = row.get("status", "")
    if dry_run and status == "dry_run":
        return row
    if status in COMPLETED_STATUSES:
        return row
    return None


def save_state(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    current_bo_iteration: int,
) -> None:
    best_row = best_full_row(rows)
    state = {
        "saved_at": now_iso(),
        "random_seed": args.seed,
        "bounds": {
            "aperture_diam": [args.diam_min, args.diam_max],
            "detector_radial_shift_mm": [args.shift_min, args.shift_max],
        },
        "completed_candidates": sorted({row.get("candidate_id", "") for row in rows if row.get("candidate_id")}),
        "completed_full_evaluations": [
            row.get("candidate_id", "")
            for row in rows
            if row.get("stage") == "full" and row.get("status") in COMPLETED_STATUSES
        ],
        "completed_cheap_evaluations": [
            row.get("candidate_id", "")
            for row in rows
            if row.get("stage") == "cheap" and row.get("status") in COMPLETED_STATUSES
        ],
        "current_bo_iteration": current_bo_iteration,
        "best_full_candidate_so_far": best_row,
        "result_csv_path": str(path.parent / "bo_hyperband_results.csv"),
        "training_history_path": str(path.parent / "bo_training_history.csv"),
    }
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline_supported_flags(run_pipeline_path: Path) -> set[str]:
    text = run_pipeline_path.read_text(encoding="utf-8")
    known_flags = {
        "--runs-dir",
        "--run-name",
        "--layout-idxs",
        "--t8-poses",
        "--aperture-diam",
        "--detector-radial-shift-mm",
        "--config-name",
        "--resume",
        "--skip-recon",
        "--cpus",
        "--pose-workers",
        "--torch-threads",
        "--torch-interop-threads",
    }
    return {flag for flag in known_flags if flag in text}


def require_pipeline_flags(flags: set[str]) -> None:
    required = {
        "--run-name",
        "--layout-idxs",
        "--t8-poses",
        "--aperture-diam",
        "--detector-radial-shift-mm",
        "--config-name",
    }
    missing = sorted(required - flags)
    if missing:
        raise RuntimeError(f"run_pipeline.py is missing required CLI flags: {', '.join(missing)}")


def build_run_name_and_dir(
    args: argparse.Namespace,
    stage: Stage,
    candidate: Candidate,
    supported_flags: set[str],
) -> tuple[str, Path, list[str]]:
    out_dir = resolve_path(args.out_dir)
    eval_slug = f"{candidate.candidate_id}_{stage.stage_name}"
    run_extra_args: list[str] = []

    if "--runs-dir" in supported_flags:
        run_name = f"evals/{eval_slug}"
        run_dir = out_dir / "evals" / eval_slug
        run_extra_args.extend(["--runs-dir", str(out_dir)])
        return run_name, run_dir, run_extra_args

    default_runs_dir = repo_dir() / "runs"
    run_dir = out_dir / "evals" / eval_slug
    try:
        run_name = str(run_dir.relative_to(default_runs_dir))
    except ValueError as exc:
        raise RuntimeError(
            "run_pipeline.py does not support --runs-dir, so --out-dir must be under ./runs"
        ) from exc
    return run_name, run_dir, run_extra_args


def add_optional_forward(
    command: list[str],
    supported_flags: set[str],
    flag: str,
    value: int | None,
) -> None:
    if value is not None and flag in supported_flags:
        command.extend([flag, str(value)])


def build_pipeline_command(
    args: argparse.Namespace,
    candidate: Candidate,
    stage: Stage,
    supported_flags: set[str],
) -> tuple[list[str], str, Path]:
    run_pipeline_path = resolve_path(args.run_pipeline)
    run_name, run_dir, run_extra_args = build_run_name_and_dir(args, stage, candidate, supported_flags)

    command = [
        args.python,
        str(run_pipeline_path),
        *run_extra_args,
        "--run-name",
        run_name,
        "--layout-idxs",
        stage.layout_idxs,
        "--t8-poses",
        stage.t8_poses,
        "--aperture-diam",
        repr(candidate.aperture_diam),
        "--detector-radial-shift-mm",
        repr(candidate.detector_radial_shift_mm),
        "--config-name",
        f"{candidate.candidate_id}_{stage.stage_name}",
    ]
    if args.resume_pipeline and "--resume" in supported_flags:
        command.append("--resume")
    if args.skip_recon and "--skip-recon" in supported_flags:
        command.append("--skip-recon")
    add_optional_forward(command, supported_flags, "--cpus", args.cpus)
    add_optional_forward(command, supported_flags, "--pose-workers", args.pose_workers)
    add_optional_forward(command, supported_flags, "--torch-threads", args.torch_threads)
    add_optional_forward(command, supported_flags, "--torch-interop-threads", args.torch_interop_threads)
    return command, run_name, run_dir


def read_ji_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "results" / "ji_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_path}")

    last_row = rows[-1]
    missing = [field for field in METRIC_FIELDS if field not in last_row]
    if missing:
        raise ValueError(f"Missing metric columns in {metrics_path}: {', '.join(missing)}")

    metrics = {field: safe_float(last_row.get(field)) for field in METRIC_FIELDS}
    if not math.isfinite(metrics["JI"]):
        raise ValueError(f"Non-finite JI in {metrics_path}: {last_row.get('JI')!r}")
    return metrics


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def empty_metrics() -> dict[str, float]:
    return {field: math.nan for field in METRIC_FIELDS}


def make_eval_result(
    candidate: Candidate,
    stage: Stage,
    metrics: dict[str, float],
    predicted_full_JI: float,
    posterior_std: float,
    acquisition_value: float,
    best_full_JI_before: float,
    promoted_to_full: bool | None,
    promotion_reason: str,
    run_name: str,
    run_dir: Path,
    status: str,
    command: list[str],
    error_message: str,
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> EvalResult:
    return EvalResult(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        iteration=candidate.iteration,
        stage=stage.stage,
        stage_name=stage.stage_name,
        aperture_diam=candidate.aperture_diam,
        detector_radial_shift_mm=candidate.detector_radial_shift_mm,
        layout_idxs=stage.layout_idxs,
        t8_poses=stage.t8_poses,
        fwhm_mean=metrics.get("fwhm_mean", math.nan),
        sensitivity_total=metrics.get("sensitivity_total", math.nan),
        sensitivity_mean=metrics.get("sensitivity_mean", math.nan),
        asci_pct=metrics.get("asci_pct", math.nan),
        JI=metrics.get("JI", math.nan),
        predicted_full_JI=predicted_full_JI,
        posterior_std=posterior_std,
        acquisition_value=acquisition_value,
        best_full_JI_before=best_full_JI_before,
        promoted_to_full=promoted_to_full,
        promotion_reason=promotion_reason,
        run_name=run_name,
        run_dir=str(run_dir),
        status=status,
        command=shlex.join(command),
        error_message=error_message,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_sec=elapsed_sec,
    )


def run_pipeline_eval(
    args: argparse.Namespace,
    candidate: Candidate,
    stage: Stage,
    supported_flags: set[str],
    predicted_full_JI: float = math.nan,
    posterior_std: float = math.nan,
    acquisition_value: float = math.nan,
    best_full_JI_before: float = math.nan,
    promoted_to_full: bool | None = None,
    promotion_reason: str = "",
) -> EvalResult:
    command, run_name, run_dir = build_pipeline_command(args, candidate, stage, supported_flags)
    started_at = now_iso()
    started = time.monotonic()

    if args.dry_run:
        print(shlex.join(command))
        finished_at = now_iso()
        return make_eval_result(
            candidate,
            stage,
            empty_metrics(),
            predicted_full_JI,
            posterior_std,
            acquisition_value,
            best_full_JI_before,
            promoted_to_full,
            promotion_reason,
            run_name,
            run_dir,
            "dry_run",
            command,
            "",
            started_at,
            finished_at,
            time.monotonic() - started,
        )

    if not args.force_rerun:
        try:
            metrics = read_ji_metrics(run_dir)
            finished_at = now_iso()
            return make_eval_result(
                candidate,
                stage,
                metrics,
                predicted_full_JI,
                posterior_std,
                acquisition_value,
                best_full_JI_before,
                promoted_to_full,
                promotion_reason or "existing_metrics",
                run_name,
                run_dir,
                "success",
                command,
                "",
                started_at,
                finished_at,
                time.monotonic() - started,
            )
        except (FileNotFoundError, ValueError):
            pass

    paths = result_paths(resolve_path(args.out_dir))
    log_path = paths["logs_dir"] / f"{candidate.candidate_id}_{stage.stage_name}.log"
    status = "success"
    error_message = ""
    metrics = empty_metrics()

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {shlex.join(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=repo_dir(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        status = "failed"
        error_message = f"run_pipeline.py exited with code {completed.returncode}\n{tail_text(log_path)}"
    else:
        try:
            metrics = read_ji_metrics(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            status = "failed"
            error_message = str(exc)

    finished_at = now_iso()
    return make_eval_result(
        candidate,
        stage,
        metrics,
        predicted_full_JI,
        posterior_std,
        acquisition_value,
        best_full_JI_before,
        promoted_to_full,
        promotion_reason,
        run_name,
        run_dir,
        status,
        command,
        error_message,
        started_at,
        finished_at,
        time.monotonic() - started,
    )


def _bo_dependencies() -> dict[str, Any]:
    try:
        import torch
        try:
            from botorch.acquisition.logei import qLogExpectedImprovement
        except ImportError:
            from botorch.acquisition.monte_carlo import qLogExpectedImprovement
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.outcome import Standardize
        from botorch.optim import optimize_acqf
        from botorch.sampling.normal import SobolQMCNormalSampler
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.mlls import ExactMarginalLogLikelihood

        try:
            from botorch.fit import fit_gpytorch_mll
        except ImportError:
            from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll
    except Exception as exc:
        raise RuntimeError(
            "BoTorch/GPyTorch are required for BO mode. "
            "Install/load them or run with --dry-run only."
        ) from exc

    return {
        "torch": torch,
        "qLogExpectedImprovement": qLogExpectedImprovement,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "optimize_acqf": optimize_acqf,
        "SobolQMCNormalSampler": SobolQMCNormalSampler,
        "MaternKernel": MaternKernel,
        "ScaleKernel": ScaleKernel,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "fit_gpytorch_mll": fit_gpytorch_mll,
    }


def train_bo_model(full_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any]]:
    if len(full_rows) < 2:
        raise RuntimeError("At least two successful full evaluations are required before BO suggestion.")

    deps = _bo_dependencies()
    torch = deps["torch"]
    points = [
        [safe_float(row["aperture_diam"]), safe_float(row["detector_radial_shift_mm"])]
        for row in full_rows
    ]
    targets = [[safe_float(row["JI"])] for row in full_rows]
    train_x = torch.tensor(normalize_X(points, args), dtype=torch.double)
    train_y = torch.tensor(targets, dtype=torch.double)

    base_kernel = deps["MaternKernel"](
        nu=2.5,
        ard_num_dims=2,
    )
    covar_module = deps["ScaleKernel"](base_kernel)
    model = deps["SingleTaskGP"](
        train_x,
        train_y,
        covar_module=covar_module,
        outcome_transform=deps["Standardize"](m=1),
    )
    mll = deps["ExactMarginalLogLikelihood"](model.likelihood, model)
    deps["fit_gpytorch_mll"](mll)
    return model, train_x, train_y, deps


def is_duplicate_candidate(
    candidate_norm: Sequence[float],
    existing_norm: Sequence[Sequence[float]],
    duplicate_tol: float,
) -> bool:
    for existing in existing_norm:
        distance = math.sqrt(
            (float(candidate_norm[0]) - float(existing[0])) ** 2
            + (float(candidate_norm[1]) - float(existing[1])) ** 2
        )
        if distance <= duplicate_tol:
            return True
    return False


def suggest_candidate_mc_ei(
    model: Any,
    train_y: Any,
    args: argparse.Namespace,
    existing_norm: Sequence[Sequence[float]],
    iteration: int,
    deps: dict[str, Any],
) -> tuple[Candidate, float, float, float]:
    torch = deps["torch"]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    try:
        sampler = deps["SobolQMCNormalSampler"](sample_shape=torch.Size([args.mc_samples]))
    except TypeError:
        sampler = deps["SobolQMCNormalSampler"](num_samples=args.mc_samples)

    acq = deps["qLogExpectedImprovement"](
        model=model,
        best_f=train_y.max(),
        sampler=sampler,
    )
    candidates, acq_value = deps["optimize_acqf"](
        acq_function=acq,
        bounds=bounds,
        q=1,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
    )
    candidate_norm = candidates.detach().view(-1).to(dtype=torch.double)

    if is_duplicate_candidate(candidate_norm.tolist(), existing_norm, args.duplicate_tol):
        random_points = torch.rand(max(args.raw_samples, 128), 2, dtype=torch.double)
        keep_mask = [
            not is_duplicate_candidate(point.tolist(), existing_norm, args.duplicate_tol)
            for point in random_points
        ]
        kept_points = random_points[keep_mask]
        if len(kept_points) == 0:
            noise = torch.randn(2, dtype=torch.double) * max(args.duplicate_tol, 1e-6)
            candidate_norm = torch.clamp(candidate_norm + noise, 0.0, 1.0)
            acq_value = acq(candidate_norm.view(1, 1, 2)).detach().view(-1)[0]
        else:
            values = acq(kept_points.unsqueeze(1)).detach().view(-1)
            best_idx = int(torch.argmax(values).item())
            candidate_norm = kept_points[best_idx]
            acq_value = values[best_idx]

    candidate_norm = torch.clamp(candidate_norm, 0.0, 1.0)
    with torch.no_grad():
        posterior = model.posterior(candidate_norm.view(1, 2))
        predicted_full_JI = float(posterior.mean.detach().view(-1)[0].item())
        posterior_std = float(posterior.variance.sqrt().detach().view(-1)[0].item())
    acquisition_value = float(acq_value.detach().view(-1)[0].item())
    aperture_diam, detector_shift = unnormalize_X([candidate_norm.tolist()], args)[0]
    candidate = Candidate(
        candidate_id=f"bo_{iteration:03d}",
        aperture_diam=aperture_diam,
        detector_radial_shift_mm=detector_shift,
        source="bo",
        iteration=iteration,
    )
    return candidate, predicted_full_JI, posterior_std, acquisition_value


def quantile(values: Sequence[float], q: float) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return math.nan
    if len(finite_values) == 1:
        return finite_values[0]
    clamped_q = min(1.0, max(0.0, q))
    position = (len(finite_values) - 1) * clamped_q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite_values[lower]
    weight = position - lower
    return finite_values[lower] * (1.0 - weight) + finite_values[upper] * weight


def promotion_decision(
    cheap_result: EvalResult,
    previous_cheap_rows: list[dict[str, str]],
    best_full_JI_before: float,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    cheap_JI = cheap_result.JI
    if cheap_result.status != "success" or not math.isfinite(cheap_JI):
        return False, "cheap_failed"

    previous_cheap_values = [safe_float(row.get("JI")) for row in previous_cheap_rows]
    if len(previous_cheap_values) < args.min_cheap_history_for_gate:
        return True, "insufficient_cheap_history"

    reasons: list[str] = []
    threshold = quantile(previous_cheap_values, args.cheap_promote_quantile)
    if math.isfinite(threshold) and cheap_JI >= threshold:
        reasons.append(f"cheap_JI_ge_q{args.cheap_promote_quantile:g}")
    if (
        math.isfinite(cheap_result.predicted_full_JI)
        and math.isfinite(best_full_JI_before)
        and cheap_result.predicted_full_JI >= best_full_JI_before
    ):
        reasons.append("predicted_full_JI_ge_best")
    if args.min_acq_value is not None and math.isfinite(cheap_result.acquisition_value):
        if cheap_result.acquisition_value >= args.min_acq_value:
            reasons.append("acquisition_value_ge_min")

    if reasons:
        return True, "+".join(reasons)
    return False, "below_promotion_thresholds"


def record_result(
    result: EvalResult,
    rows: list[dict[str, str]],
    paths: dict[str, Path],
    args: argparse.Namespace,
    current_bo_iteration: int,
) -> None:
    append_result_csv(paths["results_csv"], result)
    append_result_jsonl(paths["results_jsonl"], result)
    if is_training_result(result.to_csv_row()):
        append_training_history(paths["training_csv"], result)
    rows.append(result.to_csv_row())
    save_state(paths["state_json"], args, rows, current_bo_iteration)


def candidate_from_row(row: dict[str, str]) -> Candidate:
    return Candidate(
        candidate_id=row["candidate_id"],
        aperture_diam=safe_float(row["aperture_diam"]),
        detector_radial_shift_mm=safe_float(row["detector_radial_shift_mm"]),
        source=row.get("source", "bo"),
        iteration=int(safe_float(row.get("iteration", 0))),
    )


def run_lhs_phase(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    paths: dict[str, Path],
    supported_flags: set[str],
) -> None:
    for candidate in latin_hypercube_candidates(args):
        existing = completed_existing_result(rows, candidate.candidate_id, FULL_STAGE.stage_name, args.dry_run)
        if existing is not None and not args.force_rerun:
            continue
        best_row = best_full_row(rows)
        best_before = safe_float(best_row.get("JI")) if best_row else math.nan
        result = run_pipeline_eval(
            args,
            candidate,
            FULL_STAGE,
            supported_flags,
            best_full_JI_before=best_before,
            promoted_to_full=True,
            promotion_reason="lhs_full_warm_start",
        )
        record_result(result, rows, paths, args, current_bo_iteration=-1)
        if args.fail_fast and result.status == "failed":
            raise SystemExit(1)


def run_baseline_only(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    paths: dict[str, Path],
    supported_flags: set[str],
) -> None:
    candidate = Candidate(
        candidate_id="baseline",
        aperture_diam=args.baseline_aperture_diam,
        detector_radial_shift_mm=args.baseline_detector_radial_shift_mm,
        source="baseline",
        iteration=0,
    )
    existing = completed_existing_result(rows, candidate.candidate_id, FULL_STAGE.stage_name, args.dry_run)
    if existing is not None and not args.force_rerun:
        save_state(paths["state_json"], args, rows, current_bo_iteration=0)
        return
    best_row = best_full_row(rows)
    best_before = safe_float(best_row.get("JI")) if best_row else math.nan
    result = run_pipeline_eval(
        args,
        candidate,
        FULL_STAGE,
        supported_flags,
        best_full_JI_before=best_before,
        promoted_to_full=True,
        promotion_reason="baseline_only",
    )
    record_result(result, rows, paths, args, current_bo_iteration=0)
    if args.fail_fast and result.status == "failed":
        raise SystemExit(1)


def next_bo_candidate_from_existing(rows: list[dict[str, str]], iteration: int) -> Candidate | None:
    candidate_id = f"bo_{iteration:03d}"
    for row in reversed(rows):
        if row.get("candidate_id") == candidate_id:
            return candidate_from_row(row)
    return None


def run_bo_phase(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    paths: dict[str, Path],
    supported_flags: set[str],
) -> None:
    for iteration in range(args.bo_iters):
        candidate_id = f"bo_{iteration:03d}"
        existing_full = completed_existing_result(rows, candidate_id, FULL_STAGE.stage_name, args.dry_run)
        existing_cheap = completed_existing_result(rows, candidate_id, CHEAP_STAGE.stage_name, args.dry_run)
        if existing_full is not None and not args.force_rerun:
            save_state(paths["state_json"], args, rows, iteration)
            continue
        if (
            existing_cheap is not None
            and not args.force_rerun
            and existing_cheap.get("promoted_to_full") == "false"
        ):
            save_state(paths["state_json"], args, rows, iteration)
            continue

        full_history = get_successful_full_history(rows)
        if len(full_history) < 2:
            message = "Skipping BO phase: at least two successful full evaluations are required."
            if args.dry_run:
                print(message)
                save_state(paths["state_json"], args, rows, iteration)
                return
            raise RuntimeError(message)

        candidate = next_bo_candidate_from_existing(rows, iteration)
        predicted_full_JI = math.nan
        posterior_std = math.nan
        acquisition_value = math.nan
        if candidate is None or args.force_rerun:
            model, _train_x, train_y, deps = train_bo_model(full_history, args)
            existing_norm = normalize_X(get_candidate_params_from_rows(rows), args)
            candidate, predicted_full_JI, posterior_std, acquisition_value = suggest_candidate_mc_ei(
                model,
                train_y,
                args,
                existing_norm,
                iteration,
                deps,
            )
        else:
            candidate = Candidate(
                candidate_id=candidate.candidate_id,
                aperture_diam=candidate.aperture_diam,
                detector_radial_shift_mm=candidate.detector_radial_shift_mm,
                source="bo",
                iteration=iteration,
            )
            source_row = latest_result_for(rows, candidate_id, CHEAP_STAGE.stage_name) or {}
            predicted_full_JI = safe_float(source_row.get("predicted_full_JI"))
            posterior_std = safe_float(source_row.get("posterior_std"))
            acquisition_value = safe_float(source_row.get("acquisition_value"))

        best_row = best_full_row(rows)
        best_before = safe_float(best_row.get("JI")) if best_row else math.nan

        if existing_cheap is None or args.force_rerun:
            previous_cheap_rows = get_successful_cheap_history(rows)
            cheap_result = run_pipeline_eval(
                args,
                candidate,
                CHEAP_STAGE,
                supported_flags,
                predicted_full_JI=predicted_full_JI,
                posterior_std=posterior_std,
                acquisition_value=acquisition_value,
                best_full_JI_before=best_before,
            )
            if args.dry_run:
                cheap_result.promoted_to_full = False
                cheap_result.promotion_reason = "dry_run_not_promoted"
                record_result(cheap_result, rows, paths, args, iteration)
                save_state(paths["state_json"], args, rows, iteration)
                continue

            promoted, reason = promotion_decision(cheap_result, previous_cheap_rows, best_before, args)
            cheap_result.promoted_to_full = promoted
            cheap_result.promotion_reason = reason
            record_result(cheap_result, rows, paths, args, iteration)
            if args.fail_fast and cheap_result.status == "failed":
                raise SystemExit(1)
        else:
            promoted = existing_cheap.get("promoted_to_full") == "true"
            reason = existing_cheap.get("promotion_reason", "")

        if not promoted:
            continue

        full_result = run_pipeline_eval(
            args,
            candidate,
            FULL_STAGE,
            supported_flags,
            predicted_full_JI=predicted_full_JI,
            posterior_std=posterior_std,
            acquisition_value=acquisition_value,
            best_full_JI_before=best_before,
            promoted_to_full=True,
            promotion_reason=reason,
        )
        record_result(full_result, rows, paths, args, iteration)
        if args.fail_fast and full_result.status == "failed":
            raise SystemExit(1)


def self_test_bo(args: argparse.Namespace) -> None:
    fake_rows = [
        {
            "stage": "full",
            "status": "success",
            "aperture_diam": str(aperture_diam),
            "detector_radial_shift_mm": str(detector_shift),
            "JI": str(ji),
        }
        for aperture_diam, detector_shift, ji in [
            (0.41, -18.0, 0.10),
            (0.48, -8.0, 0.22),
            (0.55, 0.0, 0.35),
            (0.62, 9.0, 0.31),
            (0.69, 18.0, 0.18),
        ]
    ]
    model, _train_x, train_y, deps = train_bo_model(fake_rows, args)
    nu = float(model.covar_module.base_kernel.nu)
    if nu != 2.5:
        raise AssertionError(f"Expected Matérn nu=2.5, got {nu}")
    existing_norm = normalize_X(get_candidate_params_from_rows(fake_rows), args)
    candidate, predicted, posterior_std, acquisition_value = suggest_candidate_mc_ei(
        model,
        train_y,
        args,
        existing_norm,
        iteration=0,
        deps=deps,
    )
    if not args.diam_min <= candidate.aperture_diam <= args.diam_max:
        raise AssertionError("Suggested aperture_diam is outside bounds")
    if not args.shift_min <= candidate.detector_radial_shift_mm <= args.shift_max:
        raise AssertionError("Suggested detector_radial_shift_mm is outside bounds")
    print("BO self-test passed")
    print("kernel: MaternKernel(nu=2.5, ard_num_dims=2) inside ScaleKernel")
    print(f"suggested: {candidate}")
    print(f"predicted_full_JI={predicted:.6g} posterior_std={posterior_std:.6g}")
    print(f"acquisition_value={acquisition_value:.6g}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test_bo:
        self_test_bo(args)
        return 0

    paths = ensure_dirs(args)
    run_pipeline_path = resolve_path(args.run_pipeline)
    if not run_pipeline_path.exists():
        raise FileNotFoundError(f"run_pipeline.py not found: {run_pipeline_path}")
    supported_flags = run_pipeline_supported_flags(run_pipeline_path)
    require_pipeline_flags(supported_flags)

    if args.resume:
        _state = load_state(paths["state_json"])
    rows = load_results_csv(paths["results_csv"])

    if args.baseline_only:
        run_baseline_only(args, rows, paths, supported_flags)
        return 0

    run_lhs_phase(args, rows, paths, supported_flags)
    run_bo_phase(args, rows, paths, supported_flags)
    save_state(paths["state_json"], args, rows, args.bo_iters)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
