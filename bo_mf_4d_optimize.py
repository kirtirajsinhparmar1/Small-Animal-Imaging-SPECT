#!/usr/bin/env python3
"""
Multi-fidelity 4D Bayesian optimizer for active-detector SC-SPECT designs.

This optimizer is intentionally standalone.  It does not import or reuse the
older BO/Hyperband optimizer architecture.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_N_APERTURES_VALUES = [60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
DEFAULT_RING1_VALUES = [120, 160, 200, 240, 280, 320, 360, 400, 440, 480]
DEFAULT_RING2_VALUES = [240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720]

RESULTS_CSV = "bo_mf_results.csv"
SUMMARY_CSV = "bo_mf_candidate_summary.csv"
INFEASIBLE_CSV = "bo_mf_infeasible_candidates.csv"
STATE_JSON = "bo_mf_state.json"


@dataclass
class Candidate:
    candidate_id: str
    source: str
    iteration: int
    aperture_diam_mm: float
    n_apertures: int
    n_crystals_ring1: int
    n_crystals_ring2: int


@dataclass
class EvalResult:
    candidate_id: str
    stage: str
    fidelity: str
    metric_fidelity: str
    aperture_diam_mm: float
    n_apertures: int
    n_crystals_ring1: int
    n_crystals_ring2: int
    srm_row_fraction: float
    srm_row_mode: str
    srm_row_seed: int
    fwhm_mean: float
    sensitivity_total: float
    sensitivity_mean: float
    asci_pct: float
    JI: float
    status: str
    error_message: str
    run_name: str
    run_dir: str
    command: str
    started_at: str
    finished_at: str
    elapsed_sec: float


@dataclass
class CandidateSummary:
    candidate_id: str
    source: str
    iteration: int
    aperture_diam_mm: float
    n_apertures: int
    n_crystals_ring1: int
    n_crystals_ring2: int
    geometry_status: str
    feasibility_reason: str
    JI_srm50: float
    fwhm_srm50: float
    sensitivity_srm50: float
    asci_srm50: float
    elapsed_srm50: float
    JI_srm100: float
    fwhm_srm100: float
    sensitivity_srm100: float
    asci_srm100: float
    elapsed_srm100: float
    promoted_to_full: bool
    promotion_reason: str
    best_full_JI_before: float
    predicted_full_mean: float
    predicted_full_std: float
    calibrated_full_mean: float
    calibrated_full_std: float
    full_ucb: float
    calibrated_ucb: float
    acquisition_value: float
    status: str
    r_eff: float = math.nan
    aperture_angular_width: float = math.nan
    aperture_cell_angular_width: float = math.nan


@dataclass
class InfeasibleCandidate:
    candidate_id: str
    source: str
    iteration: int
    aperture_diam_mm: float
    n_apertures: int
    n_crystals_ring1: int
    n_crystals_ring2: int
    r_eff: float
    aperture_angular_width: float
    aperture_cell_angular_width: float
    geometry_status: str
    feasibility_reason: str
    timestamp: str


@dataclass
class SearchSpace:
    diam_min: float
    diam_max: float
    n_apertures_values: list[int]
    ring1_values: list[int]
    ring2_values: list[int]


@dataclass
class GPBundle:
    full_model: Any = None
    cheap_model: Any = None
    calibration_model: Any = None
    cheap_y_min: float = math.nan
    cheap_y_max: float = math.nan
    acquisition_name: str = ""
    error_message: str = ""


@dataclass
class FeasibilityResult:
    is_feasible: bool
    reason: str
    r_eff: float | None
    aperture_angular_width: float | None
    aperture_cell_angular_width: float | None

    @property
    def feasible(self) -> bool:
        return self.is_feasible


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated integer list")
    return values


def shell_join(cmd: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def finite_or_nan(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def reason_code(reason: str) -> str:
    first = str(reason or "unknown").split(";", 1)[0].strip()
    return first.split(":", 1)[0].strip() or "unknown"


def format_reason_counts(rows: Iterable[InfeasibleCandidate]) -> str:
    counts = Counter(reason_code(row.feasibility_reason) for row in rows)
    if not counts:
        return "none"
    return ", ".join(f"{reason}={count}" for reason, count in counts.most_common())


def geometry_failure_from_output(output: str) -> bool:
    lowered = output.lower()
    terms = [
        "aperture angular width",
        "not smaller than cell width",
        "infeasible",
        "invalid geometry",
        "opening ratio",
    ]
    return any(term in lowered for term in terms)


EXISTING_RUN_DIR_MESSAGE = "Run directory already exists and is not empty"
EXISTING_RUN_DIR_ERROR = (
    "Existing eval directory found. Rerun optimizer with --resume, delete the eval directory, "
    "or choose a new --out-dir."
)


def existing_run_dir_failure_from_output(output: str) -> bool:
    return EXISTING_RUN_DIR_MESSAGE.lower() in str(output or "").lower()


def output_tail(output: str, max_chars: int = 4000) -> str:
    cleaned = str(output or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[-max_chars:]


def aperture_feasibility_check(
    candidate: Candidate,
    aperture_ring_radius_mm: float,
    opening_ratio_min: float,
    opening_ratio_max: float,
) -> FeasibilityResult:
    d = float(candidate.aperture_diam_mm)
    n = int(candidate.n_apertures)
    r = float(aperture_ring_radius_mm)
    reasons: list[str] = []

    if d <= 0.0:
        reasons.append(f"invalid_aperture_diam: aperture_diam_mm <= 0 ({d:g})")
    if n <= 0:
        reasons.append(f"invalid_n_apertures: n_apertures <= 0 ({n})")
    if r <= 0.0:
        reasons.append(f"invalid_aperture_ring_radius: aperture_ring_radius_mm <= 0 ({r:g})")
    if opening_ratio_min < 0.0:
        reasons.append(f"invalid_opening_ratio_min: opening_ratio_min < 0 ({opening_ratio_min:g})")
    if opening_ratio_max <= opening_ratio_min:
        reasons.append(
            "invalid_opening_ratio_bounds: "
            f"opening_ratio_max <= opening_ratio_min ({opening_ratio_max:g} <= {opening_ratio_min:g})"
        )
    if reasons:
        return FeasibilityResult(False, "; ".join(reasons), math.nan, math.nan, math.nan)

    asin_arg = (d / 2.0) / r
    if asin_arg > 1.0:
        aperture_angular_width = math.nan
        reasons.append(f"aperture_too_large_for_radius: (aperture_diam_mm / 2) / R > 1 ({asin_arg:.6g})")
    else:
        aperture_angular_width = 2.0 * math.asin(asin_arg)

    cell_angular_width = 2.0 * math.pi / float(n)
    r_eff = float(n) * d / (2.0 * math.pi * r)

    if math.isfinite(aperture_angular_width) and aperture_angular_width >= cell_angular_width:
        reasons.append(
            "aperture_too_large_for_cell: aperture_angular_width >= cell_angular_width "
            f"({aperture_angular_width:.6g} >= {cell_angular_width:.6g})"
        )
    if r_eff < opening_ratio_min:
        reasons.append(f"opening_ratio_too_low: r_eff < opening_ratio_min ({r_eff:.6g} < {opening_ratio_min:.6g})")
    if r_eff > opening_ratio_max:
        reasons.append(f"opening_ratio_too_high: r_eff > opening_ratio_max ({r_eff:.6g} > {opening_ratio_max:.6g})")

    return FeasibilityResult(
        is_feasible=not reasons,
        reason="; ".join(reasons),
        r_eff=r_eff,
        aperture_angular_width=aperture_angular_width,
        aperture_cell_angular_width=cell_angular_width,
    )


def nearest_allowed(value: float, allowed: list[int]) -> int:
    return int(min(allowed, key=lambda candidate: abs(candidate - value)))


def candidate_key(candidate: Candidate) -> tuple[float, int, int, int]:
    return (
        round(float(candidate.aperture_diam_mm), 5),
        int(candidate.n_apertures),
        int(candidate.n_crystals_ring1),
        int(candidate.n_crystals_ring2),
    )


def normalize_candidate(candidate: Candidate, space: SearchSpace) -> np.ndarray:
    diam_span = space.diam_max - space.diam_min
    if diam_span == 0.0:
        aperture_norm = 0.0
    else:
        aperture_norm = (candidate.aperture_diam_mm - space.diam_min) / diam_span
    aperture_count_norm = (
        (candidate.n_apertures - min(space.n_apertures_values))
        / (max(space.n_apertures_values) - min(space.n_apertures_values))
    )
    ring1_norm = (
        (candidate.n_crystals_ring1 - min(space.ring1_values))
        / (max(space.ring1_values) - min(space.ring1_values))
    )
    ring2_norm = (
        (candidate.n_crystals_ring2 - min(space.ring2_values))
        / (max(space.ring2_values) - min(space.ring2_values))
    )
    return np.asarray([aperture_norm, aperture_count_norm, ring1_norm, ring2_norm], dtype=np.float64)


def materialize_candidate(
    candidate_id: str,
    source: str,
    iteration: int,
    z: Iterable[float],
    space: SearchSpace,
) -> Candidate:
    z_arr = np.clip(np.asarray(list(z), dtype=np.float64), 0.0, 1.0)
    if space.diam_min == space.diam_max:
        aperture = space.diam_min
    else:
        aperture = space.diam_min + z_arr[0] * (space.diam_max - space.diam_min)
    n_apertures_raw = min(space.n_apertures_values) + z_arr[1] * (
        max(space.n_apertures_values) - min(space.n_apertures_values)
    )
    ring1_raw = min(space.ring1_values) + z_arr[2] * (
        max(space.ring1_values) - min(space.ring1_values)
    )
    ring2_raw = min(space.ring2_values) + z_arr[3] * (
        max(space.ring2_values) - min(space.ring2_values)
    )
    return Candidate(
        candidate_id=candidate_id,
        source=source,
        iteration=iteration,
        aperture_diam_mm=round(float(aperture), 5),
        n_apertures=nearest_allowed(n_apertures_raw, space.n_apertures_values),
        n_crystals_ring1=nearest_allowed(ring1_raw, space.ring1_values),
        n_crystals_ring2=nearest_allowed(ring2_raw, space.ring2_values),
    )


def summary_from_candidate(candidate: Candidate) -> CandidateSummary:
    return CandidateSummary(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        iteration=candidate.iteration,
        aperture_diam_mm=candidate.aperture_diam_mm,
        n_apertures=candidate.n_apertures,
        n_crystals_ring1=candidate.n_crystals_ring1,
        n_crystals_ring2=candidate.n_crystals_ring2,
        geometry_status="not_checked",
        feasibility_reason="",
        JI_srm50=math.nan,
        fwhm_srm50=math.nan,
        sensitivity_srm50=math.nan,
        asci_srm50=math.nan,
        elapsed_srm50=math.nan,
        JI_srm100=math.nan,
        fwhm_srm100=math.nan,
        sensitivity_srm100=math.nan,
        asci_srm100=math.nan,
        elapsed_srm100=math.nan,
        promoted_to_full=False,
        promotion_reason="",
        best_full_JI_before=math.nan,
        predicted_full_mean=math.nan,
        predicted_full_std=math.nan,
        calibrated_full_mean=math.nan,
        calibrated_full_std=math.nan,
        full_ucb=math.nan,
        calibrated_ucb=math.nan,
        acquisition_value=math.nan,
        status="pending",
    )


def dataclass_from_dict(cls: type[Any], payload: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    filtered = {key: value for key, value in payload.items() if key in names}
    return cls(**filtered)


def write_dataclass_csv(path: Path | str, rows: list[Any], cls: type[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(cls)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_infeasible_csv(path: Path) -> list[InfeasibleCandidate]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    loaded: list[InfeasibleCandidate] = []
    for row in rows:
        loaded.append(
            InfeasibleCandidate(
                candidate_id=str(row.get("candidate_id", "")),
                source=str(row.get("source", "")),
                iteration=int(row.get("iteration") or 0),
                aperture_diam_mm=finite_or_nan(row.get("aperture_diam_mm")),
                n_apertures=int(float(row.get("n_apertures") or 0)),
                n_crystals_ring1=int(float(row.get("n_crystals_ring1") or 0)),
                n_crystals_ring2=int(float(row.get("n_crystals_ring2") or 0)),
                r_eff=finite_or_nan(row.get("r_eff")),
                aperture_angular_width=finite_or_nan(row.get("aperture_angular_width")),
                aperture_cell_angular_width=finite_or_nan(row.get("aperture_cell_angular_width")),
                geometry_status=str(row.get("geometry_status", "infeasible")),
                feasibility_reason=str(row.get("feasibility_reason", "")),
                timestamp=str(row.get("timestamp", "")),
            )
        )
    return loaded


def parse_ji_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JI CSV: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"JI CSV has no rows: {path}")
    return rows[-1]


def manual_lhs(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    values = np.empty((n, dim), dtype=np.float64)
    for axis in range(dim):
        perm = rng.permutation(n)
        values[:, axis] = (perm + rng.random(n)) / n
    return values


def generate_pool_normalized(n: int, dim: int, seed: int) -> np.ndarray:
    try:
        import torch

        engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
        return engine.draw(n).cpu().numpy().astype(np.float64)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, dim), dtype=np.float64)


def import_botorch_stack() -> dict[str, Any]:
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    try:
        from botorch.acquisition.logei import qLogExpectedImprovement
        acquisition_cls = qLogExpectedImprovement
        acquisition_name = "qLogExpectedImprovement"
    except Exception:
        from botorch.acquisition import qExpectedImprovement

        print("[WARN] qLogExpectedImprovement unavailable; falling back to qExpectedImprovement")
        acquisition_cls = qExpectedImprovement
        acquisition_name = "qExpectedImprovement"

    return {
        "torch": torch,
        "fit_gpytorch_mll": fit_gpytorch_mll,
        "SingleTaskGP": SingleTaskGP,
        "Standardize": Standardize,
        "MaternKernel": MaternKernel,
        "ScaleKernel": ScaleKernel,
        "ExactMarginalLogLikelihood": ExactMarginalLogLikelihood,
        "acquisition_cls": acquisition_cls,
        "acquisition_name": acquisition_name,
    }


def build_gp_model(train_x: np.ndarray, train_y: np.ndarray, stack: dict[str, Any]) -> Any:
    torch = stack["torch"]
    train_x_t = torch.as_tensor(train_x, dtype=torch.double)
    train_y_t = torch.as_tensor(train_y, dtype=torch.double).reshape(-1, 1)
    covar_module = stack["ScaleKernel"](stack["MaternKernel"](nu=2.5))
    model = stack["SingleTaskGP"](
        train_x_t,
        train_y_t,
        covar_module=covar_module,
        outcome_transform=stack["Standardize"](m=1),
    )
    mll = stack["ExactMarginalLogLikelihood"](model.likelihood, model)
    stack["fit_gpytorch_mll"](mll)
    return model


def gp_predict(model: Any, x: np.ndarray, stack: dict[str, Any]) -> tuple[float, float]:
    if model is None:
        return math.nan, math.nan
    torch = stack["torch"]
    with torch.no_grad():
        posterior = model.posterior(torch.as_tensor(x, dtype=torch.double).reshape(1, -1))
        mean = float(posterior.mean.reshape(-1)[0].item())
        variance = float(posterior.variance.reshape(-1)[0].clamp_min(0.0).item())
    return mean, math.sqrt(variance)


def model_training_rows(
    candidates: dict[str, Candidate],
    results: list[EvalResult],
    space: SearchSpace,
    fidelity: str,
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for result in results:
        if result.fidelity != fidelity or result.status != "success" or not is_finite(result.JI):
            continue
        candidate = candidates.get(result.candidate_id)
        if candidate is None:
            continue
        x_rows.append(normalize_candidate(candidate, space))
        y_rows.append(float(result.JI))
    if not x_rows:
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.float64)


def paired_training_rows(
    candidates: dict[str, Candidate],
    results: list[EvalResult],
    space: SearchSpace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    cheap_by_id: dict[str, EvalResult] = {}
    full_by_id: dict[str, EvalResult] = {}
    for result in results:
        if result.status != "success" or not is_finite(result.JI):
            continue
        if result.fidelity == "srm50":
            cheap_by_id[result.candidate_id] = result
        elif result.fidelity == "srm100":
            full_by_id[result.candidate_id] = result

    cheap_values = [float(row.JI) for row in cheap_by_id.values() if is_finite(row.JI)]
    if cheap_values:
        cheap_min = min(cheap_values)
        cheap_max = max(cheap_values)
    else:
        cheap_min = 0.0
        cheap_max = 1.0

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for candidate_id, cheap_result in cheap_by_id.items():
        full_result = full_by_id.get(candidate_id)
        candidate = candidates.get(candidate_id)
        if full_result is None or candidate is None:
            continue
        cheap_norm = normalize_scalar(float(cheap_result.JI), cheap_min, cheap_max)
        x_rows.append(np.concatenate([normalize_candidate(candidate, space), [cheap_norm]]))
        y_rows.append(float(full_result.JI))

    if not x_rows:
        return np.empty((0, 5), dtype=np.float64), np.empty((0,), dtype=np.float64), cheap_min, cheap_max
    return np.vstack(x_rows), np.asarray(y_rows, dtype=np.float64), cheap_min, cheap_max


def normalize_scalar(value: float, min_value: float, max_value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    span = max_value - min_value
    if span <= 1e-12:
        return 0.5
    return float(np.clip((value - min_value) / span, 0.0, 1.0))


class Optimizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.resume = bool(args.resume)
        self.out_dir = Path(args.out_dir)
        self.results_csv_path = self.out_dir / RESULTS_CSV
        self.summary_csv_path = self.out_dir / SUMMARY_CSV
        self.infeasible_csv_path = self.out_dir / INFEASIBLE_CSV
        self.state_json_path = self.out_dir / STATE_JSON
        self.space = SearchSpace(
            diam_min=args.diam_min,
            diam_max=args.diam_max,
            n_apertures_values=parse_int_list(args.n_apertures_values),
            ring1_values=parse_int_list(args.ring1_crystals_values),
            ring2_values=parse_int_list(args.ring2_crystals_values),
        )
        self.rng = np.random.default_rng(args.random_seed)
        self.candidates: dict[str, Candidate] = {}
        self.summaries: dict[str, CandidateSummary] = {}
        self.results: list[EvalResult] = []
        self.infeasible_candidates: list[InfeasibleCandidate] = []
        self.infeasible_keys: set[tuple[float, int, int, int]] = set()
        self.next_candidate_seq = 1
        self.current_bo_iteration = 0
        if self.resume:
            self.load_state()

    def load_state(self) -> None:
        if not self.state_json_path.exists():
            self.infeasible_candidates = read_infeasible_csv(self.infeasible_csv_path)
            self.infeasible_keys = {
                (
                    round(float(row.aperture_diam_mm), 5),
                    int(row.n_apertures),
                    int(row.n_crystals_ring1),
                    int(row.n_crystals_ring2),
                )
                for row in self.infeasible_candidates
            }
            print(f"[resume] No state found at {self.state_json_path}; starting new optimizer state.")
            return
        with self.state_json_path.open() as handle:
            state = json.load(handle)
        self.candidates = {
            item["candidate_id"]: dataclass_from_dict(Candidate, item)
            for item in state.get("candidates", [])
        }
        self.summaries = {
            item["candidate_id"]: dataclass_from_dict(CandidateSummary, item)
            for item in state.get("summaries", [])
        }
        self.results = [dataclass_from_dict(EvalResult, item) for item in state.get("results", [])]
        self.infeasible_candidates = [
            dataclass_from_dict(InfeasibleCandidate, item)
            for item in state.get("infeasible_candidates", [])
        ]
        csv_infeasible = read_infeasible_csv(self.infeasible_csv_path)
        known_infeasible_ids = {row.candidate_id for row in self.infeasible_candidates}
        self.infeasible_candidates.extend(
            row for row in csv_infeasible if row.candidate_id not in known_infeasible_ids
        )
        self.infeasible_keys = {
            (
                round(float(row.aperture_diam_mm), 5),
                int(row.n_apertures),
                int(row.n_crystals_ring1),
                int(row.n_crystals_ring2),
            )
            for row in self.infeasible_candidates
        }
        self.next_candidate_seq = int(state.get("next_candidate_seq", len(self.candidates) + 1))
        self.current_bo_iteration = int(state.get("current_bo_iteration", 0))
        print(
            f"[resume] Loaded {len(self.candidates)} candidates, "
            f"{len(self.results)} eval rows, {len(self.infeasible_candidates)} infeasible rows "
            f"from {self.state_json_path}"
        )

    def save_all(self) -> None:
        assert isinstance(self.results_csv_path, Path)
        assert isinstance(self.summary_csv_path, Path)
        assert isinstance(self.infeasible_csv_path, Path)
        assert isinstance(self.state_json_path, Path)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_dataclass_csv(self.results_csv_path, self.results, EvalResult)
        ordered_summaries = sorted(self.summaries.values(), key=lambda row: row.candidate_id)
        write_dataclass_csv(self.summary_csv_path, ordered_summaries, CandidateSummary)
        write_dataclass_csv(self.infeasible_csv_path, self.infeasible_candidates, InfeasibleCandidate)
        state = {
            "created_by": Path(__file__).name,
            "updated_at": now_iso(),
            "args": vars(self.args),
            "results_csv_path": str(self.results_csv_path),
            "summary_csv_path": str(self.summary_csv_path),
            "infeasible_csv_path": str(self.infeasible_csv_path),
            "state_json_path": str(self.state_json_path),
            "next_candidate_seq": self.next_candidate_seq,
            "current_bo_iteration": self.current_bo_iteration,
            "candidates": [asdict(row) for row in sorted(self.candidates.values(), key=lambda row: row.candidate_id)],
            "summaries": [asdict(row) for row in ordered_summaries],
            "results": [asdict(row) for row in self.results],
            "infeasible_candidates": [asdict(row) for row in self.infeasible_candidates],
        }
        with self.state_json_path.open("w") as handle:
            json.dump(state, handle, indent=2, allow_nan=True)

    def next_candidate_id(self) -> str:
        candidate_id = f"cand_{self.next_candidate_seq:04d}"
        self.next_candidate_seq += 1
        return candidate_id

    def existing_keys(self) -> set[tuple[float, int, int, int]]:
        return {candidate_key(candidate) for candidate in self.candidates.values()}

    def blocked_keys(self) -> set[tuple[float, int, int, int]]:
        return self.existing_keys() | self.infeasible_keys

    def register_candidate(self, candidate: Candidate) -> Candidate:
        self.candidates[candidate.candidate_id] = candidate
        summary = self.summaries.setdefault(candidate.candidate_id, summary_from_candidate(candidate))
        self.apply_feasibility_to_summary(candidate, summary)
        return candidate

    def check_candidate_feasibility(self, candidate: Candidate) -> FeasibilityResult:
        reasons: list[str] = []
        if not self.space.diam_min <= float(candidate.aperture_diam_mm) <= self.space.diam_max:
            reasons.append(
                "invalid_aperture_diam: "
                f"{candidate.aperture_diam_mm:g} outside [{self.space.diam_min:g}, {self.space.diam_max:g}]"
            )
        if int(candidate.n_apertures) not in self.space.n_apertures_values:
            reasons.append(f"invalid_n_apertures: {candidate.n_apertures} not in allowed values")
        if int(candidate.n_crystals_ring1) not in self.space.ring1_values:
            reasons.append(f"invalid_ring1_count: {candidate.n_crystals_ring1} not in allowed values")
        if int(candidate.n_crystals_ring2) not in self.space.ring2_values:
            reasons.append(f"invalid_ring2_count: {candidate.n_crystals_ring2} not in allowed values")

        aperture_check = aperture_feasibility_check(
            candidate,
            aperture_ring_radius_mm=float(self.args.aperture_ring_radius_mm),
            opening_ratio_min=float(self.args.opening_ratio_min),
            opening_ratio_max=float(self.args.opening_ratio_max),
        )
        if aperture_check.reason:
            reasons.append(aperture_check.reason)
        return FeasibilityResult(
            is_feasible=not reasons,
            reason="; ".join(reasons),
            r_eff=aperture_check.r_eff,
            aperture_angular_width=aperture_check.aperture_angular_width,
            aperture_cell_angular_width=aperture_check.aperture_cell_angular_width,
        )

    def apply_feasibility_to_summary(self, candidate: Candidate, summary: CandidateSummary) -> FeasibilityResult:
        check = self.check_candidate_feasibility(candidate)
        summary.r_eff = finite_or_nan(check.r_eff)
        summary.aperture_angular_width = finite_or_nan(check.aperture_angular_width)
        summary.aperture_cell_angular_width = finite_or_nan(check.aperture_cell_angular_width)
        if check.feasible:
            if summary.geometry_status in {"not_checked", "failed_geometry"}:
                summary.geometry_status = "preflight_feasible"
            if summary.status == "failed_geometry":
                summary.status = "pending"
            if summary.feasibility_reason.startswith("aperture_") or summary.feasibility_reason.startswith("r_eff"):
                summary.feasibility_reason = ""
        else:
            summary.geometry_status = "failed_geometry"
            summary.feasibility_reason = check.reason
            summary.status = "failed_geometry"
        return check

    def log_infeasible_candidate(self, candidate: Candidate, check: FeasibilityResult) -> bool:
        key = candidate_key(candidate)
        if key in self.infeasible_keys:
            return False

        self.infeasible_keys.add(key)
        self.candidates.setdefault(candidate.candidate_id, candidate)
        summary = self.summaries.setdefault(candidate.candidate_id, summary_from_candidate(candidate))
        summary.r_eff = finite_or_nan(check.r_eff)
        summary.aperture_angular_width = finite_or_nan(check.aperture_angular_width)
        summary.aperture_cell_angular_width = finite_or_nan(check.aperture_cell_angular_width)
        summary.geometry_status = "infeasible"
        summary.feasibility_reason = check.reason
        summary.status = "infeasible"
        self.infeasible_candidates.append(
            InfeasibleCandidate(
                candidate_id=candidate.candidate_id,
                source=candidate.source,
                iteration=candidate.iteration,
                aperture_diam_mm=candidate.aperture_diam_mm,
                n_apertures=candidate.n_apertures,
                n_crystals_ring1=candidate.n_crystals_ring1,
                n_crystals_ring2=candidate.n_crystals_ring2,
                r_eff=finite_or_nan(check.r_eff),
                aperture_angular_width=finite_or_nan(check.aperture_angular_width),
                aperture_cell_angular_width=finite_or_nan(check.aperture_cell_angular_width),
                geometry_status="infeasible",
                feasibility_reason=check.reason,
                timestamp=now_iso(),
            )
        )
        return True

    def rejection_counts_since(self, start_index: int) -> str:
        return format_reason_counts(self.infeasible_candidates[start_index:])

    def successful_result(self, candidate_id: str, fidelity: str) -> EvalResult | None:
        for result in reversed(self.results):
            if result.candidate_id == candidate_id and result.fidelity == fidelity and result.status == "success":
                return result
        return None

    def best_full_ji(self) -> float:
        values = [
            float(result.JI)
            for result in self.results
            if result.fidelity == "srm100" and result.status == "success" and is_finite(result.JI)
        ]
        return max(values) if values else math.nan

    def lhs_candidates(self, needed: int, source: str, iteration: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        existing = self.blocked_keys()
        attempts = 0
        batch_size = max(needed * 4, 32)
        rejection_start = len(self.infeasible_candidates)
        while len(candidates) < needed and attempts < self.args.lhs_max_attempts:
            attempts += 1
            z_rows = manual_lhs(batch_size, 4, self.rng)
            for z in z_rows:
                candidate = materialize_candidate(
                    candidate_id=self.next_candidate_id(),
                    source=source,
                    iteration=iteration,
                    z=z,
                    space=self.space,
                )
                key = candidate_key(candidate)
                if key in existing:
                    continue
                check = self.check_candidate_feasibility(candidate)
                if not check.feasible:
                    self.log_infeasible_candidate(candidate, check)
                    existing.add(key)
                    continue
                existing.add(key)
                candidates.append(candidate)
                if len(candidates) >= needed:
                    break
        if len(candidates) < needed:
            self.save_all()
            raise ValueError(
                f"Could only generate {len(candidates)} unique feasible LHS candidates after "
                f"{attempts} attempts; requested {needed}. "
                f"Rejection reasons: {self.rejection_counts_since(rejection_start)}. "
                "Widen the search space or relax --opening-ratio-min/--opening-ratio-max."
            )
        return candidates

    def fidelity_settings(self, candidate: Candidate, fidelity: str) -> tuple[float, str, int, str]:
        if fidelity == "srm50":
            return (
                float(self.args.cheap_srm_row_fraction),
                str(self.args.cheap_srm_row_mode),
                int(self.args.cheap_srm_row_seed),
                f"evals/{candidate.candidate_id}_srm50",
            )
        if fidelity == "srm100":
            return (
                float(self.args.full_srm_row_fraction),
                str(self.args.cheap_srm_row_mode),
                int(self.args.cheap_srm_row_seed),
                f"evals/{candidate.candidate_id}_srm100",
            )
        raise ValueError(f"Unknown fidelity: {fidelity}")

    def build_pipeline_command(self, candidate: Candidate, fidelity: str) -> tuple[list[Any], str, Path, float, str, int]:
        fraction, mode, seed, run_name = self.fidelity_settings(candidate, fidelity)

        cmd: list[Any] = [
            sys.executable,
            "run_pipeline.py",
            "--runs-dir",
            self.out_dir,
            "--run-name",
            run_name,
            "--layout-idxs",
            self.args.layout_idxs,
            "--t8-poses",
            self.args.t8_poses,
            "--aperture-diam",
            candidate.aperture_diam_mm,
            "--n-apertures",
            candidate.n_apertures,
            "--n-crystals-ring1",
            candidate.n_crystals_ring1,
            "--n-crystals-ring2",
            candidate.n_crystals_ring2,
            "--srm-row-fraction",
            fraction,
            "--srm-row-mode",
            mode,
            "--srm-row-seed",
            seed,
        ]
        optional_pairs = [
            ("--cpus", self.args.cpus),
            ("--pose-workers", self.args.pose_workers),
            ("--torch-threads", self.args.torch_threads),
            ("--torch-interop-threads", self.args.torch_interop_threads),
        ]
        for flag, value in optional_pairs:
            if value is not None:
                cmd.extend([flag, value])
        if self.args.skip_recon:
            cmd.append("--skip-recon")
        if self.resume:
            cmd.append("--resume")
        return [str(part) for part in cmd], run_name, self.out_dir / run_name, fraction, mode, seed

    def run_eval(self, candidate: Candidate, fidelity: str, stage: str) -> EvalResult:
        existing = self.successful_result(candidate.candidate_id, fidelity)
        if existing is not None:
            print(f"[skip] {candidate.candidate_id} {fidelity} already successful: JI={existing.JI}")
            return existing

        summary = self.summaries.setdefault(candidate.candidate_id, summary_from_candidate(candidate))
        feasibility = self.apply_feasibility_to_summary(candidate, summary)
        if not feasibility.feasible:
            self.log_infeasible_candidate(candidate, feasibility)
            fraction, mode, seed, run_name = self.fidelity_settings(candidate, fidelity)
            started_at = now_iso()
            result = EvalResult(
                candidate_id=candidate.candidate_id,
                stage=stage,
                fidelity=fidelity,
                metric_fidelity="srm_row50" if fidelity == "srm50" else "full",
                aperture_diam_mm=candidate.aperture_diam_mm,
                n_apertures=candidate.n_apertures,
                n_crystals_ring1=candidate.n_crystals_ring1,
                n_crystals_ring2=candidate.n_crystals_ring2,
                srm_row_fraction=fraction,
                srm_row_mode=mode,
                srm_row_seed=seed,
                fwhm_mean=math.nan,
                sensitivity_total=math.nan,
                sensitivity_mean=math.nan,
                asci_pct=math.nan,
                JI=math.nan,
                status="failed_geometry",
                error_message=feasibility.reason,
                run_name=run_name,
                run_dir=str(self.out_dir / run_name),
                command="",
                started_at=started_at,
                finished_at=now_iso(),
                elapsed_sec=0.0,
            )
            self.results.append(result)
            self.update_summary_from_result(candidate, result)
            self.save_all()
            print(f"[skip] {candidate.candidate_id} {fidelity}: failed_geometry: {feasibility.reason}")
            return result

        cmd, run_name, run_dir, fraction, mode, seed = self.build_pipeline_command(candidate, fidelity)
        command_string = shell_join(cmd)
        started_at = now_iso()
        start_time = time.monotonic()
        status = "success"
        error_message = ""
        metric_row: dict[str, Any] = {}

        if self.args.dry_run:
            print(f"[dry-run] {candidate.candidate_id} {fidelity}: {command_string}")
            status = "dry_run"
            metric_fidelity = "srm_row50" if fidelity == "srm50" else "full"
            elapsed = 0.0
        else:
            print(f"[run] {candidate.candidate_id} {fidelity}: {command_string}")
            try:
                completed = subprocess.run(cmd, cwd=Path.cwd(), check=False, capture_output=True, text=True)
                if completed.returncode != 0:
                    combined_output = f"{completed.stdout or ''}\n{completed.stderr or ''}"
                    if existing_run_dir_failure_from_output(combined_output):
                        status = "failed_existing_run_dir"
                        error_message = (
                            f"{EXISTING_RUN_DIR_ERROR} "
                            f"Pipeline exited with code {completed.returncode}. "
                            f"{output_tail(combined_output)}"
                        ).strip()
                    else:
                        status = "failed_geometry" if geometry_failure_from_output(combined_output) else "failed_pipeline"
                        error_message = (
                            f"Pipeline exited with code {completed.returncode}. "
                            f"{output_tail(combined_output)}"
                        ).strip()
                else:
                    metric_row = parse_ji_csv(run_dir / "results" / "ji_metrics.csv")
            except Exception as exc:
                status = "failed_pipeline"
                error_message = f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - start_time
            metric_fidelity = str(metric_row.get("metric_fidelity", ""))
            if status == "success" and not is_finite(metric_row.get("JI")):
                status = "failed_pipeline"
                error_message = "JI metric is missing or not finite"

        result = EvalResult(
            candidate_id=candidate.candidate_id,
            stage=stage,
            fidelity=fidelity,
            metric_fidelity=metric_fidelity,
            aperture_diam_mm=candidate.aperture_diam_mm,
            n_apertures=candidate.n_apertures,
            n_crystals_ring1=candidate.n_crystals_ring1,
            n_crystals_ring2=candidate.n_crystals_ring2,
            srm_row_fraction=fraction,
            srm_row_mode=mode,
            srm_row_seed=seed,
            fwhm_mean=finite_or_nan(metric_row.get("fwhm_mean")),
            sensitivity_total=finite_or_nan(metric_row.get("sensitivity_total")),
            sensitivity_mean=finite_or_nan(metric_row.get("sensitivity_mean")),
            asci_pct=finite_or_nan(metric_row.get("asci_pct")),
            JI=finite_or_nan(metric_row.get("JI")),
            status=status,
            error_message=error_message,
            run_name=run_name,
            run_dir=str(run_dir),
            command=command_string,
            started_at=started_at,
            finished_at=now_iso(),
            elapsed_sec=elapsed,
        )
        if result.status == "failed_geometry":
            self.log_infeasible_candidate(
                candidate,
                FeasibilityResult(
                    is_feasible=False,
                    reason=result.error_message,
                    r_eff=summary.r_eff,
                    aperture_angular_width=summary.aperture_angular_width,
                    aperture_cell_angular_width=summary.aperture_cell_angular_width,
                ),
            )
        self.results.append(result)
        self.update_summary_from_result(candidate, result)
        self.save_all()
        if result.status == "success":
            print(f"[ok] {candidate.candidate_id} {fidelity}: JI={result.JI:.6g}")
        elif result.status == "dry_run":
            print(f"[dry-run] planned {candidate.candidate_id} {fidelity}")
        else:
            print(f"[fail] {candidate.candidate_id} {fidelity}: {result.error_message}")
        return result

    def update_summary_from_result(self, candidate: Candidate, result: EvalResult) -> None:
        summary = self.summaries.setdefault(candidate.candidate_id, summary_from_candidate(candidate))
        if result.fidelity == "srm50":
            summary.JI_srm50 = result.JI
            summary.fwhm_srm50 = result.fwhm_mean
            summary.sensitivity_srm50 = result.sensitivity_mean
            summary.asci_srm50 = result.asci_pct
            summary.elapsed_srm50 = result.elapsed_sec
        elif result.fidelity == "srm100":
            summary.JI_srm100 = result.JI
            summary.fwhm_srm100 = result.fwhm_mean
            summary.sensitivity_srm100 = result.sensitivity_mean
            summary.asci_srm100 = result.asci_pct
            summary.elapsed_srm100 = result.elapsed_sec
            summary.promoted_to_full = result.status in {"success", "dry_run"}
        if result.status == "success":
            summary.geometry_status = "pipeline_success"
            summary.status = "success"
        elif result.status == "dry_run":
            summary.geometry_status = "dry_run"
            summary.status = "dry_run_planned"
        elif result.status == "failed_geometry":
            summary.geometry_status = "failed_geometry"
            summary.feasibility_reason = result.error_message
            summary.status = "failed_geometry"
        elif result.status == "failed_existing_run_dir":
            summary.geometry_status = "existing_run_dir"
            summary.feasibility_reason = result.error_message
            summary.status = "failed_existing_run_dir"
        else:
            summary.geometry_status = "pipeline_failed"
            summary.feasibility_reason = result.error_message
            summary.status = f"{result.fidelity}_failed"

    def train_models(self) -> GPBundle:
        bundle = GPBundle()
        try:
            stack = import_botorch_stack()
        except Exception as exc:
            bundle.error_message = f"BoTorch/GPyTorch import failed: {type(exc).__name__}: {exc}"
            return bundle

        bundle.acquisition_name = stack["acquisition_name"]
        try:
            full_x, full_y = model_training_rows(self.candidates, self.results, self.space, "srm100")
            cheap_x, cheap_y = model_training_rows(self.candidates, self.results, self.space, "srm50")
            cal_x, cal_y, cheap_min, cheap_max = paired_training_rows(self.candidates, self.results, self.space)
            bundle.cheap_y_min = cheap_min
            bundle.cheap_y_max = cheap_max
            if len(full_y) >= 2:
                bundle.full_model = build_gp_model(full_x, full_y, stack)
            if len(cheap_y) >= 2:
                bundle.cheap_model = build_gp_model(cheap_x, cheap_y, stack)
            if len(cal_y) >= 2:
                bundle.calibration_model = build_gp_model(cal_x, cal_y, stack)
        except Exception as exc:
            bundle.error_message = f"GP training failed: {type(exc).__name__}: {exc}"
        return bundle

    def propose_candidate(self, iteration: int, bundle: GPBundle) -> tuple[Candidate, float, str]:
        candidates: list[Candidate] = []
        normalized_rows: list[np.ndarray] = []
        seen = self.blocked_keys()
        rejection_start = len(self.infeasible_candidates)
        pool_sizes = [int(self.args.candidate_pool_size), int(self.args.candidate_pool_size) * 5]
        for attempt, pool_size in enumerate(pool_sizes):
            if candidates:
                break
            pool_z = generate_pool_normalized(
                pool_size,
                4,
                self.args.random_seed + iteration + attempt * 1000003,
            )
            for pool_idx, z in enumerate(pool_z):
                candidate = materialize_candidate(
                    candidate_id=f"pool_{iteration:04d}_{attempt}_{pool_idx:06d}",
                    source="bo_qlogei",
                    iteration=iteration,
                    z=z,
                    space=self.space,
                )
                key = candidate_key(candidate)
                if key in seen:
                    continue
                check = self.check_candidate_feasibility(candidate)
                if not check.feasible:
                    self.log_infeasible_candidate(candidate, check)
                    seen.add(key)
                    continue
                seen.add(key)
                candidates.append(candidate)
                normalized_rows.append(normalize_candidate(candidate, self.space))

        if not candidates:
            self.save_all()
            raise ValueError(
                "No feasible BO candidates found. "
                f"Rejection reasons: {self.rejection_counts_since(rejection_start)}. "
                "Widen the search space or relax --opening-ratio-min/--opening-ratio-max."
            )

        if bundle.full_model is None:
            reason = bundle.error_message or "Full GP unavailable; using first unevaluated Sobol/random candidate"
            candidate = candidates[0]
            candidate.candidate_id = self.next_candidate_id()
            candidate.source = "bo_fallback"
            return candidate, math.nan, reason

        try:
            stack = import_botorch_stack()
            torch = stack["torch"]
            full_values = [
                float(result.JI)
                for result in self.results
                if result.fidelity == "srm100" and result.status == "success" and is_finite(result.JI)
            ]
            best_f = max(full_values)
            acq = stack["acquisition_cls"](model=bundle.full_model, best_f=best_f)
            x_tensor = torch.as_tensor(np.vstack(normalized_rows), dtype=torch.double).unsqueeze(1)
            with torch.no_grad():
                scores = acq(x_tensor).detach().cpu().numpy().reshape(-1)
            best_idx = int(np.nanargmax(scores))
            candidate = candidates[best_idx]
            candidate.candidate_id = self.next_candidate_id()
            candidate.source = bundle.acquisition_name
            return candidate, float(scores[best_idx]), bundle.acquisition_name
        except Exception as exc:
            candidate = candidates[0]
            candidate.candidate_id = self.next_candidate_id()
            candidate.source = "bo_fallback"
            return candidate, math.nan, f"Acquisition scoring failed: {type(exc).__name__}: {exc}"

    def annotate_predictions(
        self,
        candidate: Candidate,
        cheap_result: EvalResult | None,
        bundle: GPBundle,
        summary: CandidateSummary,
    ) -> None:
        try:
            stack = import_botorch_stack()
        except Exception:
            return

        x_norm = normalize_candidate(candidate, self.space)
        full_mean, full_std = gp_predict(bundle.full_model, x_norm, stack)
        summary.predicted_full_mean = full_mean
        summary.predicted_full_std = full_std
        if is_finite(full_mean) and is_finite(full_std):
            summary.full_ucb = full_mean + self.args.beta_full * full_std

        if cheap_result is not None and bundle.calibration_model is not None:
            cheap_norm = normalize_scalar(float(cheap_result.JI), bundle.cheap_y_min, bundle.cheap_y_max)
            cal_x = np.concatenate([x_norm, [cheap_norm]])
            cal_mean, cal_std = gp_predict(bundle.calibration_model, cal_x, stack)
            summary.calibrated_full_mean = cal_mean
            summary.calibrated_full_std = cal_std
            if is_finite(cal_mean) and is_finite(cal_std):
                summary.calibrated_ucb = cal_mean + self.args.beta_cal * cal_std

    def decide_promotion(
        self,
        iteration: int,
        cheap_result: EvalResult,
        summary: CandidateSummary,
    ) -> tuple[bool, str]:
        best_full = summary.best_full_JI_before
        if not is_finite(best_full):
            return True, "no_full_baseline"

        if is_finite(cheap_result.JI) and cheap_result.JI >= best_full * self.args.cheap_promote_ratio:
            return True, "cheap_threshold"

        if is_finite(summary.calibrated_ucb) and summary.calibrated_ucb >= best_full * self.args.calibrated_ucb_ratio:
            return True, "calibrated_ucb"

        if is_finite(summary.full_ucb) and summary.full_ucb >= best_full * self.args.full_ucb_ratio:
            return True, "full_ucb"

        if self.args.force_full_every > 0 and iteration % self.args.force_full_every == 0:
            return True, "periodic_bo_exploration"

        return False, "not_promoted"

    def successful_paired_count(self) -> int:
        count = 0
        for candidate_id in self.candidates:
            if self.successful_result(candidate_id, "srm50") and self.successful_result(candidate_id, "srm100"):
                count += 1
        return count

    def dry_run_paired_count(self) -> int:
        count = 0
        for candidate_id in self.candidates:
            has_srm50 = any(
                result.candidate_id == candidate_id
                and result.fidelity == "srm50"
                and result.status == "dry_run"
                for result in self.results
            )
            has_srm100 = any(
                result.candidate_id == candidate_id
                and result.fidelity == "srm100"
                and result.status == "dry_run"
                for result in self.results
            )
            if has_srm50 and has_srm100:
                count += 1
        return count

    def run_warm_start(self) -> None:
        needed = self.args.n_initial - self.successful_paired_count()
        if needed <= 0:
            print(f"[warm-start] Already have {self.args.n_initial} successful paired candidates.")
            return

        if self.args.dry_run:
            planned_lhs = self.dry_run_paired_count()
            dry_needed = max(0, self.args.n_initial - planned_lhs)
            for candidate in self.lhs_candidates(dry_needed, source="lhs", iteration=0):
                self.register_candidate(candidate)
                summary = self.summaries[candidate.candidate_id]
                summary.promotion_reason = "lhs_warm_start"
                summary.promoted_to_full = True
                summary.best_full_JI_before = self.best_full_ji()
                self.save_all()
                self.run_eval(candidate, "srm50", stage="lhs_warm_start")
                self.run_eval(candidate, "srm100", stage="lhs_warm_start")
            return

        attempts = 0
        while self.successful_paired_count() < self.args.n_initial and attempts < self.args.lhs_max_attempts:
            attempts += 1
            candidate = self.lhs_candidates(1, source="lhs", iteration=0)[0]
            self.register_candidate(candidate)
            summary = self.summaries[candidate.candidate_id]
            summary.promotion_reason = "lhs_warm_start"
            summary.promoted_to_full = True
            summary.best_full_JI_before = self.best_full_ji()
            self.save_all()

            cheap = self.run_eval(candidate, "srm50", stage="lhs_warm_start")
            if cheap.status == "failed_existing_run_dir":
                self.save_all()
                raise ValueError(cheap.error_message)
            if cheap.status not in {"success", "dry_run"}:
                continue
            full = self.run_eval(candidate, "srm100", stage="lhs_warm_start")
            if full.status == "failed_existing_run_dir":
                self.save_all()
                raise ValueError(full.error_message)
            if self.args.dry_run:
                continue
            if full.status != "success":
                continue

        if not self.args.dry_run and self.successful_paired_count() < self.args.n_initial:
            raise RuntimeError(
                f"Warm start only reached {self.successful_paired_count()} successful paired candidates "
                f"after {attempts} attempts; requested {self.args.n_initial}"
            )

    def run_bo_iterations(self) -> None:
        start_iteration = self.current_bo_iteration + 1
        for iteration in range(start_iteration, self.args.bo_iters + 1):
            self.current_bo_iteration = iteration
            bundle = self.train_models() if not self.args.dry_run else GPBundle(error_message="dry-run")
            candidate, acquisition_value, proposal_reason = self.propose_candidate(iteration, bundle)
            self.register_candidate(candidate)
            summary = self.summaries[candidate.candidate_id]
            summary.acquisition_value = acquisition_value
            summary.best_full_JI_before = self.best_full_ji()
            self.save_all()

            print(f"[bo {iteration:03d}] proposed {candidate.candidate_id} via {proposal_reason}: {candidate}")
            cheap = self.run_eval(candidate, "srm50", stage=f"bo_iteration_{iteration}")
            if cheap.status == "failed_existing_run_dir":
                self.save_all()
                raise ValueError(cheap.error_message)
            if cheap.status not in {"success", "dry_run"}:
                if cheap.status != "failed_geometry":
                    summary.status = "cheap_failed"
                self.save_all()
                continue

            if self.args.dry_run:
                summary.promotion_reason = "not_promoted"
                summary.status = "dry_run_planned"
                self.save_all()
                print(f"[dry-run] {candidate.candidate_id} SRM100 would be conditional on promotion")
                continue

            self.annotate_predictions(candidate, cheap, bundle, summary)
            promote, reason = self.decide_promotion(iteration, cheap, summary)
            summary.promoted_to_full = promote
            summary.promotion_reason = reason
            self.save_all()
            print(f"[bo {iteration:03d}] promotion={promote} reason={reason}")
            if promote:
                full = self.run_eval(candidate, "srm100", stage=f"bo_iteration_{iteration}")
                if full.status == "failed_existing_run_dir":
                    self.save_all()
                    raise ValueError(full.error_message)
            else:
                summary.status = "not_promoted"
                self.save_all()

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.validate_search_space()
        self.run_warm_start()
        self.run_bo_iterations()
        self.save_all()
        if self.args.dry_run:
            self.print_dry_run_summary()
        print(f"[done] Results: {self.results_csv_path}")
        print(f"[done] Summary: {self.summary_csv_path}")
        print(f"[done] Infeasible: {self.infeasible_csv_path}")
        print(f"[done] State: {self.state_json_path}")

    def print_dry_run_summary(self) -> None:
        feasible_ids = {result.candidate_id for result in self.results if result.status == "dry_run"}
        print(
            "[dry-run summary] "
            f"feasible candidates generated={len(feasible_ids)}, "
            f"infeasible candidates rejected={len(self.infeasible_candidates)}, "
            f"rejection reasons: {format_reason_counts(self.infeasible_candidates)}"
        )

    def validate_search_space(self) -> None:
        if not self.space.diam_min <= self.space.diam_max:
            raise ValueError("--diam-min must be less than or equal to --diam-max")
        if self.args.aperture_ring_radius_mm <= 0.0:
            raise ValueError("--aperture-ring-radius-mm must be positive")
        if self.args.opening_ratio_min < 0.0:
            raise ValueError("--opening-ratio-min must be non-negative")
        if self.args.opening_ratio_max <= self.args.opening_ratio_min:
            raise ValueError("--opening-ratio-max must be greater than --opening-ratio-min")
        for name, values in [
            ("n_apertures_values", self.space.n_apertures_values),
            ("ring1_crystals_values", self.space.ring1_values),
            ("ring2_crystals_values", self.space.ring2_values),
        ]:
            if sorted(set(values)) != values:
                raise ValueError(f"{name} must be sorted and unique")


def run_self_test_bo() -> None:
    stack = import_botorch_stack()
    torch = stack["torch"]
    rng = np.random.default_rng(123)
    train_x = rng.random((16, 4), dtype=np.float64)
    train_y = (
        np.sin(train_x[:, 0] * math.pi)
        + 0.5 * train_x[:, 1]
        - 0.2 * train_x[:, 2]
        + 0.1 * train_x[:, 3]
    )
    model = build_gp_model(train_x, train_y, stack)
    base_kernel = model.covar_module.base_kernel
    ard_enabled = getattr(base_kernel, "ard_num_dims", None) is not None
    if ard_enabled:
        raise AssertionError("ARD unexpectedly enabled on MaternKernel")

    best_f = float(np.max(train_y))
    acquisition = stack["acquisition_cls"](model=model, best_f=best_f)
    pool = generate_pool_normalized(256, 4, seed=321)
    x_tensor = torch.as_tensor(pool, dtype=torch.double).unsqueeze(1)
    with torch.no_grad():
        scores = acquisition(x_tensor).detach().cpu().numpy().reshape(-1)
    suggested_z = pool[int(np.nanargmax(scores))]
    space = SearchSpace(
        diam_min=0.20,
        diam_max=1.00,
        n_apertures_values=DEFAULT_N_APERTURES_VALUES,
        ring1_values=DEFAULT_RING1_VALUES,
        ring2_values=DEFAULT_RING2_VALUES,
    )
    suggested = materialize_candidate("self_test", "self_test", 0, suggested_z, space)

    print("kernel: MaternKernel(nu=2.5) inside ScaleKernel")
    print(f"ARD: {ard_enabled}")
    print(f"acquisition: {stack['acquisition_name']}")
    print(f"suggested candidate: {suggested}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4D multi-fidelity BO optimizer for SRM50/SRM100 SC-SPECT designs")
    parser.add_argument("--n-initial", type=int, default=25)
    parser.add_argument("--bo-iters", type=int, default=20)
    parser.add_argument("--out-dir", default="runs/bo_mf_4d")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test-bo", action="store_true")

    parser.add_argument("--diam-min", type=float, default=0.20)
    parser.add_argument("--diam-max", type=float, default=1.00)
    parser.add_argument("--n-apertures-values", default="60,90,120,150,180,210,240,270,300,330,360")
    parser.add_argument("--ring1-crystals-values", default="120,160,200,240,280,320,360,400,440,480")
    parser.add_argument("--ring2-crystals-values", default="240,280,320,360,400,440,480,520,560,600,640,680,720")

    parser.add_argument("--cheap-srm-row-fraction", type=float, default=0.50)
    parser.add_argument("--cheap-srm-row-mode", default="every_k")
    parser.add_argument("--cheap-srm-row-seed", type=int, default=42)
    parser.add_argument("--full-srm-row-fraction", type=float, default=1.0)

    parser.add_argument("--cheap-promote-ratio", type=float, default=0.95)
    parser.add_argument("--calibrated-ucb-ratio", type=float, default=0.98)
    parser.add_argument("--full-ucb-ratio", type=float, default=0.98)
    parser.add_argument("--beta-full", type=float, default=2.0)
    parser.add_argument("--beta-cal", type=float, default=1.5)
    parser.add_argument("--force-full-every", type=int, default=5)

    parser.add_argument("--candidate-pool-size", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--lhs-max-attempts", type=int, default=300)
    parser.add_argument("--aperture-ring-radius-mm", type=float, default=35.0)
    parser.add_argument("--opening-ratio-min", type=float, default=0.10)
    parser.add_argument("--opening-ratio-max", type=float, default=0.50)

    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--pose-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--skip-recon", dest="skip_recon", action="store_true", default=True)
    parser.add_argument("--no-skip-recon", dest="skip_recon", action="store_false")
    parser.add_argument("--layout-idxs", default="0,1")
    parser.add_argument("--t8-poses", default="0,1,2,3,4,5,6,7")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test_bo:
        run_self_test_bo()
        return
    optimizer = Optimizer(args)
    try:
        optimizer.run()
    except ValueError as exc:
        print(f"[error] {exc}")
        try:
            optimizer.save_all()
            if args.dry_run:
                optimizer.print_dry_run_summary()
        except Exception as save_exc:
            print(f"[error] Could not save optimizer state after failure: {type(save_exc).__name__}: {save_exc}")
        if args.dry_run:
            return
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
