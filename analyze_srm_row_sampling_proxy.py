#!/usr/bin/env python3
"""
Read-only detector-row SRM/PPDF sampling proxy analysis.

This script uses already-computed full-fidelity pipeline outputs to estimate
whether sampling detector-response rows can provide a useful cheap-fidelity
proxy for sensitivity, FWHM, and diagnostic JI. It does not modify pipeline
inputs or outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    import h5py  # type: ignore
except Exception as exc:  # pragma: no cover - exercised in missing envs
    h5py = None  # type: ignore
    H5PY_IMPORT_ERROR: Exception | None = exc
else:
    H5PY_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - exercised in missing envs
    np = None  # type: ignore
    NUMPY_IMPORT_ERROR: Exception | None = exc
else:
    NUMPY_IMPORT_ERROR = None


DETS_PER_RING = [40 * 6 * 2, 40 * 9 * 2, 40 * 12 * 2, 40 * 15 * 2]
CELLS_PER_SECTOR = [6, 9, 12, 15]
N_ANGLE_BINS_40 = 40
DEFAULT_DATASET_NAME = "ppdfs"
DEFAULT_GLOB_PATTERN = "position_*_ppdfs_t8_*.hdf5"
DEFAULT_FRACTIONS = "0.25,0.50"
DEFAULT_SEEDS = "42,43,44"
DEFAULT_MODES = "every_k,ring_cell_stratified,ring_cell_random"
EPS = 1e-12


CSV_COLUMNS = [
    "run_dir",
    "data_dir",
    "results_dir",
    "status",
    "error_message",
    "n_ppdf_files",
    "first_ppdf_file",
    "ppdf_shape",
    "ppdf_dtype",
    "total_rows",
    "n_pixels",
    "pixel_shape",
    "mode",
    "fraction",
    "seed",
    "n_sampled_rows",
    "sampled_fraction_actual",
    "sampling_description",
    "ring0_sampled",
    "ring1_sampled",
    "ring2_sampled",
    "ring3_sampled",
    "angle_bins_covered",
    "min_rows_per_angle_bin",
    "max_rows_per_angle_bin",
    "crystal_side0_sampled",
    "crystal_side1_sampled",
    "full_sensitivity_from_ppdf",
    "sensitivity_total_full_csv",
    "sensitivity_mean_full_csv",
    "sensitivity_total_row_proxy",
    "sensitivity_mean_row_proxy",
    "sensitivity_abs_error",
    "sensitivity_rel_error_pct",
    "full_fwhm_from_props",
    "fwhm_full_csv",
    "fwhm_row_proxy",
    "fwhm_abs_error",
    "fwhm_rel_error_pct",
    "n_fwhm_full_valid",
    "n_fwhm_sampled_valid",
    "fwhm_status",
    "fwhm_mapping_warning",
    "asci_full_csv",
    "asci_row_proxy",
    "asci_status",
    "asci_row_proxy_mode",
    "JI_full_csv",
    "JI_row_proxy",
    "JI_row_proxy_with_full_asci_diagnostic",
    "JI_abs_error",
    "JI_rel_error_pct",
    "JI_status",
    "full_metrics_status",
]


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    data_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class PPDFMetadata:
    n_files: int
    first_ppdf_file: Path
    ppdf_shape: tuple[int, int]
    ppdf_dtype: str
    total_rows: int
    n_pixels: int
    pixel_shape: tuple[int, ...]


@dataclass(frozen=True)
class BeamPropertyData:
    status: str
    warning: str
    full_fwhm_from_props: float
    n_full_valid: int
    detector_ids: Any
    fwhm_values: Any


def require_runtime_deps() -> None:
    missing: list[str] = []
    if NUMPY_IMPORT_ERROR is not None:
        missing.append(f"numpy ({NUMPY_IMPORT_ERROR})")
    if H5PY_IMPORT_ERROR is not None:
        missing.append(f"h5py ({H5PY_IMPORT_ERROR})")
    if missing:
        raise RuntimeError(
            "Missing required runtime dependencies: "
            + ", ".join(missing)
            + ". Load/install numpy and h5py to run this analysis."
        )


def parse_csv_list(text: str, cast: Callable[[str], Any], name: str) -> list[Any]:
    values: list[Any] = []
    for item in text.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(cast(stripped))
        except Exception as exc:
            raise ValueError(f"Invalid {name} value {stripped!r}: {exc}") from exc
    if not values:
        raise ValueError(f"No {name} values were provided")
    return values


def resolve_run_and_data_dir(path_text: str) -> RunPaths:
    path = Path(path_text).expanduser()
    if (path / "data").is_dir():
        run_dir = path
        data_dir = path / "data"
    else:
        data_dir = path
        run_dir = path.parent if path.name == "data" else path
    return RunPaths(run_dir=run_dir, data_dir=data_dir, results_dir=run_dir / "results")


def find_ppdf_files(data_dir: Path, glob_pattern: str, max_files: int = 0) -> list[Path]:
    files = sorted(data_dir.glob(glob_pattern))
    if max_files > 0:
        files = files[:max_files]
    return files


def inspect_ppdf_metadata(files: list[Path], dataset_name: str) -> PPDFMetadata:
    if not files:
        raise FileNotFoundError("No PPDF files found")
    first = files[0]
    with h5py.File(first, "r") as handle:
        if dataset_name not in handle:
            raise KeyError(f"Dataset {dataset_name!r} not found in {first}")
        dataset = handle[dataset_name]
        if len(dataset.shape) != 2:
            raise ValueError(
                f"Unsupported PPDF shape in {first}: {tuple(dataset.shape)}. "
                "Expected 2-D (detector_rows, source_pixels)."
            )
        total_rows = int(dataset.shape[0])
        n_pixels = int(dataset.shape[1])
        dtype = str(dataset.dtype)

    side = int(round(math.sqrt(n_pixels)))
    pixel_shape: tuple[int, ...] = (side, side) if side * side == n_pixels else (n_pixels,)
    return PPDFMetadata(
        n_files=len(files),
        first_ppdf_file=first,
        ppdf_shape=(total_rows, n_pixels),
        ppdf_dtype=dtype,
        total_rows=total_rows,
        n_pixels=n_pixels,
        pixel_shape=pixel_shape,
    )


def build_detector_row_metadata(total_rows: int = 3360) -> dict[str, Any]:
    expected_total = int(sum(DETS_PER_RING))
    if total_rows != expected_total:
        raise ValueError(
            f"Unsupported detector row count {total_rows}; expected {expected_total}. "
            "This script only supports the current 3360-row T8 detector layout."
        )

    row_idx: list[int] = []
    ring_id: list[int] = []
    cells_per_ring_values: list[int] = []
    cell_idx: list[int] = []
    crystal_side: list[int] = []
    angle_bin_40: list[int] = []
    row_within_ring: list[int] = []

    current_row = 0
    for ring, cells_per_sector in enumerate(CELLS_PER_SECTOR):
        cells_per_ring = N_ANGLE_BINS_40 * cells_per_sector
        ring_rows = cells_per_ring * 2
        for local_row in range(ring_rows):
            cell = local_row // 2
            side = local_row % 2
            row_idx.append(current_row)
            ring_id.append(ring)
            cells_per_ring_values.append(cells_per_ring)
            cell_idx.append(cell)
            crystal_side.append(side)
            angle_bin_40.append(cell // cells_per_sector)
            row_within_ring.append(local_row)
            current_row += 1

    if current_row != total_rows:
        raise AssertionError(f"Detector metadata produced {current_row} rows, expected {total_rows}")

    metadata = {
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "ring_id": np.asarray(ring_id, dtype=np.int64),
        "cells_per_ring": np.asarray(cells_per_ring_values, dtype=np.int64),
        "cell_idx": np.asarray(cell_idx, dtype=np.int64),
        "crystal_side": np.asarray(crystal_side, dtype=np.int64),
        "angle_bin_40": np.asarray(angle_bin_40, dtype=np.int64),
        "row_within_ring": np.asarray(row_within_ring, dtype=np.int64),
    }

    for ring, expected_count in enumerate(DETS_PER_RING):
        actual_count = int(np.count_nonzero(metadata["ring_id"] == ring))
        if actual_count != expected_count:
            raise AssertionError(f"Ring {ring} has {actual_count} rows, expected {expected_count}")

    for ring in range(len(DETS_PER_RING)):
        for angle_bin in range(N_ANGLE_BINS_40):
            mask = (metadata["ring_id"] == ring) & (metadata["angle_bin_40"] == angle_bin)
            expected = CELLS_PER_SECTOR[ring] * 2
            actual = int(np.count_nonzero(mask))
            if actual != expected:
                raise AssertionError(
                    f"Ring {ring} angle bin {angle_bin} has {actual} rows, expected {expected}"
                )

    return metadata


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def allocate_group_counts(group_sizes: list[int], fraction: float) -> list[int]:
    raw = [fraction * size for size in group_sizes]
    counts = [max(1, int(math.floor(value))) for value in raw]
    target = round_half_up(fraction * sum(group_sizes))
    target = min(sum(group_sizes), max(len(group_sizes), target))

    diff = target - sum(counts)
    if diff > 0:
        order = sorted(
            range(len(group_sizes)),
            key=lambda idx: (raw[idx] - math.floor(raw[idx]), -idx),
            reverse=True,
        )
        cursor = 0
        while diff > 0:
            idx = order[cursor % len(order)]
            if counts[idx] < group_sizes[idx]:
                counts[idx] += 1
                diff -= 1
            cursor += 1
    elif diff < 0:
        order = sorted(
            range(len(group_sizes)),
            key=lambda idx: (raw[idx] - math.floor(raw[idx]), idx),
        )
        cursor = 0
        while diff < 0:
            idx = order[cursor % len(order)]
            if counts[idx] > 1:
                counts[idx] -= 1
                diff += 1
            cursor += 1

    return counts


def choose_evenly(rows: Any, n_keep: int) -> Any:
    if n_keep <= 0:
        return np.asarray([], dtype=np.int64)
    if n_keep >= len(rows):
        return np.asarray(rows, dtype=np.int64)
    positions = np.linspace(0, len(rows) - 1, n_keep, dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)[positions]


def choose_balanced_by_side(
    group_rows: Any,
    group_sides: Any,
    n_keep: int,
    *,
    rng: Any | None,
    randomize: bool,
    parity: int,
) -> Any:
    group_rows = np.asarray(group_rows, dtype=np.int64)
    group_sides = np.asarray(group_sides, dtype=np.int64)
    if n_keep >= len(group_rows):
        return group_rows

    side0 = group_rows[group_sides == 0]
    side1 = group_rows[group_sides == 1]
    n0 = n_keep // 2
    n1 = n_keep // 2
    if n_keep % 2:
        if parity % 2 == 0:
            n0 += 1
        else:
            n1 += 1

    n0 = min(n0, len(side0))
    n1 = min(n1, len(side1))

    def pick(rows: Any, count: int) -> Any:
        rows = np.asarray(rows, dtype=np.int64)
        if count <= 0:
            return np.asarray([], dtype=np.int64)
        if count >= len(rows):
            return rows
        if randomize:
            return rng.choice(rows, size=count, replace=False)
        return choose_evenly(rows, count)

    selected = np.concatenate([pick(side0, n0), pick(side1, n1)])
    if len(selected) < n_keep:
        remaining = np.setdiff1d(group_rows, selected, assume_unique=False)
        selected = np.concatenate([selected, pick(remaining, n_keep - len(selected))])
    return selected


def make_sampled_rows(
    total_rows: int,
    fraction: float,
    mode: str,
    seed: int,
    metadata: dict[str, Any],
) -> tuple[Any, float, str]:
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1.0:
        rows = np.arange(total_rows, dtype=np.int64)
        return rows, 1.0, "all_rows"

    rng = np.random.default_rng(seed)
    row_idx = metadata["row_idx"]

    if mode == "every_k":
        k = max(1, round_half_up(1.0 / fraction))
        rows = np.arange(0, total_rows, k, dtype=np.int64)
        description = f"every_k:k={k}; baseline may alias with ring/cell/side ordering"
    elif mode == "random":
        n_keep = max(1, min(total_rows, round_half_up(fraction * total_rows)))
        rows = rng.choice(row_idx, size=n_keep, replace=False)
        description = "uniform_random_all_rows; does not enforce ring/angle balance"
    elif mode in {"ring_cell_stratified", "ring_cell_random"}:
        randomize = mode == "ring_cell_random"
        selected_parts: list[Any] = []
        for ring in range(len(DETS_PER_RING)):
            ring_counts: list[int] = []
            group_rows_by_angle: list[Any] = []
            group_sides_by_angle: list[Any] = []
            for angle_bin in range(N_ANGLE_BINS_40):
                group_mask = (metadata["ring_id"] == ring) & (metadata["angle_bin_40"] == angle_bin)
                group_rows = metadata["row_idx"][group_mask]
                group_sides = metadata["crystal_side"][group_mask]
                group_rows_by_angle.append(group_rows)
                group_sides_by_angle.append(group_sides)
                ring_counts.append(int(len(group_rows)))

            keep_counts = allocate_group_counts(ring_counts, fraction)
            for angle_bin, n_keep in enumerate(keep_counts):
                selected = choose_balanced_by_side(
                    group_rows_by_angle[angle_bin],
                    group_sides_by_angle[angle_bin],
                    n_keep,
                    rng=rng,
                    randomize=randomize,
                    parity=ring + angle_bin,
                )
                selected_parts.append(selected)

        rows = np.concatenate(selected_parts) if selected_parts else np.asarray([], dtype=np.int64)
        description = (
            "ring_angle_stratified_random_side_balanced"
            if randomize
            else "ring_angle_stratified_even_side_balanced"
        )
    else:
        raise ValueError(f"Unsupported sampling mode {mode!r}")

    rows = np.unique(np.asarray(rows, dtype=np.int64))
    rows.sort()
    if rows.size == 0:
        raise ValueError(f"Sampling mode {mode!r} with fraction {fraction} selected zero rows")
    if rows[0] < 0 or rows[-1] >= total_rows:
        raise ValueError(f"Sampling mode {mode!r} produced out-of-range rows")
    return rows, float(rows.size / total_rows), description


def summarize_sample_balance(sampled_rows: Any, metadata: dict[str, Any]) -> dict[str, int]:
    sampled_mask = np.zeros(metadata["row_idx"].shape[0], dtype=bool)
    sampled_mask[np.asarray(sampled_rows, dtype=np.int64)] = True

    angle_counts = np.bincount(
        metadata["angle_bin_40"][sampled_mask],
        minlength=N_ANGLE_BINS_40,
    )
    side_counts = np.bincount(
        metadata["crystal_side"][sampled_mask],
        minlength=2,
    )
    ring_counts = np.bincount(
        metadata["ring_id"][sampled_mask],
        minlength=len(DETS_PER_RING),
    )
    nonzero_angle_counts = angle_counts[angle_counts > 0]
    return {
        "ring0_sampled": int(ring_counts[0]),
        "ring1_sampled": int(ring_counts[1]),
        "ring2_sampled": int(ring_counts[2]),
        "ring3_sampled": int(ring_counts[3]),
        "angle_bins_covered": int(np.count_nonzero(angle_counts)),
        "min_rows_per_angle_bin": int(nonzero_angle_counts.min()) if nonzero_angle_counts.size else 0,
        "max_rows_per_angle_bin": int(nonzero_angle_counts.max()) if nonzero_angle_counts.size else 0,
        "crystal_side0_sampled": int(side_counts[0]),
        "crystal_side1_sampled": int(side_counts[1]),
    }


def compute_row_sensitivity_contributions(
    files: list[Path],
    dataset_name: str,
    metadata: PPDFMetadata,
    *,
    chunk_rows: int = 128,
) -> Any:
    row_contrib = np.zeros(metadata.total_rows, dtype=np.float64)
    for path in files:
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                raise KeyError(f"Dataset {dataset_name!r} not found in {path}")
            dataset = handle[dataset_name]
            if tuple(dataset.shape) != metadata.ppdf_shape:
                raise ValueError(
                    f"Incompatible PPDF shape in {path}: {tuple(dataset.shape)}; "
                    f"expected {metadata.ppdf_shape}"
                )
            for start in range(0, metadata.total_rows, chunk_rows):
                stop = min(start + chunk_rows, metadata.total_rows)
                block = np.asarray(dataset[start:stop, :], dtype=np.float64)
                row_contrib[start:stop] += block.sum(axis=1) / metadata.n_pixels
    return row_contrib


def decode_header(raw_header: Iterable[Any]) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        for item in raw_header
    ]


def normalize_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def find_header_column(header: list[str], candidates: Iterable[str]) -> int | None:
    normalized = [normalize_name(item) for item in header]
    for candidate in candidates:
        candidate_norm = normalize_name(candidate)
        if candidate_norm in normalized:
            return normalized.index(candidate_norm)
    return None


def load_beam_properties(data_dir: Path, total_rows: int) -> BeamPropertyData:
    prop_files = sorted(data_dir.glob("beams_properties_configuration_*.hdf5"))
    if not prop_files:
        return BeamPropertyData(
            status="missing_beam_properties",
            warning="",
            full_fwhm_from_props=float("nan"),
            n_full_valid=0,
            detector_ids=None,
            fwhm_values=None,
        )

    all_detector_ids: list[Any] = []
    all_fwhm_values: list[Any] = []
    warnings: list[str] = []

    for prop_file in prop_files:
        with h5py.File(prop_file, "r") as handle:
            if "beam_properties" not in handle:
                warnings.append(f"{prop_file.name}: missing beam_properties dataset")
                continue
            dataset = handle["beam_properties"]
            data = np.asarray(dataset[:], dtype=np.float64)
            if data.size == 0:
                continue
            raw_header = dataset.attrs.get("Header")
            header = decode_header(raw_header) if raw_header is not None else []

        fwhm_col = find_header_column(header, ["FWHM (mm)", "FWHM", "fwhm (mm)"])
        detector_col = find_header_column(
            header,
            ["detector unit id", "detector id", "detector unit index", "detector index"],
        )
        if fwhm_col is None:
            warnings.append(f"{prop_file.name}: missing FWHM column; available={header}")
            continue
        if detector_col is None:
            if data.shape[0] == total_rows:
                detector_ids = np.arange(total_rows, dtype=np.int64)
                warnings.append(f"{prop_file.name}: assumed_row_order")
            else:
                warnings.append(
                    f"{prop_file.name}: missing detector id column and row count "
                    f"{data.shape[0]} does not match {total_rows}"
                )
                continue
        else:
            detector_values = np.asarray(data[:, detector_col], dtype=np.float64)
            detector_ids = np.rint(detector_values).astype(np.int64)

        fwhm_values = np.asarray(data[:, fwhm_col], dtype=np.float64)
        all_detector_ids.append(detector_ids)
        all_fwhm_values.append(fwhm_values)

    if not all_fwhm_values:
        return BeamPropertyData(
            status="failed_no_usable_fwhm",
            warning="; ".join(warnings),
            full_fwhm_from_props=float("nan"),
            n_full_valid=0,
            detector_ids=None,
            fwhm_values=None,
        )

    detector_ids = np.concatenate(all_detector_ids)
    fwhm_values = np.concatenate(all_fwhm_values)
    valid = np.isfinite(fwhm_values) & (fwhm_values > 0.0)
    return BeamPropertyData(
        status="success",
        warning="; ".join(warnings),
        full_fwhm_from_props=float(np.mean(fwhm_values[valid])) if np.any(valid) else float("nan"),
        n_full_valid=int(np.count_nonzero(valid)),
        detector_ids=detector_ids,
        fwhm_values=fwhm_values,
    )


def compute_fwhm_proxy(beam_props: BeamPropertyData, sampled_rows: Any) -> dict[str, Any]:
    if beam_props.detector_ids is None or beam_props.fwhm_values is None:
        return {
            "fwhm_row_proxy": float("nan"),
            "n_fwhm_sampled_valid": 0,
            "fwhm_status": beam_props.status,
            "fwhm_mapping_warning": beam_props.warning,
        }

    sampled_detector = np.isin(beam_props.detector_ids, np.asarray(sampled_rows, dtype=np.int64))
    valid = sampled_detector & np.isfinite(beam_props.fwhm_values) & (beam_props.fwhm_values > 0.0)
    return {
        "fwhm_row_proxy": float(np.mean(beam_props.fwhm_values[valid])) if np.any(valid) else float("nan"),
        "n_fwhm_sampled_valid": int(np.count_nonzero(valid)),
        "fwhm_status": "success" if np.any(valid) else "no_sampled_valid_fwhm",
        "fwhm_mapping_warning": beam_props.warning,
    }


def load_full_ji_metrics(run_paths: RunPaths) -> tuple[dict[str, str], str]:
    metrics_path = run_paths.results_dir / "ji_metrics.csv"
    if not metrics_path.exists():
        return {}, f"missing {metrics_path}"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, f"empty {metrics_path}"
    return rows[-1], "success"


def to_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except Exception:
        return float("nan")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return value


def rel_error_pct(value: float, reference: float) -> float:
    if not math.isfinite(value) or not math.isfinite(reference) or abs(reference) <= EPS:
        return float("nan")
    return 100.0 * (value - reference) / reference


def make_empty_row(run_paths: RunPaths) -> dict[str, Any]:
    row = {column: "" for column in CSV_COLUMNS}
    row["run_dir"] = str(run_paths.run_dir)
    row["data_dir"] = str(run_paths.data_dir)
    row["results_dir"] = str(run_paths.results_dir)
    return row


def compute_proxy_rows_for_run(
    run_dir_text: str,
    *,
    fractions: list[float],
    seeds: list[int],
    modes: list[str],
    dataset_name: str,
    glob_pattern: str,
    max_files: int,
    allow_diagnostic_full_asci: bool,
    verbose: bool,
) -> list[dict[str, Any]]:
    run_paths = resolve_run_and_data_dir(run_dir_text)
    rows: list[dict[str, Any]] = []

    try:
        ppdf_files = find_ppdf_files(run_paths.data_dir, glob_pattern, max_files)
        if not ppdf_files:
            raise FileNotFoundError(
                f"No PPDF files found in {run_paths.data_dir} matching {glob_pattern!r}"
            )
        ppdf_metadata = inspect_ppdf_metadata(ppdf_files, dataset_name)
        detector_metadata = build_detector_row_metadata(ppdf_metadata.total_rows)
        if verbose:
            print(
                f"[RUN] {run_paths.run_dir}: {len(ppdf_files)} PPDF files, "
                f"shape={ppdf_metadata.ppdf_shape}"
            )
        row_contrib = compute_row_sensitivity_contributions(
            ppdf_files,
            dataset_name,
            ppdf_metadata,
        )
        full_sensitivity_from_ppdf = float(np.sum(row_contrib))
        full_metrics, full_metrics_status = load_full_ji_metrics(run_paths)
        beam_props = load_beam_properties(run_paths.data_dir, ppdf_metadata.total_rows)
    except Exception as exc:
        for mode in modes:
            for fraction in fractions:
                for seed in seeds:
                    row = make_empty_row(run_paths)
                    row["status"] = "failed"
                    row["error_message"] = str(exc)
                    row["mode"] = mode
                    row["fraction"] = fraction
                    row["seed"] = seed
                    rows.append(row)
        return rows

    sens_total_full_csv = to_float(full_metrics.get("sensitivity_total"))
    sens_mean_full_csv = to_float(full_metrics.get("sensitivity_mean"))
    fwhm_full_csv = to_float(full_metrics.get("fwhm_mean"))
    asci_full_csv = to_float(full_metrics.get("asci_pct"))
    ji_full_csv = to_float(full_metrics.get("JI"))

    for mode in modes:
        for fraction in fractions:
            for seed in seeds:
                row = make_empty_row(run_paths)
                try:
                    sampled_rows, sampled_fraction_actual, description = make_sampled_rows(
                        ppdf_metadata.total_rows,
                        fraction,
                        mode,
                        seed,
                        detector_metadata,
                    )
                    balance = summarize_sample_balance(sampled_rows, detector_metadata)

                    scale = ppdf_metadata.total_rows / float(len(sampled_rows))
                    sensitivity_total_row = float(np.sum(row_contrib[sampled_rows]) * scale)
                    sensitivity_mean_row = sensitivity_total_row / max(ppdf_metadata.n_files, 1)
                    sensitivity_abs_error = sensitivity_total_row - full_sensitivity_from_ppdf
                    sensitivity_rel = rel_error_pct(
                        sensitivity_total_row,
                        full_sensitivity_from_ppdf,
                    )

                    fwhm_proxy = compute_fwhm_proxy(beam_props, sampled_rows)
                    fwhm_row = float(fwhm_proxy["fwhm_row_proxy"])
                    fwhm_abs_error = (
                        fwhm_row - beam_props.full_fwhm_from_props
                        if math.isfinite(fwhm_row) and math.isfinite(beam_props.full_fwhm_from_props)
                        else float("nan")
                    )
                    fwhm_rel = rel_error_pct(fwhm_row, beam_props.full_fwhm_from_props)

                    asci_status = "not_computed_requires_mask_aware_rebuild"
                    asci_proxy_mode = "not_computed"
                    ji_proxy = float("nan")
                    ji_diagnostic = float("nan")
                    ji_status = "not_computed_asci_unavailable"

                    if allow_diagnostic_full_asci and math.isfinite(asci_full_csv):
                        asci_status = "diagnostic_full_asci_available"
                        asci_proxy_mode = "full_asci_reused_for_diagnostic_only"
                        if (
                            math.isfinite(sensitivity_mean_row)
                            and math.isfinite(fwhm_row)
                            and fwhm_row > EPS
                        ):
                            ji_diagnostic = (
                                sensitivity_mean_row / (fwhm_row * fwhm_row)
                            ) * asci_full_csv / 100.0
                            ji_status = "diagnostic_full_asci"

                    if beam_props.status == "success" and math.isfinite(fwhm_row):
                        status = "success_sensitivity_fwhm"
                    else:
                        status = "success_sensitivity_only"
                    if allow_diagnostic_full_asci and math.isfinite(ji_diagnostic):
                        status = f"{status}_diagnostic_ji"

                    ji_for_error = ji_proxy if math.isfinite(ji_proxy) else ji_diagnostic
                    ji_abs_error = (
                        ji_for_error - ji_full_csv
                        if math.isfinite(ji_for_error) and math.isfinite(ji_full_csv)
                        else float("nan")
                    )
                    ji_rel = rel_error_pct(ji_for_error, ji_full_csv)

                    row.update(
                        {
                            "status": status,
                            "error_message": "",
                            "n_ppdf_files": ppdf_metadata.n_files,
                            "first_ppdf_file": str(ppdf_metadata.first_ppdf_file),
                            "ppdf_shape": str(ppdf_metadata.ppdf_shape),
                            "ppdf_dtype": ppdf_metadata.ppdf_dtype,
                            "total_rows": ppdf_metadata.total_rows,
                            "n_pixels": ppdf_metadata.n_pixels,
                            "pixel_shape": str(ppdf_metadata.pixel_shape),
                            "mode": mode,
                            "fraction": fraction,
                            "seed": seed,
                            "n_sampled_rows": len(sampled_rows),
                            "sampled_fraction_actual": sampled_fraction_actual,
                            "sampling_description": description,
                            **balance,
                            "full_sensitivity_from_ppdf": full_sensitivity_from_ppdf,
                            "sensitivity_total_full_csv": sens_total_full_csv,
                            "sensitivity_mean_full_csv": sens_mean_full_csv,
                            "sensitivity_total_row_proxy": sensitivity_total_row,
                            "sensitivity_mean_row_proxy": sensitivity_mean_row,
                            "sensitivity_abs_error": sensitivity_abs_error,
                            "sensitivity_rel_error_pct": sensitivity_rel,
                            "full_fwhm_from_props": beam_props.full_fwhm_from_props,
                            "fwhm_full_csv": fwhm_full_csv,
                            "fwhm_row_proxy": fwhm_row,
                            "fwhm_abs_error": fwhm_abs_error,
                            "fwhm_rel_error_pct": fwhm_rel,
                            "n_fwhm_full_valid": beam_props.n_full_valid,
                            "n_fwhm_sampled_valid": fwhm_proxy["n_fwhm_sampled_valid"],
                            "fwhm_status": fwhm_proxy["fwhm_status"],
                            "fwhm_mapping_warning": fwhm_proxy["fwhm_mapping_warning"],
                            "asci_full_csv": asci_full_csv,
                            "asci_row_proxy": float("nan"),
                            "asci_status": asci_status,
                            "asci_row_proxy_mode": asci_proxy_mode,
                            "JI_full_csv": ji_full_csv,
                            "JI_row_proxy": ji_proxy,
                            "JI_row_proxy_with_full_asci_diagnostic": ji_diagnostic,
                            "JI_abs_error": ji_abs_error,
                            "JI_rel_error_pct": ji_rel,
                            "JI_status": ji_status,
                            "full_metrics_status": full_metrics_status,
                        }
                    )
                except Exception as exc:
                    row.update(
                        {
                            "status": "failed",
                            "error_message": str(exc),
                            "mode": mode,
                            "fraction": fraction,
                            "seed": seed,
                            "full_metrics_status": full_metrics_status,
                        }
                    )
                rows.append(row)

    return rows


def write_rows(out_csv: Path, rows: list[dict[str, Any]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: csv_value(row.get(column, "")) for column in CSV_COLUMNS})


def float_from_row(row: dict[str, Any], key: str) -> float:
    return to_float(row.get(key))


def mean_abs(values: list[float]) -> float:
    finite = [abs(value) for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def print_summary(rows: list[dict[str, Any]], out_csv: Path, allow_diagnostic_full_asci: bool) -> None:
    successful = [row for row in rows if str(row.get("status", "")).startswith("success")]
    failed = [row for row in rows if row.get("status") == "failed"]
    print(f"\nCSV path: {out_csv}")
    print(f"Rows written: {len(rows)}")
    print(f"Successful rows: {len(successful)}")
    print(f"Failed rows: {len(failed)}")

    print("\nPer run full metrics:")
    seen_runs: set[str] = set()
    for row in rows:
        run_dir = str(row.get("run_dir", ""))
        if not run_dir or run_dir in seen_runs:
            continue
        seen_runs.add(run_dir)
        print(
            "  "
            f"{run_dir}: "
            f"JI={csv_value(float_from_row(row, 'JI_full_csv'))}, "
            f"sens_total_csv={csv_value(float_from_row(row, 'sensitivity_total_full_csv'))}, "
            f"fwhm_csv={csv_value(float_from_row(row, 'fwhm_full_csv'))}, "
            f"asci_csv={csv_value(float_from_row(row, 'asci_full_csv'))}"
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in successful:
        key = (str(row.get("mode", "")), str(row.get("fraction", "")))
        grouped.setdefault(key, []).append(row)

    print("\nGrouped proxy error summary:")
    for (mode, fraction), group_rows in sorted(grouped.items()):
        sens_errors = [float_from_row(row, "sensitivity_rel_error_pct") for row in group_rows]
        fwhm_errors = [float_from_row(row, "fwhm_rel_error_pct") for row in group_rows]
        ji_errors = [float_from_row(row, "JI_rel_error_pct") for row in group_rows]
        sampled_fracs = [float_from_row(row, "sampled_fraction_actual") for row in group_rows]
        angle_bins = [float_from_row(row, "angle_bins_covered") for row in group_rows]
        print(
            "  "
            f"{mode} fraction={fraction}: "
            f"mean_abs_sens_rel_err={csv_value(mean_abs(sens_errors))}%, "
            f"mean_abs_fwhm_rel_err={csv_value(mean_abs(fwhm_errors))}%, "
            f"mean_abs_diag_JI_rel_err={csv_value(mean_abs(ji_errors))}%, "
            f"mean_sampled_fraction={csv_value(sum(sampled_fracs) / len(sampled_fracs))}, "
            f"angle_bins_covered_min={csv_value(min(angle_bins) if angle_bins else float('nan'))}"
        )

    if allow_diagnostic_full_asci:
        print(
            "\nDiagnostic mode used full ASCI from ji_metrics.csv when partial ASCI was unavailable. "
            "Diagnostic JI is not a true row-sampled JI."
        )
    else:
        print(
            "\nASCI proxy was not computed; JI proxy is unavailable unless diagnostic full-ASCI "
            "mode is enabled."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only detector-row SRM/PPDF sampling proxy analysis"
    )
    parser.add_argument("--run-dirs", required=True, help="Comma-separated run or data directories")
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS, help="Comma-separated fractions")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated integer seeds")
    parser.add_argument("--modes", default=DEFAULT_MODES, help="Comma-separated sampling modes")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--glob-pattern", default=DEFAULT_GLOB_PATTERN)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail on first run-level error")
    parser.add_argument(
        "--allow-diagnostic-full-asci",
        action="store_true",
        help=(
            "Use full ASCI from ji_metrics.csv only for diagnostic proxy JI when partial ASCI "
            "is unavailable"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        require_runtime_deps()
        run_dirs = parse_csv_list(args.run_dirs, str, "run directory")
        fractions = parse_csv_list(args.fractions, float, "fraction")
        seeds = parse_csv_list(args.seeds, int, "seed")
        modes = parse_csv_list(args.modes, str, "mode")
        allowed_modes = {"every_k", "ring_cell_stratified", "ring_cell_random", "random"}
        unknown_modes = sorted(set(modes) - allowed_modes)
        if unknown_modes:
            raise ValueError(f"Unsupported mode(s): {unknown_modes}")
        for fraction in fractions:
            if not (0.0 < fraction <= 1.0):
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if args.max_files < 0:
            raise ValueError("--max-files must be >= 0")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    all_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rows = compute_proxy_rows_for_run(
            run_dir,
            fractions=fractions,
            seeds=seeds,
            modes=modes,
            dataset_name=args.dataset_name,
            glob_pattern=args.glob_pattern,
            max_files=args.max_files,
            allow_diagnostic_full_asci=args.allow_diagnostic_full_asci,
            verbose=args.verbose,
        )
        all_rows.extend(rows)
        if args.strict and any(row.get("status") == "failed" for row in rows):
            break

    out_csv = Path(args.out_csv)
    write_rows(out_csv, all_rows)
    print_summary(all_rows, out_csv, args.allow_diagnostic_full_asci)
    return 0 if not any(row.get("status") == "failed" for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
