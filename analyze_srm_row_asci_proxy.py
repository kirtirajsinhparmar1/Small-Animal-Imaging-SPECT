#!/usr/bin/env python3
"""
Read-only detector-row SRM/PPDF sampling proxy with ASCI reconstruction.

This script uses already-computed full-fidelity pipeline outputs to test
whether detector-response-row sampling can estimate sensitivity, FWHM, ASCI,
and proxy JI. It does not modify pipeline inputs or existing pipeline code.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
N_SECTORS = 40

DEFAULT_DATASET_NAME = "ppdfs"
DEFAULT_PPDF_GLOB = "position_*_ppdfs_t8_*.hdf5"
DEFAULT_PROPS_GLOB = "beams_properties_configuration_*.hdf5"
DEFAULT_MASKS_GLOB = "beams_masks_configuration_*.hdf5"
DEFAULT_FRACTIONS = "0.50,0.25"
DEFAULT_SEEDS = "42,43,44"
DEFAULT_MODES = "ring_cell_random,every_k,ring_cell_stratified"
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
    "asci_abs_error",
    "asci_rel_error_pct",
    "n_asci_pixels",
    "n_angle_bins",
    "n_asci_beams_used",
    "n_asci_mask_rows_seen",
    "asci_status",
    "asci_mapping_warning",
    "JI_full_csv",
    "JI_row_proxy",
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
class FullMetrics:
    status: str
    fwhm_mean: float
    sensitivity_total: float
    sensitivity_mean: float
    asci_pct: float
    ji: float


@dataclass(frozen=True)
class BeamPropertyData:
    status: str
    warning: str
    full_fwhm_from_props: float
    n_full_valid: int
    detector_ids: Any
    fwhm_values: Any
    valid_mask: Any


@dataclass(frozen=True)
class ASCIProxyResult:
    status: str
    warning: str
    asci_pct: float
    n_pixels: int
    n_angle_bins: int
    n_beams_used: int
    n_mask_rows_seen: int


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
        item = item.strip()
        if not item:
            continue
        try:
            values.append(cast(item))
        except Exception as exc:
            raise ValueError(f"Invalid {name} value {item!r}: {exc}") from exc
    if not values:
        raise ValueError(f"No {name} values were provided")
    return values


def finite_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def rel_error_pct(estimate: float, truth: float) -> float:
    if not is_finite(estimate) or not is_finite(truth) or abs(truth) <= EPS:
        return float("nan")
    return 100.0 * (estimate - truth) / truth


def fmt_shape(shape: Sequence[int]) -> str:
    return "x".join(str(int(x)) for x in shape)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def resolve_run_and_data_dir(path_text: str) -> RunPaths:
    path = Path(path_text).expanduser()
    if (path / "data").is_dir():
        run_dir = path
        data_dir = path / "data"
    elif path.name == "data":
        data_dir = path
        run_dir = path.parent
    else:
        run_dir = path
        data_dir = path / "data" if (path / "data").is_dir() else path
    return RunPaths(run_dir=run_dir, data_dir=data_dir, results_dir=run_dir / "results")


def find_files(directory: Path, pattern: str, max_files: int = 0) -> list[Path]:
    files = sorted(directory.glob(pattern))
    if max_files > 0:
        return files[:max_files]
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
    cell_idx: list[int] = []
    crystal_side: list[int] = []
    row_within_ring: list[int] = []
    cells_per_ring_values: list[int] = []
    cells_per_sector_values: list[int] = []
    angle_bin_40: list[int] = []

    row_counter = 0
    for ring, cells_per_sector in enumerate(CELLS_PER_SECTOR):
        cells_per_ring = N_SECTORS * cells_per_sector
        ring_start = row_counter
        for cell in range(cells_per_ring):
            angle_bin = cell // cells_per_sector
            for side in (0, 1):
                row_idx.append(row_counter)
                ring_id.append(ring)
                cell_idx.append(cell)
                crystal_side.append(side)
                row_within_ring.append(row_counter - ring_start)
                cells_per_ring_values.append(cells_per_ring)
                cells_per_sector_values.append(cells_per_sector)
                angle_bin_40.append(angle_bin)
                row_counter += 1

    metadata = {
        "row_idx": np.asarray(row_idx, dtype=np.int64),
        "ring_id": np.asarray(ring_id, dtype=np.int64),
        "cell_idx": np.asarray(cell_idx, dtype=np.int64),
        "crystal_side": np.asarray(crystal_side, dtype=np.int64),
        "row_within_ring": np.asarray(row_within_ring, dtype=np.int64),
        "cells_per_ring": np.asarray(cells_per_ring_values, dtype=np.int64),
        "cells_per_sector": np.asarray(cells_per_sector_values, dtype=np.int64),
        "angle_bin_40": np.asarray(angle_bin_40, dtype=np.int64),
    }

    if row_counter != expected_total:
        raise AssertionError(f"Detector metadata reconstruction produced {row_counter} rows")

    ring_counts = np.bincount(metadata["ring_id"], minlength=len(DETS_PER_RING))
    for ring, expected in enumerate(DETS_PER_RING):
        if int(ring_counts[ring]) != int(expected):
            raise AssertionError(f"Ring {ring} has {ring_counts[ring]} rows; expected {expected}")
        ring_angles = set(metadata["angle_bin_40"][metadata["ring_id"] == ring].tolist())
        if ring_angles != set(range(N_SECTORS)):
            raise AssertionError(f"Ring {ring} does not cover all {N_SECTORS} angle bins")

    return metadata


def make_sampled_rows(
    total_rows: int,
    fraction: float,
    mode: str,
    seed: int,
    metadata: dict[str, Any],
) -> tuple[Any, float, str]:
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError(f"Fraction must be in (0, 1], got {fraction}")
    if abs(fraction - 1.0) <= 1e-12:
        rows = np.arange(total_rows, dtype=np.int64)
        return rows, 1.0, "all_rows"

    if mode == "every_k":
        if math.isclose(fraction, 0.5, rel_tol=0.0, abs_tol=1e-8):
            k = 2
        elif math.isclose(fraction, 0.25, rel_tol=0.0, abs_tol=1e-8):
            k = 4
        else:
            k = max(1, int(round(1.0 / fraction)))
        rows = np.arange(0, total_rows, k, dtype=np.int64)
        description = f"every_{k}th_row_alias_prone"
    elif mode == "random":
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(round(fraction * total_rows)))
        rows = np.sort(rng.choice(total_rows, size=n_keep, replace=False).astype(np.int64))
        description = "uniform_random_rows_not_ring_balanced"
    elif mode in {"ring_cell_stratified", "ring_cell_random"}:
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for ring in range(len(DETS_PER_RING)):
            for angle_bin in range(N_SECTORS):
                group = metadata["row_idx"][
                    (metadata["ring_id"] == ring) & (metadata["angle_bin_40"] == angle_bin)
                ]
                if group.size == 0:
                    raise AssertionError(f"Empty detector group ring={ring}, angle_bin={angle_bin}")
                n_keep = max(1, int(round(fraction * int(group.size))))
                if mode == "ring_cell_random":
                    chosen = rng.choice(group, size=n_keep, replace=False)
                else:
                    positions = np.linspace(0, int(group.size) - 1, n_keep, dtype=np.int64)
                    chosen = group[positions]
                selected.extend(int(x) for x in chosen)
        rows = np.asarray(sorted(set(selected)), dtype=np.int64)
        description = f"{mode}_by_ring_and_40_angle_bins"
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    if rows.size == 0:
        raise ValueError(f"Sampling mode {mode!r} produced zero rows")
    if rows.size != np.unique(rows).size:
        raise AssertionError("Sampling produced duplicate rows")
    if int(np.min(rows)) < 0 or int(np.max(rows)) >= total_rows:
        raise AssertionError("Sampling produced row indices outside detector range")
    return rows, float(rows.size / total_rows), description


def summarize_sample_balance(sampled_rows: Any, metadata: dict[str, Any]) -> dict[str, int]:
    row_mask = np.zeros(metadata["row_idx"].shape[0], dtype=bool)
    row_mask[sampled_rows] = True
    ring_counts = np.bincount(metadata["ring_id"][row_mask], minlength=len(DETS_PER_RING))
    angle_counts = np.bincount(metadata["angle_bin_40"][row_mask], minlength=N_SECTORS)
    side_counts = np.bincount(metadata["crystal_side"][row_mask], minlength=2)
    return {
        "ring0_sampled": int(ring_counts[0]),
        "ring1_sampled": int(ring_counts[1]),
        "ring2_sampled": int(ring_counts[2]),
        "ring3_sampled": int(ring_counts[3]),
        "angle_bins_covered": int(np.count_nonzero(angle_counts)),
        "min_rows_per_angle_bin": int(np.min(angle_counts)),
        "max_rows_per_angle_bin": int(np.max(angle_counts)),
        "crystal_side0_sampled": int(side_counts[0]),
        "crystal_side1_sampled": int(side_counts[1]),
    }


def compute_detector_row_signal(
    files: list[Path],
    dataset_name: str,
    expected_shape: tuple[int, int],
    chunk_rows: int = 64,
) -> Any:
    total_rows, _ = expected_shape
    row_signal = np.zeros(total_rows, dtype=np.float64)
    for path in files:
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                raise KeyError(f"Dataset {dataset_name!r} not found in {path}")
            dataset = handle[dataset_name]
            if tuple(dataset.shape) != tuple(expected_shape):
                raise ValueError(
                    f"Incompatible PPDF shape in {path}: {tuple(dataset.shape)}; "
                    f"expected {expected_shape}"
                )
            for start in range(0, total_rows, chunk_rows):
                stop = min(start + chunk_rows, total_rows)
                row_signal[start:stop] += np.asarray(dataset[start:stop, :], dtype=np.float64).sum(axis=1)
    return row_signal


def compute_sensitivity_from_row_signal(
    row_signal: Any,
    sampled_rows: Any,
    n_pixels: int,
    n_ppdf_files: int,
) -> tuple[float, float, float]:
    full_sensitivity = float(np.sum(row_signal) / float(n_pixels))
    scale = float(row_signal.shape[0]) / float(sampled_rows.size)
    sampled_total = float(np.sum(row_signal[sampled_rows]) * scale / float(n_pixels))
    sampled_mean = sampled_total / float(max(n_ppdf_files, 1))
    return full_sensitivity, sampled_total, sampled_mean


def normalize_header_name(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def decode_header(raw_header: Any) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
        for item in list(raw_header)
    ]


def find_col(header: Sequence[str], candidates: Sequence[str]) -> int | None:
    normalized = [normalize_header_name(col) for col in header]
    for candidate in candidates:
        candidate_norm = normalize_header_name(candidate)
        if candidate_norm in normalized:
            return normalized.index(candidate_norm)
    return None


def require_col(header: Sequence[str], candidates: Sequence[str], path: Path, semantic_name: str) -> int:
    idx = find_col(header, candidates)
    if idx is None:
        raise KeyError(
            f"Missing required {semantic_name} column in {path}. "
            f"Accepted names: {candidates}. Available columns: {list(header)}"
        )
    return idx


def load_beam_properties_array(path: Path) -> tuple[Any, list[str]]:
    with h5py.File(path, "r") as handle:
        if "beam_properties" not in handle:
            raise KeyError(f"Dataset 'beam_properties' not found in {path}")
        dataset = handle["beam_properties"]
        data = dataset[:]
        if "Header" in dataset.attrs:
            header = decode_header(dataset.attrs["Header"])
        elif "Header" in handle:
            header = decode_header(handle["Header"][:])
        else:
            raise KeyError(f"No beam property Header found in {path}")
    return data, header


def valid_fwhm_mask(values: Any, fwhm_min: float | None, fwhm_max: float | None) -> Any:
    mask = np.isfinite(values) & (values > 0.0)
    if fwhm_min is not None:
        mask &= values >= float(fwhm_min)
    if fwhm_max is not None:
        mask &= values <= float(fwhm_max)
    return mask


def load_beam_properties_for_fwhm(
    data_dir: Path,
    props_pattern: str,
    total_rows: int,
    fwhm_min: float | None,
    fwhm_max: float | None,
) -> BeamPropertyData:
    files = sorted(data_dir.glob(props_pattern))
    if not files:
        return BeamPropertyData(
            status="missing_properties",
            warning=f"No beam property files matched {props_pattern}",
            full_fwhm_from_props=float("nan"),
            n_full_valid=0,
            detector_ids=np.asarray([], dtype=np.int64),
            fwhm_values=np.asarray([], dtype=np.float64),
            valid_mask=np.asarray([], dtype=bool),
        )

    detector_parts: list[Any] = []
    fwhm_parts: list[Any] = []
    warnings: list[str] = []
    status = "success"

    for path in files:
        try:
            data, header = load_beam_properties_array(path)
            fwhm_col = require_col(path=path, header=header, candidates=["FWHM (mm)", "fwhm (mm)", "FWHM"], semantic_name="FWHM")
            detector_col = find_col(
                header,
                [
                    "detector unit id",
                    "detector id",
                    "detector_id",
                    "detector idx",
                    "detector_idx",
                    "detector index",
                    "detector_index",
                ],
            )
            if detector_col is None:
                if data.shape[0] == total_rows:
                    detector_ids = np.arange(total_rows, dtype=np.int64)
                    warnings.append(f"{path.name}: assumed_row_order")
                else:
                    warnings.append(f"{path.name}: missing detector id column")
                    status = "failed_missing_detector_id"
                    continue
            else:
                detector_ids = np.asarray(np.rint(data[:, detector_col]), dtype=np.int64)
            fwhm_parts.append(np.asarray(data[:, fwhm_col], dtype=np.float64))
            detector_parts.append(detector_ids)
        except Exception as exc:
            warnings.append(f"{path.name}: {exc}")
            status = "failed_read_properties"

    if not fwhm_parts:
        return BeamPropertyData(
            status=status,
            warning="; ".join(warnings),
            full_fwhm_from_props=float("nan"),
            n_full_valid=0,
            detector_ids=np.asarray([], dtype=np.int64),
            fwhm_values=np.asarray([], dtype=np.float64),
            valid_mask=np.asarray([], dtype=bool),
        )

    detector_ids_all = np.concatenate(detector_parts)
    fwhm_values_all = np.concatenate(fwhm_parts)
    valid = valid_fwhm_mask(fwhm_values_all, fwhm_min, fwhm_max)
    full_fwhm = float(np.mean(fwhm_values_all[valid])) if np.any(valid) else float("nan")
    return BeamPropertyData(
        status=status,
        warning="; ".join(warnings),
        full_fwhm_from_props=full_fwhm,
        n_full_valid=int(np.count_nonzero(valid)),
        detector_ids=detector_ids_all,
        fwhm_values=fwhm_values_all,
        valid_mask=valid,
    )


def compute_fwhm_proxy(
    prop_data: BeamPropertyData,
    sampled_rows: Any,
) -> tuple[float, int, str, str]:
    if prop_data.detector_ids.size == 0 or prop_data.fwhm_values.size == 0:
        return float("nan"), 0, prop_data.status, prop_data.warning
    sampled = np.isin(prop_data.detector_ids, sampled_rows)
    valid = prop_data.valid_mask & sampled
    if not np.any(valid):
        return float("nan"), 0, "no_sampled_valid_fwhm", prop_data.warning
    return float(np.mean(prop_data.fwhm_values[valid])), int(np.count_nonzero(valid)), prop_data.status, prop_data.warning


def load_full_metrics(results_dir: Path) -> FullMetrics:
    path = results_dir / "ji_metrics.csv"
    if not path.exists():
        return FullMetrics("missing_ji_metrics_csv", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    try:
        with path.open("r", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return FullMetrics("empty_ji_metrics_csv", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
        row = rows[-1]
        return FullMetrics(
            status="success",
            fwhm_mean=finite_float(row.get("fwhm_mean")),
            sensitivity_total=finite_float(row.get("sensitivity_total")),
            sensitivity_mean=finite_float(row.get("sensitivity_mean")),
            asci_pct=finite_float(row.get("asci_pct")),
            ji=finite_float(row.get("JI")),
        )
    except Exception as exc:
        return FullMetrics(f"failed_read_ji_metrics_csv: {exc}", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))


def configuration_key(path: Path) -> str:
    match = re.search(r"configuration_(\d+)", path.name)
    return match.group(1) if match else path.stem


def match_property_mask_pairs(data_dir: Path, props_pattern: str, masks_pattern: str) -> tuple[list[tuple[Path, Path]], str]:
    prop_files = sorted(data_dir.glob(props_pattern))
    mask_files = sorted(data_dir.glob(masks_pattern))
    prop_by_key = {configuration_key(path): path for path in prop_files}
    mask_by_key = {configuration_key(path): path for path in mask_files}
    pairs: list[tuple[Path, Path]] = []
    warnings: list[str] = []
    for key, prop_path in prop_by_key.items():
        mask_path = mask_by_key.get(key)
        if mask_path is None:
            warnings.append(f"missing mask for configuration {key}")
            continue
        pairs.append((prop_path, mask_path))
    for key in sorted(set(mask_by_key) - set(prop_by_key)):
        warnings.append(f"missing properties for configuration {key}")
    return pairs, "; ".join(warnings)


def infer_angle_unit(values: Any, forced_unit: str) -> str:
    if forced_unit in {"rad", "deg"}:
        return forced_unit
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "rad"
    return "rad" if float(np.max(np.abs(finite))) <= (2.0 * math.pi + 1e-6) else "deg"


def angles_to_bins(values: Any, n_angle_bins: int, unit: str) -> Any:
    values = np.asarray(values, dtype=np.float64)
    if unit == "rad":
        normalized = np.mod(values, 2.0 * math.pi)
        width = (2.0 * math.pi) / float(n_angle_bins)
    else:
        normalized = np.mod(values, 360.0)
        width = 360.0 / float(n_angle_bins)
    bins = np.floor(normalized / width).astype(np.int64)
    return np.clip(bins, 0, n_angle_bins - 1)


def open_mask_dataset(path: Path) -> tuple[Any, Any]:
    handle = h5py.File(path, "r")
    try:
        if "beam_mask" not in handle:
            raise KeyError(f"Dataset 'beam_mask' not found in {path}")
        return handle, handle["beam_mask"]
    except Exception:
        handle.close()
        raise


def compute_asci_proxy(
    data_dir: Path,
    props_pattern: str,
    masks_pattern: str,
    sampled_rows: Any,
    total_rows: int,
    n_pixels: int,
    n_angle_bins: int,
    angle_unit: str,
    fwhm_min: float | None,
    fwhm_max: float | None,
) -> ASCIProxyResult:
    pairs, pair_warning = match_property_mask_pairs(data_dir, props_pattern, masks_pattern)
    if not pairs:
        return ASCIProxyResult(
            status="failed_missing_property_mask_pairs",
            warning=pair_warning or "No matched beam property/mask files found",
            asci_pct=float("nan"),
            n_pixels=n_pixels,
            n_angle_bins=n_angle_bins,
            n_beams_used=0,
            n_mask_rows_seen=0,
        )

    sampled_lookup = np.zeros(total_rows, dtype=bool)
    sampled_lookup[sampled_rows] = True
    hist = np.zeros((n_pixels, n_angle_bins), dtype=np.bool_)
    warnings: list[str] = [pair_warning] if pair_warning else []
    n_beams_used = 0
    n_mask_rows_seen = 0

    for prop_path, mask_path in pairs:
        try:
            properties, header = load_beam_properties_array(prop_path)
            detector_col = require_col(
                header,
                [
                    "detector unit id",
                    "detector id",
                    "detector_id",
                    "detector idx",
                    "detector_idx",
                    "detector index",
                    "detector_index",
                ],
                prop_path,
                "detector id",
            )
            beam_col = require_col(header, ["beam id", "beam_id", "beam index", "beam_idx", "beam"], prop_path, "beam id")
            angle_col = require_col(header, ["Angle (rad)", "angle (rad)", "angle", "Angle"], prop_path, "angle")
            fwhm_col = require_col(header, ["FWHM (mm)", "fwhm (mm)", "FWHM"], prop_path, "FWHM")
            sens_col = require_col(header, ["sensitivity"], prop_path, "sensitivity")
        except Exception as exc:
            return ASCIProxyResult(
                status="failed_property_mapping",
                warning=f"{prop_path.name}: {exc}",
                asci_pct=float("nan"),
                n_pixels=n_pixels,
                n_angle_bins=n_angle_bins,
                n_beams_used=n_beams_used,
                n_mask_rows_seen=n_mask_rows_seen,
            )

        if properties.size == 0:
            warnings.append(f"{prop_path.name}: empty beam_properties")
            continue

        detector_ids = np.asarray(np.rint(properties[:, detector_col]), dtype=np.int64)
        beam_ids = np.asarray(np.rint(properties[:, beam_col]), dtype=np.int64)
        angles = np.asarray(properties[:, angle_col], dtype=np.float64)
        fwhm = np.asarray(properties[:, fwhm_col], dtype=np.float64)
        sensitivity = np.asarray(properties[:, sens_col], dtype=np.float64)

        in_range = (detector_ids >= 0) & (detector_ids < total_rows)
        sampled = np.zeros(detector_ids.shape, dtype=bool)
        sampled[in_range] = sampled_lookup[detector_ids[in_range]]
        valid = (
            sampled
            & np.isfinite(angles)
            & np.isfinite(fwhm)
            & np.isfinite(sensitivity)
            & (fwhm > 0.0)
        )
        if fwhm_min is not None:
            valid &= fwhm >= float(fwhm_min)
        if fwhm_max is not None:
            valid &= fwhm <= float(fwhm_max)

        if not np.any(valid):
            warnings.append(f"{prop_path.name}: no valid sampled beam rows before sensitivity threshold")
            continue

        sensitivity_max = float(np.max(sensitivity[valid]))
        if math.isfinite(sensitivity_max) and sensitivity_max > 0.0:
            valid &= sensitivity > sensitivity_max * 0.01

        if not np.any(valid):
            warnings.append(f"{prop_path.name}: no valid sampled beam rows after sensitivity threshold")
            continue

        unit = infer_angle_unit(angles[valid], angle_unit)
        angle_bins = angles_to_bins(angles[valid], n_angle_bins, unit)
        detector_ids_valid = detector_ids[valid]
        beam_ids_valid = beam_ids[valid]

        try:
            mask_handle, mask_dataset = open_mask_dataset(mask_path)
            try:
                if len(mask_dataset.shape) != 2:
                    return ASCIProxyResult(
                        status="failed_mask_shape",
                        warning=f"{mask_path.name}: expected 2-D beam_mask, got {tuple(mask_dataset.shape)}",
                        asci_pct=float("nan"),
                        n_pixels=n_pixels,
                        n_angle_bins=n_angle_bins,
                        n_beams_used=n_beams_used,
                        n_mask_rows_seen=n_mask_rows_seen,
                    )
                if int(mask_dataset.shape[1]) != n_pixels:
                    return ASCIProxyResult(
                        status="failed_mask_pixel_width",
                        warning=f"{mask_path.name}: mask width {mask_dataset.shape[1]} != n_pixels {n_pixels}",
                        asci_pct=float("nan"),
                        n_pixels=n_pixels,
                        n_angle_bins=n_angle_bins,
                        n_beams_used=n_beams_used,
                        n_mask_rows_seen=n_mask_rows_seen,
                    )
                if int(mask_dataset.shape[0]) < total_rows:
                    return ASCIProxyResult(
                        status="failed_mask_row_count",
                        warning=f"{mask_path.name}: mask rows {mask_dataset.shape[0]} < detector rows {total_rows}",
                        asci_pct=float("nan"),
                        n_pixels=n_pixels,
                        n_angle_bins=n_angle_bins,
                        n_beams_used=n_beams_used,
                        n_mask_rows_seen=n_mask_rows_seen,
                    )

                n_mask_rows_seen += int(mask_dataset.shape[0])
                mask_is_bool = bool(mask_dataset.dtype == np.dtype(bool))
                for detector_id, beam_id, angle_bin in zip(detector_ids_valid, beam_ids_valid, angle_bins):
                    detector_idx = int(detector_id)
                    beam_idx = int(beam_id)
                    mask_row = np.asarray(mask_dataset[detector_idx, :])
                    if mask_is_bool:
                        pixels = mask_row > 0
                    else:
                        pixels = mask_row == beam_idx
                    if np.any(pixels):
                        hist[pixels, int(angle_bin)] = True
                    n_beams_used += 1
            finally:
                mask_handle.close()
        except Exception as exc:
            return ASCIProxyResult(
                status="failed_mask_read",
                warning=f"{mask_path.name}: {exc}",
                asci_pct=float("nan"),
                n_pixels=n_pixels,
                n_angle_bins=n_angle_bins,
                n_beams_used=n_beams_used,
                n_mask_rows_seen=n_mask_rows_seen,
            )

    if n_beams_used == 0:
        return ASCIProxyResult(
            status="failed_no_asci_beams_used",
            warning="; ".join(w for w in warnings if w),
            asci_pct=float("nan"),
            n_pixels=n_pixels,
            n_angle_bins=n_angle_bins,
            n_beams_used=0,
            n_mask_rows_seen=n_mask_rows_seen,
        )

    asci_pct = 100.0 * float(np.count_nonzero(hist)) / float(max(hist.size, 1))
    return ASCIProxyResult(
        status="success",
        warning="; ".join(w for w in warnings if w),
        asci_pct=asci_pct,
        n_pixels=n_pixels,
        n_angle_bins=n_angle_bins,
        n_beams_used=n_beams_used,
        n_mask_rows_seen=n_mask_rows_seen,
    )


def compute_ji_proxy(sensitivity_mean: float, fwhm_mean: float, asci_pct: float) -> tuple[float, str]:
    if not (is_finite(sensitivity_mean) and is_finite(fwhm_mean) and is_finite(asci_pct)):
        return float("nan"), "not_computed_missing_component"
    if fwhm_mean <= EPS:
        return float("nan"), "not_computed_nonpositive_fwhm"
    return (sensitivity_mean / (fwhm_mean ** 2)) * (asci_pct / 100.0), "success"


def base_row(run_paths: RunPaths, full_metrics: FullMetrics | None = None, metadata: PPDFMetadata | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {key: "" for key in CSV_COLUMNS}
    row.update(
        {
            "run_dir": str(run_paths.run_dir),
            "data_dir": str(run_paths.data_dir),
            "results_dir": str(run_paths.results_dir),
        }
    )
    if metadata is not None:
        row.update(
            {
                "n_ppdf_files": metadata.n_files,
                "first_ppdf_file": str(metadata.first_ppdf_file),
                "ppdf_shape": fmt_shape(metadata.ppdf_shape),
                "ppdf_dtype": metadata.ppdf_dtype,
                "total_rows": metadata.total_rows,
                "n_pixels": metadata.n_pixels,
                "pixel_shape": fmt_shape(metadata.pixel_shape),
            }
        )
    if full_metrics is not None:
        row.update(
            {
                "sensitivity_total_full_csv": full_metrics.sensitivity_total,
                "sensitivity_mean_full_csv": full_metrics.sensitivity_mean,
                "fwhm_full_csv": full_metrics.fwhm_mean,
                "asci_full_csv": full_metrics.asci_pct,
                "JI_full_csv": full_metrics.ji,
                "full_metrics_status": full_metrics.status,
            }
        )
    return row


def process_run(
    run_paths: RunPaths,
    args: argparse.Namespace,
    fractions: list[float],
    seeds: list[int],
    modes: list[str],
) -> list[dict[str, Any]]:
    full_metrics = load_full_metrics(run_paths.results_dir)
    ppdf_files = find_files(run_paths.data_dir, args.ppdf_glob_pattern, args.max_files)
    metadata = inspect_ppdf_metadata(ppdf_files, args.dataset_name)
    detector_metadata = build_detector_row_metadata(metadata.total_rows)
    row_signal = compute_detector_row_signal(ppdf_files, args.dataset_name, metadata.ppdf_shape)
    prop_data = load_beam_properties_for_fwhm(
        run_paths.data_dir,
        args.props_glob_pattern,
        metadata.total_rows,
        args.fwhm_min_mm,
        args.fwhm_max_mm,
    )

    rows: list[dict[str, Any]] = []
    for mode in modes:
        for fraction in fractions:
            for seed in seeds:
                row = base_row(run_paths, full_metrics, metadata)
                try:
                    sampled_rows, actual_fraction, description = make_sampled_rows(
                        metadata.total_rows, fraction, mode, seed, detector_metadata
                    )
                    balance = summarize_sample_balance(sampled_rows, detector_metadata)
                    full_sensitivity, sens_total_proxy, sens_mean_proxy = compute_sensitivity_from_row_signal(
                        row_signal,
                        sampled_rows,
                        metadata.n_pixels,
                        metadata.n_files,
                    )
                    fwhm_proxy, n_fwhm_sampled, fwhm_status, fwhm_warning = compute_fwhm_proxy(prop_data, sampled_rows)
                    asci_result = compute_asci_proxy(
                        run_paths.data_dir,
                        args.props_glob_pattern,
                        args.masks_glob_pattern,
                        sampled_rows,
                        metadata.total_rows,
                        metadata.n_pixels,
                        args.n_angle_bins,
                        args.angle_unit,
                        args.fwhm_min_mm,
                        args.fwhm_max_mm,
                    )
                    ji_proxy, ji_status = compute_ji_proxy(sens_mean_proxy, fwhm_proxy, asci_result.asci_pct)

                    sens_truth = full_metrics.sensitivity_total if is_finite(full_metrics.sensitivity_total) else full_sensitivity
                    fwhm_truth = full_metrics.fwhm_mean if is_finite(full_metrics.fwhm_mean) else prop_data.full_fwhm_from_props
                    asci_truth = full_metrics.asci_pct
                    ji_truth = full_metrics.ji

                    if ji_status == "success" and asci_result.status == "success":
                        status = "success_all_metrics"
                    elif is_finite(fwhm_proxy):
                        status = "success_sensitivity_fwhm_only"
                    elif is_finite(sens_total_proxy):
                        status = "success_sensitivity_only"
                    else:
                        status = "failed"

                    row.update(
                        {
                            "status": status,
                            "mode": mode,
                            "fraction": fraction,
                            "seed": seed,
                            "n_sampled_rows": int(sampled_rows.size),
                            "sampled_fraction_actual": actual_fraction,
                            "sampling_description": description,
                            **balance,
                            "full_sensitivity_from_ppdf": full_sensitivity,
                            "sensitivity_total_row_proxy": sens_total_proxy,
                            "sensitivity_mean_row_proxy": sens_mean_proxy,
                            "sensitivity_abs_error": sens_total_proxy - sens_truth if is_finite(sens_truth) else float("nan"),
                            "sensitivity_rel_error_pct": rel_error_pct(sens_total_proxy, sens_truth),
                            "full_fwhm_from_props": prop_data.full_fwhm_from_props,
                            "fwhm_row_proxy": fwhm_proxy,
                            "fwhm_abs_error": fwhm_proxy - fwhm_truth if is_finite(fwhm_truth) and is_finite(fwhm_proxy) else float("nan"),
                            "fwhm_rel_error_pct": rel_error_pct(fwhm_proxy, fwhm_truth),
                            "n_fwhm_full_valid": prop_data.n_full_valid,
                            "n_fwhm_sampled_valid": n_fwhm_sampled,
                            "fwhm_status": fwhm_status,
                            "fwhm_mapping_warning": fwhm_warning,
                            "asci_row_proxy": asci_result.asci_pct,
                            "asci_abs_error": asci_result.asci_pct - asci_truth if is_finite(asci_truth) and is_finite(asci_result.asci_pct) else float("nan"),
                            "asci_rel_error_pct": rel_error_pct(asci_result.asci_pct, asci_truth),
                            "n_asci_pixels": asci_result.n_pixels,
                            "n_angle_bins": asci_result.n_angle_bins,
                            "n_asci_beams_used": asci_result.n_beams_used,
                            "n_asci_mask_rows_seen": asci_result.n_mask_rows_seen,
                            "asci_status": asci_result.status,
                            "asci_mapping_warning": asci_result.warning,
                            "JI_row_proxy": ji_proxy,
                            "JI_abs_error": ji_proxy - ji_truth if is_finite(ji_proxy) and is_finite(ji_truth) else float("nan"),
                            "JI_rel_error_pct": rel_error_pct(ji_proxy, ji_truth),
                            "JI_status": ji_status,
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
                        }
                    )
                    if args.strict:
                        raise
                rows.append(row)
    return rows


def failed_run_row(run_paths: RunPaths, error: Exception) -> dict[str, Any]:
    row = base_row(run_paths, load_full_metrics(run_paths.results_dir))
    row.update({"status": "failed", "error_message": str(error)})
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in CSV_COLUMNS})


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(key))
        if is_finite(value):
            values.append(abs(value))
    return values


def print_metric_summary(rows: list[dict[str, Any]], key: str, label: str) -> None:
    values = numeric_values(rows, key)
    if values:
        print(f"  {label}: mean={sum(values) / len(values):.4f}% max={max(values):.4f}%")
    else:
        print(f"  {label}: unavailable")


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode", ""))
        fraction = finite_float(row.get("fraction"))
        if mode and is_finite(fraction):
            grouped.setdefault((mode, fraction), []).append(row)
    return grouped


def group_rows_with_seed(rows: list[dict[str, Any]]) -> dict[tuple[str, float, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row.get("mode", ""))
        fraction = finite_float(row.get("fraction"))
        try:
            seed = int(row.get("seed"))
        except Exception:
            continue
        if mode and is_finite(fraction):
            grouped.setdefault((mode, fraction, seed), []).append(row)
    return grouped


def ranking_key(row: dict[str, Any], key: str) -> float:
    value = finite_float(row.get(key))
    return value if is_finite(value) else -math.inf


def print_ranking_checks(rows: list[dict[str, Any]]) -> None:
    print("\nRanking checks:")
    for (mode, fraction, seed), group in sorted(group_rows_with_seed(rows).items()):
        valid = [
            row
            for row in group
            if is_finite(finite_float(row.get("JI_full_csv"))) and is_finite(finite_float(row.get("JI_row_proxy")))
        ]
        if len(valid) < 2:
            print(f"  {mode} fraction={fraction:g} seed={seed}: insufficient valid JI rows")
            continue
        full_order = [str(row["run_dir"]) for row in sorted(valid, key=lambda row: ranking_key(row, "JI_full_csv"), reverse=True)]
        proxy_order = [str(row["run_dir"]) for row in sorted(valid, key=lambda row: ranking_key(row, "JI_row_proxy"), reverse=True)]
        print(
            f"  {mode} fraction={fraction:g} seed={seed}: "
            f"exact_match={full_order == proxy_order} top1_match={full_order[0] == proxy_order[0]}"
        )


def print_recommendation(rows: list[dict[str, Any]]) -> None:
    candidates = [
        row
        for row in rows
        if math.isclose(finite_float(row.get("fraction")), 0.5, rel_tol=0.0, abs_tol=1e-8)
        and str(row.get("status")) == "success_all_metrics"
    ]
    if not candidates:
        print("\nRecommendation: Caution")
        print("  No successful 50% rows with full proxy JI were available.")
        print("  These results are preliminary and should be repeated on more candidates before BO integration.")
        return

    sens = numeric_values(candidates, "sensitivity_rel_error_pct")
    fwhm = numeric_values(candidates, "fwhm_rel_error_pct")
    asci = numeric_values(candidates, "asci_rel_error_pct")
    ji = numeric_values(candidates, "JI_rel_error_pct")
    mean_sens = sum(sens) / len(sens) if sens else math.inf
    mean_fwhm = sum(fwhm) / len(fwhm) if fwhm else math.inf
    mean_asci = sum(asci) / len(asci) if asci else math.inf
    mean_ji = sum(ji) / len(ji) if ji else math.inf

    ranking_groups = group_rows_with_seed(candidates)
    top_matches = 0
    ranking_checks = 0
    exact_matches = 0
    for group in ranking_groups.values():
        valid = [
            row
            for row in group
            if is_finite(finite_float(row.get("JI_full_csv"))) and is_finite(finite_float(row.get("JI_row_proxy")))
        ]
        if len(valid) < 2:
            continue
        full_order = [str(row["run_dir"]) for row in sorted(valid, key=lambda row: ranking_key(row, "JI_full_csv"), reverse=True)]
        proxy_order = [str(row["run_dir"]) for row in sorted(valid, key=lambda row: ranking_key(row, "JI_row_proxy"), reverse=True)]
        ranking_checks += 1
        exact_matches += int(full_order == proxy_order)
        top_matches += int(full_order[0] == proxy_order[0])

    if mean_fwhm > 5.0 or mean_asci > 5.0 or (ranking_checks and top_matches < ranking_checks):
        verdict = "Do not proceed"
    elif mean_sens < 1.0 and mean_fwhm < 2.0 and mean_asci < 2.0 and mean_ji < 3.0 and (
        ranking_checks == 0 or top_matches == ranking_checks
    ):
        verdict = "Proceed"
    else:
        verdict = "Caution"

    print(f"\nRecommendation: {verdict}")
    print(
        "  50% proxy means: "
        f"sensitivity={mean_sens:.4f}%, FWHM={mean_fwhm:.4f}%, "
        f"ASCI={mean_asci:.4f}%, JI={mean_ji:.4f}%"
    )
    if ranking_checks:
        print(f"  Ranking: exact_matches={exact_matches}/{ranking_checks}, top1_matches={top_matches}/{ranking_checks}")
    else:
        print("  Ranking: insufficient valid rows for ranking checks")
    print("  These results are preliminary and should be repeated on more candidates before BO integration.")


def print_summary(rows: list[dict[str, Any]], out_csv: Path) -> None:
    success_rows = [row for row in rows if str(row.get("status", "")).startswith("success")]
    failed_rows = [row for row in rows if row.get("status") == "failed"]
    print(f"\nCSV path: {out_csv}")
    print(f"Rows written: {len(rows)}")
    print(f"Successful rows: {len(success_rows)}")
    print(f"Failed rows: {len(failed_rows)}")

    print("\nPer run full metrics:")
    seen_runs: set[str] = set()
    for row in rows:
        run_dir = str(row.get("run_dir", ""))
        if not run_dir or run_dir in seen_runs:
            continue
        seen_runs.add(run_dir)
        print(
            f"  {run_dir}: "
            f"JI={row.get('JI_full_csv', '')} "
            f"sensitivity_total={row.get('sensitivity_total_full_csv', '')} "
            f"sensitivity_mean={row.get('sensitivity_mean_full_csv', '')} "
            f"FWHM={row.get('fwhm_full_csv', '')} "
            f"ASCI={row.get('asci_full_csv', '')}"
        )

    print("\nGrouped proxy errors:")
    for (mode, fraction), group in sorted(group_rows(rows).items()):
        print(f"  {mode} fraction={fraction:g}:")
        print_metric_summary(group, "sensitivity_rel_error_pct", "mean/max abs sensitivity rel error")
        print_metric_summary(group, "fwhm_rel_error_pct", "mean/max abs FWHM rel error")
        print_metric_summary(group, "asci_rel_error_pct", "mean/max abs ASCI rel error")
        print_metric_summary(group, "JI_rel_error_pct", "mean/max abs JI rel error")
        sampled = [finite_float(row.get("sampled_fraction_actual")) for row in group]
        sampled = [value for value in sampled if is_finite(value)]
        angle_bins = [finite_float(row.get("angle_bins_covered")) for row in group]
        angle_bins = [value for value in angle_bins if is_finite(value)]
        if sampled:
            print(f"  mean sampled fraction: {sum(sampled) / len(sampled):.6f}")
        if angle_bins:
            print(f"  min angle bins covered: {min(angle_bins):.0f}")

    print_ranking_checks(rows)
    print_recommendation(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate detector-row SRM sampling by reconstructing sensitivity, FWHM, ASCI, and proxy JI."
    )
    parser.add_argument("--run-dirs", required=True, help="Comma-separated run directories or data directories.")
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS, help="Comma-separated row fractions in (0,1].")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated integer seeds.")
    parser.add_argument("--modes", default=DEFAULT_MODES, help="Comma-separated sampling modes.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--ppdf-glob-pattern", default=DEFAULT_PPDF_GLOB)
    parser.add_argument("--props-glob-pattern", default=DEFAULT_PROPS_GLOB)
    parser.add_argument("--masks-glob-pattern", default=DEFAULT_MASKS_GLOB)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--n-angle-bins", type=int, default=360)
    parser.add_argument("--angle-unit", choices=["auto", "rad", "deg"], default="auto")
    parser.add_argument("--fwhm-min-mm", type=float, default=None)
    parser.add_argument("--fwhm-max-mm", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        require_runtime_deps()
        fractions = parse_csv_list(args.fractions, float, "fractions")
        seeds = parse_csv_list(args.seeds, int, "seeds")
        modes = parse_csv_list(args.modes, str, "modes")
        allowed_modes = {"every_k", "ring_cell_stratified", "ring_cell_random", "random"}
        unsupported = sorted(set(modes) - allowed_modes)
        if unsupported:
            raise ValueError(f"Unsupported sampling modes: {unsupported}; allowed={sorted(allowed_modes)}")
        if args.n_angle_bins <= 0:
            raise ValueError("--n-angle-bins must be positive")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    rows: list[dict[str, Any]] = []
    run_inputs = parse_csv_list(args.run_dirs, str, "run-dirs")
    for run_text in run_inputs:
        run_paths = resolve_run_and_data_dir(run_text)
        if args.verbose:
            print(f"[INFO] Processing {run_paths.run_dir} (data={run_paths.data_dir})")
        try:
            rows.extend(process_run(run_paths, args, fractions, seeds, modes))
        except Exception as exc:
            if args.strict:
                raise
            print(f"[WARN] Failed run {run_paths.run_dir}: {exc}", file=sys.stderr)
            rows.append(failed_run_row(run_paths, exc))

    out_csv = Path(args.out_csv)
    write_csv(out_csv, rows)
    print_summary(rows, out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
