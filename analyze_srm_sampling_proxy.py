#!/usr/bin/env python3
"""Validate source-pixel sampling proxies from existing full PPDF/SRM files.

This script is read-only with respect to pipeline data. It loads already
computed PPDF files, computes a full sensitivity proxy, and compares it with
balanced source-pixel subsets. It does not save runtime or alter PPDF
generation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


CSV_FIELDS = [
    "run_dir",
    "data_dir",
    "n_ppdf_files",
    "dataset_name",
    "first_ppdf_file",
    "ppdf_shape",
    "ppdf_dtype",
    "pixel_shape",
    "n_pixels",
    "mode",
    "fraction",
    "seed",
    "tile_size",
    "n_sampled",
    "full_sensitivity_mean",
    "partial_sensitivity_mean",
    "absolute_error",
    "relative_error_pct",
    "sampled_fraction_actual",
    "status",
    "error_message",
]

ALLOWED_MODES = {"grid", "checkerboard", "stratified"}


@dataclass(frozen=True)
class PixelInfo:
    response_axis: int
    pixel_shape: tuple[int, ...]
    n_pixels: int


@dataclass(frozen=True)
class PpdfMetadata:
    n_files: int
    first_ppdf_file: str
    ppdf_shape: tuple[int, ...]
    ppdf_dtype: str
    pixel_shape: tuple[int, ...]
    n_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read existing PPDF/SRM files and validate source-pixel sampling sensitivity proxies."
    )
    parser.add_argument("--run-dirs", required=True, help="Comma-separated run directories or data directories.")
    parser.add_argument("--fractions", default="0.25,0.50", help="Comma-separated sample fractions.")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated random seeds.")
    parser.add_argument("--modes", default="grid,checkerboard,stratified", help="Comma-separated sampling modes.")
    parser.add_argument("--tile-size", type=int, default=16, help="Tile/chunk size for stratified sampling.")
    parser.add_argument("--dataset-name", default="ppdfs", help="HDF5 dataset name to read.")
    parser.add_argument("--glob-pattern", default="position_*_ppdfs_t8_*.hdf5", help="PPDF filename glob.")
    parser.add_argument("--out-csv", required=True, help="CSV output path.")
    parser.add_argument("--max-files", type=int, default=0, help="If > 0, only read the first N PPDF files.")
    parser.add_argument("--verbose", action="store_true", help="Print per-run progress.")
    args = parser.parse_args()

    args.run_dirs = parse_string_list(args.run_dirs, "--run-dirs")
    args.fractions = parse_float_list(args.fractions, "--fractions")
    args.seeds = parse_int_list(args.seeds, "--seeds")
    args.modes = parse_modes(args.modes)

    if args.tile_size <= 0:
        parser.error("--tile-size must be positive")
    if args.max_files < 0:
        parser.error("--max-files must be non-negative")
    for fraction in args.fractions:
        if not 0.0 < fraction <= 1.0:
            parser.error(f"--fractions values must be > 0 and <= 1; got {fraction}")
    return args


def parse_string_list(value: str, flag_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"{flag_name} must contain at least one value")
    return items


def parse_float_list(value: str, flag_name: str) -> list[float]:
    try:
        return [float(item) for item in parse_string_list(value, flag_name)]
    except ValueError as exc:
        raise SystemExit(f"{flag_name} must contain comma-separated floats") from exc


def parse_int_list(value: str, flag_name: str) -> list[int]:
    try:
        return [int(item) for item in parse_string_list(value, flag_name)]
    except ValueError as exc:
        raise SystemExit(f"{flag_name} must contain comma-separated integers") from exc


def parse_modes(value: str) -> list[str]:
    modes = parse_string_list(value, "--modes")
    unknown = sorted(set(modes) - ALLOWED_MODES)
    if unknown:
        raise SystemExit(f"--modes contains unsupported modes: {', '.join(unknown)}")
    return modes


def infer_pixel_info(ppdf_shape: Sequence[int]) -> PixelInfo:
    shape = tuple(int(dim) for dim in ppdf_shape)
    if len(shape) == 2:
        return PixelInfo(response_axis=0, pixel_shape=(shape[1],), n_pixels=shape[1])
    if len(shape) == 3:
        return PixelInfo(response_axis=0, pixel_shape=(shape[1], shape[2]), n_pixels=shape[1] * shape[2])
    raise ValueError(
        f"Unsupported PPDF shape {shape}; expected (n_response_bins, n_pixels) "
        "or (n_response_bins, nx, ny)."
    )


def is_perfect_square(value: int) -> bool:
    side = math.isqrt(value)
    return side * side == value


def grid_shape_for_sampling(pixel_shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(pixel_shape) == 2:
        return int(pixel_shape[0]), int(pixel_shape[1])
    if len(pixel_shape) == 1 and is_perfect_square(int(pixel_shape[0])):
        side = math.isqrt(int(pixel_shape[0]))
        return side, side
    return None


def shape_to_string(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape)


def resolve_data_dir(run_dir_text: str) -> tuple[Path, Path]:
    run_dir = Path(run_dir_text)
    candidate_data_dir = run_dir / "data"
    if candidate_data_dir.exists():
        return run_dir, candidate_data_dir
    return run_dir, run_dir


def find_ppdf_files(data_dir: Path, glob_pattern: str, max_files: int) -> list[Path]:
    files = [Path(path) for path in sorted(glob.glob(os.path.join(str(data_dir), glob_pattern)))]
    if max_files > 0:
        return files[:max_files]
    return files


def available_hdf5_keys(handle: h5py.File) -> list[str]:
    keys: list[str] = []
    handle.visit(keys.append)
    return keys


def compute_per_pixel_sum(
    files: Sequence[Path],
    dataset_name: str,
) -> tuple[np.ndarray, PpdfMetadata]:
    if not files:
        raise ValueError("No PPDF files were provided")

    total_per_pixel_sum: np.ndarray | None = None
    expected_pixel_shape: tuple[int, ...] | None = None
    first_shape: tuple[int, ...] | None = None
    first_dtype = ""

    for index, path in enumerate(files):
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle:
                keys = available_hdf5_keys(handle)
                raise KeyError(
                    f"Missing dataset {dataset_name!r} in {path}. Available HDF5 entries: {keys}"
                )

            dset = handle[dataset_name]
            info = infer_pixel_info(dset.shape)
            if expected_pixel_shape is None:
                expected_pixel_shape = info.pixel_shape
                first_shape = tuple(int(dim) for dim in dset.shape)
                first_dtype = str(dset.dtype)
                total_per_pixel_sum = np.zeros(info.n_pixels, dtype=np.float64)
            elif info.pixel_shape != expected_pixel_shape:
                raise ValueError(
                    f"Incompatible pixel shape in {path}: {info.pixel_shape}; "
                    f"expected {expected_pixel_shape}"
                )

            arr = dset[...]
            per_file = np.sum(arr, axis=info.response_axis, dtype=np.float64).reshape(-1)
            if total_per_pixel_sum is None:
                raise RuntimeError("Internal error: accumulator was not initialized")
            if per_file.size != total_per_pixel_sum.size:
                raise ValueError(
                    f"Per-pixel sum from {path} has {per_file.size} pixels; "
                    f"expected {total_per_pixel_sum.size}"
                )
            total_per_pixel_sum += per_file

        if index == 0 and total_per_pixel_sum is None:
            raise RuntimeError("Internal error: first PPDF file was not accumulated")

    if total_per_pixel_sum is None or expected_pixel_shape is None or first_shape is None:
        raise RuntimeError("No PPDF data were accumulated")

    metadata = PpdfMetadata(
        n_files=len(files),
        first_ppdf_file=str(files[0]),
        ppdf_shape=first_shape,
        ppdf_dtype=first_dtype,
        pixel_shape=expected_pixel_shape,
        n_pixels=int(total_per_pixel_sum.size),
    )
    return total_per_pixel_sum, metadata


def validate_indices(indices: np.ndarray, n_pixels: int) -> np.ndarray:
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    if indices.size == 0:
        raise ValueError("Sampling produced zero pixels")
    if int(indices[0]) < 0 or int(indices[-1]) >= n_pixels:
        raise ValueError(f"Sampling produced indices outside [0, {n_pixels})")
    return indices


def require_grid_shape(pixel_shape: tuple[int, ...], mode: str) -> tuple[int, int]:
    grid_shape = grid_shape_for_sampling(pixel_shape)
    if grid_shape is None:
        raise ValueError(
            f"{mode} sampling requires 2D pixel coordinates. Pixel shape {pixel_shape} "
            "is 1D and n_pixels is not a perfect square."
        )
    return grid_shape


def make_grid_indices(pixel_shape: tuple[int, ...], fraction: float) -> np.ndarray:
    nx, ny = require_grid_shape(pixel_shape, "grid")
    x_coords, y_coords = np.indices((nx, ny))
    index_grid = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    if math.isclose(fraction, 0.25, rel_tol=0.0, abs_tol=1e-9):
        return index_grid[(x_coords % 2 == 0) & (y_coords % 2 == 0)].reshape(-1)
    if math.isclose(fraction, 0.50, rel_tol=0.0, abs_tol=1e-9):
        return index_grid[(x_coords + y_coords) % 2 == 0].reshape(-1)
    raise ValueError("grid sampling supports only fractions 0.25 and 0.50; use stratified otherwise")


def make_checkerboard_indices(pixel_shape: tuple[int, ...], fraction: float) -> np.ndarray:
    nx, ny = require_grid_shape(pixel_shape, "checkerboard")
    x_coords, y_coords = np.indices((nx, ny))
    index_grid = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
    if math.isclose(fraction, 0.50, rel_tol=0.0, abs_tol=1e-9):
        return index_grid[(x_coords + y_coords) % 2 == 0].reshape(-1)
    if math.isclose(fraction, 0.25, rel_tol=0.0, abs_tol=1e-9):
        return index_grid[(x_coords % 2 == 0) & (y_coords % 2 == 0)].reshape(-1)
    raise ValueError("checkerboard sampling supports only fractions 0.25 and 0.50")


def make_stratified_indices(
    pixel_shape: tuple[int, ...],
    fraction: float,
    seed: int,
    tile_size: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid_shape = grid_shape_for_sampling(pixel_shape)
    sampled_chunks: list[np.ndarray] = []

    if grid_shape is not None:
        nx, ny = grid_shape
        index_grid = np.arange(nx * ny, dtype=np.int64).reshape(nx, ny)
        for x_start in range(0, nx, tile_size):
            for y_start in range(0, ny, tile_size):
                tile = index_grid[x_start : x_start + tile_size, y_start : y_start + tile_size].reshape(-1)
                count = max(1, int(math.ceil(fraction * tile.size)))
                sampled_chunks.append(rng.choice(tile, size=min(count, tile.size), replace=False))
    else:
        n_pixels = int(pixel_shape[0])
        all_indices = np.arange(n_pixels, dtype=np.int64)
        for start in range(0, n_pixels, tile_size):
            chunk = all_indices[start : start + tile_size]
            count = max(1, int(math.ceil(fraction * chunk.size)))
            sampled_chunks.append(rng.choice(chunk, size=min(count, chunk.size), replace=False))

    return np.concatenate(sampled_chunks) if sampled_chunks else np.array([], dtype=np.int64)


def make_sample_indices(
    pixel_shape: tuple[int, ...],
    fraction: float,
    mode: str,
    seed: int,
    tile_size: int,
) -> np.ndarray:
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be > 0 and <= 1; got {fraction}")

    n_pixels = int(np.prod(pixel_shape))
    if fraction == 1.0:
        return np.arange(n_pixels, dtype=np.int64)
    if mode == "grid":
        indices = make_grid_indices(pixel_shape, fraction)
    elif mode == "checkerboard":
        indices = make_checkerboard_indices(pixel_shape, fraction)
    elif mode == "stratified":
        indices = make_stratified_indices(pixel_shape, fraction, seed, tile_size)
    else:
        raise ValueError(f"Unsupported sampling mode: {mode}")
    return validate_indices(indices, n_pixels)


def csv_float(value: float) -> str:
    return repr(float(value)) if math.isfinite(float(value)) else ""


def base_row(
    run_dir: Path,
    data_dir: Path,
    dataset_name: str,
    tile_size: int,
    metadata: PpdfMetadata | None = None,
) -> dict[str, str]:
    return {
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        "n_ppdf_files": str(metadata.n_files) if metadata else "",
        "dataset_name": dataset_name,
        "first_ppdf_file": metadata.first_ppdf_file if metadata else "",
        "ppdf_shape": shape_to_string(metadata.ppdf_shape) if metadata else "",
        "ppdf_dtype": metadata.ppdf_dtype if metadata else "",
        "pixel_shape": shape_to_string(metadata.pixel_shape) if metadata else "",
        "n_pixels": str(metadata.n_pixels) if metadata else "",
        "mode": "",
        "fraction": "",
        "seed": "",
        "tile_size": str(tile_size),
        "n_sampled": "",
        "full_sensitivity_mean": "",
        "partial_sensitivity_mean": "",
        "absolute_error": "",
        "relative_error_pct": "",
        "sampled_fraction_actual": "",
        "status": "",
        "error_message": "",
    }


def error_row(
    run_dir: Path,
    data_dir: Path,
    dataset_name: str,
    tile_size: int,
    error_message: str,
    metadata: PpdfMetadata | None = None,
) -> dict[str, str]:
    row = base_row(run_dir, data_dir, dataset_name, tile_size, metadata)
    row["status"] = "failed"
    row["error_message"] = error_message
    return row


def evaluate_sampling(
    total_per_pixel_sum: np.ndarray,
    metadata: PpdfMetadata,
    run_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    full_sensitivity_mean = float(np.mean(total_per_pixel_sum))
    full_has_relative_error = math.isfinite(full_sensitivity_mean) and full_sensitivity_mean != 0.0

    for mode in args.modes:
        for fraction in args.fractions:
            for seed in args.seeds:
                row = base_row(run_dir, data_dir, args.dataset_name, args.tile_size, metadata)
                row["mode"] = mode
                row["fraction"] = csv_float(fraction)
                row["seed"] = str(seed)
                row["full_sensitivity_mean"] = csv_float(full_sensitivity_mean)
                try:
                    sampled_indices = make_sample_indices(
                        metadata.pixel_shape,
                        fraction,
                        mode,
                        seed,
                        args.tile_size,
                    )
                    partial_sensitivity_mean = float(np.mean(total_per_pixel_sum[sampled_indices]))
                    absolute_error = partial_sensitivity_mean - full_sensitivity_mean
                    relative_error_pct = (
                        100.0 * absolute_error / full_sensitivity_mean
                        if full_has_relative_error
                        else math.nan
                    )
                    row["n_sampled"] = str(int(sampled_indices.size))
                    row["partial_sensitivity_mean"] = csv_float(partial_sensitivity_mean)
                    row["absolute_error"] = csv_float(absolute_error)
                    row["relative_error_pct"] = csv_float(relative_error_pct)
                    row["sampled_fraction_actual"] = csv_float(sampled_indices.size / metadata.n_pixels)
                    row["status"] = "success"
                    if not full_has_relative_error:
                        row["error_message"] = (
                            "relative_error_pct unavailable because full_sensitivity_mean is zero or non-finite"
                        )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error_message"] = str(exc)
                rows.append(row)
    return rows


def process_run(run_dir_text: str, args: argparse.Namespace) -> list[dict[str, str]]:
    run_dir, data_dir = resolve_data_dir(run_dir_text)
    files = find_ppdf_files(data_dir, args.glob_pattern, args.max_files)
    if not files:
        message = f"No PPDF files found in {data_dir} matching {args.glob_pattern!r}"
        return [error_row(run_dir, data_dir, args.dataset_name, args.tile_size, message)]

    if args.verbose:
        print(f"Reading {len(files)} PPDF files from {data_dir}")

    try:
        total_per_pixel_sum, metadata = compute_per_pixel_sum(files, args.dataset_name)
    except Exception as exc:
        return [error_row(run_dir, data_dir, args.dataset_name, args.tile_size, str(exc))]

    return evaluate_sampling(total_per_pixel_sum, metadata, run_dir, data_dir, args)


def safe_row_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if not value:
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def print_summary(out_csv: Path, rows: Sequence[dict[str, str]]) -> None:
    successful = [row for row in rows if row.get("status") == "success"]
    failed = [row for row in rows if row.get("status") != "success"]
    print(f"CSV written: {out_csv}")
    print(f"Rows written: {len(rows)}")
    print(f"Successful rows: {len(successful)}")
    print(f"Failed rows: {len(failed)}")

    full_by_run: dict[str, float] = {}
    for row in successful:
        full_value = safe_row_float(row, "full_sensitivity_mean")
        if math.isfinite(full_value):
            full_by_run.setdefault(row["run_dir"], full_value)
    for run_dir, full_value in sorted(full_by_run.items()):
        print(f"Full sensitivity mean [{run_dir}]: {full_value:.12g}")

    grouped_errors: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in successful:
        relative_error = safe_row_float(row, "relative_error_pct")
        if math.isfinite(relative_error):
            grouped_errors[(row["fraction"], row["mode"])].append(abs(relative_error))

    for (fraction, mode), values in sorted(grouped_errors.items()):
        mean_abs_relative_error = float(np.mean(values))
        print(
            f"Mean absolute relative error [{mode}, fraction={fraction}]: "
            f"{mean_abs_relative_error:.6g}%"
        )


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for run_dir_text in args.run_dirs:
        rows.extend(process_run(run_dir_text, args))

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(out_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
