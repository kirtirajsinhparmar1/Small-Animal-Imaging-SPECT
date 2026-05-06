#!/usr/bin/env python3
"""
Compute Joint Index (JI) for the local T8 pipeline.

JI = (sensitivity_mean / FWHM^2) * ASCI_pct / 100

Local file conventions:
  - position_{layout:03d}_ppdfs_t8_{pose:02d}.hdf5
  - beams_properties_configuration_{layout:03d}.hdf5
  - asci_histogram_{layout:03d}.hdf5

This script aggregates all matching files in a work directory and writes one
CSV row with the summary metrics.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

DEFAULT_IMG_NX = 200
DEFAULT_IMG_NY = 200
DEFAULT_N_BINS = 360
EPS = 1e-12


def glob_pattern_list(pattern_text: str | None) -> list[str]:
    """Resolve one glob or a comma-separated list of globs/exact paths."""
    if not pattern_text:
        return []
    matches: list[str] = []
    for pattern in pattern_text.split(","):
        item = pattern.strip()
        if not item:
            continue
        matches.extend(glob.glob(item))
    return sorted(dict.fromkeys(matches))


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _resolve_col_index(header: Sequence[str], candidates: Iterable[str]) -> int | None:
    header_norm = [_norm(col) for col in header]
    for candidate in candidates:
        cand_norm = _norm(candidate)
        if cand_norm in header_norm:
            return header_norm.index(cand_norm)
    return None


def compute_sensitivity(work_dir: str, ppdf_pattern: str | None = None):
    """
    Sum all matching PPDF files and return:
      (sensitivity_total, sensitivity_mean_per_file, n_files, matching_files)
    """
    if ppdf_pattern:
        ppdf_files = glob_pattern_list(ppdf_pattern)
    else:
        patterns = [
            os.path.join(work_dir, "position_*_ppdfs_t8_*.hdf5"),
            os.path.join(work_dir, "position_*_ppdfs.hdf5"),
            os.path.join(work_dir, "scanner_layouts_*_layout_*_subvoxels.hdf5"),
        ]

        ppdf_files: list[str] = []
        for pattern in patterns:
            ppdf_files = sorted(glob.glob(pattern))
            if ppdf_files:
                break

    aggregated_ppdfs = None
    successful = 0

    for ppdf_file in ppdf_files:
        try:
            with h5py.File(ppdf_file, "r") as f:
                ppdfs = f["ppdfs"][:].astype(np.float64, copy=False)
        except Exception as exc:
            print(f"[WARN] Failed to read PPDF file {ppdf_file}: {exc}")
            continue

        if aggregated_ppdfs is None:
            aggregated_ppdfs = ppdfs
        else:
            aggregated_ppdfs += ppdfs
        successful += 1

    if aggregated_ppdfs is None or successful == 0:
        return np.nan, np.nan, 0, ppdf_files

    per_pixel_sum = np.sum(aggregated_ppdfs, axis=0)
    sensitivity_total = float(np.mean(per_pixel_sum))
    sensitivity_mean = sensitivity_total / successful
    return sensitivity_total, sensitivity_mean, successful, ppdf_files


def compute_fwhm_and_asci(
    work_dir: str,
    *,
    img_nx: int = DEFAULT_IMG_NX,
    img_ny: int = DEFAULT_IMG_NY,
    n_bins: int = DEFAULT_N_BINS,
    prop_pattern: str | None = None,
    asci_pattern: str | None = None,
):
    """
    Aggregate FWHM and ASCI from beam-analysis outputs.
    Returns:
      (fwhm_mean, asci_pct, n_prop_files, n_asci_files)
    """
    if prop_pattern:
        prop_files = glob_pattern_list(prop_pattern)
    else:
        aggregate_prop_files = sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_[0-9][0-9][0-9].hdf5")))
        prop_files = aggregate_prop_files or sorted(glob.glob(os.path.join(work_dir, "beams_properties_configuration_*.hdf5")))

    if asci_pattern:
        asci_files = glob_pattern_list(asci_pattern)
    else:
        aggregate_asci_files = sorted(glob.glob(os.path.join(work_dir, "asci_histogram_[0-9][0-9][0-9].hdf5")))
        asci_files = aggregate_asci_files or sorted(glob.glob(os.path.join(work_dir, "asci_histogram_*.hdf5")))

    all_fwhm_values: list[float] = []
    combined_asci_hist = None
    inferred_bins = None

    for prop_file in prop_files:
        try:
            with h5py.File(prop_file, "r") as f:
                data = f["beam_properties"][:]
                header = [
                    h.decode("utf-8") if isinstance(h, (bytes, bytearray)) else str(h)
                    for h in f["beam_properties"].attrs["Header"]
                ]
        except Exception as exc:
            print(f"[WARN] Failed to read beam properties file {prop_file}: {exc}")
            continue

        if data.size == 0:
            continue

        fwhm_idx = _resolve_col_index(header, ["FWHM (mm)", "fwhm (mm)", "FWHM"])
        if fwhm_idx is None:
            fwhm_idx = 4  # layout is stable in this codebase

        fwhm_data = np.asarray(data[:, fwhm_idx], dtype=np.float64)
        valid = np.isfinite(fwhm_data)
        if np.any(valid):
            all_fwhm_values.extend(fwhm_data[valid].tolist())

    for asci_file in asci_files:
        try:
            with h5py.File(asci_file, "r") as f:
                hist = f["asci_histogram"][:]
        except Exception as exc:
            print(f"[WARN] Failed to read ASCI file {asci_file}: {exc}")
            continue

        hist = hist.astype(np.int64, copy=False)
        if combined_asci_hist is None:
            combined_asci_hist = hist
            if hist.ndim == 2:
                inferred_bins = (int(hist.shape[0]), int(hist.shape[1]))
        else:
            combined_asci_hist += hist

    fwhm_mean = float(np.mean(all_fwhm_values)) if all_fwhm_values else np.nan

    if combined_asci_hist is not None:
        n_pix, n_ang = inferred_bins if inferred_bins is not None else (img_nx * img_ny, n_bins)
        total_asci_bins = float(max(n_pix * n_ang, 1))
        asci_filled = int(np.count_nonzero(combined_asci_hist))
        asci_pct = (asci_filled / total_asci_bins) * 100.0
    else:
        asci_pct = np.nan

    return fwhm_mean, asci_pct, len(prop_files), len(asci_files)


def compute_ji(
    work_dir: str,
    *,
    img_nx: int = DEFAULT_IMG_NX,
    img_ny: int = DEFAULT_IMG_NY,
    n_bins: int = DEFAULT_N_BINS,
    ppdf_pattern: str | None = None,
    prop_pattern: str | None = None,
    asci_pattern: str | None = None,
) -> dict:
    """
    Compute the full metric bundle for a single configuration directory.
    """
    sens_total, sens_mean, n_ppdf_files, _ = compute_sensitivity(work_dir, ppdf_pattern=ppdf_pattern)
    fwhm_mean, asci_pct, n_prop_files, n_asci_files = compute_fwhm_and_asci(
        work_dir,
        img_nx=img_nx,
        img_ny=img_ny,
        n_bins=n_bins,
        prop_pattern=prop_pattern,
        asci_pattern=asci_pattern,
    )

    ji = np.nan
    if (
        not np.isnan(fwhm_mean)
        and fwhm_mean > EPS
        and not np.isnan(asci_pct)
        and not np.isnan(sens_mean)
    ):
        ji = (sens_mean / (fwhm_mean ** 2)) * asci_pct / 100.0

    return {
        "fwhm_mean": fwhm_mean,
        "sensitivity_total": sens_total,
        "sensitivity_mean": sens_mean,
        "asci_pct": asci_pct,
        "n_ppdf_files": n_ppdf_files,
        "n_prop_files": n_prop_files,
        "n_asci_files": n_asci_files,
        "JI": ji,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute JI for the local SC-SPECT pipeline")
    parser.add_argument("--work_dir", type=str, required=True,
                        help="Directory containing PPDF HDF5 + beam-analysis outputs")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Path to output CSV (appends if it already exists)")
    parser.add_argument("--config_name", type=str, default="config",
                        help="Configuration label to store in the CSV row")
    parser.add_argument("--img-nx", type=int, default=DEFAULT_IMG_NX)
    parser.add_argument("--img-ny", type=int, default=DEFAULT_IMG_NY)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--ppdf-pattern", type=str, default=None,
                        help="Optional glob for PPDF files")
    parser.add_argument("--prop-pattern", type=str, default=None,
                        help="Optional glob for beam property files")
    parser.add_argument("--asci-pattern", type=str, default=None,
                        help="Optional glob for ASCI files")
    parser.add_argument("--force-zero", action="store_true",
                        help="Write a zero-JI row for infeasible/failed configs")
    parser.add_argument("--reason", type=str, default="", help="Reason for force-zero row")
    args = parser.parse_args()

    if args.force_zero:
        results = {
            "fwhm_mean": np.nan,
            "sensitivity_total": np.nan,
            "sensitivity_mean": np.nan,
            "asci_pct": np.nan,
            "n_ppdf_files": 0,
            "n_prop_files": 0,
            "n_asci_files": 0,
            "JI": 0.0,
        }
        print(f"[{args.config_name}] FORCE_ZERO: {args.reason}")
    else:
        results = compute_ji(
            args.work_dir,
            img_nx=args.img_nx,
            img_ny=args.img_ny,
            n_bins=args.n_bins,
            ppdf_pattern=args.ppdf_pattern,
            prop_pattern=args.prop_pattern,
            asci_pattern=args.asci_pattern,
        )

    results["config"] = args.config_name
    results["work_dir"] = os.path.abspath(args.work_dir)

    df_new = pd.DataFrame([results])
    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(args.out_csv):
        df_new.to_csv(args.out_csv, mode="a", header=False, index=False)
    else:
        df_new.to_csv(args.out_csv, index=False)

    print(
        f"[{args.config_name}] "
        f"FWHM={results['fwhm_mean']:.4f}  "
        f"ASCI={results['asci_pct']:.2f}%  "
        f"Sens={results['sensitivity_mean']:.4e}  "
        f"JI={results['JI']:.6e}  "
        f"({results['n_ppdf_files']} PPDF files, {results['n_prop_files']} props, {results['n_asci_files']} asci)"
    )


if __name__ == "__main__":
    main()
