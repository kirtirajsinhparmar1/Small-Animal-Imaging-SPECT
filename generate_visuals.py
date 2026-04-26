#!/usr/bin/env python3
"""
Unified visualizer for the local SC-SPECT pipeline.

This combines the scattered visual scripts into one entrypoint:
  - geometry overview from scanner layout tensor
  - aggregated sensitivity map + histogram from PPDFs
  - ASCI map from asci_histogram_*.hdf5
  - cumulative beam-width histogram
  - beam multiplexing plots per layout
  - reconstruction snapshots / GIF from .npz
  - optional BO-style convergence plots if results_summary.csv exists

Defaults to the local repo paths under ./data and ./plots.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection
from matplotlib.ticker import PercentFormatter
from scanner_modeling._geometry_2d._io import load_scanner_geometry_from_layout, load_scanner_layouts


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(HERE, "data")
DEFAULT_PLOT_DIR = os.path.join(HERE, "plots")

def _norm(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _resolve_col_index(header: Sequence[str], candidates: Iterable[str]) -> int | None:
    header_norm = [_norm(col) for col in header]
    for candidate in candidates:
        cand_norm = _norm(candidate)
        if cand_norm in header_norm:
            return header_norm.index(cand_norm)
    return None


def _polycollection_from_vertices(vertices: torch.Tensor, **kwargs):
    if vertices is None or vertices.numel() == 0:
        return None
    return PolyCollection(vertices.cpu().tolist(), **kwargs)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pick_layout_tensor(data_dir: str, explicit: str | None = None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = sorted(glob.glob(os.path.join(data_dir, "scanner_layouts_*.tensor")))
    if not candidates:
        raise FileNotFoundError(f"No scanner_layouts_*.tensor found in {data_dir}")
    return candidates[-1]


def plot_geometry_overview(layout_file: str, layout_idx: int, plot_dir: str) -> str:
    layouts_data, _ = load_scanner_layouts(os.path.dirname(layout_file), os.path.basename(layout_file))
    plate_vertices, det_vertices, _, _ = load_scanner_geometry_from_layout(layout_idx, layouts_data)

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    pc_plate = _polycollection_from_vertices(
        plate_vertices, facecolor="none", edgecolor="tab:red", linewidth=0.7, alpha=0.8
    )
    pc_det = _polycollection_from_vertices(
        det_vertices, facecolor="none", edgecolor="tab:blue", linewidth=0.4, alpha=0.7
    )
    if pc_plate is not None:
        ax.add_collection(pc_plate)
    if pc_det is not None:
        ax.add_collection(pc_det)
    ax.set_aspect("equal")
    ax.set_title(f"Scanner geometry overview (layout {layout_idx:03d})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    out = os.path.join(plot_dir, f"geometry_overview_layout_{layout_idx:03d}.png")
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def compute_ppdf_sensitivity_map(data_dir: str, layout_idxs: Sequence[int], output_name: str = "sensitivity_map_aggregated.png"):
    all_ppdfs = []
    loaded = 0
    for idx in layout_idxs:
        for ppdf_path in sorted(glob.glob(os.path.join(data_dir, f"position_{idx:03d}_ppdfs_t8_*.hdf5"))):
            try:
                with h5py.File(ppdf_path, "r") as f:
                    all_ppdfs.append(f["ppdfs"][:].astype(np.float64, copy=False))
                loaded += 1
            except Exception as exc:
                print(f"[WARN] failed to read {ppdf_path}: {exc}")

    if not all_ppdfs:
        return None

    aggregated = np.sum(np.stack(all_ppdfs, axis=0), axis=0)
    sensitivity_1d = np.sum(aggregated, axis=0) / max(loaded, 1)
    side = int(np.sqrt(sensitivity_1d.shape[0]))
    if side * side != sensitivity_1d.shape[0]:
        raise RuntimeError(f"PPDF pixel count {sensitivity_1d.shape[0]} is not a square")

    image = sensitivity_1d.reshape(side, side)
    extent = (-side / 20.0, side / 20.0, -side / 20.0, side / 20.0)

    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    im = ax.imshow(image.T, cmap="viridis", origin="lower", extent=extent)
    fig.colorbar(im, ax=ax, label="Average sensitivity")
    ax.set_title(f"Aggregated sensitivity map ({loaded} PPDF files)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    return fig, ax, output_name


def save_sensitivity_visuals(data_dir: str, plot_dir: str, layout_idxs: Sequence[int]) -> list[str]:
    out_paths: list[str] = []

    # Per-layout PPDS-like visuals
    for idx in layout_idxs:
        ppdfs = []
        for ppdf_path in sorted(glob.glob(os.path.join(data_dir, f"position_{idx:03d}_ppdfs_t8_*.hdf5"))):
            with h5py.File(ppdf_path, "r") as f:
                ppdfs.append(f["ppdfs"][:].astype(np.float64, copy=False))
        if not ppdfs:
            continue

        aggregated = np.sum(np.stack(ppdfs, axis=0), axis=0)
        ppds = np.sum(aggregated, axis=0)
        n_pix = ppds.shape[0]
        side = int(np.sqrt(n_pix))
        if side * side != n_pix:
            continue
        sens_2d = ppds.reshape(side, side)
        sens_max = float(np.max(sens_2d))
        eps = 1e-12
        sens_rel = sens_2d / (sens_2d.max() + eps)
        sens_norm = sens_2d / (sens_max + eps)

        extent = (-side / 20.0, side / 20.0, -side / 20.0, side / 20.0)
        for arr, suffix, title in [
            (sens_2d, "map", "PPDF sensitivity map"),
            (sens_norm, "maxnorm", "PPDF sensitivity (max-normalized)"),
            (sens_rel, "relative", "PPDF sensitivity (relative)"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
            im = ax.imshow(arr.T, cmap="viridis", origin="lower", extent=extent)
            fig.colorbar(im, ax=ax)
            ax.set_title(f"{title} | layout {idx:03d}")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            out = os.path.join(plot_dir, f"ppds_{suffix}_layout_{idx:03d}.png")
            fig.savefig(out, dpi=250)
            plt.close(fig)
            out_paths.append(out)

    # Aggregated across layouts
    agg = None
    loaded = 0
    for idx in layout_idxs:
        for ppdf_path in sorted(glob.glob(os.path.join(data_dir, f"position_{idx:03d}_ppdfs_t8_*.hdf5"))):
            with h5py.File(ppdf_path, "r") as f:
                ppdfs = f["ppdfs"][:].astype(np.float64, copy=False)
            if agg is None:
                agg = ppdfs
            else:
                agg += ppdfs
            loaded += 1

    if agg is not None:
        sens_1d = np.sum(agg, axis=0) / max(loaded, 1)
        side = int(np.sqrt(sens_1d.shape[0]))
        if side * side == sens_1d.shape[0]:
            sens_2d = sens_1d.reshape(side, side)
            extent = (-side / 20.0, side / 20.0, -side / 20.0, side / 20.0)
            fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
            im = ax.imshow(sens_2d.T, cmap="viridis", origin="lower", extent=extent)
            fig.colorbar(im, ax=ax, label="Average sensitivity")
            ax.set_title(f"Aggregated sensitivity map ({loaded} PPDF files)")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            out = os.path.join(plot_dir, "sensitivity_map_aggregated.png")
            fig.savefig(out, dpi=250)
            plt.close(fig)
            out_paths.append(out)

            flat = sens_2d[sens_2d > 0].ravel()
            fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
            ax.hist(flat, bins=100, color="royalblue", alpha=0.8)
            ax.axvline(float(flat.mean()), color="red", ls="--", lw=1.5)
            ax.set_title("Aggregated sensitivity histogram")
            ax.set_xlabel("Sensitivity")
            ax.set_ylabel("Pixel count")
            out = os.path.join(plot_dir, "sensitivity_histogram_aggregated.png")
            fig.savefig(out, dpi=250)
            plt.close(fig)
            out_paths.append(out)

    return out_paths


def save_asci_visuals(data_dir: str, plot_dir: str, layout_idxs: Sequence[int], n_bins: int = 360) -> list[str]:
    out_paths: list[str] = []
    asci_sum = None
    loaded = 0
    for idx in layout_idxs:
        candidates = [
            os.path.join(data_dir, f"asci_histogram_{idx:03d}.hdf5"),
            os.path.join(data_dir, f"asci_histogram_{idx:02d}_t8_agg.hdf5"),
            os.path.join(data_dir, f"asci_histogram_{idx:02d}.hdf5"),
        ]
        asci_path = next((path for path in candidates if os.path.exists(path)), None)
        if asci_path is None:
            continue
        with h5py.File(asci_path, "r") as f:
            hist = f["asci_histogram"][:].astype(np.int64, copy=False)
        if asci_sum is None:
            asci_sum = hist
        else:
            asci_sum += hist
        loaded += 1

    if asci_sum is None:
        return out_paths

    asci_map = torch.count_nonzero(torch.from_numpy(asci_sum), dim=1).numpy() / float(n_bins)
    side = int(np.sqrt(asci_map.shape[0]))
    if side * side == asci_map.shape[0]:
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        im = ax.imshow(
            asci_map.reshape(side, side).T,
            extent=(-side / 20.0, side / 20.0, -side / 20.0, side / 20.0),
            origin="lower",
            cmap="viridis",
        )
        cbar = fig.colorbar(im, ax=ax, label="ASCI fraction of angle bins")
        cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
        cbar.update_ticks()
        ax.set_title(f"ASCI map | aggregated {loaded} file(s)")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        out = os.path.join(plot_dir, "asci_map_t8_agg.png")
        fig.savefig(out, dpi=250)
        plt.close(fig)
        out_paths.append(out)
    return out_paths


def save_beam_multiplexing_visuals(data_dir: str, plot_dir: str, layout_idxs: Sequence[int]) -> list[str]:
    out_paths: list[str] = []
    for idx in layout_idxs:
        aggregate_prop = os.path.join(data_dir, f"beams_properties_configuration_{idx:03d}.hdf5")
        aggregate_mask = os.path.join(data_dir, f"beams_masks_configuration_{idx:03d}.hdf5")
        if os.path.exists(aggregate_prop) and os.path.exists(aggregate_mask):
            prop_paths = [aggregate_prop]
            mask_paths = [aggregate_mask]
        else:
            prop_paths = sorted(glob.glob(os.path.join(data_dir, f"beams_properties_configuration_{idx:02d}_t8_*.hdf5")))
            mask_paths = sorted(glob.glob(os.path.join(data_dir, f"beams_masks_configuration_{idx:02d}_t8_*.hdf5")))
        if not prop_paths or not mask_paths:
            continue

        all_fwhm = []
        det_counts = []
        for props_path in prop_paths:
            with h5py.File(props_path, "r") as f:
                header = [h.decode("utf-8") if isinstance(h, (bytes, bytearray)) else str(h) for h in f["beam_properties"].attrs["Header"]]
                data = f["beam_properties"][:]
            fwhm_idx = _resolve_col_index(header, ["FWHM (mm)", "fwhm (mm)", "FWHM"]) or 4
            det_idx = _resolve_col_index(header, ["detector unit id", "detector unit"]) or 1
            fwhm = np.asarray(data[:, fwhm_idx], dtype=np.float64)
            det = np.asarray(data[:, det_idx], dtype=np.int64)
            valid = np.isfinite(fwhm)
            all_fwhm.append(fwhm[valid])
            good = (fwhm[valid] >= 2.0) & (fwhm[valid] <= 5.0)
            det_counts.append(det[valid][good])

        flat_fwhm = np.concatenate(all_fwhm) if all_fwhm else np.array([])
        if flat_fwhm.size:
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            plot_vals = flat_fwhm[(flat_fwhm >= 0.0) & (flat_fwhm <= 10.0)]
            ax.hist(plot_vals, bins=30, range=(0.0, 10.0), color="#4c72b0", alpha=0.85)
            ax.axvline(float(plot_vals.mean()), color="red", ls="--", lw=1.5)
            ax.axvline(float(np.median(plot_vals)), color="green", ls=":", lw=1.5)
            ax.set_xlim(0.0, 10.0)
            ax.set_title(f"Beam width histogram (T8 aggregated, layout {idx:02d})")
            ax.set_xlabel("Beam FWHM (mm)")
            ax.set_ylabel("Count")
            out = os.path.join(plot_dir, f"beam_width_histogram_t8_0to10mm_layout_{idx:02d}.png")
            fig.savefig(out, dpi=250)
            plt.close(fig)
            out_paths.append(out)

        # Detector beam counts summary
        counts_per_detector = []
        for mask_path in mask_paths:
            with h5py.File(mask_path, "r") as f:
                masks = torch.from_numpy(f["beam_mask"][:])
            counts_per_detector.append(torch.tensor([(row.unique().numel() - 1) for row in masks], dtype=torch.int64))
        if counts_per_detector:
            counts_agg = torch.stack(counts_per_detector, dim=0).max(dim=0).values
            if counts_agg.numel() > 0 and counts_agg.max() > 0:
                max_k = int(counts_agg.max().item())
                det_per_k = torch.bincount(counts_agg, minlength=max_k + 1)
                ks = np.arange(0, max_k + 1)
                fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
                ax.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)
                ax.set_xlabel("Number of beams per detector")
                ax.set_ylabel("Number of detectors")
                ax.set_title(f"Beam multiplicity (T8 aggregated, layout {idx:02d})")
                out = os.path.join(plot_dir, f"detector_beam_counts_t8_layout_{idx:02d}.png")
                fig.savefig(out, dpi=250)
                plt.close(fig)
                out_paths.append(out)

    return out_paths


def save_cumulative_beam_widths(data_dir: str, plot_dir: str) -> str | None:
    aggregate_prop_paths = sorted(glob.glob(os.path.join(data_dir, "beams_properties_configuration_[0-9][0-9][0-9].hdf5")))
    prop_paths = aggregate_prop_paths or sorted(glob.glob(os.path.join(data_dir, "beams_properties_configuration_*.hdf5")))
    all_fwhm = []
    for props_path in prop_paths:
        with h5py.File(props_path, "r") as f:
            header = [h.decode("utf-8") if isinstance(h, (bytes, bytearray)) else str(h) for h in f["beam_properties"].attrs["Header"]]
            data = f["beam_properties"][:]
        fwhm_idx = _resolve_col_index(header, ["FWHM (mm)", "fwhm (mm)", "FWHM"]) or 4
        fwhm = np.asarray(data[:, fwhm_idx], dtype=np.float64)
        all_fwhm.extend(fwhm[np.isfinite(fwhm)].tolist())

    if not all_fwhm:
        return None
    vals = np.asarray(all_fwhm)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    plot_vals = vals[(vals >= 0.0) & (vals <= 2.0)]
    ax.hist(plot_vals, bins=200, range=(0.0, 2.0), color="#4c72b0", alpha=0.9)
    ax.axvline(float(plot_vals.mean()), color="#d62728", ls="--", lw=1.5)
    ax.axvline(float(np.median(plot_vals)), color="#2ca02c", ls=":", lw=1.5)
    ax.set_title("Cumulative beam width distribution")
    ax.set_xlabel("Beam FWHM (mm)")
    ax.set_ylabel("Number of beams")
    out = os.path.join(plot_dir, "cumulative_beam_width_histogram.png")
    fig.savefig(out, dpi=250)
    plt.close(fig)
    return out


def save_recon_visuals(data_dir: str, plot_dir: str, npz_path: str | None, save_every: int = 5, no_gif: bool = False) -> list[str]:
    out_paths: list[str] = []
    if npz_path is None:
        candidates = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not candidates:
            return out_paths
        npz_path = candidates[-1]
    if not os.path.exists(npz_path):
        return out_paths

    data = np.load(npz_path, allow_pickle=True)
    if "estimates" not in data:
        return out_paths
    reconstructions = data["estimates"]
    if reconstructions.ndim != 3:
        return out_paths

    img_dim = reconstructions.shape[1]
    def to_phantom_view(arr):
        return np.flipud(arr.T) if arr.shape == (img_dim, img_dim) else arr

    for it in range(0, reconstructions.shape[0], save_every):
        frame = min(it // save_every, reconstructions.shape[0] - 1)
        img = to_phantom_view(reconstructions[frame])
        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Reconstruction iteration {it}")
        out = os.path.join(plot_dir, f"recon_iter_{it:03d}.png")
        fig.savefig(out, dpi=250)
        plt.close(fig)
        out_paths.append(out)

    final_img = to_phantom_view(reconstructions[-1])
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    ax.imshow(final_img, cmap="gray")
    ax.set_title("Final reconstruction")
    out = os.path.join(plot_dir, "final_reconstruction.png")
    fig.savefig(out, dpi=250)
    plt.close(fig)
    out_paths.append(out)

    if not no_gif:
        fig_anim, ax_anim = plt.subplots(figsize=(5, 5), layout="constrained")
        im = ax_anim.imshow(to_phantom_view(reconstructions[0]), cmap="gray", animated=True)
        ax_anim.axis("off")

        def update(frame):
            im.set_array(to_phantom_view(reconstructions[frame]))
            return [im]

        ani = FuncAnimation(fig_anim, update, frames=len(reconstructions), interval=100, blit=True)
        gif_path = os.path.join(plot_dir, "reconstruction_progress.gif")
        ani.save(gif_path, writer="pillow", fps=10)
        plt.close(fig_anim)
        out_paths.append(gif_path)

    return out_paths


def plot_bo_convergence(results_csv: str, output_dir: str) -> list[str]:
    if not os.path.exists(results_csv):
        return []
    df = pd.read_csv(results_csv)
    if "JI" not in df.columns or df.empty:
        return []
    df = df.dropna(subset=["JI"])
    if df.empty:
        return []

    out_paths: list[str] = []
    ji_vals = df["JI"].values
    best_so_far = np.maximum.accumulate(np.where(ji_vals > 0, ji_vals, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(len(best_so_far)), best_so_far, "b-o", markersize=3, linewidth=1.5)
    ax1.set_xlabel("Evaluation Number")
    ax1.set_ylabel("Best JI")
    ax1.set_title("Best JI vs Iteration")
    ax1.grid(True, alpha=0.3)
    ax2.scatter(range(len(ji_vals)), ji_vals, c=["green" if ji > 0 else "red" for ji in ji_vals], s=20, alpha=0.7)
    ax2.set_xlabel("Evaluation Number")
    ax2.set_ylabel("JI")
    ax2.set_title("JI per Evaluation")
    ax2.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out_paths.append(path)

    if all(c in df.columns for c in ["aperture_diam_mm", "n_apertures"]):
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        sc = ax.scatter(df["aperture_diam_mm"], df["n_apertures"], c=df["JI"], cmap="viridis", s=40, edgecolors="black")
        fig.colorbar(sc, ax=ax, label="JI")
        ax.set_xlabel("aperture_diam_mm")
        ax.set_ylabel("n_apertures")
        ax.set_title("Design space exploration")
        path = os.path.join(output_dir, "design_space.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)

    return out_paths


def main():
    ap = argparse.ArgumentParser(description="Generate the unified visual set for the local pipeline")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    ap.add_argument("--layout-file", default=None, help="Optional scanner_layouts_*.tensor path")
    ap.add_argument("--layout-idxs", default="0,1", help="Comma-separated layout indices for aggregate visuals")
    ap.add_argument("--npz", default=None, help="Optional reconstruction .npz path")
    ap.add_argument("--results-csv", default=None, help="Optional results CSV for BO-style plots")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.plot_dir)
    layout_idxs = [int(x) for x in args.layout_idxs.split(",") if x.strip()]
    layout_file = pick_layout_tensor(args.data_dir, args.layout_file)

    generated: list[str] = []
    try:
        generated.append(plot_geometry_overview(layout_file, layout_idxs[0], args.plot_dir))
    except Exception as exc:
        print(f"[WARN] geometry overview failed: {exc}")

    generated.extend(save_sensitivity_visuals(args.data_dir, args.plot_dir, layout_idxs))
    generated.extend(save_asci_visuals(args.data_dir, args.plot_dir, layout_idxs))

    beam_paths = save_beam_multiplexing_visuals(args.data_dir, args.plot_dir, layout_idxs)
    generated.extend(beam_paths)
    cum = save_cumulative_beam_widths(args.data_dir, args.plot_dir)
    if cum:
        generated.append(cum)

    generated.extend(save_recon_visuals(args.data_dir, args.plot_dir, args.npz, args.save_every, args.no_gif))
    if args.results_csv:
        generated.extend(plot_bo_convergence(args.results_csv, args.plot_dir))

    print("[DONE] Generated:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
