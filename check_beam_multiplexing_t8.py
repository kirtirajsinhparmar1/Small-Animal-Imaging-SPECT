#!/usr/bin/env python3
"""
check_beam_multiplexing_t8.py
--------------------------------
Same outputs as the original script, but aggregated over 8 T8 translations.

Command:
  python check_beam_multiplexing_t8.py \
    --layout 0 \
    --data-dir ./data \
    --out ./plots \
    --bins 200

Outputs (per layout):
  1) beam_width_histogram_t8_0to10mm_layout_00.png
  2) detector_beam_counts_t8_layout_00.png

Expected files (layout=00 example):
  beams_properties_configuration_00_t8_00.hdf5 ... beams_properties_configuration_00_t8_07.hdf5
  beams_masks_configuration_00_t8_00.hdf5      ... beams_masks_configuration_00_t8_07.hdf5
"""
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch

T8_TAGS = [f"t8_{i:02d}" for i in range(8)]


def read_columns(h5_path: str, desired: list[str]) -> dict[str, torch.Tensor]:
    """Return selected columns from beam_properties HDF5 as tensors."""
    with h5py.File(h5_path, "r") as f:
        header = [
            h.decode("utf-8") if isinstance(h, (bytes, bytearray)) else str(h)
            for h in f["beam_properties"].attrs["Header"]
        ]
        data = torch.from_numpy(f["beam_properties"][:])

    cols: dict[str, torch.Tensor] = {}
    for name in desired:
        if name not in header:
            raise RuntimeError(f"Column '{name}' not found in {h5_path}. Available: {header}")
        cols[name] = data[:, header.index(name)]
    return cols


def load_fwhm_and_detid(props_path: str) -> tuple[np.ndarray, torch.Tensor]:
    """Load valid FWHM and aligned detector unit IDs."""
    cols = read_columns(props_path, ["FWHM (mm)", "detector unit id"])
    fwhm_raw = cols["FWHM (mm)"].numpy()
    det_id_raw = cols["detector unit id"].to(torch.int64)

    valid = ~np.isnan(fwhm_raw)
    return fwhm_raw[valid], det_id_raw[valid]


def load_masks(masks_path: str) -> torch.Tensor:
    """Load beam_mask (n_det, n_pix)."""
    with h5py.File(masks_path, "r") as f:
        return torch.from_numpy(f["beam_mask"][:])


def beams_per_detector(masks: torch.Tensor) -> torch.Tensor:
    """
    Count beams per detector by unique non-zero IDs in each row.
    Returns (n_det,) tensor.
    """
    return torch.tensor([(row.unique().numel() - 1) for row in masks], dtype=torch.int64)


def aggregate_counts_over_t8(counts_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Aggregate beams-per-detector across T8 poses.

    Using MAX across poses keeps the meaning closest to single-file case:
      "How many beams does a detector have (worst case over T8)?"
    """
    stacked = torch.stack(counts_list, dim=0)  # (8, n_det)
    return stacked.max(dim=0).values


def main(args):
    os.makedirs(args.out, exist_ok=True)

    # ---- build expected filenames ----
    props_paths = [
        os.path.join(args.data_dir, f"beams_properties_configuration_{args.layout:02d}_{tag}.hdf5")
        for tag in T8_TAGS
    ]
    masks_paths = [
        os.path.join(args.data_dir, f"beams_masks_configuration_{args.layout:02d}_{tag}.hdf5")
        for tag in T8_TAGS
    ]

    # Validate existence (hard fail, like your T4 script)
    for p in props_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing properties file: {p}")
    for m in masks_paths:
        if not os.path.exists(m):
            raise FileNotFoundError(f"Missing masks file: {m}")

    # ================================================================
    # Figure 1: FWHM histogram aggregated over T8
    # ================================================================
    all_fwhm = []
    for p in props_paths:
        print(f"[LOAD] {p}")
        fwhm_valid, _det_id = load_fwhm_and_detid(p)
        all_fwhm.append(fwhm_valid)

    fwhm = np.concatenate(all_fwhm) if len(all_fwhm) else np.array([], dtype=np.float32)
    print(f"[INFO] Total valid beams across T8: {len(fwhm)}")

    fig1, ax1 = plt.subplots(figsize=(7, 4), layout="constrained")

    # Plot only 0–10 mm for visualization (same as your T4 script)
    fwhm_plot = fwhm[(fwhm >= 0.0) & (fwhm <= 10.0)]

    if len(fwhm_plot) > 0:
        ax1.hist(
            fwhm_plot,
            bins=args.bins,
            range=(0.0, 10.0),
            color="#4c72b0",
            alpha=0.85,
        )
        mean_v, med_v = fwhm_plot.mean(), np.median(fwhm_plot)
        ax1.axvline(mean_v, color="red", ls="--", lw=1.5, label=f"Mean {mean_v:.2f} mm")
        ax1.axvline(med_v,  color="green", ls=":",  lw=1.5, label=f"Median {med_v:.2f} mm")
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, "No FWHM in 0–10 mm", ha="center", va="center")

    ax1.set_xlim(0.0, 10.0)
    ax1.set_xlabel("Beam FWHM (mm)")
    ax1.set_ylabel("Number of beams")
    ax1.set_title(f"Distribution of Beam Widths (0–10 mm view, T8 aggregated, layout {args.layout:02d})")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    out1 = os.path.join(args.out, f"beam_width_histogram_t8_0to10mm_layout_{args.layout:02d}.png")
    fig1.savefig(out1, dpi=300)
    plt.close(fig1)
    print(f"[SAVE] {out1}")

    # ================================================================
    # Figure 2: beams-per-detector bar chart aggregated over T8 (MAX)
    # ================================================================
    counts_list = []
    for m in masks_paths:
        print(f"[LOAD] {m}")
        masks = load_masks(m)
        counts = beams_per_detector(masks)
        counts_list.append(counts)

    counts_agg = aggregate_counts_over_t8(counts_list)  # max across poses

    # Optional: print detectors with ≥1 beam in "good-width" window across all poses
    # (kept for parity with your original header text)
    # NOTE: This is computed using properties, not masks.
    good_det_ids = set()
    for p in props_paths:
        fwhm_valid, det_id_valid = load_fwhm_and_detid(p)
        good = (fwhm_valid >= args.good_lo) & (fwhm_valid <= args.good_hi)
        if np.any(good):
            for d in torch.unique(det_id_valid[good]).tolist():
                good_det_ids.add(int(d))
    print(f"[INFO] Unique detectors with ≥1 beam in {args.good_lo:.1f}–{args.good_hi:.1f} mm window (over all T8 poses): {len(good_det_ids)}")

    if counts_agg.numel() > 0 and counts_agg.max() > 0:
        max_k = int(counts_agg.max().item())
        det_per_k = torch.bincount(counts_agg, minlength=max_k + 1)  # KEEP k=0
        ks = np.arange(0, max_k + 1)

        print("\nDetectors by beam count (k), aggregated across T8 (MAX):")
        for k, c in zip(ks, det_per_k.tolist()):
            print(f"  k={k}: {c} detectors")
        print(f" Total: {det_per_k.sum().item()} detectors")

        fig2, ax2 = plt.subplots(figsize=(6, 4), layout="constrained")
        bars = ax2.bar(ks, det_per_k.numpy(), color="#55a868", alpha=0.9)

        for bar, val in zip(bars, det_per_k.tolist()):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2.set_xlabel("Number of beams per detector (k)")
        ax2.set_ylabel("Number of detectors")
        ax2.set_title(f"Beam Multiplicity Distribution (T8 aggregated, layout {args.layout:02d})")
        ax2.set_xticks(ks if max_k < 10 else np.arange(1, max_k + 1, 2))

        out2 = os.path.join(args.out, f"detector_beam_counts_t8_layout_{args.layout:02d}.png")
        fig2.savefig(out2, dpi=300)
        plt.close(fig2)
        print(f"[SAVE] {out2}")
    else:
        print("No beams found in the masks to generate multiplicity plot.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FWHM + multiplexing plots aggregated over T8 translations.")
    ap.add_argument("--layout", type=int, required=True, help="Layout index (e.g., 0)")
    ap.add_argument("--data-dir", required=True, help="Directory containing *_t8_00..07 hdf5 files")
    ap.add_argument("--out", required=True, help="Output directory for PNGs")

    ap.add_argument("--bins", type=int, default=30, help="Histogram bins for FWHM plot")
    ap.add_argument("--good-lo", type=float, default=2.0, help="Lower bound of good-width window (mm)")
    ap.add_argument("--good-hi", type=float, default=5.0, help="Upper bound of good-width window (mm)")

    args = ap.parse_args()
    main(args)
