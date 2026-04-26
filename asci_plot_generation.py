#!/usr/bin/env python3
import os
import h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR  = os.path.join(HERE, "data")
PLOT_DIR   = os.path.join(HERE, "plots")
LAYOUT_SEQ = range(2)           # change to range(24) when ready
N_BINS     = 360
FOV_SIDE   = 10                 # mm (=> extent is ±5 mm)
IMG_NX, IMG_NY = 200, 200
# -------------------------------------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

asci_hist = torch.zeros(IMG_NX * IMG_NY, N_BINS, dtype=torch.int32)

successful = 0
for idx in LAYOUT_SEQ:
    h5_path = os.path.join(INPUT_DIR, f"asci_histogram_{idx:02d}_t8_agg.hdf5")

    if not os.path.exists(h5_path):
        print(f"[WARN] Missing: {h5_path} (skipping)")
        continue

    with h5py.File(h5_path, "r") as f:
        asci_hist += torch.from_numpy(f["asci_histogram"][...])
    successful += 1

if successful == 0:
    raise RuntimeError("No T8 aggregated ASCI histogram files were found. Nothing to plot.")

# fraction of angle bins that are nonzero per pixel (0..1)
asci_map = torch.count_nonzero(asci_hist, dim=1) / float(N_BINS)

# ---- plot -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")

im = ax.imshow(
    asci_map.view(IMG_NX, IMG_NY).T,
    extent=(-FOV_SIDE/2, FOV_SIDE/2, -FOV_SIDE/2, FOV_SIDE/2),
    origin="lower",
    cmap="viridis",
)

cbar = fig.colorbar(im, ax=ax, label="ASCI (fraction of angle bins)")
cbar.formatter = PercentFormatter(xmax=1.0, decimals=1)
cbar.update_ticks()

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_title(
    f"T8-aggregated ASCI map | layouts loaded={successful} | "
    f"max={asci_map.max():.2%}, min={asci_map.min():.2%}"
)

out_path = os.path.join(PLOT_DIR, "asci_map_t8_agg.png")
fig.savefig(out_path, dpi=300)
plt.close(fig)

print(f"[DONE] Saved: {out_path}")
