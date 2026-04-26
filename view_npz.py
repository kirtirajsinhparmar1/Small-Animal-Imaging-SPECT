import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

here = os.path.dirname(os.path.abspath(__file__))
default_data_dir = os.path.abspath(os.path.join(here, "data"))
default_plots_dir = os.path.abspath(os.path.join(here, "plots"))

ap = argparse.ArgumentParser(description="Save PNG snapshots (and optional GIF) from an MLEM .npz.")
ap.add_argument("--npz", default=os.path.join(default_data_dir, "recon_mlem_torch_derenzo_filtered.npz"), help="Input .npz path")
ap.add_argument("--out-dir", default=default_plots_dir, help="Output directory for images")
ap.add_argument("--mm-per-px", type=float, default=0.05, help="mm per pixel (default: 0.05)")
ap.add_argument("--save-every", type=int, default=5, help="Saving frequency used during recon (default: 5)")
ap.add_argument("--iters", default="50,100,125,145", help="Comma-separated iterations to export (default: 50,100,125,145)")
ap.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
args = ap.parse_args()

output_dir = os.path.abspath(args.out_dir)
os.makedirs(output_dir, exist_ok=True)
print(f"Will save output images to: {output_dir}")

# --- 2. Load data ---
data = np.load(args.npz)
print(f"Keys found in the .npz file: {data.files}")
reconstructions = data["estimates"]
print(f"Shape of reconstructions array: {reconstructions.shape}")

# --- 3. Plot settings to match phantom orientation ---
IMG_DIM = reconstructions.shape[1]          # 200
SAVE_EVERY = args.save_every                 # must match your MLEM saving frequency
MM_PER_PX = args.mm_per_px                   # 200px -> 10mm (your title)
width_mm = IMG_DIM * MM_PER_PX
extent = (-width_mm/2, width_mm/2, -width_mm/2, width_mm/2)

def to_phantom_view(img_2d: np.ndarray) -> np.ndarray:
    """
    Make display consistent with your phantom plots:
    - transpose to swap row/col -> x/y
    - origin lower in imshow (below)
    If left-right still looks mirrored, change to:
        return np.fliplr(img_2d.T)
    """
    return img_2d.T

def save_image(img_2d: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(8, 8))
    plt.imshow(to_phantom_view(img_2d), cmap="gray_r", origin="lower", extent=extent, interpolation="nearest")
    plt.colorbar(label="Image Intensity")
    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

# --- 4. Save snapshots ---
target_iters = [int(x) for x in args.iters.split(",") if x.strip() != ""]

max_saved_iter = (reconstructions.shape[0] - 1) * SAVE_EVERY
print(f"Max saved iteration available ~ {max_saved_iter}")

for it in target_iters:
    frame_idx = it // SAVE_EVERY
    if frame_idx < 0 or frame_idx >= reconstructions.shape[0]:
        print(f"[WARN] Iter {it} not available (frame {frame_idx} out of range). Skipping.")
        continue

    img = reconstructions[frame_idx]
    out_path = os.path.join(output_dir, f"recon_iter_{it:03d}.png")
    save_image(img, f"Reconstruction (~Iter {it})", out_path)

# Also save the *last available* reconstruction
final_img = reconstructions[-1]
final_iter = (reconstructions.shape[0] - 1) * SAVE_EVERY
final_path = os.path.join(output_dir, f"final_reconstruction_iter_{final_iter:03d}.png")
save_image(final_img, f"Final Reconstruction (~Iter {final_iter})", final_path)

# --- 5. Optional: GIF with the same orientation ---
print("\nAttempting to create an animation...")
try:
    from matplotlib.animation import FuncAnimation

    fig_anim, ax_anim = plt.subplots(figsize=(7, 7))
    plt.close(fig_anim)

    im = ax_anim.imshow(
        to_phantom_view(reconstructions[0]),
        cmap="gray_r",
        origin="lower",
        extent=extent,
        animated=True,
        interpolation="nearest",
    )
    ax_anim.set_xlabel("x (mm)")
    ax_anim.set_ylabel("y (mm)")
    cb = fig_anim.colorbar(im, ax=ax_anim, label="Image Intensity")

    def update(frame):
        img = to_phantom_view(reconstructions[frame])
        im.set_array(img)
        im.set_clim(vmin=img.min(), vmax=img.max())
        ax_anim.set_title(f"Reconstruction after ~Iteration {frame * SAVE_EVERY}")
        return im,

    ani = FuncAnimation(fig_anim, update, frames=len(reconstructions), interval=100, blit=True)

    if args.no_gif:
        print("GIF skipped (--no-gif).")
    else:
        gif_path = os.path.join(output_dir, "reconstruction_progress.gif")
        ani.save(gif_path, writer="pillow", fps=10)
        print(f"Successfully saved animation to: {gif_path}")

except Exception as e:
    print(f"Animation skipped/failed: {e}")
