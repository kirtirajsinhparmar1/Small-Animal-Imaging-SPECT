#!/usr/bin/env python3
import argparse
import os
import time
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn


# -----------------------------
# Helpers
# -----------------------------
def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = [line.strip() for line in f if line.strip()]
    return flist


def validate_matrix_shapes(flist: list, sproj: int, sfov: int) -> None:
    for fname in flist:
        with h5py.File(fname, "r") as h5f:
            if "ppdfs" not in h5f:
                raise KeyError(f"{fname} does not contain dataset 'ppdfs'")
            current_shape = tuple(h5f["ppdfs"].shape)

        expected_shape = (sproj, sfov)
        if current_shape != expected_shape:
            raise ValueError(
                f"{fname} ppdfs shape {current_shape} does not match expected "
                f"{expected_shape}. Projection/MLEM row counts must match the "
                "current SRM fidelity."
            )


def gaussian_kernel_1d(sigma_px: float, device: torch.device, dtype=torch.float32):
    """
    Create a normalized 1D Gaussian kernel.
    radius = ceil(3*sigma)
    """
    if sigma_px <= 0:
        # no-op kernel
        k = torch.tensor([1.0], device=device, dtype=dtype)
        return k

    radius = int(np.ceil(3.0 * sigma_px))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma_px * sigma_px))
    k = k / torch.sum(k)
    return k


def gaussian_filter_2d(img_2d: torch.Tensor, fwhm_mm: float, mm_per_px: float):
    """
    Separable Gaussian blur implemented with conv2d.
    img_2d: (H, W) tensor
    """
    device = img_2d.device
    dtype = img_2d.dtype

    # FWHM -> sigma
    sigma_mm = fwhm_mm / 2.355
    sigma_px = sigma_mm / mm_per_px

    k1d = gaussian_kernel_1d(sigma_px, device=device, dtype=dtype)
    # conv2d expects (N,C,H,W)
    x = img_2d.unsqueeze(0).unsqueeze(0)

    # horizontal
    kx = k1d.view(1, 1, 1, -1)
    pad_x = (kx.shape[-1] // 2, kx.shape[-1] // 2, 0, 0)  # left,right,top,bottom
    x = F.pad(x, pad_x, mode="replicate")
    x = F.conv2d(x, kx)

    # vertical
    ky = k1d.view(1, 1, -1, 1)
    pad_y = (0, 0, ky.shape[-2] // 2, ky.shape[-2] // 2)
    x = F.pad(x, pad_y, mode="replicate")
    x = F.conv2d(x, ky)

    return x.squeeze(0).squeeze(0)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    # --- Paths ---
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(here, "data"))

    ap = argparse.ArgumentParser(description="Run chunked-matrix MLEM reconstruction + optional Gaussian post-filter.")
    ap.add_argument("--data-dir", default=default_data_dir, help="Directory containing dataset_flist.csv and projections")
    ap.add_argument("--flist", default=None, help="flist path (default: <data-dir>/dataset_flist.csv)")
    ap.add_argument("--projs", default=None, help="projections .npy path (default: <data-dir>/derenzo-projs_T8.npy)")
    ap.add_argument("--out", default=None, help="output .npz path (default: <data-dir>/recon_mlem_torch_derenzo_T8_gauss.npz)")
    ap.add_argument("--iters", type=int, default=150, help="max iterations (default: 150)")
    ap.add_argument("--save-every", type=int, default=5, help="save snapshot every N iterations (default: 5)")
    ap.add_argument("--conv-tol", type=float, default=1e-4, help="convergence tolerance (default: 1e-4)")
    ap.add_argument("--gauss-fwhm-mm", type=float, default=0.16, help="Gaussian FWHM (mm) (default: 0.16)")
    ap.add_argument("--mm-per-px", type=float, default=0.05, help="mm per pixel (default: 0.05)")
    ap.add_argument("--gauss-each-iter", action="store_true", help="apply Gaussian each iteration (not classic post-filter)")
    ap.add_argument("--no-gauss", action="store_true", help="disable Gaussian filtering")
    ap.add_argument("--device", default=None, help="cuda or cpu (default auto)")
    args = ap.parse_args()

    base_dir = os.path.abspath(args.data_dir)
    flist_path = args.flist or os.path.join(base_dir, "dataset_flist.csv")
    projs_path = args.projs or os.path.join(base_dir, "derenzo-projs_T8.npy")
    output_path = args.out or os.path.join(base_dir, "recon_mlem_torch_derenzo_T8_gauss.npz")

    # --- Geometry / dimensions ---
    IMG_DIM = 200
    SFOV = IMG_DIM * IMG_DIM

    # --- MLEM settings ---
    N_ITERATIONS = args.iters
    SAVE_EVERY = args.save_every
    CONVERGENCE_TOL = args.conv_tol

    # --- Gaussian post-filter settings ---
    # Paper uses Gaussian post-filter (HR: 0.16 mm FWHM at voxel size 0.05 mm)  [oai_citation:1‡24-2025-TMI-SCSPECT_SAI_TMI_20250908.docx](sediment://file_00000000d65071fda76a693e9555a063)
    APPLY_GAUSS_POST_FILTER = not args.no_gauss
    GAUSS_FWHM_MM = args.gauss_fwhm_mm
    MM_PER_PX = args.mm_per_px

    # If you want the filter applied after every iteration (stronger smoothing / regularization),
    # set this True. For a true "post-filter", keep False (default).
    APPLY_GAUSS_EACH_ITER = bool(args.gauss_each_iter)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load data ---
    print("Loading flist + projections...")
    flist = get_flist(flist_path)
    if len(flist) == 0:
        raise RuntimeError(f"Empty flist: {flist_path}")

    pdata_full_np = np.load(projs_path)  # expected shape: (len(flist), inferred SPROJ)
    if pdata_full_np.ndim != 2:
        raise ValueError(f"Projection array must be 2D, got shape {pdata_full_np.shape}.")
    if pdata_full_np.shape[0] != len(flist):
        raise ValueError(
            f"Projection rows ({pdata_full_np.shape[0]}) != number of system matrices ({len(flist)}). "
            f"Check flist/projection order."
        )
    SPROJ = int(pdata_full_np.shape[1])
    if SPROJ <= 0:
        raise ValueError(f"Projection width must be positive, got {SPROJ}.")
    validate_matrix_shapes(flist, SPROJ, SFOV)
    print(f"Inferred SPROJ from projections/system matrices: {SPROJ}")

    pdata_full = torch.from_numpy(pdata_full_np).to(device=device, dtype=torch.float32)

    # --- Initialization ---
    estimate = torch.ones(SFOV, device=device, dtype=torch.float32)

    estimates_history = []
    estimates_history_filt = []
    times_history = []
    diffs_history = []

    # Progress bars
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        main_task = progress.add_task("[green]MLEM Iterations", total=N_ITERATIONS)

        for it in range(N_ITERATIONS):
            t0 = time.time()
            estimate_prev = estimate.clone()

            back_projection = torch.zeros(SFOV, device=device, dtype=torch.float32)
            sensitivity_map = torch.zeros(SFOV, device=device, dtype=torch.float32)

            inner_task = progress.add_task(
                f"[cyan]  Iter {it+1}/{N_ITERATIONS}", total=len(flist), transient=True
            )

            for i, fname in enumerate(flist):
                with h5py.File(fname, "r") as h5f:
                    # (n_crystals, SFOV) -> reshape to (1, SPROJ, SFOV)
                    m_chunk = torch.from_numpy(h5f["ppdfs"][:]).to(device=device, dtype=torch.float32)
                    m_chunk = m_chunk.view(1, SPROJ, SFOV)

                p_chunk = pdata_full[i].view(1, SPROJ)

                # Forward projection: y = A * x
                y_chunk = torch.matmul(m_chunk, estimate)  # (1, SPROJ)
                y_chunk = torch.clamp(y_chunk, min=1e-12)  # avoid div0 safely

                # Ratio: r = p / y
                r_chunk = p_chunk / y_chunk  # (1, SPROJ)

                # Backprojection: A^T * r
                back_projection += torch.matmul(m_chunk.transpose(1, 2), r_chunk.unsqueeze(-1)).squeeze()

                # Sensitivity: sum over projection bins
                sensitivity_map += torch.sum(m_chunk, dim=1).squeeze()

                progress.update(inner_task, advance=1)

            sensitivity_map = torch.clamp(sensitivity_map, min=1e-12)
            estimate = estimate * (back_projection / sensitivity_map)

            # Optional: gaussian smoothing each iteration (NOT classic "post-filter")
            if APPLY_GAUSS_POST_FILTER and APPLY_GAUSS_EACH_ITER:
                est2d = estimate.view(IMG_DIM, IMG_DIM)
                est2d = gaussian_filter_2d(est2d, fwhm_mm=GAUSS_FWHM_MM, mm_per_px=MM_PER_PX)
                estimate = est2d.reshape(-1)

            # Convergence metric
            diff = torch.norm(estimate - estimate_prev) / torch.norm(estimate_prev)
            diffs_history.append(float(diff.item()))

            dt = time.time() - t0
            times_history.append(dt)

            # Save snapshots (unfiltered + filtered-for-display)
            if it % SAVE_EVERY == 0:
                est2d_cpu = estimate.view(IMG_DIM, IMG_DIM).detach().cpu()
                estimates_history.append(est2d_cpu.numpy())

                if APPLY_GAUSS_POST_FILTER:
                    est2d_f = gaussian_filter_2d(est2d_cpu.to(device), fwhm_mm=GAUSS_FWHM_MM, mm_per_px=MM_PER_PX)
                    estimates_history_filt.append(est2d_f.detach().cpu().numpy())
                else:
                    estimates_history_filt.append(est2d_cpu.numpy())

            # Update progress label
            progress.update(main_task, advance=1, description=f"[green]MLEM (diff: {diff:.2e})")

            if diff < CONVERGENCE_TOL:
                print(f"\nConverged at iter {it+1} with diff={diff:.2e}")
                progress.update(main_task, completed=N_ITERATIONS)
                break

    # Final post-filter (classic meaning)
    final_est_2d = estimate.view(IMG_DIM, IMG_DIM).detach()
    final_est_2d_filt = final_est_2d
    if APPLY_GAUSS_POST_FILTER and (not APPLY_GAUSS_EACH_ITER):
        final_est_2d_filt = gaussian_filter_2d(final_est_2d, fwhm_mm=GAUSS_FWHM_MM, mm_per_px=MM_PER_PX)

    # Save
    print("\nSaving recon results...")
    np.savez_compressed(
        output_path,
        estimates=np.array(estimates_history),                 # snapshots (unfiltered)
        estimates_gauss=np.array(estimates_history_filt),      # snapshots (filtered)
        final=final_est_2d.cpu().numpy(),
        final_gauss=final_est_2d_filt.cpu().numpy(),
        times=np.array(times_history),
        diffs=np.array(diffs_history),
        meta=np.array(
            [f"T8 MLEM | SPROJ={SPROJ} SFOV={SFOV} iters={N_ITERATIONS} "
             f"gauss={APPLY_GAUSS_POST_FILTER} fwhm_mm={GAUSS_FWHM_MM} each_iter={APPLY_GAUSS_EACH_ITER}"],
            dtype=object
        ),
    )
    print(f"Saved: {output_path}")
    print("Done.")
