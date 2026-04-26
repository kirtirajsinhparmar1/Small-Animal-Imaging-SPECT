#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import os
import h5py
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn, MofNCompleteColumn

def get_flist(input_file: str) -> list:
    with open(input_file, "r") as f:
        flist = f.readlines()
        flist = [x.strip() for x in flist]
        return flist

if __name__ == "__main__":

    # --- Setup ---
    torch.device("cpu")
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(here, "data"))
    default_phantom = os.path.join(here, "hot_rods_phantom_10.0_mm_x_10.0_mm.pt")

    ap = argparse.ArgumentParser(description="Forward-project hot-rods phantom using T8 matrices.")
    ap.add_argument("--data-dir", default=default_data_dir, help="Directory containing dataset_flist.csv and matrix files")
    ap.add_argument("--flist", default=None, help="flist path (default: <data-dir>/dataset_flist.csv)")
    ap.add_argument("--phantom", default=default_phantom, help="Phantom .pt file (default: recon/hot_rods_phantom_10.0_mm_x_10.0_mm.pt)")
    ap.add_argument("--out", default=None, help="Output .npy path (default: <data-dir>/derenzo-projs_T8.npy)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    flist_path = args.flist or os.path.join(data_dir, "dataset_flist.csv")
    out_path = args.out or os.path.join(data_dir, "derenzo-projs_T8.npy")
    flist = get_flist(flist_path)

    IMG_SIZE = 200
    sfov_expected = IMG_SIZE * IMG_SIZE
    sproj = 3360

    # --- Phantom Loading and Resizing ---
    phantom_data = torch.load(args.phantom, map_location="cpu")
    phantom_tensor = phantom_data["Phantom tensor"]

    h, w = phantom_tensor.shape
    pad_h = (IMG_SIZE - h) // 2
    pad_w = (IMG_SIZE - w) // 2

    phantom_padded = torch.nn.functional.pad(
        phantom_tensor, (pad_w, pad_w, pad_h, pad_h), "constant", 0
    )

    phantom_flat = phantom_padded.view(-1)

    print(f"Original phantom shape: {phantom_tensor.shape}")
    print(f"Padded phantom shape:   {phantom_padded.shape}")
    if phantom_flat.shape[0] != sfov_expected:
        raise ValueError("FATAL: Padded phantom size does not match expected system matrix FOV.")

    # --- Batch Processing (T8: each file is already a translated pose matrix) ---
    all_projs = []

    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    with progress:
        task = progress.add_task("[green]Projecting (T8 matrices)...", total=len(flist))
        for fname in flist:
            with h5py.File(fname, "r") as h5f:
                matrix_chunk = torch.tensor(h5f["ppdfs"][:]).view(1, sproj, sfov_expected)
                proj_chunk = torch.matmul(matrix_chunk, phantom_flat)
                all_projs.append(proj_chunk)

            progress.update(task, advance=1)

    final_projs = torch.cat(all_projs, dim=0)

    # --- Save the final result ---
    np.save(out_path, final_projs.numpy())

    print("\nProjection complete (T8).")
    print(f"Final projection shape: {final_projs.shape}")
    print(f"Saved projections to: {out_path}")
