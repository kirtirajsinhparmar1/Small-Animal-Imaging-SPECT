'''
python arg_ppdf_t8.py 0 --layout_file ./data/scanner_layouts_*.tensor --a_mm 0.8 --b_mm 0.8
python arg_ppdf_t8.py 1 --layout_file ./data/scanner_layouts_*.tensor --a_mm 0.8 --b_mm 0.8
'''
#!/usr/bin/env python3
import os
import time
import argparse
import h5py
import numpy as np
import concurrent.futures
import multiprocessing as mp
from typing import Optional
from torch import device, arange, tensor, get_num_threads

from scanner_modeling._raytracer_2d._local_functions import (
    ppdf_2d_local,
    reduced_edges_2d_local,
    sfov_properties,
    subdivision_grid_rectangle,
)
from scanner_modeling.geometry_2d import (
    fov_tensor_dict,
    load_scanner_geometry_from_layout,
    load_scanner_layouts,
)

def ellipse_offsets_t8(a_mm: float = 0.2, b_mm: float = 0.2, phase_deg: float = 0.0):
    """8 bed positions on an ellipse (a,b) in mm."""
    phase = np.deg2rad(phase_deg)
    thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
    return [(float(a_mm*np.cos(t)), float(b_mm*np.sin(t))) for t in thetas]

def _set_torch_threads(torch_threads: Optional[int], interop_threads: Optional[int]):
    # Torch may not be installed in every environment running static checks; import inside.
    if torch_threads is None and interop_threads is None:
        return
    import torch

    if torch_threads is not None:
        torch.set_num_threads(int(torch_threads))
    if interop_threads is not None:
        # Safe to call early in process; avoid calling repeatedly in the parent.
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except Exception:
            pass


def _compute_pose_worker(
    *,
    layout_idx: int,
    pose_idx: int,
    dx: float,
    dy: float,
    layout_file: str,
    output_dir: str,
    torch_threads: Optional[int],
    torch_interop_threads: Optional[int],
    skip_existing: bool,
):
    _set_torch_threads(torch_threads, torch_interop_threads)

    out_name = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
    out_path = os.path.join(output_dir, out_name)
    if skip_existing and os.path.exists(out_path):
        return out_path

    layout_dir = os.path.dirname(layout_file)
    layout_fname = os.path.basename(layout_file)
    scanner_layouts, layouts_md5 = load_scanner_layouts(layout_dir, layout_fname)

    return compute_pose(
        layout_idx=layout_idx,
        pose_idx=pose_idx,
        dx=dx,
        dy=dy,
        scanner_layouts=scanner_layouts,
        layouts_md5=layouts_md5,
        output_dir=output_dir,
    )

def compute_pose(
    *,
    layout_idx: int,
    pose_idx: int,
    dx: float,
    dy: float,
    scanner_layouts,
    layouts_md5: str,
    output_dir: str,
):
    default_device = device("cpu")
    print(f"[T8 {pose_idx:02d}] dx={dx:.3f} mm dy={dy:.3f} mm")

    # materials for raytracing (keep consistent with your pipeline)
    mu_dict = tensor([3.5, 0.5], device=default_device)

    # keep these consistent with recon grid
    FOV_NPIX = (200, 200)
    FOV_SIZE_MM = (10, 10)
    SFOV_SUBDIV = (5, 5)
    CRYSTAL_SUBS = (1, 5)
    subdivision_grid = subdivision_grid_rectangle(CRYSTAL_SUBS)

    (
        plate_objects_vertices,
        crystal_objects_vertices,
        plate_objects_edges,
        crystal_objects_edges,
    ) = load_scanner_geometry_from_layout(layout_idx, scanner_layouts)

    n_crystals_total = int(crystal_objects_vertices.shape[0])
    crystal_idx_tensor = arange(n_crystals_total)

    # Translation implemented by shifting FOV center
    fov_dict = fov_tensor_dict(
        FOV_NPIX,          # pixels
        FOV_SIZE_MM,       # mm
        (dx, dy),          # shifted center in mm
        SFOV_SUBDIV,       # sfov grid
    )

    sfov_pxs_ids, sfov_pixels_batch, sfov_corners_batch = sfov_properties(fov_dict)
    fov_n_pxs = int(fov_dict["n pixels"].prod())
    n_sfov = int(fov_dict["n subdivisions"].prod())

    sfov_pxs_ids_1d = (
        sfov_pxs_ids[:, :, 0] * fov_dict["n pixels"][0] + sfov_pxs_ids[:, :, 1]
    )

    os.makedirs(output_dir, exist_ok=True)
    out_name = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
    out_path = os.path.join(output_dir, out_name)
    if os.path.exists(out_path):
        raise FileExistsError(f"Output already exists: {out_path}")

    print(f"  → writing {out_path}")
    print(f"  crystals={n_crystals_total} | sfov={n_sfov} | threads={get_num_threads()}")

    t0 = time.time()
    with h5py.File(out_path, "w") as h5file:
        # attrs for traceability
        h5file.attrs["layout_idx"] = int(layout_idx)
        h5file.attrs["layouts_md5"] = str(layouts_md5)
        h5file.attrs["pose_idx"] = int(pose_idx)
        h5file.attrs["dx_mm"] = float(dx)
        h5file.attrs["dy_mm"] = float(dy)
        h5file.attrs["pose_tag"] = "t8"

        dset = h5file.create_dataset("ppdfs", (n_crystals_total, fov_n_pxs), dtype="f")

        for dataset_idx, crystal_idx_tensor_val in enumerate(crystal_idx_tensor):
            crystal_idx = int(crystal_idx_tensor_val.item())

            reduced_crystal_edges_sfovs = []
            reduced_plate_edges_sfovs = []
            for sfov_idx in range(n_sfov):
                reduced_plate_edges, reduced_crystal_edges = reduced_edges_2d_local(
                    sfov_idx, crystal_idx, sfov_corners_batch,
                    plate_objects_vertices, plate_objects_edges,
                    crystal_objects_vertices, crystal_objects_edges,
                    default_device,
                )
                reduced_crystal_edges_sfovs.append(reduced_crystal_edges)
                reduced_plate_edges_sfovs.append(reduced_plate_edges)

            for sfov_idx in range(n_sfov):
                ppdf_slice = ppdf_2d_local(
                    sfov_idx, crystal_idx, sfov_pixels_batch,
                    crystal_objects_vertices,
                    reduced_plate_edges_sfovs[sfov_idx],
                    reduced_crystal_edges_sfovs[sfov_idx],
                    subdivision_grid,
                    mu_dict,
                    default_device,
                )
                dset[dataset_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

            if (dataset_idx + 1) % 200 == 0:
                print(f"    computed {dataset_idx+1}/{n_crystals_total} crystals...")

    print(f"  done in {time.time()-t0:.2f} s")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index inside the .tensor file")
    ap.add_argument("--layout_file", type=str, required=True, help="path to scanner_layouts_*.tensor")
    here = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(here, "data")
    ap.add_argument("--output_dir", type=str, default=default_output_dir,
                    help="output directory for position_###_ppdfs_t8_##.hdf5 (default: ./data)")
    ap.add_argument("--a_mm", type=float, default=0.2, help="ellipse semi-axis a (X) in mm")
    ap.add_argument("--b_mm", type=float, default=0.2, help="ellipse semi-axis b (Y) in mm")
    ap.add_argument("--phase_deg", type=float, default=0.0, help="phase rotate the 8 positions (deg)")
    ap.add_argument("--pose_idx", type=int, default=None,
                    help="run only this single pose index (0-7). If omitted, run all 8.")
    ap.add_argument("--pose-workers", type=int, default=1,
                    help="when running all 8 poses, number of parallel worker processes (default: 1)")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="torch.set_num_threads() inside each worker (default: auto)")
    ap.add_argument("--torch-interop-threads", type=int, default=None,
                    help="torch.set_num_interop_threads() inside each worker (default: auto)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="if set, skip poses whose output HDF5 already exists")
    args = ap.parse_args()

    layout_dir = os.path.dirname(args.layout_file)
    layout_fname = os.path.basename(args.layout_file)
    scanner_layouts, layouts_md5 = load_scanner_layouts(layout_dir, layout_fname)

    if not (0 <= args.layout_idx < len(scanner_layouts)):
        raise ValueError(f"layout_idx={args.layout_idx} out of range 0..{len(scanner_layouts)-1}")

    poses = ellipse_offsets_t8(args.a_mm, args.b_mm, args.phase_deg)

    if args.pose_idx is not None:
        # Single pose mode
        if not (0 <= args.pose_idx < len(poses)):
            raise ValueError(f"pose_idx={args.pose_idx} out of range 0..{len(poses)-1}")
        print(f"--- T8 PPDF | layout={args.layout_idx} | pose={args.pose_idx} | a={args.a_mm} b={args.b_mm} ---")
        dx, dy = poses[args.pose_idx]
        out_name = f"position_{args.layout_idx:03d}_ppdfs_t8_{args.pose_idx:02d}.hdf5"
        out_path = os.path.join(os.path.abspath(args.output_dir), out_name)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            return
        compute_pose(
            layout_idx=args.layout_idx,
            pose_idx=args.pose_idx,
            dx=dx, dy=dy,
            scanner_layouts=scanner_layouts,
            layouts_md5=layouts_md5,
            output_dir=args.output_dir,
        )
    else:
        # All 8 poses
        print(f"--- T8 PPDFs | layout={args.layout_idx} | a={args.a_mm} b={args.b_mm} | poses=8 ---")
        pose_workers = int(args.pose_workers or 1)
        if pose_workers <= 1:
            for i, (dx, dy) in enumerate(poses):
                compute_pose(
                    layout_idx=args.layout_idx,
                    pose_idx=i,
                    dx=dx, dy=dy,
                    scanner_layouts=scanner_layouts,
                    layouts_md5=layouts_md5,
                    output_dir=args.output_dir,
                )
        else:
            # Use separate worker processes per pose; each worker reloads the layout tensor.
            # This avoids pickling large torch tensors and keeps outputs disjoint (one HDF5 per pose).
            max_workers = min(pose_workers, len(poses))
            print(f"[parallel] pose_workers={max_workers} | torch_threads={args.torch_threads} | torch_interop_threads={args.torch_interop_threads}")

            ctx = mp.get_context("spawn")
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                for i, (dx, dy) in enumerate(poses):
                    futures.append(
                        ex.submit(
                            _compute_pose_worker,
                            layout_idx=args.layout_idx,
                            pose_idx=i,
                            dx=dx,
                            dy=dy,
                            layout_file=os.path.abspath(args.layout_file),
                            output_dir=os.path.abspath(args.output_dir),
                            torch_threads=args.torch_threads,
                            torch_interop_threads=args.torch_interop_threads,
                            skip_existing=bool(args.skip_existing),
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    out_path = fut.result()
                    print(f"[parallel] finished: {out_path}")

if __name__ == "__main__":
    main()
