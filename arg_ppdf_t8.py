'''
python arg_ppdf_t8.py 0 --layout_file ./data/scanner_layouts_*.tensor --a_mm 0.8 --b_mm 0.8
python arg_ppdf_t8.py 1 --layout_file ./data/scanner_layouts_*.tensor --a_mm 0.8 --b_mm 0.8
'''
#!/usr/bin/env python3
import os
import time
import argparse
import numpy as np
import concurrent.futures
import multiprocessing as mp
from typing import Optional

from srm_row_sampling import make_sampled_detector_rows, write_row_sampling_metadata

def ellipse_offsets_t8(a_mm: float = 0.2, b_mm: float = 0.2, phase_deg: float = 0.0):
    """8 bed positions on an ellipse (a,b) in mm."""
    phase = np.deg2rad(phase_deg)
    thetas = np.linspace(0, 2*np.pi, 8, endpoint=False) + phase
    return [(float(a_mm*np.cos(t)), float(b_mm*np.sin(t))) for t in thetas]


def parse_pose_idxs(value: str) -> list[int]:
    poses = []
    seen = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            pose = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid pose index: {item}") from exc
        if pose < 0 or pose > 7:
            raise ValueError(f"pose index must be in [0, 7], got {pose}")
        if pose in seen:
            raise ValueError(f"Duplicate pose index: {pose}")
        poses.append(pose)
        seen.add(pose)
    if not poses:
        raise ValueError("At least one pose index is required")
    return poses

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


def _to_numpy_int64(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def _scalar_or_none(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return value.item()
    return value


def _layout_metadata(scanner_layouts, layout_idx: int) -> dict:
    if not isinstance(scanner_layouts, dict):
        return {}
    top_level = {
        key: scanner_layouts.get(key)
        for key in (
            "active_detector_unit_indices",
            "n_total_detector_rows",
            "n_active_detector_rows",
            "n_crystals_ring1",
            "n_crystals_ring2",
            "n_apertures",
            "aperture_diam_mm",
        )
        if key in scanner_layouts
    }
    layout_entry = scanner_layouts.get(f"position {layout_idx:03d}", {})
    if isinstance(layout_entry, dict):
        top_level.update(layout_entry)
    return top_level


def _extra_sampling_attrs(layout_meta: dict) -> dict:
    return {
        key: _scalar_or_none(layout_meta.get(key))
        for key in (
            "n_crystals_ring1",
            "n_crystals_ring2",
            "n_apertures",
            "aperture_diam_mm",
            "n_active_detector_rows",
        )
    }


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
    srm_row_fraction: float,
    srm_row_mode: str,
    srm_row_seed: int,
):
    _set_torch_threads(torch_threads, torch_interop_threads)
    from scanner_modeling.geometry_2d import load_scanner_layouts

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
        srm_row_fraction=srm_row_fraction,
        srm_row_mode=srm_row_mode,
        srm_row_seed=srm_row_seed,
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
    srm_row_fraction: float = 1.0,
    srm_row_mode: str = "ring_cell_random",
    srm_row_seed: int = 42,
):
    from torch import device, get_num_threads, tensor
    from scanner_modeling._raytracer_2d._local_functions import (
        ppdf_2d_local,
        reduced_edges_2d_local,
        sfov_properties,
        subdivision_grid_rectangle,
    )
    from scanner_modeling.geometry_2d import (
        fov_tensor_dict,
        load_scanner_geometry_from_layout,
    )

    default_device = device("cpu")
    import h5py

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
    layout_meta = _layout_metadata(scanner_layouts, layout_idx)
    active_detector_unit_indices = _to_numpy_int64(
        layout_meta.get("active_detector_unit_indices")
    )
    if active_detector_unit_indices is None:
        active_detector_unit_indices = np.arange(n_crystals_total, dtype=np.int64)

    sampled_rows = make_sampled_detector_rows(
        total_rows=n_crystals_total,
        fraction=srm_row_fraction,
        mode=srm_row_mode,
        seed=srm_row_seed,
        active_detector_unit_indices=active_detector_unit_indices,
    )
    n_active_rows = int(len(active_detector_unit_indices))
    n_sampled_rows = len(sampled_rows)
    print(
        "  SRM row sampling: "
        f"fraction={srm_row_fraction}, mode={srm_row_mode}, seed={srm_row_seed}, "
        f"sampled={n_sampled_rows}/{n_active_rows} active detector rows, "
        f"total_identity={n_crystals_total}"
    )

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

        dset = h5file.create_dataset("ppdfs", (n_sampled_rows, fov_n_pxs), dtype="f")
        write_row_sampling_metadata(
            h5file,
            sampled_rows,
            fraction=srm_row_fraction,
            mode=srm_row_mode,
            seed=srm_row_seed,
            total_rows=n_crystals_total,
            active_detector_unit_indices=active_detector_unit_indices,
            extra_attrs=_extra_sampling_attrs(layout_meta),
        )

        for local_row_idx, crystal_idx in enumerate(sampled_rows):
            crystal_idx = int(crystal_idx)

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
                dset[local_row_idx, sfov_pxs_ids_1d[sfov_idx]] = ppdf_slice.cpu().numpy()

            if (local_row_idx + 1) % 200 == 0:
                print(f"    computed {local_row_idx+1}/{n_sampled_rows} sampled detector rows...")

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
    ap.add_argument("--pose-idxs", dest="pose_idxs", type=str, default=None,
                    help="comma-separated pose indices to run, e.g. 0,2,4,6")
    ap.add_argument("--pose-workers", type=int, default=1,
                    help="when running multiple poses, number of parallel worker processes (default: 1)")
    ap.add_argument("--torch-threads", type=int, default=None,
                    help="torch.set_num_threads() inside each worker (default: auto)")
    ap.add_argument("--torch-interop-threads", type=int, default=None,
                    help="torch.set_num_interop_threads() inside each worker (default: auto)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="if set, skip poses whose output HDF5 already exists")
    ap.add_argument("--srm-row-fraction", type=float, default=1.0,
                    help="fraction of detector-response rows to compute and store (default: 1.0)")
    ap.add_argument("--srm-row-mode", type=str, default="ring_cell_random",
                    choices=["all", "every_k", "evenly_spaced", "random", "ring_cell_stratified", "ring_cell_random"],
                    help="detector-row sampling mode (default: ring_cell_random)")
    ap.add_argument("--srm-row-seed", type=int, default=42,
                    help="random seed for detector-row sampling modes (default: 42)")
    args = ap.parse_args()

    from scanner_modeling.geometry_2d import load_scanner_layouts

    layout_dir = os.path.dirname(args.layout_file)
    layout_fname = os.path.basename(args.layout_file)
    scanner_layouts, layouts_md5 = load_scanner_layouts(layout_dir, layout_fname)

    if not (0 <= args.layout_idx < len(scanner_layouts)):
        raise ValueError(f"layout_idx={args.layout_idx} out of range 0..{len(scanner_layouts)-1}")

    poses = ellipse_offsets_t8(args.a_mm, args.b_mm, args.phase_deg)

    if args.pose_idx is not None and args.pose_idxs is not None:
        raise ValueError("Use either --pose_idx or --pose-idxs, not both")

    if args.pose_idx is not None:
        if not (0 <= args.pose_idx < len(poses)):
            raise ValueError(f"pose_idx={args.pose_idx} out of range 0..{len(poses)-1}")
        selected_pose_indices = [args.pose_idx]
    elif args.pose_idxs is not None:
        selected_pose_indices = parse_pose_idxs(args.pose_idxs)
    else:
        selected_pose_indices = list(range(len(poses)))

    selected_pose_label = ",".join(str(pose_idx) for pose_idx in selected_pose_indices)

    if len(selected_pose_indices) == 1:
        pose_idx = selected_pose_indices[0]
        print(
            f"--- T8 PPDF | layout={args.layout_idx} | pose={pose_idx} | "
            f"selected poses={selected_pose_label} | a={args.a_mm} b={args.b_mm} ---"
        )
        dx, dy = poses[pose_idx]
        out_name = f"position_{args.layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
        out_path = os.path.join(os.path.abspath(args.output_dir), out_name)
        if args.skip_existing and os.path.exists(out_path):
            print(f"[skip] exists: {out_path}")
            return
        compute_pose(
            layout_idx=args.layout_idx,
            pose_idx=pose_idx,
            dx=dx, dy=dy,
            scanner_layouts=scanner_layouts,
            layouts_md5=layouts_md5,
            output_dir=args.output_dir,
            srm_row_fraction=args.srm_row_fraction,
            srm_row_mode=args.srm_row_mode,
            srm_row_seed=args.srm_row_seed,
        )
    else:
        selected_jobs = [(pose_idx, poses[pose_idx]) for pose_idx in selected_pose_indices]
        print(
            f"--- T8 PPDFs | layout={args.layout_idx} | selected poses={selected_pose_label} | "
            f"a={args.a_mm} b={args.b_mm} ---"
        )
        pose_workers = int(args.pose_workers or 1)
        if pose_workers <= 1:
            for pose_idx, (dx, dy) in selected_jobs:
                out_name = f"position_{args.layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
                out_path = os.path.join(os.path.abspath(args.output_dir), out_name)
                if args.skip_existing and os.path.exists(out_path):
                    print(f"[skip] exists: {out_path}")
                    continue
                compute_pose(
                    layout_idx=args.layout_idx,
                    pose_idx=pose_idx,
                    dx=dx, dy=dy,
                    scanner_layouts=scanner_layouts,
                    layouts_md5=layouts_md5,
                    output_dir=args.output_dir,
                    srm_row_fraction=args.srm_row_fraction,
                    srm_row_mode=args.srm_row_mode,
                    srm_row_seed=args.srm_row_seed,
                )
        else:
            # Use separate worker processes per pose; each worker reloads the layout tensor.
            # This avoids pickling large torch tensors and keeps outputs disjoint (one HDF5 per pose).
            max_workers = min(pose_workers, len(selected_jobs))
            print(
                f"[parallel] pose_workers={max_workers} | "
                f"torch_threads={args.torch_threads} | "
                f"torch_interop_threads={args.torch_interop_threads}"
            )

            ctx = mp.get_context("spawn")
            futures = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
                for pose_idx, (dx, dy) in selected_jobs:
                    futures.append(
                        ex.submit(
                            _compute_pose_worker,
                            layout_idx=args.layout_idx,
                            pose_idx=pose_idx,
                            dx=dx,
                            dy=dy,
                            layout_file=os.path.abspath(args.layout_file),
                            output_dir=os.path.abspath(args.output_dir),
                            torch_threads=args.torch_threads,
                            torch_interop_threads=args.torch_interop_threads,
                            skip_existing=bool(args.skip_existing),
                            srm_row_fraction=args.srm_row_fraction,
                            srm_row_mode=args.srm_row_mode,
                            srm_row_seed=args.srm_row_seed,
                        )
                    )
                for fut in concurrent.futures.as_completed(futures):
                    out_path = fut.result()
                    print(f"[parallel] finished: {out_path}")

if __name__ == "__main__":
    main()
