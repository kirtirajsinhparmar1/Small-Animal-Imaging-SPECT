# '''
# python arg_extract_beam_masks.py 0 --t4
# python arg_extract_beam_masks.py 1 --t4
# '''
# #!/usr/bin/env python3
# import sys
# import os
# import argparse
# from torch import tensor, arange, cat

# from beam_property_extract import (
#     beams_boundaries_radians,
#     get_beams_masks,
#     get_beams_combined_mask,
#     sample_ppdf_on_arc_2d_local,
# )
# from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
# from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
# from geometry_2d_utils import (
#     fov_tensor_dict,
#     pixels_coordinates,
#     pixels_to_detector_unit_rads,
# )
# from ppdf_io import load_ppdfs_data_from_hdf5
# from beam_property_io import (
#     initialize_beam_masks_hdf5,
#     append_to_hdf5_dataset,
# )

# # Paper T4 positions (x,y) in mm (same ordering you used for raytracing)
# T4_OFFSETS_XY = [
#     (-0.4, -0.4),
#     ( 0.4,  0.4),
#     (-0.4,  0.4),
#     ( 0.4, -0.4),
# ]


# def run_one(layout_idx: int, ppdf_filename: str, out_hdf5_filename: str, fov_center_xy=(0.0, 0.0)):
#     print(f"\n--- Beam mask extraction ---")
#     print(f"[INFO] layout_idx       : {layout_idx}")
#     print(f"[INFO] PPDF input file  : {ppdf_filename}")
#     print(f"[INFO] FOV center (mm)  : {fov_center_xy}")
#     print(f"[INFO] Output HDF5      : {out_hdf5_filename}")

#     scanner_layouts_dir = "./data"
#     scanner_layouts_filename = (
#         "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
#     )
#     ppdfs_dataset_dir = "./data"
#     out_dir = "./data"
#     os.makedirs(out_dir, exist_ok=True)

#     # Load layouts
#     scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
#         scanner_layouts_dir, scanner_layouts_filename
#     )

#     # IMPORTANT: fov center must match the pose used to generate the PPDF file
#     fov_dict = fov_tensor_dict(
#         n_pixels=(200, 200),
#         mm_per_pixel=(0.05, 0.05),
#         center_coordinates=(float(fov_center_xy[0]), float(fov_center_xy[1])),
#     )
#     fov_n_pixels_int = int(fov_dict["n pixels"].prod())

#     # Init output HDF5
#     out_hdf5_file, beams_masks_dataset = initialize_beam_masks_hdf5(
#         fov_n_pixels_int, out_hdf5_filename, out_dir
#     )
#     print(f"[INFO] Saving to: {os.path.join(out_dir, out_hdf5_filename)}")

#     # Load geometry for the layout
#     plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
#         int(layout_idx), scanner_layouts_data
#     )

#     # Load corresponding PPDFs
#     ppdfs = load_ppdfs_data_from_hdf5(
#         ppdfs_dataset_dir, ppdf_filename, fov_dict
#     )

#     # Precompute hull points (depends on FOV, so must be rebuilt per pose)
#     detector_unit_centers = detector_units_vertices.mean(dim=1)
#     fov_corners = (
#         tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
#         * fov_dict["size in mm"]
#         * 0.5
#     )
#     hull_points_batch = cat(
#         (
#             fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
#             detector_unit_centers.unsqueeze(1),
#         ),
#         dim=1,
#     )
#     hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

#     # Pixel xy coordinates must match this pose's fov_dict
#     fov_points_xy = pixels_coordinates(fov_dict)

#     n_detector_units = int(detector_units_vertices.shape[0])
#     detector_units_sequence = arange(0, n_detector_units)
#     print(f"[INFO] Processing {n_detector_units} detector units...")

#     for detector_unit_idx in detector_units_sequence:
#         ppdf_data_2d = ppdfs[detector_unit_idx].view(
#             int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
#         )

#         hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

#         sampled_ppdf, sampling_rads, sampling_points = sample_ppdf_on_arc_2d_local(
#             ppdf_data_2d,
#             detector_unit_centers[detector_unit_idx],
#             hull_2d,
#             fov_dict,
#         )

#         beams_boundaries = beams_boundaries_radians(
#             sampled_ppdf, sampling_rads, threshold=0.01
#         )

#         fov_points_rads = pixels_to_detector_unit_rads(
#             fov_points_xy,
#             detector_unit_centers[detector_unit_idx],
#         )

#         beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
#         combined_beams_masks = get_beams_combined_mask(beams_masks)

#         append_to_hdf5_dataset(beams_masks_dataset, combined_beams_masks)

#         if (int(detector_unit_idx) + 1) % 200 == 0:
#             print(f"  ... processed {int(detector_unit_idx) + 1}/{n_detector_units} detector units.")

#     print(f"\n[DONE] Layout {layout_idx} output dataset shape: {list(beams_masks_dataset.shape)}")
#     out_hdf5_file.close()
#     print(f"[DONE] Saved: {os.path.join(out_dir, out_hdf5_filename)}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("layout_idx", type=int, help="layout index to process (e.g., 0)")
#     ap.add_argument("--t4", action="store_true", help="use T4 PPDF files and matching FOV center shifts")
#     ap.add_argument("--pose", type=int, default=None, help="T4 pose index (0..3). If omitted, runs all 4 poses.")
#     args = ap.parse_args()

#     layout_idx = args.layout_idx

#     if not args.t4:
#         # original: one file
#         ppdf_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"
#         out_hdf5_filename = f"beams_masks_configuration_{layout_idx:02d}.hdf5"
#         run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(0.0, 0.0))
#         sys.exit(0)

#     # T4: 1 pose or all poses
#     if args.pose is not None:
#         if not (0 <= args.pose <= 3):
#             raise ValueError("--pose must be 0..3")

#         dx, dy = T4_OFFSETS_XY[args.pose]
#         ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t4_{args.pose:02d}.hdf5"
#         out_hdf5_filename = f"beams_masks_configuration_{layout_idx:02d}_t4_{args.pose:02d}.hdf5"
#         run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(dx, dy))
#     else:
#         for pose_idx in range(4):
#             dx, dy = T4_OFFSETS_XY[pose_idx]
#             ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t4_{pose_idx:02d}.hdf5"
#             out_hdf5_filename = f"beams_masks_configuration_{layout_idx:02d}_t4_{pose_idx:02d}.hdf5"
#             run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(dx, dy))

#!/usr/bin/env python3
import sys
import os
import argparse
import h5py
from typing import Optional
from torch import tensor, arange, cat, float32

from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_combined_mask,
    sample_ppdf_on_arc_2d_local,
)
from convex_hull_helper import convex_hull_2d, sort_points_for_hull_batch_2d
from geometry_2d_io import load_scanner_layout_geometries, load_scanner_layouts
from geometry_2d_utils import (
    fov_tensor_dict,
    pixels_coordinates,
    pixels_to_detector_unit_rads,
)
from ppdf_io import load_ppdfs_data_from_hdf5
from beam_property_io import (
    initialize_beam_masks_hdf5,
    append_to_hdf5_dataset,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_HERE, "data")
DEFAULT_LAYOUT_TENSOR = (
    "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
)
N_T8_POSES = 8

def _pick_default_layout_file(data_dir: str) -> Optional[str]:
    try:
        import glob

        cands = sorted(glob.glob(os.path.join(data_dir, "scanner_layouts_*.tensor")))
        return cands[-1] if cands else None
    except Exception:
        return None


def read_pose_center_from_ppdf_h5(ppdf_path: str):
    """Read dx_mm, dy_mm from the PPDF HDF5 attrs."""
    with h5py.File(ppdf_path, "r") as f:
        dx = float(f.attrs.get("dx_mm", 0.0))
        dy = float(f.attrs.get("dy_mm", 0.0))
        pose_tag = str(f.attrs.get("pose_tag", ""))
        pose_idx = int(f.attrs.get("pose_idx", -1))
    return dx, dy, pose_tag, pose_idx


def load_aggregated_t8_ppdfs(data_dir: str, layout_idx: int):
    """Load and sum the 8 T8 PPDF pose files for one layout."""
    aggregated = None
    loaded = 0
    for pose_idx in range(N_T8_POSES):
        ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
        ppdf_path = os.path.join(data_dir, ppdf_filename)
        if not os.path.exists(ppdf_path):
            print(f"[WARN] Missing T8 PPDF pose file: {ppdf_path}")
            continue
        with h5py.File(ppdf_path, "r") as f:
            ppdfs = tensor(f["ppdfs"][:], dtype=float32)
        if aggregated is None:
            aggregated = ppdfs
        else:
            aggregated += ppdfs
        loaded += 1

    if aggregated is None:
        raise FileNotFoundError(f"No T8 PPDF files found for layout {layout_idx:03d} in {data_dir}")
    print(f"[INFO] Aggregated {loaded}/{N_T8_POSES} T8 PPDF pose files for layout {layout_idx:03d}")
    return aggregated


def run_one(
    layout_idx: int,
    ppdf_filename: str,
    out_hdf5_filename: str,
    *,
    data_dir: str,
    layout_file: str,
    fov_center_xy=(0.0, 0.0),
    ppdfs_override=None,
):
    print(f"\n--- Beam mask extraction (T8) ---")
    print(f"[INFO] layout_idx       : {layout_idx}")
    print(f"[INFO] PPDF input file  : {ppdf_filename}")
    print(f"[INFO] FOV center (mm)  : {fov_center_xy}")
    print(f"[INFO] Output HDF5      : {out_hdf5_filename}")
    print(f"[INFO] Layout tensor    : {layout_file}")

    scanner_layouts_dir = os.path.dirname(layout_file)
    scanner_layouts_filename = os.path.basename(layout_file)
    ppdfs_dataset_dir = data_dir
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load layouts
    scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )

    # Aggregate T8 baseline mode intentionally uses the centered FOV, matching Omer's pipeline.
    fov_dict = fov_tensor_dict(
        n_pixels=(200, 200),
        mm_per_pixel=(0.05, 0.05),
        center_coordinates=(float(fov_center_xy[0]), float(fov_center_xy[1])),
    )
    fov_n_pixels_int = int(fov_dict["n pixels"].prod())

    # Init output HDF5
    out_hdf5_file, beams_masks_dataset = initialize_beam_masks_hdf5(
        fov_n_pixels_int, out_hdf5_filename, out_dir
    )
    print(f"[INFO] Saving to: {os.path.join(out_dir, out_hdf5_filename)}")

    # Load geometry for the layout
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        int(layout_idx), scanner_layouts_data
    )

    # Load corresponding PPDFs. T8 baseline mode passes the already summed tensor.
    if ppdfs_override is None:
        ppdfs = load_ppdfs_data_from_hdf5(
            ppdfs_dataset_dir, ppdf_filename, fov_dict
        )
    else:
        ppdfs = ppdfs_override

    # Precompute hull points for this FOV.
    detector_unit_centers = detector_units_vertices.mean(dim=1)
    fov_corners = (
        tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        * fov_dict["size in mm"]
        * 0.5
    )
    hull_points_batch = cat(
        (
            fov_corners.unsqueeze(0).expand(detector_units_vertices.shape[0], -1, -1),
            detector_unit_centers.unsqueeze(1),
        ),
        dim=1,
    )
    hull_points_batch = sort_points_for_hull_batch_2d(hull_points_batch)

    # Pixel xy coordinates must match this fov_dict.
    fov_points_xy = pixels_coordinates(fov_dict)

    n_detector_units = int(detector_units_vertices.shape[0])
    detector_units_sequence = arange(0, n_detector_units)
    print(f"[INFO] Processing {n_detector_units} detector units...")

    for detector_unit_idx in detector_units_sequence:
        ppdf_data_2d = ppdfs[detector_unit_idx].view(
            int(fov_dict["n pixels"][0]), int(fov_dict["n pixels"][1])
        )

        hull_2d = convex_hull_2d(hull_points_batch[detector_unit_idx])

        sampled_ppdf, sampling_rads, sampling_points = sample_ppdf_on_arc_2d_local(
            ppdf_data_2d,
            detector_unit_centers[detector_unit_idx],
            hull_2d,
            fov_dict,
        )

        beams_boundaries = beams_boundaries_radians(
            sampled_ppdf, sampling_rads, threshold=0.01
        )

        fov_points_rads = pixels_to_detector_unit_rads(
            fov_points_xy,
            detector_unit_centers[detector_unit_idx],
        )

        beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
        combined_beams_masks = get_beams_combined_mask(beams_masks)

        append_to_hdf5_dataset(beams_masks_dataset, combined_beams_masks)

        if (int(detector_unit_idx) + 1) % 200 == 0:
            print(f"  ... processed {int(detector_unit_idx) + 1}/{n_detector_units} detector units.")

    print(f"\n[DONE] Layout {layout_idx} output dataset shape: {list(beams_masks_dataset.shape)}")
    out_hdf5_file.close()
    print(f"[DONE] Saved: {os.path.join(out_dir, out_hdf5_filename)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index to process (e.g., 0)")
    ap.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory containing PPDFs and where beams_masks_*.hdf5 are written (default: ./data)",
    )
    ap.add_argument(
        "--layout-file",
        default=None,
        help="path to scanner_layouts_*.tensor (default: newest ./data/scanner_layouts_*.tensor, else DEFAULT_LAYOUT_TENSOR)",
    )
    ap.add_argument("--t8", action="store_true", help="sum T8 PPDF pose files and extract one mask file per layout")
    ap.add_argument("--pose", type=int, default=None, help="legacy per-pose T8 mode is no longer supported")
    args = ap.parse_args()

    layout_idx = args.layout_idx
    data_dir = os.path.abspath(args.data_dir)
    layout_file = args.layout_file or _pick_default_layout_file(data_dir) or os.path.join(data_dir, DEFAULT_LAYOUT_TENSOR)
    layout_file = os.path.abspath(layout_file)

    if not args.t8:
        # original: one file
        ppdf_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"
        out_hdf5_filename = f"beams_masks_configuration_{layout_idx:02d}.hdf5"
        run_one(
            layout_idx,
            ppdf_filename,
            out_hdf5_filename,
            data_dir=data_dir,
            layout_file=layout_file,
            fov_center_xy=(0.0, 0.0),
        )
        return

    if args.pose is not None:
        raise ValueError("--pose is incompatible with T8 aggregate baseline mode")

    aggregated_ppdfs = load_aggregated_t8_ppdfs(data_dir, layout_idx)
    out_hdf5_filename = f"beams_masks_configuration_{layout_idx:03d}.hdf5"
    run_one(
        layout_idx,
        f"position_{layout_idx:03d}_ppdfs_t8_00..07.hdf5 (summed)",
        out_hdf5_filename,
        data_dir=data_dir,
        layout_file=layout_file,
        fov_center_xy=(0.0, 0.0),
        ppdfs_override=aggregated_ppdfs,
    )


if __name__ == "__main__":
    main()
