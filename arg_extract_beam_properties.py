# '''
# python arg_extract_beam_properties.py 0 --t4
# python arg_extract_beam_properties.py 1 --t4
# '''
# #!/usr/bin/env python3
# import sys
# import os
# import argparse
# from torch import cat, tensor, arange

# from beam_property_extract import (
#     beams_boundaries_radians,
#     get_beams_masks,
#     get_beams_weighted_center,
#     get_beam_width,
#     get_beams_angle_radian,
#     get_beams_basic_properties,
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
#     initialize_beam_properties_hdf5,
#     append_to_hdf5_dataset,
#     stack_beams_properties,
# )

# # Paper T4 positions (x,y) in mm (same ordering you used for raytracing)
# T4_OFFSETS_XY = [
#     (-0.4, -0.4),
#     ( 0.4,  0.4),
#     (-0.4,  0.4),
#     ( 0.4, -0.4),
# ]


# def run_one(layout_idx: int, ppdf_filename: str, out_hdf5_filename: str, fov_center_xy=(0.0, 0.0)):
#     print(f"\n--- Beam property extraction ---")
#     print(f"[INFO] layout_idx       : {layout_idx}")
#     print(f"[INFO] PPDF input file  : {ppdf_filename}")
#     print(f"[INFO] FOV center (mm)  : {fov_center_xy}")
#     print(f"[INFO] Output HDF5      : {out_hdf5_filename}")

#     # --- Define paths ---
#     scanner_layouts_dir = "./data"
#     scanner_layouts_filename = (
#         "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
#     )
#     ppdfs_dataset_dir = "./data"
#     out_dir = "./data"
#     os.makedirs(out_dir, exist_ok=True)

#     # --- Load layouts and set FOV ---
#     scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
#         scanner_layouts_dir, scanner_layouts_filename
#     )

#     fov_dict = fov_tensor_dict(
#         n_pixels=(200, 200),
#         mm_per_pixel=(0.05, 0.05),
#         center_coordinates=(float(fov_center_xy[0]), float(fov_center_xy[1])),
#     )

#     # --- Init output HDF5 for this run ---
#     out_hdf5_file, beam_properties_dataset = initialize_beam_properties_hdf5(
#         out_hdf5_filename, out_dir
#     )
#     print(f"[INFO] Saving to: {os.path.join(out_dir, out_hdf5_filename)}")

#     # --- Load geometry & PPDFs for this layout/pose ---
#     plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
#         int(layout_idx), scanner_layouts_data
#     )

#     ppdfs = load_ppdfs_data_from_hdf5(
#         ppdfs_dataset_dir, ppdf_filename, fov_dict
#     )

#     # --- Precompute things that do NOT change per-detector ---
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

#     # pixel xy coordinates are fixed for this fov_dict (depends on center!)
#     fov_points_xy = pixels_coordinates(fov_dict)

#     n_detector_units = int(detector_units_vertices.shape[0])
#     detector_units_sequence = arange(0, n_detector_units)
#     print(f"[INFO] Processing {n_detector_units} detector units...")

#     # --- Main loop ---
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

#         # Convert all pixel xy points into angles for this detector
#         fov_points_rads = pixels_to_detector_unit_rads(
#             fov_points_xy, detector_unit_centers[detector_unit_idx]
#         )

#         beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
#         if beams_masks.shape[0] == 0:
#             continue

#         beams_weighted_centers = get_beams_weighted_center(
#             beams_masks, fov_points_xy, ppdf_data_2d
#         )

#         beams_fwhm, _, _, _ = get_beam_width(
#             beams_weighted_centers,
#             detector_unit_centers[detector_unit_idx],
#             beams_masks,
#             ppdf_data_2d,
#             fov_dict,
#         )

#         beams_angle = get_beams_angle_radian(
#             beams_weighted_centers, detector_unit_centers[detector_unit_idx]
#         )

#         beams_sizes, beams_relative_sensitivity, beams_absolute_sensitivity = get_beams_basic_properties(
#             beams_masks, ppdf_data_2d, fov_points_xy
#         )

#         stacked_beams_properties = stack_beams_properties(
#             int(layout_idx),
#             int(detector_unit_idx),
#             angles=beams_angle,
#             fwhms=beams_fwhm,
#             sizes=beams_sizes,
#             relative_sensitivities=beams_relative_sensitivity,
#             absolute_sensitivities=beams_absolute_sensitivity,
#             weighted_centers=beams_weighted_centers,
#         )

#         if stacked_beams_properties.numel():
#             append_to_hdf5_dataset(beam_properties_dataset, stacked_beams_properties)

#         if (int(detector_unit_idx) + 1) % 200 == 0:
#             print(f"  ... processed {int(detector_unit_idx) + 1}/{n_detector_units} detector units.")

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
#         # original behavior: single file
#         ppdf_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"
#         out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}.hdf5"
#         run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(0.0, 0.0))
#         sys.exit(0)

#     # T4 behavior
#     if args.pose is not None:
#         if not (0 <= args.pose <= 3):
#             raise ValueError("--pose must be 0..3")

#         dx, dy = T4_OFFSETS_XY[args.pose]
#         ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t4_{args.pose:02d}.hdf5"
#         out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}_t4_{args.pose:02d}.hdf5"
#         run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(dx, dy))
#     else:
#         # run all 4 poses
#         for pose_idx in range(4):
#             dx, dy = T4_OFFSETS_XY[pose_idx]
#             ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t4_{pose_idx:02d}.hdf5"
#             out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}_t4_{pose_idx:02d}.hdf5"
#             run_one(layout_idx, ppdf_filename, out_hdf5_filename, fov_center_xy=(dx, dy))

#!/usr/bin/env python3
import sys
import os
import argparse
import h5py
from typing import Optional
from torch import cat, tensor, arange

from beam_property_extract import (
    beams_boundaries_radians,
    get_beams_masks,
    get_beams_weighted_center,
    get_beam_width,
    get_beams_angle_radian,
    get_beams_basic_properties,
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
    initialize_beam_properties_hdf5,
    append_to_hdf5_dataset,
    stack_beams_properties,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(_HERE, "data")
DEFAULT_LAYOUT_TENSOR = (
    "scanner_layouts_24a4365260a3f68491dfa8ca55e0ecc2_rot2_ang1p0deg_trans1x1_step0p0x0p0.tensor"
)

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


def run_one(
    layout_idx: int,
    ppdf_filename: str,
    out_hdf5_filename: str,
    *,
    data_dir: str,
    layout_file: str,
    fov_center_xy,
):
    print(f"\n--- Beam property extraction (T8) ---")
    print(f"[INFO] layout_idx       : {layout_idx}")
    print(f"[INFO] PPDF input file  : {ppdf_filename}")
    print(f"[INFO] FOV center (mm)  : {fov_center_xy}")
    print(f"[INFO] Output HDF5      : {out_hdf5_filename}")
    print(f"[INFO] Layout tensor    : {layout_file}")

    # --- Define paths ---
    scanner_layouts_dir = os.path.dirname(layout_file)
    scanner_layouts_filename = os.path.basename(layout_file)
    ppdfs_dataset_dir = data_dir
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Load layouts and set FOV ---
    scanner_layouts_data, layouts_unique_id = load_scanner_layouts(
        scanner_layouts_dir, scanner_layouts_filename
    )

    fov_dict = fov_tensor_dict(
        n_pixels=(200, 200),
        mm_per_pixel=(0.05, 0.05),
        center_coordinates=(float(fov_center_xy[0]), float(fov_center_xy[1])),
    )

    # --- Init output HDF5 for this run ---
    out_hdf5_file, beam_properties_dataset = initialize_beam_properties_hdf5(
        out_hdf5_filename, out_dir
    )
    print(f"[INFO] Saving to: {os.path.join(out_dir, out_hdf5_filename)}")

    # --- Load geometry & PPDFs for this layout/pose ---
    plates_vertices, detector_units_vertices = load_scanner_layout_geometries(
        int(layout_idx), scanner_layouts_data
    )

    ppdfs = load_ppdfs_data_from_hdf5(
        ppdfs_dataset_dir, ppdf_filename, fov_dict
    )

    # --- Precompute things that do NOT change per-detector ---
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

    # pixel xy coordinates are fixed for this fov_dict (depends on center!)
    fov_points_xy = pixels_coordinates(fov_dict)

    n_detector_units = int(detector_units_vertices.shape[0])
    detector_units_sequence = arange(0, n_detector_units)
    print(f"[INFO] Processing {n_detector_units} detector units...")

    # --- Main loop ---
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

        # Convert all pixel xy points into angles for this detector
        fov_points_rads = pixels_to_detector_unit_rads(
            fov_points_xy, detector_unit_centers[detector_unit_idx]
        )

        beams_masks = get_beams_masks(fov_points_rads, beams_boundaries)
        if beams_masks.shape[0] == 0:
            continue

        beams_weighted_centers = get_beams_weighted_center(
            beams_masks, fov_points_xy, ppdf_data_2d
        )

        beams_fwhm, _, _, _ = get_beam_width(
            beams_weighted_centers,
            detector_unit_centers[detector_unit_idx],
            beams_masks,
            ppdf_data_2d,
            fov_dict,
        )

        beams_angle = get_beams_angle_radian(
            beams_weighted_centers, detector_unit_centers[detector_unit_idx]
        )

        beams_sizes, beams_relative_sensitivity, beams_absolute_sensitivity = get_beams_basic_properties(
            beams_masks, ppdf_data_2d, fov_points_xy
        )

        stacked_beams_properties = stack_beams_properties(
            int(layout_idx),
            int(detector_unit_idx),
            angles=beams_angle,
            fwhms=beams_fwhm,
            sizes=beams_sizes,
            relative_sensitivities=beams_relative_sensitivity,
            absolute_sensitivities=beams_absolute_sensitivity,
            weighted_centers=beams_weighted_centers,
        )

        if stacked_beams_properties.numel():
            append_to_hdf5_dataset(beam_properties_dataset, stacked_beams_properties)

        if (int(detector_unit_idx) + 1) % 200 == 0:
            print(f"  ... processed {int(detector_unit_idx) + 1}/{n_detector_units} detector units.")

    out_hdf5_file.close()
    print(f"[DONE] Saved: {os.path.join(out_dir, out_hdf5_filename)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index to process (e.g., 0)")
    ap.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory containing PPDFs and where beams_properties_*.hdf5 are written (default: ./data)",
    )
    ap.add_argument(
        "--layout-file",
        default=None,
        help="path to scanner_layouts_*.tensor (default: newest ./data/scanner_layouts_*.tensor, else DEFAULT_LAYOUT_TENSOR)",
    )
    ap.add_argument("--t8", action="store_true", help="use T8 PPDF files (position_XXX_ppdfs_t8_YY.hdf5)")
    ap.add_argument("--pose", type=int, default=None, help="T8 pose index (0..7). If omitted, runs all 8 poses.")
    args = ap.parse_args()

    layout_idx = args.layout_idx
    data_dir = os.path.abspath(args.data_dir)
    layout_file = args.layout_file or _pick_default_layout_file(data_dir) or os.path.join(data_dir, DEFAULT_LAYOUT_TENSOR)
    layout_file = os.path.abspath(layout_file)

    if not args.t8:
        # fallback: original single-file behavior
        ppdf_filename = f"position_{layout_idx:03d}_ppdfs.hdf5"
        out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}.hdf5"
        run_one(
            layout_idx,
            ppdf_filename,
            out_hdf5_filename,
            data_dir=data_dir,
            layout_file=layout_file,
            fov_center_xy=(0.0, 0.0),
        )
        return

    # T8 behavior
    ppdf_dir = data_dir

    if args.pose is not None:
        if not (0 <= args.pose <= 7):
            raise ValueError("--pose must be 0..7 for T8")

        ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t8_{args.pose:02d}.hdf5"
        ppdf_path = os.path.join(ppdf_dir, ppdf_filename)
        dx, dy, pose_tag, pose_idx = read_pose_center_from_ppdf_h5(ppdf_path)

        out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}_t8_{args.pose:02d}.hdf5"
        run_one(
            layout_idx,
            ppdf_filename,
            out_hdf5_filename,
            data_dir=data_dir,
            layout_file=layout_file,
            fov_center_xy=(dx, dy),
        )

    else:
        # run all 8 poses
        for pose_idx in range(8):
            ppdf_filename = f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
            ppdf_path = os.path.join(ppdf_dir, ppdf_filename)
            dx, dy, pose_tag, _ = read_pose_center_from_ppdf_h5(ppdf_path)

            out_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}_t8_{pose_idx:02d}.hdf5"
            run_one(
                layout_idx,
                ppdf_filename,
                out_hdf5_filename,
                data_dir=data_dir,
                layout_file=layout_file,
                fov_center_xy=(dx, dy),
            )


if __name__ == "__main__":
    main()
