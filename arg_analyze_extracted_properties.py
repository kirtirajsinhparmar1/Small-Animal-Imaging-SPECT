# import torch
# import h5py
# import os
# import sys

# if __name__ == "__main__":
#     # --- MODIFICATION: Get layout_idx from a command-line argument ---
#     if len(sys.argv) != 2:
#         print("Usage: python generate_asci_histogram_task.py <layout_idx>")
#         sys.exit(1)
#     try:
#         layout_idx = int(sys.argv[1])
#     except ValueError:
#         print(f"Error: <layout_idx> must be an integer. Received: {sys.argv[1]}")
#         sys.exit(1)

#     print(f"--- Starting ASCI histogram generation for layout index: {layout_idx} ---")

#     # --- Configuration ---
#     n_bins = 360
#     angular_bin_boundaries = torch.arange(n_bins + 1) / 180 * torch.pi
#     input_dir = "./data"
#     FWHM_MIN_MM = 0
#     FWHM_MAX_MM = 9

#     # --- Data Loading ---
#     print("Loading beams properties and masks...")
#     try:
#         # Load the beams properties
#         beams_properties_hdf5_filename = f"beams_properties_configuration_{layout_idx:02d}.hdf5"
#         with h5py.File(os.path.join(input_dir, beams_properties_hdf5_filename), "r") as f:
#             layout_beams_properties = torch.from_numpy(f["beam_properties"][:])
#             beam_properties_header = f["beam_properties"].attrs["Header"]

#         # Load the beams masks for the layout
#         beams_masks_hdf5_filename = f"beams_masks_configuration_{layout_idx:02d}.hdf5"
#         with h5py.File(os.path.join(input_dir, beams_masks_hdf5_filename), "r") as beams_masks_hdf5:
#             beams_masks = torch.from_numpy(beams_masks_hdf5["beam_mask"][:])
#     except FileNotFoundError as e:
#         print(f"Error: Input file not found for layout {layout_idx}. Make sure previous steps ran successfully.")
#         print(e)
#         sys.exit(1)

#     # --- Data Processing ---
#     print("Processing data and building histogram...")
    
#     # Digitize the angles
#     digitized_angles = torch.bucketize(
#         layout_beams_properties[:, 3], angular_bin_boundaries, right=False
#     )
#     # Note: The original script had this block duplicated. One is sufficient.
#     layout_beams_properties = torch.cat(
#         (layout_beams_properties, (digitized_angles - 1).unsqueeze(1).float()),
#         dim=1,
#     )
    
#     # Filter out beams with NaN angles
#     layout_beams_properties_filtered = layout_beams_properties[
#         torch.isnan(layout_beams_properties[:, 3]) == False
#     ]

#     print(f"Applying FWHM filter (keeping beams between {FWHM_MIN_MM} and {FWHM_MAX_MM} mm)...")
#     # IMPORTANT: Assuming FWHM is in column 4. Verify with your data header.
#     fwhm_column_index = 4
#     fwhm_values = layout_beams_properties_filtered[:, fwhm_column_index]

#     # Create a boolean mask for beams within the desired FWHM range
#     fwhm_mask = (fwhm_values >= FWHM_MIN_MM) & (fwhm_values <= FWHM_MAX_MM)

#     # Apply the mask
#     layout_beams_properties_filtered = layout_beams_properties_filtered[fwhm_mask]

#     print(f"  {layout_beams_properties_filtered.shape[0]} beams remaining after FWHM filter.")

#     # FILTERING BASED ON MPXI
#     # Find this index by printing beam_properties_header (uncomment the print statement above)
#     # MULTIPLEX_COLUMN_INDEX = 10  # <-- FIXME: Change this to the correct column index
#     # DESIRED_MULTIPLEX_VALUE = 1   # <-- FIXME: Change this to the index value you want to keep

#     # print(f"Applying multiplexing index filter (keeping value: {DESIRED_MULTIPLEX_VALUE})...")
    
#     # if layout_beams_properties_filtered.shape[0] > 0:
#     #     # Get the multiplexing index values from the correct column
#     #     multiplex_values = layout_beams_properties_filtered[:, MULTIPLEX_COLUMN_INDEX]
        
#     #     # Create a boolean mask to keep only the desired index
#     #     multiplex_mask = (multiplex_values == DESIRED_MULTIPLEX_VALUE)
        
#     #     # Apply the mask
#     #     layout_beams_properties_filtered = layout_beams_properties_filtered[multiplex_mask]
#     #     print(f"  {layout_beams_properties_filtered.shape[0]} beams remaining after multiplexing filter.")
#     # else:
#     #     print("  Skipping multiplexing filter, no beams left.")

#     # Filter by sensitivity (optional, kept from original script)
#     if layout_beams_properties_filtered.shape[0] > 0:
#         beams_sensitivity_max = layout_beams_properties_filtered[:, 7].max()
#         layout_beams_properties_filtered = layout_beams_properties_filtered[
#             layout_beams_properties_filtered[:, 7] > beams_sensitivity_max * 0.01
#         ]
#     else:
#         print("Warning: No beams remaining after FWHM filter to apply sensitivity filter.")
    
#     print(f"Found {layout_beams_properties_filtered.shape[0]} valid beams to process.")

#     # Initialize the histogram for ASCI map
#     asci_histogram = torch.zeros((200 * 200, n_bins), dtype=torch.int32)

#     # Loop through the beams and populate the histogram
#     for beam_props in layout_beams_properties_filtered:
#         detector_idx = int(beam_props[1])
#         beam_idx = int(beam_props[2])
#         # The angle bin index is now the last column (index 11 after cat)
#         angle_bin_idx = int(beam_props[-1]) 
        
#         # Ensure angle_bin_idx is valid before using it
#         if 0 <= angle_bin_idx < n_bins:
#             asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

#     # --- Save the Output ---
#     # output_dir = "../../../data/sc_mph_30det_3mm_base_layout_rotated_10custom_rot/filtered_outputs"
#     # os.makedirs(output_dir, exist_ok=True)
#     asci_histogram_filename = os.path.join(input_dir, f"asci_histogram_{layout_idx:02d}.hdf5")
#     print(f"Saving histogram to: {asci_histogram_filename}")
#     with h5py.File(asci_histogram_filename, "w") as f:
#         f.create_dataset("asci_histogram", data=asci_histogram.numpy())

#     print(f"--- Finished ASCI histogram generation for layout index: {layout_idx} ---")

#!/usr/bin/env python3
import torch
import h5py
import os
import sys
import argparse

def load_beam_properties(path: str):
    with h5py.File(path, "r") as f:
        props = torch.from_numpy(f["beam_properties"][:])
        header_raw = f["beam_properties"].attrs["Header"]
        header = [h.decode("utf-8") if isinstance(h, bytes) else str(h) for h in header_raw]
    return props, header

def load_beam_masks(path: str):
    with h5py.File(path, "r") as f:
        masks = torch.from_numpy(f["beam_mask"][:])
    return masks

def get_col_idx(header, name: str):
    if name not in header:
        raise RuntimeError(f"Column '{name}' not found. Available columns: {header}")
    return header.index(name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index (e.g., 0)")
    ap.add_argument("--t8", action="store_true",
                    help="aggregate across 8 T8 pose files (_t8_00..07)")
    ap.add_argument("--pose", type=int, default=None,
                    help="If set (0..7), analyze only that single T8 pose file")
    ap.add_argument("--n-bins", type=int, default=360)
    ap.add_argument("--fwhm-min", type=float, default=0.0)
    ap.add_argument("--fwhm-max", type=float, default=9.0)
    ap.add_argument("--img-nx", type=int, default=200)
    ap.add_argument("--img-ny", type=int, default=200)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--input-dir", type=str, default=os.path.join(here, "data"),
                    help="directory containing beams_*.hdf5 inputs and where asci_histogram_*.hdf5 is written (default: ./data)")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="extra suffix added to output filename")
    args = ap.parse_args()

    layout_idx = args.layout_idx
    input_dir = args.input_dir

    n_bins = args.n_bins
    angular_bin_boundaries = torch.arange(n_bins + 1) / 180.0 * torch.pi

    FWHM_MIN_MM = args.fwhm_min
    FWHM_MAX_MM = args.fwhm_max

    IMG_NX, IMG_NY = args.img_nx, args.img_ny
    n_pix = IMG_NX * IMG_NY

    # Decide which “system matrices” to aggregate
    if not args.t8:
        pose_tags = [None]   # base only (single file)
    else:
        if args.pose is not None:
            if not (0 <= args.pose <= 7):
                raise ValueError("--pose must be 0..7")
            pose_tags = [args.pose]
        else:
            pose_tags = list(range(8))  # T8 poses

    asci_histogram_total = torch.zeros((n_pix, n_bins), dtype=torch.int32)

    total_valid_beams_used = 0
    total_missing_pairs = 0

    for pose in pose_tags:
        if pose is None:
            props_file = os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5")
            masks_file = os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5")
            pose_label = "base"
        else:
            props_file = os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:02d}_t8_{pose:02d}.hdf5")
            masks_file = os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}_t8_{pose:02d}.hdf5")
            pose_label = f"t8_{pose:02d}"

        if not os.path.exists(props_file) or not os.path.exists(masks_file):
            print(f"[WARN] Missing files for {pose_label}.")
            print(f"       props: {props_file} exists={os.path.exists(props_file)}")
            print(f"       masks: {masks_file} exists={os.path.exists(masks_file)}")
            total_missing_pairs += 1
            continue

        print(f"\n--- Processing {pose_label} ---")
        print(f"  props: {props_file}")
        print(f"  masks: {masks_file}")

        layout_beams_properties, header = load_beam_properties(props_file)
        beams_masks = load_beam_masks(masks_file)

        # ---- Find relevant columns robustly via header ----
        try:
            angle_col = get_col_idx(header, "angle (rad)")
        except Exception:
            angle_col = 3  # fallback

        try:
            fwhm_col = get_col_idx(header, "FWHM (mm)")
        except Exception:
            fwhm_col = 4  # fallback

        sens_col = None
        for cand in [
            "absolute sensitivity", "absolute sensitivity (a.u.)", "absolute_sensitivity",
            "relative sensitivity", "relative_sensitivity", "sensitivity"
        ]:
            if cand in header:
                sens_col = get_col_idx(header, cand)
                break
        if sens_col is None:
            sens_col = 7  # fallback

        # ---- Digitize angles into bins ----
        digitized_angles = torch.bucketize(
            layout_beams_properties[:, angle_col], angular_bin_boundaries, right=False
        )
        layout_beams_properties = torch.cat(
            (layout_beams_properties, (digitized_angles - 1).unsqueeze(1).float()),
            dim=1,
        )

        # ---- Filter NaN angles ----
        valid_angle_mask = ~torch.isnan(layout_beams_properties[:, angle_col])
        props_f = layout_beams_properties[valid_angle_mask]

        # ---- Filter by FWHM window ----
        fwhm_values = props_f[:, fwhm_col]
        fwhm_mask = (fwhm_values >= FWHM_MIN_MM) & (fwhm_values <= FWHM_MAX_MM)
        props_f = props_f[fwhm_mask]

        print(f"  Beams after angle+FWHM filtering: {props_f.shape[0]}")

        # ---- Optional sensitivity threshold (same behavior as your current code) ----
        if props_f.shape[0] > 0:
            beams_sensitivity_max = props_f[:, sens_col].max()
            props_f = props_f[props_f[:, sens_col] > beams_sensitivity_max * 0.01]
        else:
            print("  [WARN] No beams left after FWHM filter; skipping sensitivity filter.")

        print(f"  Beams after sensitivity filter: {props_f.shape[0]}")

        # ---- Accumulate histogram for this pose ----
        asci_histogram_pose = torch.zeros((n_pix, n_bins), dtype=torch.int32)

        # Assumption same as your original code:
        # props_f columns [1]=detector_idx, [2]=beam_idx
        det_col = 1
        beam_col = 2

        for beam_props in props_f:
            detector_idx = int(beam_props[det_col].item())
            beam_idx = int(beam_props[beam_col].item())
            angle_bin_idx = int(beam_props[-1].item())  # last column

            if not (0 <= angle_bin_idx < n_bins):
                continue
            if detector_idx < 0 or detector_idx >= beams_masks.shape[0]:
                continue

            asci_histogram_pose[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

        asci_histogram_total += asci_histogram_pose
        total_valid_beams_used += int(props_f.shape[0])
        print(f"  Added {pose_label} histogram into total.")

    if total_missing_pairs == len(pose_tags):
        print("\n[ERROR] No valid (properties, masks) pairs were found. Nothing to save.")
        sys.exit(1)

    # ---- Save output ----
    if args.t8:
        out_name = f"asci_histogram_{layout_idx:02d}_t8_agg.hdf5"
    else:
        out_name = f"asci_histogram_{layout_idx:02d}.hdf5"

    if args.out_suffix:
        out_name = out_name.replace(".hdf5", f"_{args.out_suffix}.hdf5")

    out_path = os.path.join(input_dir, out_name)
    print(f"\nSaving combined histogram to: {out_path}")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram_total.numpy())
        f.attrs["layout_idx"] = int(layout_idx)
        f.attrs["t8_aggregated"] = int(args.t8)
        f.attrs["n_bins"] = int(n_bins)
        f.attrs["fwhm_min_mm"] = float(FWHM_MIN_MM)
        f.attrs["fwhm_max_mm"] = float(FWHM_MAX_MM)
        f.attrs["img_nx"] = int(IMG_NX)
        f.attrs["img_ny"] = int(IMG_NY)
        f.attrs["total_valid_beams_used"] = int(total_valid_beams_used)

    print(f"[DONE] total_valid_beams_used = {total_valid_beams_used}")
    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
