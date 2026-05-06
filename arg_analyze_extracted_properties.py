import argparse
import os

import h5py
import torch


def _decode_header(raw_header):
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in raw_header
    ]


def _normalize_column_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def load_beam_properties(path: str):
    with h5py.File(path, "r") as f:
        if "beam_properties" not in f:
            raise KeyError(f"Dataset 'beam_properties' not found in {path}")
        dataset = f["beam_properties"]
        if "Header" not in dataset.attrs:
            raise KeyError(f"Attribute 'Header' not found on dataset 'beam_properties' in {path}")
        props = torch.from_numpy(dataset[:])
        header = _decode_header(dataset.attrs["Header"])
    return props, header


def load_beam_masks(path: str):
    with h5py.File(path, "r") as f:
        if "beam_mask" not in f:
            raise KeyError(f"Dataset 'beam_mask' not found in {path}")
        masks = torch.from_numpy(f["beam_mask"][:])
    return masks


def get_required_col_idx(header, required_name: str, file_path: str) -> int:
    normalized_required = _normalize_column_name(required_name)
    normalized_header = [_normalize_column_name(name) for name in header]
    if normalized_required not in normalized_header:
        raise RuntimeError(
            f"Required column {required_name!r} not found in {file_path}. "
            f"Available columns: {header}"
        )
    return normalized_header.index(normalized_required)


def get_layout_files(input_dir: str, layout_idx: int, t8: bool):
    if t8:
        return (
            os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:03d}.hdf5"),
            os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:03d}.hdf5"),
            os.path.join(input_dir, f"asci_histogram_{layout_idx:03d}.hdf5"),
        )

    return (
        os.path.join(input_dir, f"beams_properties_configuration_{layout_idx:02d}.hdf5"),
        os.path.join(input_dir, f"beams_masks_configuration_{layout_idx:02d}.hdf5"),
        os.path.join(input_dir, f"asci_histogram_{layout_idx:02d}.hdf5"),
    )


def build_asci_histogram(
    layout_beams_properties: torch.Tensor,
    header,
    beams_masks: torch.Tensor,
    props_file: str,
    n_bins: int,
    n_pix: int,
    fwhm_min_mm: float,
    fwhm_max_mm: float,
):
    angle_col = get_required_col_idx(header, "Angle (rad)", props_file)
    fwhm_col = get_required_col_idx(header, "FWHM (mm)", props_file)
    sensitivity_col = get_required_col_idx(header, "sensitivity", props_file)
    detector_col = get_required_col_idx(header, "detector unit id", props_file)
    beam_col = get_required_col_idx(header, "beam id", props_file)

    if beams_masks.ndim != 2:
        raise ValueError(f"Expected beam_mask dataset to be 2-D, got shape {tuple(beams_masks.shape)}")
    if beams_masks.shape[1] != n_pix:
        raise ValueError(
            f"Beam mask pixel count mismatch: expected {n_pix}, got {beams_masks.shape[1]}"
        )

    angle_values = layout_beams_properties[:, angle_col]
    fwhm_values = layout_beams_properties[:, fwhm_col]
    sensitivity_values = layout_beams_properties[:, sensitivity_col]

    finite_mask = (
        torch.isfinite(angle_values)
        & torch.isfinite(fwhm_values)
        & torch.isfinite(sensitivity_values)
    )
    fwhm_mask = (fwhm_values >= fwhm_min_mm) & (fwhm_values <= fwhm_max_mm)
    props_f = layout_beams_properties[finite_mask & fwhm_mask]

    print(f"  Beams after finite+FWHM filtering: {props_f.shape[0]}")

    if props_f.shape[0] > 0:
        sensitivity_values_f = props_f[:, sensitivity_col]
        sensitivity_max = sensitivity_values_f.max()
        props_f = props_f[sensitivity_values_f > sensitivity_max * 0.01]
    else:
        print("  [WARN] No beams left after finite+FWHM filtering; skipping sensitivity filter.")

    print(f"  Beams after sensitivity filter: {props_f.shape[0]}")

    asci_histogram = torch.zeros((n_pix, n_bins), dtype=torch.int32)
    if props_f.shape[0] == 0:
        return asci_histogram, 0

    angular_bin_boundaries = torch.arange(
        n_bins + 1,
        dtype=props_f.dtype,
        device=props_f.device,
    ) / 180.0 * torch.pi
    angle_bin_idxs = torch.bucketize(
        props_f[:, angle_col], angular_bin_boundaries, right=False
    ) - 1

    for beam_props, angle_bin_idx_value in zip(props_f, angle_bin_idxs):
        angle_bin_idx = int(angle_bin_idx_value.item())
        if not (0 <= angle_bin_idx < n_bins):
            continue

        detector_idx = int(beam_props[detector_col].item())
        beam_idx = int(beam_props[beam_col].item())
        if detector_idx < 0 or detector_idx >= beams_masks.shape[0]:
            continue

        asci_histogram[beams_masks[detector_idx] == beam_idx, angle_bin_idx] += 1

    return asci_histogram, int(props_f.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_idx", type=int, help="layout index (e.g., 0)")
    ap.add_argument(
        "--t8",
        action="store_true",
        help="use aggregate-first T8 layout-level beam property and mask files",
    )
    ap.add_argument(
        "--pose",
        type=int,
        default=None,
        help="legacy per-pose T8 analysis is not supported in aggregate-first mode",
    )
    ap.add_argument("--n-bins", type=int, default=360)
    ap.add_argument("--fwhm-min", type=float, default=0.0)
    ap.add_argument("--fwhm-max", type=float, default=9.0)
    ap.add_argument("--img-nx", type=int, default=200)
    ap.add_argument("--img-ny", type=int, default=200)
    ap.add_argument("--input-dir", type=str, default="./data")
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help="extra suffix added to output filename",
    )
    args = ap.parse_args()

    if args.t8 and args.pose is not None:
        raise ValueError("--pose is legacy per-pose T8 mode and is not supported in aggregate-first mode")
    if args.fwhm_min > args.fwhm_max:
        raise ValueError("--fwhm-min must be <= --fwhm-max")

    layout_idx = args.layout_idx
    input_dir = args.input_dir
    n_pix = args.img_nx * args.img_ny

    props_file, masks_file, out_path = get_layout_files(input_dir, layout_idx, args.t8)
    if args.out_suffix:
        out_path = out_path.replace(".hdf5", f"_{args.out_suffix}.hdf5")

    missing_files = [path for path in (props_file, masks_file) if not os.path.exists(path)]
    if missing_files:
        raise FileNotFoundError(
            "Required ASCI input file(s) missing for aggregate-first analysis: "
            + ", ".join(missing_files)
        )

    print(f"Processing layout {layout_idx}")
    print(f"  props: {props_file}")
    print(f"  masks: {masks_file}")

    layout_beams_properties, header = load_beam_properties(props_file)
    beams_masks = load_beam_masks(masks_file)

    asci_histogram, total_valid_beams_used = build_asci_histogram(
        layout_beams_properties=layout_beams_properties,
        header=header,
        beams_masks=beams_masks,
        props_file=props_file,
        n_bins=args.n_bins,
        n_pix=n_pix,
        fwhm_min_mm=args.fwhm_min,
        fwhm_max_mm=args.fwhm_max,
    )

    print(f"\nSaving histogram to: {out_path}")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("asci_histogram", data=asci_histogram.numpy())
        f.attrs["layout_idx"] = int(layout_idx)
        f.attrs["t8_aggregated"] = int(args.t8)
        f.attrs["n_bins"] = int(args.n_bins)
        f.attrs["fwhm_min_mm"] = float(args.fwhm_min)
        f.attrs["fwhm_max_mm"] = float(args.fwhm_max)
        f.attrs["img_nx"] = int(args.img_nx)
        f.attrs["img_ny"] = int(args.img_ny)
        f.attrs["total_valid_beams_used"] = int(total_valid_beams_used)
        f.attrs["beam_properties_file"] = os.path.basename(props_file)
        f.attrs["beam_masks_file"] = os.path.basename(masks_file)

    print(f"[DONE] total_valid_beams_used = {total_valid_beams_used}")
    print("[DONE] Finished.")


if __name__ == "__main__":
    main()
