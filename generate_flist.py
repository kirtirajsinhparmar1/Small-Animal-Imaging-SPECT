if __name__ == "__main__":
    import argparse
    import os
    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.abspath(os.path.join(here, "data"))

    ap = argparse.ArgumentParser(description="Generate dataset_flist.csv for T8 system matrices.")
    ap.add_argument("--data-dir", default=default_data_dir, help="Directory containing position_###_ppdfs_t8_##.hdf5 files")
    ap.add_argument("--out", default=None, help="Output flist path (default: <data-dir>/dataset_flist.csv)")
    ap.add_argument("--layout-idxs", default="0,1", help="Comma-separated layout indices (default: 0,1)")
    ap.add_argument("--n-poses", type=int, default=8, help="Number of T8 poses per layout (default: 8)")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_path = args.out or os.path.join(data_dir, "dataset_flist.csv")

    layout_idxs = [int(x) for x in args.layout_idxs.split(",") if x.strip() != ""]
    pose_idxs = np.arange(args.n_poses)
    fnames = [f"position_{layout:03d}_ppdfs_t8_{pose:02d}.hdf5" for layout in layout_idxs for pose in pose_idxs]

    with open(out_path, "w") as f:
        for fname in fnames:
            f.write(os.path.join(data_dir, fname) + "\n")

    print(f"Wrote {len(fnames)} entries to: {out_path}")
