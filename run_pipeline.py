#!/usr/bin/env python3
"""Run the ALO SPECT workflow end-to-end.

This file is intentionally an orchestrator only. It calls the existing stage
scripts in order and keeps their inputs/outputs inside one run directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


def parse_layout_idxs(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise argparse.ArgumentTypeError(f"Invalid layout range: {item}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("At least one layout index is required")
    return sorted(set(values))


def parse_iter_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_t8_poses(value: str) -> list[int]:
    poses: list[int] = []
    seen: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            pose_idx = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid T8 pose index: {item}") from exc
        if not 0 <= pose_idx <= 7:
            raise argparse.ArgumentTypeError(f"T8 pose index must be in [0, 7], got {pose_idx}")
        if pose_idx in seen:
            raise argparse.ArgumentTypeError(f"Duplicate T8 pose index: {pose_idx}")
        poses.append(pose_idx)
        seen.add(pose_idx)
    if not poses:
        raise argparse.ArgumentTypeError("At least one T8 pose index is required")
    return poses


def default_cpus() -> int:
    raw_value = os.environ.get("SLURM_CPUS_PER_TASK")
    if raw_value:
        try:
            return max(1, int(raw_value))
        except ValueError:
            pass
    return 16


def stringify_config(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [stringify_config(item) for item in value]
    if isinstance(value, dict):
        return {key: stringify_config(item) for key, item in value.items()}
    return value


def add_if_set(cmd_args: list[object], flag: str, value: object | None) -> None:
    """Forward output-affecting stage options only when explicitly provided."""
    if value is not None:
        cmd_args.extend([flag, value])


RESUME_CONFIG_KEYS = [
    "layout_idxs",
    "t8_poses",
    "layout_file",
    "aperture_diam",
    "detector_radial_shift_mm",
    "n_apertures",
    "n_crystals_ring1",
    "n_crystals_ring2",
    "scint_radial_mm",
    "ring_thickness",
    "a_mm",
    "b_mm",
    "phase_deg",
    "srm_row_fraction",
    "srm_row_mode",
    "srm_row_seed",
    "n_bins",
    "fwhm_min",
    "fwhm_max",
    "img_nx",
    "img_ny",
]


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_dir = Path(__file__).resolve().parent
        self.run_dir = (self.repo_dir / args.runs_dir / args.run_name).resolve()
        self.data_dir = self.run_dir / "data"
        self.plots_dir = self.run_dir / "plots"
        self.results_dir = self.run_dir / "results"
        self.recon_dir = self.run_dir / "recon"
        self.logs_dir = self.run_dir / "logs"
        self.stage_counter = 0
        self.layout_file: Path | None = None

    def command(self, script_name: str, *args: object) -> list[str]:
        return [sys.executable, str(self.repo_dir / script_name), *[str(arg) for arg in args]]

    def command_with_pairs(self, script_name: str, args: Iterable[object]) -> list[str]:
        return [sys.executable, str(self.repo_dir / script_name), *[str(arg) for arg in args]]

    def prepare_run_dir(self) -> None:
        dirs = [self.data_dir, self.plots_dir, self.results_dir, self.recon_dir, self.logs_dir]
        if self.args.dry_run:
            print("Dry run: would create run directories:")
            for path in dirs:
                print(f"  {path}")
            return

        if self.run_dir.exists() and not self.args.resume and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists and is not empty: {self.run_dir}. "
                "Use --resume only when its existing files match this run configuration, or choose a new --run-name."
            )

        for path in dirs:
            path.mkdir(parents=True, exist_ok=True)

        config_path = self.results_dir / "pipeline_config.json"
        if self.args.resume and config_path.exists():
            self.validate_resume_config(config_path)
        elif self.args.resume and self.run_dir.exists() and self.has_resume_outputs():
            raise FileNotFoundError(
                f"Refusing --resume without {config_path}. "
                "Use a new --run-name so reduced-T8 outputs cannot reuse stale analysis files."
            )

        config = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "repo_dir": str(self.repo_dir),
            "run_dir": str(self.run_dir),
            "data_dir": str(self.data_dir),
            "plots_dir": str(self.plots_dir),
            "results_dir": str(self.results_dir),
            "recon_dir": str(self.recon_dir),
            "args": stringify_config(vars(self.args)),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    def validate_resume_config(self, config_path: Path) -> None:
        try:
            previous = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Cannot resume with unreadable pipeline config {config_path}: {exc}") from exc

        previous_args = previous.get("args", {})
        current_args = stringify_config(vars(self.args))
        mismatches: list[str] = []
        for key in RESUME_CONFIG_KEYS:
            if previous_args.get(key) != current_args.get(key):
                mismatches.append(
                    f"{key}: existing={previous_args.get(key)!r}, requested={current_args.get(key)!r}"
                )
        if mismatches:
            preview = "\n".join(mismatches)
            raise ValueError(
                f"Refusing --resume because output-affecting options differ in {config_path}:\n{preview}"
            )

    def has_resume_outputs(self) -> bool:
        output_patterns = [
            "scanner_layouts_*.tensor",
            "position_*_ppdfs_t8_*.hdf5",
            "beams_masks_configuration_*.hdf5",
            "beams_properties_configuration_*.hdf5",
            "asci_histogram_*.hdf5",
        ]
        if (self.results_dir / "ji_metrics.csv").exists():
            return True
        return any(
            next(self.data_dir.glob(pattern), None) is not None
            for pattern in output_patterns
        )

    def run_cmd(self, label: str, cmd: list[str], cwd: Path | None = None) -> None:
        self.stage_counter += 1
        slug = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
        display_cmd = shlex.join(cmd)
        cwd = cwd or self.repo_dir

        if self.args.dry_run:
            print(f"[dry-run] {label}")
            print(f"  cwd: {cwd}")
            print(f"  cmd: {display_cmd}")
            return

        log_path = self.logs_dir / f"{self.stage_counter:02d}_{slug}.log"
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(self.repo_dir)
            if not existing_pythonpath
            else f"{self.repo_dir}{os.pathsep}{existing_pythonpath}"
        )
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        if self.args.torch_threads:
            env.setdefault("OMP_NUM_THREADS", str(self.args.torch_threads))
            env.setdefault("MKL_NUM_THREADS", str(self.args.torch_threads))

        print(f"\n[{self.stage_counter:02d}] {label}")
        print(f"$ {display_cmd}")
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"cwd: {cwd}\n")
            log_file.write(f"cmd: {display_cmd}\n\n")
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"{label} failed with exit code {rc}. See {log_path}")

    def expect_exists(self, path: Path, label: str) -> None:
        if self.args.dry_run:
            return
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} at {path}")

    def expect_all_exist(self, paths: Iterable[Path], label: str) -> None:
        if self.args.dry_run:
            return
        missing = [path for path in paths if not path.exists()]
        if missing:
            preview = "\n".join(str(path) for path in missing[:10])
            raise FileNotFoundError(f"Missing {label} files:\n{preview}")

    def expect_glob(self, pattern: str, min_count: int, label: str) -> None:
        if self.args.dry_run:
            return
        matches = sorted(self.run_dir.glob(pattern))
        if len(matches) < min_count:
            raise FileNotFoundError(
                f"Expected at least {min_count} {label} files matching {pattern}; found {len(matches)}"
            )

    def existing_layout_tensors(self) -> list[Path]:
        return sorted(self.data_dir.glob("scanner_layouts_*.tensor"), key=lambda path: path.stat().st_mtime)

    def discover_layout_file(self) -> Path:
        if self.args.dry_run:
            return self.data_dir / "scanner_layouts_<generated>.tensor"
        tensors = self.existing_layout_tensors()
        if not tensors:
            raise FileNotFoundError(f"No scanner_layouts_*.tensor found in {self.data_dir}")
        return tensors[-1]

    def generated_ppdf_files(self) -> list[Path]:
        return [
            self.data_dir / f"position_{layout_idx:03d}_ppdfs_t8_{pose_idx:02d}.hdf5"
            for layout_idx in self.args.layout_idxs
            for pose_idx in self.args.t8_poses
        ]

    def unexpected_ppdf_files(self) -> list[Path]:
        selected = {
            (layout_idx, pose_idx)
            for layout_idx in self.args.layout_idxs
            for pose_idx in self.args.t8_poses
        }
        unexpected: list[Path] = []
        for path in sorted(self.data_dir.glob("position_*_ppdfs_t8_*.hdf5")):
            try:
                stem_parts = path.stem.split("_")
                layout_idx = int(stem_parts[1])
                pose_idx = int(stem_parts[-1])
            except (IndexError, ValueError):
                unexpected.append(path)
                continue
            if (layout_idx, pose_idx) not in selected:
                unexpected.append(path)
        return unexpected

    def unexpected_analysis_files(self) -> list[Path]:
        expected = {
            *self.generated_mask_files(),
            *self.generated_property_files(),
            *self.generated_asci_files(),
        }
        unexpected: list[Path] = []
        for pattern in [
            "beams_masks_configuration_*.hdf5",
            "beams_properties_configuration_*.hdf5",
            "asci_histogram_*.hdf5",
        ]:
            for path in sorted(self.data_dir.glob(pattern)):
                if path not in expected:
                    unexpected.append(path)
        return unexpected

    def ensure_no_unexpected_analysis_files(self) -> None:
        if self.args.dry_run:
            return
        unexpected = self.unexpected_analysis_files()
        if unexpected:
            preview = "\n".join(str(path) for path in unexpected[:10])
            raise FileExistsError(
                "Found beam/ASCI files outside the requested layout subset. "
                "They would contaminate downstream JI aggregation:\n"
                f"{preview}\nUse a clean --run-name or rebuild the stale run directory."
            )

    def generated_mask_files(self) -> list[Path]:
        return [
            self.data_dir / f"beams_masks_configuration_{layout_idx:03d}.hdf5"
            for layout_idx in self.args.layout_idxs
        ]

    def generated_property_files(self) -> list[Path]:
        return [
            self.data_dir / f"beams_properties_configuration_{layout_idx:03d}.hdf5"
            for layout_idx in self.args.layout_idxs
        ]

    def generated_asci_files(self) -> list[Path]:
        return [
            self.data_dir / f"asci_histogram_{layout_idx:03d}.hdf5"
            for layout_idx in self.args.layout_idxs
        ]

    def pose_indices_for_layout(self, layout_idx: int) -> list[int]:
        if self.args.dry_run:
            return []

        prefix = f"position_{layout_idx:03d}_ppdfs_t8_"
        indices: list[int] = []
        for path in sorted(self.data_dir.glob(f"{prefix}*.hdf5")):
            try:
                indices.append(int(path.stem.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                continue

        if not indices:
            raise FileNotFoundError(
                f"No T8 PPDF pose files found for layout {layout_idx:02d} in {self.data_dir}"
            )
        return sorted(set(indices))

    def stage_geometry(self) -> None:
        if self.args.layout_file is not None:
            self.layout_file = Path(self.args.layout_file).expanduser().resolve()
            if not self.layout_file.exists():
                raise FileNotFoundError(f"Requested layout tensor does not exist: {self.layout_file}")
            print(f"Skipping geometry generation; using requested layout {self.layout_file}")
            return

        if not self.args.dry_run:
            tensors = self.existing_layout_tensors()
            if tensors and self.args.resume:
                self.layout_file = tensors[-1]
                print(f"Skipping geometry generation; using existing layout {self.layout_file}")
                return
            if tensors and not self.args.resume:
                raise FileExistsError(
                    f"Layout tensor already exists in {self.data_dir}. Use --resume or a new --run-name."
                )

        cmd_args: list[object] = ["--output_dir", self.data_dir]
        add_if_set(cmd_args, "--aperture_diam", self.args.aperture_diam)
        cmd_args.extend(["--detector-radial-shift-mm", self.args.detector_radial_shift_mm])
        add_if_set(cmd_args, "--n_apertures", self.args.n_apertures)
        add_if_set(cmd_args, "--n-crystals-ring1", self.args.n_crystals_ring1)
        add_if_set(cmd_args, "--n-crystals-ring2", self.args.n_crystals_ring2)
        add_if_set(cmd_args, "--scint_radial_mm", self.args.scint_radial_mm)
        add_if_set(cmd_args, "--ring_thickness", self.args.ring_thickness)
        cmd = self.command_with_pairs("generate_mph_scanner_circularfov.py", cmd_args)
        self.run_cmd("geometry generation", cmd, cwd=self.run_dir)
        self.layout_file = self.discover_layout_file()
        self.move_geometry_plot()
        self.expect_exists(self.layout_file, "layout tensor")

    def move_geometry_plot(self) -> None:
        source = self.run_dir / "scspect_hr_stationary.png"
        target = self.plots_dir / "scspect_hr_stationary.png"
        if self.args.dry_run:
            print(f"[dry-run] geometry plot would be moved to {target}")
            return
        if source.exists():
            source.replace(target)

    def stage_ppdf(self) -> None:
        unexpected = [] if self.args.dry_run else self.unexpected_ppdf_files()
        if unexpected:
            preview = "\n".join(str(path) for path in unexpected[:10])
            raise FileExistsError(
                "Found T8 PPDF pose files outside the requested --t8-poses subset. "
                "They would contaminate T8 aggregation:\n"
                f"{preview}\nUse a clean --run-name or remove/rebuild the stale run directory."
            )

        if not self.args.dry_run and not self.args.resume:
            existing = [
                path
                for layout_idx in self.args.layout_idxs
                for path in sorted(self.data_dir.glob(f"position_{layout_idx:03d}_ppdfs_t8_*.hdf5"))
            ]
            if existing:
                raise FileExistsError(
                    f"PPDF output already exists: {existing[0]}. Use --resume or a new --run-name."
                )

        for layout_idx in self.args.layout_idxs:
            cmd_args: list[object] = [
                layout_idx,
                "--layout_file",
                self.layout_file,
                "--output_dir",
                self.data_dir,
                "--pose-idxs",
                ",".join(str(pose_idx) for pose_idx in self.args.t8_poses),
                "--pose-workers",
                self.args.pose_workers,
            ]
            add_if_set(cmd_args, "--a_mm", self.args.a_mm)
            add_if_set(cmd_args, "--b_mm", self.args.b_mm)
            add_if_set(cmd_args, "--phase_deg", self.args.phase_deg)
            add_if_set(cmd_args, "--torch-threads", self.args.torch_threads)
            add_if_set(cmd_args, "--torch-interop-threads", self.args.torch_interop_threads)
            cmd_args.extend([
                "--srm-row-fraction",
                self.args.srm_row_fraction,
                "--srm-row-mode",
                self.args.srm_row_mode,
                "--srm-row-seed",
                self.args.srm_row_seed,
            ])
            if self.args.resume:
                cmd_args.append("--skip-existing")
            self.run_cmd(
                f"PPDF layout {layout_idx:02d} poses {','.join(f'{pose_idx:02d}' for pose_idx in self.args.t8_poses)}",
                self.command_with_pairs("arg_ppdf_t8.py", cmd_args),
            )
        self.expect_all_exist(self.generated_ppdf_files(), "selected T8 PPDF")

    def stage_masks(self) -> None:
        self.ensure_no_unexpected_analysis_files()
        for layout_idx in self.args.layout_idxs:
            out_path = self.data_dir / f"beams_masks_configuration_{layout_idx:03d}.hdf5"
            if not self.args.dry_run and out_path.exists():
                if self.args.resume:
                    print(f"Skipping beam masks layout {layout_idx:03d}; output exists")
                    continue
                raise FileExistsError(f"Beam mask output already exists: {out_path}")
            cmd = self.command_with_pairs(
                "arg_extract_beam_masks.py",
                [
                    layout_idx,
                    "--data-dir",
                    self.data_dir,
                    "--layout-file",
                    self.layout_file,
                    "--t8",
                ],
            )
            self.run_cmd(f"beam masks layout {layout_idx:03d}", cmd)
        self.expect_all_exist(self.generated_mask_files(), "beam mask")

    def stage_properties(self) -> None:
        self.ensure_no_unexpected_analysis_files()
        for layout_idx in self.args.layout_idxs:
            out_path = self.data_dir / f"beams_properties_configuration_{layout_idx:03d}.hdf5"
            if not self.args.dry_run and out_path.exists():
                if self.args.resume:
                    print(f"Skipping beam properties layout {layout_idx:03d}; output exists")
                    continue
                raise FileExistsError(f"Beam property output already exists: {out_path}")
            cmd = self.command_with_pairs(
                "arg_extract_beam_properties.py",
                [
                    layout_idx,
                    "--data-dir",
                    self.data_dir,
                    "--layout-file",
                    self.layout_file,
                    "--t8",
                ],
            )
            self.run_cmd(f"beam properties layout {layout_idx:03d}", cmd)
        self.expect_all_exist(self.generated_property_files(), "beam property")

    def stage_asci(self) -> None:
        self.ensure_no_unexpected_analysis_files()
        for layout_idx in self.args.layout_idxs:
            out_path = self.data_dir / f"asci_histogram_{layout_idx:03d}.hdf5"
            if not self.args.dry_run and out_path.exists():
                if self.args.resume:
                    print(f"Skipping ASCI analysis layout {layout_idx:03d}; output exists")
                    continue
                raise FileExistsError(f"ASCI output already exists: {out_path}")
            cmd = self.command_with_pairs(
                "arg_analyze_extracted_properties.py",
                self.asci_args(layout_idx),
            )
            self.run_cmd(f"ASCI aggregate layout {layout_idx:03d}", cmd)
        self.expect_all_exist(self.generated_asci_files(), "ASCI aggregate")

    def asci_args(self, layout_idx: int) -> list[object]:
        cmd_args: list[object] = [layout_idx, "--input-dir", self.data_dir, "--t8"]
        add_if_set(cmd_args, "--n-bins", self.args.n_bins)
        add_if_set(cmd_args, "--fwhm-min", self.args.fwhm_min)
        add_if_set(cmd_args, "--fwhm-max", self.args.fwhm_max)
        add_if_set(cmd_args, "--img-nx", self.args.img_nx)
        add_if_set(cmd_args, "--img-ny", self.args.img_ny)
        return cmd_args

    def stage_ji(self) -> Path:
        self.ensure_no_unexpected_analysis_files()
        ji_csv = self.results_dir / "ji_metrics.csv"
        if not self.args.dry_run and ji_csv.exists():
            if self.args.resume:
                print(f"Skipping JI calculation; output exists at {ji_csv}")
                return ji_csv
            raise FileExistsError(f"JI output already exists: {ji_csv}")

        config_name = self.args.config_name or self.args.run_name
        cmd_args: list[object] = [
            "--work_dir",
            self.data_dir,
            "--out_csv",
            ji_csv,
            "--config_name",
            config_name,
        ]
        add_if_set(cmd_args, "--img-nx", self.args.img_nx)
        add_if_set(cmd_args, "--img-ny", self.args.img_ny)
        add_if_set(cmd_args, "--n-bins", self.args.n_bins)
        cmd_args.extend([
            "--ppdf-pattern",
            ",".join(str(path) for path in self.generated_ppdf_files()),
            "--prop-pattern",
            ",".join(str(path) for path in self.generated_property_files()),
            "--asci-pattern",
            ",".join(str(path) for path in self.generated_asci_files()),
        ])
        if self.args.ji_force_zero:
            cmd_args.append("--force-zero")
            add_if_set(cmd_args, "--reason", self.args.ji_zero_reason)
        cmd = self.command_with_pairs("6_calc_ji.py", cmd_args)
        self.run_cmd("JI metric calculation", cmd)
        self.expect_exists(ji_csv, "JI CSV")
        return ji_csv

    def stage_visuals(self, ji_csv: Path) -> None:
        cmd = self.command_with_pairs(
            "generate_visuals.py",
            [
                "--data-dir",
                self.data_dir,
                "--plot-dir",
                self.plots_dir,
                "--layout-file",
                self.layout_file,
                "--layout-idxs",
                ",".join(str(idx) for idx in self.args.layout_idxs),
                "--results-csv",
                ji_csv,
            ],
        )
        if self.args.no_gif:
            cmd.append("--no-gif")
        self.run_cmd("physics and summary visuals", cmd)
        self.expect_glob("plots/*.png", 1, "physics plot")

    def stage_flist(self) -> Path:
        flist = self.results_dir / "dataset_flist.csv"
        if not self.args.dry_run and flist.exists():
            if self.args.resume:
                print(f"Skipping flist generation; output exists at {flist}")
                return flist
            raise FileExistsError(f"Flist output already exists: {flist}")

        cmd = self.command_with_pairs(
            "generate_flist.py",
            self.flist_args(flist),
        )
        self.run_cmd("flist generation", cmd)
        self.expect_exists(flist, "flist")
        return flist

    def flist_args(self, flist: Path) -> list[object]:
        cmd_args: list[object] = [
            "--data-dir",
            self.data_dir,
            "--out",
            flist,
            "--layout-idxs",
            ",".join(str(idx) for idx in self.args.layout_idxs),
            "--t8-poses",
            ",".join(str(pose_idx) for pose_idx in self.args.t8_poses),
        ]
        return cmd_args

    def resolve_phantom_source(self) -> Path:
        if not self.args.phantom:
            raise ValueError("resolve_phantom_source requires an explicit --phantom path")
        return Path(self.args.phantom).expanduser().resolve()

    def resolve_phantom(self) -> Path:
        source = self.resolve_phantom_source()
        target = self.recon_dir / source.name
        if self.args.dry_run:
            print(f"[dry-run] phantom source would be {source}")
            print(f"[dry-run] projection would use run-local phantom {target}")
            return target
        if not source.exists():
            raise FileNotFoundError(
                f"Explicit phantom file does not exist: {source}"
            )
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    def stage_projection(self, flist: Path) -> Path:
        projs = self.recon_dir / "derenzo-projs_T8.npy"
        if not self.args.dry_run and projs.exists():
            if self.args.resume:
                print(f"Skipping projection; output exists at {projs}")
                return projs
            raise FileExistsError(f"Projection output already exists: {projs}")

        cmd = self.command_with_pairs(
            "projection_t8.py",
            self.projection_args(flist, projs),
        )
        self.run_cmd("projection", cmd)
        self.expect_exists(projs, "projection array")
        return projs

    def projection_args(self, flist: Path, projs: Path) -> list[object]:
        cmd_args: list[object] = [
            "--data-dir",
            self.data_dir,
            "--flist",
            flist,
            "--out",
            projs,
        ]
        if self.args.phantom:
            cmd_args.extend(["--phantom", self.resolve_phantom()])
        return cmd_args

    def stage_mlem(self, flist: Path, projs: Path) -> Path:
        recon_npz = self.recon_dir / "recon_mlem_torch_derenzo_T8_gauss.npz"
        if not self.args.dry_run and recon_npz.exists():
            if self.args.resume:
                print(f"Skipping MLEM reconstruction; output exists at {recon_npz}")
                return recon_npz
            raise FileExistsError(f"MLEM output already exists: {recon_npz}")

        cmd_args: list[object] = [
            "--data-dir",
            self.data_dir,
            "--flist",
            flist,
            "--projs",
            projs,
            "--out",
            recon_npz,
        ]
        add_if_set(cmd_args, "--iters", self.args.mlem_iters)
        add_if_set(cmd_args, "--save-every", self.args.mlem_save_every)
        add_if_set(cmd_args, "--conv-tol", self.args.conv_tol)
        add_if_set(cmd_args, "--gauss-fwhm-mm", self.args.gauss_fwhm_mm)
        add_if_set(cmd_args, "--mm-per-px", self.args.mm_per_px)
        if self.args.device:
            cmd_args.extend(["--device", self.args.device])
        if self.args.no_gauss:
            cmd_args.append("--no-gauss")
        if self.args.gauss_each_iter:
            cmd_args.append("--gauss-each-iter")

        self.run_cmd(
            "MLEM reconstruction",
            self.command_with_pairs("mlem_torch_gpf_nonmpi.py", cmd_args),
        )
        self.expect_exists(recon_npz, "reconstruction NPZ")
        return recon_npz

    def stage_view_npz(self, recon_npz: Path) -> None:
        out_dir = self.plots_dir / "recon"
        cmd_args: list[object] = ["--npz", recon_npz, "--out-dir", out_dir]
        add_if_set(cmd_args, "--mm-per-px", self.args.mm_per_px)
        add_if_set(cmd_args, "--save-every", self.args.mlem_save_every)
        if self.args.view_iters is not None:
            cmd_args.extend(["--iters", ",".join(str(idx) for idx in self.args.view_iters)])
        cmd = self.command_with_pairs("view_npz.py", cmd_args)
        if self.args.no_gif:
            cmd.append("--no-gif")
        self.run_cmd("reconstruction visuals", cmd)
        self.expect_glob("plots/recon/*.png", 1, "reconstruction plot")

    def run(self) -> None:
        self.prepare_run_dir()
        self.stage_geometry()
        self.stage_ppdf()
        self.stage_masks()
        self.stage_properties()
        self.stage_asci()
        ji_csv = self.stage_ji()
        self.stage_visuals(ji_csv)
        flist = self.stage_flist()

        if self.args.skip_recon:
            print("Skipping projection, MLEM, and NPZ viewing because --skip-recon was set")
        else:
            if self.args.srm_row_fraction < 1.0:
                print(
                    "Warning: SRM-row sampled output is a cheap-fidelity proxy and "
                    "should not be used as final reconstruction unless intentionally requested."
                )
            projs = self.stage_projection(flist)
            recon_npz = self.stage_mlem(flist, projs)
            self.stage_view_npz(recon_npz)

        if self.args.dry_run:
            print("\nDry run complete. No stage commands were executed.")
        else:
            print("\nPipeline complete")
            print(f"Run directory: {self.run_dir}")
            print(f"Data: {self.data_dir}")
            print(f"Results: {self.results_dir}")
            print(f"Plots: {self.plots_dir}")
            print(f"Recon: {self.recon_dir}")
            print(f"Logs: {self.logs_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run geometry, T8 PPDF, beam analysis, JI, visuals, projection, MLEM, and NPZ viewing."
    )
    parser.add_argument("--runs-dir", default="runs", help="Directory under the repo for run outputs.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to a timestamp.")
    parser.add_argument("--config-name", default=None, help="Configuration label written into JI results.")
    parser.add_argument(
        "--layout-idxs",
        type=parse_layout_idxs,
        required=True,
        help="Comma-separated layout indices to process. Required because layout selection changes outputs.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed outputs in the selected run directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and output layout without executing stages.")

    parser.add_argument("--aperture-diam", type=float, default=None)
    parser.add_argument(
        "--layout-file",
        type=Path,
        default=None,
        help="Use an existing scanner layout tensor instead of generating geometry.",
    )
    parser.add_argument("--detector-radial-shift-mm", type=float, default=0.0)
    parser.add_argument("--n-apertures", type=int, default=None)
    parser.add_argument("--n-crystals-ring1", "--n_crystals_ring1", dest="n_crystals_ring1", type=int, default=480)
    parser.add_argument("--n-crystals-ring2", "--n_crystals_ring2", dest="n_crystals_ring2", type=int, default=720)
    parser.add_argument("--scint-radial-mm", type=float, default=None)
    parser.add_argument("--ring-thickness", type=float, default=None)

    parser.add_argument("--a-mm", type=float, default=None, help="T8 aperture spacing a passed to arg_ppdf_t8.py.")
    parser.add_argument("--b-mm", type=float, default=None, help="T8 aperture spacing b passed to arg_ppdf_t8.py.")
    parser.add_argument("--phase-deg", type=float, default=None)
    parser.add_argument(
        "--t8-poses",
        type=parse_t8_poses,
        default=parse_t8_poses("0,1,2,3,4,5,6,7"),
        help="Comma-separated T8 pose indices to generate/use, e.g. 0,2,4,6. Defaults to all 8 poses.",
    )
    parser.add_argument("--cpus", type=int, default=default_cpus())
    parser.add_argument("--pose-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--srm-row-fraction", type=float, default=1.0)
    parser.add_argument("--srm-row-mode", type=str, default="ring_cell_random")
    parser.add_argument("--srm-row-seed", type=int, default=42)

    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--fwhm-min", type=float, default=None)
    parser.add_argument("--fwhm-max", type=float, default=None)
    parser.add_argument("--img-nx", type=int, default=None)
    parser.add_argument("--img-ny", type=int, default=None)
    parser.add_argument("--ji-force-zero", action="store_true")
    parser.add_argument("--ji-zero-reason", default=None)

    parser.add_argument("--skip-recon", action="store_true", help="Stop after flist generation.")
    parser.add_argument("--phantom", default=None, help="Explicit phantom .pt file for projection.")
    parser.add_argument("--mlem-iters", type=int, default=None)
    parser.add_argument("--mlem-save-every", type=int, default=None)
    parser.add_argument("--conv-tol", type=float, default=None)
    parser.add_argument("--gauss-fwhm-mm", type=float, default=None)
    parser.add_argument("--mm-per-px", type=float, default=None)
    parser.add_argument("--device", default=None, help="Optional torch device passed to MLEM.")
    parser.add_argument("--no-gauss", action="store_true")
    parser.add_argument("--gauss-each-iter", action="store_true")
    parser.add_argument("--view-iters", type=parse_iter_list, default=None)
    parser.add_argument("--no-gif", action="store_true")
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    args.cpus = max(1, args.cpus)
    selected_pose_count = max(1, len(args.t8_poses))
    if args.pose_workers is None:
        args.pose_workers = min(selected_pose_count, max(1, args.cpus // 4))
    else:
        args.pose_workers = max(1, args.pose_workers)
    args.pose_workers = min(args.pose_workers, selected_pose_count, 8)
    if args.torch_threads is None:
        args.torch_threads = max(1, args.cpus // args.pose_workers)
    else:
        args.torch_threads = max(1, args.torch_threads)
    if args.torch_interop_threads is not None:
        args.torch_interop_threads = max(1, args.torch_interop_threads)
    return args


def main() -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args())
    try:
        Pipeline(args).run()
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
