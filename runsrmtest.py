#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FRACTIONS = "0.10,0.15,0.25,0.40,0.50,0.60"
VALID_LABELS = ("baseline", "srm10", "srm15", "srm25", "srm40", "srm50", "srm60")
METRIC_COLUMNS = [
    "fwhm_mean",
    "sensitivity_total",
    "sensitivity_mean",
    "asci_pct",
    "JI",
]
JI_COLUMNS = [
    "metric_fidelity",
    "srm_row_sampled",
    "srm_row_fraction",
    "srm_row_mode",
    "srm_row_seed",
    "srm_total_rows",
    "srm_active_rows",
    "srm_sampled_rows",
    "srm_sampled_fraction_actual",
    "n_crystals_ring1",
    "n_crystals_ring2",
    "config",
    "work_dir",
]
RECON_STAT_COLUMNS = [
    "recon_npz_path",
    "recon_status",
    "n_recon_frames",
    "recon_shape",
    "final_recon_min",
    "final_recon_max",
    "final_recon_mean",
    "final_recon_sum",
]
IMAGE_COMPARE_COLUMNS = [
    "recon_rmse_vs_srm100",
    "recon_nrmse_vs_srm100",
    "recon_mae_vs_srm100",
    "recon_corr_vs_srm100",
    "recon_rel_l2_error_vs_srm100",
    "recon_max_abs_diff_vs_srm100",
    "recon_norm_rmse_vs_srm100",
    "recon_norm_nrmse_vs_srm100",
    "recon_norm_mae_vs_srm100",
    "recon_norm_corr_vs_srm100",
    "recon_norm_rel_l2_error_vs_srm100",
    "recon_norm_max_abs_diff_vs_srm100",
]
PHANTOM_COMPARE_COLUMNS = [
    "phantom_rmse",
    "phantom_nrmse",
    "phantom_mae",
    "phantom_corr",
    "phantom_rel_l2_error",
]
EPS = 1e-12


def parse_csv_floats(value: str) -> list[float]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def parse_only(value: str | None) -> set[str] | None:
    if not value:
        return None
    labels = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = sorted(labels.difference(VALID_LABELS))
    if unknown:
        raise ValueError(f"Unknown --only labels: {', '.join(unknown)}")
    return labels


def fraction_label(fraction: float) -> str:
    return f"srm{int(round(fraction * 100.0))}"


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def rel_diff_pct(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or abs(baseline) <= EPS:
        return None
    return 100.0 * (value - baseline) / baseline


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def tail_text(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def read_last_ji_row(run_dir: Path) -> dict[str, str]:
    path = run_dir / "results" / "ji_metrics.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else {}


def find_recon_npz(run_dir: Path) -> tuple[Path | None, str]:
    roots = [run_dir / "recon", run_dir / "data", run_dir / "results"]
    candidates: list[tuple[int, int, Path]] = []
    for root_order, root in enumerate(roots):
        if not root.exists():
            continue
        for path in root.rglob("*.npz"):
            name_score = 0 if "filtered" in path.name.lower() else 1
            try:
                with np.load(path, allow_pickle=True) as data:
                    has_estimates = "estimates" in data.files
            except Exception:
                has_estimates = False
            key_score = 0 if has_estimates else 1
            candidates.append((root_order * 10 + key_score, name_score, path))

    if not candidates:
        return None, "missing_recon_npz"
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return candidates[0][2], "success"


def load_final_recon(run_dir: Path) -> tuple[np.ndarray | None, dict[str, Any]]:
    npz_path, status = find_recon_npz(run_dir)
    stats: dict[str, Any] = {
        "recon_npz_path": str(npz_path) if npz_path else "",
        "recon_status": status,
        "n_recon_frames": "",
        "recon_shape": "",
        "final_recon_min": "",
        "final_recon_max": "",
        "final_recon_mean": "",
        "final_recon_sum": "",
    }
    if npz_path is None:
        return None, stats

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "estimates" in data.files and len(data["estimates"]) > 0:
                estimates = data["estimates"]
                final = np.asarray(estimates[-1], dtype=np.float64)
                stats["n_recon_frames"] = int(len(estimates))
            elif "final_gauss" in data.files:
                final = np.asarray(data["final_gauss"], dtype=np.float64)
                stats["n_recon_frames"] = 1
            elif "final" in data.files:
                final = np.asarray(data["final"], dtype=np.float64)
                stats["n_recon_frames"] = 1
            else:
                stats["recon_status"] = "missing_estimates_key"
                return None, stats
    except Exception as exc:
        stats["recon_status"] = "failed_load_recon_npz"
        stats["error_message"] = str(exc)
        return None, stats

    stats["recon_shape"] = "x".join(str(dim) for dim in final.shape)
    stats["final_recon_min"] = float(np.min(final))
    stats["final_recon_max"] = float(np.max(final))
    stats["final_recon_mean"] = float(np.mean(final))
    stats["final_recon_sum"] = float(np.sum(final))
    return final, stats


def load_phantom(path: str | None, img_size: int = 200) -> np.ndarray | None:
    if not path:
        return None
    import torch

    data = torch.load(path, map_location="cpu")
    phantom = data["Phantom tensor"].detach().cpu()
    h, w = phantom.shape
    if h > img_size or w > img_size:
        raise ValueError(f"Phantom shape {tuple(phantom.shape)} is larger than {img_size}x{img_size}")
    pad_h_before = (img_size - h) // 2
    pad_h_after = img_size - h - pad_h_before
    pad_w_before = (img_size - w) // 2
    pad_w_after = img_size - w - pad_w_before
    padded = torch.nn.functional.pad(
        phantom,
        (pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        "constant",
        0,
    )
    return padded.numpy().astype(np.float64, copy=False)


def image_metrics(a: np.ndarray | None, b: np.ndarray | None, prefix: str) -> dict[str, Any]:
    names = {
        "rmse": f"{prefix}_rmse",
        "nrmse": f"{prefix}_nrmse",
        "mae": f"{prefix}_mae",
        "corr": f"{prefix}_corr",
        "rel_l2": f"{prefix}_rel_l2_error",
        "max_abs_diff": f"{prefix}_max_abs_diff",
    }
    result = {name: "" for name in names.values()}
    if a is None or b is None or a.shape != b.shape:
        return result

    af = np.asarray(a, dtype=np.float64).ravel()
    bf = np.asarray(b, dtype=np.float64).ravel()
    diff = af - bf
    rmse = float(np.sqrt(np.mean(diff * diff)))
    result[names["rmse"]] = rmse
    result[names["nrmse"]] = float(rmse / (np.max(bf) - np.min(bf) + EPS))
    result[names["mae"]] = float(np.mean(np.abs(diff)))
    if np.std(af) <= EPS or np.std(bf) <= EPS:
        result[names["corr"]] = ""
    else:
        result[names["corr"]] = float(np.corrcoef(af, bf)[0, 1])
    result[names["rel_l2"]] = float(np.linalg.norm(diff) / (np.linalg.norm(bf) + EPS))
    result[names["max_abs_diff"]] = float(np.max(np.abs(diff)))
    return result


def normalized_image(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    denom = max(abs(float(np.sum(image))), EPS)
    return image / denom


def classify_metric(row: dict[str, Any]) -> str:
    ji = safe_float(row.get("JI_abs_rel_error_pct"))
    fwhm = safe_float(row.get("fwhm_mean_abs_rel_error_pct"))
    asci = safe_float(row.get("asci_pct_abs_rel_error_pct"))
    sens = safe_float(row.get("sensitivity_total_abs_rel_error_pct"))
    if None in (ji, fwhm, asci, sens):
        return "Unknown"
    if ji <= 3 and fwhm <= 2 and asci <= 2 and sens <= 1:
        return "Excellent"
    if ji <= 5 and fwhm <= 3 and asci <= 5 and sens <= 2:
        return "Good"
    return "Risky"


def classify_recon(row: dict[str, Any]) -> str:
    nrmse = safe_float(row.get("recon_norm_nrmse_vs_srm100"))
    corr = safe_float(row.get("recon_norm_corr_vs_srm100"))
    if nrmse is None or corr is None:
        return "Unknown"
    if nrmse <= 0.03 and corr >= 0.98:
        return "Excellent"
    if nrmse <= 0.05 and corr >= 0.95:
        return "Good"
    return "Risky"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SRM fraction reconstruction validation sweep.")
    parser.add_argument("--out-dir", default="runs/srm_recon_fraction_test")
    parser.add_argument("--aperture-diam", type=float, default=0.4)
    parser.add_argument("--n-apertures", type=int, default=180)
    parser.add_argument("--n-crystals-ring1", type=int, default=480)
    parser.add_argument("--n-crystals-ring2", type=int, default=720)
    parser.add_argument("--fractions", default=DEFAULT_FRACTIONS)
    parser.add_argument("--baseline-fraction", type=float, default=1.0)
    parser.add_argument("--srm-row-mode", default="evenly_spaced")
    parser.add_argument("--srm-row-seed", type=int, default=42)
    parser.add_argument("--layout-idxs", default="0,1")
    parser.add_argument("--t8-poses", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--phantom", default=None)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--pose-workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-baseline", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


class SweepRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_dir = Path(args.out_dir)
        self.summary_dir = self.out_dir / "summary"
        self.raw_csv = self.summary_dir / "srm_recon_fraction_raw.csv"
        self.comparison_csv = self.summary_dir / "srm_recon_fraction_comparison.csv"
        self.summary_md = self.summary_dir / "srm_recon_fraction_summary.md"
        self.commands_sh = self.summary_dir / "srm_recon_fraction_commands.sh"
        self.state_json = self.summary_dir / "srm_recon_fraction_state.json"
        self.raw_rows: list[dict[str, Any]] = []
        self.comparison_rows: list[dict[str, Any]] = []
        self.commands: list[str] = []
        self.completed: list[str] = []
        self.failed: dict[str, str] = {}
        self.recons: dict[str, np.ndarray | None] = {}
        self.phantom = load_phantom(args.phantom) if args.phantom else None

    def selected_specs(self) -> list[dict[str, Any]]:
        fractions = parse_csv_floats(self.args.fractions)
        selected = parse_only(self.args.only)
        specs = [
            {
                "label": "baseline",
                "fraction": float(self.args.baseline_fraction),
                "run_name": "evals/baseline_srm100_recon",
                "is_baseline": True,
            }
        ]
        for fraction in fractions:
            label = fraction_label(fraction)
            specs.append(
                {
                    "label": label,
                    "fraction": float(fraction),
                    "run_name": f"evals/{label}_recon",
                    "is_baseline": False,
                }
            )
        if selected is None:
            return specs
        need_baseline = "baseline" in selected or any(label != "baseline" for label in selected)
        filtered = []
        for spec in specs:
            if spec["label"] == "baseline" and need_baseline:
                filtered.append(spec)
            elif spec["label"] in selected:
                filtered.append(spec)
        return filtered

    def build_command(self, spec: dict[str, Any]) -> list[str]:
        cmd = [
            self.args.python,
            "run_pipeline.py",
            "--runs-dir",
            str(self.out_dir),
            "--run-name",
            spec["run_name"],
            "--layout-idxs",
            self.args.layout_idxs,
            "--t8-poses",
            self.args.t8_poses,
            "--aperture-diam",
            str(self.args.aperture_diam),
            "--n-apertures",
            str(self.args.n_apertures),
            "--n-crystals-ring1",
            str(self.args.n_crystals_ring1),
            "--n-crystals-ring2",
            str(self.args.n_crystals_ring2),
            "--srm-row-fraction",
            str(spec["fraction"]),
            "--srm-row-mode",
            self.args.srm_row_mode,
            "--srm-row-seed",
            str(self.args.srm_row_seed),
            "--cpus",
            str(self.args.cpus),
            "--pose-workers",
            str(self.args.pose_workers),
            "--torch-threads",
            str(self.args.torch_threads),
            "--torch-interop-threads",
            str(self.args.torch_interop_threads),
        ]
        if self.args.phantom:
            cmd.extend(["--phantom", self.args.phantom])
        if self.args.resume:
            cmd.append("--resume")
        return cmd

    def run_command(self, spec: dict[str, Any], cmd: list[str]) -> tuple[str, float, str]:
        run_dir = self.out_dir / spec["run_name"]
        if is_nonempty_dir(run_dir) and not self.args.resume:
            return (
                "failed",
                0.0,
                "Run exists. Use --resume or choose a new --out-dir.",
            )

        log_path = self.summary_dir / f"{spec['label']}.log"
        tail = deque(maxlen=120)
        started = time.time()
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                tail.append(line)
            return_code = proc.wait()
        elapsed = time.time() - started
        if return_code != 0:
            return "failed", elapsed, tail_text("".join(tail))
        return "success", elapsed, ""

    def make_raw_row(
        self,
        spec: dict[str, Any],
        cmd: list[str],
        status: str,
        elapsed_sec: float,
        error_message: str = "",
    ) -> dict[str, Any]:
        run_dir = self.out_dir / spec["run_name"]
        ji_row = read_last_ji_row(run_dir) if status == "success" else {}
        final_recon, recon_stats = load_final_recon(run_dir) if status == "success" else (None, {})
        self.recons[spec["label"]] = final_recon

        row: dict[str, Any] = {
            "label": spec["label"],
            "fraction": spec["fraction"],
            "status": status,
            "command": shell_join(cmd),
            "elapsed_sec": elapsed_sec,
            "run_dir": str(run_dir),
            "error_message": error_message,
        }
        for name in METRIC_COLUMNS + JI_COLUMNS:
            row[name] = ji_row.get(name, "")
        for name in RECON_STAT_COLUMNS:
            row[name] = recon_stats.get(name, "")
        if self.phantom is not None and final_recon is not None:
            row.update(image_metrics(final_recon, self.phantom, "phantom"))
        else:
            for name in PHANTOM_COMPARE_COLUMNS:
                row[name] = ""
        return row

    def build_comparison_rows(self) -> None:
        baseline = next((row for row in self.raw_rows if row["label"] == "baseline" and row["status"] == "success"), None)
        baseline_recon = self.recons.get("baseline")
        self.comparison_rows = []
        if baseline is None:
            return

        baseline_elapsed = safe_float(baseline.get("elapsed_sec"))
        for row in self.raw_rows:
            if row["label"] == "baseline":
                continue
            comp: dict[str, Any] = {
                "label": row["label"],
                "fraction": row["fraction"],
                "status": row["status"],
                "elapsed_sec": row.get("elapsed_sec", ""),
                "speedup_vs_srm100": "",
                "srm_active_rows": row.get("srm_active_rows", ""),
                "srm_sampled_rows": row.get("srm_sampled_rows", ""),
                "srm_sampled_fraction_actual": row.get("srm_sampled_fraction_actual", ""),
            }
            elapsed = safe_float(row.get("elapsed_sec"))
            if baseline_elapsed is not None and elapsed is not None and elapsed > 0:
                comp["speedup_vs_srm100"] = baseline_elapsed / elapsed

            for metric in METRIC_COLUMNS:
                value = safe_float(row.get(metric))
                base_value = safe_float(baseline.get(metric))
                comp[f"{metric}_abs_diff"] = "" if value is None or base_value is None else value - base_value
                rel = rel_diff_pct(value, base_value)
                comp[f"{metric}_rel_diff_pct"] = "" if rel is None else rel
                comp[f"{metric}_abs_rel_error_pct"] = "" if rel is None else abs(rel)

            recon = self.recons.get(row["label"])
            comp.update(image_metrics(recon, baseline_recon, "recon"))
            comp.update(
                image_metrics(
                    normalized_image(recon),
                    normalized_image(baseline_recon),
                    "recon_norm",
                )
            )
            if self.phantom is not None:
                comp.update(image_metrics(recon, self.phantom, "phantom"))
            else:
                for name in PHANTOM_COMPARE_COLUMNS:
                    comp[name] = ""
            comp["metric_class"] = classify_metric(comp)
            comp["recon_class"] = classify_recon(comp)
            self.comparison_rows.append(comp)

    def write_csv(self, path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def write_commands(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        with self.commands_sh.open("w") as handle:
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
            for command in self.commands:
                handle.write(command + "\n")

    def write_state(self) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "args": vars(self.args),
            "completed_labels": self.completed,
            "failed_labels": sorted(self.failed),
            "run_dirs": {row["label"]: row["run_dir"] for row in self.raw_rows},
            "recon_paths": {row["label"]: row.get("recon_npz_path", "") for row in self.raw_rows},
            "output_paths": {
                "raw_csv": str(self.raw_csv),
                "comparison_csv": str(self.comparison_csv),
                "summary_md": str(self.summary_md),
                "commands_sh": str(self.commands_sh),
                "state_json": str(self.state_json),
            },
            "errors": self.failed,
        }
        with self.state_json.open("w") as handle:
            json.dump(state, handle, indent=2)

    def write_markdown(self) -> None:
        baseline = next((row for row in self.raw_rows if row["label"] == "baseline"), None)
        acceptable = [
            row
            for row in self.comparison_rows
            if row.get("metric_class") in ("Excellent", "Good")
            and row.get("recon_class") in ("Excellent", "Good")
        ]
        if acceptable:
            recommended = min(acceptable, key=lambda row: float(row["fraction"]))
            recommendation = (
                f"Smallest acceptable fraction: {recommended['label']} "
                f"({float(recommended['fraction']):.2f})."
            )
        else:
            recommendation = "No partial SRM fraction met both metric and reconstruction Good thresholds."

        lines = [
            "# SRM Reconstruction Fraction Sweep",
            "",
            "## Reference Geometry",
            "",
            f"- aperture_diam_mm: {self.args.aperture_diam}",
            f"- n_apertures: {self.args.n_apertures}",
            f"- n_crystals_ring1: {self.args.n_crystals_ring1}",
            f"- n_crystals_ring2: {self.args.n_crystals_ring2}",
            f"- srm_row_mode: {self.args.srm_row_mode}",
            "",
            "## Baseline SRM100",
            "",
        ]
        if baseline:
            lines.extend(
                [
                    f"- status: {baseline.get('status', '')}",
                    f"- FWHM: {fmt(baseline.get('fwhm_mean'))}",
                    f"- sensitivity_total: {fmt(baseline.get('sensitivity_total'))}",
                    f"- sensitivity_mean: {fmt(baseline.get('sensitivity_mean'))}",
                    f"- ASCI: {fmt(baseline.get('asci_pct'))}",
                    f"- JI: {fmt(baseline.get('JI'))}",
                    f"- reconstruction: {baseline.get('recon_npz_path', '')}",
                    "",
                ]
            )

        lines.extend(
            [
                "## Metric Comparison",
                "",
                "| label | sampled rows | FWHM err % | sensitivity_total err % | ASCI err % | JI err % | metric class | speedup |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in self.comparison_rows:
            lines.append(
                "| {label} | {rows} | {fwhm} | {sens} | {asci} | {ji} | {cls} | {speed} |".format(
                    label=row.get("label", ""),
                    rows=fmt(row.get("srm_sampled_rows")),
                    fwhm=fmt(row.get("fwhm_mean_abs_rel_error_pct")),
                    sens=fmt(row.get("sensitivity_total_abs_rel_error_pct")),
                    asci=fmt(row.get("asci_pct_abs_rel_error_pct")),
                    ji=fmt(row.get("JI_abs_rel_error_pct")),
                    cls=row.get("metric_class", ""),
                    speed=fmt(row.get("speedup_vs_srm100")),
                )
            )

        lines.extend(
            [
                "",
                "## Reconstruction Comparison",
                "",
                "| label | norm NRMSE | norm corr | norm rel L2 | recon class |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in self.comparison_rows:
            lines.append(
                "| {label} | {nrmse} | {corr} | {rel_l2} | {cls} |".format(
                    label=row.get("label", ""),
                    nrmse=fmt(row.get("recon_norm_nrmse_vs_srm100")),
                    corr=fmt(row.get("recon_norm_corr_vs_srm100")),
                    rel_l2=fmt(row.get("recon_norm_rel_l2_error_vs_srm100")),
                    cls=row.get("recon_class", ""),
                )
            )

        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                recommendation,
                "",
                "SRM100 remains the final scientific reporting metric. Partial SRM should only be used for cheap-fidelity screening.",
                "",
            ]
        )
        self.summary_md.write_text("\n".join(lines))

    def write_outputs(self) -> None:
        raw_fields = [
            "label",
            "fraction",
            "status",
            "command",
            "elapsed_sec",
            "run_dir",
            *METRIC_COLUMNS,
            *JI_COLUMNS,
            *RECON_STAT_COLUMNS,
            *PHANTOM_COMPARE_COLUMNS,
            "error_message",
        ]
        comparison_fields = [
            "label",
            "fraction",
            "status",
            "elapsed_sec",
            "speedup_vs_srm100",
            "srm_active_rows",
            "srm_sampled_rows",
            "srm_sampled_fraction_actual",
        ]
        for metric in METRIC_COLUMNS:
            comparison_fields.extend(
                [
                    f"{metric}_abs_diff",
                    f"{metric}_rel_diff_pct",
                    f"{metric}_abs_rel_error_pct",
                ]
            )
        comparison_fields.extend(IMAGE_COMPARE_COLUMNS)
        comparison_fields.extend(PHANTOM_COMPARE_COLUMNS)
        comparison_fields.extend(["metric_class", "recon_class"])

        self.build_comparison_rows()
        self.write_csv(self.raw_csv, self.raw_rows, raw_fields)
        self.write_csv(self.comparison_csv, self.comparison_rows, comparison_fields)
        self.write_commands()
        self.write_state()
        self.write_markdown()

    def dry_run(self, specs: list[dict[str, Any]]) -> None:
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            cmd = self.build_command(spec)
            command = shell_join(cmd)
            self.commands.append(command)
            print(command)
        self.write_commands()
        self.write_state()

    def run(self) -> None:
        specs = self.selected_specs()
        if not specs:
            raise RuntimeError("No SRM labels selected.")

        self.summary_dir.mkdir(parents=True, exist_ok=True)
        if self.args.dry_run:
            self.dry_run(specs)
            return

        for spec in specs:
            if self.args.stop_after_baseline and not spec["is_baseline"]:
                break

            cmd = self.build_command(spec)
            self.commands.append(shell_join(cmd))
            print(f"\n=== Running {spec['label']} ({spec['fraction']}) ===")
            status, elapsed_sec, error_message = self.run_command(spec, cmd)
            row = self.make_raw_row(spec, cmd, status, elapsed_sec, error_message)
            self.raw_rows.append(row)

            if status == "success":
                self.completed.append(spec["label"])
            else:
                self.failed[spec["label"]] = error_message

            self.write_outputs()

            if spec["is_baseline"] and status != "success":
                raise RuntimeError("Baseline SRM100 failed; comparison is impossible.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        runner = SweepRunner(args)
        runner.run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
