#!/usr/bin/env python3
"""Run a standalone SRM-fraction fidelity sweep against an SRM100 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "fwhm_mean",
    "sensitivity_total",
    "sensitivity_mean",
    "asci_pct",
    "JI",
)

JI_FIELDS = (
    "fwhm_mean",
    "sensitivity_total",
    "sensitivity_mean",
    "asci_pct",
    "JI",
    "metric_fidelity",
    "srm_row_sampled",
    "srm_row_fraction",
    "srm_row_mode",
    "srm_row_seed",
    "srm_total_rows",
    "srm_active_rows",
    "srm_sampled_rows",
    "srm_sampled_fraction_actual",
    "config",
    "work_dir",
)


@dataclass
class RunResult:
    fraction_label: str
    expected_fraction: float
    run_name: str
    run_dir: str
    command: str
    status: str
    error_message: str
    elapsed_sec: float | None
    ji_metrics_elapsed_sec: float | None = None
    fwhm_mean: float | None = None
    sensitivity_total: float | None = None
    sensitivity_mean: float | None = None
    asci_pct: float | None = None
    JI: float | None = None
    metric_fidelity: str | None = None
    srm_row_sampled: str | None = None
    srm_row_fraction: float | None = None
    srm_row_mode: str | None = None
    srm_row_seed: int | None = None
    srm_total_rows: int | None = None
    srm_active_rows: int | None = None
    srm_sampled_rows: int | None = None
    srm_sampled_fraction_actual: float | None = None
    config: str | None = None
    work_dir: str | None = None


def parse_fraction_list(value: str) -> list[float]:
    fractions = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not fractions:
        raise argparse.ArgumentTypeError("At least one partial fraction is required.")
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise argparse.ArgumentTypeError(
                f"Partial fractions must be in (0, 1); got {fraction}."
            )
    return fractions


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def fraction_label(fraction: float) -> str:
    return f"srm{int(round(fraction * 100))}"


def has_fixed_baseline_reference(args: argparse.Namespace) -> bool:
    fields = (
        args.baseline_reference_fwhm,
        args.baseline_reference_sensitivity_total,
        args.baseline_reference_sensitivity_mean,
        args.baseline_reference_asci,
        args.baseline_reference_ji,
    )
    if all(value is None for value in fields):
        return False
    if any(value is None for value in fields):
        raise ValueError(
            "Fixed baseline reference requires all of: "
            "--baseline-reference-fwhm, --baseline-reference-sensitivity-total, "
            "--baseline-reference-sensitivity-mean, --baseline-reference-asci, "
            "--baseline-reference-ji."
        )
    return True


def build_fixed_baseline_result(args: argparse.Namespace) -> RunResult:
    return RunResult(
        fraction_label="srm100",
        expected_fraction=args.baseline_fraction,
        run_name="historical_reference_srm100",
        run_dir=str(args.baseline_reference_run_dir or ""),
        command="",
        status="success",
        error_message="",
        elapsed_sec=args.baseline_reference_elapsed_sec,
        fwhm_mean=args.baseline_reference_fwhm,
        sensitivity_total=args.baseline_reference_sensitivity_total,
        sensitivity_mean=args.baseline_reference_sensitivity_mean,
        asci_pct=args.baseline_reference_asci,
        JI=args.baseline_reference_ji,
        metric_fidelity="full",
        srm_row_sampled="False",
        srm_row_fraction=args.baseline_fraction,
        srm_row_mode=None,
        srm_row_seed=None,
        srm_total_rows=3360,
        srm_active_rows=3360,
        srm_sampled_rows=3360,
        srm_sampled_fraction_actual=1.0,
        config="historical_reference",
        work_dir=str(args.baseline_reference_run_dir or ""),
    )


def shell_join(parts: list[str]) -> str:
    return shlex.join(parts)


def build_pipeline_command(args: argparse.Namespace, run_name: str, fraction: float) -> list[str]:
    command = [
        sys.executable,
        "run_pipeline.py",
        "--runs-dir",
        str(args.out_dir),
        "--run-name",
        run_name,
        "--layout-idxs",
        args.layout_idxs,
        "--t8-poses",
        args.t8_poses,
        "--aperture-diam",
        str(args.aperture_diam),
        "--n-apertures",
        str(args.n_apertures),
        "--n-crystals-ring1",
        str(args.n_crystals_ring1),
        "--n-crystals-ring2",
        str(args.n_crystals_ring2),
        "--srm-row-fraction",
        str(fraction),
        "--srm-row-mode",
        args.srm_row_mode,
        "--srm-row-seed",
        str(args.srm_row_seed),
    ]
    if args.layout_file is not None:
        command.extend(["--layout-file", str(args.layout_file)])
    for flag in ("cpus", "pose_workers", "torch_threads", "torch_interop_threads"):
        value = getattr(args, flag)
        if value is not None:
            command.extend([f"--{flag.replace('_', '-')}", str(value)])
    if args.skip_recon:
        command.append("--skip-recon")
    if args.resume:
        command.append("--resume")
    return command


def read_last_ji_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def result_from_ji_row(
    *,
    label: str,
    expected_fraction: float,
    run_name: str,
    run_dir: Path,
    command: list[str],
    elapsed_sec: float,
    row: dict[str, str],
) -> RunResult:
    return RunResult(
        fraction_label=label,
        expected_fraction=expected_fraction,
        run_name=run_name,
        run_dir=str(run_dir),
        command=shell_join(command),
        status="success",
        error_message="",
        elapsed_sec=elapsed_sec,
        ji_metrics_elapsed_sec=parse_float(
            row.get("elapsed_sec") or row.get("runtime_sec")
        ),
        fwhm_mean=parse_float(row.get("fwhm_mean")),
        sensitivity_total=parse_float(row.get("sensitivity_total")),
        sensitivity_mean=parse_float(row.get("sensitivity_mean")),
        asci_pct=parse_float(row.get("asci_pct")),
        JI=parse_float(row.get("JI")),
        metric_fidelity=row.get("metric_fidelity") or None,
        srm_row_sampled=row.get("srm_row_sampled") or None,
        srm_row_fraction=parse_float(row.get("srm_row_fraction")),
        srm_row_mode=row.get("srm_row_mode") or None,
        srm_row_seed=parse_int(row.get("srm_row_seed")),
        srm_total_rows=parse_int(row.get("srm_total_rows")),
        srm_active_rows=parse_int(row.get("srm_active_rows")),
        srm_sampled_rows=parse_int(row.get("srm_sampled_rows")),
        srm_sampled_fraction_actual=parse_float(row.get("srm_sampled_fraction_actual")),
        config=row.get("config") or None,
        work_dir=row.get("work_dir") or None,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_rel_diff_pct(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return 100.0 * (value - baseline) / baseline


def classify_row(row: dict[str, Any]) -> str:
    values = (
        row.get("JI_abs_rel_error_pct"),
        row.get("fwhm_abs_rel_error_pct"),
        row.get("asci_abs_rel_error_pct"),
        row.get("sensitivity_total_abs_rel_error_pct"),
    )
    if any(value is None for value in values):
        return "Failed"
    if (
        values[0] <= 3
        and values[1] <= 2
        and values[2] <= 2
        and values[3] <= 1
    ):
        return "Excellent"
    if (
        values[0] <= 5
        and values[1] <= 3
        and values[2] <= 5
        and values[3] <= 2
    ):
        return "Good"
    return "Risky"


def build_comparison_rows(
    baseline: RunResult, partials: list[RunResult]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in partials:
        row: dict[str, Any] = {
            "fraction_label": result.fraction_label,
            "expected_fraction": result.expected_fraction,
            "status": result.status,
            "elapsed_sec": result.elapsed_sec,
            "speedup_vs_srm100": (
                baseline.elapsed_sec / result.elapsed_sec
                if baseline.elapsed_sec and result.elapsed_sec
                else None
            ),
            "srm_active_rows": result.srm_active_rows,
            "srm_sampled_rows": result.srm_sampled_rows,
            "srm_sampled_fraction_actual": result.srm_sampled_fraction_actual,
        }
        for metric in METRIC_FIELDS:
            current = getattr(result, metric)
            baseline_value = getattr(baseline, metric)
            abs_diff = (
                current - baseline_value
                if current is not None and baseline_value is not None
                else None
            )
            rel_diff = safe_rel_diff_pct(current, baseline_value)
            row[metric] = current
            row[f"{metric}_abs_diff"] = abs_diff
            row[f"{metric}_rel_diff_pct"] = rel_diff
            row[f"{metric}_abs_rel_error_pct"] = (
                abs(rel_diff) if rel_diff is not None else None
            )
        row["classification"] = classify_row(row)
        rows.append(row)
    return rows


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def write_summary_markdown(
    path: Path,
    args: argparse.Namespace,
    baseline: RunResult | None,
    comparison_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# SRM Fraction Fidelity Sweep",
        "",
        "## Reference geometry",
        "",
        f"- aperture_diam_mm: {args.aperture_diam}",
        f"- n_apertures: {args.n_apertures}",
        f"- n_crystals_ring1: {args.n_crystals_ring1}",
        f"- n_crystals_ring2: {args.n_crystals_ring2}",
        f"- row mode: {args.srm_row_mode}",
        f"- row seed: {args.srm_row_seed}",
        "",
    ]
    if baseline is None or baseline.status != "success":
        lines.extend(
            [
                "## Baseline SRM100 metrics",
                "",
                "Baseline failed; comparison metrics are unavailable.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Baseline SRM100 metrics",
                "",
                f"- FWHM mean: {fmt(baseline.fwhm_mean, 6)}",
                f"- sensitivity_total: {fmt(baseline.sensitivity_total, 6)}",
                f"- sensitivity_mean: {fmt(baseline.sensitivity_mean, 6)}",
                f"- ASCI percentage: {fmt(baseline.asci_pct, 6)}",
                f"- JI: {fmt(baseline.JI, 8)}",
                f"- runtime sec: {fmt(baseline.elapsed_sec, 2)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Comparison",
            "",
            "| Fidelity | Sampled rows | Runtime (s) | Speedup | JI err % | FWHM err % | ASCI err % | Sens total err % | Class |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in comparison_rows:
        lines.append(
            "| "
            f"{str(row.get('fraction_label', '')).upper()} | "
            f"{fmt(row.get('srm_sampled_rows'), 0)} | "
            f"{fmt(row.get('elapsed_sec'), 2)} | "
            f"{fmt(row.get('speedup_vs_srm100'), 2)} | "
            f"{fmt(row.get('JI_abs_rel_error_pct'), 3)} | "
            f"{fmt(row.get('fwhm_abs_rel_error_pct'), 3)} | "
            f"{fmt(row.get('asci_abs_rel_error_pct'), 3)} | "
            f"{fmt(row.get('sensitivity_total_abs_rel_error_pct'), 3)} | "
            f"{row.get('classification', '')} |"
        )

    accepted = [
        row for row in comparison_rows
        if row["classification"] in {"Excellent", "Good"}
    ]
    smallest_acceptable = min(accepted, key=lambda row: row["expected_fraction"], default=None)
    lines.extend(["", "## Recommendation", ""])
    if smallest_acceptable is None:
        lines.append("No partial SRM fidelity met the configured Good/Excellent thresholds.")
    else:
        lines.append(
            "Smallest acceptable fidelity: "
            f"{smallest_acceptable['fraction_label'].upper()} "
            f"({smallest_acceptable['classification']})."
        )
        lines.append(
            "Best tradeoff recommendation: "
            f"{smallest_acceptable['fraction_label'].upper()} "
            "because it is the lowest-cost fidelity meeting the configured thresholds."
        )
    srm50 = next((row for row in comparison_rows if row["fraction_label"] == "srm50"), None)
    if srm50 is not None:
        lines.append(
            "SRM50 classification: "
            f"{srm50['classification']}."
        )
    lines.extend(
        [
            "",
            "Final scientific reporting should still use SRM100.",
            "Partial SRM should only be used as cheap BO/Hyperband fidelity.",
            "",
            "## Notes",
            "",
            "- Excellent: JI <= 3%, FWHM <= 2%, ASCI <= 2%, sensitivity_total <= 1%.",
            "- Good: JI <= 5%, FWHM <= 3%, ASCI <= 5%, sensitivity_total <= 2%.",
            "- Risky: anything outside those thresholds.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def save_state(
    path: Path,
    *,
    args: argparse.Namespace,
    results: list[RunResult],
    output_paths: dict[str, str],
) -> None:
    completed = [
        result.fraction_label for result in results
        if result.status == "success"
    ]
    failed = [
        result.fraction_label for result in results
        if result.status != "success"
    ]
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_fractions": completed,
        "failed_fractions": failed,
        "run_dirs": {result.fraction_label: result.run_dir for result in results},
        "output_paths": output_paths,
        "reference_geometry": {
            "aperture_diam_mm": args.aperture_diam,
            "n_apertures": args.n_apertures,
            "n_crystals_ring1": args.n_crystals_ring1,
            "n_crystals_ring2": args.n_crystals_ring2,
            "layout_file": str(args.layout_file) if args.layout_file is not None else "",
        },
        "row_mode": args.srm_row_mode,
        "seed": args.srm_row_seed,
        "results": [asdict(result) for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def ensure_run_dir_policy(run_dir: Path, resume: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"Run directory already exists and is not empty: {run_dir}. "
            "Use --resume or a new --out-dir."
        )


def run_one(
    args: argparse.Namespace,
    *,
    label: str,
    fraction: float,
    run_name: str,
) -> RunResult:
    run_dir = args.out_dir / run_name
    command = build_pipeline_command(args, run_name, fraction)
    ensure_run_dir_policy(run_dir, args.resume)
    if args.dry_run:
        print(shell_join(command))
        return RunResult(
            fraction_label=label,
            expected_fraction=fraction,
            run_name=run_name,
            run_dir=str(run_dir),
            command=shell_join(command),
            status="dry_run",
            error_message="",
            elapsed_sec=None,
        )

    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed_sec = time.monotonic() - started
    if proc.returncode != 0:
        error_message = (proc.stderr or proc.stdout or "").strip()
        return RunResult(
            fraction_label=label,
            expected_fraction=fraction,
            run_name=run_name,
            run_dir=str(run_dir),
            command=shell_join(command),
            status="failed",
            error_message=error_message,
            elapsed_sec=elapsed_sec,
        )

    ji_csv = run_dir / "results" / "ji_metrics.csv"
    if not ji_csv.exists():
        return RunResult(
            fraction_label=label,
            expected_fraction=fraction,
            run_name=run_name,
            run_dir=str(run_dir),
            command=shell_join(command),
            status="failed",
            error_message=f"Missing JI CSV: {ji_csv}",
            elapsed_sec=elapsed_sec,
        )
    row = read_last_ji_row(ji_csv)
    return result_from_ji_row(
        label=label,
        expected_fraction=fraction,
        run_name=run_name,
        run_dir=run_dir,
        command=command,
        elapsed_sec=elapsed_sec,
        row=row,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate partial SRM fidelities against an SRM100 baseline."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/srm_fraction_sweep_baseline"))
    parser.add_argument("--aperture-diam", type=float, default=0.4)
    parser.add_argument("--n-apertures", type=int, default=180)
    parser.add_argument("--n-crystals-ring1", type=int, default=480)
    parser.add_argument("--n-crystals-ring2", type=int, default=720)
    parser.add_argument(
        "--layout-file",
        type=Path,
        default=None,
        help="Existing scanner layout tensor to reuse for every sweep run.",
    )
    parser.add_argument("--fractions", type=parse_fraction_list, default=parse_fraction_list("0.10,0.15,0.25,0.40,0.50,0.60"))
    parser.add_argument("--baseline-fraction", type=float, default=1.0)
    parser.add_argument("--baseline-reference-fwhm", type=float, default=None)
    parser.add_argument("--baseline-reference-sensitivity-total", type=float, default=None)
    parser.add_argument("--baseline-reference-sensitivity-mean", type=float, default=None)
    parser.add_argument("--baseline-reference-asci", type=float, default=None)
    parser.add_argument("--baseline-reference-ji", type=float, default=None)
    parser.add_argument("--baseline-reference-elapsed-sec", type=float, default=None)
    parser.add_argument("--baseline-reference-run-dir", type=Path, default=None)
    parser.add_argument("--srm-row-mode", default="evenly_spaced")
    parser.add_argument("--srm-row-seed", type=int, default=42)
    parser.add_argument("--layout-idxs", default="0,1")
    parser.add_argument("--t8-poses", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--pose-workers", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--torch-interop-threads", type=int, default=None)
    parser.add_argument("--skip-recon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    use_fixed_baseline = has_fixed_baseline_reference(args)
    summary_dir = args.out_dir / "summary"
    raw_csv = summary_dir / "srm_fraction_sweep_raw.csv"
    comparison_csv = summary_dir / "srm_fraction_sweep_comparison.csv"
    summary_md = summary_dir / "srm_fraction_sweep_summary.md"
    commands_sh = summary_dir / "srm_fraction_sweep_commands.sh"
    state_json = summary_dir / "srm_fraction_sweep_state.json"
    output_paths = {
        "raw_csv": str(raw_csv),
        "comparison_csv": str(comparison_csv),
        "summary_markdown": str(summary_md),
        "commands_manifest": str(commands_sh),
        "state_json": str(state_json),
    }

    run_specs = [
        *(
            []
            if use_fixed_baseline
            else [("srm100", args.baseline_fraction, "evals/baseline_srm100")]
        ),
        *[
            (fraction_label(fraction), fraction, f"evals/{fraction_label(fraction)}")
            for fraction in args.fractions
        ],
    ]
    commands = [
        shell_join(build_pipeline_command(args, run_name, fraction))
        for _, fraction, run_name in run_specs
    ]
    summary_dir.mkdir(parents=True, exist_ok=True)
    commands_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n")

    baseline = build_fixed_baseline_result(args) if use_fixed_baseline else None
    results: list[RunResult] = [baseline] if baseline is not None else []
    try:
        for label, fraction, run_name in run_specs:
            result = run_one(args, label=label, fraction=fraction, run_name=run_name)
            results.append(result)
            if label == "srm100":
                baseline = result
                if result.status not in {"success", "dry_run"}:
                    break
            raw_rows = [asdict(item) for item in results]
            write_csv(raw_csv, raw_rows)
            comparison_rows = (
                build_comparison_rows(
                    baseline,
                    [item for item in results if item.fraction_label != "srm100"],
                )
                if baseline is not None and baseline.status == "success"
                else []
            )
            write_csv(comparison_csv, comparison_rows)
            write_summary_markdown(summary_md, args, baseline, comparison_rows)
            save_state(state_json, args=args, results=results, output_paths=output_paths)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        save_state(state_json, args=args, results=results, output_paths=output_paths)
        return 2

    raw_rows = [asdict(item) for item in results]
    write_csv(raw_csv, raw_rows)
    partials = [item for item in results if item.fraction_label != "srm100"]
    comparison_rows = (
        build_comparison_rows(baseline, partials)
        if baseline is not None and baseline.status == "success"
        else []
    )
    write_csv(comparison_csv, comparison_rows)
    write_summary_markdown(summary_md, args, baseline, comparison_rows)
    save_state(state_json, args=args, results=results, output_paths=output_paths)

    if baseline is not None and baseline.status not in {"success", "dry_run"}:
        print("Baseline SRM100 failed; comparison sweep stopped.", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"Dry run complete. Commands written to {commands_sh}")
    else:
        failed = [item.fraction_label for item in partials if item.status != "success"]
        if failed:
            print(f"Completed with failed fractions: {', '.join(failed)}")
        else:
            print("SRM fraction sweep complete.")
        print(f"Raw CSV: {raw_csv}")
        print(f"Comparison CSV: {comparison_csv}")
        print(f"Summary: {summary_md}")
        print(f"State: {state_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
