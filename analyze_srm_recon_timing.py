#!/usr/bin/env python3
"""Read-only timing audit for SRM reconstruction fraction sweeps.

This script inspects an existing runsrmtest.py sweep without rerunning any
pipeline stages. It reads summary CSVs, log tails, file metadata, HDF5 dataset
shapes, and NPZ headers to explain whether elapsed_sec values are clean timing
measurements or contaminated by --resume/skipped stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - project normally has h5py
    h5py = None


STAGES = [
    "geometry",
    "ppdf",
    "beam_masks",
    "beam_properties",
    "asci",
    "ji",
    "visuals",
    "flist",
    "projection",
    "mlem",
    "view",
]

LOG_TERMS = [
    "[skip]",
    "skip",
    "exists",
    "resume",
    "already exists",
    "done in",
    "elapsed",
    "seconds",
    "computed",
    "processing",
    "finished",
    "error",
    "traceback",
]

STAGE_LOG_KEYWORDS = {
    "geometry": ["generate_mph", "scanner_circularfov", "geometry", "layout"],
    "ppdf": ["arg_ppdf", "ppdf", "t8"],
    "beam_masks": ["beam_masks", "beam mask"],
    "beam_properties": ["beam_properties", "beam properties"],
    "asci": ["analyze_extracted", "asci"],
    "ji": ["6_calc_ji", "calc_ji", "ji_metrics", "ji"],
    "visuals": ["generate_visuals", "visual", "plot"],
    "flist": ["generate_flist", "flist", "dataset_flist"],
    "projection": ["projection_t8", "projection"],
    "mlem": ["mlem_torch", "mlem"],
    "view": ["view_npz", "view"],
}

STAGE_OUTPUT_PATTERNS = {
    "geometry": [
        ("data", "scanner_layouts_*.tensor"),
        ("data", "scanner_layouts_*.pt"),
        ("data", "scanner_layouts*"),
    ],
    "ppdf": [("data", "position_*_ppdfs_t8_*.hdf5")],
    "beam_masks": [("data", "beams_masks_configuration_*.hdf5")],
    "beam_properties": [("data", "beams_properties_configuration_*.hdf5")],
    "asci": [("data", "asci_histogram_*.hdf5")],
    "ji": [("results", "ji_metrics.csv")],
    "visuals": [("plots", "*")],
    "flist": [
        ("recon", "dataset_flist.csv"),
        ("data", "dataset_flist.csv"),
        ("results", "dataset_flist.csv"),
    ],
    "projection": [
        ("recon", "*projection*.npy"),
        ("recon", "*projection*.npz"),
        ("data", "*projection*.npy"),
        ("data", "*projection*.npz"),
    ],
    "mlem": [
        ("recon", "recon_mlem*.npz"),
        ("recon", "*mlem*.npz"),
        ("recon", "*.npz"),
    ],
    "view": [
        ("plots", "*recon*"),
        ("plots", "*mlem*"),
        ("plots", "*.png"),
    ],
}

RAW_COLUMNS = [
    "label",
    "fraction",
    "elapsed_sec_from_raw",
    "command_has_resume",
    "srm_sampled_rows",
    "srm_active_rows",
    "ppdf_count",
    "ppdf_shape_first",
    "ppdf_total_mb",
    "mask_count",
    "mask_shape_first",
    "prop_count",
    "prop_shape_first",
    "asci_count",
    "asci_shape_first",
    "recon_npz_path",
    "recon_npz_exists",
    "recon_estimates_shape",
    "clean_runtime_likely",
    "timing_trust_level",
    "likely_reason_for_weird_timing",
]

STAGE_COLUMNS = [
    "label",
    "stage",
    "status",
    "parsed_duration_sec",
    "log_file",
    "output_count",
    "output_total_mb",
    "oldest_output_mtime",
    "newest_output_mtime",
    "evidence",
]


@dataclass
class FileGroupInfo:
    count: int = 0
    total_mb: float = 0.0
    oldest_mtime: str = ""
    newest_mtime: str = ""
    paths: list[str] | None = None


@dataclass
class LogInfo:
    path: str
    size_bytes: int
    mtime: str
    first_lines: list[str]
    last_lines: list[str]
    flags: dict[str, bool]
    durations: list[float]


@dataclass
class StageAudit:
    label: str
    stage: str
    status: str
    parsed_duration_sec: str
    log_file: str
    output_count: int
    output_total_mb: str
    oldest_output_mtime: str
    newest_output_mtime: str
    evidence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of SRM reconstruction sweep timing."
    )
    parser.add_argument(
        "--sweep-dir",
        default="runs/srm_recon_fraction_test",
        help="Existing SRM reconstruction sweep directory.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def to_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def format_float(value: float | int | str | None, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        value = to_float(value)
    try:
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def timestamp(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return ""


def resolve_run_dir(run_dir_value: str, sweep_dir: Path) -> Path:
    path = Path(run_dir_value)
    if path.exists():
        return path
    candidate = sweep_dir / run_dir_value
    if candidate.exists():
        return candidate
    return path


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    result = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def collect_stage_outputs(run_dir: Path, stage: str) -> list[Path]:
    paths: list[Path] = []
    for rel_dir, pattern in STAGE_OUTPUT_PATTERNS.get(stage, []):
        root = run_dir / rel_dir
        if root.exists():
            paths.extend(root.glob(pattern))
    return sorted(p for p in unique_paths(paths) if p.exists())


def file_group_info(paths: list[Path]) -> FileGroupInfo:
    if not paths:
        return FileGroupInfo(count=0, total_mb=0.0, oldest_mtime="", newest_mtime="", paths=[])
    mtimes = []
    total = 0
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        mtimes.append(stat.st_mtime)
    if not mtimes:
        return FileGroupInfo(count=0, total_mb=0.0, oldest_mtime="", newest_mtime="", paths=[])
    return FileGroupInfo(
        count=len(paths),
        total_mb=total / (1024 * 1024),
        oldest_mtime=datetime.fromtimestamp(min(mtimes)).isoformat(timespec="seconds"),
        newest_mtime=datetime.fromtimestamp(max(mtimes)).isoformat(timespec="seconds"),
        paths=[str(p) for p in paths],
    )


def scan_log(path: Path) -> LogInfo:
    first_lines: list[str] = []
    last_lines: deque[str] = deque(maxlen=80)
    flags = {term: False for term in LOG_TERMS}
    durations: list[float] = []
    duration_patterns = [
        re.compile(r"done\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", re.I),
        re.compile(r"elapsed(?:\s+time)?\s*[:=]?\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", re.I),
        re.compile(r"finished\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", re.I),
        re.compile(r"\btime\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|sec|seconds)?", re.I),
    ]

    try:
        with path.open("r", errors="replace") as handle:
            for idx, line in enumerate(handle):
                text = line.rstrip("\n")
                lower = text.lower()
                if idx < 20:
                    first_lines.append(text)
                last_lines.append(text)
                for term in LOG_TERMS:
                    if term in lower:
                        flags[term] = True
                for pattern in duration_patterns:
                    for match in pattern.finditer(text):
                        durations.append(float(match.group(1)))
    except OSError:
        pass

    try:
        stat = path.stat()
        size = stat.st_size
    except OSError:
        size = 0

    return LogInfo(
        path=str(path),
        size_bytes=size,
        mtime=timestamp(path),
        first_lines=first_lines,
        last_lines=list(last_lines),
        flags=flags,
        durations=durations,
    )


def collect_logs(run_dir: Path) -> list[LogInfo]:
    logs_dir = run_dir / "logs"
    if not logs_dir.exists():
        return []
    paths = sorted(p for p in logs_dir.rglob("*") if p.is_file())
    return [scan_log(path) for path in paths]


def log_matches_stage(log: LogInfo, stage: str) -> bool:
    text = " ".join(
        [Path(log.path).name.lower()]
        + [line.lower() for line in log.first_lines[:5]]
        + [line.lower() for line in log.last_lines[-10:]]
    )
    return any(keyword in text for keyword in STAGE_LOG_KEYWORDS.get(stage, []))


def summarize_log_evidence(logs: list[LogInfo]) -> str:
    if not logs:
        return ""
    pieces = []
    for log in logs:
        active_flags = [term for term, present in log.flags.items() if present]
        duration = max(log.durations) if log.durations else None
        first = " | ".join(log.first_lines[:2]).strip()
        last = " | ".join(log.last_lines[-4:]).strip()
        pieces.append(
            json.dumps(
                {
                    "file": log.path,
                    "size_bytes": log.size_bytes,
                    "mtime": log.mtime,
                    "flags": active_flags,
                    "duration_sec": duration,
                    "first_lines": first[:800],
                    "last_lines": last[:1600],
                },
                sort_keys=True,
            )
        )
    return "\n".join(pieces)


def parsed_duration(logs: list[LogInfo]) -> str:
    values = [duration for log in logs for duration in log.durations]
    if not values:
        return ""
    return format_float(max(values), 3)


def stage_status(command_has_resume: bool, outputs: list[Path], logs: list[LogInfo]) -> str:
    has_outputs = bool(outputs)
    flags = Counter()
    for log in logs:
        for term, present in log.flags.items():
            if present:
                flags[term] += 1

    if flags["traceback"] or flags["error"]:
        return "failed"
    if not has_outputs:
        return "missing"
    if flags["[skip]"] or flags["skip"]:
        return "skipped_existing"
    if flags["already exists"] or (flags["exists"] and command_has_resume):
        return "reused_old_output"
    if flags["processing"] or flags["computed"] or flags["finished"] or flags["done in"]:
        return "ran_this_resume" if command_has_resume else "ran"
    if command_has_resume:
        return "reused_old_output"
    return "unknown"


def h5_dataset_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    if h5py is None:
        return shapes
    try:
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if hasattr(obj, "shape"):
                    try:
                        shapes[name] = tuple(int(v) for v in obj.shape)
                    except TypeError:
                        pass

            handle.visititems(visitor)
    except Exception:
        return shapes
    return shapes


def first_non_index_shape(path: Path, preferred: list[str]) -> tuple[int, ...] | None:
    shapes = h5_dataset_shapes(path)
    for name in preferred:
        if name in shapes:
            return shapes[name]
    for name, shape in shapes.items():
        lower = name.lower()
        if "sampled_detector" in lower or "active_detector" in lower:
            continue
        return shape
    return None


def ppdf_shape_summary(paths: list[Path]) -> tuple[str, str]:
    first_dims = set()
    second_dims = set()
    for path in paths:
        shapes = h5_dataset_shapes(path)
        shape = shapes.get("ppdfs")
        if shape and len(shape) >= 2:
            first_dims.add(shape[0])
            second_dims.add(shape[1])
    first = ",".join(str(v) for v in sorted(first_dims))
    second = ",".join(str(v) for v in sorted(second_dims))
    return first, second


def npz_headers(path: Path) -> dict[str, tuple[int, ...]]:
    headers: dict[str, tuple[int, ...]] = {}
    if not path.exists():
        return headers
    try:
        with zipfile.ZipFile(path, "r") as archive:
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    continue
                key = member[:-4]
                with archive.open(member, "r") as handle:
                    version = np.lib.format.read_magic(handle)
                    shape, _, _ = np.lib.format._read_array_header(handle, version)
                    headers[key] = tuple(int(v) for v in shape)
    except Exception:
        return headers
    return headers


def selected_recon_headers(recon_path_value: str, run_dir: Path) -> tuple[bool, str]:
    if not recon_path_value:
        return False, ""
    path = Path(recon_path_value)
    if not path.exists():
        candidate = run_dir / recon_path_value
        path = candidate if candidate.exists() else path
    if not path.exists():
        return False, ""
    headers = npz_headers(path)
    if "estimates" in headers:
        return True, "x".join(str(v) for v in headers["estimates"])
    if headers:
        key = sorted(headers)[0]
        return True, f"{key}=" + "x".join(str(v) for v in headers[key])
    return True, ""


def inspect_run(row: dict[str, str], sweep_dir: Path) -> tuple[dict[str, Any], list[StageAudit]]:
    label = row.get("label", "")
    run_dir = resolve_run_dir(row.get("run_dir", ""), sweep_dir)
    command = row.get("command", "")
    command_has_resume = "--resume" in command.split()
    logs = collect_logs(run_dir)

    stage_audits: list[StageAudit] = []
    status_by_stage: dict[str, str] = {}
    output_info_by_stage: dict[str, FileGroupInfo] = {}
    for stage in STAGES:
        outputs = collect_stage_outputs(run_dir, stage)
        info = file_group_info(outputs)
        output_info_by_stage[stage] = info
        stage_logs = [log for log in logs if log_matches_stage(log, stage)]
        status = stage_status(command_has_resume, outputs, stage_logs)
        status_by_stage[stage] = status
        stage_audits.append(
            StageAudit(
                label=label,
                stage=stage,
                status=status,
                parsed_duration_sec=parsed_duration(stage_logs),
                log_file=";".join(log.path for log in stage_logs),
                output_count=info.count,
                output_total_mb=format_float(info.total_mb, 3),
                oldest_output_mtime=info.oldest_mtime,
                newest_output_mtime=info.newest_mtime,
                evidence=summarize_log_evidence(stage_logs),
            )
        )

    ppdf_paths = collect_stage_outputs(run_dir, "ppdf")
    ppdf_first_dims, ppdf_second_dims = ppdf_shape_summary(ppdf_paths)
    ppdf_info = output_info_by_stage["ppdf"]

    mask_paths = collect_stage_outputs(run_dir, "beam_masks")
    mask_shape = first_non_index_shape(mask_paths[0], ["beam_mask", "beam_masks"]) if mask_paths else None

    prop_paths = collect_stage_outputs(run_dir, "beam_properties")
    prop_shape = first_non_index_shape(prop_paths[0], ["beam_properties", "properties"]) if prop_paths else None

    asci_paths = collect_stage_outputs(run_dir, "asci")
    asci_shape = first_non_index_shape(asci_paths[0], ["asci_histogram"]) if asci_paths else None

    recon_exists, recon_shape = selected_recon_headers(row.get("recon_npz_path", ""), run_dir)

    problematic = {
        "skipped_existing",
        "reused_old_output",
        "missing",
        "failed",
        "unknown",
    }
    contaminated = command_has_resume or any(status in problematic for status in status_by_stage.values())
    clean_runtime_likely = not contaminated
    if command_has_resume or any(
        status in {"skipped_existing", "reused_old_output"} for status in status_by_stage.values()
    ):
        trust = "low"
    elif any(status in {"missing", "failed", "unknown"} for status in status_by_stage.values()):
        trust = "medium"
    else:
        trust = "high"

    likely_reason = explain_run_timing(command_has_resume, status_by_stage)

    raw = {
        "label": label,
        "fraction": row.get("fraction", ""),
        "elapsed_sec_from_raw": row.get("elapsed_sec", ""),
        "command_has_resume": command_has_resume,
        "srm_sampled_rows": row.get("srm_sampled_rows", ""),
        "srm_active_rows": row.get("srm_active_rows", ""),
        "ppdf_count": ppdf_info.count,
        "ppdf_shape_first": ppdf_first_dims,
        "ppdf_shape_second": ppdf_second_dims,
        "ppdf_total_mb": format_float(ppdf_info.total_mb, 3),
        "mask_count": output_info_by_stage["beam_masks"].count,
        "mask_shape_first": shape_first(mask_shape),
        "prop_count": output_info_by_stage["beam_properties"].count,
        "prop_shape_first": shape_first(prop_shape),
        "asci_count": output_info_by_stage["asci"].count,
        "asci_shape_first": shape_first(asci_shape),
        "recon_npz_path": row.get("recon_npz_path", ""),
        "recon_npz_exists": recon_exists,
        "recon_estimates_shape": recon_shape,
        "clean_runtime_likely": clean_runtime_likely,
        "timing_trust_level": trust,
        "likely_reason_for_weird_timing": likely_reason,
    }
    for stage in STAGES:
        raw[f"{stage}_status"] = status_by_stage[stage]
    return raw, stage_audits


def shape_first(shape: tuple[int, ...] | None) -> str:
    if not shape:
        return ""
    return str(shape[0])


def explain_run_timing(command_has_resume: bool, statuses: dict[str, str]) -> str:
    skipped = [stage for stage, status in statuses.items() if status == "skipped_existing"]
    reused = [stage for stage, status in statuses.items() if status == "reused_old_output"]
    ran = [stage for stage, status in statuses.items() if status in {"ran", "ran_this_resume"}]
    failed = [stage for stage, status in statuses.items() if status == "failed"]
    missing = [stage for stage, status in statuses.items() if status == "missing"]
    if failed:
        return "failed stages detected: " + ",".join(failed)
    if command_has_resume and (skipped or reused):
        return "resume-contaminated timing; skipped/reused stages: " + ",".join(skipped + reused)
    if command_has_resume and ran:
        return "resume run with partial recomputation; ran stages: " + ",".join(ran)
    if command_has_resume:
        return "command used --resume; elapsed_sec is not clean runtime evidence"
    if missing:
        return "missing stage outputs: " + ",".join(missing)
    return "no obvious resume contamination detected"


def compare_with_baseline(raw_rows: list[dict[str, Any]]) -> dict[str, str]:
    baseline = next((row for row in raw_rows if row.get("label") == "baseline"), None)
    if baseline is None:
        baseline = next((row for row in raw_rows if "baseline" in str(row.get("label", "")).lower()), None)
    notes: dict[str, str] = {}
    if baseline is None:
        return notes
    baseline_statuses = {stage: baseline.get(f"{stage}_status", "") for stage in STAGES}
    baseline_clean = str(baseline.get("clean_runtime_likely", "")).lower() == "true"
    for row in raw_rows:
        label = str(row.get("label", ""))
        if row is baseline:
            notes[label] = "baseline timing reference"
            continue
        mismatches = []
        for stage in STAGES:
            b_status = baseline_statuses.get(stage)
            r_status = row.get(f"{stage}_status", "")
            if b_status != r_status:
                mismatches.append(f"{stage}:baseline={b_status},run={r_status}")
        clean = str(row.get("clean_runtime_likely", "")).lower() == "true"
        if baseline_clean and clean and not mismatches:
            notes[label] = "speedup may be comparable; both runs look clean"
        elif mismatches:
            notes[label] = "stage mismatch with baseline; speedup invalid: " + "; ".join(mismatches[:6])
        else:
            notes[label] = "speedup invalid or low trust because baseline/run used resume or reused outputs"
    return notes


def write_summary(
    path: Path,
    sweep_dir: Path,
    raw_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, str]],
    baseline_notes: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    low_trust = [row for row in raw_rows if row.get("timing_trust_level") == "low"]
    resume_rows = [row for row in raw_rows if str(row.get("command_has_resume", "")).lower() == "true"]
    label_to_comp = {row.get("label", ""): row for row in comparison_rows}

    def row_line(row: dict[str, Any]) -> str:
        label = row.get("label", "")
        comp = label_to_comp.get(label, {})
        speedup = comp.get("speedup_vs_srm100", "")
        note = baseline_notes.get(label, row.get("likely_reason_for_weird_timing", ""))
        return (
            f"| {label} | {row.get('fraction', '')} | {row.get('elapsed_sec_from_raw', '')} | "
            f"{speedup} | {row.get('timing_trust_level', '')} | "
            f"{row.get('clean_runtime_likely', '')} | {note} |"
        )

    srm25 = next((row for row in raw_rows if str(row.get("label", "")).lower() == "srm25"), None)
    srm50 = next((row for row in raw_rows if str(row.get("label", "")).lower() == "srm50"), None)
    srm60 = next((row for row in raw_rows if str(row.get("label", "")).lower() == "srm60"), None)

    lines = [
        "# SRM Reconstruction Timing Audit",
        "",
        f"Sweep directory: `{sweep_dir}`",
        "",
        "## Summary",
        "",
        f"- Runs inspected: {len(raw_rows)}",
        f"- Runs launched with `--resume`: {len(resume_rows)}",
        f"- Low-trust timing rows: {len(low_trust)}",
        "",
        "Elapsed times should be treated as clean runtime only when `clean_runtime_likely` is true. "
        "A resumed run can skip or reuse expensive outputs, so elapsed time may measure only the remaining work.",
        "",
        "## Per-Fraction Timing Interpretation",
        "",
        "| Label | Fraction | Raw elapsed_sec | Reported speedup | Trust | Clean runtime likely | Interpretation |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    lines.extend(row_line(row) for row in raw_rows)
    lines.extend(
        [
            "",
            "## Specific Timing Questions",
            "",
            "### Why SRM25 looked fast",
            "",
            specific_label_note(srm25, baseline_notes),
            "",
            "### Why SRM50 looked slow",
            "",
            specific_label_note(srm50, baseline_notes),
            "",
            "### Why SRM60 looked slow",
            "",
            specific_label_note(srm60, baseline_notes),
            "",
            "## Speedup Validity",
            "",
        ]
    )
    if any(str(row.get("clean_runtime_likely", "")).lower() != "true" for row in raw_rows):
        lines.append(
            "At least one run has non-clean timing evidence. Fraction speedups from this sweep should not be used as final timing claims."
        )
    else:
        lines.append("All inspected runs look clean enough for preliminary timing comparisons.")

    lines.extend(
        [
            "",
            "## Recommended Clean Timing Experiment",
            "",
            "Option A: fresh isolated rerun for timing only, using a new output directory and no `--resume`:",
            "",
            "```bash",
            "python runsrmtest.py \\",
            "  --out-dir runs/srm_recon_fraction_timing_clean \\",
            "  --only baseline,srm25,srm50",
            "```",
            "",
            "Option B: if reconstruction is too expensive, first run a metric-only timing experiment with `--skip-recon` through the pipeline runner.",
            "",
            "This audit does not run either command.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def specific_label_note(row: dict[str, Any] | None, baseline_notes: dict[str, str]) -> str:
    if row is None:
        return "No row found for this label."
    label = str(row.get("label", ""))
    reason = row.get("likely_reason_for_weird_timing", "")
    baseline_note = baseline_notes.get(label, "")
    statuses = ", ".join(
        f"{stage}={row.get(f'{stage}_status', '')}"
        for stage in ["ppdf", "beam_masks", "beam_properties", "projection", "mlem"]
    )
    return f"{label}: {reason}. {baseline_note}. Key stages: {statuses}."


def main() -> int:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    summary_dir = sweep_dir / "summary"
    raw_csv = summary_dir / "srm_recon_fraction_raw.csv"
    comparison_csv = summary_dir / "srm_recon_fraction_comparison.csv"
    audit_dir = summary_dir / "timing_audit"

    raw_input_rows = read_csv_rows(raw_csv)
    comparison_rows = read_csv_rows(comparison_csv)
    if not raw_input_rows:
        raise SystemExit(f"No raw sweep CSV found or no rows: {raw_csv}")

    raw_output_rows: list[dict[str, Any]] = []
    stage_output_rows: list[dict[str, Any]] = []
    for row in raw_input_rows:
        raw_row, stage_rows = inspect_run(row, sweep_dir)
        raw_output_rows.append(raw_row)
        stage_output_rows.extend(asdict(stage_row) for stage_row in stage_rows)

    baseline_notes = compare_with_baseline(raw_output_rows)
    for row in raw_output_rows:
        note = baseline_notes.get(str(row.get("label", "")))
        if note and row.get("label") != "baseline":
            row["stage_mismatch_with_baseline"] = "stage mismatch" in note
            row["speedup_valid"] = "may be comparable" in note
            row["likely_reason_for_weird_timing"] = (
                str(row.get("likely_reason_for_weird_timing", "")) + " | " + note
            )
        else:
            row["stage_mismatch_with_baseline"] = False
            row["speedup_valid"] = False

    raw_fieldnames = (
        RAW_COLUMNS
        + [f"{stage}_status" for stage in STAGES]
        + ["stage_mismatch_with_baseline", "speedup_valid"]
    )
    write_csv(audit_dir / "srm_recon_timing_audit_raw.csv", raw_output_rows, raw_fieldnames)
    write_csv(audit_dir / "srm_recon_stage_timing_audit.csv", stage_output_rows, STAGE_COLUMNS)
    write_summary(
        audit_dir / "srm_recon_timing_audit_summary.md",
        sweep_dir,
        raw_output_rows,
        comparison_rows,
        baseline_notes,
    )

    print(f"Wrote timing audit to {audit_dir}")
    print(f"Rows inspected: {len(raw_output_rows)}")
    print("No pipeline commands were run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
