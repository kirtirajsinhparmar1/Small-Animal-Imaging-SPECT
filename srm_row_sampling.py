"""Detector-row sampling helpers for SRM/PPDF generation.

The detector-response rows are ordered ring-major, then cell index, then
crystal side. SRM50 keeps 50% of those detector-response rows.
"""

from __future__ import annotations

from contextlib import nullcontext
from os import PathLike
from typing import Any, Optional

import numpy as np


DETS_PER_RING = [40 * 6 * 2, 40 * 9 * 2, 40 * 12 * 2, 40 * 15 * 2]
CELLS_PER_SECTOR = [6, 9, 12, 15]
N_SECTORS = 40
DEFAULT_TOTAL_ROWS = 3360
_SUPPORTED_DETS_PER_RING = {DEFAULT_TOTAL_ROWS: DETS_PER_RING}


def _as_valid_detector_rows(name: str, rows, total_rows: int) -> np.ndarray:
    out = np.sort(np.asarray(rows, dtype=np.int64))
    try:
        validate_sampled_rows(out, total_rows)
    except Exception as exc:
        raise type(exc)(f"{name}: {exc}") from exc
    return out


def _default_active_rows(total_rows: int) -> np.ndarray:
    return np.arange(total_rows, dtype=np.int64)


def _balanced_sector_counts(n_rows: int) -> list[int]:
    base = n_rows // N_SECTORS
    rem = n_rows % N_SECTORS
    counts = [base] * N_SECTORS
    for i in range(rem):
        sector = int(np.floor(i * N_SECTORS / rem))
        counts[sector] += 1
    return counts


def build_detector_row_metadata(total_rows: int = DEFAULT_TOTAL_ROWS) -> dict:
    """Return detector-row metadata arrays for the historical 3360-row identity layout."""
    if total_rows != DEFAULT_TOTAL_ROWS:
        raise ValueError(
            f"Unsupported detector identity layout: got total_rows={total_rows}; "
            f"expected the historical baseline layout with {DEFAULT_TOTAL_ROWS} rows."
        )

    dets_per_ring = _SUPPORTED_DETS_PER_RING[total_rows]
    expected_total = sum(dets_per_ring)
    if expected_total != total_rows:
        raise ValueError(
            f"Invalid detector constants: ring counts sum to {expected_total}, "
            f"expected {total_rows}."
        )

    row_idx = np.arange(total_rows, dtype=np.int64)
    ring_id = np.empty(total_rows, dtype=np.int64)
    cell_idx = np.empty(total_rows, dtype=np.int64)
    crystal_side = np.empty(total_rows, dtype=np.int64)
    row_within_ring = np.empty(total_rows, dtype=np.int64)
    cells_per_sector = np.empty(total_rows, dtype=np.int64)
    angle_bin_40 = np.empty(total_rows, dtype=np.int64)

    start = 0
    for ring, ring_rows in enumerate(dets_per_ring):
        stop = start + ring_rows
        within = np.arange(ring_rows, dtype=np.int64)
        ring_cell_idx = within // 2
        if total_rows == DEFAULT_TOTAL_ROWS:
            ring_cells_per_sector = CELLS_PER_SECTOR[ring]
            expected_ring_rows = N_SECTORS * ring_cells_per_sector * 2
            if ring_rows != expected_ring_rows:
                raise ValueError(
                    f"Invalid ring {ring} detector count: got {ring_rows}, "
                    f"expected {expected_ring_rows}."
                )
            ring_angle_bin = ring_cell_idx // ring_cells_per_sector
            ring_cells_per_sector_array = np.full(
                ring_rows, ring_cells_per_sector, dtype=np.int64
            )
        else:
            sector_counts = _balanced_sector_counts(ring_rows)
            ring_angle_bin = np.empty(ring_rows, dtype=np.int64)
            ring_cells_per_sector_array = np.empty(ring_rows, dtype=np.int64)
            sector_start = 0
            for angle_bin, sector_count in enumerate(sector_counts):
                sector_stop = sector_start + sector_count
                ring_angle_bin[sector_start:sector_stop] = angle_bin
                ring_cells_per_sector_array[sector_start:sector_stop] = max(
                    1, sector_count // 2
                )
                sector_start = sector_stop
            if sector_start != ring_rows:
                raise ValueError(
                    f"Invalid sector metadata for ring {ring}: built "
                    f"{sector_start}, expected {ring_rows}."
                )

        if ring_angle_bin.size != ring_rows:
            raise ValueError(f"Internal metadata error for ring {ring}.")
        if int(ring_angle_bin.min()) != 0 or int(ring_angle_bin.max()) != N_SECTORS - 1:
            raise ValueError(
                f"Invalid angle-bin range for ring {ring}: "
                f"{int(ring_angle_bin.min())}..{int(ring_angle_bin.max())}."
            )

        ring_id[start:stop] = ring
        cell_idx[start:stop] = ring_cell_idx
        crystal_side[start:stop] = within % 2
        row_within_ring[start:stop] = within
        cells_per_sector[start:stop] = ring_cells_per_sector_array
        angle_bin_40[start:stop] = ring_angle_bin
        start = stop

    if start != total_rows:
        raise ValueError(f"Invalid detector metadata length: built {start}, expected {total_rows}.")

    return {
        "row_idx": row_idx,
        "ring_id": ring_id,
        "cell_idx": cell_idx,
        "crystal_side": crystal_side,
        "row_within_ring": row_within_ring,
        "cells_per_sector": cells_per_sector,
        "angle_bin_40": angle_bin_40,
    }


def validate_sampled_rows(sampled_rows, total_rows: int) -> None:
    """Validate sampled detector row indices."""
    rows = np.asarray(sampled_rows)

    if rows.ndim != 1:
        raise ValueError(f"sampled_rows must be a 1-D array; got shape {rows.shape}.")
    if rows.size == 0:
        raise ValueError("sampled_rows must be non-empty.")
    if not np.issubdtype(rows.dtype, np.integer):
        raise TypeError(f"sampled_rows must contain integer rows; got dtype {rows.dtype}.")
    if int(rows.min()) < 0 or int(rows.max()) >= total_rows:
        raise ValueError(
            f"sampled_rows must be within [0, {total_rows}); "
            f"got min={int(rows.min())}, max={int(rows.max())}."
        )
    if np.unique(rows).size != rows.size:
        raise ValueError("sampled_rows contains duplicate row indices.")


def make_sampled_detector_rows(
    total_rows: int,
    fraction: float,
    mode: str = "ring_cell_random",
    seed: int = 42,
    active_detector_unit_indices: Optional[Any] = None,
) -> np.ndarray:
    """Build a sorted int64 detector-row sample from active original detector IDs."""
    if not (fraction > 0 and fraction <= 1):
        raise ValueError(f"fraction must be > 0 and <= 1; got {fraction}.")

    if active_detector_unit_indices is None:
        active_rows = _default_active_rows(total_rows)
    else:
        active_rows = _as_valid_detector_rows(
            "active_detector_unit_indices", active_detector_unit_indices, total_rows
        )

    if fraction == 1.0 or mode == "all":
        rows = active_rows.copy()
        validate_sampled_rows(rows, total_rows)
        return rows

    if mode == "every_k":
        k = round(1 / fraction)
        if k < 1:
            raise ValueError(f"Invalid every_k step {k} for fraction={fraction}.")
        rows = active_rows[::k]
    elif mode == "evenly_spaced":
        n_rows = max(1, round(fraction * active_rows.size))
        raw_positions = np.linspace(0, active_rows.size - 1, num=n_rows)
        positions = np.rint(raw_positions).astype(np.int64, copy=False)
        unique_positions = np.unique(positions)
        if unique_positions.size != n_rows:
            used = set(int(pos) for pos in unique_positions)
            fill_positions = [
                pos for pos in range(active_rows.size)
                if pos not in used
            ][: n_rows - unique_positions.size]
            positions = np.sort(
                np.concatenate(
                    [unique_positions, np.asarray(fill_positions, dtype=np.int64)]
                )
            )
        else:
            positions = np.sort(unique_positions)
        rows = active_rows[positions]
    elif mode == "random":
        n_rows = round(fraction * active_rows.size)
        if n_rows < 1:
            raise ValueError(
                f"fraction={fraction} selects zero rows from "
                f"{active_rows.size} active detector rows."
            )
        rng = np.random.default_rng(seed)
        rows = rng.choice(active_rows, size=n_rows, replace=False).astype(np.int64, copy=False)
    elif mode in {"ring_cell_stratified", "ring_cell_random"}:
        metadata = build_detector_row_metadata(total_rows)
        active_mask = np.zeros(total_rows, dtype=bool)
        active_mask[active_rows] = True
        rng = np.random.default_rng(seed)
        chunks = []

        for ring in range(len(DETS_PER_RING)):
            for angle_bin in range(N_SECTORS):
                mask = (
                    active_mask
                    & (metadata["ring_id"] == ring)
                    & (metadata["angle_bin_40"] == angle_bin)
                )
                group_rows = metadata["row_idx"][mask]
                if group_rows.size == 0:
                    continue

                n_group_rows = max(1, round(fraction * group_rows.size))
                if mode == "ring_cell_stratified":
                    positions = np.linspace(
                        0,
                        group_rows.size - 1,
                        num=n_group_rows,
                        dtype=np.int64,
                    )
                    chunks.append(group_rows[positions])
                else:
                    chunks.append(rng.choice(group_rows, size=n_group_rows, replace=False))

        if not chunks:
            raise ValueError("No active detector rows were available for ring/angle sampling.")
        rows = np.concatenate(chunks).astype(np.int64, copy=False)
    else:
        raise ValueError(
            "Unsupported sampling mode "
            f"{mode!r}; expected one of all, every_k, evenly_spaced, random, "
            "ring_cell_stratified, ring_cell_random."
        )

    rows = np.sort(rows.astype(np.int64, copy=False))
    validate_sampled_rows(rows, total_rows)
    return rows


def summarize_sampled_rows(
    sampled_rows,
    total_rows: int = DEFAULT_TOTAL_ROWS,
    active_detector_unit_indices: Optional[Any] = None,
) -> dict:
    """Return balance and coverage metrics for sampled detector rows."""
    rows = np.asarray(sampled_rows, dtype=np.int64)
    validate_sampled_rows(rows, total_rows)
    if active_detector_unit_indices is None:
        active_rows = _default_active_rows(total_rows)
    else:
        active_rows = _as_valid_detector_rows(
            "active_detector_unit_indices", active_detector_unit_indices, total_rows
        )
    if not np.isin(rows, active_rows).all():
        raise ValueError("sampled_rows must be a subset of active_detector_unit_indices.")

    sampled_mask = np.zeros(total_rows, dtype=bool)
    sampled_mask[rows] = True

    if total_rows in _SUPPORTED_DETS_PER_RING:
        metadata = build_detector_row_metadata(total_rows)
        ring_counts = [
            int(np.count_nonzero(sampled_mask & (metadata["ring_id"] == ring)))
            for ring in range(len(_SUPPORTED_DETS_PER_RING[total_rows]))
        ]
        angle_counts = np.array(
            [
                np.count_nonzero(sampled_mask & (metadata["angle_bin_40"] == angle_bin))
                for angle_bin in range(N_SECTORS)
            ],
            dtype=np.int64,
        )
        crystal_side0 = int(np.count_nonzero(sampled_mask & (metadata["crystal_side"] == 0)))
        crystal_side1 = int(np.count_nonzero(sampled_mask & (metadata["crystal_side"] == 1)))
    else:
        ring_counts = _summarize_known_ring_counts(rows, total_rows)
        angle_counts = np.array([], dtype=np.int64)
        crystal_side0 = int(np.count_nonzero(rows % 2 == 0))
        crystal_side1 = int(np.count_nonzero(rows % 2 == 1))

    return {
        "n_sampled_rows": int(rows.size),
        "sampled_fraction_actual": float(rows.size / active_rows.size),
        "ring0_sampled": ring_counts[0],
        "ring1_sampled": ring_counts[1],
        "ring2_sampled": ring_counts[2],
        "ring3_sampled": ring_counts[3],
        "angle_bins_covered": int(np.count_nonzero(angle_counts > 0)) if angle_counts.size else 0,
        "min_rows_per_angle_bin": int(angle_counts.min()) if angle_counts.size else 0,
        "max_rows_per_angle_bin": int(angle_counts.max()) if angle_counts.size else 0,
        "crystal_side0_sampled": crystal_side0,
        "crystal_side1_sampled": crystal_side1,
    }


def _summarize_known_ring_counts(rows: np.ndarray, total_rows: int) -> list[int]:
    return [int(rows.size), 0, 0, 0]


def write_row_sampling_metadata(
    h5file,
    sampled_rows,
    *,
    fraction: float,
    mode: str,
    seed: int,
    total_rows: int,
    active_detector_unit_indices: Optional[Any] = None,
    extra_attrs: Optional[dict] = None,
) -> None:
    """Write row-sampling metadata and sampled row indices to an open HDF5 file."""
    rows = np.sort(np.asarray(sampled_rows, dtype=np.int64))
    validate_sampled_rows(rows, total_rows)
    if active_detector_unit_indices is None:
        active_rows = _default_active_rows(total_rows)
    else:
        active_rows = _as_valid_detector_rows(
            "active_detector_unit_indices", active_detector_unit_indices, total_rows
        )
    if not np.isin(rows, active_rows).all():
        raise ValueError("sampled rows must be a subset of active detector rows.")
    summary = summarize_sampled_rows(rows, total_rows, active_rows)

    h5file.attrs["srm_row_sampled"] = bool(rows.size != active_rows.size)
    h5file.attrs["srm_row_fraction"] = float(fraction)
    h5file.attrs["srm_row_mode"] = str(mode)
    h5file.attrs["srm_row_seed"] = int(seed)
    h5file.attrs["srm_total_rows"] = int(total_rows)
    h5file.attrs["srm_active_rows"] = int(active_rows.size)
    h5file.attrs["srm_sampled_rows"] = int(rows.size)
    h5file.attrs["srm_sampled_fraction_actual"] = summary["sampled_fraction_actual"]
    if extra_attrs:
        for key, value in extra_attrs.items():
            if value is not None:
                h5file.attrs[str(key)] = value

    for key, value in summary.items():
        if isinstance(value, (bool, int, float, np.integer, np.floating)):
            h5file.attrs[f"srm_{key}"] = value

    if "active_detector_unit_indices" in h5file:
        del h5file["active_detector_unit_indices"]
    h5file.create_dataset("active_detector_unit_indices", data=active_rows, dtype="int64")

    if "sampled_detector_unit_indices" in h5file:
        del h5file["sampled_detector_unit_indices"]
    h5file.create_dataset("sampled_detector_unit_indices", data=rows, dtype="int64")


def read_row_sampling_metadata(h5file_or_path, ppdf_shape_first_dim: int | None = None) -> dict:
    """Read row-sampling metadata from an HDF5 object or path.

    If no sampled-row dataset exists, assume a full dense row layout using
    ppdf_shape_first_dim.
    """
    if isinstance(h5file_or_path, (str, bytes, PathLike)):
        import h5py

        context = h5py.File(h5file_or_path, "r")
    else:
        context = nullcontext(h5file_or_path)

    with context as h5file:
        attrs = {key: _decode_h5_attr(value) for key, value in h5file.attrs.items()}

        if "sampled_detector_unit_indices" in h5file:
            rows = np.asarray(h5file["sampled_detector_unit_indices"][:], dtype=np.int64)
            total_rows = int(attrs.get("srm_total_rows", ppdf_shape_first_dim or rows.size))
            validate_sampled_rows(rows, total_rows)
            active_rows = None
            if "active_detector_unit_indices" in h5file:
                active_rows = np.asarray(h5file["active_detector_unit_indices"][:], dtype=np.int64)
                validate_sampled_rows(active_rows, total_rows)
                if not np.isin(rows, active_rows).all():
                    raise ValueError(
                        "sampled_detector_unit_indices must be a subset of "
                        "active_detector_unit_indices."
                    )
                attrs["srm_active_rows"] = int(attrs.get("srm_active_rows", active_rows.size))
            else:
                is_full = (
                    not bool(attrs.get("srm_row_sampled", rows.size != total_rows))
                    or rows.size == total_rows
                )
                active_rows = (
                    _default_active_rows(total_rows)
                    if is_full
                    else rows.copy()
                )
            attrs["sampled_detector_unit_indices"] = rows
            attrs["sampled_rows"] = rows
            attrs["active_detector_unit_indices"] = active_rows
            attrs["active_rows"] = active_rows
            return attrs

        if ppdf_shape_first_dim is None:
            raise ValueError(
                "sampled_detector_unit_indices is missing and ppdf_shape_first_dim "
                "was not provided for dense-row fallback."
            )

        rows = np.arange(ppdf_shape_first_dim, dtype=np.int64)
        attrs.update(
            {
                "srm_row_sampled": False,
                "srm_row_fraction": 1.0,
                "srm_sampled_fraction_actual": 1.0,
                "srm_total_rows": int(ppdf_shape_first_dim),
                "srm_active_rows": int(ppdf_shape_first_dim),
                "srm_sampled_rows": int(ppdf_shape_first_dim),
                "sampled_detector_unit_indices": rows,
                "sampled_rows": rows,
                "active_detector_unit_indices": rows,
                "active_rows": rows,
            }
        )
        return attrs


def _assert_compatible_rows(reference_rows, current_rows, row_name: str, path: str = "") -> None:
    reference = np.asarray(reference_rows, dtype=np.int64)
    current = np.asarray(current_rows, dtype=np.int64)
    location = f" for {path}" if path else ""

    if reference.shape != current.shape:
        raise ValueError(
            f"Incompatible {row_name}{location}: "
            f"shape {reference.shape} != {current.shape}."
        )
    if not np.array_equal(reference, current):
        mismatch = np.flatnonzero(reference != current)
        first = int(mismatch[0]) if mismatch.size else -1
        if first >= 0:
            raise ValueError(
                f"Incompatible {row_name}{location}: first mismatch at "
                f"position {first}: reference={int(reference[first])}, current={int(current[first])}."
            )
        raise ValueError(f"Incompatible {row_name}{location}: arrays differ.")


def assert_compatible_sampled_rows(reference_rows, current_rows, path: str = "") -> None:
    """Raise if two sampled-row arrays are not identical."""
    _assert_compatible_rows(reference_rows, current_rows, "sampled detector rows", path)


def assert_compatible_active_rows(reference_rows, current_rows, path: str = "") -> None:
    """Raise if two active-row arrays are not identical."""
    _assert_compatible_rows(reference_rows, current_rows, "active detector rows", path)


def _decode_h5_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
