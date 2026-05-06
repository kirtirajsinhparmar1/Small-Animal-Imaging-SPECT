import os
import math
import random
import argparse
import torch
from typing import List, Tuple, Dict
from torch import save as torch_save, Tensor

# --- project helpers ---
# helper.py provides:
# - generate_md5_from_tensors: creates a unique hash for geometry identification
# - plot_polygons_from_vertices_2d_mpl: draws (N,4,2) polygons on a Matplotlib axis
# - positions_parameters: creates motion poses (rotation + translation grid)
# - transform_to_positions_2d_batch: applies each pose transform to a batch of 2D points
# - OutDataDict: type alias for the output dictionary format
try:
    from helper import (
        generate_md5_from_tensors,          # hash id unique to geometry
        plot_polygons_from_vertices_2d_mpl, # draws (N,4,2) polygons on matplotlib axis
        positions_parameters,               # motion poses
        transform_to_positions_2d_batch,    # transforms points by poses
        OutDataDict,
    )
except ImportError as e:
    print(f"Error importing from helper.py: {e}")
    raise  # if helper.py can't be imported, script stops


# -------------------------------
# Geometry builders
# -------------------------------

def build_sc_spect_detector_rings(
    ring_inner_diameters_mm: List[float],
    detectors_per_ring: List[int],
    scint_tangential_mm: float = 0.84,      # width around the ring (tangential width)
    intra_cell_gap_mm: float = 0.84,        # air gap between the pair of crystals in the same cell
    scint_radial_thickness_mm: float = 6.0, # thickness in radial direction (toward/away from center)
    detector_radial_shift_mm: float = 0.0,  # uniform shift of all detector inner radii
    dtype=torch.float32,
    device: str = "cpu",
) -> Tuple[Tensor, Dict]:
    """
    Create (N,4,2) rectangles for each scintillator on 4 concentric rings.

    Output:
      - base_detector_units: shape (N,4,2)
        N = total scintillators (expected 3360)
        4 vertices per rectangle
        2 coordinates per vertex (x,y)

    Enforces non-overlap by checking:
      arc_per_cell >= (2*W + gap)
      because each cell has 2 crystals placed tangentially with a gap.
    """
    assert len(ring_inner_diameters_mm) == 4
    assert len(detectors_per_ring) == 4

    W = scint_tangential_mm
    H = scint_radial_thickness_mm
    gap = intra_cell_gap_mm

    # each "cell" contains 2 crystals (left/right tangentially)
    # their centers are separated by:
    pair_sep = W + gap
    half_pair = pair_sep / 2.0

    polys: List[Tensor] = []   # collects all crystal rectangles (each as (4,2))
    stats: List[Dict] = []     # per-ring diagnostic info

    for ring_idx, (inner_d, n_scint) in enumerate(zip(ring_inner_diameters_mm, detectors_per_ring)):
        n_cells = int(n_scint) // 2  # 2 crystals per cell
        r_in = inner_d / 2.0 + detector_radial_shift_mm
        if r_in <= 0.0:
            raise ValueError(
                f"Ring {ring_idx+1}: invalid shifted inner radius {r_in:.3f} mm "
                f"from inner diameter {inner_d:.3f} mm and detector_radial_shift_mm="
                f"{detector_radial_shift_mm:.3f} mm"
            )
        r_c = r_in + H / 2.0         # radius of rectangle centers
        dtheta = 2.0 * math.pi / n_cells
        arc_per_cell = r_c * dtheta

        required_span = 2 * W + gap
        clearance = arc_per_cell - required_span
        if clearance <= 0:
            raise ValueError(
                f"Ring {ring_idx+1}: overlap (arc/cell={arc_per_cell:.3f} < {required_span:.3f} mm)"
            )

        stats.append({
            "ring": ring_idx + 1,
            "inner_diameter_mm": inner_d,
            "detector_radial_shift_mm": detector_radial_shift_mm,
            "shifted_inner_radius_mm": r_in,
            "shifted_inner_diameter_mm": 2.0 * r_in,
            "n_scintillators": n_scint,
            "n_cells": n_cells,
            "radius_center_mm": r_c,
            "arc_per_cell_mm": arc_per_cell,
            "required_span_mm": required_span,
            "clearance_mm": clearance
        })

        for i in range(n_cells):
            th = i * dtheta

            # unit vectors at angle th:
            # r_hat points outward (radial)
            # t_hat points tangentially (perpendicular to r_hat)
            t_hat = torch.tensor([-math.sin(th), math.cos(th)], dtype=dtype, device=device)
            r_hat = torch.tensor([ math.cos(th), math.sin(th)], dtype=dtype, device=device)

            # center of the cell before splitting into 2 crystals
            slot_center = r_hat * r_c

            # place two crystals per cell at +/- half_pair along tangential direction
            for sgn in (-1.0, 1.0):
                c = slot_center + (sgn * half_pair) * t_hat

                # rectangle vertices (W tangential, H radial)
                v1 = c + (W/2.0)*t_hat + (H/2.0)*r_hat
                v2 = c - (W/2.0)*t_hat + (H/2.0)*r_hat
                v3 = c - (W/2.0)*t_hat - (H/2.0)*r_hat
                v4 = c + (W/2.0)*t_hat - (H/2.0)*r_hat

                polys.append(torch.stack([v1, v2, v3, v4], dim=0))

    base_detector_units = (
        torch.stack(polys, dim=0)
        if polys else torch.zeros((0, 4, 2), dtype=dtype, device=device)
    )

    return base_detector_units, {"rings": stats, "total_scintillators": base_detector_units.shape[0]}


def sample_apertures_uniform_ring(
    n: int,
    r_in: float,           # inner radius of metal ring (mm)
    thickness: float,      # ring thickness (mm)
    d_min: float,          # min center spacing (mm) (diagnostic check only)
    r_ap: float,           # aperture radius (mm)
    seed: int | None = None
) -> Tensor:
    """
    Place apertures evenly around the ring at the mid-thickness radius.

    Output:
      - centers_tensor: (n,2) with (x,y) aperture centers
    """
    if seed is not None:
        random.seed(seed)  # kept for reproducibility (even though placement is deterministic here)

    r_center = r_in + thickness / 2.0
    angles = [2.0 * math.pi * i / n for i in range(n)]

    centers: List[Tuple[float, float]] = []
    for a in angles:
        centers.append((r_center * math.cos(a), r_center * math.sin(a)))

    centers_tensor = torch.tensor(centers, dtype=torch.float32)

    # diagnostic: chord spacing between adjacent apertures
    if n > 1:
        chord = 2.0 * r_center * math.sin(math.pi / n)
        if chord < d_min:
            print(
                f"[warn] even-angle spacing chord={chord:.3f} mm < d_min={d_min:.3f} mm "
                f"(n={n}, r_center={r_center:.3f} mm)"
            )

    return centers_tensor


# -------------------------------
# Main (SC-SPECT: detectors fixed, collimator can rotate)
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SC-SPECT scanner geometry")
    parser.add_argument("--aperture_diam", "--aperture-diam", dest="aperture_diam", type=float, default=0.4,
                        help="Aperture diameter in mm (default: 0.4)")
    parser.add_argument("--n_apertures", type=int, default=180,
                        help="Number of apertures on HR ring (default: 180)")
    parser.add_argument("--scint_radial_mm", type=float, default=6.0,
                        help="Scintillator radial thickness in mm (default: 6.0)")
    parser.add_argument("--ring_thickness", type=float, default=2.5,
                        help="HR collimator ring thickness in mm (default: 2.5)")
    parser.add_argument(
        "--detector-radial-shift-mm",
        "--detector_radial_shift_mm",
        dest="detector_radial_shift_mm",
        type=float,
        default=0.0,
        help=(
            "Uniform radial shift in mm applied to all detector ring inner radii. "
            "Positive moves detector rings outward; negative moves inward. Ring spacing remains fixed."
        ),
    )
    _here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--output_dir", type=str, default=os.path.join(_here, "data"),
                        help="Output directory for .tensor file (default: ./data)")
    cli_args = parser.parse_args()

    # ===== HR ring parameters =====
    RING_INNER_DIAM_MM = 67.5
    RING_THICKNESS_MM  = cli_args.ring_thickness
    APERTURE_DIAM_MM   = cli_args.aperture_diam
    MIN_SPACING_MM     = 0.8
    APERTURE_COUNT     = cli_args.n_apertures
    FOV_DIAMETER_MM    = 10.0

    SEED = int(os.getenv("HR_APERTURE_SEED", "2025"))

    # ===== Detector ring parameters =====
    RING_INNER_DIAMS_MM = [260.0, 390.0, 520.0, 650.0]
    DETECTOR_RADIAL_SHIFT_MM = cli_args.detector_radial_shift_mm
    DETS_PER_RING       = [40*6*2, 40*9*2, 40*12*2, 40*15*2]  # 3360 total
    SCINT_TANGENT_MM    = 0.84
    SCINT_RADIAL_MM     = cli_args.scint_radial_mm
    INTRA_GAP_MM        = 0.84
    SCINT_AXIAL_MM      = 20.0

    print("Generating SC-SPECT base geometry")

    # --- 1) Build detector rectangles (fixed across poses) ---
    base_detector_units, geom_stats = build_sc_spect_detector_rings(
        ring_inner_diameters_mm=RING_INNER_DIAMS_MM,
        detectors_per_ring=DETS_PER_RING,
        scint_tangential_mm=SCINT_TANGENT_MM,
        intra_cell_gap_mm=INTRA_GAP_MM,
        scint_radial_thickness_mm=SCINT_RADIAL_MM,
        detector_radial_shift_mm=DETECTOR_RADIAL_SHIFT_MM,
        dtype=torch.float32,
        device="cpu",
    )

    print("\n=== Detector geometry summary ===")
    for rs in geom_stats["rings"]:
        print(
            f"Ring {rs['ring']}: inner Ø {rs['inner_diameter_mm']} mm | R_center={rs['radius_center_mm']:.2f} mm | "
            f"cells={rs['n_cells']} | scint={rs['n_scintillators']} | "
            f"arc/cell={rs['arc_per_cell_mm']:.3f} mm | clearance={rs['clearance_mm']:.3f} mm"
        )
    print(f"Total scintillators: {geom_stats['total_scintillators']} (expected 3360)")

    # --- 2) Build analytic aperture centers (base, before motion) ---
    r_in = RING_INNER_DIAM_MM / 2.0
    r_out = r_in + RING_THICKNESS_MM
    r_ap = APERTURE_DIAM_MM / 2.0

    base_aperture_centers = sample_apertures_uniform_ring(
        n=APERTURE_COUNT,
        r_in=r_in,
        thickness=RING_THICKNESS_MM,
        d_min=MIN_SPACING_MM,
        r_ap=r_ap,
        seed=SEED
    )  # (180,2)

    # --- 2b) Build tungsten plate segments (the actual attenuating geometry for raytracer) ---
    r_center_ap = r_in + RING_THICKNESS_MM / 2.0
    dtheta_cell = 2.0 * math.pi / APERTURE_COUNT
    dtheta_open = APERTURE_DIAM_MM / r_center_ap  # approximate angular opening for 0.4mm arc length

    if dtheta_open >= dtheta_cell:
        raise ValueError(
            f"Aperture angular width ({dtheta_open:.4f} rad) "
            f"is not smaller than cell width ({dtheta_cell:.4f} rad)."
        )

    plate_segments_list: List[Tensor] = []

    def make_wedge_quad(r_in_val: float, r_out_val: float, th1: float, th2: float) -> Tensor:
        """
        Create a quadrilateral approximating a ring sector between angles th1 and th2.
        Vertex order: inner-th1, outer-th1, outer-th2, inner-th2.
        """
        v1 = torch.tensor([r_in_val * math.cos(th1), r_in_val * math.sin(th1)], dtype=torch.float32)
        v2 = torch.tensor([r_out_val * math.cos(th1), r_out_val * math.sin(th1)], dtype=torch.float32)
        v3 = torch.tensor([r_out_val * math.cos(th2), r_out_val * math.sin(th2)], dtype=torch.float32)
        v4 = torch.tensor([r_in_val * math.cos(th2), r_in_val * math.sin(th2)], dtype=torch.float32)
        return torch.stack([v1, v2, v3, v4], dim=0)

    # for each cell: make left wall and right wall around the aperture opening
    for i in range(APERTURE_COUNT):
        th_center = i * dtheta_cell

        th_left_mid  = th_center - 0.5 * dtheta_cell
        th_right_mid = th_center + 0.5 * dtheta_cell

        th_open_left  = th_center - 0.5 * dtheta_open
        th_open_right = th_center + 0.5 * dtheta_open

        plate_segments_list.append(make_wedge_quad(r_in, r_out, th_left_mid,  th_open_left))
        plate_segments_list.append(make_wedge_quad(r_in, r_out, th_open_right, th_right_mid))

    plate_segments = torch.stack(plate_segments_list, dim=0)  # (360,4,2)
    print(f"Created {plate_segments.shape[0]} tungsten plate segments "
          f"({plate_segments.shape[0]//2} cells with 2 walls each).")

    # --- 3) MD5 over detectors + analytic circles (centers + radius column) ---
    centers_with_r = torch.cat(
        [base_aperture_centers, torch.full((APERTURE_COUNT, 1), r_ap)],
        dim=1
    )
    base_scanner_md5 = generate_md5_from_tensors(base_detector_units, centers_with_r)
    print(f"MD5 of base scanner (detectors + analytic apertures): {base_scanner_md5}")

    # --- 4) Motion: 2 rotations (0° and 1°), no translations ---
    # IMPORTANT:
    # - We want the collimator (apertures + tungsten segments) to rotate
    # - The detectors remain fixed (same base_detector_units used in every layout)
    n_rotations_for_motion = 2
    angle_step_deg_for_motion = 1.0
    n_shifts_grid_for_motion = [1, 1]
    shift_step_mm_for_motion = [0.0, 0.0]
    print("\nDefining motion: rotate collimator (2 steps), detectors fixed")

    motion_positions = positions_parameters(
        n_rotations_for_motion,
        angle_step_deg_for_motion,
        n_shifts_grid_for_motion,
        shift_step_mm_for_motion
    )
    n_total_positions = int(motion_positions.shape[0])
    assert n_total_positions == n_rotations_for_motion * (n_shifts_grid_for_motion[0] * n_shifts_grid_for_motion[1])

    # --- 4a) Rotate aperture centers for each pose ---
    transformed_centers_flat = transform_to_positions_2d_batch(
        motion_positions,
        base_aperture_centers.reshape(-1, 2)
    )
    transformed_centers = transformed_centers_flat.reshape(n_total_positions, APERTURE_COUNT, 2)

    # --- 4b) Rotate tungsten plate segments for each pose (THIS is what raytracer uses) ---
    transformed_segments_flat = transform_to_positions_2d_batch(
        motion_positions,
        plate_segments.reshape(-1, 2)
    )
    transformed_plate_segments = transformed_segments_flat.reshape(
        n_total_positions,
        plate_segments.shape[0],
        plate_segments.shape[1],  # 4 vertices
        2
    )

    print("motion_positions:", motion_positions.shape)
    print("base_aperture_centers:", base_aperture_centers.shape)
    print("transformed_centers:", transformed_centers.shape)
    print("plate_segments:", plate_segments.shape)
    print("transformed_plate_segments:", transformed_plate_segments.shape)

    # --- 5) Package + save layouts ---
    out_data: OutDataDict = {  # type: ignore
        "scanner MD5": base_scanner_md5,
        "paper_config": {
            "detector_rings_inner_diameters_mm": RING_INNER_DIAMS_MM,
            "detector_radial_shift_mm": DETECTOR_RADIAL_SHIFT_MM,
            "detector_rings_shifted_inner_radii_mm": [
                inner_d / 2.0 + DETECTOR_RADIAL_SHIFT_MM for inner_d in RING_INNER_DIAMS_MM
            ],
            "detector_rings_shifted_inner_diameters_mm": [
                inner_d + 2.0 * DETECTOR_RADIAL_SHIFT_MM for inner_d in RING_INNER_DIAMS_MM
            ],
            "detectors_per_ring": DETS_PER_RING,
            "scintillator_mm": [SCINT_TANGENT_MM, SCINT_RADIAL_MM, SCINT_AXIAL_MM],
            "intra_cell_gap_mm": INTRA_GAP_MM,
            "ring_inner_diameter_mm": RING_INNER_DIAM_MM,
            "ring_thickness_mm": RING_THICKNESS_MM,
            "aperture_diameter_mm": APERTURE_DIAM_MM,
            "aperture_min_spacing_mm": MIN_SPACING_MM,
            "aperture_count": APERTURE_COUNT,
            "fov_diameter_mm": FOV_DIAMETER_MM
        },
        "motion_parameters": {
            "n_rotational_steps_defined": n_rotations_for_motion,
            "rotation_step_deg": angle_step_deg_for_motion,
            "n_translational_shifts_grid": n_shifts_grid_for_motion,
            "translational_step_size_mm": shift_step_mm_for_motion,
            "generated_n_positions": n_total_positions
        },
        "layouts": {}
    }

    # store each pose as one layout entry
    for p in range(n_total_positions):
        layout_key = f"position {p:03d}"
        out_data["layouts"][layout_key] = {
            "position": motion_positions[p],                 # [angle_rad, x_mm, y_mm]
            "detector units": base_detector_units,           # FIXED
            "plate circles": {
                "centers": transformed_centers[p],           # ROTATED
                "radius_mm": r_ap
            },
            "plate segments": transformed_plate_segments[p]  # ROTATED (important for raytracer)
        }

    # filename reflects motion
    def _fmt(x: float) -> str:
        return f"{x}".replace("-", "m").replace(".", "p")

    motion_id = (
        f"rot{n_rotations_for_motion}"
        f"_ang{_fmt(angle_step_deg_for_motion)}deg"
        f"_trans{n_shifts_grid_for_motion[0]}x{n_shifts_grid_for_motion[1]}"
        f"_step{_fmt(shift_step_mm_for_motion[0])}x{_fmt(shift_step_mm_for_motion[1])}"
    )

    out_file_name = f"scanner_layouts_{base_scanner_md5}_{motion_id}.tensor"
    if cli_args.output_dir:
        os.makedirs(cli_args.output_dir, exist_ok=True)
        out_file_name = os.path.join(cli_args.output_dir, out_file_name)
    print(f"\nSaving SC-SPECT layouts to:\n  {out_file_name}")
    torch_save(out_data, out_file_name)
    print("Saved.")

    # --- 6) Visualization (cosmetic only) ---
    # NOTE: this plot shows the BASE (unrotated) drawing for quick sanity check.
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, Circle as MplCircle

    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw detectors (fixed)
    if base_detector_units.numel() > 0:
        plot_polygons_from_vertices_2d_mpl(
            base_detector_units, ax,
            facecolor='lightblue', edgecolor='blue', alpha=0.7,
            label="Detectors (Rings 1–4, 3,360)"
        )

    # Draw HR ring annulus (base)
    ann = Wedge((0, 0), r_out, 0, 360, width=(r_out - r_in),
                facecolor='0.85', edgecolor='0.4', linewidth=1.0, zorder=2)
    ax.add_patch(ann)

    # Draw analytic apertures as circles (base)
    bg = ax.figure.get_facecolor()
    for cx, cy in base_aperture_centers.tolist():
        ax.add_patch(MplCircle((cx, cy), r_ap, facecolor=bg,
                               edgecolor='black', linewidth=1.0, zorder=3))

    # FOV circle (10 mm)
    fov_r = FOV_DIAMETER_MM / 2.0
    ax.add_patch(MplCircle((0, 0), fov_r, edgecolor='red', facecolor='none',
                           linestyle='--', linewidth=2,
                           label=f'Effective FOV (D={FOV_DIAMETER_MM} mm)'))

    ax.set_aspect('equal', adjustable='box')
    max_coord_det = torch.abs(base_detector_units).max().item() if base_detector_units.numel() else 0.0
    plot_lim = max(max_coord_det, r_out + 50, fov_r + 50) * 1.05
    ax.set_xlim([-plot_lim, plot_lim])
    ax.set_ylim([-plot_lim, plot_lim])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("SC-SPECT (Top View) — HR Ring with 180 Apertures (Base Pose)")
    ax.legend(fontsize='small')
    ax.grid(True)
    plt.savefig("scspect_hr_stationary.png")
    print("Saved top-view figure to scspect_hr_stationary.png")

    # --- 7) Geometry self-checks ---
    print("\n=== GEOMETRY CHECKS ===")
    print("detectors tensor:", tuple(base_detector_units.shape))          # expect (3360,4,2)
    print("aperture centers:", tuple(base_aperture_centers.shape))        # expect (180,2)
    print("plate segments:", tuple(plate_segments.shape))                 # expect (360,4,2)
    print("n_positions:", n_total_positions)                               # expect 2

    # min center spacing among apertures
    D = torch.cdist(base_aperture_centers, base_aperture_centers)
    D.fill_diagonal_(1e9)
    dmin = float(D.min().item())
    print(f"min center spacing among apertures: {dmin:.4f} mm (>= {MIN_SPACING_MM} mm)")
    assert dmin >= MIN_SPACING_MM - 1e-6, "Aperture spacing violated!"

    # ring-wall radial clearance
    R_mid = r_in + RING_THICKNESS_MM / 2.0
    print(f"radial clearance (inner/outer): {(R_mid - r_ap) - r_in:.3f} mm / {r_out - (R_mid + r_ap):.3f} mm")

    # detector-cell clearance (no overlap)
    W = SCINT_TANGENT_MM
    H = SCINT_RADIAL_MM
    gap = INTRA_GAP_MM
    required_span = 2 * W + gap
    ring_centers = [d/2.0 + DETECTOR_RADIAL_SHIFT_MM + H/2.0 for d in RING_INNER_DIAMS_MM]
    cells = [n//2 for n in DETS_PER_RING]
    min_clear = min(r_c * (2*math.pi / n_cell) - required_span for r_c, n_cell in zip(ring_centers, cells))
    print(f"min detector cell clearance: {min_clear:.3f} mm (> 0 means no overlap)")
    assert min_clear > 0.0, "Detector cell overlap!"

    print("GEOMETRY OK ✅")
