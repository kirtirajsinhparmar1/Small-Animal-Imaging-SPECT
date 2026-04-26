from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
import torch

from typing import Tuple, Dict, Any, Sequence, Union # Added Sequence, Union
from torch import Tensor, tensor, pi, cos, sin, tan, asin, arange, stack, empty, cat, bmm, linspace, meshgrid, float32 # Added more specific torch imports
import hashlib

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def plot_polygons_from_vertices_2d_mpl(
    vertices: torch.Tensor, ax: Axes, **kwargs
):
    p = PolyCollection(vertices.tolist(), **kwargs)
    ax.add_collection(p)
    return p


def plate_random_aperture_dev(input: Tensor) -> Dict[str, Tensor]:
    """
    Generate a random aperture for a plate.
    The apertures is defined by a polygons each with 4 vertices.

    Parameters
    ----------
    input : Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    Dict[str, Tensor]
        A dictionary containing the following:

        - "all cell polygons": `rank-3` tensor, `shape`: (n_cells, 4, 2), containing the vertices of all cells
        - "merged polygons": `rank-3` `tensor, `shape`: (n_merged_polygons, 4, 2), containing the vertices of the merged polygons
    """

    n_segments = input[0]
    inner_radius = input[1]
    thickness = input[2]
    aperture_unit = input[3]
    ratio = input[4]

    rad_tan = torch.tan(torch.pi / n_segments)
    y_half_front = rad_tan * inner_radius

    m = int(2 * ratio * y_half_front / aperture_unit)
    biased_ratios = (
        torch.tensor([m, m + 1]) * aperture_unit / y_half_front * 0.5
    )

    ratio_diff_abs = torch.abs(biased_ratios - ratio)
    m = m + 1 if ratio_diff_abs[1] < ratio_diff_abs[0] else m

    actual_ratio = m * aperture_unit / y_half_front * 0.5
    min_offset = thickness * rad_tan
    y_half_back = y_half_front + min_offset
    n_middle_polygons = int(2 * (y_half_front - min_offset) / aperture_unit)

    m = m - 1 if m >= n_middle_polygons else m
    offset = y_half_front - n_middle_polygons * aperture_unit * 0.5
    y_start = -y_half_front + offset
    y_end = y_half_front - offset
    x_start = inner_radius
    x_end = inner_radius + thickness

    low_corner_polygon = torch.tensor(
        [
            [x_start, -y_half_front],
            [x_end, -y_half_back],
            [x_end, y_start],
            [x_start, y_start],
        ]
    ).unsqueeze(0)

    up_corner_polygon = torch.tensor(
        [
            [x_start, y_end],
            [x_end, y_end],
            [x_end, y_half_back],
            [x_start, y_half_front],
        ]
    ).unsqueeze(0)

    middle_polygons_vertices = torch.empty((n_middle_polygons, 4, 2))
    middle_polygons_vertices[:, [0, 3], 0] = (
        (torch.ones(n_middle_polygons) * x_start).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [1, 2], 0] = (
        (torch.ones(n_middle_polygons) * x_end).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [0, 1], 1] = (
        (torch.arange(n_middle_polygons) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    middle_polygons_vertices[:, [2, 3], 1] = (
        ((torch.arange(n_middle_polygons) + 1) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    mask = torch.ones((n_middle_polygons), dtype=torch.bool)
    mask[torch.sort(torch.randperm(n_middle_polygons)[:m])[0]] = False
    mask = torch.cat(
        (
            torch.ones((1), dtype=torch.bool),
            mask,
            torch.ones((1), dtype=torch.bool),
        )
    )
    all_cells = torch.cat(
        (low_corner_polygon, middle_polygons_vertices, up_corner_polygon),
        dim=0,
    )
    indices = torch.argwhere(mask).squeeze()
    data = torch.stack(
        (
            indices,
            torch.cat(
                (torch.tensor([100]), torch.diff(indices, dim=0))
            ).squeeze(),
            torch.cat(
                (torch.diff(indices, dim=0), torch.tensor([100]))
            ).squeeze(),
        ),
        dim=1,
    ).squeeze()
    partitions = torch.stack(
        (data[data[:, 1] > 1, 0], data[data[:, 2] > 1, 0]), dim=1
    )
    merged_polygons = torch.cat(
        (
            all_cells[partitions[:, 0], :2, :],
            all_cells[partitions[:, 1], 2:, :],
        ),
        dim=1,
    )

    print(
        f"{'Number of aperture cell':28s}:{m}\n{'Actual aperture ratio':28s}:{actual_ratio}\n"
    )
    return {
        "all cell polygons": all_cells,
        "merged polygons": merged_polygons,
        "partitions": partitions,
        "mask": mask,
    }


def plate_random_apertures(input: Tensor) -> tuple[Tensor, Tensor]:
    """
    Generate a plate with random apertures.
    The apertures is defined by a polygons each with 4 vertices.

    Parameters
    ----------
    input : torch.Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    torch.Tensor
        A tensor of shape (n_polygons, 4, 2) containing the vertices of the plate polygons.
        Each cell is defined by 4 vertices.
    """

    n_segments = input[0]
    inner_radius = input[1]
    thickness = input[2]
    aperture_unit = input[3]
    ratio = input[4]

    rad_tan = torch.tan(torch.pi / n_segments)
    y_half_front = rad_tan * inner_radius

    m = int(2 * ratio * y_half_front / aperture_unit)
    biased_ratios = (
        torch.tensor([m, m + 1]) * aperture_unit / y_half_front * 0.5
    )

    ratio_diff_abs = torch.abs(biased_ratios - ratio)
    m = m + 1 if ratio_diff_abs[1] < ratio_diff_abs[0] else m

    min_offset = thickness * rad_tan
    y_half_back = y_half_front + min_offset
    n_middle_polygons = int(2 * (y_half_front - min_offset) / aperture_unit)

    m = m - 1 if m >= n_middle_polygons else m
    offset = y_half_front - n_middle_polygons * aperture_unit * 0.5
    y_start = -y_half_front + offset
    y_end = y_half_front - offset
    x_start = inner_radius
    x_end = inner_radius + thickness

    # Calculate the actual ratio
    actual_ratio = m * aperture_unit / y_half_front * 0.5

    low_corner_polygon = torch.tensor(
        [
            [x_start, -y_half_front],
            [x_end, -y_half_back],
            [x_end, y_start],
            [x_start, y_start],
        ]
    ).unsqueeze(0)

    up_corner_polygon = torch.tensor(
        [
            [x_start, y_end],
            [x_end, y_end],
            [x_end, y_half_back],
            [x_start, y_half_front],
        ]
    ).unsqueeze(0)

    middle_polygons_vertices = torch.empty((n_middle_polygons, 4, 2))
    middle_polygons_vertices[:, [0, 3], 0] = (
        (torch.ones(n_middle_polygons) * x_start).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [1, 2], 0] = (
        (torch.ones(n_middle_polygons) * x_end).unsqueeze(1).expand(-1, 2)
    )
    middle_polygons_vertices[:, [0, 1], 1] = (
        (torch.arange(n_middle_polygons) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )
    middle_polygons_vertices[:, [2, 3], 1] = (
        ((torch.arange(n_middle_polygons) + 1) * aperture_unit + y_start)
        .unsqueeze(1)
        .expand(-1, 2)
    )

    mask = torch.ones((n_middle_polygons), dtype=torch.bool)
    mask[torch.sort(torch.randperm(n_middle_polygons)[:m])[0]] = False
    mask = torch.cat(
        (
            torch.ones((1), dtype=torch.bool),
            mask,
            torch.ones((1), dtype=torch.bool),
        )
    )
    all_cells = torch.cat(
        (low_corner_polygon, middle_polygons_vertices, up_corner_polygon),
        dim=0,
    )
    indices = torch.argwhere(mask).squeeze()
    data = torch.stack(
        (
            indices,
            torch.cat(
                (torch.tensor([100]), torch.diff(indices, dim=0))
            ).squeeze(),
            torch.cat(
                (torch.diff(indices, dim=0), torch.tensor([100]))
            ).squeeze(),
        ),
        dim=1,
    ).squeeze()
    partitions = torch.stack(
        (data[data[:, 1] > 1, 0], data[data[:, 2] > 1, 0]), dim=1
    )
    merged_polygons = torch.cat(
        (
            all_cells[partitions[:, 0], :2, :],
            all_cells[partitions[:, 1], 2:, :],
        ),
        dim=1,
    )
    return merged_polygons, actual_ratio


def plates_random_apertures(
    input: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    plates_random_apertures(input, n, step) -> Tensor
    Generate a full circle of plates with random apertures.

    Args
    ----------
    input : torch.Tensor
        A tensor of shape (5,) containing the following values:

        - n_segments: Number of segments of the plate
        - inner_radius: Inner radius of the plate
        - thickness: Thickness of the plate
        - aperture_unit: Unit size of the aperture
        - ratio: Ratio of the aperture to the inner radius

    Returns
    -------
    plates_vertices: torch.Tensor, shape (n * m, 4, 2)
        A tensor containing the vertices of the plate segments
    """

    n = int(input[0])  # number of plate segments
    # rotation angles
    rotations = torch.arange(0, n) * 2 * torch.pi / n
    # rotation matrix
    rotation_matrices = torch.stack(
        (
            torch.cos(rotations),
            -torch.sin(rotations),
            torch.sin(rotations),
            torch.cos(rotations),
        ),
        dim=1,
    ).reshape(-1, 2, 2)

    # Preallocate the plates_vertices tensor
    # shape: (0, 4, 2)
    plates_vertices = torch.empty((0, 4, 2), dtype=torch.float32)
    ratios = torch.empty((0), dtype=torch.float32)
    for idx in range(n):
        # Generate random apertures for each segment independently
        plate_vertices, ratio = plate_random_apertures(input)
        ratios = torch.cat((ratios, ratio.unsqueeze(0)), dim=0)
        # Rotate the vertices for each segment
        rotated_vertices = torch.bmm(
            rotation_matrices[idx]
            .unsqueeze(0)
            .expand(plate_vertices.shape[0] * 4, -1, -1),
            plate_vertices.view(-1, 2).unsqueeze(2),
        ).reshape(-1, 4, 2)

        # Concatenate the rotated vertices to the plates_vertices tensor
        plates_vertices = torch.cat(
            (
                plates_vertices,
                rotated_vertices,
            ),
            dim=0,
        )

    return plates_vertices, ratios.mean()


def rotate_and_repeat_4gon(
    input: torch.Tensor, n: int, step: Union[float, Tensor] # Changed type hint slightly for clarity
) -> torch.Tensor:
    """
    rotate_and_repeat_4gon(input, n, step) -> Tensor


    Rotate and repeat the vertices of the quadrilateral (polygons
    with 4 vertices).

    Args
    ----------
    input: torch.Tensor, shape (m, 4, 2)
      A tensor containing the vertices of the quadrilaterals.

    n: int
      The number of segments to repeat the vertices.
        - Example: 6

    step:
      The step size for the rotation. Unit is radian.
        - Example: `2 * torch.pi / n`


    Returns
    -------
    rotated_vertices: torch.Tensor, shape (n * m, 4, 2)

        A tensor containing the rotated and repeated vertices.
    """
    # Rotate the vertices for each segment
    m_polygons = input.shape[0]
    
    # rotation angles: shape (n * m_polygons * 4)
    rotations = (torch.arange(0, n, dtype=input.dtype, device=input.device) * step).repeat_interleave(
        4 * m_polygons
    )

    # rotation matrix: shape (n * m_polygons * 4, 2, 2)
    rotation_matrix = torch.stack(
        (
            torch.cos(rotations),
            -torch.sin(rotations),
            torch.sin(rotations),
            torch.cos(rotations),
        ),
        dim=1,
    ).reshape(-1, 2, 2)

    # tiled input: shape (n * m_polygons, 4, 2)
    # then reshape to (n * m_polygons * 4, 2, 1) for bmm
    tiled_input_reshaped = input.tile(n, 1, 1).reshape(-1, 2).unsqueeze(2)
    
    # rotate and repeat
    # output of bmm: (n * m_polygons * 4, 2, 1)
    # reshape to (n * m_polygons, 4, 2)
    output = torch.bmm(
        rotation_matrix,
        tiled_input_reshaped,
    ).reshape(-1, 4, 2)
    return output


def cell_grid_2d(input: Tensor) -> Tensor:
    """
    Generate a grid of cells in 2D space. Each cell is a rectangle defined by
    4 vertices.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(5, )`
      - `input[0]` : `float` : radial distance to the FOV center in mm
      - `input[1]` : `float` : x size of a cell in mm
      - `input[2]` : `float` : y size of a cell in mm
      - `input[3]` : `float` : number of cells in x direction
      - `input[4]` : `float` : number of cells in y direction

    Returns
    -------

    `Tensor`:
      `rank-4` tensor of shape `(M, N, 4, 2)` containing the vertices of the grid cells.
    """
    x_borders = torch.arange(int(input[3]) + 1) * input[1] + input[0]
    y_borders = (
        torch.arange(int(input[4]) + 1) * input[2] - input[2] * input[4] * 0.5
    )

    x_grid, y_grid = torch.meshgrid(x_borders, y_borders, indexing="ij")
    return torch.stack(
        [
            torch.stack([x_grid[:-1, :-1], y_grid[:-1, :-1]], dim=-1),
            torch.stack([x_grid[1:, :-1], y_grid[:-1, :-1]], dim=-1),
            torch.stack([x_grid[1:, 1:], y_grid[1:, 1:]], dim=-1),
            torch.stack([x_grid[:-1, 1:], y_grid[1:, 1:]], dim=-1),
        ],
        dim=-2,
    )


def grid_cells_random_mask_batch(input: Tensor) -> Tensor:
    """
    Generate a randomized mask for the grid cells with given occupancy.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(4, )`
      - `input[0]` : `float` : ratio, number of occupied cells over total
      number of cells of a panel
      - `input[1]` : `float` : number of cells in x direction
      - `input[2]` : `float` : number of cells in y direction
      - `input[3]` : `float` : batch size

    Returns
    -------

    `Tensor`:
      `rank-2` tensor of shape `(input[3], input[1] * input[2])` containing the mask for the grid cells.
    """
    n_total = int(input[1] * input[2])
    shape = (int(input[3]), int(input[1:3].prod()))
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[
        torch.arange(int(input[3])).unsqueeze(-1),
        torch.multinomial(
            torch.ones(shape, dtype=torch.float),
            int(torch.prod(input[:-1])),
            replacement=False,
        ),
    ] = True
    return mask


def grid_cells_random_full_mask_batch(input: Tensor) -> Tensor:
    """
    Generate a randomized mask for the grid cells with given occupancy. \\
    The last column of the mask is fully occupied.

    Parameters
    ----------
    `input` : `Tensor`

      `rank-1` `Tensor` of shape `(4, )`
      - `input[0]` : `float` : ratio, number of occupied cells over total
      number of cells of a panel
      - `input[1]` : `float` : number of cells in x direction
      - `input[2]` : `float` : number of cells in y direction
      - `input[3]` : `float` : batch size

    Returns
    -------

    `Tensor`:
      `rank-2` tensor of shape `(input[3], input[1] * input[2])` containing the mask for the grid cells.
    """

    cells_mask = grid_cells_random_mask_batch(input).view(
        int(input[3]), int(input[1]), int(input[2])
    )
    full_column_mask = torch.ones(
        (int(input[3]), 1, int(input[2])), dtype=torch.bool
    )
    concatenated_mask = torch.cat(
        [
            cells_mask,
            full_column_mask,
        ],
        dim=1,
    )
    return concatenated_mask


def single_crystal(input: Tensor) -> Tensor:
    """
    Generate a single crystal polygon based on the size.

    Parameters
    ----------
    input : Tensor
        `rank-1` tensor of shape (2, ) representing the
        size definition of the crystal.

    Returns
    -------
    output : Tensor
        `rank-3` tensor of shape (1, 4, 2) representing the vertices of the crystal.
    """

    return torch.tensor(
        [
            [-input[0] / 2, -input[1] / 2],
            [input[0] / 2, -input[1] / 2],
            [input[0] / 2, input[1] / 2],
            [-input[0] / 2, input[1] / 2],
        ]
    ).unsqueeze(0)


def detector_units_panels(
    cell_grid: Tensor,
    inner_cell: Tensor,
    outer_cell: Tensor,
    **kwargs,
) -> Tensor:
    """
    Generates a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    cell_grid : Tensor, shape (N_r, N_c, 4, 2)
        `rank-4` tensor containing the vertices of the cell grid of a panel.\\
        `N_r` is the number of rows of cells in a grid \\
        `N_c` is the number of columns of cells in a grid \\

    inner_cell : Tensor, shape (M, 4, 2)
        `rank-3` tensor containing the vertices of detector units in a single
        cell in the inner layers.\\
        `M` is the number of detector units

    outer_cell : Tensor, shape (L, 4, 2)
        `rank-3` tensor containing the vertices of the outer detector units 
        in a single cell in the outermost layer.\\
        `L` is the number of outer detector units. `Default` is `1`.

    Returns
    -------
    out: Tensor
        A tensor containing the vertices of the detector units.

    """
    n_detector_panels = kwargs.get("n_detector_panels", 6)
    inner_cell_columns = kwargs.get("inner_cell_columns", [0, 1, 2, 3, 4, 5, 6])
    outer_cell_columns = kwargs.get("outer_cell_columns", [7])

    inner_cells_array = cell_grid[inner_cell_columns, :, :, :].view(-1, 4, 2)
    outer_cells_array = cell_grid[outer_cell_columns, :, :, :].view(-1, 4, 2)
    inner_cell_centers = inner_cells_array.mean(dim=-2)
    outer_cell_centers = outer_cells_array.mean(dim=-2)
    inner_units = inner_cell.repeat(
        inner_cell_centers.shape[0], 1, 1
    ) + inner_cell_centers.repeat_interleave(
        inner_cell.shape[0], dim=0
    ).unsqueeze(
        1
    ).expand(
        -1, 4, -1
    )
    outer_units = outer_cell.repeat(
        outer_cell_centers.shape[0], 1, 1
    ) + outer_cell_centers.repeat_interleave(
        outer_cell.shape[0], dim=0
    ).unsqueeze(
        1
    ).expand(
        -1, 4, -1
    )
    panel_detector_units = torch.cat(
        (
            inner_units,
            outer_units,
        ),
        dim=0,
    ).view(-1, 4, 2)
    out = rotate_and_repeat_4gon(
        panel_detector_units.view(-1, 4, 2),
        n=n_detector_panels,
        step=torch.pi / n_detector_panels * 2,
    )
    return out


def cell_detector_units(crystal_size: Tensor, centers: Tensor) -> Tensor:
    """
    Generates a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    crystal_size : Tensor, shape (6,)
        A tensor containing the parameters for the scanner layout.

        The parameters are as follows:
        - `crystal_size[0]`: detector unit size in x direction
        - `crystal_size[1]`: detector unit size in y direction

    centers : Tensor, shape (N, 2)
        A tensor containing the center coordinates of the detector units.
    Returns
    -------
    out: Tensor
        A tensor containing the vertices of the detector units.

    """

    out = single_crystal(crystal_size).repeat(
        centers.size(0), 1, 1
    ) + centers.unsqueeze(1).repeat_interleave(4, dim=1)
    return out


def scanner_layout_random_last_full(
    input: Tensor, **kwargs
) -> Tuple[Tensor, Tensor]:
    """
    Generate a randomized scanner layout based on the input parameters.

    Parameters
    ----------
    input : Tensor
        `rank-1` tensor of shape (5,) containing the following values:
        - `input[0]`: ratio of the aperture to plate tangential length
        - `input[1]`: ratio of the occupied cells over total number of cells of a panel
        - `input[2]`: inner radius of the plate
        - `input[3]`: detector array to plate distance

    kwargs : Keyword Arguments, optional

        Additional parameters for the function. The following parameters can be provided:

        | Property                     | Description                                          | Type           | Default Value     |
        |:-----------------------------|:-----------------------------------------------------|:---------------|:------------------|
        | `inner_unit_size`            | Size of the inner layer detector unit in mm          |`Tuple`         |`(2.4, 2.4)`       |
        | `outer_unit_size`            | Size of the outer layer detector unit in mm          |`Tuple`         |`(3.0, 3.0)`       |
        | `inner_unit_centers_cell`    | Center coordinates of the inner layer detector units |`Tensor (N, 2)` |`tensor([[0, 0]])` |
        | `outer_unit_centers_cell`    | Center coordinates of the outer layer detector units |`Tensor (L, 2)` |`tensor([[0, 0]])` |
        | `cell_size`                  | Size of the cells in mm                              |`Tuple`         |`(3.36, 3.36)`     |
        | `n_cells`                    | Number of cells in x and y direction                 |`Tuple`         |`(8, 32)`          |
        | `n_plate_segments`           | Number of plate segments                             |`int`           |`6`                |
        | `n_detector_panels`          | Number of detector panels                            |`int`           |`6`                |
        | `plate_thickness`            | Thickness of the plate in mm                         |`float`         |`2.0`              |
        | `aperture_unit_size`         | Size of the aperture unit in mm                      |`float`         |`2.0`              |

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing:
        - `rank-3` tensor of shape (M, 4, 2) containing the vertices of plate segments
        - `rank-3` tensor of shape (N, 4, 2) containing the vertices of the detector units
    """
    inner_unit_size = kwargs.get("inner_unit_size", (2.4, 2.4))
    inner_unit_centers_cell = kwargs.get(
        "inner_unit_centers_cell", torch.tensor([[0, 0]]).view(-1, 2)
    )
    outer_unit_size = kwargs.get("outer_unit_size", (3.0, 3.0))
    outer_unit_centers_cell = kwargs.get(
        "outer_unit_centers_cell", torch.tensor([[0, 0]]).view(-1, 2)
    )
    cell_size = kwargs.get("cell_size", (3.36, 3.36))
    n_cells = kwargs.get("n_cells", (8, 32))
    n_plate_segments = kwargs.get("n_plate_segments", 6)
    n_detector_panels = kwargs.get("n_detector_panels", 6)
    plate_thickness = kwargs.get("plate_thickness", 2.0)
    aperture_unit_size = kwargs.get("aperture_unit_size", 2.0)

    plate_segments, aperture_ratio = plates_random_apertures(
        torch.tensor(
            [
                n_plate_segments,
                input[2],
                plate_thickness,
                aperture_unit_size,
                input[0],
            ]
        )
    )
    inner_units_cell_array = cell_detector_units(
        torch.tensor(inner_unit_size),
        inner_unit_centers_cell,
    )
    outer_units_cell_array = cell_detector_units(
        torch.tensor(outer_unit_size),
        outer_unit_centers_cell,
    )

    panel_cell_grid = cell_grid_2d(
        torch.tensor(
            [
                float(input[2] + input[3] + plate_thickness),
                cell_size[0],
                cell_size[1],
                n_cells[0],
                n_cells[1],
            ]
        )
    )

    detector_units = detector_units_panels(
        panel_cell_grid,
        inner_units_cell_array,
        outer_units_cell_array,
        **kwargs,
    )
    scanner_cells_mask = grid_cells_random_full_mask_batch(
        torch.tensor([input[1], n_cells[0] - 1, n_cells[1], n_detector_panels])
    )

    return plate_segments, detector_units[scanner_cells_mask.view(-1)]


def rotate_polygon_batch(angles: Tensor, polygons: Tensor) -> Tensor:
    """
    Rotate a batch of polygons by given angles.

    Parameters
    -----------

        angles : Tensor, shape (`N_Batch`, 1)

            The angles in radians to rotate the polygons.

        polygons : Tensor

            The polygons to rotate, shape (`N_Batch`, 4, 2).

    Returns
    -------

        Tensor: The rotated polygons, shape (`N_Batch`, 4, 2).
    """
    rotation_matrix = (
        torch.stack(
            (
                torch.cos(angles),
                -torch.sin(angles),
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=1,
        )
        .view(-1, 2, 2)
        .repeat(polygons.size(1), 1, 1)
    )
    return torch.bmm(
        rotation_matrix.view(-1, 2, 2),
        polygons.view(-1, 2, 1),
    ).view(
        polygons.size(0), polygons.size(1), 2
    )  # (N, 4, 2)


def rotate_vertices_2d_batch(angle: Tensor, vertices_batch: Tensor) -> Tensor:
    """
    Rotate a batch of 2D vertices by given angles.

    Parameters
    ----------
    angle : Tensor
        The angles in radians to rotate the vertices.

    vertices_batch : Tensor
        The vertices to rotate, shape (..., 2).

    Returns
    -------
    Tensor
        The rotated vertices, shape (..., 2).
    """
    # Get number of vertices
    n_vertices = int(torch.prod(torch.tensor(vertices_batch.size()[:-1])))
    rotation_matrix = (
        torch.stack(
            (
                torch.cos(angle),
                -torch.sin(angle),
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=0,
        )
        .view(-1, 2, 2)
        .repeat(n_vertices, 1, 1)
    )  # (N, 2, 2)
    return torch.bmm(rotation_matrix, vertices_batch.view(-1, 2, 1)).view(
        vertices_batch.shape
    )


def generate_sha256_from_tensors(*tensors):
    hash_obj = hashlib.sha256()
    for tensor_item in tensors: # Renamed to avoid conflict with torch.tensor
        hash_obj.update(tensor_item.numpy().tobytes())
    return hash_obj.hexdigest()


def generate_md5_from_tensors(*tensors):
    hash_obj = hashlib.md5()
    for tensor_item in tensors: # Renamed to avoid conflict with torch.tensor
        hash_obj.update(tensor_item.numpy().tobytes())
    return hash_obj.hexdigest()


# --- Functions from the second script provided by user (for transforming layouts) ---
OutDataDict = Dict[str, Union[str, Dict]]

def positions_parameters(
     n_rotations: int,
    angle_step_deg: float,  # The NEW parameter we are adding
    n_shifts: Sequence[int],
    shift_step: Sequence[float]
):            
    """
    Generate a set of angles and shifts for the scanner layout.
    """
    angle_step_rad = angle_step_deg * pi / 180.0
    angles = arange(0, n_rotations, dtype=torch.float32) * angle_step_rad
    shifts_indices = stack(
        meshgrid([arange(n_shifts[0]), arange(n_shifts[1])]),
        dim=-1,
    )
    if n_shifts[0] > 1 and n_shifts[1] > 1 : # Avoid flip if one dim is 1
      shifts_indices[1::2, :, 1] = shifts_indices[1::2, :, 1].flip(1)
    
    shifts = shifts_indices * tensor(shift_step, dtype=float32)
    shifts = shifts - (shifts.view(-1, 2)).to(dtype=float32).mean(0)
    return cat(
        [
            angles.repeat(n_shifts[0] * n_shifts[1]).view(-1, 1),
            shifts.view(-1, 2).repeat_interleave(n_rotations, dim=0),
        ],
        dim=1,
    )


def translation_matrix_2d_batch(translations: Tensor):
    n = translations.size(0)
    # Get the translation matrix in (2+1)-D

    translation_matrix = (
        tensor([[1, 0], [0, 1], [0, 0]], dtype=float32)
        .unsqueeze(0)
        .repeat(n, 1, 1)
    )  # (n, 3, 2)
    translation_matrix = cat(
        [
            translation_matrix,
            cat(
                [translations, tensor([[1]], dtype=float32).repeat(n, 1)], dim=1
            ).unsqueeze(2),
        ],
        dim=2,
    )  # (n, 3, 3)
    return translation_matrix


def rotation_matrix_2d_batch(angles: Tensor):

    rotation_matrix = stack(
        (
            cos(angles),
            -sin(angles),
            sin(angles),
            cos(angles),
        ),
        dim=1,
    ).view(-1, 2, 2)
    return rotation_matrix


def transform_to_positions_2d_batch(positions: Tensor, points_2d_batch: Tensor):
    # positions: (m, 3)
    # points_2d_batch, shape: (n, 2)
    m = positions.shape[0]
    n = points_2d_batch.shape[0]
    # Get the rotation matrix in 2-D
    rotation_matrix_batch = rotation_matrix_2d_batch(
        positions[:, 0]
    )  # (m, 2, 2)
    rotation_matrix_batch = rotation_matrix_batch.unsqueeze(1).expand(
        -1, n, -1, -1
    )  # (m, n, 2, 2)
    # Expand the points to (m, n, 2, 1)

    points = (
        points_2d_batch.unsqueeze(0).view(1, n, 2, 1).expand(m, n, 2, 1)
    )  # (m, n, 2, 1)
    # Perform the rotation
    rotated_points = bmm(
        rotation_matrix_batch.reshape(-1, 2, 2), points.reshape(-1, 2, 1)
    )  # (m*n, 2, 1)

    # Pad the points with one on the third dimension
    rotated_points = cat(
        [
            rotated_points,
            tensor([[[1]]], dtype=float32).repeat(
                rotated_points.shape[0], 1, 1
            ),
        ],
        dim=1,
    ).view(
        -1, 3, 1
    )  # (m*n, 3, 1)
    # Get the translation matrix in (2+1)-D
    translation_matrix_batch = translation_matrix_2d_batch(
        positions[:, 1:]
    )  # (m, 3, 3)
    # Expand the translation matrix to (m, n, 3, 3)
    translation_matrix_batch = translation_matrix_batch.unsqueeze(1).expand(
        -1, n, -1, -1
    )  # (m, n, 3, 3)
    # Perform the translation
    return bmm(
        translation_matrix_batch.reshape(-1, 3, 3),
        rotated_points.reshape(-1, 3, 1),
    )[:, :2, 0].view(
        m, n, 2
    )  # (m, n, 2)

# --- NEW MPH-SPECT GEOMETRY FUNCTION ---
def generate_mph_spect_geometry(pinhole_aperture_mm: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the 2D geometry for an MPH-SPECT scanner.

    Parameters
    ----------
    pinhole_aperture_mm : float
        The diameter of the pinhole aperture in millimeters.
        Default is 0.25 mm.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - detector_units: Tensor of shape (N_scintillators, 4, 2)
        - plate_segments: Tensor of shape (N_collimator_segments, 4, 2)
    """

    # Detector parameters
    N_DETECTOR_PANELS = 15
    N_SCINT_PER_PANEL = 75
    SCINT_TANGENTIAL_WIDTH = 0.84  # mm
    SCINT_RADIAL_THICKNESS = 6.0   # mm
    R_DET_INNER_FACE = 150.0       # mm
    R_DET_OUTER_FACE = R_DET_INNER_FACE + SCINT_RADIAL_THICKNESS

    # Collimator parameters
    N_PINHOLES = 15
    R_COLL_INNER = 22.0  # mm (44mm internal diameter / 2)
    R_COLL_OUTER = 26.0  # mm (22mm inner radius + 4mm wall thickness)
    R_PINHOLE_CENTER = (R_COLL_INNER + R_COLL_OUTER) / 2.0  # 24.0 mm
    PINHOLE_OPENING_ANGLE_DEG = 30.0
    
    r_pinhole = pinhole_aperture_mm / 2.0 # radius of pinhole aperture
    theta_ph_rad = tensor(PINHOLE_OPENING_ANGLE_DEG / 2.0 * pi / 180.0) # half opening angle in radians

    # --- Generate Detector Units ---
    # Create vertices for one panel (centered on positive x-axis, facing origin)
    # Panel tangential extent: N_SCINT_PER_PANEL * SCINT_TANGENTIAL_WIDTH
    panel_scintillators = empty((N_SCINT_PER_PANEL, 4, 2), dtype=torch.float32)
    for i in range(N_SCINT_PER_PANEL):
        # y-coordinate of the center of the i-th scintillator in the panel
        y_center = (i - (N_SCINT_PER_PANEL - 1) / 2.0) * SCINT_TANGENTIAL_WIDTH
        y_min = y_center - SCINT_TANGENTIAL_WIDTH / 2.0
        y_max = y_center + SCINT_TANGENTIAL_WIDTH / 2.0
        
        # Vertices in CCW order: (front_ymin, back_ymin, back_ymax, front_ymax)
        # Front face is at R_DET_INNER_FACE (smaller x for panel on positive x-axis)
        # Back face is at R_DET_OUTER_FACE (larger x)
        panel_scintillators[i, 0, :] = tensor([R_DET_INNER_FACE, y_min])
        panel_scintillators[i, 1, :] = tensor([R_DET_OUTER_FACE, y_min])
        panel_scintillators[i, 2, :] = tensor([R_DET_OUTER_FACE, y_max])
        panel_scintillators[i, 3, :] = tensor([R_DET_INNER_FACE, y_max])

    detector_units = rotate_and_repeat_4gon(
        input=panel_scintillators,
        n=N_DETECTOR_PANELS,
        step=2 * pi / N_DETECTOR_PANELS
    )

    # --- Generate Collimator Plate Segments (Solid Parts) ---
    # For knife-edge pinhole at R_PINHOLE_CENTER, opening angle theta_ph_rad (half angle)
    # Pinhole channel width at R_COLL_INNER:
    # dist_center_to_inner_surface = R_PINHOLE_CENTER - R_COLL_INNER
    # y_inner_halfwidth = r_pinhole + dist_center_to_inner_surface * tan(theta_ph_rad)
    
    # For a knife-edge pinhole, the opening is defined by the aperture diameter and the opening angle.
    # The "void" starts at the pinhole aperture itself.
    # The angle subtended by the pinhole channel at the origin can be approximated.
    # Half-width of the pinhole channel where it intersects the inner radius R_COLL_INNER
    # This assumes the pinhole aperture is *at* R_PINHOLE_CENTER.
    # The knife edge extends inwards and outwards from R_PINHOLE_CENTER along lines defined by theta_ph_rad.
    # Let's calculate the projected width of the channel at R_COLL_INNER and R_COLL_OUTER
    
    # Point where the inner cone of the pinhole (towards origin) intersects R_COLL_INNER
    # (assuming pinhole aperture r_pinhole is at radial distance R_PINHOLE_CENTER)
    # The channel widens as it goes from R_PINHOLE_CENTER towards R_COLL_INNER
    channel_half_width_at_R_inner = r_pinhole + (R_PINHOLE_CENTER - R_COLL_INNER) * tan(theta_ph_rad)
    angular_half_width_at_R_inner = asin(channel_half_width_at_R_inner / R_COLL_INNER)

    # Point where the outer cone of the pinhole (away from origin) intersects R_COLL_OUTER
    # The channel widens as it goes from R_PINHOLE_CENTER towards R_COLL_OUTER
    channel_half_width_at_R_outer = r_pinhole + (R_COLL_OUTER - R_PINHOLE_CENTER) * tan(theta_ph_rad)
    angular_half_width_at_R_outer = asin(channel_half_width_at_R_outer / R_COLL_OUTER)

    # Angle of the axis of each pinhole.
    # First pinhole axis is at angle 0.
    # The solid segment is between the end of one pinhole's void and the start of the next.
    
    # For a segment starting after pinhole 0 (axis at 0 rad) and ending before pinhole 1 (axis at 2*pi/N_PINHOLES):
    # Start angle of the segment = axis_pinhole_0 + effective_angular_half_width_of_void
    # End angle of the segment   = axis_pinhole_1 - effective_angular_half_width_of_void

    # Using the wider of the two angular half-widths to define the void for simplicity.
    # This creates a trapezoidal void. The solid segments are between these.
    # A solid segment starts at angle_start and ends at angle_end for the first segment.
    # angle_start_first_segment_inner = angular_half_width_at_R_inner
    # angle_end_first_segment_inner   = (2 * pi / N_PINHOLES) - angular_half_width_at_R_inner
    
    # angle_start_first_segment_outer = angular_half_width_at_R_outer
    # angle_end_first_segment_outer   = (2 * pi / N_PINHOLES) - angular_half_width_at_R_outer

    # Ensure there's material left
    if angular_half_width_at_R_inner >= (pi / N_PINHOLES):
        raise ValueError(f"Pinhole aperture {pinhole_aperture_mm}mm and opening angle {PINHOLE_OPENING_ANGLE_DEG} deg "
                         "are too large for the inner collimator radius, "
                         "resulting in no solid collimator material between pinholes at the inner surface.")
    if angular_half_width_at_R_outer >= (pi / N_PINHOLES):
         raise ValueError(f"Pinhole aperture {pinhole_aperture_mm}mm and opening angle {PINHOLE_OPENING_ANGLE_DEG} deg "
                         "are too large for the outer collimator radius, "
                         "resulting in no solid collimator material between pinholes at the outer surface.")


    # Create vertices for one solid collimator segment
    # This segment is between the void of pinhole 0 (centered at angle 0)
    # and the void of pinhole 1 (centered at angle 2*pi/N_PINHOLES)
    
    # Vertices for the first segment (angles relative to x-axis)
    # Point 0: Inner radius, start angle (after void of pinhole 0)
    # Point 1: Outer radius, start angle (after void of pinhole 0)
    # Point 2: Outer radius, end angle (before void of pinhole 1)
    # Point 3: Inner radius, end angle (before void of pinhole 1)
    
    segment_vertices = empty((1, 4, 2), dtype=torch.float32) # Shape (1, 4, 2)

    start_angle_inner = angular_half_width_at_R_inner
    end_angle_inner   = (2 * pi / N_PINHOLES) - angular_half_width_at_R_inner
    start_angle_outer = angular_half_width_at_R_outer
    end_angle_outer   = (2 * pi / N_PINHOLES) - angular_half_width_at_R_outer

    # CCW order
    segment_vertices[0, 0, :] = tensor([R_COLL_INNER * cos(start_angle_inner), 
                                        R_COLL_INNER * sin(start_angle_inner)])
    segment_vertices[0, 1, :] = tensor([R_COLL_OUTER * cos(start_angle_outer), 
                                        R_COLL_OUTER * sin(start_angle_outer)])
    segment_vertices[0, 2, :] = tensor([R_COLL_OUTER * cos(end_angle_outer), 
                                        R_COLL_OUTER * sin(end_angle_outer)])
    segment_vertices[0, 3, :] = tensor([R_COLL_INNER * cos(end_angle_inner), 
                                        R_COLL_INNER * sin(end_angle_inner)])
    
    plate_segments = rotate_and_repeat_4gon(
        input=segment_vertices,
        n=N_PINHOLES, # Number of solid segments = number of pinholes
        step=2 * pi / N_PINHOLES
    )
    
    return detector_units, plate_segments



def generate_mph_spect_geometry(
    pinhole_diameter_mm: float = 3.0,
    n_pinholes: int = 18,
    collimator_ring_radius_mm: float = 215.0,
    pinhole_to_detector_distance_mm: float = 542.0,
    scint_tangential_mm: float = 3.5,
    scint_radial_thickness_mm: float = 6.0,
    scintillators_per_pinhole_projection_arc_tangential: int = 50, # Number of scintillators to cover a reasonable arc per pinhole
    pinhole_channel_length_mm: float = 20.0 # Assuming a reasonable length for the straight pinhole channel
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the 2D geometry for the MPH-SPECT scanner based on specs.
    Pinhole: Straight hole.
    Detector: Single ring.

    Parameters
    ----------
    pinhole_diameter_mm : float
        Diameter of the straight pinhole.
    n_pinholes : int
        Number of pinholes evenly distributed.
    collimator_ring_radius_mm : float
        Radius of the collimator ring where pinholes are located.
    pinhole_to_detector_distance_mm : float
        Distance from the pinhole center to the inner face of the detector ring.
    scint_tangential_mm : float
        Tangential width of each scintillator crystal.
    scint_radial_thickness_mm : float
        Radial thickness of each scintillator crystal.
    scintillators_per_pinhole_projection_arc_tangential : int
        Approximate number of scintillators to place in the tangential direction
        to cover the projection from a single pinhole. This is an estimation.
    pinhole_channel_length_mm : float
        The length of the straight pinhole channel. This also defines the
        collimator "plate" thickness at the pinhole location.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - detector_units: Tensor of shape (N_scintillators, 4, 2)
        - plate_segments: Tensor of shape (N_collimator_segments, 4, 2)
    """
    R_COLL = collimator_ring_radius_mm
    R_DET_INNER_FACE = R_COLL + pinhole_to_detector_distance_mm
    R_DET_OUTER_FACE = R_DET_INNER_FACE + scint_radial_thickness_mm

    pinhole_radius_mm = pinhole_diameter_mm / 2.0

    # --- Generate Collimator Plate Segments (Solid Parts) ---
    # For straight holes, the collimator is a ring of certain thickness.
    # The "plate_segments" are the solid parts between the pinhole openings.
    
    # Angular width of one pinhole opening at the collimator ring radius
    # For a straight hole, the opening is just the diameter.
    # This is the angle subtended by the pinhole diameter at the *center* of the collimator ring.
    # angular_width_of_pinhole_opening = 2 * torch.asin(pinhole_radius_mm / R_COLL)
    angular_width_of_pinhole_opening = 2 * torch.asin(torch.tensor(pinhole_radius_mm / R_COLL, dtype=torch.float32)) # CORRECTED LINE

    angle_step_pinholes = 2 * torch.pi / n_pinholes

    if angular_width_of_pinhole_opening >= angle_step_pinholes:
        raise ValueError(
            f"Pinhole diameter {pinhole_diameter_mm}mm is too large for {n_pinholes} pinholes "
            f"on a ring of radius {R_COLL}mm. No solid material between pinholes."
        )

    # Define one solid collimator segment
    # It starts after one pinhole opening and ends before the next.
    # Collimator thickness here is approximated by pinhole_channel_length_mm
    # The "front" of the collimator is R_COLL - pinhole_channel_length_mm / 2
    # The "back" of the collimator is R_COLL + pinhole_channel_length_mm / 2
    R_COLL_SEG_INNER = R_COLL - pinhole_channel_length_mm / 2.0
    R_COLL_SEG_OUTER = R_COLL + pinhole_channel_length_mm / 2.0


    angle_start_first_segment = angular_width_of_pinhole_opening / 2.0
    angle_end_first_segment = angle_step_pinholes - (angular_width_of_pinhole_opening / 2.0)

    segment_vertices = torch.empty((1, 4, 2), dtype=torch.float32)
    segment_vertices[0, 0, :] = torch.tensor([R_COLL_SEG_INNER * torch.cos(angle_start_first_segment),
                                              R_COLL_SEG_INNER * torch.sin(angle_start_first_segment)])
    segment_vertices[0, 1, :] = torch.tensor([R_COLL_SEG_OUTER * torch.cos(angle_start_first_segment),
                                              R_COLL_SEG_OUTER * torch.sin(angle_start_first_segment)])
    segment_vertices[0, 2, :] = torch.tensor([R_COLL_SEG_OUTER * torch.cos(angle_end_first_segment),
                                              R_COLL_SEG_OUTER * torch.sin(angle_end_first_segment)])
    segment_vertices[0, 3, :] = torch.tensor([R_COLL_SEG_INNER * torch.cos(angle_end_first_segment),
                                              R_COLL_SEG_INNER * torch.sin(angle_end_first_segment)])

    plate_segments = rotate_and_repeat_4gon(
        input=segment_vertices,
        n=n_pinholes,
        step=angle_step_pinholes
    )

    # --- Generate Detector Units ---
    # Detectors form a continuous ring.
    # Total tangential length of detector ring approx: 2 * pi * R_DET_INNER_FACE
    # Total number of scintillators needed for a full ring:
    total_circumference_det_ring = 2 * torch.pi * (R_DET_INNER_FACE + R_DET_OUTER_FACE) / 2.0 # Mid-radius
    n_total_scintillators_ring = int(torch.ceil(torch.tensor(total_circumference_det_ring / scint_tangential_mm, dtype=torch.float32)))    
    
    # Adjust to be an even number for symmetry if desired, or multiple of n_pinholes
    if n_total_scintillators_ring % 2 != 0:
        n_total_scintillators_ring +=1
    
    all_detector_units_list = []
    
    angle_per_scint = 2 * torch.pi / n_total_scintillators_ring

    for i in range(n_total_scintillators_ring):
        scint_center_angle = i * angle_per_scint # This is likely a float
        
        # Calculate vertices for one scintillator crystal oriented radially
        # For a crystal in a ring, its "tangential" dimension defines an arc.
        # We approximate with a polygon whose vertices lie on the inner/outer radii.
        
        half_angular_width_scint = scint_tangential_mm / (2.0 * ((R_DET_INNER_FACE + R_DET_OUTER_FACE) / 2.0) ) # This is likely a float
        
        angle_s0_float = scint_center_angle - half_angular_width_scint
        angle_s1_float = scint_center_angle + half_angular_width_scint

        # Convert angles to tensors before using with torch.cos/sin
        angle_s0 = torch.tensor(angle_s0_float, dtype=torch.float32)
        angle_s1 = torch.tensor(angle_s1_float, dtype=torch.float32)

        crystal_vertices = torch.empty((1, 4, 2), dtype=torch.float32)
        # The error was occurring on the next line:
        crystal_vertices[0, 0, :] = torch.tensor([R_DET_INNER_FACE * torch.cos(angle_s0), R_DET_INNER_FACE * torch.sin(angle_s0)])
        crystal_vertices[0, 1, :] = torch.tensor([R_DET_OUTER_FACE * torch.cos(angle_s0), R_DET_OUTER_FACE * torch.sin(angle_s0)])
        crystal_vertices[0, 2, :] = torch.tensor([R_DET_OUTER_FACE * torch.cos(angle_s1), R_DET_OUTER_FACE * torch.sin(angle_s1)])
        crystal_vertices[0, 3, :] = torch.tensor([R_DET_INNER_FACE * torch.cos(angle_s1), R_DET_INNER_FACE * torch.sin(angle_s1)])
        all_detector_units_list.append(crystal_vertices)

    if not all_detector_units_list:
        detector_units = torch.empty((0, 4, 2), dtype=torch.float32)
    else:
        detector_units = torch.cat(all_detector_units_list, dim=0)
        
    return detector_units, plate_segments

    import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import torch
from torch import Tensor, stack
import math

import torch

def generate_linear_multi_plate_geometry(
    aperture_width_x=140.0,
    aperture_height_y=6.0,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    n_detector_layers=4,
    detector_height_mm=50.0,
    detector_spacing_mm=40.0,
    detector_offset_mm=30.0,
):
    """
    Generate a linear MPH-SPECT geometry with:
      - One rectangular aperture (collimator) plate centered at (0,0)
      - Several detector plates stacked below in Y (each offset)
    Returns:
        detector_units: (N_layers, 4, 2)
        plate_segments: (1, 4, 2)
    """

    # === Aperture Plate ===
    half_w, half_h = aperture_width_x / 2, aperture_height_y / 2
    plate = torch.tensor([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ]).unsqueeze(0)  # (1, 4, 2)

    # === Detector Plates ===
    detector_layers = []
    det_half_w = aperture_width_x / 2
    det_half_h = detector_height_mm / 2

    for i in range(n_detector_layers):
        # Stack downward from aperture
        offset_y = -(i + 1) * detector_spacing_mm - detector_offset_mm
        rect = torch.tensor([
            [-det_half_w, offset_y - det_half_h],
            [ det_half_w, offset_y - det_half_h],
            [ det_half_w, offset_y + det_half_h],
            [-det_half_w, offset_y + det_half_h],
        ])
        detector_layers.append(rect)

    detectors = torch.stack(detector_layers)  # (N_layers, 4, 2)

    return detectors, plate

def generate_uniform_linear_geometry(
    aperture_width_x=140.0,
    aperture_height_y=6.0,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    n_detector_layers=4,
    layer_spacing_mm=40.0,
    crystal_size=(2.0, 6.0),
    detectors_x=64,
    detectors_y=16,
):
    """
    Generate planar MPH-SPECT geometry with:
      - One aperture plate containing pinholes (centered)
      - Uniform rectangular detector layers below the aperture
    """
    import torch

    crystal_w, crystal_h = crystal_size

    # === 1. Define aperture plate ===
    aperture_y = 0.0  # plate centered at y=0
    aperture_top = aperture_y + aperture_height_y / 2
    aperture_bottom = aperture_y - aperture_height_y / 2

    # === 2. Define pinholes along aperture plate ===
    n_pinholes = int(aperture_width_x / (pinhole_diameter_mm / aperture_open_ratio))
    pinhole_spacing = aperture_width_x / n_pinholes
    pinhole_x_positions = torch.linspace(
        -aperture_width_x / 2 + pinhole_spacing / 2,
        aperture_width_x / 2 - pinhole_spacing / 2,
        n_pinholes,
    )
    pinhole_y_positions = torch.full_like(pinhole_x_positions, aperture_y)
    pinhole_r = pinhole_diameter_mm / 2.0
    circle = torch.stack(
        [
            torch.cos(torch.linspace(0, 2 * torch.pi, 16)),
            torch.sin(torch.linspace(0, 2 * torch.pi, 16)),
        ],
        dim=1,
    )
    pinhole_vertices = torch.stack(
        [
            pinhole_x_positions[:, None] + pinhole_r * circle[None, :, 0],
            pinhole_y_positions[:, None] + pinhole_r * circle[None, :, 1],
        ],
        dim=-1,
    )

    base_plate_segments = pinhole_vertices

    # === 3. Generate detector layers ===
    detectors = []
    centers_x = (torch.arange(detectors_x) - detectors_x / 2 + 0.5) * crystal_w
    centers_y = (torch.arange(detectors_y) - detectors_y / 2 + 0.5) * crystal_h

    rect = torch.tensor(
        [[-crystal_w / 2, -crystal_h / 2],
         [crystal_w / 2, -crystal_h / 2],
         [crystal_w / 2, crystal_h / 2],
         [-crystal_w / 2, crystal_h / 2]]
    )

    # First layer starts below aperture_bottom
    first_layer_y = aperture_bottom - 20.0  # small gap (20 mm)
    for i in range(n_detector_layers):
        layer_y_offset = first_layer_y - i * layer_spacing_mm
        grid_x, grid_y = torch.meshgrid(centers_x, centers_y + layer_y_offset, indexing="xy")
        centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        layer_vertices = centers[:, None, :] + rect[None, :, :]
        detectors.append(layer_vertices)

    base_detector_units = torch.cat(detectors, dim=0)

    return base_detector_units, base_plate_segments

import torch

def generate_planar_multi_plate_geometry_v2(
    aperture_width_x=140.0,
    aperture_height_y=6.0,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    n_layers=4,
    layer_spacing_mm=40.0
):
    """
    Generates a simplified linear MPH-SPECT geometry:
      - A rectangular aperture plate with pinhole cutouts
      - Stacked rectangular detector plates behind it
    Returns (detector_units, plate_segments)
    """

    # --- Define aperture plate centered at (0, 0)
    half_w, half_h = aperture_width_x / 2, aperture_height_y / 2
    plate = torch.tensor([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ]).unsqueeze(0)  # shape (1, 4, 2)

    # --- Define detector layers stacked below aperture
    detector_layers = []
    det_width, det_height = aperture_width_x, 50.0
    half_dw, half_dh = det_width / 2, det_height / 2

    for i in range(n_layers):
        offset_y = -(i + 1) * layer_spacing_mm - 30.0  # stack downward
        rect = torch.tensor([
            [-half_dw, offset_y - half_dh],
            [ half_dw, offset_y - half_dh],
            [ half_dw, offset_y + half_dh],
            [-half_dw, offset_y + half_dh],
        ])
        detector_layers.append(rect)

    detectors = torch.stack(detector_layers)  # shape (n_layers, 4, 2)

    return detectors, plate

    base_detectors, base_plate = generate_linear_multi_plate_geometry(
    aperture_width_x=140.0,
    aperture_height_y=6.0,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    n_detector_layers=4,          # ✅ correct name
    detector_spacing_mm=40.0,     # ✅ correct name
    detector_offset_mm=30.0       # optional, default is 30
)

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def generate_linear_slim_geometry(
    aperture_width_x=140.0,
    aperture_height_y=6.0,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    n_layers=4,
    layer_spacing_mm=40.0,
):
    """
    Generate a simple planar MPH-SPECT geometry:
      - One aperture (collimator) plate centered at Y=0
      - Multiple rectangular detector plates stacked downward in Y
    Returns:
        detector_units: (N_layers, 4, 2)
        aperture_plate: (1, 4, 2)
    """

    # === Aperture plate (at origin) ===
    half_w, half_h = aperture_width_x / 2, aperture_height_y / 2
    aperture_plate = torch.tensor([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ]).unsqueeze(0)  # (1, 4, 2)

    # === Detector layers (rectangles below) ===
    det_half_w = aperture_width_x / 2
    det_half_h = 5.0  # thin height for clarity
    detector_units = []

    for i in range(n_layers):
        y_offset = -(i + 1) * layer_spacing_mm - 10.0  # start below aperture
        rect = torch.tensor([
            [-det_half_w, y_offset - det_half_h],
            [ det_half_w, y_offset - det_half_h],
            [ det_half_w, y_offset + det_half_h],
            [-det_half_w, y_offset + det_half_h],
        ])
        detector_units.append(rect)

    detector_units = torch.stack(detector_units)  # (N_layers, 4, 2)
    return detector_units, aperture_plate


def visualize_linear_spect_geometry(
    base_detector_units: torch.Tensor,
    base_plate_segments: torch.Tensor,
    aperture_width_mm: float = 140.0,
    aperture_height_mm: float = 6.0,
    aperture_to_first_detector_mm: float = 30.0,
    inter_layer_spacing_mm: float = 8.0,
    dense_layer_offset_mm: float = 30.0,
    fov_diameter_mm: float = 70.0,
    save_path: str = "mph_linear_layout_visualized.png"
):
    """
    Visualize planar MPH-SPECT geometry with 4 detector layers + aperture plate.

    - 3 self-collimating layers (blue shades)
    - 1 dense detector layer (green)
    - aperture plate + pinholes in front
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect("equal", adjustable="box")

    # --- Colors for detector layers ---
    detector_colors = ["#90caf9", "#64b5f6", "#1976d2", "#66bb6a"]  # 3 blues + 1 green
    n_layers = 4
    n_per_layer = base_detector_units.shape[0] // n_layers if n_layers > 0 else 0

    # --- Plot detector layers (from back to front) ---
    if base_detector_units.numel() > 0 and n_per_layer > 0:
        for layer_idx in reversed(range(n_layers)):  # draw farthest (dense) first
            start = layer_idx * n_per_layer
            end = (layer_idx + 1) * n_per_layer
            color = detector_colors[layer_idx % len(detector_colors)]
            plot_polygons_from_vertices_2d_mpl(
                base_detector_units[start:end],
                ax,
                facecolor=color,
                edgecolor="black",
                alpha=0.6,
                label=f"Detector Layer {layer_idx + 1}",
                zorder=layer_idx + 1
            )
    # --- Plot aperture plate (in front of detectors) ---
    aperture_rect_x = -aperture_width_mm / 2
    aperture_rect_y = -aperture_to_first_detector_mm - aperture_height_mm / 2
    aperture_rect = Rectangle(
        (aperture_rect_x, aperture_rect_y),
        aperture_width_mm,
        aperture_height_mm,
        edgecolor="black",
        facecolor="lightgray",
        lw=1.5,
        label="Aperture Plate",
        zorder=10  # in front
    )
    ax.add_patch(aperture_rect)

    # --- Plot pinholes (on top of aperture) ---
    if base_plate_segments.numel() > 0:
        plot_polygons_from_vertices_2d_mpl(
            base_plate_segments,
            ax,
            facecolor="none",
            edgecolor="red",
            alpha=0.9,
            label="Pinholes",
            zorder=11
        )

    # --- Plot circular FOV reference ---
    circ = Circle(
        (0, 10),
        radius=fov_diameter_mm / 2,
        edgecolor="darkred",
        facecolor="none",
        linestyle="--",
        lw=2,
        label=f"FOV (D={fov_diameter_mm} mm)",
        zorder=12
    )
    ax.add_patch(circ)

    # --- Axis guides ---
    ax.axhline(0, color="gray", linestyle=":", lw=1, zorder=0)
    ax.axvline(0, color="gray", linestyle=":", lw=1, zorder=0)

    # --- Auto-scaling limits ---
    max_extent = 0
    if base_detector_units.numel() > 0:
        max_extent = torch.abs(base_detector_units).max().item()
    if base_plate_segments.numel() > 0:
        max_extent = max(max_extent, torch.abs(base_plate_segments).max().item())

    plot_limit = max(max_extent, fov_diameter_mm / 2 + 40, aperture_to_first_detector_mm + 100)
    ax.set_xlim([-plot_limit, plot_limit])
    ax.set_ylim([-plot_limit, plot_limit * 0.2])

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Planar MPH-SPECT Geometry: Aperture + 4 Detector Layers")
    ax.legend(fontsize="small", loc="upper right", ncol=1)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Saved color-coded visualization to: {save_path}")

import matplotlib.pyplot as plt

def visualize_linear_geometry(
    detectors,
    aperture,
    fov_diameter_mm=16.0,
    save_path="mph_planar_slim.png",
):
    """
    Visualize planar MPH-SPECT layout (detectors + aperture + FOV).
    """
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("white")

    # Plot detectors (stacked blue plates)
    for i, det in enumerate(detectors):
        poly = plt.Polygon(
            det.numpy(),
            closed=True,
            facecolor='royalblue',
            edgecolor='black',
            alpha=0.6,
        )
        ax.add_patch(poly)
        ax.text(
            det[:, 0].mean(),
            det[:, 1].mean(),
            f"D{i+1}",
            color='white',
            ha='center',
            va='center',
            fontsize=8,
        )

    # Plot aperture (gray)
    ap_poly = plt.Polygon(
        aperture[0].numpy(),
        closed=True,
        facecolor='gray',
        edgecolor='black',
        alpha=0.9,
        label="Aperture",
    )
    ax.add_patch(ap_poly)

    # Add FOV circle
    circ = plt.Circle(
        (0, 0),
        radius=fov_diameter_mm / 2,
        edgecolor='darkred',
        facecolor='none',
        linestyle='--',
        lw=2,
        label=f"FOV ({fov_diameter_mm:.1f}mm)",
    )
    ax.add_patch(circ)

    # === Fix: Scale limits so all detectors are visible ===
    all_y = torch.cat([detectors[..., 1].flatten(), aperture[..., 1].flatten()])
    all_x = torch.cat([detectors[..., 0].flatten(), aperture[..., 0].flatten()])
    y_min, y_max = all_y.min().item(), all_y.max().item()
    x_min, x_max = all_x.min().item(), all_x.max().item()

    y_range = y_max - y_min
    x_range = x_max - x_min

    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.1 * y_range)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Planar MPH-SPECT Geometry (Slim)")

    ax.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved improved visualization → {save_path}")

import torch
import matplotlib.pyplot as plt

def visualize_linear_geometry_with_pinholes(
    detectors,
    aperture,
    pinhole_diameter_mm=1.6,
    aperture_open_ratio=0.125,
    fov_diameter_mm=16.0,
    save_path="mph_planar_slim_with_pinholes.png",
):
    """
    Visualize planar MPH-SPECT layout:
      - 4 detector plates
      - 1 aperture plate
      - pinholes drawn on aperture
      - FOV circle in front
    """

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_facecolor("white")

    # --- 1. Plot detectors (first, so they appear behind everything else)
    for i, det in enumerate(detectors):
        poly = plt.Polygon(
            det.numpy(),
            closed=True,
            facecolor='royalblue',
            edgecolor='black',
            alpha=0.6,
            zorder=1,
        )
        ax.add_patch(poly)
        ax.text(
            det[:, 0].mean(),
            det[:, 1].mean(),
            f"D{i+1}",
            color='white',
            ha='center',
            va='center',
            fontsize=8,
            zorder=5,
        )

    # --- 2. Plot aperture plate
    ap_poly = plt.Polygon(
        aperture[0].numpy(),
        closed=True,
        facecolor='gray',
        edgecolor='black',
        alpha=0.9,
        label="Aperture",
        zorder=2,
    )
    ax.add_patch(ap_poly)

    # --- 3. Compute and plot pinhole positions
    ap = aperture[0]
    width = ap[:, 0].max() - ap[:, 0].min()
    height = ap[:, 1].max() - ap[:, 1].min()

    # spacing between pinholes = open_ratio * width
    n_pinholes = int(width / (pinhole_diameter_mm / aperture_open_ratio))
    x_positions = torch.linspace(ap[:, 0].min() + pinhole_diameter_mm,
                                 ap[:, 0].max() - pinhole_diameter_mm,
                                 n_pinholes)
    y_center = ap[:, 1].mean()

    for x in x_positions:
        circ = plt.Circle(
            (x.item(), y_center),
            radius=pinhole_diameter_mm / 2,
            facecolor='white',
            edgecolor='red',
            linewidth=1.2,
            alpha=1.0,
            zorder=3,
        )
        ax.add_patch(circ)

    # --- 4. Add FOV circle (on top of everything else)
    fov = plt.Circle(
        (0, 0),
        radius=fov_diameter_mm / 2,
        edgecolor='darkred',
        facecolor='none',
        linestyle='--',
        lw=2,
        label=f'FOV ({fov_diameter_mm:.1f}mm)',
        zorder=4,  # higher zorder = on top
    )
    ax.add_patch(fov)

    # --- 5. Scale and label
    all_y = torch.cat([detectors[..., 1].flatten(), aperture[..., 1].flatten()])
    all_x = torch.cat([detectors[..., 0].flatten(), aperture[..., 0].flatten()])
    y_min, y_max = all_y.min().item(), all_y.max().item()
    x_min, x_max = all_x.min().item(), all_x.max().item()
    y_range = y_max - y_min
    x_range = x_max - x_min

    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Planar MPH-SPECT Geometry with Pinhole Aperture (Slim)")
    ax.legend(loc="upper right", fontsize=8)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Saved visualization with pinholes → {save_path}")
