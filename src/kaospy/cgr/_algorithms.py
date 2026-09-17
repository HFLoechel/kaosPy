from numpy import zeros, ndarray
from math import sin, pi, floor, cos
from typing import Dict, Optional

"""
Functions for calculating Chaos Game Representations (CGR).
"""


def calc_scaling_factor(num_symbols: int, scaling_factor: Optional[float] = None) -> float:
    """
    Calculate or validate the scaling factor for a given number of symbols.

    :param num_symbols: Number of symbols.
    :type num_symbols: int
    :param scaling_factor: Optional user-defined scaling factor. If None,
                           it is calculated automatically.
    :type scaling_factor: float | None
    :return: The validated or computed scaling factor.
    :rtype: float
    :raises ValueError: If ``num_symbols`` is not a positive integer or if
                        ``scaling_factor`` is outside the range [0, 1].
    """
    if not isinstance(num_symbols, int) or num_symbols <= 0:
        raise ValueError("num_symbols must be a positive integer")

    if scaling_factor is None:
        scaling_factor = 1 - (sin(pi * (1 / num_symbols)) / (sin(pi * (1 / num_symbols)) + sin(pi * (1 / num_symbols + 2 * (floor(num_symbols / 4) / num_symbols)))))

    else:
        if not (0 <= scaling_factor <= 1):
            raise ValueError(f"Scaling factor must be between 0 and 1, got {scaling_factor}")

    return scaling_factor


def build_vertices_dict(
    symbols: list[str],
    radius: float,
    use_square_for_four: bool = True
) -> Dict[str, tuple[float, float]]:
    """
    Create a dictionary mapping each symbol to a vertex coordinate.

    By default, sets with exactly four symbols are mapped to the
    vertices of a square. For all other symbol counts, vertices are
    distributed evenly using a circular arrangement.

    :param symbols: List of symbols used for the CGR representation.
    :type symbols: list[str]
    :param radius: Radius for vertex distribution.
    :type radius: float
    :param use_square_for_four: If True, use square vertex coordinates
                                for exactly four symbols. If False,
                                distribute all vertices using the circular
                                arrangement.
    :type use_square_for_four: bool
    :return: Dictionary mapping symbols to vertex coordinates.
    :rtype: dict[str, tuple[float, float]]
    """
    num_symbols = len(symbols)

    if num_symbols == 0:
        return {}

    if num_symbols == 4 and use_square_for_four:
        square_vertices = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        vertices = [(x * radius, y * radius) for x, y in square_vertices]
    else:
        vertices = distr_pts(num_symbols, radius)

    vertices_dict = {symbols[i]: vertices[i] for i in range(num_symbols)}
    return vertices_dict


def compute_cgr_points(
    sequence: list[str],
    scaling_factor: float,
    vertices_dict: dict[str, tuple[float, float]]
) -> tuple[ndarray, ndarray]:
    """
    Compute the Chaos Game Representation (CGR) point coordinates for a given sequence.

    Each symbol in the sequence is mapped to a vertex coordinate from
    `vertices_dict`. Starting at (0, 0), each new point is computed by
    iteratively moving a proportion (`scaling_factor`) of the distance
    from the current point towards the corresponding vertex.

    :param sequence: Normalized sequence of symbols for which the CGR is computed.
                     Each symbol must exist as a key in `vertices_dict`.
    :type sequence: list[str]
    :param scaling_factor: Value between 0 and 1 determining how strongly each point
                           moves toward the corresponding vertex on each iteration.
    :type scaling_factor: float
    :param vertices_dict: Dictionary mapping each symbol to its vertex coordinate.
    :type vertices_dict: dict[str, tuple[float, float]]
    :return: Two NumPy arrays containing the x- and y-coordinates of the CGR points.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """

    x = zeros(len(sequence))
    y = zeros(len(sequence))
    last_x = 0
    last_y = 0

    for i, symbol in enumerate(sequence):
        vertex_x, vertex_y = vertices_dict[symbol]
        last_x += (vertex_x - last_x) * scaling_factor
        last_y += (vertex_y - last_y) * scaling_factor
        x[i] = last_x
        y[i] = last_y

    return x, y


def distr_pts(num_symbols: int, r: float) -> list[tuple[float, float]]:
    """
    Compute evenly distributed vertices on a circle with radius `r`.

    The vertices are placed at angles defined by:

        angle_i = π * (2(i + 1) + 1) / num_symbols

    This produces coordinates suitable for placing CGR vertices
    symmetrically on a circle for arbitrary numbers of symbols.

    :param num_symbols: Number of vertices to distribute.
    :type num_symbols: int
    :param r: Radius of the circle on which the vertices are placed.
    :type r: float
    :return: List of (x, y) coordinate tuples for the distributed vertices.
    :rtype: list[tuple[float, float]]
    :raises ValueError: If ``num_symbols`` is not a positive integer.
    """

    if num_symbols <= 0:
        raise ValueError("n must be a positive integer")

    angles = [pi * ((2 * (i + 1) + 1) / num_symbols) for i in range(num_symbols)]
    return [(r * sin(a), r * cos(a)) for a in angles]
