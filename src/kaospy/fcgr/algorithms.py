import numpy as np

"""
    Functions and calculations for Frequency Chaos Game Representation (FCGR).
"""


def _calc_index(
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        res: int,
        radius: float,
        ignore_grid_line_points: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map CGR coordinates to matrix indices for an FCGR matrix.

    Each CGR point is assigned to a cell in the FCGR matrix based on the
    given resolution and coordinate system radius.

    :param x: X-coordinates of CGR points.
    :type x: list[float] | np.ndarray
    :param y: Y-coordinates of CGR points.
    :type y: list[float] | np.ndarray
    :param res: Resolution of the FCGR matrix (number of rows and columns).
                Must be a positive integer.
    :type res: int
    :param radius: Radius defining the CGR coordinate range.
                   Must be positive.
    :type radius: float
    :param ignore_grid_line_points: If True, points located on FCGR grid lines are excluded.
    :type ignore_grid_line_points: bool
    :return: Tuple containing the row and column indices of the FCGR matrix.
    :rtype: tuple[np.ndarray, np.ndarray]
    :raises ValueError: If ``res`` is not a positive integer or if
                        ``radius`` is not positive.
    """

    if not isinstance(res, int) or res <= 0:
        raise ValueError("Resolution must be a positive integer.")

    if radius <= 0:
        raise ValueError("Radius must be positive.")

    x = np.asarray(x)
    y = np.asarray(y)

    cell = 2 * radius / res

    if ignore_grid_line_points:
        valid_points = ~((np.isclose((x + radius) % cell, 0)) | (np.isclose((y + radius) % cell, 0)))
        x = x[valid_points]
        y = y[valid_points]

    matrix_x = np.floor((radius - y) * res / (2 * radius)).astype(int)  # col
    matrix_y = np.floor((x + radius) * res / (2 * radius)).astype(int)  # row

    matrix_x = np.clip(matrix_x, 0, res - 1)
    matrix_y = np.clip(matrix_y, 0, res - 1)

    return matrix_x, matrix_y


def _compute_fcgr_matrix(
        x: list[float] | np.ndarray,
        y: list[float] | np.ndarray,
        res: int,
        radius: float,
        ignore_grid_line_points: bool = False
) -> np.ndarray:
    """
    Compute the Frequency Chaos Game Representation (FCGR) matrix from
    CGR coordinates.

    Each CGR point is assigned to a cell of the FCGR matrix. The resulting
    matrix contains the number of points falling into each cell.

    :param x: X-coordinates of CGR points.
    :type x: list[float] | np.ndarray
    :param y: Y-coordinates of CGR points.
    :type y: list[float] | np.ndarray
    :param res: Resolution of the FCGR matrix (number of rows and columns).
    :type res: int
    :param radius: Radius defining the CGR coordinate range.
    :type radius: float
    :param ignore_grid_line_points: If True, points located on FCGR grid lines are excluded.
    :type ignore_grid_line_points: bool
    :return: FCGR matrix of shape (res, res), where each cell contains the
             number of CGR points assigned to that region.
    :rtype: np.ndarray
    """

    matrix = np.zeros((res, res), dtype=int)

    matrix_x, matrix_y = _calc_index(x, y, res, radius, ignore_grid_line_points=ignore_grid_line_points)

    np.add.at(matrix, (matrix_x, matrix_y), 1)

    return matrix


def _euclidean_distance(
        fcgr_matrix1: np.ndarray,
        fcgr_matrix2: np.ndarray
) -> float:
    """
    Compute the Euclidean distance between two FCGR matrices.

    :param fcgr_matrix1: First FCGR matrix.
    :type fcgr_matrix1: np.ndarray
    :param fcgr_matrix2: Second FCGR matrix.
    :type fcgr_matrix2: np.ndarray
    :return: Euclidean distance between the two matrices.
    :rtype: float
    :raises ValueError: If the matrices have different shapes.
    """

    if fcgr_matrix1.shape != fcgr_matrix2.shape:
        raise ValueError("FCGR matrices must have the same shape.")

    return np.linalg.norm(fcgr_matrix1 - fcgr_matrix2)


def _manhattan_distance(
        fcgr_matrix1: np.ndarray,
        fcgr_matrix2: np.ndarray
) -> float:
    """
    Compute the Manhattan (cityblock) distance between two FCGR matrices.

    :param fcgr_matrix1: First FCGR matrix.
    :type fcgr_matrix1: np.ndarray
    :param fcgr_matrix2: Second FCGR matrix.
    :type fcgr_matrix2: np.ndarray
    :return: Manhattan distance between the two matrices.
    :rtype: float
    :raises ValueError: If the matrices have different shapes.
    """

    if fcgr_matrix1.shape != fcgr_matrix2.shape:
        raise ValueError("FCGR matrices must have the same shape.")

    return np.sum(np.abs(fcgr_matrix1 - fcgr_matrix2))


def _cosine_distance(
        fcgr_matrix1: np.ndarray,
        fcgr_matrix2: np.ndarray
) -> float:
    """
    Compute the cosine distance between two FCGR matrices.

    :param fcgr_matrix1: First FCGR matrix.
    :type fcgr_matrix1: np.ndarray
    :param fcgr_matrix2: Second FCGR matrix.
    :type fcgr_matrix2: np.ndarray
    :return: Cosine distance between the two matrices.
    :rtype: float
    :raises ValueError: If the matrices have different shapes or if one
                        of the matrices has zero norm.
    """

    if fcgr_matrix1.shape != fcgr_matrix2.shape:
        raise ValueError("FCGR matrices must have the same shape.")

    f1, f2 = fcgr_matrix1.flatten(), fcgr_matrix2.flatten()

    norm_product = np.linalg.norm(f1) * np.linalg.norm(f2)

    if norm_product == 0:
        raise ValueError("Cosine distance is undefined for zero vectors.")

    return 1 - np.dot(f1, f2) / norm_product


_DISTANCE_METHODS = {
    "euclidean": _euclidean_distance,
    "manhattan": _manhattan_distance,
    "cosine": _cosine_distance,
}


def compute_fcgr_distance(
        fcgr_matrix1: np.ndarray,
        fcgr_matrix2: np.ndarray,
        method: str = "euclidean"
) -> float:
    """
    Compute the distance between two FCGR matrices using the specified method.

    Available distance methods are:
    - "euclidean": Euclidean distance
    - "manhattan": Manhattan (cityblock) distance
    - "cosine": Cosine distance

    :param fcgr_matrix1: First FCGR matrix.
    :type fcgr_matrix1: np.ndarray
    :param fcgr_matrix2: Second FCGR matrix.
    :type fcgr_matrix2: np.ndarray
    :param method: Distance method to use.
                   Options are "euclidean", "manhattan", and "cosine"
    :type method: str
    :return: Computed distance value.
    :rtype: float
    :raises ValueError: If an unknown distance method is provided.
    """

    if method not in _DISTANCE_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Available methods: {list(_DISTANCE_METHODS.keys())}"
        )

    return _DISTANCE_METHODS[method](fcgr_matrix1, fcgr_matrix2)
