from typing import Union, List, Optional
import numpy as np
from matplotlib.axes import Axes

from kaospy.cgr.api import CGR
from kaospy.visualizations.plot import cgr_plot, fcgr_plot
from kaospy.fcgr.algorithms import _compute_fcgr_matrix, compute_fcgr_distance


class FCGR(CGR):
    """
    Frequency Chaos Game Representation (FCGR) extending CGR with a frequency-based matrix representation.

    FCGR inherits the CGR coordinate representation and additionally provides a
    frequency matrix derived from the CGR coordinates. It can also be initialized
    from an existing CGR object.
    """

    def __init__(
            self,
            sequence: Union[str, int, List[Union[str, int]]] = None,
            cgr_obj: Optional[CGR] = None,
            symbols: Optional[Union[str, List[str]]] = None,
            scaling_factor: Optional[float] = None,
            radius: float = 1.0,
            resolution: int = 100,
            ignore_grid_line_points: bool = False,
            use_square_for_four: bool = True
    ) -> None:
        """
        Initialize a Frequency Chaos Game Representation (FCGR).

        An FCGR can be created from a new input sequence or from an
        existing CGR object. If a CGR object is provided, its calculated
        CGR coordinates and parameters are reused.

        Parameters:
            sequence (str | int | list[str | int] | None):
                Input sequence used to generate the CGR.

            cgr_obj (CGR | None):
                Existing CGR object used as input.

            symbols (str | list[str] | None):
                Optional predefined or user-defined symbols.

            scaling_factor (float | None):
                Optional scaling factor controlling the
                movement towards vertices.

            radius (float):
                Radius used for distributing CGR vertices.

            resolution (int):
                Resolution of the FCGR matrix.

            ignore_grid_line_points (bool):
                If True, points located on FCGR grid lines are excluded.

            use_square_for_four (bool):
                If True, use square vertex coordinates
                for exactly four symbols. If False,
                distribute all vertices using the circular
                arrangement.
        """

        if isinstance(sequence, CGR):
            cgr_obj = sequence
            sequence = None

        if cgr_obj is not None:
            self._sequence: list[str] = cgr_obj.sequence
            self._symbols: list[str] = cgr_obj.symbols
            self._scaling_factor: float = cgr_obj.scaling_factor
            self._radius: float = cgr_obj.radius
            self._vertices_dict: dict[str, tuple[float, float]] = cgr_obj.vertices
            self._x_coords: np.ndarray = cgr_obj.x_coords.copy()
            self._y_coords: np.ndarray = cgr_obj.y_coords.copy()
            self._use_square_for_four: bool = cgr_obj.use_square_for_four

        else:
            super().__init__(
                sequence,
                symbols,
                scaling_factor,
                radius,
                use_square_for_four
            )

        if resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")

        self._resolution: int = resolution
        self._ignore_grid_line_points: bool = ignore_grid_line_points

        self._fcgr_matrix: np.ndarray = self._compute_matrix()

    @property
    def matrix(self) -> np.ndarray:
        """
        Return the FCGR matrix.

        Returns:
            numpy.ndarray:
                Read-only view of the FCGR matrix.
        """

        matrix = self._fcgr_matrix.view()
        matrix.flags.writeable = False

        return matrix

    @property
    def resolution(self) -> int:
        """
        Return the resolution of the FCGR matrix.

        Returns:
            int:
                Resolution of the FCGR matrix.
        """

        return self._resolution

    @property
    def ignore_grid_line_points(self) -> bool:
        """
        Return whether FCGR grid line points are ignored.

        Returns:
            bool:
                True if points located on FCGR grid lines are ignored,
                otherwise False.
        """

        return self._ignore_grid_line_points

    def _compute_matrix(self) -> np.ndarray:
        """
        Compute the FCGR matrix from the stored CGR coordinates.

        Grid line point handling is controlled by the
        ``ignore_grid_line_points`` setting.

        Returns:
            numpy.ndarray:
                Computed FCGR matrix.
        """

        return _compute_fcgr_matrix(
            self._x_coords,
            self._y_coords,
            self._resolution,
            self._radius,
            ignore_grid_line_points=self._ignore_grid_line_points
        )

    def set_resolution(
            self,
            resolution: int,
            ignore_grid_line_points: Optional[bool] = None
    ) -> None:
        """
        Update the FCGR resolution and recompute the FCGR matrix.

        Parameters:
            resolution (int):
                New resolution of the FCGR matrix.

            ignore_grid_line_points (bool | None):
                If provided, update whether grid line points are ignored.
                If None, keep the current setting.

        Returns:
            None
        """

        if resolution <= 0:
            raise ValueError("Resolution must be a positive integer.")

        if ignore_grid_line_points is not None:
            self._ignore_grid_line_points = ignore_grid_line_points

        self._resolution = resolution
        self._fcgr_matrix = self._compute_matrix()

    def plot(
            self,
            kind: str = "fcgr",
            vertices: bool = False,
            labels: bool = False,
            point_size: float = 1,
            vertex_size: float = 25,
            cmap: str = "gray_r",
            norm=None,
            interpolation: str = "nearest",
            ax: Axes | None = None
    ):
        """
        Plot the FCGR matrix or the underlying CGR representation.

        Parameters:
            kind (str):
                Representation to plot. Supported values are
                "fcgr" and "cgr".

            vertices (bool):
                If True, plot symbol vertices.

            labels (bool):
                If True, add symbol labels.

            point_size (float):
                Size of CGR points when plotting the CGR representation.

            vertex_size (float):
                Size of plotted symbol vertices.

            cmap (str):
                Colormap used for displaying the FCGR matrix.

            norm (matplotlib.colors.Normalize | None):
                Normalization applied to the FCGR values.

            interpolation (str):
                Interpolation method used for displaying the FCGR matrix.

            ax (matplotlib.axes.Axes | None):
                Existing Matplotlib axes to draw the representation on.
                If None, a new figure and axes are created.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                Tuple containing the Matplotlib figure and axes objects.
        """

        if kind == "cgr":
            return cgr_plot(
                self._x_coords,
                self._y_coords,
                self._vertices_dict,
                vertices=vertices,
                labels=labels,
                point_size=point_size,
                vertex_size=vertex_size,
                ax=ax
            )

        if kind == "fcgr":
            return fcgr_plot(
                self._fcgr_matrix,
                self._resolution,
                self._vertices_dict,
                vertices=vertices,
                labels=labels,
                vertex_size=vertex_size,
                cmap=cmap,
                norm=norm,
                interpolation=interpolation,
                ax=ax
            )

        raise ValueError("kind must be 'fcgr' or 'cgr'")

    def distance(self, other: "FCGR", method: str = "euclidean") -> float:
        """
        Compute the distance between this FCGR and another FCGR.

        Both FCGR objects must have the same resolution and the same
        grid line handling configuration.

        Parameters:
            other (FCGR):
                Another FCGR object to compare with.

            method (str):
                Distance metric to use. Supported methods include
                "euclidean", "manhattan", and "cosine".

        Returns:
            float:
                Computed distance value.
        """

        if not isinstance(other, FCGR):
            raise TypeError("`other` must be an instance of FCGR")

        if self._resolution != other._resolution:
            raise ValueError(
                "FCGR objects must have the same resolution for distance computation."
            )

        if self._ignore_grid_line_points != other._ignore_grid_line_points:
            raise ValueError(
                "FCGR objects must use the same grid line handling "
                "for distance computation."
            )

        return compute_fcgr_distance(
            self._fcgr_matrix,
            other._fcgr_matrix,
            method=method
        )

    def __str__(self) -> str:
        """
        Return a description of the FCGR object.

        Returns:
            str:
                String representation of the FCGR.
        """

        return (
            f"FCGR("
            f"sequence_length={len(self._sequence)}, "
            f"symbols={self._symbols}, "
            f"resolution={self._resolution}, "
            f"scaling_factor={self._scaling_factor}, "
            f"radius={self._radius}"
            f")"
        )

    __repr__ = __str__
