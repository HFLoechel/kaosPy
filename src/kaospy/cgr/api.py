from typing import Union, List, Optional

import numpy as np
from matplotlib.axes import Axes

# noinspection PyProtectedMember
from . import _algorithms
# noinspection PyProtectedMember
from ..sequences import _utils
from kaospy.visualizations.plot import cgr_plot


class CGR:
    """
    Chaos Game Representation (CGR) for a given sequence.

    The input sequence is normalized internally and transformed
    into CGR coordinates based on the defined symbols, scaling factor,
    and radius.
    """

    def __init__(
            self,
            sequence: Union[str, int, List[Union[str, int]]],
            symbols: Optional[Union[str, List[str]]] = None,
            scaling_factor: Optional[float] = None,
            radius: float = 1.0,
            use_square_for_four: bool = True
    ) -> None:
        """
        Initialize a Chaos Game Representation (CGR).

        The input sequence is normalized internally and mapped to
        symbol vertices. The CGR coordinates are computed based on
        the provided or automatically calculated scaling factor.

        Parameters:
            sequence (str | int | list[str | int]):
                Input sequence. Can be a string, integer,
                or list of strings/integers.

            symbols (str | list[str] | None):
                Optional predefined or user-defined symbols.
                If None, symbols are generated automatically
                from the sequence.

            scaling_factor (float | None):
                Optional scaling factor controlling the
                movement towards the corresponding vertex.
                If None, it is calculated automatically.

            radius (float):
                Radius used for distributing the vertices.

            use_square_for_four (bool):
                If True, use square vertex coordinates
                for exactly four symbols. If False,
                distribute all vertices using the circular
                arrangement.
        """

        self._sequence: List[str] = _utils.normalize_sequence(sequence)
        self._symbols: List[str] = _utils.build_symbols(
            self._sequence,
            symbols
        )

        num_symbols = len(self._symbols)

        self._scaling_factor: float = _algorithms.calc_scaling_factor(
            num_symbols,
            scaling_factor
        )

        self._radius: float = radius
        self._use_square_for_four: bool = use_square_for_four

        self._vertices_dict: dict[str, tuple[float, float]] = (
            _algorithms.build_vertices_dict(
                self._symbols,
                self._radius,
                self._use_square_for_four
            )
        )

        self._x_coords: np.ndarray
        self._y_coords: np.ndarray

        self._x_coords, self._y_coords = _algorithms.compute_cgr_points(
            self._sequence,
            self._scaling_factor,
            self._vertices_dict
        )

    @property
    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the Chaos Game Representation (CGR) coordinates.

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]:
                Tuple containing the x- and y-coordinates of the CGR points.
        """
        x_coords = self._x_coords.view()
        y_coords = self._y_coords.view()

        x_coords.flags.writeable = False
        y_coords.flags.writeable = False

        return x_coords, y_coords

    @property
    def x_coords(self) -> np.ndarray:
        """
        Return the x-coordinates of the Chaos Game Representation (CGR).

        Returns:
            numpy.ndarray:
                Read-only view of the x-coordinates of the CGR points.
        """

        x_coords = self._x_coords.view()
        x_coords.flags.writeable = False

        return x_coords

    @property
    def y_coords(self) -> np.ndarray:
        """
        Return the y-coordinates of the Chaos Game Representation (CGR).

        Returns:
            numpy.ndarray:
                Read-only view of the y-coordinates of the CGR points.
        """

        y_coords = self._y_coords.view()
        y_coords.flags.writeable = False

        return y_coords

    def coordinate_at(self, index: int) -> np.ndarray:
        """
        Return the CGR coordinate at the specified index.

        Parameters:
            index (int):
                Index of the CGR point.

        Returns:
            numpy.ndarray:
                Array containing the x- and y-coordinate of the CGR point.
        """

        return np.array(
            [
                self._x_coords[index],
                self._y_coords[index]
            ]
        )

    @property
    def sequence(self) -> list[str]:
        """
        Return the normalized sequence used for the CGR.

        Returns:
            list[str]:
                Copy of the normalized sequence as a list of symbols.
        """

        return self._sequence.copy()

    @property
    def symbols(self) -> list[str]:
        """
        Return the symbols used for the CGR.

        Returns:
            list[str]:
                Copy of the symbols used in the CGR.
        """

        return self._symbols.copy()

    @property
    def scaling_factor(self) -> float:
        """
        Return the scaling factor used for the CGR.

        Returns:
            float:
                Scaling factor used for generating CGR coordinates.
        """

        return self._scaling_factor

    @property
    def radius(self) -> float:
        """
        Return the radius used for distributing CGR vertices.

        Returns:
            float:
                Radius used for the CGR vertex distribution.
        """

        return self._radius

    @property
    def vertices(self) -> dict[str, tuple[float, float]]:
        """
        Return the vertices used for the CGR.

        Returns:
            dict[str, tuple[float, float]]:
                Copy of the dictionary mapping symbols to their vertex coordinates.
        """

        return self._vertices_dict.copy()

    @property
    def use_square_for_four(self) -> bool:
        """
        Return whether a square arrangement is used for four symbols.

        Returns:
            bool:
                True if a square arrangement is used for exactly four symbols,
                otherwise False.
        """

        return self._use_square_for_four

    def plot(
            self,
            vertices: bool = False,
            labels: bool = False,
            point_size: float = 1,
            vertex_size: float = 25,
            ax: Axes | None = None
    ):
        """
        Plot the CGR.

        Parameters:
            vertices (bool):
                If True, plot the symbol vertices used for the CGR.

            labels (bool):
                If True, add symbol labels to the plotted vertices.

            point_size (float):
                Size of the CGR points in the scatter plot.

            vertex_size (float):
                Size of the plotted vertices.

            ax (matplotlib.axes.Axes | None):
                Existing Matplotlib axes to draw the CGR on.
                If None, a new figure and axes are created.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
                Tuple containing the Matplotlib figure and axes objects.
        """

        return cgr_plot(
            self.x_coords,
            self.y_coords,
            self.vertices,
            vertices=vertices,
            labels=labels,
            point_size=point_size,
            vertex_size=vertex_size,
            ax=ax
        )

    def __str__(self) -> str:
        """
        Return a description of the CGR object.

        Returns:
            str:
                String representation of the CGR.
        """

        return (
            f"CGR("
            f"sequence_length={len(self._sequence)}, "
            f"symbols={self._symbols}, "
            f"scaling_factor={self._scaling_factor}, "
            f"radius={self._radius}"
            f")"
        )

    __repr__ = __str__

    def __len__(self) -> int:
        """
        Return the length of the sequence represented by the CGR.

        Returns:
            int:
                Number of symbols in the sequence.
        """

        return len(self._sequence)
