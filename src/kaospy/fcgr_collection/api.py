from typing import Union, List, Optional
import numpy as np

from kaospy import FCGR

from kaospy.fcgr_collection.algorithm import _build_fcgrs, _mean_fcgr, _variance_fcgr, _median_fcgr, _std_fcgr
from kaospy.matrix.api import Matrix


class FCGRCollection:
    """
    Collection of FCGR objects with shared representation parameters.

    An FCGRCollection stores multiple FCGR objects that are defined in the
    same FCGR space. All contained FCGR objects must have identical
    representation parameters.
    """

    def __init__(
            self,
            sequences: Optional[dict[str, str]] = None,
            symbols: Optional[Union[str, List[str]]] = None,
            scaling_factor: Optional[float] = None,
            radius: float = 1.0,
            resolution: int = 100,
            ignore_grid_line_points: bool = False,
            use_square_for_four: bool = True
    ) -> None:
        """
        Initialize an FCGRCollection.

        The collection can optionally be initialized from a dictionary of
        sequences. Each sequence is converted into an FCGR object and added
        to the collection.

        Parameters:
            sequences (dict[str, str] | None):
                Dictionary mapping sequence identifiers to biological sequences.

            symbols (str | list[str] | None):
                Optional predefined or user-defined symbols.

            scaling_factor (float | None):
                Optional scaling factor controlling movement towards CGR vertices.

            radius (float):
                Radius used for distributing CGR vertices.

            resolution (int):
                Resolution of the FCGR matrices.

            ignore_grid_line_points (bool):
                If True, points located on FCGR grid lines are ignored during
                FCGR computation.

            use_square_for_four (bool):
                If True, use square vertex coordinates for exactly four symbols.
                If False, distribute all vertices using a circular arrangement.
        """

        if resolution <= 0:
            raise ValueError(
                "Resolution must be a positive integer."
            )

        self._fcgrs: dict[str, FCGR] = {}

        # Store resolved FCGR parameters.
        # Symbols are resolved after the first FCGR is created.
        self._symbols: Optional[list[str]] = None
        self._scaling_factor: Optional[float] = scaling_factor
        self._radius: float = radius
        self._resolution: int = resolution
        self._ignore_grid_line_points: bool = ignore_grid_line_points
        self._use_square_for_four: bool = use_square_for_four

        if sequences is not None:

            fcgrs = _build_fcgrs(
                sequences,
                symbols=symbols,
                scaling_factor=scaling_factor,
                radius=radius,
                resolution=resolution,
                ignore_grid_line_points=ignore_grid_line_points,
                use_square_for_four=use_square_for_four
            )

            first_fcgr = next(iter(fcgrs.values()))

            self._symbols = first_fcgr.symbols

            if self._scaling_factor is None:
                self._scaling_factor = first_fcgr.scaling_factor

            for name, fcgr in fcgrs.items():
                self.append(name, fcgr)

    def append(
            self,
            name: str,
            fcgr: FCGR
    ) -> None:
        """
        Add an FCGR object to the collection.

        The FCGR object must have the same representation parameters as the
        collection.

        Args:
            name (str):
                Identifier of the FCGR object.

            fcgr (FCGR):
                FCGR object to add.

        Raises:
            TypeError:
                If fcgr is not an FCGR instance.

            ValueError:
                If the FCGR representation parameters do not match the collection.
        """

        if not isinstance(fcgr, FCGR):
            raise TypeError(
                "`fcgr` must be an instance of FCGR."
            )

        if fcgr.symbols != self.symbols:
            raise ValueError(
                "FCGR symbols do not match collection symbols."
            )

        if fcgr.scaling_factor != self.scaling_factor:
            raise ValueError(
                "FCGR scaling factor does not match collection."
            )

        if fcgr.radius != self.radius:
            raise ValueError(
                "FCGR radius does not match collection."
            )

        if fcgr.resolution != self.resolution:
            raise ValueError(
                "FCGR resolution does not match collection."
            )

        if fcgr.ignore_grid_line_points != self.ignore_grid_line_points:
            raise ValueError(
                "FCGR grid line handling does not match collection."
            )

        self._fcgrs[name] = fcgr

    @property
    def symbols(self) -> list[str] | None:

        """
        Return the symbols used for the FCGR representations in the collection.

        Returns:
            list[str]:
                Copy of the symbols defining the FCGR alphabet.
        """

        return None if self._symbols is None else self._symbols.copy()

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
    def use_square_for_four(self) -> bool:
        """
        Return whether a square arrangement is used for four symbols.

        Returns:
            bool:
                True if a square arrangement is used for exactly four symbols,
                otherwise False.
        """

        return self._use_square_for_four

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

    def number_of_fcgrs(self) -> int:
        """
        Return the number of FCGR objects stored in the collection.

        Returns:
            int:
                Number of FCGR representations stored in the collection.
        """

        return len(self._fcgrs)

    def __len__(self) -> int:
        """
        Return the number of FCGR objects stored in the collection.

        Returns:
            int:
                Number of FCGR representations.
        """

        return len(self._fcgrs)

    def fcgr_names(self) -> list[str]:
        """
        Return the names of all FCGR objects in the collection.

        Returns:
            list[str]:
                List of FCGR names stored in the collection.
        """

        return list(self._fcgrs.keys())

    def remove(
            self,
            name: str
    ) -> None:
        """
        Remove an FCGR object from the collection.

        Parameters:
            name (str):
                Identifier of the FCGR object to remove.

        Raises:
            KeyError:
                If no FCGR object with the given identifier exists.
        """

        if name not in self._fcgrs:
            raise KeyError(
                f"No FCGR object with identifier '{name}' found in collection."
            )

        del self._fcgrs[name]

    def get_fcgr(
            self,
            name: str
    ) -> FCGR:
        """
        Return an FCGR object stored in the collection.

        Parameters:
            name (str):
                Identifier of the FCGR object.

        Returns:
            FCGR:
                Requested FCGR object.

        Raises:
            KeyError:
                If no FCGR object with the given identifier exists.
        """

        if name not in self._fcgrs:
            raise KeyError(
                f"No FCGR object with identifier '{name}' found in collection."
            )

        return self._fcgrs[name]

    @property
    def fcgrs(self) -> dict[str, FCGR]:
        """
        Return all FCGR objects stored in the collection.

        Returns:
            dict[str, FCGR]:
                Copy of the dictionary mapping identifiers to FCGR objects.
        """

        return self._fcgrs.copy()

    def get_matrices(self) -> list[np.ndarray]:
        """
        Return all FCGR matrices stored in the collection.

        The returned matrices are read-only views and cannot modify the
        FCGR objects stored in the collection.

        Returns:
            list[numpy.ndarray]:
                List of read-only views of FCGR matrices.
        """

        return [
            fcgr.matrix
            for fcgr in self._fcgrs.values()
        ]

    def __str__(self) -> str:
        """
        Return a human-readable description of the FCGR collection.

        Returns:
            str:
                String representation of the collection.
        """

        return (
            f"FCGRCollection("
            f"number_of_fcgrs={len(self._fcgrs)}, "
            f"symbols={self.symbols}, "
            f"resolution={self.resolution}, "
            f"scaling_factor={self.scaling_factor}, "
            f"radius={self.radius}"
            f")"
        )

    def mean(self) -> Matrix:
        """
        Compute the pixel-wise mean FCGR matrix of the collection.

        The resulting matrix represents the average frequency value
        at each FCGR position across all FCGR objects in the collection.

        Returns:
            Matrix:
                Matrix containing the pixel-wise mean FCGR values.
        """

        return Matrix(_mean_fcgr(
            self.get_matrices())
        )

    def variance(self) -> Matrix:
        """
        Compute the pixel-wise variance of the FCGR matrices in the collection.

        Each matrix position is treated independently across all stored FCGR
        representations.

        Returns:
            Matrix:
                Matrix containing the pixel-wise variance of the FCGR matrices.
        """

        return Matrix(
            _variance_fcgr(
                self.get_matrices()
            )
        )

    def median(self) -> Matrix:
        """
        Compute the pixel-wise median of the FCGR matrices in the collection.

        Each matrix position is treated independently across all stored FCGR
        representations.

        Returns:
            Matrix:
                Matrix containing the pixel-wise median of the FCGR matrices.
        """

        return Matrix(
            _median_fcgr(
                self.get_matrices()
            )
        )

    def std(self) -> Matrix:
        """
        Compute the pixel-wise standard deviation of the FCGR matrices
        in the collection.

        Each matrix position is treated independently across all stored FCGR
        representations.

        Returns:
            Matrix:
                Matrix containing the pixel-wise standard deviation
                of the FCGR matrices.
        """

        return Matrix(
            _std_fcgr(
                self.get_matrices()
            )
        )

    def feature_matrix(self) -> np.ndarray:
        """
        Convert all FCGR matrices into a two-dimensional feature matrix.

        Each FCGR matrix is flattened into a one-dimensional feature vector.
        The resulting array contains one feature vector per FCGR object and is
        suitable as input for machine learning algorithms or numerical analyses.

        Returns:
            numpy.ndarray:
                Array of shape (n_fcgrs, n_features), where each row represents
                one FCGR representation.
        """

        return np.array([
            matrix.flatten()
            for matrix in self.get_matrices()
        ])

    def stacked_matrix(self) -> np.ndarray:
        """
        Return all FCGR matrices as a three-dimensional NumPy array.

        The resulting array contains one FCGR matrix per collection element
        and can be used for image-based analyses or tensor operations.

        Returns:
            numpy.ndarray:
                Array with shape (n_fcgrs, resolution, resolution).
        """

        return np.array(self.get_matrices())
