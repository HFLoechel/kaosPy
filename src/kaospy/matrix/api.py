import numpy as np

from kaospy.visualizations.plot import matrix_plot


class Matrix:
    """
    Container class for numerical matrices.

    The Matrix class provides a lightweight wrapper around NumPy arrays and
    offers visualization functionality through the PyKaos plotting interface.
    It is used to represent processed FCGR-derived matrices and statistical
    summaries.

    Parameters:
        array (array-like):
            Input matrix data converted internally to a NumPy array.
    """

    def __init__(
            self,
            array
    ) -> None:
        """
        Initialize a Matrix object.

        Parameters:
            array (array-like):
                Matrix values.
        """

        self._values = np.asarray(array)

    @property
    def values(self) -> np.ndarray:
        """
        Return the underlying NumPy array.

        Returns:
            numpy.ndarray:
                Matrix values.
        """

        return self._values

    def plot(
            self,
            **kwargs
    ):
        """
        Plot the matrix using the PyKaos visualization backend.

        Parameters:
            **kwargs:
                Additional keyword arguments passed to the plotting function.

       Returns:
             tuple:
                Matplotlib figure, axes, and image object.
        """

        return matrix_plot(
            self._values,
            **kwargs
        )


    def __str__(self) -> str:
        """
        Return a formatted string representation of the matrix.

        Returns:
            str:
                Human-readable matrix representation.
        """

        return (
            "Matrix(\n"
            + np.array2string(
                self._values,
                formatter={
                    "float_kind": lambda x: f"{x:.2f}"
                }
            )
            + "\n)"
        )

    def __repr__(self) -> str:
        """
        Return the string representation of the Matrix object.

        Returns:
            str:
                Formatted matrix representation.
        """

        return self.__str__()

    def __sub__(self, other):
        """
        Subtract another Matrix object element-wise.

        Parameters:
            other (Matrix):
                Matrix to subtract from this matrix.

        Returns:
            Matrix:
                New Matrix containing the element-wise difference.

        Raises:
            TypeError:
                If other is not a Matrix instance.
        """

        if not isinstance(other, Matrix):
            raise TypeError(
                "Can only subtract Matrix objects."
            )

        return Matrix(
            self.values - other.values
        )