"""
    Functions and calculations for collections of Frequency Chaos Game Representation (FCGR).
"""

from kaospy import FCGR
import numpy as np


def _build_fcgrs(
        sequences: dict[str, str],
        **kwargs
) -> dict[str, FCGR]:
    """
    Build FCGR objects from a collection of input sequences.

    Each sequence is converted into an FCGR object while preserving the
    provided sequence identifiers.

    :param sequences: Dictionary mapping sequence identifiers to input
                      sequences.
    :type sequences: dict[str, str]
    :param kwargs: Additional arguments passed to the FCGR constructor.
    :type kwargs: dict

    :return: Dictionary mapping sequence identifiers to FCGR objects.
    :rtype: dict[str, FCGR]
    """

    fcgrs = {}

    for header, seq in sequences.items():
        fcgrs[header] = FCGR(
            sequence=seq,
            **kwargs
        )

    return fcgrs


def _mean_fcgr(
        matrices: list[np.ndarray]
) -> np.ndarray:
    """
    Compute the mean FCGR matrix from a collection of FCGR matrices.

    :param matrices: List of FCGR matrices.
    :type matrices: list[numpy.ndarray]

    :return: Mean FCGR matrix.
    :rtype: numpy.ndarray
    """

    return np.mean(
        matrices,
        axis=0
    )


def _variance_fcgr(
        matrices: list[np.ndarray]
) -> np.ndarray:
    """
    Compute the pixel-wise variance of a collection of FCGR matrices.

    The variance is calculated independently for each matrix position
    across all FCGR representations.

    :param matrices: List of FCGR matrices.
    :type matrices: list[numpy.ndarray]

    :return: Variance FCGR matrix.
    :rtype: numpy.ndarray
    """

    return np.var(
        matrices,
        axis=0
    )


def _median_fcgr(
        matrices: list[np.ndarray]
) -> np.ndarray:
    """
    Compute the pixel-wise median of a collection of FCGR matrices.

    :param matrices: List of FCGR matrices.
    :type matrices: list[numpy.ndarray]

    :return: Median FCGR matrix.
    :rtype: numpy.ndarray
    """

    return np.median(
        matrices,
        axis=0
    )


def _std_fcgr(
        matrices: list[np.ndarray]
) -> np.ndarray:
    """
    Compute the pixel-wise standard deviation of a collection of FCGR matrices.

    The standard deviation is calculated independently for each matrix
    position across all FCGR representations.

    :param matrices: List of FCGR matrices.
    :type matrices: list[numpy.ndarray]

    :return: Standard deviation FCGR matrix.
    :rtype: numpy.ndarray
    """

    return np.std(
        matrices,
        axis=0
    )
