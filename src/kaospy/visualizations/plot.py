import matplotlib.pyplot as plt
import numpy as np

"""
    Functions for plotting.
"""

def fcgr_plot(
        fcgr_matrix,
        resolution,
        vertices_dict,
        vertices=False,
        labels=False,
        vertex_size=25,
        cmap="gray_r",
        norm=None,
        interpolation="nearest",
        ax=None
):
    """
    Create a plot of a Frequency Chaos Game Representation (FCGR).

    The function visualizes the FCGR matrix and optionally adds the
    corresponding symbol vertices and labels.

    Parameters:
        fcgr_matrix (numpy.ndarray):
            FCGR matrix to visualize.

        resolution (int):
            Resolution of the FCGR matrix.

        vertices_dict (dict[str, tuple[float, float]]):
            Mapping from symbols to CGR vertex coordinates.

        vertices (bool):
            If True, plot the symbol vertices used for the FCGR.

        labels (bool):
            If True, add symbol labels to the plotted vertices.

        vertex_size (float):
            Size of the plotted vertices.

        cmap (str):
            Colormap used for displaying the FCGR matrix.

        norm (matplotlib.colors.Normalize | None):
            Normalization applied to the FCGR values.

        interpolation (str):
            Interpolation method used for displaying the FCGR matrix.

        ax (matplotlib.axes.Axes | None):
            Existing Matplotlib axes to draw the FCGR on.
            If None, a new figure and axes are created.

    Returns:
        tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
            Matplotlib figure and axes objects.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(
        fcgr_matrix,
        cmap=cmap,
        norm=norm,
        interpolation=interpolation
    )

    transformed_vertices = {}

    if labels or vertices:
        transformed_vertices = {
            label: (
                (x * 1.35 + 1) / 2 * (resolution - 1),
                resolution - 1 -
                (y * 1.35 + 1) / 2 * (resolution - 1)
            )
            for label, (x, y) in vertices_dict.items()
        }

    if labels:
        for label, (gx, gy) in transformed_vertices.items():
            ax.text(
                gx,
                gy,
                label,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    if vertices:
        for label, (gx, gy) in transformed_vertices.items():
            ax.scatter(
                gx,
                gy,
                s=vertex_size
            )

    ax.axis("off")

    return fig, ax

def cgr_plot(
        x_coords,
        y_coords,
        vertices_dict,
        vertices=False,
        labels=False,
        point_size=1,
        vertex_size=25,
        ax=None
):
    """
    Create a plot of a Chaos Game Representation (CGR).

    The function visualizes the CGR coordinates and optionally adds the
    corresponding symbol vertices and labels. An existing Matplotlib
    axes object can be provided to draw the plot into an existing figure.

    Parameters:
        x_coords (numpy.ndarray):
            x-coordinates of the CGR points.

        y_coords (numpy.ndarray):
            y-coordinates of the CGR points.

        vertices_dict (dict[str, tuple[float, float]]):
            Mapping from symbols to their vertex coordinates.

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
            Matplotlib figure and axes objects.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.scatter(
        x_coords,
        y_coords,
        s=point_size,
        color="black"
    )

    if vertices:
        for label, (gx, gy) in vertices_dict.items():
            ax.scatter(
                gx,
                gy,
                color="red",
                s=vertex_size
            )

    if labels:
        for label, (gx, gy) in vertices_dict.items():
            ax.text(
                gx * 1.1,
                gy * 1.1,
                label,
                color="black",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold"
            )

    ax.axis("off")

    return fig, ax



def matrix_plot(
    matrix,
    cmap="gray_r",
    ax=None,
    vmin=None,
    vmax=None,
    symmetric=False,
    colorbar=False,
    colorbar_label=None,
    title=None
):
    if symmetric:
        max_abs = np.abs(matrix).max()
        vmin = -max_abs
        vmax = max_abs

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    im = ax.imshow(
        matrix,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    ax.axis("off")

    if title is not None:
        ax.set_title(
            title,
            pad=12
        )

    if colorbar:
        fig.colorbar(
            im,
            ax=ax,
            shrink=0.7,
            label=colorbar_label
        )

    return fig, ax, im