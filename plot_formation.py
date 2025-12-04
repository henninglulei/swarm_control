"""plot_formation.py

Python equivalent of the MATLAB `plot_formation.m` in this folder.

Usage:
    - As a module: import `plot_formation` and call `plot_formation(z)` where
      `z` is an (N x 2) array of agent positions.
    - As a script: it will try to load `data.mat` from the same folder and
      use the first 2D array it finds with second-dimension >= 2.

Requires: numpy, scipy, matplotlib
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


# topology graph (copied from the MATLAB file)
B = np.array([
    [1, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1],
    [-1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
    [0, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0],
    [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 1, -1, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 0],
], dtype=float)


def plot_formation(z: np.ndarray,
                   axis_limits: tuple[float, float, float, float] = (-2, 2, -2, 2),
                   show: bool = True,
                   savepath: Optional[str] = None) -> None:
    """Plot a formation defined by node positions `z`.

    Parameters
    - z: (N x D) array-like. Only first two columns are used (x, y).
    - axis_limits: (xmin, xmax, ymin, ymax) for the plot axes.
    - show: whether to call `plt.show()`.
    - savepath: optional path to save the figure (PNG, PDF, ...).
    """
    z = np.asarray(z)
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("z must be a 2D array with at least two columns (x,y)")

    N = B.shape[0]
    M = B.shape[1]

    # allow z to have a different number of rows: only plot edges where both
    # endpoints exist in the provided z array
    z_rows = z.shape[0]

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')

    # draw edges
    for i in range(M):
        nodes = np.where(B[:, i] != 0)[0]
        if nodes.size < 2:
            continue
        u, v = int(nodes[0]), int(nodes[1])
        # check that both endpoints exist in z
        if u < z_rows and v < z_rows:
            xcoords = [z[u, 0], z[v, 0]]
            ycoords = [z[u, 1], z[v, 1]]
            ax.plot(xcoords, ycoords, color='k', linewidth=1.5)

    # draw nodes
    ax.plot(z[:, 0], z[:, 1], 'r.', markersize=12)

    #xmin, xmax, ymin, ymax = axis_limits
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight')

    if show:
        plt.grid(True)
        plt.show()


def _find_z_in_mat(d: dict) -> Optional[np.ndarray]:
    """Return the first suitable array from a loaded mat-file dict.

    Criteria: numpy.ndarray, ndim==2 and shape[1] >= 2
    """
    for k, v in d.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 2:
            return v
    return None


def _main_try_load_and_plot():
    # try to load data.mat located next to this file
    folder = os.path.dirname(__file__)
    mat_path = os.path.join(folder, 'data.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Cannot find 'data.mat' in {folder}")

    data = loadmat(mat_path)
    z = _find_z_in_mat(data)
    if z is None:
        raise ValueError("No suitable 2D array found in data.mat (need shape (N,>=2))")

    # If MATLAB saved column vectors as Nx1 or 1xN, loadmat might transpose
    # but we assume the file contains an (N x 2) or (2 x N) style array.
    if z.shape[0] < z.shape[1] and z.shape[1] == 2:
        # probably transposed; transpose back
        z = z.T

    plot_formation(z)


if __name__ == '__main__':
    try:
        _main_try_load_and_plot()
    except Exception as e:
        print('Error while trying to auto-load and plot data.mat:', e)
        print("You can also call `plot_formation(z)` directly with your numpy array.")
