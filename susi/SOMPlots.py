"""SOMPlots functions.

Copyright (c) 2019-2020, Felix M. Riese.
All rights reserved.

"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_estimation_map(estimation_map, cbar_label="Variable in unit",
                        cmap="viridis", fontsize=20):
    """Plot estimation map.

    Parameters
    ----------
    estimation_map : np.array
        Estimation map of the size (n_rows, n_columns)
    cbar_label : str, optional
        Label of the colorbar, by default "Variable in unit"
    cmap : str, optional (default="viridis")
        Colormap
    fontsize : int, optional (default=20)
        Fontsize of the labels

    Returns
    -------
    ax : pyplot.axis
        Plot axis

    """
    _, ax = plt.subplots(1, 1, figsize=(7, 5))
    img = ax.imshow(estimation_map, cmap=cmap)
    ax.set_xlabel("SOM columns", fontsize=fontsize)
    ax.set_ylabel("SOM rows", fontsize=fontsize)
    # ax.set_xticklabels(fontsize=fontsize)
    # ax.set_yticklabels(fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    # colorbar
    cbar = plt.colorbar(img, ax=ax)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.set_ylabel(cbar_label, fontsize=fontsize, labelpad=10)
    for label in cbar.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    plt.grid(b=False)

    return ax


def plot_som_histogram(bmu_list, n_rows, n_columns, n_datapoints_cbar=5,
                       fontsize=22):
    """Plot 2D Histogram of SOM.

    Plot 2D Histogram with one bin for each SOM node. The content of one
    bin is the number of datapoints matched to the specific node.

    Parameters
    ----------
    bmu_list  : list of (int, int) tuples
        Position of best matching units (row, column) for each datapoint
    n_rows : int, optional (default=10)
        Number of rows for the SOM grid
    n_columns : int, optional (default=10)
        Number of columns for the SOM grid
    n_datapoints_cbar : int, optional (default=5)
        Maximum number of datapoints shown on the colorbar
    fontsize : int, optional (default=22)
        Fontsize of the labels

    Returns
    -------
    ax : pyplot.axis
        Plot axis

    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # colormap
    cmap = plt.cm.viridis
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'mcm', cmaplist, cmap.N)

    bounds = np.arange(0., n_datapoints_cbar+1, 1.)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax2 = fig.add_axes([0.96, 0.12, 0.03, 0.76])
    cbar = matplotlib.colorbar.ColorbarBase(
        ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds,
        boundaries=bounds, format='%1i', extend="max")
    cbar.ax.set_ylabel('Number of datapoints', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.hist2d([x[0] for x in bmu_list], [x[1] for x in bmu_list],
              bins=[n_rows, n_columns], cmin=1, cmap=cmap, norm=norm)

    for label in cbar.ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    ax.set_xlabel("SOM columns", fontsize=fontsize)
    ax.set_ylabel("SOM rows", fontsize=fontsize)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)

    # to be compatible with plt.imshow:
    ax.invert_yaxis()

    plt.grid(b=False)

    return ax


def plot_umatrix(u_matrix, n_rows, n_colums, cmap="Greys", fontsize=18):
    """Plot u-matrix.

    Parameters
    ----------
    u_matrix : np.array
        U-matrix containing the distances between all nodes of the
        unsupervised SOM. Shape = (n_rows*2-1, n_columns*2-1)
    n_rows : int, optional (default=10)
        Number of rows for the SOM grid
    n_columns : int, optional (default=10)
        Number of columns for the SOM grid
    cmap : str, optional (default="Greys)
        Colormap
    fontsize : int, optional (default=18)
        Fontsize of the labels

    Returns
    -------
    ax : pyplot.axis
        Plot axis

    """
    _, ax = plt.subplots(figsize=(6, 6))
    img = ax.imshow(u_matrix.squeeze(), cmap=cmap)
    ax.set_xticks(np.arange(0, n_colums*2+1, 20))
    ax.set_xticklabels(np.arange(0, n_colums+1, 10))
    ax.set_yticks(np.arange(0, n_rows*2+1, 20))
    ax.set_yticklabels(np.arange(0, n_rows+1, 10))

    # ticks and labels
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    ax.set_ylabel("SOM rows", fontsize=fontsize)
    ax.set_xlabel("SOM columns", fontsize=fontsize)

    # colorbar
    cbar = plt.colorbar(img, ax=ax, fraction=0.04, pad=0.04)
    cbar.ax.set_ylabel('Distance measure (a.u.)', rotation=90,
                       fontsize=fontsize, labelpad=20)
    cbar.ax.tick_params(labelsize=fontsize)

    return ax


def plot_nbh_dist_weight_matrix(som, it_frac=0.1):
    """Plot neighborhood distance weight matrix in 3D.

    Parameters
    ----------
    som : susi.SOMClustering or related
        Trained (un)supervised SOM
    it_frac : float, optional (default=0.1)
        Fraction of `som.n_iter_unsupervised` for the plot state.

    Returns
    -------
    ax : pyplot.axis
        Plot axis

    """
    nbh_func = som.calc_neighborhood_func(
        curr_it=som.n_iter_unsupervised*it_frac,
        mode=som.neighborhood_mode_unsupervised)
    dist_weight_matrix = som.get_nbh_distance_weight_matrix(
        neighborhood_func=nbh_func, bmu_pos=[som.n_rows//2, som.n_columns//2])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(som.n_rows)
    y = np.arange(som.n_columns)
    X, Y = np.meshgrid(x, y)
    Z = dist_weight_matrix.reshape(som.n_rows, som.n_columns)

    surf = ax.plot_surface(
        X, Y, Z, cmap=matplotlib.cm.coolwarm, antialiased=False,
        rstride=1, cstride=1, linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    return ax
