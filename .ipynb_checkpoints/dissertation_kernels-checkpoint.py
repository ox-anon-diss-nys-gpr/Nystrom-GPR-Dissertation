# Import libraries
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv, gamma

def gaussian_kernel_vectors(x, y, bandwidth):
    """
    Compute the Gaussian (RBF) kernel between two vectors.

    Parameters:
        x (ndarray): First vector.
        y (ndarray): Second vector.
        bandwidth (float): Bandwidth (length scale) parameter.

    Returns:
        float: Kernel value.
    """
    squared_dist = np.sum((x - y) ** 2)
    return np.exp(-squared_dist / (2 * bandwidth ** 2))


def gaussian_kernel_matrices(X, Y, bandwidth):
    """
    Compute the Gaussian (RBF) kernel matrix between two datasets.

    Parameters:
        X (ndarray): Matrix of shape (n_samples_X, n_features).
        Y (ndarray): Matrix of shape (n_samples_Y, n_features).
        bandwidth (float): Bandwidth (length scale) parameter.

    Returns:
        ndarray: Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    sq_dists = cdist(X, Y, 'sqeuclidean')  # Efficient pairwise squared distances
    return np.exp(-sq_dists / (2 * bandwidth ** 2))


def matern_kernel_matrices(X, Y, bandwidth, nu):
    """
    Compute the Matern kernel matrix between two datasets.

    Parameters:
        X (ndarray): Matrix of shape (n_samples_X, n_features).
        Y (ndarray): Matrix of shape (n_samples_Y, n_features).
        bandwidth (float): Bandwidth (length scale) parameter.
        nu (float): Smoothness parameter of the Matern kernel.

    Returns:
        ndarray: Kernel matrix of shape (n_samples_X, n_samples_Y).
    """
    dists = cdist(X, Y, 'euclidean') 

    if nu == 0.5:
        return np.exp(-dists / bandwidth)
    
    elif nu == 1.5:
        sqrt3 = np.sqrt(3)
        factor = sqrt3 * dists / bandwidth
        return (1.0 + factor) * np.exp(-factor)
    
    elif nu == 2.5:
        sqrt5 = np.sqrt(5)
        factor = sqrt5 * dists / bandwidth
        return (1.0 + factor + factor**2 / 3.0) * np.exp(-factor)
    
    else:
        dists = np.where(dists == 0, 1e-10, dists)
        factor = np.sqrt(2 * nu) * dists / bandwidth
        coeff = (2 ** (1 - nu)) / gamma(nu)
        return coeff * (factor ** nu) * kv(nu, factor)