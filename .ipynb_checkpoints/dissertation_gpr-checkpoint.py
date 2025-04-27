# Import libraries
import numpy as np
from dissertation_kernels import (
    gaussian_kernel_vectors,
    gaussian_kernel_matrices,
    matern_kernel_matrices
)

def full_gpr_gaussian(X_train, y_train, X_test, noise, bandwidth):
    """
    Perform exact Gaussian Process Regression using the Gaussian (RBF) kernel.

    Parameters:
        X_train (ndarray): Training inputs of shape (n_train, n_features).
        y_train (ndarray): Training targets of shape (n_train,).
        X_test (ndarray): Test inputs of shape (n_test, n_features).
        noise (float): Standard deviation of the observation noise.
        bandwidth (float): Bandwidth parameter for the Gaussian kernel.

    Returns:
        mean (ndarray): Predictive mean of shape (n_test,).
        cov (ndarray): Predictive covariance matrix of shape (n_test, n_test).
    """
    K = gaussian_kernel_matrices(X_train, X_train, bandwidth) + noise**2 * np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    
    # Compute alpha = K_inv @ y using Cholesky
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    K_star = gaussian_kernel_matrices(X_test, X_train, bandwidth)
    mean = K_star @ alpha

    # Compute predictive covariance
    v = np.linalg.solve(L, K_star.T)
    K_star_star = gaussian_kernel_matrices(X_test, X_test, bandwidth)
    cov = K_star_star - v.T @ v

    return mean, cov


def full_gpr_matern(X_train, y_train, X_test, noise, bandwidth, nu):
    """
    Perform exact Gaussian Process Regression using the Matern kernel.

    Parameters:
        X_train (ndarray): Training inputs of shape (n_train, n_features).
        y_train (ndarray): Training targets of shape (n_train,).
        X_test (ndarray): Test inputs of shape (n_test, n_features).
        noise (float): Standard deviation of the observation noise.
        bandwidth (float): Bandwidth parameter for the Matern kernel.
        nu (float): Smoothness parameter of the Matern kernel.

    Returns:
        mean (ndarray): Predictive mean of shape (n_test,).
        cov (ndarray): Predictive covariance matrix of shape (n_test, n_test).
    """
    K = matern_kernel_matrices(X_train, X_train, bandwidth, nu) + noise**2 * np.eye(len(X_train))
    L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    K_star = matern_kernel_matrices(X_test, X_train, bandwidth, nu)
    mean = K_star @ alpha

    v = np.linalg.solve(L, K_star.T)
    K_star_star = matern_kernel_matrices(X_test, X_test, bandwidth, nu)
    cov = K_star_star - v.T @ v

    return mean, cov


def approximation_gpr(X_train, X_test, y_train, noise, bandwidth, indices, rank):
    """
    Perform approximate Gaussian Process Regression using the Nyström method
    with selected landmark indices and the Gaussian kernel.

    Parameters:
        X_train (ndarray): Training inputs of shape (n_train, n_features).
        X_test (ndarray): Test inputs of shape (n_test, n_features).
        y_train (ndarray): Training targets of shape (n_train,).
        noise (float): Standard deviation of the observation noise.
        bandwidth (float): Bandwidth parameter for the Gaussian kernel.
        indices (array-like): Indices of selected landmark points.
        rank (int): Number of landmark points.

    Returns:
        mean (ndarray): Predictive mean of shape (n_test,).
    """
    X_m = X_train[indices]  # Landmark points
    K_mn = gaussian_kernel_matrices(X_m, X_train, bandwidth)
    K_mm = K_mn[:, indices]  # More efficient than recomputing K_mm separately

    # Regularised Nyström system matrix
    A = noise**2 * K_mm + K_mn @ K_mn.T + 1e-6 * np.eye(rank)

    # Solve for weights
    v = K_mn @ y_train
    w = np.linalg.solve(A, v)

    # Compute mean prediction
    K_star_m = gaussian_kernel_matrices(X_test, X_m, bandwidth)
    mean = K_star_m @ w

    return mean