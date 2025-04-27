# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from dissertation_nystrom import nystrom_approximation
from dissertation_kernels import gaussian_kernel_matrices
from dissertation_sampling import *

def generate_spiral(n_points=10000, t_max=64):
    """
    Generate a 2D spiral dataset.

    Parameters:
        n_points (int): Number of data points.
        t_max (float): Maximum parameter value controlling the spiral tightness.

    Returns:
        ndarray: Coordinates of the spiral shape of shape (n_points, 2).
    """
    t = np.sort(t_max * np.random.rand(n_points))
    x = np.exp(0.2 * t) * np.cos(t)
    y = np.exp(0.2 * t) * np.sin(t)
    spiral_coords = np.column_stack((x, y))
    return spiral_coords


if __name__ == "__main__":
    # Set parameters
    color = 'gray'
    lw = 2
    bandwidth = 1000.0  # Gaussian kernel bandwidth
    rank = 50  # Number of landmarks per method

    # Generate dataset
    spiral_coords = generate_spiral(n_points=10000, t_max=64)

    # Compute full kernel matrix and its trace
    K_full = gaussian_kernel_matrices(spiral_coords, spiral_coords, bandwidth)
    full_trace = np.trace(K_full)

    # Eigendecomposition for kernel K-means++
    eigvals, eigvecs = np.linalg.eigh(K_full)
    tol = 1e-10
    positive_idx = eigvals > tol
    eigvals_filtered = eigvals[positive_idx]
    eigvecs_filtered = eigvecs[:, positive_idx]
    L = eigvecs_filtered @ np.diag(np.sqrt(eigvals_filtered))

    # --- Landmark Visualisations ---

    # Uniform sampling
    uniform_samples = sample_uniformly(spiral_coords, rank)
    plt.figure(figsize=(6, 6))
    plt.scatter(spiral_coords[:, 0], spiral_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(spiral_coords[uniform_samples, 0], spiral_coords[uniform_samples, 1], color='blue', s=100)
    plt.axis('equal')
    plt.title("Uniform Sampling")
    plt.show()

    # RPCholesky
    rpcholesky_samples, F = sample_rpcholesky(spiral_coords, rank, bandwidth)
    plt.scatter(spiral_coords[:, 0], spiral_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(spiral_coords[rpcholesky_samples, 0], spiral_coords[rpcholesky_samples, 1], color='orange', s=100)
    plt.axis('equal')
    plt.title("RPCholesky Sampling")
    plt.show()

    # RRLS
    rrls_samples = sample_rrls(spiral_coords, rank, bandwidth)
    plt.scatter(spiral_coords[:, 0], spiral_coords[:, 1], color=color, alpha=0.5)
    plt.scatter(spiral_coords[rrls_samples, 0], spiral_coords[rrls_samples, 1], color='green', s=100)
    plt.axis('equal')
    plt.title("RRLS Sampling")
    plt.show()

    # Linear K-Means++
    lin_kmeans_indices, _ = sample_linear_kmeans(spiral_coords, rank)
    plt.scatter(spiral_coords[:, 0], spiral_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(spiral_coords[lin_kmeans_indices, 0], spiral_coords[lin_kmeans_indices, 1], color='red', s=100)
    plt.axis('equal')
    plt.title("Linear K-Means++ Sampling")
    plt.show()

    # Kernel K-Means++
    ker_kmeans_indices = sample_kernel_kmeans(L, rank)
    plt.scatter(spiral_coords[:, 0], spiral_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(spiral_coords[ker_kmeans_indices, 0], spiral_coords[ker_kmeans_indices, 1], color='purple', s=100)
    plt.axis('equal')
    plt.title("Kernel K-Means++ Sampling")
    plt.show()

    # --- Approximation Quality Evaluation ---
    
    # Ranks and trials for averaging errors
    ranks = np.arange(10, 101, 10)
    num_trials = 10
    
    # Containers for errors
    methods = ['Uniform', 'RRLS', 'RPCholesky', 'Linear K-Means++', 'Kernel K-Means++']
    errors = {method: [] for method in methods}
    
    for r in ranks:
        # Temporary trial-wise error storage
        err_uniform, err_rrls, err_rpchol, err_lkm, err_kkm = [], [], [], [], []
    
        for t in range(num_trials):
            # Uniform
            idx = sample_uniformly(spiral_coords, r, random_state=t)
            err_uniform.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)
    
            # RRLS
            idx = sample_rrls(spiral_coords, rank=r, bandwidth=bandwidth, random_state=t)
            err_rrls.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)
    
            # RPCholesky
            idx, F = sample_rpcholesky(spiral_coords, rank=r, bandwidth=bandwidth, random_state=t)
            err_rpchol.append(np.abs(full_trace - np.trace(F @ F.T)) / full_trace)
    
            # Linear K-Means++
            idx, _ = sample_linear_kmeans(spiral_coords, rank=r, random_state=t)
            err_lkm.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)
    
            # Kernel K-Means++
            idx = sample_kernel_kmeans(L, rank=r, random_state=t)
            err_kkm.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)
    
        # Store mean errors
        errors['Uniform'].append(np.mean(err_uniform))
        errors['RRLS'].append(np.mean(err_rrls))
        errors['RPCholesky'].append(np.mean(err_rpchol))
        errors['Linear K-Means++'].append(np.mean(err_lkm))
        errors['Kernel K-Means++'].append(np.mean(err_kkm))

    # --- Plot Error Results ---
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ranks, errors[method], marker='o', label=method)

    plt.xlabel('Rank (Number of Landmarks)')
    plt.ylabel('Relative Trace Error')
    plt.title('Relative Trace Error vs. Rank on Spiral Dataset')
    plt.legend()
    plt.show()