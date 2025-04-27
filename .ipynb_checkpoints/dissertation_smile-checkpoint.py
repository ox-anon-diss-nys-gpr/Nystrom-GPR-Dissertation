# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from dissertation_nystrom import nystrom_approximation
from dissertation_kernels import gaussian_kernel_matrices
from dissertation_sampling import *

def circle_interior_points(center, radius, num_points):
    """
    Sample uniformly from the interior of a circle using polar coordinates.

    Parameters:
        center (tuple): (x, y) coordinates of the circle center.
        radius (float): Radius of the circle.
        num_points (int): Number of points to sample.

    Returns:
        ndarray: Sampled points of shape (num_points, 2).
    """
    r = radius * np.sqrt(np.random.rand(num_points))
    theta = 2 * np.pi * np.random.rand(num_points)
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.column_stack((x, y))


if __name__ == "__main__":
    # === Smiley Face Construction ===
    color = 'gray'
    lw = 2
    face_radius = 10
    eye_radius = 1
    mouth_radius = 6
    bandwidth = 3.0  # Kernel bandwidth

    # Face outline
    theta = np.linspace(0, 2 * np.pi, 9000)
    face_x = face_radius * np.cos(theta)
    face_y = face_radius * np.sin(theta)
    face_coords = np.column_stack((face_x, face_y))

    # Eyes (interior samples)
    left_eye = circle_interior_points(center=(-3, 3), radius=eye_radius, num_points=50)
    right_eye = circle_interior_points(center=(3, 3), radius=eye_radius, num_points=50)

    # Mouth arc (210° to 330°) shifted down
    theta_mouth = np.linspace(210 * np.pi / 180, 330 * np.pi / 180, 900)
    mouth_x = mouth_radius * np.cos(theta_mouth)
    mouth_y = mouth_radius * np.sin(theta_mouth) - 2
    mouth_coords = np.column_stack((mouth_x, mouth_y))

    # Combine all parts
    smiley_coords = np.concatenate((face_coords, left_eye, right_eye, mouth_coords), axis=0)

    # === Kernel Matrix & Eigendecomposition ===
    K_full = gaussian_kernel_matrices(smiley_coords, smiley_coords, bandwidth)
    full_trace = np.trace(K_full)

    eigvals, eigvecs = np.linalg.eigh(K_full)
    positive_idx = eigvals > 1e-10
    eigvals_filtered = eigvals[positive_idx]
    eigvecs_filtered = eigvecs[:, positive_idx]
    L = eigvecs_filtered @ np.diag(np.sqrt(eigvals_filtered))  # For kernel K-means++

    # === Landmark Visualisations ===

    # Uniform
    idx = sample_uniformly(smiley_coords, rank=50)
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(smiley_coords[:, 0], smiley_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(smiley_coords[idx, 0], smiley_coords[idx, 1], color='blue', s=100)
    plt.axis('equal')
    plt.title("Uniform Sampling")
    plt.show()

    # RPCholesky
    idx, F = sample_rpcholesky(smiley_coords, rank=50, bandwidth=bandwidth)
    plt.scatter(smiley_coords[:, 0], smiley_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(smiley_coords[idx, 0], smiley_coords[idx, 1], color='orange', s=100)
    plt.axis('equal')
    plt.title("RPCholesky Sampling")
    plt.show()

    # RRLS
    idx = sample_rrls(smiley_coords, rank=50, bandwidth=bandwidth)
    plt.scatter(smiley_coords[:, 0], smiley_coords[:, 1], color=color, alpha=0.5)
    plt.scatter(smiley_coords[idx, 0], smiley_coords[idx, 1], color='green', s=100)
    plt.axis('equal')
    plt.title("RRLS Sampling")
    plt.show()

    # Linear K-Means++
    idx, _ = sample_linear_kmeans(smiley_coords, rank=50)
    plt.scatter(smiley_coords[:, 0], smiley_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(smiley_coords[idx, 0], smiley_coords[idx, 1], color='red', s=100)
    plt.axis('equal')
    plt.title("Linear K-Means++ Sampling")
    plt.show()

    # Kernel K-Means++
    idx = sample_kernel_kmeans(L, rank=50)
    plt.scatter(smiley_coords[:, 0], smiley_coords[:, 1], color=color, linewidth=lw)
    plt.scatter(smiley_coords[idx, 0], smiley_coords[idx, 1], color='purple', s=100)
    plt.axis('equal')
    plt.title("Kernel K-Means++ Sampling")
    plt.show()

    # === Approximation Error Analysis ===
    ranks = np.arange(10, 101, 10)
    num_trials = 10
    methods = ['Uniform', 'RRLS', 'RPCholesky', 'Linear K-Means++', 'Kernel K-Means++']
    errors = {method: [] for method in methods}

    for r in ranks:
        err_uniform, err_rrls, err_rpchol, err_lkm, err_kkm = [], [], [], [], []

        for t in range(num_trials):
            # Uniform
            idx = sample_uniformly(smiley_coords, r, random_state=t)
            err_uniform.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

            # RRLS
            idx = sample_rrls(smiley_coords, rank=r, bandwidth=bandwidth, random_state=t)
            err_rrls.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

            # RPCholesky
            idx, F = sample_rpcholesky(smiley_coords, rank=r, bandwidth=bandwidth, random_state=t)
            err_rpchol.append(np.abs(full_trace - np.trace(F @ F.T)) / full_trace)

            # Linear K-Means++
            idx, _ = sample_linear_kmeans(smiley_coords, rank=r, random_state=t)
            err_lkm.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

            # Kernel K-Means++
            idx = sample_kernel_kmeans(L, rank=r, random_state=t)
            err_kkm.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

        # Average over trials
        errors['Uniform'].append(np.mean(err_uniform))
        errors['RRLS'].append(np.mean(err_rrls))
        errors['RPCholesky'].append(np.mean(err_rpchol))
        errors['Linear K-Means++'].append(np.mean(err_lkm))
        errors['Kernel K-Means++'].append(np.mean(err_kkm))

    # === Error Plot ===
    plt.figure(figsize=(7, 5))
    for method in methods:
        plt.plot(ranks, errors[method], marker='o', label=method)

    plt.xlabel('Rank (Number of Landmarks)')
    plt.ylabel('Relative Trace Error')
    plt.yscale('log')
    plt.title('Relative Trace Error vs. Rank on Smiley Dataset')
    plt.legend()
    plt.show()