# Import libraries
import numpy as np
from sklearn.cluster import KMeans
import scipy.linalg as spl
from dissertation_kernels import gaussian_kernel_vectors, gaussian_kernel_matrices


def sample_uniformly(X, rank, random_state=None):
    """
    Uniformly sample landmark indices from dataset X.

    Parameters:
        X (ndarray): Input dataset of shape (n_samples, n_features).
        rank (int): Number of landmarks to sample.
        random_state (int or None): Seed for reproducibility.

    Returns:
        ndarray: Sorted array of sampled indices.
    """
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], size=rank, replace=False)
    return np.sort(indices)


def sample_rpcholesky(X, rank, bandwidth, random_state=None):
    """
    Sample landmark points using RPCholesky (pivoted Cholesky) method.

    Parameters:
        X (ndarray): Input dataset of shape (n_samples, n_features).
        rank (int): Number of landmarks.
        bandwidth (float): Bandwidth parameter for the Gaussian kernel.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple:
            S (ndarray): Selected landmark indices of shape (rank,).
            F (ndarray): Feature matrix such that K ≈ F @ F.T
    """
    if random_state is not None:
        np.random.seed(random_state)

    N, d = X.shape
    F = np.zeros((N, rank))
    S = np.empty(rank, dtype=np.int64)
    d_vec = np.array([gaussian_kernel_vectors(x, x, bandwidth) for x in X])

    for i in range(rank):
        p = d_vec / d_vec.sum()
        s = np.random.choice(N, p=p)
        S[i] = s

        diff = X - X[s]
        g = np.exp(-np.sum(diff**2, axis=1) / (2 * bandwidth**2))

        if i > 0:
            g -= F[:, :i] @ F[s, :i].T

        pivot_val = g[s]
        if pivot_val <= 0:
            raise ValueError(f"Nonpositive pivot value encountered at step {i}.")

        F[:, i] = g / np.sqrt(pivot_val)
        d_vec = np.maximum(d_vec - F[:, i]**2, 0)

    return S, F


def sample_rrls(X, rank, bandwidth, accelerated_flag=False,
                random_state=None, lmbda_0=0, return_leverage_score=False):
    """
    Sample landmark points using Recursive Ridge Leverage Score (RRLS) sampling.

    Parameters:
        X (ndarray): Input dataset.
        rank (int): Number of landmarks to sample.
        bandwidth (float): Gaussian kernel bandwidth.
        accelerated_flag (bool): Not currently used.
        random_state (int or None): Seed for reproducibility.
        lmbda_0 (float): Base regularisation parameter.
        return_leverage_score (bool): Whether to return final leverage scores.

    Returns:
        indices (ndarray): Selected landmark indices.
        (Optional) leverage_scores (ndarray): Corresponding leverage scores.
    """
    rng = np.random.RandomState(random_state)

    n_oversample = np.log(rank)
    k = int(np.ceil(rank / (4 * n_oversample)))
    n_levels = int(np.ceil(np.log(X.shape[0] / rank) / np.log(2)))
    perm = rng.permutation(X.shape[0])

    size_list = [X.shape[0]]
    for l in range(1, n_levels + 1):
        size_list.append(int(np.ceil(size_list[l - 1] / 2)))

    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones(indices.shape[0])

    k_diag = np.array([gaussian_kernel_vectors(x, x, bandwidth) for x in X])

    for l in reversed(range(n_levels)):
        current_indices = perm[:size_list[l]]
        KS = gaussian_kernel_matrices(X[current_indices], X[indices], bandwidth)
        SKS = KS[sample]

        if k >= SKS.shape[0]:
            lmbda = 1e-5
        else:
            eigvals = spl.eigvalsh(SKS * weights[:, None] * weights[None, :],
                                   subset_by_index=(SKS.shape[0] - k, SKS.shape[0] - 1))
            lmbda = (np.sum(np.diag(SKS) * weights**2) - np.sum(eigvals)) / k

        lmbda = max(lmbda, lmbda_0 * SKS.shape[0])
        if lmbda == lmbda_0 * SKS.shape[0]:
            print(f"Set lambda to {lmbda:.2e}.")

        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = KS @ R

        if l != 0:
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) *
                                        np.maximum(0.0, (k_diag[current_indices] - np.sum(R * KS, axis=1))))
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            if sample.size == 0:
                leverage_score[:] = rank / size_list[l]
                sample = rng.choice(size_list[l], size=rank, replace=False)
            weights = np.sqrt(1.0 / leverage_score[sample])
        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) *
                                        np.maximum(0.0, (k_diag[current_indices] - np.sum(R * KS, axis=1))))
            p = leverage_score / leverage_score.sum()
            sample = rng.choice(X.shape[0], size=rank, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        final_scores = np.zeros(X.shape[0])
        final_scores[perm[:len(leverage_score)]] = leverage_score
        return indices, final_scores
    else:
        return indices


def sample_linear_kmeans(X, rank, random_state=None):
    """
    Select landmarks using standard K-Means++ in input space.

    Parameters:
        X (ndarray): Input dataset.
        rank (int): Number of clusters.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple:
            indices (ndarray): Landmark indices (medoids).
            centers (ndarray): Cluster centers in input space.
    """
    kmeans = KMeans(n_clusters=rank, random_state=random_state)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    indices = [np.argmin(np.linalg.norm(X - c, axis=1)) for c in centers]
    return np.array(indices), centers


def sample_kernel_kmeans(L, rank, random_state=None):
    """
    Select landmarks using K-Means++ in feature space L (e.g. from spectral kernel approximation).

    Parameters:
        L (ndarray): Low-rank kernel matrix factor such that K ≈ L @ L.T
        rank (int): Number of clusters.
        random_state (int or None): Seed for reproducibility.

    Returns:
        ndarray: Landmark indices (medoids in feature space).
    """
    kmeans = KMeans(n_clusters=rank, random_state=random_state)
    kmeans.fit(L)
    centers = kmeans.cluster_centers_

    cluster_indices = [np.argmin(np.linalg.norm(L - c, axis=1)) for c in centers]
    return np.array(cluster_indices)