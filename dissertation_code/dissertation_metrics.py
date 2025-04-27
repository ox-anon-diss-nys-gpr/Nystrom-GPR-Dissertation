# Import libraries
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
from dissertation_gpr import approximation_gpr
from dissertation_sampling import *


def compute_rmse(y_true, y_pred):
    """
    Compute root mean squared error (RMSE).

    Parameters:
        y_true (ndarray): True target values.
        y_pred (ndarray): Predicted target values.

    Returns:
        float: RMSE between true and predicted values.
    """
    return np.sqrt(np.mean((y_true.ravel() - y_pred.ravel()) ** 2))


def suggest_bandwidth_range(X, sample_size=1000):
    """
    Suggest a range of candidate bandwidths based on pairwise distances.

    Parameters:
        X (ndarray): Input data.
        sample_size (int): Number of points to use for estimation (to reduce cost).

    Returns:
        list: List of candidate bandwidths (rounded integers between percentiles).
    """
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    dists = pairwise_distances(X_sample, metric='euclidean')
    triu_indices = np.triu_indices_from(dists, k=1)
    flat_dists = dists[triu_indices]

    stats = {
        "min": np.min(flat_dists),
        "25th percentile": np.percentile(flat_dists, 25),
        "median": np.median(flat_dists),
        "mean": np.mean(flat_dists),
        "75th percentile": np.percentile(flat_dists, 75),
        "max": np.max(flat_dists)
    }

    print("Suggested bandwidth candidates from pairwise distances (on a sample):")
    for name, val in stats.items():
        print(f"{name:<18}: {val:.4f}")

    lower = int(np.floor(stats["25th percentile"]))
    upper = int(np.ceil(stats["75th percentile"]))
    return list(range(lower, upper + 1))


def cv_sampling_method(X, y, ranks, noise_list, bandwidth_list, method, cv_folds=5, random_state=None):
    """
    Cross-validation to select the best noise and bandwidth parameters for a given sampling method.

    Parameters:
        X (ndarray): Input features.
        y (ndarray): Targets.
        ranks (list): List of landmark counts to evaluate.
        noise_list (list): List of candidate noise levels.
        bandwidth_list (list): List of candidate kernel bandwidths.
        method (str): Sampling method name ('uniform', 'rpcholesky', 'rrls', 'linear_kmeans').
        cv_folds (int): Number of folds in cross-validation.
        random_state (int or None): Seed for reproducibility.

    Returns:
        tuple: Lists of best noise levels and bandwidths per rank.
    """
    best_configs = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    for rank in ranks:
        print(f"\n=== Rank: {rank} ===")
        best_rmse = float('inf')
        best_combo = None

        for noise in noise_list:
            for bandwidth in bandwidth_list:
                rmse_scores = []

                for train_index, val_index in kf.split(X):
                    X_tr, X_val = X[train_index], X[val_index]
                    y_tr, y_val = y[train_index], y[val_index]

                    # Sampling
                    if method == 'uniform':
                        indices = sample_uniformly(X_tr, rank)
                    elif method == 'rpcholesky':
                        indices, _ = sample_rpcholesky(X_tr, rank, bandwidth)
                    elif method == 'rrls':
                        indices = sample_rrls(X_tr, rank, bandwidth)
                    elif method == 'linear_kmeans':
                        indices, _ = sample_linear_kmeans(X_tr, rank)
                    else:
                        raise ValueError(f"Unknown method '{method}'")

                    # GPR prediction
                    y_pred = approximation_gpr(X_tr, X_val, y_tr, noise, bandwidth, indices, rank)
                    rmse = compute_rmse(y_val, y_pred)
                    rmse_scores.append(rmse)

                avg_rmse = np.mean(rmse_scores)
                print(f"Noise={noise}, Bandwidth={bandwidth} → Avg RMSE={avg_rmse:.4f}")

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_combo = (noise, bandwidth)

        best_configs[rank] = (*best_combo, best_rmse)
        print(f"→ Best for Rank={rank}: Noise={best_combo[0]}, Bandwidth={best_combo[1]}, RMSE={best_rmse:.4f}")

    best_noise_list = [best_configs[r][0] for r in ranks]
    best_bandwidth_list = [best_configs[r][1] for r in ranks]

    return best_noise_list, best_bandwidth_list


def benchmark_sampling_method(X_train, y_train, X_test, y_test,
                               ranks, noise_list, bandwidth_list,
                               num_trials, method):
    """
    Benchmark RMSE and runtime of Nyström GPR with a given sampling method.

    Parameters:
        X_train, y_train: Training data and targets.
        X_test, y_test: Test data and targets.
        ranks (list): List of landmark counts to evaluate.
        noise_list (list): Best noise per rank.
        bandwidth_list (list): Best bandwidth per rank.
        num_trials (int): Number of repetitions per rank.
        method (str): Sampling method name.

    Returns:
        tuple: RMSE results, sampling time results, GPR time results.
               Each as a list of (rank, value).
    """
    rmse_results = []
    sampling_time_results = []
    gpr_time_results = []

    for rank, noise, bandwidth in zip(ranks, noise_list, bandwidth_list):
        print(f"Running GPR with Nyström ({method}) - Rank: {rank}")

        trial_rmses = []
        trial_sampling_times = []
        trial_gpr_times = []

        for trial in range(num_trials):
            # Time sampling
            start_sampling = time.time()
            if method == 'uniform':
                indices = sample_uniformly(X_train, rank)
            elif method == 'rpcholesky':
                indices, _ = sample_rpcholesky(X_train, rank, bandwidth)
            elif method == 'rrls':
                indices = sample_rrls(X_train, rank, bandwidth)
            elif method == 'linear_kmeans':
                indices, _ = sample_linear_kmeans(X_train, rank)
            else:
                raise ValueError(f"Unknown method '{method}'")
            sampling_time = time.time() - start_sampling

            # Time GPR
            start_gpr = time.time()
            y_pred = approximation_gpr(X_train, X_test, y_train, noise, bandwidth, indices, rank)
            gpr_time = time.time() - start_gpr

            rmse = compute_rmse(y_test, y_pred)

            trial_rmses.append(rmse)
            trial_sampling_times.append(sampling_time)
            trial_gpr_times.append(gpr_time)

        avg_rmse = np.mean(trial_rmses)
        avg_sampling_time = np.mean(trial_sampling_times)
        avg_gpr_time = np.mean(trial_gpr_times)

        print(f"Rank {rank}: Avg RMSE = {avg_rmse:.4f}, "
              f"Sampling Time = {avg_sampling_time:.4f}s, "
              f"GPR Time = {avg_gpr_time:.4f}s")

        rmse_results.append((rank, avg_rmse))
        sampling_time_results.append((rank, avg_sampling_time))
        gpr_time_results.append((rank, avg_gpr_time))

    return rmse_results, sampling_time_results, gpr_time_results


def extract_and_rescale(rmse_list, y_std):
    """
    Rescale RMSE values using target standard deviation.

    Parameters:
        rmse_list (list): List of (rank, rmse).
        y_std (float): Standard deviation of original targets.

    Returns:
        list: Rescaled RMSE values.
    """
    return [rmse * y_std for (_, rmse) in rmse_list]


def extract_runtime(runtime_list):
    """
    Extract runtime values from benchmark results.

    Parameters:
        runtime_list (list): List of (rank, runtime).

    Returns:
        list: Raw runtime values.
    """
    return [runtime for (_, runtime) in runtime_list]