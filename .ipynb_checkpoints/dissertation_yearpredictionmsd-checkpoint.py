# === Import libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from dissertation_metrics import *
from dissertation_gpr import approximation_gpr
from dissertation_sampling import *
from dissertation_nystrom import nystrom_approximation
from dissertation_kernels import gaussian_kernel_matrices


if __name__ == "__main__":
    # === Load and preprocess YearPredictionMSD dataset ===
    file_path = r"C:\Users\mamun\Downloads\yearpredictionmsd\YearPredictionMSD.txt"
    column_names = ['Year'] + [f'Feature_{i}' for i in range(1, 91)]
    data = pd.read_csv(file_path, header=None, names=column_names)

    # Split into training and test sets
    train_data = data.iloc[:463715]
    test_data = data.iloc[463715:]

    X_train = train_data.drop(columns=['Year'])
    y_train = train_data['Year']
    X_test = test_data.drop(columns=['Year'])
    y_test = test_data['Year']

    # Standardize inputs and targets
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    y_std = y_scaler.scale_[0]

    # === Parameters ===
    ranks = [250, 500, 750, 1000]
    noise_candidates = [1e-2, 1e-1]
    bandwidth_candidates = suggest_bandwidth_range(X_train_scaled)
    cv_folds = 3
    num_trials = 10

    # Subsample for CV
    X_sub = X_train_scaled[:30000]
    y_sub = y_train_scaled[:30000]

    # === Hyperparameter selection ===
    print("Cross-validating bandwidth and noise per method...")

    uniform_noise, uniform_bandwidth = cv_sampling_method(X_sub, y_sub, ranks, noise_candidates, bandwidth_candidates, method='uniform', cv_folds=cv_folds)
    rpcholesky_noise, rpcholesky_bandwidth = cv_sampling_method(X_sub, y_sub, ranks, noise_candidates, bandwidth_candidates, method='rpcholesky', cv_folds=cv_folds)
    rrls_noise, rrls_bandwidth = cv_sampling_method(X_sub, y_sub, ranks, noise_candidates, bandwidth_candidates, method='rrls', cv_folds=cv_folds)
    linear_kmeans_noise, linear_kmeans_bandwidth = cv_sampling_method(X_sub, y_sub, ranks, noise_candidates, bandwidth_candidates, method='linear_kmeans', cv_folds=cv_folds)

    # === Benchmark GPR prediction accuracy and runtime ===
    print("\nBenchmarking sampling methods...")
    
    uniform_rmse, uniform_sampling_time, uniform_gpr_time = benchmark_sampling_method(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        ranks, uniform_noise, uniform_bandwidth, num_trials, method='uniform'
    )
    rpcholesky_rmse, rpcholesky_sampling_time, rpcholesky_gpr_time = benchmark_sampling_method(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        ranks, rpcholesky_noise, rpcholesky_bandwidth, num_trials, method='rpcholesky'
    )
    rrls_rmse, rrls_sampling_time, rrls_gpr_time = benchmark_sampling_method(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        ranks, rrls_noise, rrls_bandwidth, num_trials, method='rrls'
    )
    linear_kmeans_rmse, linear_kmeans_sampling_time, linear_kmeans_gpr_time = benchmark_sampling_method(
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled,
        ranks, linear_kmeans_noise, linear_kmeans_bandwidth, num_trials, method='linear_kmeans'
    )
    
    # === Plot RMSE and runtime vs rank ===
    methods = ['Uniform', 'RPCholesky', 'RRLS', 'Linear K-Means++']
    
    rmse_data = {
        'Uniform': extract_and_rescale(uniform_rmse, y_std),
        'RPCholesky': extract_and_rescale(rpcholesky_rmse, y_std),
        'RRLS': extract_and_rescale(rrls_rmse, y_std),
        'Linear K-Means++': extract_and_rescale(linear_kmeans_rmse, y_std)
    }
    
    sampling_runtime_data = {
        'Uniform': extract_runtime(uniform_sampling_time),
        'RPCholesky': extract_runtime(rpcholesky_sampling_time),
        'RRLS': extract_runtime(rrls_sampling_time),
        'Linear K-Means++': extract_runtime(linear_kmeans_sampling_time)
    }
    
    gpr_runtime_data = {
        'Uniform': extract_runtime(uniform_gpr_time),
        'RPCholesky': extract_runtime(rpcholesky_gpr_time),
        'RRLS': extract_runtime(rrls_gpr_time),
        'Linear K-Means++': extract_runtime(linear_kmeans_gpr_time)
    }
    
    # RMSE Plot
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ranks, rmse_data[method], marker='o', label=method)
    plt.xlabel("Rank")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Rank (YearPredictionMSD Dataset)")
    plt.legend()
    plt.show()
    
    # Sampling Runtime Plot
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ranks, sampling_runtime_data[method], marker='o', label=method)
    plt.xlabel("Rank")
    plt.ylabel("Sampling Time (s)")
    plt.title("Sampling Time vs Rank (YearPredictionMSD Dataset)")
    plt.legend()
    plt.show()
    
    # GPR Runtime Plot
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ranks, gpr_runtime_data[method], marker='o', label=method)
    plt.xlabel("Rank")
    plt.ylabel("GPR Time (s)")
    plt.title("GPR Time vs Rank (YearPredictionMSD Dataset)")
    plt.legend()
    plt.show()

    # Full Runtime Plot
    combined = {}
    for method in sampling_runtime_data:
        combined[method] = [
            s + g for s, g in zip(sampling_runtime_data[method], gpr_runtime_data[method])
        ]
    plt.figure(figsize=(6, 4))
    for method in methods:
        plt.plot(ranks, combined[method], marker='o', label=method)
    plt.xlabel("Rank")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime vs Rank (YearPredictionMSD Dataset)")
    plt.legend()
    plt.show()

    # === Trace error computation ===
    print("\nEvaluating relative trace error...")

    bandwidth = 8.0  # Fixed bandwidth for trace evaluation
    errors = {method: [] for method in methods}

    for r in ranks:
        print(f"→ Rank {r}")
        err_uniform, err_rrls, err_rpchol, err_linear_kmeans = [], [], [], []

        for t in range(num_trials):
            print(f"  Trial {t+1}/{num_trials}")
            idx_sample = np.random.choice(X_train_scaled.shape[0], 10000, replace=False)
            X_sample = X_train_scaled[idx_sample]

            K_full = gaussian_kernel_matrices(X_sample, X_sample, bandwidth)
            full_trace = np.trace(K_full)

            # Uniform
            idx = sample_uniformly(X_sample, r, random_state=t)
            err_uniform.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

            # RRLS
            idx = sample_rrls(X_sample, r, bandwidth, random_state=t)
            err_rrls.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

            # RPCholesky
            idx, F = sample_rpcholesky(X_sample, r, bandwidth, random_state=t)
            err_rpchol.append(np.abs(full_trace - np.trace(F @ F.T)) / full_trace)

            # Linear K-Means++
            idx, _ = sample_linear_kmeans(X_sample, r, random_state=t)
            err_linear_kmeans.append(np.abs(full_trace - np.trace(nystrom_approximation(K_full, idx))) / full_trace)

        errors['Uniform'].append(np.mean(err_uniform))
        errors['RRLS'].append(np.mean(err_rrls))
        errors['RPCholesky'].append(np.mean(err_rpchol))
        errors['Linear K-Means++'].append(np.mean(err_linear_kmeans))

    # Trace Error Plot
    plt.figure(figsize=(8, 6))
    for method in methods:
        plt.plot(ranks, errors[method], marker='o', label=method)
    plt.xlabel('Rank')
    plt.ylabel('Relative Trace Error')
    plt.yscale('log')
    plt.title("Relative Trace Error vs Rank (YearPredictionMSD)")
    plt.legend()
    plt.show()