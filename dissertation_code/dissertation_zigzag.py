# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from dissertation_gpr import full_gpr_matern, full_gpr_gaussian

def generate_data_zigzag(n, noise):
    """
    Generate a synthetic zigzag dataset with added Gaussian noise.

    Parameters:
        n (int): Number of data points.
        noise (float): Standard deviation of the Gaussian noise.

    Returns:
        X (ndarray): Input features of shape (n, 1).
        y (ndarray): Noisy target values of shape (n,).
    """
    X = np.linspace(0, 1, n)
    y = 1 - 4 * np.abs((X * 10) % 1 - 0.5)  # Zigzag pattern
    y += np.random.normal(0, noise, size=n)  # Add Gaussian noise
    return X.reshape(-1, 1), y


if __name__ == "__main__":
    # Generate training data
    X_train, y_train = generate_data_zigzag(n=50, noise=0.1)

    # Generate dense test points
    X_test = np.linspace(0, 1, 1000).reshape(-1, 1)

    # Compute ground truth (without noise)
    true_function = 1 - 4 * np.abs((X_test.ravel() * 10) % 1 - 0.5)

    # Gaussian Process Regression with Matérn kernel (ν = 1.5)
    mean_matern, cov_matern = full_gpr_matern(
        X_train, y_train, X_test, noise=0.1, bandwidth=0.1, nu=1.5)
    std_matern = np.sqrt(np.diag(cov_matern))  # Standard deviation

    # Gaussian Process Regression with Gaussian kernel
    mean_gaussian, cov_gaussian = full_gpr_gaussian(
        X_train, y_train, X_test, noise=0.1, bandwidth=0.1)
    std_gaussian = np.sqrt(np.diag(cov_gaussian))

    # Plot results
    plt.figure(figsize=(6.2, 4.2))

    # True function
    plt.plot(X_test, true_function, 'g--', label='True Function')

    # Training data
    plt.scatter(X_train, y_train, color='gray', alpha=0.6, label='Training Data')

    # Matérn prediction
    plt.plot(X_test, mean_matern, 'b-', label='GPR Prediction (Matérn)')
    plt.fill_between(X_test.ravel(),
                     mean_matern - 2 * std_matern,
                     mean_matern + 2 * std_matern,
                     color='blue', alpha=0.2)

    # Gaussian prediction
    plt.plot(X_test, mean_gaussian, 'r-', label='GPR Prediction (Gaussian)')
    plt.fill_between(X_test.ravel(),
                     mean_gaussian - 2 * std_gaussian,
                     mean_gaussian + 2 * std_gaussian,
                     color='red', alpha=0.2)

    # Final plot setup
    plt.title("Exact GPR with Matérn vs Gaussian Kernel")
    plt.legend()
    plt.show()