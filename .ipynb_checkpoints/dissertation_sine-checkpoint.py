# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from dissertation_gpr import full_gpr_gaussian

def generate_data_sine(n, noise):
    """
    Generate a noisy sine wave dataset.

    Parameters:
        n (int): Number of data points.
        noise (float): Standard deviation of Gaussian noise.

    Returns:
        X (ndarray): Input features of shape (n, 1).
        y (ndarray): Noisy sine values of shape (n,).
    """
    X = np.random.uniform(0, 1, size=n)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, noise, size=n)
    return X.reshape(-1, 1), y

if __name__ == "__main__":
    # Generate training data
    X_train, y_train = generate_data_sine(n=10, noise=0.01)

    # Generate dense test inputs
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)

    # GPR models with different bandwidths
    settings = [
        {"noise": 0.01, "bandwidth": 0.05, "label": "Bandwidth = 0.05"},
        {"noise": 0.01, "bandwidth": 0.1,  "label": "Bandwidth = 0.1"},
        {"noise": 0.01, "bandwidth": 0.2,  "label": "Bandwidth = 0.2"},
    ]

    plt.figure(figsize=(7, 5))

    # Training data
    plt.scatter(X_train.ravel(), y_train.ravel(), color='red', alpha=0.6, label='Training Data')

    # True sine function
    plt.plot(X_test.ravel(), np.sin(2 * np.pi * X_test.ravel()), 'g--', label='True Function')

    # Colours for different GPRs
    colours = ['blue', 'purple', 'orange']

    # GPR predictions
    for idx, setting in enumerate(settings):
        mean, cov = full_gpr_gaussian(X_train, y_train, X_test,
                                      noise=setting["noise"],
                                      bandwidth=setting["bandwidth"])
        std = np.sqrt(np.diag(cov))

        # Plot mean
        plt.plot(X_test.ravel(), mean.ravel(), label=f'GPR Prediction ({setting["label"]})', color=colours[idx])

        # Plot uncertainty (±2 std dev)
        plt.fill_between(X_test.ravel(),
                         (mean - 2 * std).ravel(),
                         (mean + 2 * std).ravel(),
                         color=colours[idx], alpha=0.2)

    # Final plot setup
    plt.title("Exact GPR with Different Hyperparemeters")
    plt.legend()
    plt.show()