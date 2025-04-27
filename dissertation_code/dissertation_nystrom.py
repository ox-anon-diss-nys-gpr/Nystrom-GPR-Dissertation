# Import libraries
import numpy as np
from numpy.linalg import pinv

def nystrom_approximation(K, indices):
    """
    Compute the Nyström approximation of a positive semi-definite kernel matrix.

    Parameters:
        K (ndarray): Full kernel matrix of shape (n_samples, n_samples).
        indices (array-like): Indices of landmark points used for approximation.

    Returns:
        ndarray: Nyström-approximated kernel matrix of shape (n_samples, n_samples),
                 computed as K_nm @ K_mm_pinv @ K_nm.T
    """
    # Extract relevant submatrices
    K_nm = K[:, indices]                   # All rows, landmark columns
    K_mm = K[np.ix_(indices, indices)]     # Landmark submatrix (symmetric)
    
    # Compute pseudoinverse of the landmark submatrix
    K_mm_pinv = pinv(K_mm)

    # Return the Nyström approximation
    return K_nm @ K_mm_pinv @ K_nm.T