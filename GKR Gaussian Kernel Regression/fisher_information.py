"""
Fisher Information Calculator
Extracted from AgeYY/speed_grid_cell_information repository

This module computes Fisher Information from a fitted model that provides
mean predictions and covariance matrices. 

Usage:
    1. Fit your model (e.g., GKR model) with covariance estimation
    2. Use compute_fisher_info_from_model() to compute Fisher Information
"""

import numpy as np


def compute_jacobian_central(f, X, h=1e-5, *args, **kwargs):
    """
    Estimate the Jacobian matrix J of a vector-valued function f at points X using central differences. 

    Parameters: 
    -----------
    f :  callable
        The function should accept an array of shape (N, D) and return an array of shape (N, P).
    X : ndarray
        2D NumPy array of shape (N, D), where N is the number of points and D is the dimension of the input space.
    h :  float
        Small step size for estimating the derivative.  Defaults to 1e-5.
    *args : tuple
        Additional positional arguments for the function f.
    **kwargs : dict
        Additional keyword arguments for the function f. 
    
    Returns: 
    --------
    J : ndarray
        A 3D NumPy array of shape (N, P, D), where J[n, : , i] represents the estimated partial derivatives 
        of the P outputs with respect to the i-th input variable, evaluated at the n-th point. 
    """
    N, D = X.shape  # Number of points (N) and dimensions of input space (D)
    
    # Precompute the output shape by evaluating the function at the first data point
    y = f(X[0]. reshape(1, -1), *args, **kwargs)
    P = y.shape[1]  # Number of dimensions in the output space
    
    # Initialize the Jacobian matrix with zeros
    J = np.zeros((N, P, D))
    
    for i in range(D):  # Iterate over each dimension
        # Shift the nth data point in the positive and negative direction along the ith dimension
        X_plus_h = np.copy(X)
        X_minus_h = np.copy(X)
        X_plus_h[: , i] += h / 2.0
        X_minus_h[:, i] -= h / 2.0
            
        # Evaluate the function at the shifted data points
        y_plus = f(X_plus_h, *args, **kwargs)
        y_minus = f(X_minus_h, *args, **kwargs)

        # Estimate the partial derivative using central difference
        J[:, :, i] = (y_plus - y_minus) / h
    
    return J


def compute_fisher_info(jacobian, precision_matrix):
    """
    Compute the Fisher information matrix. 
    
    The Fisher information is computed as:  F = J^T * Σ^{-1} * J
    where J is the Jacobian matrix and Σ^{-1} is the precision matrix (inverse of covariance).
    
    Parameters:
    -----------
    jacobian : ndarray
        The Jacobian matrix with shape (n_sample, n_feature, n_label).
        n_feature is the dimension of the output space (e.g., neural activity).
        n_label is the dimension of the input space (e.g., position, speed).
    precision_matrix : ndarray
        The precision matrix with shape (n_sample, n_feature, n_feature).
        This is the inverse of the covariance matrix. 
    
    Returns:
    --------
    fisher :  ndarray
        The Fisher information matrix with shape (n_sample, n_label, n_label).
    """
    return np.einsum('nij,nik,nkl->njl', jacobian, precision_matrix, jacobian)


def compute_total_fisher(jacobian, precision_matrix):
    """
    Compute the total Fisher information (trace of Fisher information matrix).
    
    Parameters:
    -----------
    jacobian : ndarray
        The Jacobian matrix with shape (n_sample, n_feature, n_label).
    precision_matrix : ndarray
        The precision matrix with shape (n_sample, n_feature, n_feature).
    
    Returns: 
    --------
    total_fisher : ndarray
        The total Fisher information for each sample, shape (n_sample,).
    """
    fisher = compute_fisher_info(jacobian, precision_matrix)
    return np.trace(fisher, axis1=1, axis2=2)


def compute_fisher_info_from_model(model, query_points, h):
    """
    Compute Fisher information from a fitted model. 
    
    This is the main function to use after fitting a model (e.g., GKR model).
    The model should have a predict() method that returns (mean, covariance).
    
    Parameters:
    -----------
    model : object
        A fitted model with a predict(query, return_cov=True/False) method.
        - predict(query, return_cov=False) should return (mean, None)
        - predict(query, return_cov=True) should return (mean, covariance)
    query_points : ndarray
        Query points with shape (n_sample, n_label).
        For example, [x, y, speed] coordinates.
    h :  float
        Step size for numerical differentiation.  Default is 0.01.
    
    Returns: 
    --------
    fisher :  ndarray
        Fisher information matrices with shape (n_sample, n_label, n_label).
    total_fisher : ndarray
        Total Fisher information (trace) for each sample, shape (n_sample,).
    feamap : ndarray
        Predicted feature map (mean) with shape (n_sample, n_feature).
    cov : ndarray
        Predicted covariance matrices with shape (n_sample, n_feature, n_feature).
    """
    # Get predictions from the model
    feamap, cov = model.predict(query_points, return_cov=True)
    
    # Create a wrapper function for Jacobian computation (mean only)
    def model_mean_wrapper(query):
        return model.predict(query, return_cov=False)[0]
    
    # Compute Jacobian matrix using central differences
    jacobian = compute_jacobian_central(model_mean_wrapper, query_points, h=h)
    
    # Compute precision matrix (inverse of covariance)
    precision_matrix = np.linalg.inv(cov)
    
    # Compute Fisher information
    fisher = compute_fisher_info(jacobian, precision_matrix)
    total_fisher = np.trace(fisher, axis1=1, axis2=2)
    
    return fisher, total_fisher, feamap, cov


def compute_fisher_for_direction(direction_vector, jacobian, covariance):
    """
    Compute Fisher information for a specific direction. 
    
    Parameters:
    -----------
    direction_vector :  ndarray
        Direction vectors with shape (n_sample, n_feature).
    jacobian : ndarray
        Jacobian matrix with shape (n_sample, n_feature, n_label).
    covariance : ndarray
        Covariance matrices with shape (n_sample, n_feature, n_feature).
    
    Returns:
    --------
    fisher_matrices : ndarray
        Fisher information matrices for the given direction.
    """
    # Project variance onto direction
    d_norm = direction_vector / np. linalg.norm(direction_vector, axis=1)[:, np.newaxis]
    variance_projection = np.einsum('ij,ijk,ik->i', d_norm, covariance, d_norm)
    inv_s = 1.0 / variance_projection
    
    # Compute derivative along direction
    derivative = np.einsum('ij,ijk->ik', direction_vector, jacobian)
    fisher_matrices = np. einsum('ij,ik,i->ijk', derivative, derivative, inv_s)
    
    return fisher_matrices


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Example:  Computing Fisher information from a mock model
    
    # Create a simple mock model for demonstration
    class MockModel:
        """A simple mock model for demonstration purposes."""
        
        def __init__(self, n_feature=6):
            self.n_feature = n_feature
        
        def predict(self, query, return_cov=True):
            n_sample = query. shape[0]
            # Mock mean prediction (random)
            mean = np.random.randn(n_sample, self.n_feature)
            
            if return_cov: 
                # Mock covariance (positive definite)
                cov = np.zeros((n_sample, self.n_feature, self.n_feature))
                for i in range(n_sample):
                    A = np.random.randn(self.n_feature, self.n_feature)
                    cov[i] = A @ A.T + np.eye(self.n_feature) * 0.1
                return mean, cov
            return mean, None
    
    # Create mock model and query points
    model = MockModel(n_feature=6)
    n_samples = 100
    n_labels = 3  # e.g., [x, y, speed]
    query_points = np.random.randn(n_samples, n_labels)
    
    # Compute Fisher information
    fisher, total_fisher, feamap, cov = compute_fisher_info_from_model(
        model, query_points, h=0.01
    )
    
    print("=" * 50)
    print("Fisher Information Computation Results")
    print("=" * 50)
    print(f"Query points shape: {query_points.shape}")
    print(f"Feature map shape: {feamap.shape}")
    print(f"Covariance shape: {cov.shape}")
    print(f"Fisher information shape: {fisher.shape}")
    print(f"Total Fisher information shape: {total_fisher.shape}")
    print(f"Mean total Fisher:  {total_fisher.mean():.4f}")
    print("=" * 50)