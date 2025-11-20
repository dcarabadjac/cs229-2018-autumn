import numpy as np


def run_kmeans(x, mu, K):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        mu: Initial cluster means of shape (k, n).

    Returns:
        mu: cluster centroids after convergence
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 30

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    prev_mu = None
    
    while it < max_iter and (prev_mu is None or np.linalg.norm(mu - prev_mu) >= eps):
        prev_mu = mu
        dists = np.linalg.norm(x[:, None, :] - mu[None, :, :], axis=2) #m, k
        c = np.argmin(dists, axis=1) #m
        R = (c[:, None] == np.arange(K)[None, :]).astype(int) # m, k
        mu = R.T@x/R.sum(axis=0)[:, None]      
        it += 1
    return c, mu