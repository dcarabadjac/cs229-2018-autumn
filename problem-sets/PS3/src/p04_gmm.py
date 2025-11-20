import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import multivariate_normal

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, k = w.shape
    nom = np.zeros((m, k))
    
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        for j, (muj, sigmaj, phij) in enumerate(zip(mu, sigma, phi)):
            rv = multivariate_normal(mean=muj, cov=sigmaj)
            px_for_zj = rv.pdf(x)
            pz = phij
            nom[:, j] = px_for_zj*pz
        denom = np.sum(nom, axis=1, keepdims=True) #p(x)
        w = nom/denom #shape m, k

        sumw = np.sum(w, axis=0, keepdims=True).T
        phi = sumw/m
        mu = w.T@x/sumw

        sigma = np.zeros_like(sigma)
        for j in range(k):
            sigma[j] = w.T[j].reshape(1, -1)*(x - mu[j]).T@(x - mu[j])/ sumw[j]

        prev_ll = ll
        ll = np.sum(np.log(denom))
        if it%10 == 0:
            print(it, ll)
        it += 1
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, k = w.shape
    mtilde = z.shape[0]
    nom = np.zeros((m, k))
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        
        for j, (muj, sigmaj, phij) in enumerate(zip(mu, sigma, phi)):
            rv = multivariate_normal(mean=muj, cov=sigmaj)
            px_for_zj = rv.pdf(x)
            pz = phij
            nom[:, j] = px_for_zj*pz
        denom = np.sum(nom, axis=1, keepdims=True) #p(x)
        w = nom/denom #shape m, k

        z = z.astype(int).ravel()
        counts = np.bincount(z, minlength=K).reshape(-1, 1)
        onehot = (z[:, None] == np.arange(K))  # (mtilde, k)
        
        sumw = np.sum(w, axis=0, keepdims=True).T
        phi = (sumw+alpha*counts)/(m + alpha*mtilde)
        mu = (w.T@x + alpha*onehot.T@x_tilde)/(sumw + alpha*counts)

        sigma = np.zeros_like(sigma)
        for j in range(k):
            sigma[j] = w.T[j].reshape(1, -1)*(x - mu[j]).T@(x - mu[j]) + alpha*(x_tilde[z==j] - mu[j]).T@(x_tilde[z==j] - mu[j])
            sigma[j] = sigma[j]/(sumw[j] + alpha*counts[j])
        prev_ll = ll
        ll = np.sum(np.log(denom))
        if it%10 == 0:
            print(it, ll)
        it += 1
    return w


# *** START CODE HERE ***
# Helper functions
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(with_supervision=True, trial_num=t)
        # *** END CODE HERE ***
