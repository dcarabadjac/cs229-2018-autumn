import numpy as np
from . import util
from .linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m = x.shape[0]
        n = x.shape[1]
        unmask_zeros = (y == 0)
        unmask_ones  = (y == 1)
        
        phi = np.sum(y)/m
        mu0 = np.sum(x[unmask_zeros], axis=0)/(m-np.sum(y))
        mu1 = np.sum(x[unmask_ones], axis=0)/np.sum(y)
        
        Mu  = np.zeros_like(x)
        Mu[unmask_zeros] = mu0
        Mu[unmask_ones] = mu1    
        Sigma = 1/m*(x-Mu).T @ (x-Mu)
        sigma_inv = np.linalg.inv(Sigma)
        
        self.theta = np.zeros(x.shape[1]+1)
        self.theta[1:] = sigma_inv@(mu1 - mu0)
        self.theta[0]  = -np.log((1-phi)/phi) - 1/2*(mu1@(sigma_inv@mu1) - mu0@(sigma_inv@mu0))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))    

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return self.sigmoid(x@self.theta[1:] + self.theta[0])
