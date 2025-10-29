import numpy as np
from . import util
from .linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    
    def __init__(self):
        super().__init__()
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    def hessian(self, x):
        result = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            z = self.theta @ x[i]
            result += self.sigmoid(z)*(1-self.sigmoid(z))*np.outer(x[i], x[i])
        return result/x.shape[0]
    
    def inv_hessian(self, x):
        return np.linalg.inv(self.hessian(x))

    def grad_lost_function(self, x, y):
        result = x.T@(self.predict(x) - y)
        return result/x.shape[0]

    def fit(self, x, y, epsilon=1e-5):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        res = 10e8
        self.theta = np.zeros(x.shape[1])
        niter = 0
        while res > epsilon:
            if niter < self.max_iter:
                niter += 1
                theta_old = self.theta
                self.theta = self.theta - self.inv_hessian(x)@self.grad_lost_function(x, y)
                res = np.linalg.norm(self.theta-theta_old, ord=1) 
            else:
                raise RunTimeError(f"Reached maximal number of iterations {self.max_iter}")

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return self.sigmoid(x@self.theta)
