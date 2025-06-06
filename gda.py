# Imports - you can add any other permitted libraries
import numpy as np
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.


class GaussianDiscriminantAnalysis:
    # Assume Binary Classification
    def __init__(self):
        self.mu0 = []
        self.mu1 = []
        self.sigma = []
        self.sigma1 = []
        self.sigma2 = []
        self.C = 0
        self.assume_same_cov = False
        self.normalize_mean = 0
        self.normalize_var = 1
        pass
    
    def fit(self, X, y, assume_same_covariance=False):
        """
        Fit the Gaussian Discriminant Analysis model to the data.
        Remember to normalize the input data X before fitting the model.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target labels - 0 or 1.
        
        learning_rate : float
            The learning rate to use in the update rule.
        
        Returns
        -------
        Parameters: 
            If assume_same_covariance = True - 3-tuple of numpy arrays mu_0, mu_1, sigma 
            If assume_same_covariance = False - 4-tuple of numpy arrays mu_0, mu_1, sigma_0, sigma_1
            The parameters learned by the model.
        """
        
        self.assume_same_cov = assume_same_covariance
        phi = np.sum(y == 1) / len(y)
        X = self.normalize(X,True)
        m,n = X.shape
        X0 = X[y==0]
        mu0 = np.mean(X0, axis = 0)
        X1 = X[y==1]
        mu1 = np.mean(X1, axis = 0)
        self.mu0 = mu0
        self.mu1 = mu1
        if (assume_same_covariance):
            sigma = np.zeros((n,n))
            for i in range(m):
                mu = np.zeros(n)
                if (y[i] == 0):
                    mu = mu0
                else:
                    mu = mu1
                sigma += (np.outer(X[i] - mu,X[i]-mu))
            sigma = sigma/m
            self.sigma = sigma
            self.C = np.log(phi/(1-phi))
            return mu0, mu1, sigma
        
        else:
            sigma0 = np.zeros((n,n))
            sigma1 = np.zeros((n,n))
            for i in range(m):
                if (y[i] == 0):
                    sigma0 += (np.outer(X[i] - mu0,X[i]-mu0))
                else:
                    sigma1 += (np.outer(X[i] - mu1,X[i]-mu1))
            if (len(X0) > 0):
                sigma0 = sigma0/len(X0)
            if (len(X1) > 0):
                sigma1 = sigma1/len(X1)
            self.sigma1 = sigma1
            self.sigma0 = sigma0
            det0 = np.linalg.det(sigma0)
            det1 = np.linalg.det(sigma1)
            self.C = np.log(phi/(1-phi)) + (0.5 * np.log(det0/det1))
            return mu0, mu1, sigma0, sigma1

    def predict(self, X):
        """
        Predict the target values for the input data.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        y_pred : numpy array of shape (n_samples,)
            The predicted target label.
        """
        X = (X-self.normalize_mean)/self.normalize_var
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if (self.assume_same_cov):
                if (self.equation(X[i][0], X[i][1],self.sigma, self.sigma, self.mu1, self.mu0, self.C) > 0):
                    pred[i] = 1
            else:
                if (self.equation(X[i][0], X[i][1],self.sigma1, self.sigma0, self.mu1, self.mu0, self.C) > 0):
                    pred[i] = 1
        return pred
    
    def normalize(self, X, store_val = False):
        mean = np.mean(X,axis = 0)
        var = np.std(X,axis = 0)
        if (store_val):
            self.normalize_mean = mean
            self.normalize_var = var
        X = (X-mean)/var
        return X
    
    def equation(self,x1,x2, sigma1, sigma0, mu1, mu0, C):
        sigma0_inv = np.linalg.inv(sigma0)
        sigma1_inv = np.linalg.inv(sigma1)
        X = np.array([x1, x2])
        term1 = ((X.T @ (sigma1_inv - sigma0_inv)) @ X)
        term2 = 2*(X.T @ ((sigma0_inv @ mu0) - (sigma1_inv @ mu1)))
        term3 = ((mu1.T @ sigma1_inv) @ mu1) - ((mu0.T @ sigma0_inv) @ mu0)
        return (C-(0.5 * (term1 + term2 + term3)))
        