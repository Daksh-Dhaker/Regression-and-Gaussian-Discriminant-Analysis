# Imports - you can add any other permitted libraries
import numpy as np
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LogisticRegressor:
    # Assume Binary Classification
    def __init__(self):
        self.theta = np.array([])
        self.normalize_mean = 0
        self.normalize_std = 1
        pass
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Newton's Method.
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
        List of Parameters: numpy array of shape (n_iter, n_features+1,)
            The list of parameters obtained after each iteration of Newton's Method.
        """
        X = self.normalize(X)
        ones_col = np.ones((X.shape[0],1))
        X = np.hstack((ones_col,X))
        m,n = X.shape
        theta_t = np.ones(n)
        theta_t_1 = np.zeros(n)
        self.theta = theta_t_1
        feature_matrix = []
        
        for _ in range(2000):
            h_theta = self.sigmoid(X @ theta_t)  
            grad = self.gradient(h_theta, X, y)
            H = self.hessian(h_theta, X)
            try:
                theta_t_1 = theta_t + (learning_rate * (np.linalg.inv(H) @ grad))
            except:
                H = H + ((1e-10) * np.eye(H.shape[0]))
                theta_t_1 = theta_t + (learning_rate * (np.linalg.inv(H) @ grad))
            feature_matrix.append(theta_t_1)
            self.theta = theta_t_1
            
            if (self.convergence(theta_t_1, theta_t)):
                break
            theta_t = theta_t_1
        
        return np.array(feature_matrix)
    
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
        X = (X-self.normalize_mean)/self.normalize_std
        ones_col = np.ones((X.shape[0],1))
        X = np.hstack((ones_col,X))
        m,n = X.shape
        h_theta = self.sigmoid(X @ self.theta)
        y = np.zeros(m)
        for i in range(m):
            if (h_theta[i] > 0.5):
                y[i] = 1
            else:
                y[i] = 0
        return y
        
    def sigmoid(self, x):
        ans = 1/(1 + np.exp(-x))
        return ans
    
    def normalize(self, X):
        mean = np.mean(X,axis = 0)
        var = np.std(X,axis = 0)
        self.normalize_mean = mean
        self.normalize_std = var
        X = (X-mean)/var
        return X
    
    def gradient(self,h_theta,X,y):
        m,n = X.shape
        ans = np.zeros(n)
        for i in range(m):
            ans = ans + (X[i] * (y[i] - h_theta[i]))
        return ans
    
    def hessian(self, h_theta, X):
        m,n = X.shape
        ans = np.zeros((n,n))
        for i in range(m):
            X_out = np.outer(X[i],X[i])
            ans = ans + (h_theta[i] * (1-h_theta[i]) * X_out)
        return ans
    
    def convergence(self, theta_t,theta_t_1, epislon = 1e-6):
        norm = np.linalg.norm(theta_t_1 - theta_t, ord = 2)
        if norm < epislon:
            return True
        return False
           
