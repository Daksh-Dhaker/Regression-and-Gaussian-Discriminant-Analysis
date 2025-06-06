# Imports - you can add any other permitted libraries
import numpy as np
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

class LinearRegressor:
    def __init__(self):
        self.theta = []
        self.J_theta_values =[]
        pass
    
    def fit(self, X, y, learning_rate=0.01):
        """
        Fit the linear regression model to the data using Gradient Descent.
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input data.
            
        y : numpy array of shape (n_samples,)
            The target values.

        learning_rate : float
            The learning rate to use in the update rule.
            
        Returns
        -------
        List of Parameters: numpy array of shape (n_iter, n_features,)
            The list of parameters obtained after each iteration of Gradient Descent.
        """
        ones_col = np.ones((X.shape[0],1))
        X = np.hstack((ones_col,X))
        m, n = X.shape
        theta_t = np.zeros(n)
        theta_t_1 = np.zeros(n)
        self.theta = theta_t_1
        feature_matrix = []
        
        for _ in range(2000):
            theta_t_1 = theta_t - (learning_rate * self.grad(theta_t,X,y))
            feature_matrix.append(theta_t_1)
            self.J_theta_values.append(self.J_theta(theta_t_1,X,y))
            self.theta = theta_t_1
            if (self.convergence(theta_t,theta_t_1, X, y)):
                break
            theta_t = theta_t_1
        
        # print("Jtheta" ,self.J_theta_values)
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
            The predicted target values.
        """
        ones_col = np.ones((X.shape[0],1))
        X = np.hstack((ones_col,X))
        y = np.dot(X,self.theta)
        return y
    
    def grad(self,theta_t,X,y):
        m, n = X.shape
        y = y.reshape(-1)
        ans = (-1/len(X))*(X.T @ (y-(X @ theta_t)))
        return ans
    
    def convergence(self, theta_t,theta_t_1, X, y, epislon = 1e-10):
        norm = abs(self.J_theta(theta_t,X, y) - self.J_theta(theta_t_1, X, y))
        if norm < epislon:
            return True
        return False

    def J_theta(self, theta_t, X, y):
        j_theta = 0
        m, n = X.shape
        for i in range(m):
            j_theta += (y[i] - np.dot(theta_t,X[i]))**2
        j_theta = (j_theta/(2*m))
        
        return j_theta 
    
        
        