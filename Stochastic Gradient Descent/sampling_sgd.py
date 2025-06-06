# Imports - you can add any other permitted libraries
import numpy as np
# You may add any other functions to make your code more modular. However,
# do not change the function signatures (name and arguments) of the given functions,
# as these functions will be called by the autograder.

def generate(N, theta, input_mean, input_sigma, noise_sigma):
    """
    Generate normally distributed input data and target values
    Note that we have 2 input features
    Parameters
    ----------
    N : int
        The number of samples to generate.
        
    theta : numpy array of shape (3,)
        The true parameters of the linear regression model.
        
    input_mean : numpy array of shape (2,)
        The mean of the input data.
        
    input_sigma : numpy array of shape (2,)
        The standard deviation of the input data.
        
    noise_sigma : float
        The standard deviation of the Gaussian noise.
        
    Returns
    -------
    X : numpy array of shape (N, 2)
        The input data.
        
    y : numpy array of shape (N,)
        The target values.
    """
    x1 = np.random.normal(input_mean[0], np.sqrt(input_sigma[0]), N)
    x2 = np.random.normal(input_mean[1], np.sqrt(input_sigma[1]), N)
    X = np.vstack([np.ones(N), x1, x2]).T
    y = X @ theta + np.random.normal(0, np.sqrt(noise_sigma), N)
    X = np.column_stack((x1, x2))
    return X, y

class StochasticLinearRegressor:
    def __init__(self):
        self.theta = None
        self.batch_size = 8000
        self.learning_rate = 0.001
        self.max_epochs = 4000
        self.tol = 1e-5
        self.theta_list = []
        self.plot_list = []
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
        
        n_samples, n_features = X.shape
        self.learning_rate = learning_rate
        batch_sizes = [1,80,8000,800000]
        max_epochs = [3,100,350,12000]
        tolerance = [1e-3,1e-4,1e-4,1e-7]
        iterations = [3,5,5,5]
        result = []
        
        for batch_num in range(4):
            self.batch_size = batch_sizes[batch_num]
            self.max_epochs = max_epochs[batch_num]
            self.tol = tolerance[batch_num]
            self.theta = np.zeros(n_features)
            loss_history = []
            itr = 0
            plot_history = []
            
            convergence_acheived = False
            
            features_matrix = []
            for epoch in range(self.max_epochs):
                
                indices = np.random.permutation(n_samples)
                X_shuffled, y_shuffled = X[indices], y[indices]

                for start_idx in range(0, n_samples, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, n_samples)
                    X_batch, y_batch = X_shuffled[start_idx:end_idx], y_shuffled[start_idx:end_idx]

                    gradient = self.grad(X_batch,y_batch)
                    self.theta = self.theta - (self.learning_rate * gradient)
                    plot_history.append(self.theta)
                    
                    loss = self.J_theta(X_batch,y_batch)
                    loss_history.append(loss)
                    

                    if len(loss_history) > 10:
                        rolling_mean_diff = abs(np.mean(loss_history[-10:]) - np.mean(loss_history[-11:-1]))
                        
                        if rolling_mean_diff < self.tol:
                            itr += 1
                            if (itr >= iterations[batch_num]):
                                convergence_acheived = True
                        else:
                            itr = 0
                    
                    if convergence_acheived:
                        break
                
                features_matrix.append(self.theta) 
                if convergence_acheived:
                    break
            
            self.plot_list.append(np.array(plot_history))
            print(self.theta)
            result.append(np.array(features_matrix))
        
        self.theta_list = [result[0][-1], result[1][-1], result[2][-1], result[3][-1]]
        print(result[0][-1], result[1][-1], result[2][-1], result[3][-1])
        # print(len(result))
        return result
    
    def grad(self,X,y):
        ans = (-1 / len(X))*(X.T @(y - (X@self.theta)))
        return ans
    
    def J_theta(self,X,y):
        ans = np.mean((y - (X@ self.theta))**2)/2
        return ans
    
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
        final_ans = []
        for i in range (4):
            y = np.dot(X,self.theta_list[i])
            final_ans.append(y)
        
        return final_ans
    
    
