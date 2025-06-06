import numpy as np
import matplotlib.pyplot as plt
from sampling_sgd import StochasticLinearRegressor, generate
import argparse

def theoretical_theta(X,y):
    ones_col = np.ones((X.shape[0],1))
    X = np.hstack((ones_col,X))
    theta_closed = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return theta_closed

def plot_theta_trajectory(theta_array,number):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    theta0, theta1, theta2 = theta_array[:, 0], theta_array[:, 1], theta_array[:, 2]

    ax.plot3D(theta0, theta1, theta2, 'b-', marker='o',markersize=0.1)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Theta 2')
    ax.set_title('Theta Trajectory in 3D Space')

    if (number == 0):
        plt.savefig('plot_1.png')
    elif (number == 1):
        plt.savefig('plot_80.png')
    elif (number == 2):
        plt.savefig('plot_8000.png')
    else:
        plt.savefig('plot_800000.png')
    
    plt.show()
    

def train_test_split(X, y, threshold):
    size = X.shape[0]
    indices = np.arange(size)
    np.random.shuffle(indices)

    split_idx = int(threshold * size)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    return X_train, y_train, X_test, y_test

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def Loss(theta,X,y):
    ones_col = np.ones((X.shape[0],1))
    X = np.hstack((ones_col,X))
    ans = np.mean((y - (X@ theta))**2)/2
    return ans

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="command-line arguments")
    parser.add_argument("function", type=str, help="which function to call ?")
    args = parser.parse_args()
    
    l = StochasticLinearRegressor()
    learning_rate = 0.001
    if (args.function == "theoretical"):
        X, y = generate(1000000,np.array([3,1,2]),np.array([3,-1]),np.array([4,4]),2)
        print("Theoretical Theta : [",*theoretical_theta(X, y).flatten(),"]")
    elif (args.function == "plot"):
        X, y = generate(1000000,np.array([3,1,2]),np.array([3,-1]),np.array([4,4]),2)
        X_train, y_train, X_test, y_test = train_test_split(X,y,0.8)
        ans = l.fit(X_train,y_train,learning_rate)
        plot_theta_trajectory(l.plot_list[0],0)
        plot_theta_trajectory(l.plot_list[1],1)
        plot_theta_trajectory(l.plot_list[2],2)
        plot_theta_trajectory(l.plot_list[3],3)
    elif (args.function == "MSE"):
        X, y = generate(1000000,np.array([3,1,2]),np.array([3,-1]),np.array([4,4]),2)
        X_train, y_train, X_test, y_test = train_test_split(X,y,0.8)
        ans = l.fit(X_train,y_train,learning_rate)
        predictions = l.predict(X_test)
        predictions_train = l.predict(X_train)
        mse_1 = mean_squared_error(y_test, predictions[0])
        mse_train_1 = mean_squared_error(y_train, predictions_train[0])
        mse_80 = mean_squared_error(y_test, predictions[1])
        mse_train_80 = mean_squared_error(y_train, predictions_train[1])
        mse_8000 = mean_squared_error(y_test, predictions[2])
        mse_train_8000 = mean_squared_error(y_train, predictions_train[2])
        mse_800000 = mean_squared_error(y_test, predictions[3])
        mse_train_800000 = mean_squared_error(y_train, predictions_train[3])
        print("Parameter vector : ",ans[2][-1])
        print("MSE Test 1: ",mse_1)
        print("MSE Train 1: ", mse_train_1)
        print("MSE Test 80: ",mse_80)
        print("MSE Train 80: ", mse_train_80)
        print("MSE Test 8000: ",mse_8000)
        print("MSE Train 8000: ", mse_train_8000)
        print("MSE Test 800000: ",mse_800000)
        print("MSE Train 800000: ", mse_train_800000)
    elif (args.function == "parameters"):
        X, y = generate(1000000,np.array([3,1,2]),np.array([3,-1]),np.array([4,4]),2)
        X_train, y_train, X_test, y_test = train_test_split(X,y,0.8)
        ans = l.fit(X_train,y_train,learning_rate)
        parameters = [ans[0][-1], ans[1][-1], ans[2][-1], ans[3][-1]]
        print("Parameters : ")
        print(parameters)
        print("Epochs : ")
        print(len(ans[0]), len(ans[1]), len(ans[2]), len(ans[3]))
        print("Hops : ")
        print(len(l.plot_list[0]), len(l.plot_list[1]), len(l.plot_list[2]), len(l.plot_list[3]))
        print("Loss : ")
        print(Loss(ans[0][-1],X,y), Loss(ans[1][-1],X,y), Loss(ans[2][-1],X,y), Loss(ans[3][-1],X,y))
    else:
        pass 

