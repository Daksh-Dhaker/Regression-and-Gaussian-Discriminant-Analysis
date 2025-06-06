import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegressor
import argparse

def plot(m,c,linear_X,linear_Y):
    x_values = np.linspace(np.min(linear_X), np.max(linear_X), 100)  
    y_values = m * x_values + c  
    plt.scatter(linear_X, linear_Y, color='blue', s=20, label='Data Points')
    plt.plot(x_values, y_values, color='orange', linewidth=2, label=f'y = {m:.2f}x + {c:.2f}')
    plt.xlabel("X values")
    plt.ylabel("Y values")
    plt.title("Scatter Plot with Regression Line")
    plt.legend()
    plt.savefig('q2.png')
    plt.show()

def three_D_plot_save(data, label):
    linear_XX = pd.read_csv(data, header=None).values.flatten()
    linear_YY = pd.read_csv(label, header=None).values.flatten()
    theta0_min, theta1_min = 6, 29
    theta0_range = np.linspace(0, theta0_min+1, 100)
    theta1_range = np.linspace(0, theta1_min+1, 100)
    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    J_values = np.zeros_like(T0)

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta0 = T0[i, j]
            theta1 = T1[i, j]
            J_values[i, j] = np.sum((linear_YY - (theta1 * linear_XX + theta0)) ** 2)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T0, T1, J_values, cmap='coolwarm', alpha=0.9)
    ax.contourf(T0, T1, J_values, zdir='z', offset=np.min(J_values) - 100, cmap='coolwarm')
    ax.set_xlabel(r'$\theta_0$ (Bias)')
    ax.set_ylabel(r'$\theta_1$ (Slope)')
    ax.set_zlabel(r'$J(\theta)$ (Error Function)')
    ax.set_title('3D Half-Bowl of Error Function)')
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('J_theta.png')
    plt.show()

def three_D_plot_show(data, label, theta_values):
    linear_XX = pd.read_csv(data, header=None).values.flatten()
    linear_YY = pd.read_csv(label, header=None).values.flatten()
    theta0_min, theta1_min = 6, 29
    theta0_range = np.linspace(0, theta0_min+1, 100)
    theta1_range = np.linspace(0, theta1_min+1, 100)

    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    J_values = np.zeros_like(T0)

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta0 = T0[i, j]
            theta1 = T1[i, j]
            J_values[i, j] = np.sum((linear_YY - (theta1 * linear_XX + theta0)) ** 2)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T0, T1, J_values, cmap='coolwarm', alpha=0.9)
    ax.set_xlabel(r'$\theta_0$ (Bias)')
    ax.set_ylabel(r'$\theta_1$ (Slope)')
    ax.set_zlabel(r'$J(\theta)$ (Error Function)')
    ax.set_title('3D Half-Bowl of Error Function)')

    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    J_values_added_points = []
    for (theta0, theta1) in theta_values:
        if not plt.fignum_exists(fig.number):  
            break
        J_theta = np.sum((linear_YY - (theta1 * linear_XX + theta0)) ** 2)
        J_values_added_points.append(J_theta)
        ax.scatter(theta0, theta1, J_theta, color='black', s=30)
        plt.pause(0.2) 
        
    plt.show()

    
def contour_plot(data, label, theta_values):
    linear_XX = pd.read_csv(data, header=None).values.flatten() 
    linear_YY = pd.read_csv(label, header=None).values.flatten() 
    
    theta1_optimal = np.sum(linear_XX * linear_YY) / np.sum(linear_XX**2) 
    theta0_optimal = np.mean(linear_YY) - theta1_optimal * np.mean(linear_XX)  

    theta0_range = np.linspace(theta0_optimal - 50, theta0_optimal + 50, 100)  
    theta1_range = np.linspace(theta1_optimal - 50, theta1_optimal + 50, 100)  

    T0, T1 = np.meshgrid(theta0_range, theta1_range)
    J_values = np.zeros_like(T0)

    for i in range(T0.shape[0]):
        for j in range(T0.shape[1]):
            theta0 = T0[i, j]
            theta1 = T1[i, j]
            J_values[i, j] = np.sum((linear_YY - (theta1 * linear_XX + theta0)) ** 2) 

    fig = plt.figure(figsize=(10, 7))
    plt.contourf(T0, T1, J_values, levels=50, cmap='coolwarm')
    plt.colorbar(label=r'$J(	heta)$ (Error Function)')
    plt.xlabel(r'$\theta_0$ (Bias)')
    plt.ylabel(r'$\theta_1$ (Slope)')
    plt.title('Contour Plot of Error Function with Gradient Descent Path')

    for (theta0, theta1) in theta_values:
        if not plt.fignum_exists(fig.number):  
            break  
        plt.scatter(theta0, theta1, color='red', s=5)
        plt.pause(0.2) 

    plt.show()

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)   
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Command Line Arguments")
    try:
        parser.add_argument("function", type=str, help="Name of Function")
        parser.add_argument("data", type=str, help="Path to data.csv file")
        parser.add_argument("label", type=str, help="Path to labels.csv file")
        
        args = parser.parse_args()
    
        X = pd.read_csv(args.data, header=None)
        y = pd.read_csv(args.label, header=None)
        X = X.to_numpy()
        y = y.to_numpy()

        l = LinearRegressor()
        learning_rate = 0.1
        ans = l.fit(X,y,learning_rate)
        print("parameter vector : ", ans[-1])
        print("Hops : ", len(ans))
        if (args.function == "parameters"):
            ones_col = np.ones((X.shape[0],1))
            X = np.hstack((ones_col,X))
            print("Loss : ", l.J_theta(ans[-1],X,y))
        elif (args.function == "plot"):
            plot(ans[-1][1],ans[-1][0],X, y)
        elif (args.function == "show"):
            three_D_plot_show(args.data,args.label,ans)
        elif (args.function == "save"):
            three_D_plot_save(args.data, args.label)
        elif (args.function == "contour"):
            contour_plot(args.data,args.label,ans)
        elif (args.function == "MSE"):
            predictions = l.predict(X)
            mse = mean_squared_error(y, predictions)
            print("Parameter vector : ",ans[-1])
            print("MSE : ",mse)
        else:
            pass
        
    except Exception as e:
        print(e)
        print("Usage : python3 analysis.py <Function> <path_to_data> <path_to_labels>.")
      
        
        