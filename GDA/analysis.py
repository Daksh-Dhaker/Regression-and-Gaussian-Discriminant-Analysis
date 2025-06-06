import numpy as np
import matplotlib.pyplot as plt
from gda import GaussianDiscriminantAnalysis
import argparse

def equation(x1,x2, sigma1, sigma0, mu1, mu0, C):
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    X = np.array([x1, x2])
    term1 = ((X.T @ (sigma1_inv - sigma0_inv)) @ X)
    term2 = 2*(X.T @ ((sigma0_inv @ mu0) - (sigma1_inv @ mu1)))
    term3 = ((mu1.T @ sigma1_inv) @ mu1) - ((mu0.T @ sigma0_inv) @ mu0)
    return (C-(0.5 * (term1 + term2 + term3)))

def plot_points(X,y):
    class_0 = X[y==0]
    class_1 = X[y==1]
    class_0_x1 = class_0[:,0]
    class_0_x2 = class_0[:,1]
    class_1_x1 = class_1[:,0]
    class_1_x2 = class_1[:,1]
    plt.scatter(class_0_x1, class_0_x2, label="Class 0", marker='o', color ='red', edgecolors='k')
    plt.scatter(class_1_x1, class_1_x2, label="Class 1", marker='x', color='blue')
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.legend()
    plt.title("Training Data")
    plt.savefig("points.png")
    plt.show()

def plot(X,y,mu0,mu1,sigma0,sigma1,same_cov = False):
    class_0 = X[y==0]
    class_1 = X[y==1]
    class_0_x1 = class_0[:,0]
    class_0_x2 = class_0[:,1]
    class_1_x1 = class_1[:,0]
    class_1_x2 = class_1[:,1]
    plt.scatter(class_0_x1, class_0_x2, label="Class 0", marker='o', color ='red', edgecolors='k')
    plt.scatter(class_1_x1, class_1_x2, label="Class 1", marker='x', color='blue')
    
    phi  = np.sum(y == 1) / len(y)
    det0 = np.linalg.det(sigma0)
    det1 = np.linalg.det(sigma1)
    try: 
        C = np.log(phi/(1-phi)) + (0.5 * np.log(det0/det1))
    except:
        C = np.log(phi/(1-phi))
    
    x1_vals = np.linspace((X[:, 0].min())-1, (X[:, 0].max())+1, 400)  
    x2_vals = np.linspace((X[:, 1].min())-1, (X[:, 1].max())+1, 400)  
    
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros_like(X1)
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            Z[j, i] = equation(X1[j, i], X2[j, i],sigma1, sigma0, mu1, mu0, C) 
    contour = plt.contour(X1, X2, Z, levels=[0], colors='blue')
    if (same_cov):
        plt.plot([], [], color='blue', label="Linear Decision Boundary")
        plt.title("Linear Decision Boundary")
    else:
        plt.plot([], [], color='blue', label="Quadratic Decision Boundary")
        plt.title("Quadratic Decision Boundary")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    if (same_cov):
        plt.savefig("plot_linear.png")
    else:
        plt.savefig("plot_quad.png")
    plt.show()
    
def get_accuracy(predictions, y_test):
    ans = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(predictions)):
        if (predictions[i] == y_test[i]):
            ans += 1
            if (predictions[i] == 1):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if (predictions[i] == 1):
                false_pos += 1
            else:
                false_neg += 1
                
    ans = ans/len(predictions)
    print("True Positives : ", true_pos)
    print("False Positives : ", false_pos)
    print("True Negatives : ", true_neg)
    print("False Negatives : ", false_neg)
    return ans

def plot_both(X,y,mu0,mu1,sigma,sigma0,sigma1):
    phi  = np.sum(y == 1) / len(y)
    C2 = np.log(phi/(1-phi))
    theta = np.linalg.inv(sigma) @ (mu1-mu0)
    theta0 = C2-(0.5 * (((mu1.T @ np.linalg.inv(sigma)) @ mu1) - ((mu0.T @ np.linalg.inv(sigma)) @ mu0))) 
    theta = np.concatenate(([theta0], theta))
    y = y.flatten()
    x1_vals_1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_vals_1 = - (theta[0] + (theta[1] * x1_vals_1)) / theta[2]
    
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    class_0_x1 = class_0[:, 0]
    class_0_x2 = class_0[:, 1]
    class_1_x1 = class_1[:, 0]
    class_1_x2 = class_1[:, 1]
    plt.scatter(class_0_x1, class_0_x2, label="Class 0", marker='o', color='red', edgecolors='k')
    plt.scatter(class_1_x1, class_1_x2, label="Class 1", marker='x', color='blue')
    
    C = 0
    det0 = np.linalg.det(sigma0)
    det1 = np.linalg.det(sigma1)
    try: 
        C = np.log(phi/(1-phi)) + (0.5 * np.log(det0/det1))
    except:
        C = np.log(phi/(1-phi))
    x1_vals = np.linspace(-3, 3, 400)  
    x2_vals = np.linspace(-3, 3, 400)  
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = np.zeros_like(X1)
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            Z[j, i] = equation(X1[j, i], X2[j, i],sigma1, sigma0, mu1, mu0, C) 
    contour = plt.contour(X1, X2, Z, levels=[0], colors='blue')
    
    plt.plot(x1_vals_1, x2_vals_1, color='green', label="Linear Decision Boundary")
    plt.plot([], [], color='blue', label="Quadratic Decision Boundary")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.savefig("plot_both.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command-line arguments")
    try:
        parser.add_argument("function", type=str, help="Name of Function")
        parser.add_argument("data", type=str, help="Path to data.csv file")
        parser.add_argument("label", type=str, help="Path to labels.csv file")
        args = parser.parse_args()
    
        X = np.loadtxt(args.data, delimiter=None)
        y = np.loadtxt(args.label, dtype=str)
        y = np.where(y == 'Alaska', 0, 1)

        l = GaussianDiscriminantAnalysis()
        
        if (args.function == "same_cov"):
            ans = l.fit(X,y,True)
            print("mu0 : ",ans[0])
            print("mu1 : ",ans[1])
            print("sigma : ")
            print(ans[2])
        elif (args.function == "plot_points"):
            X = l.normalize(X)
            plot_points(X,y)
        elif (args.function == "plot_linear"):
            ans = l.fit(X,y,True)
            X = l.normalize(X)
            plot(X,y,ans[0],ans[1],ans[2],ans[2],True)
        elif (args.function == "diff_cov"):
            ans = l.fit(X,y,False)
            print("mu0 : ",ans[0])
            print("mu1 : ",ans[1])
            print("sigma0 : ")
            print(ans[2])
            print("sigma1 : ")
            print(ans[3])
        elif (args.function == "plot_quad"):
            ans = l.fit(X,y,False)
            X = l.normalize(X)
            plot(X,y,ans[0],ans[1],ans[2],ans[3],False)
        elif (args.function == "plot_both"):
            ans1 = l.fit(X,y,True)
            ans2 = l.fit(X,y,False)
            X = l.normalize(X)
            plot_both(X,y,ans1[0],ans1[1],ans1[2],ans2[2],ans2[3])
        elif (args.function == "accuracy"):
            print("same cov : ")
            ans1 = l.fit(X,y,True)
            pred = l.predict(X)
            print("Accuracy : ",get_accuracy(pred,y))
            print("diff cov : ")
            ans2 = l.fit(X,y,False)
            pred = l.predict(X)
            print("Accuracy : ",get_accuracy(pred,y)) 
        else:
            pass
        
    except Exception as e:
        print(e)



