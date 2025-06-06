import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressor
import argparse


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

def plot(X, y, theta):
    Y = y.flatten()
    class_0 = X[Y==0]
    class_1 = X[Y==1]
    class_0_x1 = class_0[:,0]
    class_0_x2 = class_0[:,1]
    class_1_x1 = class_1[:,0]
    class_1_x2 = class_1[:,1]
    plt.scatter(class_0_x1, class_0_x2, label="Class 0", marker='o', color ='red', edgecolors='k')
    plt.scatter(class_1_x1, class_1_x2, label="Class 1", marker='x', color='blue')
    
    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2_vals = - (theta[0] + (theta[1] * x1_vals)) / theta[2]
    eq_line = f"{theta[0]:.2f} + {theta[1]:.2f}x₁ + {theta[2]:.2f}x₂ = 0"
    
    plt.plot(x1_vals, x2_vals, color='green', label=f"Decision Boundary\n({eq_line})")
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.legend()
    plt.title("Logistic Regression Decision Boundary")
    plt.savefig('plot.png')
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="command-line arguments")
    try:
        parser.add_argument("function", type=str, help="Name of Function")
        parser.add_argument("data", type=str, help="Path to data.csv file")
        parser.add_argument("label", type=str, help="Path to labels.csv file")
        
        args = parser.parse_args()
    
        X = pd.read_csv(args.data, header=None)
        y = pd.read_csv(args.label, header=None)
        X = X.to_numpy()
        y = y.to_numpy()

        l = LogisticRegressor()
        ans = l.fit(X,y)
        
        if (args.function == "plot"):
            print("parameter vector : ", ans[-1])
            X = l.normalize(X)
            plot(X, y, ans[-1])
        elif (args.function == "parameters"):
            print("parameter vector : ", ans[-1])
            print("Hops : ",len(ans))
        elif (args.function == "accuracy"):
            print("parameter vector : ", ans[-1])
            predictions = l.predict(X)
            accuracy = get_accuracy(predictions,y)
            print("Accuracy : ",accuracy)      
        else:
            pass
        
    except Exception as e:
        print(e)
