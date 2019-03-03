"""
May 31, 2018

    MNIST digit recognition testing.
    Hands on Machine Learning by Aurelien Geron Chapter 3.

Scharmann
"""
from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

if __name__ == "__main__":

    mnist = fetch_mldata('MNIST original')
    X, y = mnist["data"], mnist["target"]

    #some_digit_image = data.reshape(28,28)
    #plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    #plt.axis("off")
    #plt.show()
    
    # MNIST contains 70000 samples where the first 60000 are for training while
    # the remainder for testing
    X_train, y_train, X_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    for i in range(0,10):

        y_train_i = (y_train == i)
        y_test_i = (y_test == i)

        sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=0.001)
        sgd_clf.fit(X_train, y_train_i)
            
        # Number of folds
        skfolds = StratifiedKFold(n_splits=2, random_state=42)
           
        # Perform K-Fold Cross Validation 
        for train_index, test_index in skfolds.split(X_train, y_train_i):
                
            clone_clf = clone(sgd_clf)
            X_train_folds = X_train[train_index]
            y_train_folds = y_train_i[train_index]
            X_test_fold = X_train[test_index]
            y_test_fold = y_train_i[test_index]

            clone_clf.fit(X_train_folds, y_train_folds)
            y_pred = clone_clf.predict(X_test_fold)
            n_correct = sum(y_pred == y_test_fold)
            print(i, ": ", n_correct / len(y_pred))

    
