from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


def metrics(y, predictions):
    fp = np.sum(np.all([predictions == 1, y == 0], axis=0))
    tp = np.sum(np.all([predictions == 1, y == 1], axis=0))
    fn = np.sum(np.all([predictions == 0, y == 1], axis=0))

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    F1 = (2 * precision * recall) / (precision +
                                     recall) if (precision + recall) > 0 else 0
    return precision, recall, F1


def train_validation_splits(df):

    # Fraud Transactions
    fraud = df[df['Class'] == 1]
    # Normal Transactions
    normal = df[df['Class'] == 0]
    print('normal:', normal.shape[0])
    print('fraud:', fraud.shape[0])

    normal_test_start = int(normal.shape[0] * .2)
    fraud_test_start = int(fraud.shape[0] * .5)
    normal_train_start = normal_test_start * 2

    val_normal = normal[:normal_test_start]
    val_fraud = fraud[:fraud_test_start]
    validation_set = pd.concat([val_normal, val_fraud], axis=0)

    test_normal = normal[normal_test_start:normal_train_start]
    test_fraud = fraud[fraud_test_start:fraud.shape[0]]
    test_set = pd.concat([test_normal, test_fraud], axis=0)

    Xval = validation_set.iloc[:, :-1]
    Yval = validation_set.iloc[:, -1]

    Xtest = test_set.iloc[:, :-1]
    Ytest = test_set.iloc[:, -1]

    train_set = normal[normal_train_start:normal.shape[0]]
    Xtrain = train_set.iloc[:, :-1]

    return Xtrain.to_numpy(), Xtest.to_numpy(), Xval.to_numpy(), Ytest.to_numpy(), Yval.to_numpy()


def estimate_gaussian_params(X):
    """
    Calculates the gaussian parameters for each feature.

    Arguments:
    X: dataset
    """
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)

    return mu, sigma

def selectThreshold(yval, pval):
    e_values = pval
    bestF1 = 0
    bestEpsilon = 0

    for epsilon in e_values:
        predictions = pval < epsilon

        (precision, recall, F1) = metrics(yval, predictions)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
