import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import re
import torch


def is_classification_target(target: str):
    return target in {
        "Excess electrolyte",
        "Post 1C charge relaxation fit MSE_outlier",
        "Post C/2 charge relaxation fit MSE_outlier",
        "Post 1C or Post C/2_outlier",
        "1C discharge capacity_3bins",
        "C/3 discharge capacity_3bins",
    }


def get_optimal_n_components_minimum(means: np.ndarray):
    """

    Given an array of means the index (number of components) that has the
    minimum error.

    Args:
        means (np.ndarray)    Shape (n_components, ),
                                mean error across n bootstrap splits
    """
    n_components = np.argmin(means) + 1
    threshold = np.min(means)
    return n_components, threshold


def minimum_test(
    X: pd.DataFrame,
    y: str,
    metadata: pd.DataFrame,
    max_n_components: int = 30,
    n_splits: int = 5,
    printout: bool = False,
):
    """

    Given a dataset and target, use KFold CV to determine
    the optimal number of PLSR components.

    Args:
        X (pd.DataFrame)           Features to train on
        y (str)                    True output values
        metadata (pd.DataFrame)    Groups for K-Fold CV
        max_n_components (int)     Maximum number of PLSR components to test for 1-SD test
        n_splits (int)             Number of K-Fold splits to use
        printout (bool)            Whether to print iteration updates during training
                                   and plot mean/std of error per number of components.
    """

    means = []
    stds = []

    # For each component option
    for n_components in range(1, max_n_components + 1):

        if printout:
            if n_components % 10 == 0:
                print("n_components = ", n_components)

        splitter = GroupKFold(n_splits=n_splits)

        # Get metrics for 5 CV splits
        maes = []
        for i, (idx_train, idx_test) in enumerate(
            splitter.split(X, groups=metadata["measurement_id"])
        ):
            # print("\tBootstrap iteration ", i)
            X_train, X_test = X[idx_train], X[idx_test]
            y_train, y_test = y[idx_train].numpy(), y[idx_test].numpy()

            X_train, X_test = (
                torch.tensor(X_train).float(),
                torch.tensor(X_test).float(),
            )
            y_train, y_test = (
                torch.tensor(y_train).float(),
                torch.tensor(y_test).float(),
            )
            X_train = torch.nn.Flatten()(X_train).numpy()
            X_test = torch.nn.Flatten()(X_test).numpy()

            # Fit model
            pls = PLSRegression(n_components=n_components, scale=True).fit(
                X_train, y_train.numpy()
            )
            yhat_test = pls.predict(X_test)

            # Save score
            mae_score = mean_absolute_error(y_test, yhat_test)
            maes.append(mae_score)

        means.append(np.mean(maes))
        stds.append(np.std(maes))

    # Return the number of components with minimum error (too noisy to do 1SD test)
    n_components, threshold = get_optimal_n_components_minimum(means)

    if printout:

        print("n_components: ", n_components)

        plt.figure()
        plt.errorbar(list(range(1, max_n_components + 1)), means, yerr=stds)
        plt.hlines(
            xmin=1,
            xmax=max_n_components + 1,
            y=threshold,
            linestyles="dashed",
            colors="red",
        )
        plt.show()

    return n_components
