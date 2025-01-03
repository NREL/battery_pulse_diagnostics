"""
Main script to run XGBoost model using repeated random train/test sampling, referred
to here as "bootstrapping".
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
import torch
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

from utils.data_utils import load_data, filter_relevant_extracted_features
from utils.modeling_utils import is_classification_target

import warnings

warnings.filterwarnings("ignore")


pd.options.mode.chained_assignment = None  # default='warn'

RANDOM_STATE = 42


if __name__ == "__main__":

    ################# User Parameters ####################
    filename_raw_features = "data/data_for_ml.h5"
    filename_extracted_features = "data/data_for_ml_extracted_features.h5"
    test_size = 0.20

    # No. bootstrap iterations
    n_splits = 50

    # How often to print output
    print_freq = 2
    
    # If True, fit on [OCV, T] only. Otherwise fit on all features.
    baseline = False

    # Whether to fit on raw or extracted features
    raw_features = True

    # Whether to return the probability or predicted class
    # when performing classification
    return_classification_probabilities = False
    #######################################################

    pulses = [
        "Static Rapid",
        "Static HPPC",
        "Static PsRP 1",
        "Static PsRP 2 Chg",
        "Static PsRP 2 Dis",
        "Dynamic PsRP 1 C/2",
        "Dynamic PsRP 2 1C",
        "Dynamic PsRP 1 1C",
        "Dynamic PsRP 2 C/2",
        "DCIR",
    ]

    targets = [
        "1C discharge capacity",
        "C/10 discharge capacity",
        "C/5 discharge capacity",
        "C/3 discharge capacity",
        "C/2 discharge capacity",
        "P/3 discharge capacity",
        "Charge depleting cycle charge throughput",
        "Charge sustaining cycle charge efficiency",
        "soc",
        "Post 1C charge relaxation fit MSE",
        "1C discharge capacity_3bins",
        "Post 1C charge relaxation fit MSE_outlier",
        "C/3 discharge capacity_3bins",
        "Post C/2 charge relaxation fit MSE_outlier",
        "Post 1C or Post C/2_outlier",
        "Post 1C charge relaxation fit Max Error",
        "Post 1C charge relaxation fit Max Error_outlier",
        "Cell volume",
        "Electrode stack thickness",
        "Excess electrolyte",
        "Volume growth",
        "Thickness growth",
    ]

    for cell_type in ["A", "B", "C", "D"]:

        # Save results as a dictionary
        bootstrap_results = dict()

        tests = load_data(filename=filename_raw_features, cell_type=cell_type)

        for target in targets:

            for pulse in pulses:
                print(pulse, ", ", target)

                r2s = []
                maes = []
                df = tests[pulse]
                df = df[df[target].notna()]

                # Keep only the identifier columns: cell_id, measurement_id,
                # temp, rate, etc. This is a little clunky but since the
                # identifier columns are slightly different per type of pulse,
                # set up the results df backwards by removing the common
                # unwanted columns.
                unwanted_cols = [
                    "Post C/10 charge relaxation fit MSE",
                    "Post C/5 charge relaxation fit MSE",
                    "Post C/3 charge relaxation fit MSE",
                    "Post P/3 charge relaxation fit MSE",
                    "Post C/2 charge relaxation fit MSE",
                    "Unnamed: 0",
                ]
                results = df.loc[:, list(~df.columns.isin(targets + unwanted_cols))]
                results = results.filter(
                    regex=r"^(?!{}*)".format("voltage|polarization|current")
                )
                results[target] = df[target]

                # Keep measurement IDs together when splitting
                splitter = GroupShuffleSplit(
                    n_splits=n_splits, test_size=test_size, random_state=RANDOM_STATE
                )
                features_selected = []
                for i, (idx_train, idx_test) in enumerate(
                    splitter.split(df, groups=df["measurement_id"])
                ):
                    if i % print_freq == 0:
                        print("\tBootstrap iteration ", i)
                    train, test = df.iloc[idx_train], df.iloc[idx_test]

                    if not raw_features:
                        # Fitting on TSFresh extracted features. Filter only the
                        # the relevant extracted features based on the train set.
                        if target != "Post 1C charge relaxation fit MSE_outlier":
                            train, test = filter_relevant_extracted_features(
                                train, test, target
                            )
                            features_selected.append(train.columns)

                    # Add instantaneous resistance feature
                    for p in train.filter(regex="polarization"):
                        time = p.split("_")[-1]
                        train[f"P/I_{time}"] = train[p] / train[f"current_{time}"]
                        test[f"P/I_{time}"] = test[p] / test[f"current_{time}"]
                    train = train.replace([np.inf, -np.inf], 0)
                    test = test.replace([np.inf, -np.inf], 0)
                    train = train.replace(np.nan, 0)
                    test = test.replace(np.nan, 0)

                    # Define input features
                    if baseline is True:
                        if pulse.startswith('Static'):
                            train_filter = train.filter(regex="voltage_0.0s|temperature")
                            test_filter = test.filter(regex="voltage_0.0s|temperature")
                        elif pulse.startswith('Dynamic'):
                            train["current_mean"] = train.filter(regex="current").mean(axis=1)
                            test["current_mean"] = test.filter(regex="current").mean(axis=1)
                            max_s = len(train.filter(regex='voltage').columns) - 1
                            train['voltagef-voltagei'] = train[f"voltage_{max_s}.0s"] - train[f"voltage_0.0s"]
                            test['voltagef-voltagei'] = test[f"voltage_{max_s}.0s"] - test[f"voltage_0.0s"]
                            train_filter = train.filter(regex=f"voltagef-voltagei|voltage_0.0s|voltage_{max_s}.0s|current_mean|temperature")
                            test_filter = test.filter(regex=f"voltagef-voltagei|voltage_0.0s|voltage_{max_s}.0s|current_mean|temperature")
                    else:
                        if pulse == 'DCIR':
                            regex="voltage|0p1s|1s|4s|10s|V0|temperature_ambient_f"
                        else:
                            regex="voltage|P/I|polarization|current|temperature"

                    # Fit and predict
                    if is_classification_target(target):

                        sample_weights = class_weight.compute_sample_weight(
                            "balanced", y=train[target]
                        )

                        if target.endswith("outlier"):
                            count_0 = (train[target] == 0).sum()
                            count_1 = (train[target] == 1).sum()
                            scale_pos_weight = count_0 / count_1
                        else:
                            scale_pos_weight = 1

                        le = LabelEncoder()
                        y_train = le.fit_transform(train[target])
                        xgb = XGBClassifier(scale_pos_weight=scale_pos_weight).fit(
                            train.filter(regex=regex),
                            y_train,
                            sample_weight=sample_weights,
                        )
                        preds = xgb.predict(test.filter(regex=regex))
                        if return_classification_probabilities:
                            preds = prob_class1 = xgb.predict_proba(
                                test.filter(regex=regex)
                            )[:, 1]
                        else:
                            preds = le.inverse_transform(preds)

                    else:
                        xgb = XGBRegressor().fit(
                            train.filter(regex=regex),
                            train[target],
                        )
                        preds = xgb.predict(test.filter(regex=regex))

                    # Save predicted value per cell
                    test_cells = test[["sample_id"]]
                    test_cells.loc[:, i] = preds
                    results = results.merge(test_cells, how="left", on="sample_id")

                    r2s.append(r2_score(test[target], preds))
                    maes.append(mean_absolute_error(test[target], preds))

                print(
                    "\tr2 score: mean = %0.4f, std = %0.4f"
                    % (np.mean(r2s), np.std(r2s))
                )
                print(
                    "\tmae score: mean = %0.4f, std = %0.4f"
                    % (np.mean(maes), np.std(maes))
                )

                bootstrap_results[f"{target}, {pulse}"] = results
                bootstrap_results[
                    f"{target}, {pulse}, features_per_iteration"
                ] = features_selected

                torch.save(
                    bootstrap_results,
                    f"results/bootstrap_results_{cell_type}.pth",
                )
