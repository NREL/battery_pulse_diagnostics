"""
Main script to compare models using KFold Cross-Validation.
"""

import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from xgboost import XGBRegressor, XGBClassifier

from utils.data_utils import load_data, format_data_for_nn, TARGETS
from utils.modeling_utils import minimum_test, is_classification_target
from utils.nn_models import (
    NeuralNetwork,
    BiLSTM,
    CNNBiLSTM,
    NeuralNetworkClassifier,
    BiLSTMClassifier,
    CNNBiLSTMClassifier,
)

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


RANDOM_STATE = 42


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred.squeeze(), y.squeeze()).item()

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss


def evaluate(model, X_test, y_test, ax=None, classifier=False, title="", label=""):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    y_hat = model(X_test)

    y_test = y_test.detach().numpy()
    y_hat = y_hat.flatten().detach().numpy()

    if classifier:
        p = precision_score(y_test, y_hat.round())
        r = recall_score(y_test, y_hat.round())
        f1 = f1_score(y_test, y_hat.round())
        score_str = f"precision={p}\nrecall={r}\nf1={f1}"
    else:
        score_str = f"r2 = {r2_score(y_test, y_hat)}\nmae = {mean_absolute_error(y_test, y_hat)}"

    if ax is not None:
        ax.scatter(x=y_test, y=y_hat, label=label)
        ax.axline((y_test.min(), y_test.min()), (y_test.max(), y_test.max()))
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.legend()
        ax.set_title(f"{title}\n{score_str}")
    else:
        print(score_str)
        return y_hat


def plot_lines(arr):
    col = 0
    plt.figure()
    for line in arr:
        plt.plot(line[:, col])
    plt.show()


def train(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    test_dataloader,
    lr_scheduler=None,
    n_epochs=100,
):
    train_losses = []
    val_losses = []
    for t in range(n_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)

        size = len(train_dataloader.dataset)
        batch_size = train_dataloader.batch_size
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):

            # Zero the gradients for each batch
            optimizer.zero_grad()

            # Compute prediction for this batch
            pred = model(X)

            # Compute loss and its gradients
            train_loss = loss_fn(pred.squeeze(), y)
            train_loss.backward()

            # Adjust weights
            optimizer.step()

            # Report metrics
            train_loss, current = train_loss.item(), batch * batch_size + len(X)
        print(f"loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")

        #####
        test_loss = test_loop(test_dataloader, model, loss_fn)

        train_losses.append(train_loss)
        val_losses.append(test_loss)

        if lr_scheduler is not None:
            lr_scheduler.step()

    print("Done!")
    return train_losses, val_losses


def plot_loss_curve(n_epochs, train_losses, val_losses, ax, title):
    ax.scatter(
        x=list(range(1, n_epochs + 1)), y=train_losses, marker="o", label="train"
    )
    ax.scatter(x=list(range(1, n_epochs + 1)), y=val_losses, marker="o", label="test")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.set_title(title)


def reset_random_seeds():
    os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    random.seed(RANDOM_STATE)


if __name__ == "__main__":

    reset_random_seeds()

    ############### User-defined parameters ##################

    # Data parameters
    filename = "data/data_for_ml.h5"
    cell_type = "A"  # {None, "A", "B", "C", "D"}
    n_splits = 5

    # Training parameters
    n_epochs = 100
    learning_rate = 1e-3

    # Output parameters
    plot = False

    # 1. Load and split data
    data = load_data(filename, cell_type=cell_type)
    data = format_data_for_nn(data, TARGETS)

    for feature in ["Static HPPC", "Static PsRP 1", "Dynamic PsRP 1 C/2"]:
        for target in [
            "1C discharge capacity",
            "C/3 discharge capacity",
            "Charge depleting cycle charge throughput",
            "Charge sustaining cycle charge efficiency",
            "soc",
            "Post 1C charge relaxation fit MSE_outlier",
        ]:

            print(f"Predicting {target} using {feature}...")

            if not is_classification_target(target):
                loss_fn = nn.MSELoss()
            else:
                loss_fn = nn.BCELoss()

            X = data[feature]["X"]
            y = data[feature]["y"][target]
            metadata = data[feature]["metadata"]

            # Shuffle the data
            np.random.seed(42)
            idx = np.random.permutation(y.index)
            X = X[idx]
            y = y.reindex(idx).reset_index(drop=True)
            metadata = metadata.reindex(idx).reset_index(drop=True)

            splitter = GroupKFold(n_splits=n_splits)

            results_per_sample = pd.DataFrame(
                {"sample_id": metadata["sample_id"], target: y}
            )

            # Loop over each split
            for i, (idx_train, idx_test) in enumerate(
                splitter.split(X, groups=metadata["measurement_id"])
            ):
                print("\tKFold iteration ", i)
                X_train, X_test = X[idx_train], X[idx_test]
                y_train, y_test = y[idx_train].to_numpy(), y[idx_test].to_numpy()

                X_train, X_test = (
                    torch.tensor(X_train).float(),
                    torch.tensor(X_test).float(),
                )
                y_train, y_test = (
                    torch.tensor(y_train).float(),
                    torch.tensor(y_test).float(),
                )

                metadata_train = metadata.loc[idx_train]

                train_dataloader = DataLoader(
                    TensorDataset(X_train, y_train),
                    batch_size=8,
                    shuffle=True,
                )
                test_dataloader = DataLoader(
                    TensorDataset(X_test, y_test),
                    batch_size=8,
                    shuffle=True,
                )

                # Number of samples, number of Time steps, number of Features
                N, T, F = X_train.shape

                # Fit baseline non-neural network models: XGBoost and PLSR
                X_train_for_xgb = nn.Flatten()(X_train).numpy()
                X_test_for_xgb = nn.Flatten()(X_test).numpy()
                if is_classification_target(target):
                    xgb = XGBClassifier().fit(X_train_for_xgb, y_train.numpy())
                else:
                    xgb = XGBRegressor().fit(X_train_for_xgb, y_train.numpy())
                y_hat_xgb = xgb.predict(X_test_for_xgb)
                y_hat_xgb_train = xgb.predict(X_train_for_xgb)

                if not is_classification_target(target):
                    X_train_for_pls = nn.Flatten()(X_train).numpy()
                    X_test_for_pls = nn.Flatten()(X_test).numpy()
                    n_components = minimum_test(
                        X_train_for_pls,
                        y_train,
                        metadata_train,
                        max_n_components=20,
                        n_splits=5,
                    )
                    print("best n_components = ", n_components)
                    pls = PLSRegression(n_components=n_components, scale=True).fit(
                        X_train_for_pls, y_train.numpy()
                    )
                    y_hat_pls = pls.predict(X_test_for_pls)
                    y_hat_pls_train = pls.predict(X_train_for_pls)
                else:
                    y_hat_pls = np.empty(y_test.shape)

                # 2. Fit model
                if is_classification_target(target):
                    model_nn = NeuralNetworkClassifier(in_features=T * F)
                    model_bilstm = BiLSTMClassifier(
                        n_features=F,
                        n_hidden=16,
                        seq_len=T,
                        n_layers=3,
                        bidirectional=True,
                    )
                    model_cnnbilstm = CNNBiLSTMClassifier(
                        n_features=F,
                        n_hidden=16,
                        seq_len=T,
                        n_layers=3,
                        bidirectional=True,
                    )

                else:
                    model_nn = NeuralNetwork(in_features=T * F)
                    model_bilstm = BiLSTM(
                        n_features=F,
                        n_hidden=16,
                        seq_len=T,
                        n_layers=3,
                        bidirectional=True,
                    )
                    model_cnnbilstm = CNNBiLSTM(
                        n_features=F,
                        n_hidden=16,
                        seq_len=T,
                        n_layers=3,
                        bidirectional=True,
                    )

                optimizer = torch.optim.Adam(model_nn.parameters(), lr=learning_rate)
                train_losses_nn, val_losses_nn = train(
                    model_nn, optimizer, loss_fn, train_dataloader, test_dataloader
                )

                optimizer = torch.optim.Adam(
                    model_bilstm.parameters(), lr=learning_rate
                )
                train_losses_bilstm, val_losses_bilstm = train(
                    model_bilstm,
                    optimizer,
                    loss_fn,
                    train_dataloader,
                    test_dataloader,
                    n_epochs=n_epochs,
                )

                optimizer = torch.optim.Adam(
                    model_cnnbilstm.parameters(), lr=learning_rate
                )
                train_losses_cnnbilstm, val_losses_cnnbilstm = train(
                    model_cnnbilstm,
                    optimizer,
                    loss_fn,
                    train_dataloader,
                    test_dataloader,
                    n_epochs=n_epochs,
                )
                if plot:
                    fig, axes = plt.subplots(2, 4, figsize=(20, 20))
                    plot_loss_curve(
                        n_epochs,
                        train_losses_nn,
                        val_losses_nn,
                        axes[0, 0],
                        title=f"Neural Network\nLearning curve",
                    )
                    plot_loss_curve(
                        n_epochs,
                        train_losses_bilstm,
                        val_losses_bilstm,
                        axes[0, 1],
                        title=f"BiLSTM\nLearning curve",
                    )

                    evaluate(
                        model_nn,
                        X_train,
                        y_train,
                        axes[1, 0],
                        title="Neural Network",
                        label="train",
                        classifier=True,
                    )
                    evaluate(
                        model_nn,
                        X_test,
                        y_test,
                        axes[1, 0],
                        title="Neural Network",
                        label="test",
                        classifier=True,
                    )

                    evaluate(
                        model_bilstm,
                        X_train,
                        y_train,
                        axes[1, 1],
                        title="Bidirectional LSTM",
                        label="train",
                        classifier=True,
                    )
                    evaluate(
                        model_bilstm,
                        X_test,
                        y_test,
                        axes[1, 1],
                        title="Bidirectional LSTM",
                        label="test",
                        classifier=True,
                    )

                    axes[0, 3].set_axis_off()
                    axes[1, 3].scatter(y_test.numpy(), y_hat_xgb, label="test")
                    axes[1, 3].scatter(y_train.numpy(), y_hat_xgb_train, label="train")
                    axes[1, 3].axline(
                        (y_test.numpy().min(), y_test.numpy().min()),
                        (y_test.numpy().max(), y_test.numpy().max()),
                    )
                    axes[1, 3].set_title(
                        f"XGBoost\nr2 = {r2_score(y_test.numpy(), y_hat_xgb)}\nmae = {mean_absolute_error(y_test.numpy(), y_hat_xgb)}"
                    )
                    axes[1, 3].legend()
                    fig.suptitle(f"Predicting {target} from {feature}")
                    fig.tight_layout()
                    fig.savefig(
                        f"_debugging_images/{target.replace('/', '_')}_{feature.replace('/', '_')}.png"
                    )
                    print()

                # 3. Make predictions
                y_hat_nn = evaluate(model_nn, X_test, y_test)
                y_hat_bilstm = evaluate(model_bilstm, X_test, y_test)
                y_hat_cnnbilstm = evaluate(model_cnnbilstm, X_test, y_test)

                # Save results per sample
                samples = metadata[["sample_id"]]
                samples.loc[idx_test, f"{i}_XGB"] = y_hat_xgb.squeeze()
                samples.loc[idx_test, f"{i}_PLSR"] = y_hat_pls.squeeze()
                samples.loc[idx_test, f"{i}_NN"] = y_hat_nn.squeeze()
                samples.loc[idx_test, f"{i}_BiLSTM"] = y_hat_bilstm.squeeze()
                samples.loc[idx_test, f"{i}_CNNBiLSTM"] = y_hat_cnnbilstm.squeeze()

                results_per_sample = results_per_sample.merge(
                    samples, how="left", on="sample_id"
                )

                torch.save(
                    results_per_sample,
                    f"results/{cell_type}_{target.replace('/', '_')}_{feature.replace('/', '_')}.pth",
                )
