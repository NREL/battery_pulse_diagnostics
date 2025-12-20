# main.py
# Run Everything


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import target_vars, exogenous_vars
from data_loader import load_data, prepare_data_by_cell_type
from models import get_models
from trainer import evaluate_model
from result import save_results, save_model
from visualizer import (
    plot_avg_rmse_r2_bar,
    plot_avg_rmse_r2_box,
    plot_train_test_avg_bar,
    plot_loss_curves,
    plot_rmse_vs_epoch,
    plot_model_comparison_avg
)

# Number of splits (seeds) to run
NUM_SEEDS = 5  


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

def main():
    file_path = 'data/data_for_ml.h5'
    key = 'Dynamic_PsRP_1_1C'
    cell_type = 'A'

    data = load_data(file_path, key)
    X, numeric = prepare_data_by_cell_type(data, cell_type, target_vars, exogenous_vars)
    y = numeric['C/10 discharge capacity']

    models = get_models(X, y, tune_xgb=True)
    trained_models = {}
    results = {}
    loss_histories_all = {}
    test_loss_histories_all = {}

    os.makedirs("results", exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        all_splits = []
        all_train_losses = []
        all_test_losses = []

        for split_seed in range(NUM_SEEDS):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed)
            eval_df, trained_model, loss_histories, test_loss_histories = evaluate_model(
                model, X_train, y_train, X_test, y_test,
                return_trained_model=True, return_loss_history=True
            )
            all_splits.append(eval_df)

            if any(rnn_type in name for rnn_type in ['RNN', 'LSTM', 'GRU']):
                all_train_losses.append(loss_histories)
                all_test_losses.append(test_loss_histories)

        combined_eval_df = pd.concat(all_splits, ignore_index=True)
        trained_models[name] = trained_model
        results[name] = combined_eval_df

        if any(rnn_type in name for rnn_type in ['RNN', 'LSTM', 'GRU']):
            loss_histories_all[name] = all_train_losses
            test_loss_histories_all[name] = all_test_losses

    save_results({k: v.to_dict(orient='records') for k, v in results.items()}, cell_type)

    for name, model in trained_models.items():
        save_model(model, name.replace(" ", "_"), cell_type)

    results_long = pd.concat([
        df.assign(Model=name) for name, df in results.items()
    ], ignore_index=True)
    results_long['CellType'] = cell_type
    results_long.to_csv("results/results_long.csv", index=False)

    plot_avg_rmse_r2_bar(results_long)
    plot_avg_rmse_r2_box(results_long)
    plot_train_test_avg_bar(results_long)
    plot_loss_curves(trained_models)
    plot_rmse_vs_epoch(loss_histories_all, test_loss_histories_all)
    plot_model_comparison_avg(results_long)

if __name__ == "__main__":
    main()



