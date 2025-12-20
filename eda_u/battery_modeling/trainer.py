# trainer.py
# Train and Evaluate Modelss

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   return_trained_model=False, return_indices=False, return_loss_history=False):
    results = []
    trained_model = None
    test_indices = X_test.index.tolist() if hasattr(X_test, 'index') else list(range(len(X_test)))
    loss_histories = []
    test_loss_histories = []

    model_copy = model.__class__(**model.get_params())
    if hasattr(model_copy, "set_params") and "random_state" in model_copy.get_params():
        model_copy.set_params(random_state=42)

    model_copy.fit(X_train, y_train)

    if hasattr(model_copy, "safe_predict"):
        y_pred_test = model_copy.safe_predict(X_test)
        y_pred_train = model_copy.safe_predict(X_train)
    else:
        y_pred_test = model_copy.predict(X_test)
        y_pred_train = model_copy.predict(X_train)

    if y_pred_test is None or y_pred_train is None:
        print("Skipping evaluation due to insufficient sequence length.")
        return pd.DataFrame(), None, [], []

    y_test_trimmed = y_test[:len(y_pred_test)]
    y_train_trimmed = y_train[:len(y_pred_train)]

    def safe_r2(y_true, y_pred):
        return r2_score(y_true, y_pred) if len(y_true) > 1 else None

    if hasattr(model_copy, "loss_history"):
        loss_histories = model_copy.loss_history
    if hasattr(model_copy, "test_loss_history"):
        test_loss_histories = model_copy.test_loss_history

    trained_model = model_copy

    results.append({
        "r2_test": safe_r2(y_test_trimmed, y_pred_test),
        "rmse_test": mean_squared_error(y_test_trimmed, y_pred_test) ** 0.5,
        "mae_test": mean_absolute_error(y_test_trimmed, y_pred_test),
        "r2_train": safe_r2(y_train_trimmed, y_pred_train),
        "rmse_train": mean_squared_error(y_train_trimmed, y_pred_train) ** 0.5,
        "mae_train": mean_absolute_error(y_train_trimmed, y_pred_train),
    })

    eval_df = pd.DataFrame(results)

    if return_indices and return_trained_model and return_loss_history:
        return eval_df, trained_model, test_indices, loss_histories, test_loss_histories
    elif return_indices and return_trained_model:
        return eval_df, trained_model, test_indices
    elif return_trained_model and return_loss_history:
        return eval_df, trained_model, loss_histories, test_loss_histories
    elif return_trained_model:
        return eval_df, trained_model
    elif return_loss_history:
        return eval_df, loss_histories, test_loss_histories
    elif return_indices:
        return eval_df, test_indices
    else:
        return eval_df


