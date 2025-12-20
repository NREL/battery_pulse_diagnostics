# models.py
# Define Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV

# # PyTorch RNN
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader



def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Earlystopping 
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_model_state = None
        self.best_fc_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, fc):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_fc_state = fc.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_fc_state = fc.state_dict()
            self.counter = 0

# Base RNN model class
class BaseRNNModel(BaseEstimator, RegressorMixin):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2, lr=0.01, epochs=50, seq_len=150, batch_size=64,
                 rnn_type='RNN', patience=3, delta=0.0, lr_scheduler='plateau', lr_factor=0.5, lr_patience=3, min_lr=1e-6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.patience = patience
        self.delta = delta
        self.lr_scheduler = lr_scheduler
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.min_lr = min_lr

        self.model = None
        self.fc = None
        self.scaler = None
        self.y_scaler = None
        self.loss_history = []
        self.test_loss_history = []
        self.validation_indices = []

    def reshape_sequences(self, X, y):
        num_samples = X.shape[0] // self.seq_len
        X_trimmed = X[:num_samples * self.seq_len]
        y_trimmed = y[:num_samples * self.seq_len]
        n_features = X_trimmed.shape[1]
        X_seq = X_trimmed.reshape(num_samples, self.seq_len, n_features)
        self.input_size = n_features
        y_seq = np.array(y_trimmed).reshape(num_samples, self.seq_len).mean(axis=1)
        return X_seq, y_seq

    def safe_predict(self, X):
        if self.scaler is None or self.y_scaler is None:
            raise ValueError("Model must be fitted before prediction.")
        X_scaled = self.scaler.transform(X)
        dummy_y = np.zeros(X_scaled.shape[0])
        if len(X_scaled) < self.seq_len:
            return None
        X_seq, _ = self.reshape_sequences(X_scaled, dummy_y)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)
        with torch.no_grad():
            out, _ = self.model(X_tensor)
            preds = self.fc(out[:, -1, :])
        preds_np = preds.numpy().flatten()
        return self.y_scaler.inverse_transform(preds_np.reshape(-1, 1)).flatten()

    def fit(self, X, y):
        set_global_seed(42)
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        X_seq, y_seq = self.reshape_sequences(X_scaled, y_scaled)

        split_idx = int(0.8 * len(X_seq))
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

        self.validation_indices = list(range(split_idx * self.seq_len, (split_idx + len(X_val)) * self.seq_len))

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if self.rnn_type == 'RNN':
            self.model = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.model = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.model = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        else:
            raise ValueError("Unsupported rnn_type")

        self.fc = nn.Linear(self.hidden_size, 1)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.model.parameters()) + list(self.fc.parameters()), lr=self.lr)

        if self.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=self.lr_factor, patience=self.lr_patience, min_lr=self.min_lr
            )
        elif self.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=self.lr_factor
            )
        else:
            scheduler = None

        early_stopper = EarlyStopping(patience=self.patience, delta=self.delta)

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                out, _ = self.model(xb)
                preds = self.fc(out[:, -1, :])
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            self.loss_history.append(np.sqrt(np.mean(train_losses)))

            self.model.eval()
            with torch.no_grad():
                out_val, _ = self.model(torch.tensor(X_val, dtype=torch.float32))
                preds_val = self.fc(out_val[:, -1, :])
                val_loss = criterion(preds_val, torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
                val_rmse = np.sqrt(val_loss.item())
                self.test_loss_history.append(val_rmse)

                if scheduler:
                    if self.lr_scheduler == 'plateau':
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                print(f"Epoch {epoch+1}: Learning rate is {optimizer.param_groups[0]['lr']}")

                early_stopper(val_loss.item(), self.model, self.fc)
                if early_stopper.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if early_stopper.best_model_state and early_stopper.best_fc_state:
            self.model.load_state_dict(early_stopper.best_model_state)
            self.fc.load_state_dict(early_stopper.best_fc_state)

        return self


class RNNRegressor(BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type='RNN', **kwargs)

class LSTMRegressor(BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type='LSTM', **kwargs)

class GRURegressor(BaseRNNModel):
    def __init__(self, **kwargs):
        super().__init__(rnn_type='GRU', **kwargs)


def get_models(X, y, tune_xgb=False, rnn_hidden_sizes=[32]):
    models = {
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(verbosity=0),
        'LightGBM': LGBMRegressor(verbose=-1),
        'Dummy Regressor': DummyRegressor(strategy='mean')
    }

    for size in rnn_hidden_sizes:
        models[f'RNN Hidden {size}'] = RNNRegressor(input_size=X.shape[1], hidden_size=size)
        models[f'LSTM Hidden {size}'] = LSTMRegressor(input_size=X.shape[1], hidden_size=size)
        models[f'GRU Hidden {size}'] = GRURegressor(input_size=X.shape[1], hidden_size=size)

    if tune_xgb:
        param_grid = {
            'max_depth': [1, 2, 3],
            'min_child_weight': [1, 3],
            'learning_rate': [0.01, 0.1]
        }
        xgb = XGBRegressor(n_estimators=10, random_state=5, verbosity=0)
        grid_search = GridSearchCV(xgb, param_grid=param_grid, scoring='r2', cv=3)
        grid_search.fit(X, y)
        models['Tuned XGBoost'] = grid_search.best_estimator_

    return models
