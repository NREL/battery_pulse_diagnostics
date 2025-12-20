# data_loader.py
# Load and Prepare Data


import pandas as pd
import h5py

def explore_h5_keys(filepath):
    with h5py.File(filepath, 'r') as f:
        return list(f.keys())

def load_data(filepath, key):
    data = pd.read_hdf(filepath, key=key)
    data['direction_num'] = data['direction'].map({'charge': 1, 'discharge': 0})
    data['cell_type'] = data['cell_id'].astype(str).str[0]

    drop_cols = [
        'cell_id', 'Unnamed: 0', 'measurement_id', 'direction',
        'Charge depleting cycle charge throughput',
        'Charge sustaining cycle charge efficiency',
        'Post C/10 charge relaxation fit MSE',
        'Post C/5 charge relaxation fit MSE',
        'Post C/3 charge relaxation fit MSE',
        'Post P/3 charge relaxation fit MSE',
        'Post C/2 charge relaxation fit MSE',
        'Post 1C charge relaxation fit MSE',
        'Thickness growth', 'Volume growth'
    ]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns])
    return data

def prepare_data_by_cell_type(data, cell_type, target_vars, exogenous_vars):
    filtered = data[data['cell_type'] == cell_type]
    numeric = filtered.select_dtypes(include='number')
    exclude_cols = set(target_vars + exogenous_vars)
    feature_cols = [col for col in numeric.columns if col.strip() not in exclude_cols]
    X = numeric[feature_cols].copy()
    return X, numeric

