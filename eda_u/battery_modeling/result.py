# result.py
# Saves and loads results

import os
import json
import joblib
import torch

def save_results(results_dict, cell_type, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'model_results_celltype_{cell_type}.json')
    with open(path, 'w') as f:
        json.dump(results_dict, f, indent=4)

def load_results(cell_type, output_dir='results'):
    path = os.path.join(output_dir, f'model_results_celltype_{cell_type}.json')
    with open(path, 'r') as f:
        return json.load(f)

def save_model(model, name, cell_type, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    path_prefix = os.path.join(output_dir, f'{name}_celltype_{cell_type}')

    if model.__class__.__name__ in ['RNNRegressor', 'LSTMRegressor', 'GRURegressor']:
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'fc_state_dict': model.fc.state_dict(),
            'scaler': model.scaler,
            'y_scaler': model.y_scaler
        }, f"{path_prefix}.pth")
        print(f"PyTorch model saved at {path_prefix}.pth")
    else:
        joblib.dump(model, f"{path_prefix}.pkl")
        print(f"Sklearn model saved at {path_prefix}.pkl")

def load_model(name, cell_type, model_class, output_dir='results'):
    path_prefix = os.path.join(output_dir, f'{name}_celltype_{cell_type}')
    if model_class.__name__ in ['RNNRegressor', 'LSTMRegressor', 'GRURegressor']:
        model = model_class()
        checkpoint = torch.load(f"{path_prefix}.pth")
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.fc.load_state_dict(checkpoint['fc_state_dict'])
        model.scaler = checkpoint['scaler']
        model.y_scaler = checkpoint.get('y_scaler', None)
        print(f"PyTorch model loaded from {path_prefix}.pth")
        return model
    else:
        print(f"Sklearn model loaded from {path_prefix}.pkl")
        return joblib.load(f"{path_prefix}.pkl")


