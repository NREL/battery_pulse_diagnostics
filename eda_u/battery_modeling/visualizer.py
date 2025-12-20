# visulaizer.py
# to visulaize the rsults


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Simplify model names
def simplify_model_names(df):
    df['Model'] = df['Model'].replace({
        r'RNN.*': 'RNN',
        r'LSTM.*': 'LSTM',
        r'GRU.*': 'GRU'
    }, regex=True)
    return df

# Average RMSE and R² bar chart
def plot_avg_rmse_r2_bar(results_long):
    results_long = simplify_model_names(results_long)
    summary = results_long.groupby('Model').agg({
        'r2_test': 'mean',
        'rmse_test': 'mean'
    }).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=summary, x='Model', y='rmse_test', ax=axs[0], palette='Blues_d')
    axs[0].set_title('Average RMSE (Test)')
    axs[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=summary, x='Model', y='r2_test', ax=axs[1], palette='Greens_d')
    axs[1].set_title('Average R² (Test)')
    axs[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig("results/avg_rmse_r2_bar.png")
    plt.close()


# Box plots for RMSE and R²
def plot_avg_rmse_r2_box(results_long):
    results_long = simplify_model_names(results_long)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=results_long, x='Model', y='rmse_test', ax=axs[0], palette='Blues')
    axs[0].set_title('RMSE Distribution (Test)')
    axs[0].tick_params(axis='x', rotation=45)
    sns.boxplot(data=results_long, x='Model', y='r2_test', ax=axs[1], palette='Greens')
    axs[1].set_title('R² Distribution (Test)')
    axs[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("results/avg_rmse_r2_box.png")
    plt.close()

# Train vs Test bar plot with error bars
def plot_train_test_avg_bar(results_long):
    results_long = simplify_model_names(results_long)
    selected_models = ['RNN', 'LSTM', 'GRU', 'Dummy Regressor']
    filtered = results_long[results_long['Model'].isin(selected_models)]
    summary = filtered.groupby('Model').agg({
        'r2_train': ['mean', 'std'],
        'r2_test': ['mean', 'std'],
        'rmse_train': ['mean', 'std'],
        'rmse_test': ['mean', 'std']
    }).reset_index()
    summary.columns = ['Model', 'R2_Train_Mean', 'R2_Train_Std', 'R2_Test_Mean', 'R2_Test_Std',
                       'RMSE_Train_Mean', 'RMSE_Train_Std', 'RMSE_Test_Mean', 'RMSE_Test_Std']
    x = np.arange(len(summary['Model']))
    width = 0.35
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    axs[0].bar(x - width/2, summary['RMSE_Train_Mean'], width,
               yerr=summary['RMSE_Train_Std'], capsize=5, label='Train', color='skyblue')
    axs[0].bar(x + width/2, summary['RMSE_Test_Mean'], width,
               yerr=summary['RMSE_Test_Std'], capsize=5, label='Test', color='salmon')
    axs[0].set_title('RMSE: Train vs Test')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(summary['Model'])
    axs[0].legend()
    axs[1].bar(x - width/2, summary['R2_Train_Mean'], width,
               yerr=summary['R2_Train_Std'], capsize=5, label='Train', color='skyblue')
    axs[1].bar(x + width/2, summary['R2_Test_Mean'], width,
               yerr=summary['R2_Test_Std'], capsize=5, label='Test', color='salmon')
    axs[1].set_title('R²: Train vs Test')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(summary['Model'])
    axs[1].legend()
    plt.suptitle('Train vs Test Performance Comparison')
    plt.tight_layout()
    plt.show()
    plt.savefig("results/train_test_avg_bar_rnn.png")
    plt.close()

# Loss curves for RNN models
def plot_loss_curves(trained_models):
    for name, model in trained_models.items():
        if hasattr(model, 'loss_history') and model.loss_history:
            plt.figure(figsize=(10, 5))
            plt.plot(model.loss_history, label='Train Loss', marker='o')
            if hasattr(model, 'test_loss_history') and model.test_loss_history:
                plt.plot(model.test_loss_history, label='Test Loss', marker='x')
            plt.title(f'{name} - Train & Test Loss vs Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            plt.savefig(f"results/{name.replace(' ', '_')}_loss_curve.png")
            plt.close()

# RMSE vs Epoch for all seeds
def plot_rmse_vs_epoch(train_losses_dict, test_losses_dict):
    for name in train_losses_dict:
        plt.figure(figsize=(10, 6))
        for seed, (train_loss, test_loss) in enumerate(zip(train_losses_dict[name], test_losses_dict[name])):
            plt.plot(test_loss, label=f'Seed {seed} Test RMSE', linestyle='--')
            plt.plot(train_loss, label=f'Seed {seed} Train RMSE', linestyle='-')
        plt.title(f'{name} - RMSE vs Epoch (All Seeds)')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f"results/{name.replace(' ', '_')}_rmse_vs_epoch.png")
        plt.close()

# Comparison plot for RNN, LSTM, GRU vs XGBoost and LightGBM
plot_model_comparison_avg = plot_train_test_avg_bar

# Bar chart for selected models
def plot_selected_models_bar(results_long):
    results_long = simplify_model_names(results_long)
    selected_order = ['RNN', 'LSTM', 'GRU', 'LightGBM', 'XGBoost']
    filtered = results_long[results_long['Model'].isin(selected_order)]
    summary = filtered.groupby('Model').agg({
        'r2_test': 'mean',
        'rmse_test': 'mean'
    }).reindex(selected_order).reset_index()
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=summary, x='Model', y='rmse_test', ax=axs[0], palette='Blues_d', order=selected_order)
    axs[0].set_title('Average RMSE (Test)')
    axs[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=summary, x='Model', y='r2_test', ax=axs[1], palette='Greens_d', order=selected_order)
    axs[1].set_title('Average R² (Test)')
    axs[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/selected_models_avg_bar.png")
    plt.show()
    plt.close()

# Box plots for selected models
def plot_selected_models_box(results_long):
    results_long = simplify_model_names(results_long)
    selected_order = ['RNN', 'LSTM', 'GRU', 'LightGBM', 'XGBoost']
    filtered = results_long[results_long['Model'].isin(selected_order)]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.boxplot(data=filtered, x='Model', y='rmse_test', ax=axs[0], palette='Blues', order=selected_order)
    axs[0].set_title('RMSE Distribution (Test)')
    axs[0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=filtered, x='Model', y='r2_test', ax=axs[1], palette='Greens', order=selected_order)
    axs[1].set_title('R² Distribution (Test)')
    axs[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("results/selected_models_box.png")
    plt.show()
    plt.close()




