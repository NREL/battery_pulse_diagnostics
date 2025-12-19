
from neural_network import BasicRNN, DataFrameDataset, NeuralNetwork, ToTensor, train, test
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import random
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


keys = [
    "Charge_Depleting",
    "Charge_Sustaining",
    "Rate_Test_C/2",
    "Rate_Test_1C",
    "PsRP_1_C/2",
    "PsRP_1_1C",
    "PsRP_2_C/2",
    "PsRP_2_1C",
    # "Charge_Sustaining_Time_Variable",
    # "PsRP_2_C/2_Time_Variable"
]

input_size_dict = {
    "Charge_Depleting": 360,
    "Charge_Sustaining": 5400,
    "Rate_Test_C/2":2700,
    "Rate_Test_1C":1440,
    "PsRP_1_C/2":2700,
    "PsRP_1_1C":1400,
    "PsRP_2_C/2":2700,
    "PsRP_2_1C":1440,
    # "Charge_Sustaining_Time_Variable": 1800,
    # "PsRP_2_C/2_Time_Variable": 900
}
targets = [
    "C/3 discharge capacity",
    "soc_mean"
]

key, input_size = keys[0], input_size_dict[keys[0]]
target = targets[1]
overfit_testing = False

history = []

final_train_test = []

splits = 1
# kfold = GroupKFold(n_splits=splits, shuffle=True, random_state=42)

hdf = pd.HDFStore("data/data_partial_charge_for_ml_fixed.h5", mode="r")
# hdf = pd.HDFStore("data/data_for_ml_boundary_conditions.h5", mode="r")
df_all = hdf.get(key)
for cell_type in ["C_"]:# "A_", "B_", "C_", "D_"]:
    df_c = df_all[df_all["cell_id"].str.startswith(cell_type)].reset_index(drop=True)

    # for i, (train_index, test_index) in enumerate(kfold.split(df, groups=df["measurement_id"])):
    for i in range(splits):
        
        if key == "Charge_Sustaining" or key == "Charge_Depleting":
            df_train = df_c[df_c["split_type"] == "training"].reset_index(drop=True).groupby("measurement_id").sample(frac=0.1, random_state=42+i).reset_index(drop=True)
            df_test = df_c[df_c["split_type"] == "testing"].reset_index(drop=True)
        else:
            splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42+i)
            df = df_c.groupby("measurement_id").sample(frac=0.1, random_state=42+i).reset_index(drop=True)
            idx_train, idx_test = next(splitter.split(df, groups=df["measurement_id"]))
            df_train = df.iloc[idx_train].reset_index(drop=True)
            df_test = df.iloc[idx_test].reset_index(drop=True)
        
        if overfit_testing:
            random.seed(42)
            overfit_idxs = random.sample(range(len(df_train)), 5)
            df_train = df_train.iloc[overfit_idxs].reset_index(drop=True)
            df_test = df_train.copy().reset_index(drop=True)
            print(f"Overfit testing enabled, using 5 random samples for cell type {cell_type}")
            print(overfit_idxs)

        # Only use if wanting to test boundary condition data
        # if key == "Charge_Depleting" or key == "Charge_Sustaining":
        #     num_features = 4
        # else:
        #     num_features = 5
        num_features = 3
        # model = NeuralNetwork(input_size=input_size, hidden_size=128, output_size=1).to(device)
        model = BasicRNN(input_size=num_features, hidden_size=128, output_size=1).to(device)
        print(model)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


        train_dataloader = DataLoader(DataFrameDataset(df_train, target=target, key=key, num_features=num_features, transform=ToTensor()), batch_size=64, shuffle=True)
        test_dataloader = DataLoader(DataFrameDataset(df_test, target=target, key=key, num_features=num_features, transform=ToTensor()), batch_size=64, shuffle=False)

        epochs = 20
        history_ = []
        for t in range(epochs):
            print(f"Epoch {t+1}-------------------------------")
            train_loss = train(train_dataloader, model, loss_fn, optimizer) 
            test_loss = test(test_dataloader, model, loss_fn)
            history_.append((train_loss, test_loss))
        print("Done!")

        
        history.append(history_)
        final_train_test.append((train_loss, test_loss))

        torch.save(model.state_dict(), f"results/neural_network/c_testing/RNN_result_graph_{key.replace('/', '')}_{target.replace('/', '')}_{cell_type}model.pth")

print("Final Train/Test Losses:")
for train_loss, test_loss in final_train_test:
    print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")


fig, ax = plt.subplots(2,2, figsize=(10,8))
ax = ax.flatten()

for i, cell_type in enumerate(["A", "B", "C", "D"]):
    history_ = history[i]
    ax[i].plot(range(1, epochs + 1), [h[0] for h in history_], label='Train MSE')
    ax[i].plot(range(1, epochs + 1), [h[1] for h in history_], label='Test MSE')
    ax[i].set_xlabel('Epoch')
    ax[i].set_ylabel('Average MSE')
    ax[i].set_title(f'Cell {cell_type} MSE over Epochs')
    ax[i].legend()
fig.tight_layout()
plt.show()