import pandas as pd
from pathlib import Path
import os
import numpy as np
import scipy as sci

import sys

sys.path.append("../")

hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()

_A_csv = pd.read_csv("data/LG JH3 Cell characterization planning sheet.csv")
_B_csv = pd.read_csv("data/Nissan Leaf Cell characterization planning sheet.csv")
_C_csv = pd.read_csv("data/Ford Fusion Cell characterization planning sheet .csv")
_D_csv = pd.read_csv("data/A123 M1 Cell characterization planning sheet.csv")

A_csv = _A_csv[_A_csv['Step Number'].notna()].reset_index(drop=True).set_index('Step Number')
B_csv = _B_csv[_B_csv['Step Number'].notna()].reset_index(drop=True).set_index('Step Number')
C_csv = _C_csv[_C_csv['Step Number'].notna()].reset_index(drop=True).set_index('Step Number')
D_csv = _D_csv[_D_csv['Step Number'].notna()].reset_index(drop=True).set_index('Step Number')


columns_of_interest = [
    "Control Current",
    "Control Voltage",
    "Control Power",
    "Limit Current",
    "Limit Voltage",
    "Limit Power",
    "Limit Inequality",
    "Limit Time",
]

for key in keys:
    df = hdf.get(key)
    print(key)

    if "_A_" in key:
        cell_id_prefix = "A"
    elif "Leaf" in key:
        cell_id_prefix = "B"
    elif "_C_" in key:
        cell_id_prefix = "C"
    elif "A123" in key:
        cell_id_prefix = "D"

    if cell_id_prefix == "A":
        for step in df['Step'].unique():
            if step in A_csv.index:
                for col in columns_of_interest:
                    df.loc[df['Step'] == step, col] = A_csv.loc[step, col]

    elif cell_id_prefix == "B":
        for step in df['Step'].unique():
            if step in B_csv.index:
                for col in columns_of_interest:
                    df.loc[df['Step'] == step, col] = B_csv.loc[step, col]

    elif cell_id_prefix == "C":
        for step in df['Step'].unique():
            if step in C_csv.index:
                for col in columns_of_interest:
                    df.loc[df['Step'] == step, col] = C_csv.loc[step, col]
                    
    elif cell_id_prefix == "D":
        for step in df['Step'].unique():
            if step in D_csv.index:
                for col in columns_of_interest:
                    df.loc[df['Step'] == step, col] = D_csv.loc[step, col]

    is_hdf = any(["data_raw_with_boundary_conditions.h5" in key for key in os.listdir(Path('data'))])

    if not is_hdf:
        print("Writing new HDF5 file...")
        df.to_hdf(
            "data/data_raw_with_boundary_conditions.h5", key=key[1:], mode="w", complevel=9
        )
    else:
        print("Appending to existing HDF5 file...")
        df.to_hdf("data/data_raw_with_boundary_conditions.h5", key=key[1:], complevel=9)