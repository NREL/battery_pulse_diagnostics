import os
from pathlib import Path
import pandas as pd
import numpy as np
from utils.data_utils import join_targets_to_features

import sys
sys.path.append("../")

### Duplication of code in 05_process_data_for_ml.py to read partial charge csvs into h5 file

### Authored by Ethan Tenney, SULI intern Fall 2025

# Features
features_charge_depleting = pd.read_csv("data/features_partial_charge_v2_depleting_cycles.csv")
features_charge_sustaining = pd.read_csv("data/features_partial_charge_v2_sustaining_cycles.csv")
features_rate_test_C2 = pd.read_csv("data/features_partial_charge_v2_rate_test_C2_charge_cc.csv")
features_rate_test_1C = pd.read_csv("data/features_partial_charge_v2_rate_test_1C_charge_cc.csv")
features_psrp_1_Cb2 = pd.read_csv("data/features_partial_charge_v2_psrp_1_C2_charge.csv")
features_psrp_1_1C = pd.read_csv("data/features_partial_charge_v2_psrp_1_1C_charge.csv")
features_psrp_2_Cb2 = pd.read_csv("data/features_partial_charge_v2_psrp_2_C2_charge.csv")
features_psrp_2_1C = pd.read_csv("data/features_partial_charge_v2_psrp_2_1C_charge.csv")
# features_charge_sustaining_time_variable = pd.read_csv("data/features_partial_charge_sustaining_cycle_time_variable.csv")
# features_psrp_2_C2_time_variable = pd.read_csv("data/features_partial_charge_psrp_2_C2_charge_time_variable.csv")

# dictionary of keys for dataframes in h5 file
features_raw = {
    "Charge_Depleting": features_charge_depleting,
    "Charge_Sustaining": features_charge_sustaining,
    "Rate_Test_C/2": features_rate_test_C2,
    "Rate_Test_1C": features_rate_test_1C,
    "PsRP_1_C/2": features_psrp_1_Cb2,
    "PsRP_1_1C": features_psrp_1_1C,
    "PsRP_2_C/2": features_psrp_2_Cb2,
    "PsRP_2_1C": features_psrp_2_1C,
    # "Charge_Sustaining_Time_Variable": features_charge_sustaining_time_variable,
    # "PsRP_2_C/2_Time_Variable": features_psrp_2_C2_time_variable
}

# Targets
targets_raw = pd.read_csv("data/targets_soh_fixed_id.csv")
idx_targets = np.array([4, 5, 6, 7, 8, 9, 10, 11, 17])
target_variables = targets_raw.columns[idx_targets]

for feature_key in features_raw:

    # Create a dataframe of a feature set (one pulse measurement type) joined with all of its possible targets
    features_and_targets = join_targets_to_features(
        features_raw[feature_key], targets_raw
    )

    # Check if the output file has already been created
    is_hdf = any(["data_for_ml_boundary_conditions.h5" in key for key in os.listdir(Path('data'))])

    # Write to HDF5 database
    if not is_hdf:
        print(feature_key)
        features_and_targets.to_hdf(
            "data/data_for_ml_boundary_conditions.h5", key=feature_key, mode="w", complevel=9
        )
    else:
        print(feature_key)
        features_and_targets.to_hdf("data/data_for_ml_boundary_conditions.h5", key=feature_key, complevel=9)