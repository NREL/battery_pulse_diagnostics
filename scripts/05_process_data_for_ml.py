import os
from pathlib import Path
import pandas as pd
import numpy as np
from utils.data_utils import join_targets_to_features

import sys
sys.path.append("../")

# Features
features_dcir = pd.read_csv("data/features_dcir_no_interpolation.csv")
features_static_hppc = pd.read_csv("data/features_pulse_hppc.csv")
features_static_rapid = pd.read_csv("data/features_pulse_rapid.csv")
features_static_psrp1 = pd.read_csv("data/features_pulse_psrp1.csv")
features_static_psrp2_chg = pd.read_csv("data/features_pulse_psrp2_charging.csv")
features_static_psrp2_dis = pd.read_csv("data/features_pulse_psrp2_discharging.csv")
features_dynamic_psrp1_Cb2 = pd.read_csv("data/features_pulse_psrp1_Cb2.csv")
features_dynamic_psrp1_1C = pd.read_csv("data/features_pulse_psrp1_1C.csv")
features_dynamic_psrp2_Cb2 = pd.read_csv("data/features_pulse_psrp2_Cb2.csv")
features_dynamic_psrp2_1C = pd.read_csv("data/features_pulse_psrp2_1C.csv")

features_raw = {
    "DCIR": features_dcir,
    "Static_HPPC": features_static_hppc,
    "Static_Rapid": features_static_rapid,
    "Static_PsRP_1": features_static_psrp1,
    "Static_PsRP_2_Chg": features_static_psrp2_chg,
    "Static_PsRP_2_Dis": features_static_psrp2_dis,
    "Dynamic_PsRP_1_C/2": features_dynamic_psrp1_Cb2,
    "Dynamic_PsRP_1_1C": features_dynamic_psrp1_1C,
    "Dynamic_PsRP_2_C/2": features_dynamic_psrp2_Cb2,
    "Dynamic_PsRP_2_1C": features_dynamic_psrp2_1C,
}

# Targets
targets_raw = pd.read_csv("data/targets_soh.csv")
idx_targets = np.array([4, 5, 6, 7, 8, 9, 10, 11, 17])
target_variables = targets_raw.columns[idx_targets]

for feature_key in features_raw:

    # Create a dataframe of a feature set (one pulse measurement type) joined with all of its possible targets
    features_and_targets = join_targets_to_features(
        features_raw[feature_key], targets_raw
    )

    # Check if the output file has already been created
    is_hdf = any(["data_for_ml.h5" in key for key in os.listdir(Path('data'))])

    # Write to HDF5 database
    if not is_hdf:
        print(feature_key)
        features_and_targets.to_hdf(
            "data/data_for_ml.h5", key=feature_key, mode="w", complevel=9
        )
    else:
        print(feature_key)
        features_and_targets.to_hdf("data/data_for_ml.h5", key=feature_key, complevel=9)
