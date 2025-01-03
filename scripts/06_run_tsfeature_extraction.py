import os
import pandas as pd
from pathlib import Path
import time
import torch
from tsfresh import extract_features
import sys

def stack_df(df):
    voltages = df.filter(regex="voltage")
    voltages = voltages.rename(
        columns={t: float(t.split("_")[1][:-1]) for t in voltages.columns}
    )
    voltages = (
        voltages.stack()
        .reset_index(level=[0, 1])
        .rename(columns={"level_0": "column_id", "level_1": "time", 0: "voltage"})
    )

    currents = df.filter(regex="current")
    currents = currents.rename(
        columns={t: float(t.split("_")[1][:-1]) for t in currents.columns}
    )
    currents = (
        currents.stack()
        .reset_index(level=[0, 1])
        .rename(columns={"level_0": "column_id", "level_1": "time", 0: "current"})
    )

    polars = df.filter(regex="polarization")
    polars = polars.rename(
        columns={t: float(t.split("_")[1][:-1]) for t in polars.columns}
    )
    polars = (
        polars.stack()
        .reset_index(level=[0, 1])
        .rename(columns={"level_0": "column_id", "level_1": "time", 0: "polarization"})
    )

    voltages["current"] = currents["current"]
    voltages["polarization"] = polars["polarization"]

    return voltages


def format_extracted_features(
    output_fn: str = "data/data_for_ml_extracted_features.h5",
):

    # Load the extracted features
    features = torch.load("data/extracted_features/extracted_features.pth")

    # Load the already-processed data
    data = pd.HDFStore("data/data_for_ml.h5")
    static_hppc = data.get("Static_HPPC")
    static_rapid = data.get("Static_Rapid")
    static_psrp1 = data.get("Static_PsRP_1")
    static_psrp2_chg = data.get("Static_PsRP_2_Chg")
    static_psrp2_dis = data.get("Static_PsRP_2_Dis")
    dynamic_psrp1_Cb2 = data.get("Dynamic_PsRP_1_C/2")
    dynamic_psrp1_1C = data.get("Dynamic_PsRP_1_1C")
    dynamic_psrp2_Cb2 = data.get("Dynamic_PsRP_2_C/2")
    dynamic_psrp2_1C = data.get("Dynamic_PsRP_2_1C")
    already_processed = {
        "Static HPPC": static_hppc,
        "Static Rapid": static_rapid,
        "Static PsRP 1": static_psrp1,
        "Static PsRP 2 Chg": static_psrp2_chg,
        "Static PsRP 2 Dis": static_psrp2_dis,
        "Dynamic PsRP 1 C/2": dynamic_psrp1_Cb2,
        "Dynamic PsRP 1 1C": dynamic_psrp1_1C,
        "Dynamic PsRP 2 C/2": dynamic_psrp2_Cb2,
        "Dynamic PsRP 2 1C": dynamic_psrp2_1C,
    }

    # For each pulse type
    for key in features.keys():

        print(key)

        # Get the extracted features
        extracted_feats = features[key]

        # and the rest of the values (cell id, temp, etc.)
        other_vars = (
            already_processed[key]
            .filter(regex=r"^(?!{}*)".format("voltage|polarization|current"))
            .drop(columns="Unnamed: 0")
        )

        # Concatenate them
        new_features_df = pd.concat([other_vars, extracted_feats])

        # Check if the output file has already been created
        is_hdf = any([output_fn in key for key in os.listdir(Path())])

        # Write to HDF5 database
        if not is_hdf:
            new_features_df.to_hdf(
                output_fn, key=key.replace(" ", "_"), mode="w", complevel=9
            )
        else:
            new_features_df.to_hdf(output_fn, key=key.replace(" ", "_"), complevel=9)


if __name__ == "__main__":
    sys.path.append("../")

    data = pd.HDFStore("data/data_for_ml.h5")

    static_hppc = data.get("Static_HPPC")
    static_rapid = data.get("Static_Rapid")
    static_psrp1 = data.get("Static_PsRP_1")
    static_psrp2_chg = data.get("Static_PsRP_2_Chg")
    static_psrp2_dis = data.get("Static_PsRP_2_Dis")
    dynamic_psrp1_Cb2 = data.get("Dynamic_PsRP_1_C/2")
    dynamic_psrp1_1C = data.get("Dynamic_PsRP_1_1C")
    dynamic_psrp2_Cb2 = data.get("Dynamic_PsRP_2_C/2")
    dynamic_psrp2_1C = data.get("Dynamic_PsRP_2_1C")

    data.close()

    tests = {
        "Static HPPC": static_hppc,
        "Static Rapid": static_rapid,
        "Static PsRP 1": static_psrp1,
        "Static PsRP 2 Chg": static_psrp2_chg,
        "Static PsRP 2 Dis": static_psrp2_dis,
        "Dynamic PsRP 1 C/2": dynamic_psrp1_Cb2,
        "Dynamic PsRP 1 1C": dynamic_psrp1_1C,
        "Dynamic PsRP 2 C/2": dynamic_psrp2_Cb2,
        "Dynamic PsRP 2 1C": dynamic_psrp2_1C,
    }

    results = dict()

    for key in tests:

        # Add polarization feature
        voltage_cols = tests[key].filter(regex="voltage").columns
        for v in voltage_cols:  # [1:]
            tests[key][f"polarization_{v.split('_')[1]}"] = (
                tests[key][v] - tests[key][voltage_cols[0]]
            )

        print("Extracting features for ", key)
        t0 = time.time()

        df = tests[key]
        df_stacked = stack_df(df)
        extracted_features = extract_features(
            df_stacked, column_id="column_id", column_sort="time"
        )

        results[key] = extracted_features
        print("Time elapsed: ", time.time() - t0)

        torch.save("data/extracted_features/extracted_features.pth")

    # Format the results into analagous format to "data/data_for_ml.h5"
    format_extracted_features("data/data_for_ml_extracted_features.h5")
