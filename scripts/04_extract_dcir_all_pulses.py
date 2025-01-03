import pandas as pd
import numpy as np
import utils.data_utils as utils
import re

import sys
sys.path.append("../")

hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()

# Extract HPPC resistance at 50% SOC, across varied temperature, pulse rates, and charge/discharge direction
rates = ["2C"]  # "C/10", "C/2", "1C"
directions = ["charge", "discharge"]
times = ["0p1s", "1s", "4s", "10s"]
flag_colnames = False


output = {
    "measurement_id": [],
    "temperature_ambient": [],
    "cell_id": [],
    "soc": [],
    "voltage_0.0s": [],
    "0p1s": [],
    "1s": [],
    "4s": [],
    "10s": [],
}
for ii, key in enumerate(keys):
    print(key)
    df = hdf.get(key)

    # cell id
    cell_id_num = re.search(r"Cell\d\d", key, re.IGNORECASE).group()[-2:]
    if "_A_" in key:
        cell_id_prefix = "A_"
    elif "Leaf" in key:
        cell_id_prefix = "B_"
    elif "_C_" in key:
        if "_TX" in key:
            cell_id_prefix = "C_TX_"
        else:
            cell_id_prefix = "C_"
    elif "A123" in key:
        cell_id_prefix = "D_"

    # DCIR, Only keep pulses that do not hit voltage limits to avoid inconsistent resistance calculations
    dcir = utils.get_hppc_pulse_resistances(df)
    hppc = utils.get_hppc_pulses(df)
    dcir = (
        dcir.loc[
            np.logical_and(
                ~dcir["is_voltage_limit_discharge"], ~dcir["is_voltage_limit_charge"]
            )
        ]
        .reset_index()
        .copy()
    )
    dcir["voltage_0.0s"] = hppc["voltage_0.0s"]
    # Temperature should not vary within meausurement precision during HPPC, so just a single valute
    temp = np.mean(dcir["temp_mean"])
    # For each rate, 4 pulses (charge/discharge, in both charging/discharging directions), resistance calculated at 4 times
    for i, rate in enumerate(rates):
        print(rate)
        for j, dir in enumerate(directions):
            print(dir)
            for k, time in enumerate(times):
                print(time)
                dcir_temp = (
                    dcir.loc[
                        np.logical_and(dcir["rate"] == rate, dcir["direction"] == dir)
                    ]
                    .reset_index()
                    .copy()
                )
                # Some cells may have valid measurements only at low rates, so higher rate data will be missing
                if dcir_temp.shape[0] > 0:
                    soc_ = dcir_temp["soc"].to_list()
                    v0_ = dcir_temp["voltage_0.0s"].to_list()
                    dcir_average = (
                        (
                            dcir_temp["charge_resistance_" + time]
                            + dcir_temp["discharge_resistance_" + time]
                        )
                        / 2
                    ).to_list()
                    output[time] += dcir_average
                    assert len(soc_) == len(dcir_average)
                    n_pulses = len(soc_)
                    if k == 0:
                        output["voltage_0.0s"] += v0_
                        output["soc"] += soc_
                        output["measurement_id"] += [ii] * n_pulses
                        output["temperature_ambient"] += [temp] * n_pulses
                        output["cell_id"] += [cell_id_prefix + cell_id_num] * n_pulses

                else:
                    soc_ = [np.nan]
                    v0_ = [np.nan]
                    dcir_average = [np.nan]
                    output[time] += dcir_average
                    n_pulses = len(soc_)
                    if k == 0:
                        output["voltage_0.0s"] += v0_
                        output["soc"] += soc_
                        output["measurement_id"] += [ii] * n_pulses
                        output["temperature_ambient"] += [temp] * n_pulses
                        output["cell_id"] += [cell_id_prefix + cell_id_num] * n_pulses

            pd.DataFrame(output).to_csv("data/features_dcir_no_interpolation.csv")

hdf.close()
