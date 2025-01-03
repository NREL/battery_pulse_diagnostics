import pandas as pd
import numpy as np
import utils.data_utils as utils
import re

hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()

# Extract HPPC resistance at 50% SOC, across varied temperature, pulse rates, and charge/discharge direction
rates = ["C/10", "C/2", "1C", "2C"]
directions = ["charge", "discharge"]
times = ["0p1s", "1s", "4s", "10s"]
flag_colnames = False
for ii, key in enumerate(keys):
    print(key)
    df = hdf.get(key)
    # cell id
    cell_id_num = re.search(r"Cell\d\d", key, re.IGNORECASE).group()[-2:]
    # DCIR, Only keep pulses that do not hit voltage limits to avoid inconsistent resistance calculations
    dcir = utils.get_hppc_pulse_resistances(df)
    dcir = (
        dcir.loc[
            np.logical_and(
                ~dcir["is_voltage_limit_discharge"], ~dcir["is_voltage_limit_charge"]
            )
        ]
        .reset_index()
        .copy()
    )
    # Temperature should not vary within meausurement precision during HPPC, so just a single valute
    temp = np.mean(dcir["temp_mean"])
    # For each rate, 4 pulses (charge/discharge, in both charging/discharging directions), resistance calculated at 4 times
    for i, rate in enumerate(rates):
        for j, dir in enumerate(directions):
            for k, time in enumerate(times):
                dcir_temp = (
                    dcir.loc[
                        np.logical_and(dcir["rate"] == rate, dcir["direction"] == dir)
                    ]
                    .reset_index()
                    .copy()
                )
                # Some cells may have valid measurements only at low rates, so higher rate data will be missing
                if dcir_temp.shape[0] > 0:
                    soc = dcir_temp["soc"].to_numpy()
                    idx_sort = np.argsort(soc)
                    dcir_dis_50soc = np.interp(
                        0.5,
                        soc[idx_sort],
                        dcir_temp["discharge_resistance_" + time].to_numpy()[idx_sort],
                    )
                    dcir_chg_50soc = np.interp(
                        0.5,
                        soc[idx_sort],
                        dcir_temp["charge_resistance_" + time].to_numpy()[idx_sort],
                    )
                else:
                    dcir_dis_50soc = np.nan
                    dcir_chg_50soc = np.nan

                if dir == "charge":
                    dir_name = "charging"
                else:
                    dir_name = "discharging"
                if i == 0 and j == 0 and k == 0:
                    dcir_50soc = np.array((dcir_dis_50soc, dcir_chg_50soc))
                    if not flag_colnames:
                        col_name = [
                            "discharge_resistance_"
                            + time
                            + "_"
                            + rate
                            + "_"
                            + dir_name,
                            "charge_resistance_" + time + "_" + rate + "_" + dir_name,
                        ]
                else:
                    dcir_50soc = np.concatenate(
                        (dcir_50soc, np.array((dcir_dis_50soc, dcir_chg_50soc)))
                    )
                    if not flag_colnames:
                        col_name = col_name + [
                            "discharge_resistance_"
                            + time
                            + "_"
                            + rate
                            + "_"
                            + dir_name,
                            "charge_resistance_" + time + "_" + rate + "_" + dir_name,
                        ]
    # Only need to write column names once for all the data files
    if not flag_colnames:
        flag_colnames = True
    col_name = ["measurement_id", "cell_id", "temperature_ambient"] + col_name

    # modifiers by cell type
    if "_A_" in key:
        cell_id_prefix = "A_"
    elif "Leaf" in key:
        cell_id_prefix = "B_"
    elif "_C_" in key:
        if "_TX" in key:
            cell_id_prefix = "C_TX_"
        elif "_NY" in key:
            cell_id_prefix = "C_NY_"
        else:
            cell_id_prefix = "C_"
    elif "A123" in key:
        cell_id_prefix = "D_"

    # instantiate container or concatenate new value
    if key == keys[0]:
        temperature = temp
        dcir_ = dcir_50soc
        cell_id = cell_id_prefix + cell_id_num
        measurement_id = ii
    else:
        _temperature = temp
        _dcir_ = dcir_50soc
        _cell_id = cell_id_prefix + cell_id_num

        dcir_ = np.row_stack((dcir_, _dcir_))
        cell_id = np.row_stack((cell_id, _cell_id))
        temperature = np.row_stack((temperature, _temperature))
        measurement_id = np.row_stack((measurement_id, ii))
    del df

dcir = np.hstack((measurement_id, cell_id, temperature, dcir_))
dcir = pd.DataFrame(dcir, columns=col_name)
dcir.to_csv("data/features_dcir.csv")

hdf.close()
