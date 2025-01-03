import pandas as pd
import numpy as np
import utils.data_utils as utils
import re
from scipy.integrate import trapz
from scipy.optimize import curve_fit

import sys
sys.path.append("../")

def exponential_relaxation(x, A, tau, C):
    return A * np.exp(-x / tau) + C


hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()
# Extract SOH metrics
for i, key in enumerate(keys):
    df = hdf.get(key)
    # cell id
    cell_id_num = re.search(r"Cell\d\d", key).group()[-2:]
    # capacities
    capacity = utils.get_discharge_capacities_cc(df)
    # temperature
    temp = capacity.loc[0, "temp_min"]
    # Charge depleting drive cycle (US06) cumulative relative charge throughput
    is_us06 = ["Charge depleting" in seg_desc for seg_desc in df["Segment Description"]]
    df_ = utils.retime(df[is_us06])
    us06_charge_throughput = (
        trapz(np.abs(df_["Current, A"]), x=df_["Segment Time, S"]) / 3600
    )
    # Charge sustaining cycle (FCR) charge efficiency
    is_fcr = ["Charge sustaining" in seg_desc for seg_desc in df["Segment Description"]]
    df_ = utils.retime(df[is_fcr])
    current = df_["Current, A"].to_numpy()
    discharge_current = current.copy()
    discharge_current[current > 0] = 0
    charge_current = current.copy()
    charge_current[current <= 0] = 0
    charge_amphr = trapz(charge_current) / 3600
    discharge_amphr = trapz(discharge_current) / 3600
    fcr_efficiency = charge_amphr / np.abs(discharge_amphr)

    # post-charge relaxation exponential fit error (li-plating proxy)
    dfs = utils.get_after_charge_rests(df)
    for df_ in dfs:
        seg_descr = df_.loc[1, "Segment Description"]
        if "_A_" in key or "_C_" in key:
            initial_guess = (0.01, 500, 4.1)  # Initial guess for parameters (A, tau, C)
        elif "A123" in key:
            initial_guess = (0.01, 500, 3.5)
        else:
            initial_guess = (0.01, 500, 8.2)  # Initial guess for parameters (A, tau, C)
        params, _ = curve_fit(
            exponential_relaxation,
            df_["Segment Time, S"][:500],
            df_["Voltage, V"][:500],
            p0=initial_guess,
        )
        # Extract the fitted parameters
        A_fit, tau_fit, C_fit = params
        # Calculate the fitted curve using the fitted parameters
        v_fit = exponential_relaxation(df_["Segment Time, S"], A_fit, tau_fit, C_fit)
        mse = np.mean((v_fit - df_["Voltage, V"]) ** 2)
        if "C/10" in seg_descr:
            mse_Cb10 = mse
        elif "C/5" in seg_descr:
            mse_Cb5 = mse
        elif "C/3" in seg_descr:
            mse_Cb3 = mse
        elif "P/3" in seg_descr:
            mse_Pb3 = mse
        elif "C/2" in seg_descr:
            mse_Cb2 = mse
        elif "1C" in seg_descr:
            mse_1C = mse

        max_error = np.max((v_fit - df_["Voltage, V"]) ** 2)
        if "C/10" in seg_descr:
            max_error_Cb10 = max_error
        elif "C/5" in seg_descr:
            max_error_Cb5 = max_error
        elif "C/3" in seg_descr:
            max_error_Cb3 = max_error
        elif "P/3" in seg_descr:
            max_error_Pb3 = max_error
        elif "C/2" in seg_descr:
            max_error_Cb2 = max_error
        elif "1C" in seg_descr:
            max_error_1C = max_error

    # modifiers by cell type
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
    
    # instantiate container or concatenate new value
    if key == keys[0]:
        temperature = temp
        targets_soh = capacity["discharge capacity"].to_numpy().reshape(1, 6)
        cd_throughput = us06_charge_throughput
        cs_efficiency = fcr_efficiency
        post_Cb10_charge_relax_mse = mse_Cb10
        post_Cb5_charge_relax_mse = mse_Cb5
        post_Cb3_charge_relax_mse = mse_Cb3
        post_Pb3_charge_relax_mse = mse_Pb3
        post_Cb2_charge_relax_mse = mse_Cb2
        post_1C_charge_relax_mse = mse_1C
        post_Cb10_charge_relax_max_error = max_error_Cb10
        post_Cb5_charge_relax_max_error = max_error_Cb5
        post_Cb3_charge_relax_max_error = max_error_Cb3
        post_Pb3_charge_relax_max_error = max_error_Pb3
        post_Cb2_charge_relax_max_error = max_error_Cb2
        post_1C_charge_relax_max_error = max_error_1C
        cell_id = cell_id_prefix + cell_id_num
        measurement_id = i
    else:
        _temperature = temp
        _targets_soh = capacity["discharge capacity"].to_numpy().reshape(1, 6)
        _cd_throughput = us06_charge_throughput
        _cs_efficiency = fcr_efficiency
        _cell_id = cell_id_prefix + cell_id_num

        temperature = np.row_stack((temperature, _temperature))
        targets_soh = np.row_stack((targets_soh, _targets_soh))
        cd_throughput = np.row_stack((cd_throughput, _cd_throughput))
        cs_efficiency = np.row_stack((cs_efficiency, _cs_efficiency))
        post_Cb10_charge_relax_mse = np.row_stack(
            (post_Cb10_charge_relax_mse, mse_Cb10)
        )
        post_Cb5_charge_relax_mse = np.row_stack((post_Cb5_charge_relax_mse, mse_Cb5))
        post_Cb3_charge_relax_mse = np.row_stack((post_Cb3_charge_relax_mse, mse_Cb3))
        post_Pb3_charge_relax_mse = np.row_stack((post_Pb3_charge_relax_mse, mse_Pb3))
        post_Cb2_charge_relax_mse = np.row_stack((post_Cb2_charge_relax_mse, mse_Cb2))
        post_1C_charge_relax_mse = np.row_stack((post_1C_charge_relax_mse, mse_1C))
        post_Cb10_charge_relax_max_error = np.row_stack(
            (post_Cb10_charge_relax_max_error, max_error_Cb10)
        )
        post_Cb5_charge_relax_max_error = np.row_stack(
            (post_Cb5_charge_relax_max_error, max_error_Cb5)
        )
        post_Cb3_charge_relax_max_error = np.row_stack(
            (post_Cb3_charge_relax_max_error, max_error_Cb3)
        )
        post_Pb3_charge_relax_max_error = np.row_stack(
            (post_Pb3_charge_relax_max_error, max_error_Pb3)
        )
        post_Cb2_charge_relax_max_error = np.row_stack(
            (post_Cb2_charge_relax_max_error, max_error_Cb2)
        )
        post_1C_charge_relax_max_error = np.row_stack(
            (post_1C_charge_relax_max_error, max_error_1C)
        )

        cell_id = np.row_stack((cell_id, _cell_id))
        measurement_id = np.row_stack((measurement_id, i))
    del df

targets_soh = np.hstack(
    (
        measurement_id,
        cell_id,
        temperature,
        targets_soh,
        cd_throughput,
        cs_efficiency,
        post_Cb10_charge_relax_mse,
        post_Cb5_charge_relax_mse,
        post_Cb3_charge_relax_mse,
        post_Pb3_charge_relax_mse,
        post_Cb2_charge_relax_mse,
        post_1C_charge_relax_mse,
        post_Cb10_charge_relax_max_error,
        post_Cb5_charge_relax_max_error,
        post_Cb3_charge_relax_max_error,
        post_Pb3_charge_relax_max_error,
        post_Cb2_charge_relax_max_error,
        post_1C_charge_relax_max_error,
    )
)
col = [
    "measurement_id",
    "cell_id",
    "temperature_ambient",
    "C/10 discharge capacity",
    "C/5 discharge capacity",
    "C/3 discharge capacity",
    "C/2 discharge capacity",
    "1C discharge capacity",
    "P/3 discharge capacity",
    "Charge depleting cycle charge throughput",
    "Charge sustaining cycle charge efficiency",
    "Post C/10 charge relaxation fit MSE",
    "Post C/5 charge relaxation fit MSE",
    "Post C/3 charge relaxation fit MSE",
    "Post P/3 charge relaxation fit MSE",
    "Post C/2 charge relaxation fit MSE",
    "Post 1C charge relaxation fit MSE",
    "Post C/10 charge relaxation fit Max Error",
    "Post C/5 charge relaxation fit Max Error",
    "Post C/3 charge relaxation fit Max Error",
    "Post P/3 charge relaxation fit Max Error",
    "Post C/2 charge relaxation fit Max Error",
    "Post 1C charge relaxation fit Max Error",
]
targets_soh = pd.DataFrame(targets_soh, columns=col)
targets_soh.to_csv("data/targets_soh.csv")

hdf.close()
