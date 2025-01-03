import pandas as pd
import numpy as np
import utils.data_utils as utils
import re

import sys
sys.path.append("../")

hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()

for ii, key in enumerate(keys):
    print(key)
    df = hdf.get(key)
    # cell id
    cell_id_num = re.search(r"Cell\d\d", key).group()[-2:]
    # modifiers by cell type
    if "_A_" in key:
        cell_id_prefix = "A_"
        nominal_capacity = 64
    elif "Leaf" in key:
        cell_id_prefix = "B_"
        nominal_capacity = 66
    elif "_C_" in key:
        nominal_capacity = 26
        if "_TX" in key:
            cell_id_prefix = "C_TX_"
        else:
            cell_id_prefix = "C_"
    elif "A123" in key:
        cell_id_prefix = "D_"

    # HPPC pulses
    _df_hppc = utils.get_hppc_pulses(df, nominal_capacity)
    # rapid 2C pulses
    _df_rapid = utils.get_rapid_pulses(df, nominal_capacity)
    # PsRP 1 pulses
    _df_psrp1 = utils.get_psrp1_pulses(df, nominal_capacity)
    # C/2 + PsRP 1 pulses
    _df_psrp1_Cb2 = utils.get_psrp1_Cb2_pulses(df, nominal_capacity)
    # 1C + PsRP 1 pulses
    _df_psrp1_1C = utils.get_psrp1_1C_pulses(df, nominal_capacity)
    # PsRP 2 pulses (Charging and discharging)
    _df_psrp2_charging = utils.get_psrp2_pulses_charging(df, nominal_capacity)
    if not key == "/Leaf/data_230719_ReCellML_Nissan_Leaf_Cell05_charac_45oC":
        _df_psrp2_discharging = utils.get_psrp2_pulses_discharging(df, nominal_capacity)
    # C/2 + PsRP 2 pulses
    _df_psrp2_Cb2 = utils.get_psrp2_Cb2_pulses(df, nominal_capacity)
    # 1C + PsRP 2 pulses
    _df_psrp2_1C = utils.get_psrp2_1C_pulses(df, nominal_capacity)

    # attach measurement meta data so we can link to SOH values
    cell_id = cell_id_prefix + cell_id_num
    measurement_id = ii
    _df_hppc["measurement_id"] = measurement_id
    _df_rapid["measurement_id"] = measurement_id
    _df_psrp1["measurement_id"] = measurement_id
    _df_psrp1_Cb2["measurement_id"] = measurement_id
    _df_psrp1_1C["measurement_id"] = measurement_id
    _df_psrp2_charging["measurement_id"] = measurement_id
    _df_psrp2_discharging["measurement_id"] = measurement_id
    _df_psrp2_Cb2["measurement_id"] = measurement_id
    _df_psrp2_1C["measurement_id"] = measurement_id
    _df_hppc["cell_id"] = cell_id
    _df_rapid["cell_id"] = cell_id
    _df_psrp1["cell_id"] = cell_id
    _df_psrp1_Cb2["cell_id"] = cell_id
    _df_psrp1_1C["cell_id"] = cell_id
    _df_psrp2_charging["cell_id"] = cell_id
    _df_psrp2_discharging["cell_id"] = cell_id
    _df_psrp2_Cb2["cell_id"] = cell_id
    _df_psrp2_1C["cell_id"] = cell_id

    # instantiate container or concatenate new value
    if key == keys[0]:
        df_hppc = _df_hppc
        df_rapid = _df_rapid
        df_psrp1 = _df_psrp1
        df_psrp1_Cb2 = _df_psrp1_Cb2
        df_psrp1_1C = _df_psrp1_1C
        df_psrp2_charging = _df_psrp2_charging
        df_psrp2_discharging = _df_psrp2_discharging
        df_psrp2_Cb2 = _df_psrp2_Cb2
        df_psrp2_1C = _df_psrp2_1C
    else:
        df_hppc = pd.concat([df_hppc, _df_hppc])
        df_rapid = pd.concat([df_rapid, _df_rapid])
        df_psrp1 = pd.concat([df_psrp1, _df_psrp1])
        df_psrp1_Cb2 = pd.concat([df_psrp1_Cb2, _df_psrp1_Cb2])
        df_psrp1_1C = pd.concat([df_psrp1_1C, _df_psrp1_1C])
        df_psrp2_charging = pd.concat([df_psrp2_charging, _df_psrp2_charging])
        df_psrp2_discharging = pd.concat([df_psrp2_discharging, _df_psrp2_discharging])
        df_psrp2_Cb2 = pd.concat([df_psrp2_Cb2, _df_psrp2_Cb2])
        df_psrp2_1C = pd.concat([df_psrp2_1C, _df_psrp2_1C])

    del df

df_hppc.to_csv("data/features_pulse_hppc.csv")
df_rapid.to_csv("data/features_pulse_rapid.csv")
df_psrp1.to_csv("data/features_pulse_psrp1.csv")
df_psrp1_Cb2.to_csv("data/features_pulse_psrp1_Cb2.csv")
df_psrp1_1C.to_csv("data/features_pulse_psrp1_1C.csv")
df_psrp2_charging.to_csv("data/features_pulse_psrp2_charging.csv")
df_psrp2_discharging.to_csv("data/features_pulse_psrp2_discharging.csv")
df_psrp2_Cb2.to_csv("data/features_pulse_psrp2_Cb2.csv")
df_psrp2_1C.to_csv("data/features_pulse_psrp2_1C.csv")

hdf.close()
