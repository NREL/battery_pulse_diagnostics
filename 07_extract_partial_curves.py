import pandas as pd
import numpy as np
# from pathlib import Path
# import sys
# sys.path.append("../../")
import utils.data_utils as utils
import re

import sys
sys.path.append("../")

hdf = pd.HDFStore("data/data_raw_with_boundary_conditions.h5", mode="r")
keys = hdf.keys()

# cell_types = ["/A/", "/B/", "/C/", "/D/"]
# cell_letter = ["A", "B", "C", "D"]

# for cell_type, letter in zip(cell_types, cell_letter):
print("Processing...")
collated_data = []
# keys = [key for key in keys_all if key.startswith(cell_type)]

print("processing charge depleting and sustaining cycles...")

for i, key in enumerate(keys):
    continue
    # if key == '/D/data_240624_ReCellML_A123_Cell07_charac_30C' or \
    #     key == '/D/data_240626_ReCellML_A123_Cell10_charac_30C' or \
    #     key == '/D/data_240624_ReCellML_A123_Cell15_charac_30C':
    #     continue
    # if not key.startswith(cell_type):
    #     continue
    

    print(key)
    df = hdf.get(key)
    # cell id
    cell_id_num = re.search(r"[Cc]ell\d\d", key).group()[-2:]
    # modifiers by cell type
    if "_A_" in key:
        cell_id_prefix = "A_"
        nominal_capacity = 64
        charge_depleting_pattern_length = 599 # seconds
    elif "Leaf" in key:
        cell_id_prefix = "B_"
        nominal_capacity = 66
        charge_depleting_pattern_length = 599 # seconds
    elif "_C_" in key:
        nominal_capacity = 26
        charge_depleting_pattern_length = 600 # seconds
        if "_TX" in key:
            cell_id_prefix = "C_TX_"
        else:
            cell_id_prefix = "C_"
    elif "A123" in key:
        nominal_capacity = 2.3
        cell_id_prefix = "D_"
        charge_depleting_pattern_length = 600 # seconds

    

    _df_charge_depleting = utils.get_charge_depleting_cycle(df, charge_depleting_pattern_length) #, nominal_capacity)
    _df_charge_sustaining = utils.get_charge_sustaining_cycle(df, cell_id_prefix) #, nominal_capacity)
    # _df_rate_test_C2 = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "Rate test C/2 cycle charge CC", 900) #, nominal_capacity)
    # _df_rate_test_1C = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "Rate test 1C cycle charge CC", 480) #, nominal_capacity)
    # _df_psrp_1_C2 = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "PsRP 1 C/2 diagnostic cycle charge", 900) #, nominal_capacity)
    # _df_psrp_1_1C = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "PsRP 1 1C diagnostic cycle charge", 480) #, nominal_capacity)
    # _df_psrp_2_C2 = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "PsRP 2 C/2 diagnostic cycle charge", 900) #, nominal_capacity)
    # _df_psrp_2_1C = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, "PsRP 2 1C diagnostic cycle charge", 480) #, nominal_capacity)
    # _df_charge_sustaining_time_variable = utils.get_charge_sustaining_cycle_time_variable(df, cell_id_prefix) #, nominal_capacity)
    # _df_psrp_2_C2_time_variable = utils.get_charge_cycle_time_variable(df, cell_id_prefix, cell_id_num, "PsRP 2 C/2 diagnostic cycle charge") #, nominal_capacity)  

    data = [
        _df_charge_depleting,
        _df_charge_sustaining,
        # _df_rate_test_C2,
        # _df_rate_test_1C,
        # _df_psrp_1_C2,
        # _df_psrp_1_1C,
        # _df_psrp_2_C2,
        # _df_psrp_2_1C,
        # _df_charge_sustaining_time_variable,
        # _df_psrp_2_C2_time_variable,
    ]

    # attach measurement meta data so we can link to SOH values
    cell_id = cell_id_prefix + cell_id_num
    measurement_id = i
    for i, _df in enumerate(data):
        if _df is None:
            continue
        new_df = _df.copy()
        new_df["measurement_id"] = measurement_id
        new_df["cell_id"] = cell_id
        if key == keys[0]:
            collated_data.append(new_df)
        else:
            # print(i)
            collated_data[i] = pd.concat([collated_data[i], new_df])

    # _df_charge_depleting["measurement_id"] = measurement_id
    # _df_charge_sustaining["measurement_id"] = measurement_id

    # _df_charge_depleting["cell_id"] = cell_id
    # _df_charge_sustaining["cell_id"] = cell_id

    # #instantiate or append to container
    # if key == keys[0]:
    #     df_charge_depleting = _df_charge_depleting
    #     df_charge_sustaining = _df_charge_sustaining # assuming the first key is not none
    # else:
    #     df_charge_depleting = pd.concat([df_charge_depleting, _df_charge_depleting])
    #     df_charge_sustaining = pd.concat([df_charge_sustaining, _df_charge_sustaining])
    
    del df

#save to file
# print("Saving data...")
# collated_data[0].to_csv("data/features_partial_charge_v2_depleting_cycles.csv") 
# collated_data[1].to_csv("data/features_partial_charge_v2_sustaining_cycles.csv")  

# del collated_data
print("processing rate test and psrp charge cycles...")

data_labels = [
    ("Rate test C/2 cycle charge CC", 900, "data/features_partial_charge_v2_rate_test_C2_charge_cc.csv"),
    ("Rate test 1C cycle charge CC", 480, "data/features_partial_charge_v2_rate_test_1C_charge_cc.csv"),
    ("PsRP 1 C/2 diagnostic cycle charge", 900, "data/features_partial_charge_v2_psrp_1_C2_charge.csv"),
    ("PsRP 1 1C diagnostic cycle charge", 480, "data/features_partial_charge_v2_psrp_1_1C_charge.csv"),
    ("PsRP 2 C/2 diagnostic cycle charge", 900, "data/features_partial_charge_v2_psrp_2_C2_charge.csv"),
    ("PsRP 2 1C diagnostic cycle charge", 480, "data/features_partial_charge_v2_psrp_2_1C_charge.csv"),
]

for label, length, filename in data_labels:
    print(f"processing {label}...")
    collated_df = pd.DataFrame()
    for i, key in enumerate(keys):
        print(key)
        df = hdf.get(key)
        # cell id
        cell_id_num = re.search(r"[Cc]ell\d\d", key).group()[-2:]
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

        _df_cycle = utils.get_charge_cycle(df, cell_id_prefix, cell_id_num, label, length) #, nominal_capacity)
        if _df_cycle is None:
            continue

        measurement_id = i
        cell_id = cell_id_prefix + cell_id_num
        new_df = _df_cycle.copy()
        new_df["measurement_id"] = measurement_id
        new_df["cell_id"] = cell_id

        collated_df = pd.concat([collated_df, new_df])

        del df

    collated_df.to_csv(filename)

# collated_data[2].to_csv("data/features_partial_charge_rate_test_C2_charge_cc.csv")
# collated_data[3].to_csv("data/features_partial_charge_rate_test_1C_charge_cc.csv")
# collated_data[4].to_csv("data/features_partial_charge_psrp_1_C2_charge.csv")
# collated_data[5].to_csv("data/features_partial_charge_psrp_1_1C_charge.csv")
# collated_data[6].to_csv("data/features_partial_charge_psrp_2_C2_charge.csv")
# collated_data[7].to_csv("data/features_partial_charge_psrp_2_1C_charge.csv")
# collated_data[8].to_csv("data/features_partial_charge_sustaining_cycle_time_variable.csv")
# collated_data[9].to_csv("data/features_partial_charge_psrp_2_C2_charge_time_variable.csv")
print("Done.")


hdf.close()