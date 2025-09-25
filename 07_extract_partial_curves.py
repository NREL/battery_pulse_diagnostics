import pandas as pd
import numpy as np
# from pathlib import Path
# import sys
# sys.path.append("../../")
import utils.data_utils as utils
import re

import sys
sys.path.append("../")

hdf = pd.HDFStore("data/data_raw.h5", mode="r")
keys = hdf.keys()

for i, key in enumerate(keys):
    # if key == '/D/data_240624_ReCellML_A123_Cell07_charac_30C' or \
    #     key == '/D/data_240626_ReCellML_A123_Cell10_charac_30C' or \
    #     key == '/D/data_240624_ReCellML_A123_Cell15_charac_30C':
    #     continue
    print(key)
    df = hdf.get(key)
    # cell id
    cell_id_num = re.search(r"[Cc]ell\d\d", key).group()[-2:]
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
        nominal_capacity = 2.3
        cell_id_prefix = "D_"

    _df_charge_depleting = utils.get_charge_depleting_cycle(df) #, nominal_capacity)
    # _df_charge_sustaining = utils.get_charge_sustaining_cycle(df) #, nominal_capacity)

    # attach measurement meta data so we can link to SOH values
    cell_id = cell_id_prefix + cell_id_num
    measurement_id = i
    _df_charge_depleting["measurement_id"] = measurement_id
    # _df_charge_sustaining["measurement_id"] = measurement_id

    _df_charge_depleting["cell_id"] = cell_id
    # _df_charge_sustaining["cell_id"] = cell_id

    #instantiate or append to container
    if key == keys[0]:
        df_charge_depleting = _df_charge_depleting
        # df_charge_sustaining = _df_charge_sustaining
    else:
        df_charge_depleting = pd.concat([df_charge_depleting, _df_charge_depleting])
        # df_charge_sustaining = pd.concat([df_charge_sustaining, _df_charge_sustaining])
    
    del df

#save to file
df_charge_depleting.to_csv("data/features_charge_depleting_cycles.csv") 
# df_charge_sustaining.to_csv("data/features_charge_sustaining_cycles.csv")   

hdf.close()