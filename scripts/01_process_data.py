import pandas as pd
from pathlib import Path
import os
import numpy as np
import scipy as sci

fpath = Path("Data files")
new_fpath = fpath / "processed"
files = os.listdir(fpath)
files = [file for file in files if ".csv" in file]

# Check if the hdf5 database already exists
is_hdf = any(["data/data_raw.h5" in key for key in os.listdir(Path())])

# Segment names and step numbers [first, last]
segments = {
    # Equilibration cycle
    #'Equilibration cycle': [5,10],
    "Equilibration cycle charge CC": [5],  # 0
    "Equilibration cycle charge CV": [6],  # 1
    "Equilibration cycle after charge rest": [7],  # 2
    "Equilibration cycle discharge CC": [8],  # 3
    "Equilibration cycle discharge CV": [9],  # 4
    "Equilibration cycle after discharge rest": [10],  # 5
    # Rate test C/10 cycle
    #'Rate test C/10 cycle': [11,16],
    "Rate test C/10 cycle charge CC": [11],  # 6
    "Rate test C/10 cycle charge CV": [12],
    "Rate test C/10 cycle after charge rest": [13],
    "Rate test C/10 cycle discharge CC": [14],
    "Rate test C/10 cycle discharge CV": [15],
    "Rate test C/10 cycle after discharge rest": [16],  # 11
    # Rate test C/5 cycle
    #'Rate test C/5 cycle': [17,22],
    "Rate test C/5 cycle charge CC": [17],  # 12
    "Rate test C/5 cycle charge CV": [18],
    "Rate test C/5 cycle after charge rest": [19],
    "Rate test C/5 cycle discharge CC": [20],
    "Rate test C/5 cycle discharge CV": [21],
    "Rate test C/5 cycle after discharge rest": [22],  # 17
    # Rate test C/3 cycle
    #'Rate test C/3 cycle': [23,28],
    "Rate test C/3 cycle charge CC": [23],  # 18
    "Rate test C/3 cycle charge CV": [24],
    "Rate test C/3 cycle after charge rest": [25],
    "Rate test C/3 cycle discharge CC": [26],
    "Rate test C/3 cycle discharge CV": [27],
    "Rate test C/3 cycle after discharge rest": [28],  # 23
    # Rate test P/3 cycle
    #'Rate test P/3 cycle': [29,33],
    "Rate test P/3 cycle charge CP": [29],  # 24
    "Rate test P/3 cycle charge CV": [30],
    "Rate test P/3 cycle after charge rest": [31],
    "Rate test P/3 cycle discharge CP": [32],
    "Rate test P/3 cycle discharge CV": [33],  # 28
    # HPPC
    "HPPC": [54, 197],  # 29
    #'HPPC charge rest 1': [54],
    "HPPC charge 2C rapid pulses": [55, 58],
    #'HPPC charge rest 2': [59],
    "HPPC charge C/10 pulses": [60, 62],
    #'HPPC charge rest 3': [63],
    "HPPC charge C/2 pulses": [64, 66],
    #'HPPC charge rest 4': [67],
    "HPPC charge 1C pulses": [68, 70],
    #'HPPC charge rest 5': [71],
    "HPPC charge 2C pulses": [72, 74],
    #'HPPC charge rest 6': [75],
    "HPPC charge PsRP 1": [76, 105],
    #'HPPC charge rest 7': [106],
    "HPPC charge PsRP 2": [107, 123],
    #'HPPC charge rest 8': [124],
    "HPPC charge 10 SOC C/3 charge": [125],
    #'HPPC discharge rest 1': [126],
    "HPPC discharge 2C rapid pulses": [127, 130],
    #'HPPC discharge rest 2': [131],
    "HPPC discharge C/10 pulses": [132, 134],
    #'HPPC discharge rest 3': [135],
    "HPPC discharge C/2 pulses": [136, 138],
    #'HPPC discharge rest 4': [139],
    "HPPC discharge 1C pulses": [140, 142],
    #'HPPC discharge rest 5': [143],
    "HPPC discharge 2C pulses": [144, 146],
    #'HPPC discharge rest 6': [147],
    "HPPC discharge PsRP 1": [148, 177],
    #'HPPC discharge rest 7': [178],
    "HPPC discharge PsRP 2": [179, 193],
    "HPPC discharge 10 SOC C/3 charge": [197],  # 45
    # Drive cycles
    "Charge depleting cycle": [198, 796],  # 46
    "Charge sustaining cycle": [797, 2596],  # 47
    # Rate test C/2 cycle
    #'Rate test C/2 cycle': [2598,2603],
    "Rate test C/2 cycle charge CC": [2598],  # 48
    "Rate test C/2 cycle charge CV": [2599],
    "Rate test C/2 cycle after charge rest": [2600],
    "Rate test C/2 cycle discharge CC": [2601],
    "Rate test C/2 cycle discharge CV": [2602],
    "Rate test C/2 cycle after discharge rest": [2603],  # 53
    # PsRP 1 C/2 diagnostic cycle
    #'PsRP 1 C/2 diagnostic cycle': [2604,2670],
    "PsRP 1 C/2 diagnostic cycle charge 10 SOC CC 1": [2604],  # 54
    "PsRP 1 C/2 diagnostic cycle charge pulses": [2605, 2634],
    "PsRP 1 C/2 diagnostic cycle charge 10 SOC CC 2": [2635],
    "PsRP 1 C/2 diagnostic cycle charge CV": [2636],
    "PsRP 1 C/2 diagnostic cycle charge rest": [2637],
    "PsRP 1 C/2 diagnostic cycle discharge 10 SOC CC 1": [2638],
    "PsRP 1 C/2 diagnostic cycle discharge pulses": [2639, 2668],
    "PsRP 1 C/2 diagnostic cycle discharge 10 SOC CC 2": [2669],
    "PsRP 1 C/2 diagnostic cycle discharge CV": [2670],  # 62
    # PsRP 2 C/2 diagnostic cycle
    #'PsRP 2 C/2 diagnostic cycle': [2672,2708],
    "PsRP 2 C/2 diagnostic cycle charge 10 SOC CC 1": [2672],  # 63
    "PsRP 2 C/2 diagnostic cycle charge pulses": [2673, 2687],
    "PsRP 2 C/2 diagnostic cycle charge 10 SOC CC 2": [2688],
    "PsRP 2 C/2 diagnostic cycle charge CV": [2689],
    "PsRP 2 C/2 diagnostic cycle charge rest": [2690],
    "PsRP 2 C/2 diagnostic cycle discharge 10 SOC CC 1": [2691],
    "PsRP 2 C/2 diagnostic cycle discharge pulses": [2692, 2706],
    "PsRP 2 C/2 diagnostic cycle discharge 10 SOC CC 2": [2707],
    "PsRP 2 C/2 diagnostic cycle discharge CV": [2708],  # 71
    # Rate test 1C cycle
    #'Rate test 1C cycle': [2710,2715],
    "Rate test 1C cycle charge CC": [2710],  # 72
    "Rate test 1C cycle charge CV": [2711],
    "Rate test 1C cycle after charge rest": [2712],
    "Rate test 1C cycle discharge CC": [2713],
    "Rate test 1C cycle discharge CV": [2714],
    "Rate test 1C cycle after discharge rest": [2715],  # 77
    # PsRP 1 1C diagnostic cycle
    #'PsRP 1 1C diagnostic cycle': [2716,2782],
    "PsRP 1 1C diagnostic cycle charge 10 SOC CC 1": [2716],  # 78
    "PsRP 1 1C diagnostic cycle charge pulses": [2717, 2746],
    "PsRP 1 1C diagnostic cycle charge 10 SOC CC 2": [2747],
    "PsRP 1 1C diagnostic cycle charge CV": [2748],
    "PsRP 1 1C diagnostic cycle charge rest": [2749],
    "PsRP 1 1C diagnostic cycle discharge 10 SOC CC 1": [2750],
    "PsRP 1 1C diagnostic cycle discharge pulses": [2751, 2780],
    "PsRP 1 1C diagnostic cycle discharge 10 SOC CC 2": [2781],
    "PsRP 1 1C diagnostic cycle discharge CV": [2782],  # 86
    # PsRP 2 1C diagnostic cycle
    #'PsRP 2 1C diagnostic cycle': [2784,2824],
    "PsRP 2 1C diagnostic cycle charge 10 SOC CC 1": [2784],  # 87
    "PsRP 2 1C diagnostic cycle charge pulses": [2785, 2801],
    "PsRP 2 1C diagnostic cycle charge 10 SOC CC 2": [2802],
    "PsRP 2 1C diagnostic cycle charge CV": [2803],
    "PsRP 2 1C diagnostic cycle charge rest": [2804],
    "PsRP 2 1C diagnostic cycle discharge 10 SOC CC 1": [2805],
    "PsRP 2 1C diagnostic cycle discharge pulses": [2806, 2822],
    "PsRP 2 1C diagnostic cycle discharge 10 SOC CC 2": [2823],
    "PsRP 2 1C diagnostic cycle discharge CV": [2824],  # 95
}

# SOC stages (charge, discharge)
soc_stages = {
    # Equilibration cycle
    "Equilibration cycle charge": [5, 7],  # 0
    "Equilibration cycle discharge": [8, 10],  # 1
    # Rate test C/10 cycle
    "Rate test C/10 cycle charge": [11, 13],  # 2
    "Rate test C/10 cycle discharge": [14, 16],  # 3
    # Rate test C/5 cycle
    "Rate test C/5 cycle charge": [17, 19],  # 4
    "Rate test C/5 cycle discharge": [20, 22],  # 5
    # Rate test C/3 cycle
    "Rate test C/3 cycle charge": [23, 25],  # 6
    "Rate test C/3 cycle discharge": [26, 28],  # 7
    # Rate test P/3 cycle
    "Rate test P/3 cycle charge CP": [29, 31],  # 8
    "Rate test P/3 cycle discharge CP": [32, 33],  # 9
    # HPPC
    #'HPPC': [54,197],
    "HPPC charge": [54, 125],  # 10
    "HPPC discharge": [126, 197],  # 11
    # Rate test C/2 cycle - C/2 charge capacity used for CS cycle capacity, has to occur before
    "Rate test C/2 cycle charge": [2598, 2600],  # 17
    "Rate test C/2 cycle discharge": [2601, 2603],  # 18
    # Drive cycles
    "CD cycle charge": [36, 39],  # 12
    "CD cycle discharge": [198, 796],  # 13
    "CS cycle charge": [41, 44],  # 14
    "CS cycle discharge": [797, 2596],  # 15
    "post CS cycle discharge": [45, 46],  # 16
    # PsRP 1 C/2 diagnostic cycle
    #'PsRP 1 C/2 diagnostic cycle': [2604,2670],
    "PsRP 1 C/2 diagnostic cycle charge": [2604, 2637],  # 19
    "PsRP 1 C/2 diagnostic cycle discharge": [2638, 2670],  # 20
    # PsRP 2 C/2 diagnostic cycle
    "PsRP 2 C/2 diagnostic cycle charge": [2672, 2690],  # 21
    "PsRP 2 C/2 diagnostic cycle discharge": [2691, 2708],  # 22
    # Rate test 1C cycle
    "Rate test 1C cycle charge": [2710, 2712],  # 23
    "Rate test 1C cycle discharge CC": [2713, 2715],  # 24
    # PsRP 1 1C diagnostic cycle
    "PsRP 1 1C diagnostic cycle charge": [2716, 2749],  # 25
    "PsRP 1 1C diagnostic cycle discharge": [2750, 2782],  # 26
    # PsRP 2 1C diagnostic cycle
    "PsRP 2 1C diagnostic cycle charge": [2784, 2804],  # 27
    "PsRP 2 1C diagnostic cycle discharge": [2805, 2824],  # 28
}

# Iterate through files, processing and saving in HDF5 format
for i, file in enumerate(files):
    print("Processing " + file)
    # Encodings that work for cell type 'A' data (degree sign on temperature column breaks many of them)
    # cp1250-1258, iso8859_2 to 4 and 7 to 10 and 13, koi8_t, kz1048.
    # Seem to be the same for Nissan data (300 A channels, different formatting slightly, but same problem)
    if "LG_JH3" in file:
        df = pd.read_csv(
            fpath / file,
            encoding="cp1250",
            skiprows=range(17),
            skipfooter=1,
            engine="python",
        )
    elif "Nissan_Leaf" in file:
        df = pd.read_csv(fpath / file, encoding="cp1250", skipfooter=1, engine="python")
    elif "FF" in file:
        df = pd.read_csv(
            fpath / file,
            encoding="cp1250",
            skiprows=range(17),
            skipfooter=1,
            engine="python",
        )
    elif "A123" in file:
        df = pd.read_csv(
            fpath / file,
            encoding='cp1250',
            skiprows=range(17), 
            skipfooter=1, 
            engine='python')
    else:
        NameError("Cell type not recognized for identifing file formatting")

    # Remove columns with no data
    vars_to_remove_substring = "U"
    vars_to_remove = [var for var in df.columns if vars_to_remove_substring in var]
    df = df.drop(vars_to_remove, axis=1)

    # Some Leaf files have super noisy data on their 1 mV resolution voltage data channels, remove these columns
    if "Leaf" in file:
        if np.any(np.abs(df["Voltage, V"] - df["Cell Voltage A1, V"]) > 1):
            df = df.drop(
                ["Cell Voltage A1, V", "Cell Voltage A2, V", "Cell Voltage A3, V"],
                axis=1,
            )
            print('    Warning: Dropped "Cell Voltage" columns due to noisy data')

    # 'FF' data without 'chain': charge-depleting drive cycle has one extra step, subtract 1 from all high step numbers for segmenting
    if "FF" in file and not "_chain_" in file:
        mask_fix = df["Step"] >= 797
        df.loc[mask_fix, "Step"] += -1

    # 'FF' data with 'chain' : need to update step numbers for script chain to be consistent with script with subroutines
     # All 'A123' data was also recorded using the script chain
    if ("FF" in file and "_chain_" in file) or "A123" in file:
        is_chain = True
        idx_link_ends = np.where(df["Data Acquisition Flag"] == "Q1")[0]
        idx_link_starts = idx_link_ends + 1
        idx_link_starts = np.concatenate(([0], idx_link_starts))
        idx_link_ends = np.concatenate((idx_link_ends, [df.shape[0]]))
        for i, idx in enumerate(zip(idx_link_starts, idx_link_ends)):
            if i > 0:
                if i == 1:
                    df.loc[range(idx[0], idx[1]), "Step"] += 1
                else:
                    idx_link = range(idx[0], idx[1])
                    df_link = df.loc[idx_link]
                    steps_link = df_link["Step"].to_numpy()
                    if i == 2:
                        # subroutine steps
                        mask_subroutine = steps_link >= 4
                        steps_link[mask_subroutine] += 50
                        # main routine steps
                        steps_link[~mask_subroutine] += 33
                    elif i == 3:
                        # subroutine steps
                        mask_subroutine = steps_link >= 6
                        steps_link[mask_subroutine] += 192
                        # main routine steps
                        steps_link[~mask_subroutine] += 36
                        # fix final US06 drive cycle step
                        steps_link[steps_link == 797] = 796
                    elif i == 4:
                        # subroutine steps
                        mask_subroutine = steps_link >= 6
                        steps_link[mask_subroutine] += 791
                        # main routine steps
                        steps_link[~mask_subroutine] += 41
                    elif i == 5:
                        # subroutine steps
                        mask_subroutine = steps_link >= 4
                        steps_link[mask_subroutine] += 2593
                        # main routine steps
                        steps_link[~mask_subroutine] += 42
                    elif i == 6:
                        # subroutine steps
                        mask_subroutine = steps_link >= 5
                        steps_link[mask_subroutine] += 2704
                        # main routine steps
                        steps_link[~mask_subroutine] += 49
                    df.loc[idx_link, "Step"] = steps_link
    else:
        is_chain = False

    # Add segment descriptions
    for key in segments:
        idx_seg = segments[key]
        if len(idx_seg) == 1:
            idx_seg_start = idx_seg[0]
            idx_seg_end = idx_seg[0] + 1
        else:
            idx_seg_start = idx_seg[0]
            idx_seg_end = idx_seg[1] + 1
        # Mask off all segments in this step number range
        is_seg = np.isin(df["Step"].values, np.arange(idx_seg_start, idx_seg_end))
        # Name these steps
        df.loc[is_seg, "Segment Description"] = key
        # Calculate segment time
        df_seg = df[is_seg].reset_index(drop=True)
        df_seg["Segment Time, S"] = (
            df_seg["Total Time, S"] - df_seg["Total Time, S"].values[0]
        )
        # Check for repeated segments (more than 2 minute gap in the sequence)
        segment_starts = np.argwhere(
            np.diff(df_seg["Segment Time, S"], prepend=0) > 120
        ).squeeze()
        if segment_starts.size > 1:
            segment_starts = np.append([0], segment_starts)
            for i, idx in enumerate(segment_starts):
                # Extract this segment part
                if i == segment_starts.size - 1:
                    df_seg_part = df_seg.loc[idx:].copy().reset_index(drop=True)
                else:
                    df_seg_part = (
                        df_seg.loc[idx : segment_starts[i + 1] - 1]
                        .copy()
                        .reset_index(drop=True)
                    )
                df_seg_part["Segment Time, S"] = (
                    df_seg_part["Total Time, S"]
                    - df_seg_part["Total Time, S"].values[0]
                )
                # Write segement time to this segment part
                is_seg_part = np.isin(df["Total Time, S"], df_seg_part["Total Time, S"])
                df.loc[is_seg_part, "Segment Time, S"] = df_seg_part[
                    "Segment Time, S"
                ].values
        else:
            # Write segement time to this segment part
            df.loc[is_seg, "Segment Time, S"] = df_seg["Segment Time, S"].values
    # Set dtype of new column to string to avoid HDF5 having to pickle the column
    df["Segment Description"] = df["Segment Description"].astype(str)

    # Calculate SOC
    df["SOC"] = np.full((df.shape[0]), np.nan)
    for key in soc_stages:
        # FF_chain files missing a specific step... avoid indexing empty df. Otherwise, run the code
        if not(is_chain and key == 'post CS cycle discharge'):
            idx_seg = soc_stages[key]
            idx_seg_start = idx_seg[0]
            idx_seg_end = idx_seg[1] + 1
            is_seg = np.isin(df["Step"].values, np.arange(idx_seg_start, idx_seg_end))
            df_seg = df[is_seg].reset_index(drop=True)
            df_seg["Segment Time, S"] = (
                df_seg["Total Time, S"] - df_seg["Total Time, S"].values[0]
            )
            amp_seconds = sci.integrate.cumulative_trapezoid(
                df_seg["Current, A"], x=df_seg["Segment Time, S"], initial=0
            )
            if not "CS cycle" in key:
                if " charge" in key:
                    capacity = np.max(amp_seconds)
                    soc = amp_seconds / capacity
                    if key == "Rate test C/2 cycle charge" and not is_chain:
                        # 'FF_chain' files have this step incorrect, use discharge to set C/2 capacity instead
                        cap_ = capacity
                else:
                    capacity = np.min(amp_seconds)
                    if capacity == 0:
                        capacity = np.max(amp_seconds)
                    soc = 1 - (amp_seconds / capacity)
                    if is_chain and key == "Rate test C/2 cycle discharge":
                        # 'FF_chain' files have charge step incorrect, use discharge to set C/2 capacity instead
                        cap_ = capacity
            else:
                if key == "CS cycle charge":
                    soc = amp_seconds / cap_
                    prior_soc = soc[-1]
                elif key == "CS cycle discharge":
                    soc = prior_soc + (amp_seconds / cap_)
                    prior_soc = soc[-1]
                else:  # 'post CS cycle discharge'
                    capacity = np.min(amp_seconds)
                    if capacity == 0:
                        capacity = np.max(amp_seconds)
                    soc = prior_soc - ((amp_seconds / capacity) * prior_soc)
            df.loc[is_seg, "SOC"] = soc

    # Generate key for file
    if "LG_JH3" in file:
        file_key = "/A/data_" + file[:-4].replace('LG_JH3', 'A')
    elif "Nissan_Leaf" in file:
        file_key = "/B/data_" + file[:-4]
    elif "FF" in file:
        file_key = "/C/data_" + file[:-4].replace('FF', 'C')
    elif "A123" in file:
        file_key = "/D/data_" + file[:-4]
    else:
        file_key = file[:-4]

    # Move file to processed folder
    os.rename(fpath / file, new_fpath / file)

    # Write to HDF5 database
    if file == files[0] and not is_hdf:
        df.to_hdf("data/data_raw.h5", key=file_key, mode="w", complevel=9)
    else:
        df.to_hdf("data/data_raw.h5", key=file_key, complevel=9)