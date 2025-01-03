# Battery Pulse Diagnostics

Using rapid DC pulse sequences to predict SOC, discharge capacity, and safety metrics measured from multiple different commercially produced lithium-ion batteries.

## Setup

### Environment
```
$ conda create env -f environment.yml
$ conda activate battery_pulse_diagnostics
```

NOTE: May need to run $ pip install --upgrade pandas "dask[complete]" $ for tsfresh to work.

## Data preprocessing
1. Download CSV files from ...
2. In this order, run:
    scripts/01_process_data.py
    scripts/02_extract_pulses.py
    scripts/03_extract_targets.py
    scripts/04_extract_dcir_all_pulses.py
    scripts/04_extract_dcir.py
    scripts/05_process_data_for_ml.py

    These files only need to be run once, and they generate:
    * data_raw.hdf5
    * data_for_ml.hdf5

    * data/
        * features_pulse_hppc.csv
        * features_pulse_rapid.csv
        * features_pulse_psrp1.csv
        * features_pulse_psrp1_Cb2.csv
        * features_pulse_psrp1_1C.csv
        * features_pulse_psrp2_charging.csv
        * features_pulse_psrp2_discharging.csv
        * features_pulse_psrp2_Cb2.csv
        * features_pulse_psrp2_1C.csv
        * targets_soh.csv
        * features_dcir.csv

## Model fit
There are two main scripts for model fitting. `run_kfoldcv.py` runs KFold cross-validation and is set up to compare multiple models including PLSR, XGBoost, and several neural network architectures. KFold CV is used due to the computational cost of each model run, in particular for the neural network models. `run_bootstrap_xgboost.py` runs an XGBoost model with repeated random train/test sampling, as the lower complexity of XGBoost allows for more model runs. Results are saved and visualized in the notebooks/ folder.
