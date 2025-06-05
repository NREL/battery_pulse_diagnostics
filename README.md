# Battery Pulse Diagnostics

Using rapid DC pulse sequences to predict SOC, discharge capacity, and safety metrics measured from multiple different commercially produced lithium-ion batteries.

This analysis uses electrochemical, X-ray CT, and physical characterization to collect various features (DC pulse sequences) and targets (capacity, drive cycle performance, state-of-charge, safety related metrics) to train machine learning models, quantifying the ability for DC pulse sequences to be used to estimate the state of commercial Li-ion batteries. 4 different types of commercial lithium-ion batteries were tested after lab- and field-aging, with electrochemical measurements recorded at 3 operating temperatures.

![Electrochemical characterization](/assets/echem.jpg)

![Data set overview](/assets/dataset.jpg)

## Setup

### Environment
```
$ conda create env -f environment.yml
$ conda activate battery_pulse_diagnostics
```

NOTE: May need to run $ pip install --upgrade pandas "dask[complete]" $ for tsfresh to work.

## Data preprocessing
The processed and raw data can be downloaded from Zenodo: [https://doi.org/10.5281/zenodo.14597394](https://doi.org/10.5281/zenodo.14597394).

Raw electrochemical characterization test data was processed using the following scripts:
1. scripts/01_process_data.py (Raw files processed by this script are not included; this script generates the 'data_raw.h5' data base used by further processing scripts)
2. scripts/02_extract_pulses.py
3. scripts/03_extract_targets.py
4. scripts/04_extract_dcir_all_pulses.py
5. scripts/04_extract_dcir.py (Only returns DCIR at 50% SOC for data exploration, versus the prior script, which returns DCIR from all HPPC-type pulses)
6. scripts/05_process_data_for_ml.py

    These files only need to be run once, and they generate:
    * data/
        * data_raw.h5
        * data_for_ml.h5
        * data_for_ml_extracted_features.h5
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
        * features_dcir_no_interpolation.csv

Another approach studied was extracting timeseries features using the `tsfresh` feature library, documented in `scripts/06_run_tsfeature_extraction.py`, which generates the 'data/data_for_ml_extracted_features.h5' file. This was found to be computationally expensive and not improve predictive performance compared to just using the raw data.

## Model fitting
There are two main scripts for model fitting. `run_kfoldcv.py` runs KFold cross-validation and is set up to compare multiple models including PLSR, XGBoost, and several neural network architectures. KFold CV is used due to the computational cost of each model run, in particular for the neural network models. `run_bootstrap_xgboost.py` runs an XGBoost model with repeated random train/test sampling, as the lower complexity of XGBoost allows for more model runs. Results are saved in the 'results/' directory and code visualizing the results is reported in the 'notebooks/' directory.

## Authors
Code development: Paul Gasper, Nina Prakash
Data acquisition: Paul Gasper, Bryce Knutson, Thomas Bethel, Amariah Condon
Analysis: Paul Gasper, Nina Prakash, Bryce Knutson, Peter Attia
