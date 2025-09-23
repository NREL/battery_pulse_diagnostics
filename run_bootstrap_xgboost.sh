#!/bin/bash
#SBATCH --job-name=run1
#SBATCH --nodes=1
#SBATCH --account=mlbatt
#SBATCH --time=1:00:00
#SBATCH --output=count_samples_%j.out
#SBATCH --error=count_samples_%j.err
#SBATCH --mail-user=Umme.Nur.Habiba@nrel.gov
#SBATCH --mail-type=ALL
 
# Load required modules (adjust as needed for your system)
module load conda
conda activate /projects/mlbatt/etenney/conda_env/battery_pulse_diagnostics
 
# Change to the directory containing the script
cd /projects/mlbatt/etenney/battery_pulse_diagnostics
 
# Run the job                                                          
srun python run_bootstrap_xgboost.py
 