#!/bin/bash
#SBATCH --partition=gpu  # Specify the partition name where GPUs are available
#SBATCH --nodes=1                    # Number of nodes requested
#SBATCH --gres=gpu:1                 # Number of GPUs requested
#SBATCH --time=10:00:00              # Requested wall-clock time (format: hours:minutes:seconds)
#SBATCH --job-name=hw2_v2            # Job name
#SBATCH --output=output_v2.log       # Output file path
#SBATCH --mem=16G

module load GCCcore/11.3.0
module load Python/3.10.4
source ../../../venv/bin/activate

python main.py