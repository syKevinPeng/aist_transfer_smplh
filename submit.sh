#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=/fs/nexus-projects/PhysicsFall/transfermodel/slrum-output/%j.out
#SBATCH --error=/fs/nexus-projects/PhysicsFall/transfermodel/slrum-output/%j.err

# Activate conda environment
source /nfshomes/peng2000/scratch/miniforge3/etc/profile.d/conda.sh
conda activate physicsfall

# Run the script
# siyuan: 0(running 3240814),1 (running),2, 3, 4, 5
# Seong: 6,7,8,9
python AIST_transfer.py --batch_num 1