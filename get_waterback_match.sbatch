#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=precisionhealth_owned1
#SBATCH --partition=precisionhealth

python get_image_hash.py --which_back 'water'