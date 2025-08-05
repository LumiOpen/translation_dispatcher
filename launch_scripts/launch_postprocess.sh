#!/bin/bash
#SBATCH --job-name=postproc  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch/2.5

python postprocess.py \
        --translation_output_file $translate_output_file  \
        --complete_preprocessed_file $preprocessed_file  \
        --final_output_file $final_output_file \
        --dataset_type $dataset_type \  
