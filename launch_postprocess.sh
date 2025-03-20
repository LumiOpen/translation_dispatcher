#!/bin/bash
#SBATCH --job-name=postproc_translation  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000615  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch

trg_lang_3char="fin"

python postprocess.py \
        --translation_output_file oasst2_translation_output.jsonl  \
        --complete_preprocessed_file oasst2_preprocessed.jsonl  \
        --final_output_file oasst2_translated_$trg_lang_3char.jsonl 
