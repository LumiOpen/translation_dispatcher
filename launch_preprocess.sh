#!/bin/bash
#SBATCH --job-name=preproc  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000615  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch

trg_lang_3char="dan"

python preprocess.py \
        --input_file tulu3-sft-dedup-700k.jsonl \
        --translation_input_file tulu3-sft-dedup-700k_translation_input_$trg_lang_3char.jsonl \
        --preprocessed_file tulu3-sft-dedup-700k_preprocessed_$trg_lang_3char.jsonl \
        --trg_lang $trg_lang_3char \
        --prompt_format user_assistant \
