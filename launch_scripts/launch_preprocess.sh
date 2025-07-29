#!/bin/bash
#SBATCH --job-name=preproc  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch/2.5


python preprocess.py \
        --input_file /scratch/project_462000353/posttraining_data/SFTTrainer_format/eng/tulu-3-sft-mixture-rip/llama-70B-scored-with-orig-best/train.jsonl \
        --translation_input_file tulu3-sft-700k_translation_input_fin.jsonl \
        --preprocessed_file tulu3-sft-700k_preprocessed_fin.jsonl \
        --trg_lang fin \
        --prompt_format user_assistant \
        --n_shot 8 \
        --dataset_type sft \
        --roles_to_translate user \



