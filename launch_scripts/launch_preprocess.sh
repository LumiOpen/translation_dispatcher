#!/bin/bash
#SBATCH --job-name=preproc_${trg}  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=debug       # Partition (queue) name
#SBATCH --ntasks=1              # One task (process)
#SBATCH --cpus-per-task=128     # Number of cores (threads)
#SBATCH --time=00:30:00         # Run time (hh:mm:ss)
#SBATCH --account=project_462000353  # Project for billing


module use /appl/local/csc/modulefiles/
module load pytorch/2.5

echo "Preprocessing $input_file"
python preprocess.py \
        --input_file $input_file \
        --translation_input_file $translate_input_file \
        --preprocessed_file $preprocessed_file \
        --trg_lang $lang \
        --prompt_format user_assistant \
        --n_shot 8 \
        --dataset_type $dataset_type \
        --roles_to_translate user \


