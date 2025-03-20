#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1
#SBATCH --partition=standard-g
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000615
#SBATCH --mem=480G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


###
# configure the following.

# If you update GPUS_PER_TASK you must also update SBATCH --ntasks-per-node to
# the correct number of tasks per node. basically, int(8 / GPUS_PER_TASK)

INPUT_FILE=tulu3-sft-dedup-700k_translation_input.jsonl
OUTPUT_FILE=tulu3-sft-dedup-700k_translation_output.jsonl

# jq-like path string to find the prompt within the jsonl row.
PROMPT_PATH='content'
# PROMPT_PATH='.messages[0].content'

# generation parameters
BATCH_SIZE=64       # number of prompts in a batch
NUM_GENERATIONS=1 # generations per prompt

# sampling params
# MIN_P=0.05
# TOP_P=1.00
TEMPERATURE=0

# MODEL=meta-llama/Llama-3.1-8B-Instruct
# MODEL=LumiOpen/Viking-7B
MODEL=/scratch/project_462000444/zosaelai2/models/Poro-34B-hf

MAX_MODEL_LEN=2048 # let's cut off here
MAX_TOKENS=1024     # max tokens to gneerate
GPUS_PER_TASK=2    # enough for the model and large batch size
#
# BE SURE TO UPDATE SBATCH --ntasks-per-node to work with GPUS_PER_TASK
#

# end configuration
###


export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

mkdir -p logs output pythonuserbase
export PYTHONUSERBASE=./pythonuserbase
export PYTHONPATH="/scratch/project_462000444/zosaelai2/.dispatcher_venv/lib/python3.10/site-packages"

module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_462000444/zosaelai2/.dispatcher_venv/bin/activate 

export HF_HOME="/scratch/project_462000444/cache"

# pip install git+https://github.com/LumiOpen/dispatcher.git
#pip install --upgrade transformers vllm
#source /scratch/project_462000353/jburdge/venv/bin/activate

python -m dispatcher.server \
    --infile $INPUT_FILE \
    --outfile $OUTPUT_FILE \
    --host 0.0.0.0 \
    --port ${DISPATCHER_PORT} &

sleep 10

srun -l bash -c '
    # Compute the starting GPU index for this task.
    # SLURM_LOCALID is the index of the task on this node.
    start_gpu=$(( SLURM_LOCALID * '"$GPUS_PER_TASK"' ))
    GPU_IDS=""
    for (( i=0; i < '"$GPUS_PER_TASK"'; i++ )); do
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="$(( start_gpu + i ))"
        else
            GPU_IDS="${GPU_IDS},$(( start_gpu + i ))"
        fi
    done
    export CUDA_VISIBLE_DEVICES=$GPU_IDS

    # Set ports uniquely per task (to avoid collisions)
    export MASTER_PORT=$(( 7000 + SLURM_LOCALID ))
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch
    export PYTHONUSERBASE=./pythonuserbase
    python inference.py \
        --batch_size '"$BATCH_SIZE"' \
        --dispatcher_server ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --prompt_path "'"$PROMPT_PATH"'" \
        --num_generations '"$NUM_GENERATIONS"' \
        --max_model_len '"$MAX_MODEL_LEN"' \
        --max_tokens '"$MAX_TOKENS"' \
        --temperature '"$TEMPERATURE"' \
        --tensor_parallel_size '"$GPUS_PER_TASK"' \
        --model_path '"$MODEL"'
'

