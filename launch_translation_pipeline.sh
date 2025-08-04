# Ensure at least 4 arguments are provided
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <model_path> <input_file> <trg_lang> <dataset_type>"
  exit 1
fi

# Model to use for translation
model_path="$1"
model_name=$(basename "$model_path")
echo "model_name: $model_name"

# Path to input file
input_file="$2"
filename=$(basename "$input_file")

# Target language
trg_lang="$3"

# Dataset type (sft or dpo)
dataset_type="$4"

preprocessed_filename=preprocessed_${trg_lang}_$filename
translate_input_filename=translation_input_${trg_lang}_$filename
translate_output_filename=translation_output_${trg_lang}_$filename
echo "preprocessed_filename: $preprocessed_filename"
echo "translate_input_filename: $translate_input_filename"
echo "translate_output_filename: $translate_output_filename"
echo "dataset_type: $dataset_type"


echo "-----------------------------"
echo "Running translation pipeline"
echo "-----------------------------"
echo "| PREPROCESSING "
job_id=$(sbatch --job-name="preproc_$trg_lang" --export=input_file=$input_file,translate_input_file=$translate_input_filename,preprocessed_file=$preprocessed_filename,lang=$trg_lang,dataset_type=$dataset_type launch_preprocess.sh | awk '{print $4}')
echo "|--> Submitted preprocessing job | Job ID: $job_id"
echo "|--> Waiting for preprocessing job to finish"
# Wait until the job is no longer in the queue
while squeue -j $job_id > /dev/null 2>&1 && squeue -j $job_id | grep -q "$job_id"; do
    echo "|--> Preproc job $job_id is still running..."
    sleep 10  # wait 10 seconds before checking again
done
echo "|--> Preproc job $job_id has FINISHED."

echo "| TRANSLATION"
job_id=$(sbatch --job-name="translate_$trg_lang" --export=translate_input_file=$translate_input_filename,translate_output_file=$translate_output_filename,model_name=$model_name launch_inference.sh | awk '{print $4}')
echo "|--> Submitted translation job | Job ID: $job_id"