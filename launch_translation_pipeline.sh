# Ensure at least 4 arguments are provided
if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <model_path> <input_file> <trg_lang> <dataset_type>"
  exit 1
fi

# Model to use for translation
model_name="$1"
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
final_output_filename=final_output_${trg_lang}_$filename
echo "preprocessed_filename: $preprocessed_filename"
echo "translate_input_filename: $translate_input_filename"
echo "translate_output_filename: $translate_output_filename"
echo "final_output_filename: $final_output_filename"


echo "-----------------------------"
echo "Running translation pipeline"
echo "-----------------------------"
echo "| PREPROCESSING "
preproc_job_id=$(sbatch --export=input_file=$input_file,translate_input_file=$translate_input_filename,preprocessed_file=$preprocessed_filename,lang=$trg_lang,dataset_type=$dataset_type launch_scripts/launch_preprocess.sh | awk '{print $4}')
echo "|--> Submitted preprocessing job with ID: $preproc_job_id"
echo "| TRANSLATION"
translate_job_id=$(sbatch --dependency=afterok:$preproc_job_id --export=ALL,input=$translate_input_filename,output=$translate_output_filename,model=$model_name launch_scripts/launch_inference.sh | awk '{print $4}')
echo "|--> Submitted translation job with ID: $translate_job_id"
echo "| POSTPROCESSING"
postproc_job_id=$(sbatch --dependency=afterok:$translate_job_id --export=ALL,output=$translate_output_filename,preprocessed=$preprocessed_filename,final=$final_output_filename,lang=$trg_lang launch_scripts/launch_postprocess.sh | awk '{print $4}')
echo "|--> Submitted postprocessing job with ID: $postproc_job_id"
echo "-----------------------------------"
echo "Translation pipeline jobs SUBMITTED "
echo "-----------------------------------"