# Translation Dispatcher

This codebase is based on the [dispatcher codebase](https://github.com/LumiOpen/dispatcher). This is desgined for large translation jobs and uses vLLM for inference.

### Preprocessing

(1) The preprocessing script splits a text sample into lines (anything separated by "\n"). 

(2) It also removes code blocks (anything inside ```) so that they will not be passed to the translator.

(3) The few-shot translation prompt is prepended to each line that needs translation. Few-shot prompts will differ by target lanuage. Indicate the target language in `trg_lang`.

(4) Finally a sample_id and line_id will be added to each entry in order to assemble the translated text correctly during postprocessing.

**How to use the script**

```
python preprocess.py \
        --input_file <sft_dataset.jsonl> \
        --translation_input_file <filename_for_translation_input.jsonl> \
        --preprocessed_file <filename_for_postprocessing_input.jsonl> \
        --trg_lang <three_letter_lang_code> \
        --prompt_format <prompt_format> \
```

**Example**

```
python preprocess.py \
        --input_file oasst2.jsonl \
        --translation_input_file oasst2_translation_input_fin.jsonl \
        --preprocessed_file oasst2_preprocessed_fin.jsonl \
        --trg_lang fin \
        --prompt_format user_assistant \
```

### Inference

This is the actual translation job where the lines to be translated are passed to the dispatcher. 

Use `launch_inference.sh` to launch the job. Set INPUT_FILE to the name of the translation input file from the preprocessing step. Set OUTPUT_FILE to the filename you want for the translation output.


### Postprocessing

Postprocessing will assemble the translated lines from the same sample and combine them with any lines that were left untranslated (i.e. code blocks). The output will be jsonl in the SFT format.

**How to use**

```
python postprocess.py \
        --translation_output_file <translation_output.jsonl>  \
        --complete_preprocessed_file <filename_for_postprocessing_input.jsonl>  \
        --final_output_file <translated_sft.jsonl> 

```

**Example use**

```
python postprocess.py \
        --translation_output_file oasst2_translation_output_fin.jsonl  \
        --complete_preprocessed_file oasst2_preprocessed_fin.jsonl  \
        --final_output_file oasst2_translated_fin.jsonl 
```
