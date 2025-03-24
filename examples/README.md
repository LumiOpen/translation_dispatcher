### Example scripts

## Inference

Example age of dispatcher server.

This is an implementation of a simple command line inference utility.  Simply
prepare your json, edit the sbatch config if necessary, and launch the script.

```bash
# download some data.
# we'll put it in the .messages[0].content format like many chat prompts are,
# though this is not at all necessary, you could just update the PROMPT_PATH in
# the sbatch config to be ".prompt", but I wanted to use a more complicate
# PROMPT_PATH for the demo.
python -c "import json, datasets; [(json.dump({'messages': [{'role': 'user', 'content': item['prompt']}]}, f), f.write('\n')) for item in datasets.load_dataset('argilla/prompt-collective')['train'] for f in [open('input.jsonl', 'a')]]"

# review settings in lumi_launch.sh and launch
sbatch lumi_launch.sh
```

The script will inference all the prompts.  If the job exits before the work is
done, requeuing the job will cause it to continue from where it left off.

The job will exit once all work is complete.


# OOM errors

If you get OOM killed while VLLM is loading and you are using a converted
checkpoint from megatron, check that the checkpoint is sharded into multiple
pytorch files the way huggingface prefers it, rather than a single pytorch.bin.

LLM will use a lot of memory if there is only one large file, and if you are
loading multiple copies simlutaneously you may run into issues.

You can cause huggingface to reshard the model using a code snippet like this.

```python
from transformers import AutoModelForCausalLM

model_path="/scratch/project_xxx...."
save_path="./newdir"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
model.save_pretrained(save_path)
# you'll probably want to copy the tokenizer over too.
```


# Huggingface timeouts

If you are launching a large number of workers, you may be more likely to run
into timeouts checking huggingface for updated model versions.  If you have
already downlaoded the model, You can avoid this by directly referencing the
huggingface cached version fo the model, which you can find like this:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="meta-llama/Llama-2-7b-hf", filename="config.json")
print("Cache path:", path)
```
