Example usage of dispatcher server.

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
