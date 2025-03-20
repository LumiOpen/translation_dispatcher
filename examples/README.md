Example usage

```bash
# download some data.  we'll put it in the .messages[0].content format like many chat prompts are.
python -c "import json, datasets; [(json.dump({'messages': [{'role': 'user', 'content': item['instruction']}]}, f), f.write('\n')) for item in datasets.load_dataset('argilla/10Kprompts-mini')['train'] for f in [open('input.jsonl', 'a')]]"

# review settings in lumi_launch.sh 

sbatch lumi_launch.sh
```
