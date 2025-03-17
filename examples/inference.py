import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
import argparse
import time

from dispatcher.client import WorkClient
from dispatcher.models import WorkStatus

class Generator:
    def __init__(
        self,
        model_path: str,
        num_generations: int = 1,
        tensor_parallel_size: int = 1,
        max_model_len: int = 16384,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        max_tokens: int = 4096,
    ):
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_tokens,
        )

    def generate_responses(self, prompts: list[str]) -> list[list[str]]:
        """Generate responses for multiple prompts in a batch.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of lists, where each inner list contains the generated responses for a prompt
        """
        chat_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False
            )
            for prompt in prompts
        ]
        
        outputs = self.model.generate(
            prompts=chat_prompts,
            sampling_params=self.sampling_params
        )
        
        # Organize results by prompt
        results = []
        for output_group in outputs:
            results.append([output.text for output in output_group.outputs])
            
        return results

    def process_prompts(self, prompt_data_batch: list[dict]) -> list[dict]:
        """Process a batch of prompts and generate responses for each.
        
        Args:
            prompt_data_batch: List of prompt data dictionaries
            
        Returns:
            List of result dictionaries with responses added
        """
        # Extract prompt texts
        prompt_texts = [data.get("prompt", "") for data in prompt_data_batch]
        
        # Generate responses for all prompts
        all_responses = self.generate_responses(prompt_texts)
        
        # Combine results back with original data
        results = []
        for i, prompt_data in enumerate(prompt_data_batch):
            result = prompt_data.copy()
            result["responses"] = all_responses[i]
            results.append(result)
            
        return results


def get_work(dispatcher_server, batch_size=1):
    client = WorkClient(dispatcher_server)
    print(f"Using dispatcher server at {dispatcher_server}, batch size: {batch_size}")
    while True:
        resp = client.get_work(batch_size=batch_size)
        if resp.status == WorkStatus.ALL_WORK_COMPLETE:
            print("All work complete. Exiting.")
            break
        elif resp.status == WorkStatus.RETRY:
            print(f"No work available; retry in {resp.retry_in} seconds.")
            time.sleep(resp.retry_in)
            continue
        elif resp.status == WorkStatus.SERVER_UNAVAILABLE:
            print("Server is unavailable. Exiting.")
            break
        elif resp.status == WorkStatus.OK:
            # Return the entire batch at once
            yield resp.items
            # Don't submit results here; it will be done after processing
        else:
            print("Unexpected status from server; exiting.")
            break


def extract_by_path(data, path):
    """
    Extract a value from nested dictionary using a path string.
    
    Args:
        data: The dictionary or list to extract from
        path: A string path like "messages[0].content" or ".prompt"
    
    Returns:
        The extracted value or None if not found
    """
    if not path or not path.strip():
        return data
    
    current = data
    path = path.strip()
    
    # If path starts with a dot, remove it
    if path.startswith('.'):
        path = path[1:]
    
    parts = []
    # Parse the path
    i = 0
    while i < len(path):
        if path[i] == '[':
            # Find the closing bracket
            end = path.find(']', i)
            if end == -1:
                raise ValueError(f"Invalid path: unclosed bracket in {path}")
            
            # Get the index
            index = path[i+1:end]
            try:
                index = int(index)
            except ValueError:
                # Strip quotes if present
                if (index.startswith('"') and index.endswith('"')) or \
                   (index.startswith("'") and index.endswith("'")):
                    index = index[1:-1]
            
            parts.append(index)
            i = end + 1
        elif path[i] == '.':
            i += 1
        else:
            # Find the end of the key
            end = i
            while end < len(path) and path[end] not in '.[':
                end += 1
            
            parts.append(path[i:end])
            i = end
    
    # Navigate the path
    for part in parts:
        try:
            if isinstance(current, list) and isinstance(part, int):
                if 0 <= part < len(current):
                    current = current[part]
                else:
                    return None
            elif isinstance(current, dict):
                if part in current:
                    current = current[part]
                else:
                    return None
            else:
                return None
        except (TypeError, KeyError, IndexError):
            return None
    
    return current

def main():
    parser = argparse.ArgumentParser(description='Simple Generation Tool')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model for generation')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help='Maximum model context length')
    
    # Generation parameters
    parser.add_argument('--num_generations', type=int, default=1,
                        help='Number of generations per prompt')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling parameter (nucleus sampling)')
    parser.add_argument('--min_p', type=float, default=0.05,
                        help='Min-p sampling parameter (excludes tokens below this probability)')
    parser.add_argument('--max_tokens', type=int, default=4096,
                        help='Maximum number of tokens to generate')
    
    # Input processing
    parser.add_argument('--prompt_path', type=str, default=".messages[0].content",
                        help='JSON path to extract prompt from input (e.g., ".messages[0].content" or ".prompt")')

    
    # Dispatcher and batch parameters
    parser.add_argument('--dispatcher_server', type=str, required=True,
                        help='Dispatcher server in host:port format')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of prompts to process in a single batch')
    
    args = parser.parse_args()
    
    generator = Generator(
        model_path=args.model_path,
        num_generations=args.num_generations,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )
    
    client = WorkClient(args.dispatcher_server)

    for work_batch in get_work(args.dispatcher_server, args.batch_size):
        try:
            # Prepare batch of prompts
            prompt_data_batch = []
            
            for work in work_batch:
                try:
                    # Parse the JSON content
                    if work.content.strip().startswith('{'):
                        # It's a JSON object
                        row = json.loads(work.content)
                        
                        # Extract the prompt using the provided path
                        prompt = extract_by_path(row, args.prompt_path)
                        
                        if prompt is None:
                            print(f"Warning: Could not extract prompt using path: {args.prompt_path}")
                            # Fall back to the content itself as a last resort
                            prompt = work.content
                    else:
                        # Assume the content itself is the prompt (plain text)
                        row = {"raw_content": work.content}
                        prompt = work.content
                    
                    prompt_data = {
                        "prompt": prompt,
                        "original": row,
                        "_work_item": work
                    }
                    prompt_data_batch.append(prompt_data)
                except Exception as e:
                    print(f"Error parsing work item: {e}")
                    work.set_error(f"Error parsing work item: {str(e)}")
                    # Submit this result immediately since it won't be part of the batch
                    client.submit_results([work])
            
            if not prompt_data_batch:
                print("No valid work items in batch, continuing...")
                continue
                
            # Process the batch
            results = generator.process_prompts(prompt_data_batch)
            
            # Set results for each work item
            for result in results:
                work_item = result.pop("_work_item")  # Remove the work item reference
                work_item.set_result(json.dumps(result))
            
            # Submit all results back to the dispatcher
            client.submit_results(work_batch)
            print(f"Processed batch of {len(work_batch)} prompts")
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Try to submit errors for each work item
            for work in work_batch:
                try:
                    work.set_error(f"Batch processing error: {str(e)}")
                except:
                    pass
            # Submit all results, even with errors
            client.submit_results(work_batch)

if __name__ == "__main__":
    main()
