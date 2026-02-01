import os
# Use Flash Attention instead of FlashInfer (FlashInfer not available on all platforms)
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

from datasets import load_dataset
from vllm import LLM, SamplingParams
from typing import Callable
from pathlib import Path
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

#print(ds['train'][:5])

"""
Evaluate a language model on a list of prompts,
compute evaluation metrics, and serialize results to disk.
"""
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    step_number=0
    ) -> dict[str, float]:

    prompt_file = Path('cs336_alignment/prompts/r1_zero.prompt')
    if prompt_file.exists():
        content = prompt_file.read_text()
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    formatted_prompts = [content.format(question=prompt) for prompt in prompts]
    outputs = vllm_model.generate(formatted_prompts, eval_sampling_params)

    # Collect results for serialization
    results = []
    total_accuracy = 0.0
    total_format_reward = 0.0

    for i in range(len(prompts)):
        generated_text = outputs[i].outputs[0].text
        scores = reward_fn(generated_text, ground_truths[i])

        results.append({
            "example": prompts[i],
            "ground_truth": ground_truths[i],
            "generation": generated_text,
            "scores": scores
        })

        # Accumulate metrics
        total_accuracy += 1.0 if scores["answer_reward"] > 0 else 0.0
        total_format_reward += scores["format_reward"]

    # Compute averages
    avg_accuracy = total_accuracy / len(prompts)
    avg_format_reward = total_format_reward / len(prompts)

    # Serialize to disk as JSONL (one JSON object per line)
    output_path = Path(f"evaluation_results_step_{step_number}.jsonl")
    with output_path.open('w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"Saved {len(results)} evaluation results to {output_path}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Format Reward: {avg_format_reward:.4f}")

    return {
        "accuracy": avg_accuracy,
        "format_reward": avg_format_reward
    }


if __name__ == "__main__":
    ds = load_dataset("gsm8k", "main")
    prompts = [example['question'] for example in ds['test']]
    ground_truths = [example['answer'] for example in ds['test']]
    sampling_params = SamplingParams(
        temperature=0.7,  # Lower temperature for more focused outputs
        top_p=0.95,       # Slightly lower top_p
        max_tokens=2048,  # More tokens for reasoning
        stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
        include_stop_str_in_output=True
    )

    # Load Qwen 2.5 Math 1.5B model using vLLM
    model = LLM(model="Qwen/Qwen2.5-Math-1.5B-Instruct")
    evaluate_vllm(model, r1_zero_reward_fn, prompts, ground_truths, sampling_params)


