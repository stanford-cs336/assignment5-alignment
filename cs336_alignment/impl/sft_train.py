
import os
import re
import threading
import random

from datasets import load_dataset
from vllm import LLM, SamplingParams
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from torch.nn.utils import clip_grad_norm_
import wandb

from pathlib import Path
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.impl.mathBaseline import evaluate_vllm
from cs336_alignment.impl.util import tokenize_prompt_and_output, get_response_log_probs, \
    sft_microbatch_train_step

# Check for single GPU mode
SINGLE_GPU = os.environ.get("SINGLE_GPU", "0") == "1"
TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:0" if SINGLE_GPU else "cuda:1"
VLLM_GPU_MEMORY = 0.45 if SINGLE_GPU else 0.85

ds = load_dataset("gsm8k", "main")

def load_r1_zero_prompt_template() -> str:
    """Load the r1_zero prompt template from file (without the trailing <think>)."""
    prompt_file = Path('cs336_alignment/prompts/r1_zero.prompt')
    if prompt_file.exists():
        content = prompt_file.read_text()
        # Remove the trailing "<think>" since we'll include it in the answer
        if content.rstrip().endswith('<think>'):
            content = content.rstrip()[:-7].rstrip()  # Remove <think> and trailing whitespace
        return content + "\n"  # Add newline for clean formatting
    else:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")


def format_gsm8k_answer_for_r1_zero(gsm8k_answer: str) -> str:
    """
    Convert GSM8K answer format to r1_zero format.

    GSM8K format:
        Natalia sold 48/2 = <<48/2=24>>24 clips in May.
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
        #### 72

    r1_zero format:
        <think>Natalia sold 48/2 = 24 clips in May.
        Natalia sold 48+24 = 72 clips altogether in April and May.</think> <answer>72</answer>
    """
    # Remove <<...>> calculation annotations
    cleaned = re.sub(r'<<[^>]+>>', '', gsm8k_answer)

    # Split on #### to get reasoning and final answer
    if '####' in cleaned:
        parts = cleaned.split('####')
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        # Fallback if no #### marker
        reasoning = cleaned.strip()
        final_answer = reasoning

    # Format as r1_zero expected output (full <think>...</think> wrapper)
    return f"<think>{reasoning}</think> <answer>{final_answer}</answer>"

sampling_params = SamplingParams(
    temperature=0.7,  # Lower temperature for more focused outputs
    top_p=0.95,       # Slightly lower top_p
    max_tokens=2048,  # More tokens for reasoning
    stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
    include_stop_str_in_output=True
)

micro_batch_size = 4

evaluation_step = 72

# Load Qwen 2.5 Math 1.5B model using vLLM


def run_async_evaluation(vllm_instance, reward_fn, prompts, ground_truths, sampling_params, step):
    """Run evaluation asynchronously in a background thread."""
    eval_metrics = evaluate_vllm(vllm_instance, reward_fn, prompts, ground_truths, sampling_params, step)

    # Log sample generations to wandb for visual inspection
    num_samples = 3
    results_file = Path(f"evaluation_results_step_{step}.jsonl")
    if results_file.exists():
        with results_file.open('r') as f:
            all_results = [json.loads(line) for line in f]

        # Sample a few examples
        sample_results = random.sample(all_results, min(num_samples, len(all_results)))

        # Create a wandb table for nice visualization
        table = wandb.Table(columns=["question", "generation", "ground_truth", "format_reward", "answer_reward"])
        for r in sample_results:
            table.add_data(
                r["example"][:200] + "..." if len(r["example"]) > 200 else r["example"],
                r["generation"][:500] + "..." if len(r["generation"]) > 500 else r["generation"],
                r["ground_truth"][:200] + "..." if len(r["ground_truth"]) > 200 else r["ground_truth"],
                r["scores"]["format_reward"],
                r["scores"]["answer_reward"]
            )

        wandb.log({
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/format_reward": eval_metrics["format_reward"],
            "eval/sample_generations": table,
            "eval_step": step
        })

        # Also print samples to console for immediate visibility
        print(f"\n{'='*60}")
        print(f"Sample generations at step {step}:")
        print(f"{'='*60}")
        for i, r in enumerate(sample_results):
            print(f"\n--- Sample {i+1} ---")
            print(f"Question: {r['example'][:100]}...")
            print(f"Generation: {r['generation'][:300]}...")
            print(f"Format reward: {r['scores']['format_reward']}, Answer reward: {r['scores']['answer_reward']}")
        print(f"{'='*60}\n")
    else:
        wandb.log({
            "eval/accuracy": eval_metrics["accuracy"],
            "eval/format_reward": eval_metrics["format_reward"],
            "eval_step": step
        })

    return eval_metrics




def train(model, vllm_instance, train_prompts, train_answers,
          optimizer=None, checkpoint_prefix="epoch", epoch_count=1,
          init_wandb=True, run_eval=True):
    """
    Train model on provided prompts and answers.

    Args:
        model: The model to train
        vllm_instance: vLLM instance for evaluation
        train_prompts: List of formatted prompts
        train_answers: List of formatted answers
        optimizer: Optional optimizer (creates new one if None)
        checkpoint_prefix: Prefix for checkpoint naming (e.g., "epoch" -> "epoch_1")
        epoch_count: Number of epochs to train
        init_wandb: Whether to initialize wandb (set False if already initialized)
        run_eval: Whether to run evaluation during training
    """
    # Initialize wandb with custom metrics (only if requested)
    if init_wandb:
        wandb.init(project="math-sft")
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    gradient_accumulation_steps = 16
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    model.train()  # Set to training mode
    if torch.cuda.is_available():
        model = model.to(TRAIN_DEVICE)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # Tokenize formatted training data
    tokenized = tokenize_prompt_and_output(train_prompts, train_answers, tokenizer)
    input_ids = tokenized['input_ids']
    labels = tokenized['labels']
    response_mask = tokenized['response_mask']
    epoch_size = input_ids.shape[0]

    if torch.cuda.is_available():
        input_ids = input_ids.to(TRAIN_DEVICE)
        labels = labels.to(TRAIN_DEVICE)
        response_mask = response_mask.to(TRAIN_DEVICE)

    # Use provided optimizer or create new one
    if optimizer is None:
        optimizer = AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    print(f"Starting training for {epoch_count} epochs")
    print(f"Dataset size: {epoch_size}, Micro-batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {micro_batch_size * gradient_accumulation_steps}")
    print("=" * 80)

    global_step = 0
    eval_thread = None  # Track running evaluation thread

    prompts_valid = [example['question'] for example in ds['test']]
    ground_truths_valid = [example['answer'] for example in ds['test']]

    '''# Run initial evaluation at step 0 to see baseline performance
    print("Running initial evaluation at step 0...")
    load_policy_into_vllm_instance(model, vllm_instance)
    # Validation data (keep raw for evaluation - mathBaseline.py handles formatting)
    eval_thread = threading.Thread(
        target=run_async_evaluation,
        args=(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, sampling_params, global_step)
    )
    eval_thread.start()'''

    # Create index list for shuffling
    indices = list(range(epoch_size))

    for epoch in range(epoch_count):
        print(f"\nEpoch {epoch + 1}/{epoch_count}")
        epoch_losses = []

        # Shuffle indices at the start of each epoch
        random.shuffle(indices)

        for idx, start in enumerate(range(0, epoch_size, micro_batch_size)):
            # Get shuffled indices for this micro-batch
            batch_indices = indices[start:start + micro_batch_size]

            # Forward pass with shuffled data
            micro_batch = input_ids[batch_indices]
            micro_labels = labels[batch_indices]
            micro_response_mask = response_mask[batch_indices]
            log_prob = get_response_log_probs(model, micro_batch, micro_labels)['log_probs']

            # Backward pass
            loss, _ = sft_microbatch_train_step(log_prob, micro_response_mask, gradient_accumulation_steps)
            epoch_losses.append(loss.item())

            if (idx + 1) % gradient_accumulation_steps == 0:
                # Update weights every `gradient_accumulation_steps` batches
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log every N optimizer steps
                if global_step % 10 == 0:
                    avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                    print(f"  Step {global_step} | Loss: {loss.item():.4f} | Avg Loss (last 10): {avg_loss:.4f}")
                    wandb.log({"train/loss": loss.item(), "train_step": global_step})

                # Evaluate every N optimizer steps (async) - only if run_eval is True
                if run_eval and global_step % evaluation_step == 0:
                    # Wait for previous evaluation to finish before starting new one
                    if eval_thread is not None and eval_thread.is_alive():
                        eval_thread.join()

                    load_policy_into_vllm_instance(model, vllm_instance)
                    eval_thread = threading.Thread(
                        target=run_async_evaluation,
                        args=(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, sampling_params, global_step)
                    )
                    eval_thread.start()


        # End of epoch logging
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        print("=" * 80)

        # Save checkpoint after each epoch
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_name = f"{checkpoint_prefix}_{epoch + 1}"
        model.save_pretrained(checkpoint_dir / checkpoint_name)
        tokenizer.save_pretrained(checkpoint_dir / checkpoint_name)
        print(f"Saved checkpoint to {checkpoint_dir / checkpoint_name}")

    # Wait for any remaining evaluation to complete
    if eval_thread is not None and eval_thread.is_alive():
        print("Waiting for final evaluation to complete...")
        eval_thread.join()

    print("\nTraining complete!")
    return model, tokenizer, optimizer

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


if __name__ == "__main__":

    # Load prompt template
    prompt_template = load_r1_zero_prompt_template()

    # Load and format dataset

    # Format training data with r1_zero template
    raw_questions = [example['question'] for example in ds['train']]
    raw_answers = [example['answer'] for example in ds['train']]

    # Apply r1_zero formatting to prompts and answers
    formatted_prompts = [prompt_template.format(question=q) for q in raw_questions]
    formatted_answers = [format_gsm8k_answer_for_r1_zero(a) for a in raw_answers]
    model_to_train = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    vllm_instance = init_vllm("Qwen/Qwen2.5-Math-1.5B-Instruct", device=VLLM_DEVICE, seed=42, gpu_memory_utilization=VLLM_GPU_MEMORY)

    train(model_to_train, vllm_instance, formatted_prompts, formatted_answers, epoch_count=3)
