
import os

from datasets import load_dataset
from vllm import LLM, SamplingParams
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
from torch.nn.utils import clip_grad_norm_
import wandb

from typing import Callable
from pathlib import Path
import json
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.impl.mathBaseline import evaluate_vllm
from cs336_alignment.impl.util import tokenize_prompt_and_output, get_response_log_probs, masked_normalize, \
    sft_microbatch_train_step

sampling_params = SamplingParams(
    temperature=0.7,  # Lower temperature for more focused outputs
    top_p=0.95,       # Slightly lower top_p
    max_tokens=2048,  # More tokens for reasoning
    stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
    include_stop_str_in_output=True
)

micro_batch_size = 16

evaluation_step = 300

# Load Qwen 2.5 Math 1.5B model using vLLM


def train():
    # Initialize wandb with custom metrics
    wandb.init(project="math-sft")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    gradient_accumulation_steps = 4
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    model.train()  # Set to training mode
    if torch.cuda.is_available():
        model = model.to("cuda:0")  # Explicitly use GPU 0 for training

    vllm_instance = init_vllm("Qwen/Qwen2.5-Math-1.5B-Instruct", device="cuda:1", seed=42)  # GPU 1 for evaluation

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    ds = load_dataset("gsm8k", "main", trust_remote_code=True)
    prompts = [example['question'] for example in ds['train']]
    ground_truths = [example['answer'] for example in ds['train']]
    prompts_valid = [example['question'] for example in ds['test']]
    ground_truths_valid = [example['answer'] for example in ds['test']]
    tokenized = tokenize_prompt_and_output(prompts, ground_truths, tokenizer)
    input_ids = tokenized['input_ids']
    labels = tokenized['labels']
    response_mask = tokenized['response_mask']
    epoch_size = input_ids.shape[0]
    epoch_count = 3

    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda:0")
        labels = labels.to("cuda:0")
        response_mask = response_mask.to("cuda:0")

    optimizer = AdamW(model.parameters(), lr=1e-5)
    optimizer.zero_grad()

    print(f"Starting training for {epoch_count} epochs")
    print(f"Dataset size: {epoch_size}, Micro-batch size: {micro_batch_size}")
    print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {micro_batch_size * gradient_accumulation_steps}")
    print("=" * 80)

    global_step = 0

    for epoch in range(epoch_count):
        print(f"\nEpoch {epoch + 1}/{epoch_count}")
        epoch_losses = []

        for idx, i in enumerate(range(0, epoch_size, micro_batch_size)):
            # Forward passƒ
            micro_batch = input_ids[i:i + micro_batch_size]
            micro_labels = labels[i:i + micro_batch_size]
            micro_response_mask = response_mask[i:i + micro_batch_size]
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

            if global_step % evaluation_step == 0:
                load_policy_into_vllm_instance(model, vllm_instance)
                eval_metrics = evaluate_vllm(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, sampling_params, global_step)
                wandb.log({
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/format_reward": eval_metrics["format_reward"],
                    "eval_step": global_step
                })


        # End of epoch logging
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        print("=" * 80)

        # Save checkpoint after each epoch
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        model.save_pretrained(checkpoint_dir / f"epoch_{epoch + 1}")
        tokenizer.save_pretrained(checkpoint_dir / f"epoch_{epoch + 1}")
        print(f"Saved checkpoint to {checkpoint_dir / f'epoch_{epoch + 1}'}")

    print("\nTraining complete!")
    return model, tokenizer

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
    train()
