import os
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import torch
import wandb
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from pathlib import Path

from cs336_alignment.impl.sft_train import init_vllm, load_r1_zero_prompt_template, \
    train, load_policy_into_vllm_instance, run_async_evaluation, sampling_params as eval_sampling_params

# Check for single GPU mode
SINGLE_GPU = os.environ.get("SINGLE_GPU", "0") == "1"
TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:0" if SINGLE_GPU else "cuda:1"
VLLM_GPU_MEMORY = 0.85  # Full memory since we run sequentially in single GPU mode

n_ei_steps = 5
batch_size = 256
rollout_count = 8
sampling_params = SamplingParams(
    n=rollout_count,
    temperature=0.7,  # Lower temperature for more focused outputs
    top_p=0.95,       # Slightly lower top_p
    max_tokens=2048,  # More tokens for reasoning
    min_tokens=4,
    stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
    include_stop_str_in_output=True
)


def free_vllm(vllm_instance):
    """Free vLLM instance and clear GPU memory."""
    del vllm_instance
    gc.collect()
    torch.cuda.empty_cache()
    print("Freed vLLM memory")


def free_model(model):
    """Free training model and clear GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Freed training model memory")


def expert_iteration():
    checkpoint_path = Path("checkpoints/epoch_2")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize wandb once for the entire EI run
    wandb.init(project="expert-iteration")
    wandb.define_metric("ei_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    # Load prompt template
    prompt_template = load_r1_zero_prompt_template()

    # Load and format dataset
    ds = load_dataset("gsm8k", "main")

    # Format training data with r1_zero template
    raw_questions = [example['question'] for example in ds['train']]
    raw_answers = [example['answer'] for example in ds['train']]

    # Apply r1_zero formatting to prompts only (answers used raw for reward function)
    formatted_prompts = [prompt_template.format(question=q) for q in raw_questions]

    # Validation data for evaluation
    prompts_valid = [example['question'] for example in ds['test']]
    ground_truths_valid = [example['answer'] for example in ds['test']]

    current_checkpoint = str(checkpoint_path)

    if SINGLE_GPU:
        # Single GPU mode: sequential loading/freeing to fit in memory
        _run_single_gpu_mode(
            tokenizer, formatted_prompts, raw_questions, raw_answers,
            prompts_valid, ground_truths_valid, current_checkpoint
        )
    else:
        # Dual GPU mode: training model on GPU 0, vLLM on GPU 1
        # Both stay loaded throughout - use load_policy_into_vllm_instance to update weights
        _run_dual_gpu_mode(
            tokenizer, formatted_prompts, raw_questions, raw_answers,
            prompts_valid, ground_truths_valid, current_checkpoint
        )


def _run_dual_gpu_mode(tokenizer, formatted_prompts, raw_questions, raw_answers,
                       prompts_valid, ground_truths_valid, current_checkpoint):
    """
    Dual GPU mode: training model stays on GPU 0, vLLM stays on GPU 1.
    No loading/freeing during the loop - just update vLLM weights in place.
    """
    # Load both models once at the start
    print("\n[Init] Loading training model on GPU 0...")
    model = AutoModelForCausalLM.from_pretrained(current_checkpoint).to(TRAIN_DEVICE)

    print("[Init] Loading vLLM on GPU 1...")
    vllm_instance = init_vllm(current_checkpoint, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=VLLM_GPU_MEMORY)

    # Create optimizer once to maintain momentum across EI steps
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for ei_step in range(n_ei_steps):
        print(f"\n{'='*60}")
        print(f"Starting EI step {ei_step}")
        print(f"{'='*60}")

        # === PHASE 1: Generation with vLLM ===
        current_batch_prompts = formatted_prompts[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_raw_answers = raw_answers[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_questions = raw_questions[ei_step*batch_size:(ei_step+1)*batch_size]

        print(f"Generating {len(current_batch_prompts)} x {rollout_count} = {len(current_batch_prompts) * rollout_count} rollouts...")
        current_batch_generations = vllm_instance.generate(current_batch_prompts, sampling_params)

        # Log first 2 prompts and answers
        for log_idx in range(min(2, len(current_batch_prompts))):
            print(f"\n--- Sample {log_idx + 1} ---")
            print(f"Question: {current_batch_questions[log_idx][:200]}...")
            print(f"Ground truth answer: {current_batch_raw_answers[log_idx][:200]}...")

        # Filter successful generations
        current_batch_sft_prompts = []
        current_batch_sft_answers = []
        for i in range(len(current_batch_prompts)):
            for j in range(len(current_batch_generations[i].outputs)):
                current_generated_text = current_batch_generations[i].outputs[j].text
                scores = r1_zero_reward_fn(current_generated_text, current_batch_raw_answers[i])

                # Log generations and rewards for first 2 prompts
                if i < 2:
                    print(f"  Prompt {i} Gen {j}: {current_generated_text[:150]}...")
                    print(f"    Rewards: format={scores['format_reward']}, answer={scores['answer_reward']}")

                if scores["answer_reward"] > 0 and scores["format_reward"] > 0:
                    current_batch_sft_prompts.append(current_batch_prompts[i])
                    current_batch_sft_answers.append(current_generated_text)

        print(f"\n{'='*60}\n")
        print(f"EI step {ei_step}: {len(current_batch_sft_prompts)} successful generations out of {len(current_batch_prompts) * rollout_count} total")
        wandb.log({"ei/successful_generations": len(current_batch_sft_prompts), "ei_step": ei_step})

        # === PHASE 2: Training ===
        if len(current_batch_sft_prompts) > 0:
            print(f"\n[EI Step {ei_step}] Training on GPU 0...")

            # Train (pass None for vllm_instance since we're not doing eval during training)
            model, _, optimizer = train(
                model, None, current_batch_sft_prompts, current_batch_sft_answers,
                optimizer=optimizer,
                checkpoint_prefix=f"ei_step_{ei_step}",
                epoch_count=1,
                init_wandb=False,
                run_eval=False
            )

            # Update vLLM weights from training model (fast, no reload needed)
            print(f"[EI Step {ei_step}] Updating vLLM weights in place...")
            load_policy_into_vllm_instance(model, vllm_instance)
        else:
            print(f"EI step {ei_step}: No successful generations, skipping training")

        # === PHASE 3: Evaluation ===
        print(f"\n[EI Step {ei_step}] Running evaluation...")
        run_async_evaluation(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, eval_sampling_params, ei_step)

    # Save final model
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint_dir / "ei_final")
    tokenizer.save_pretrained(checkpoint_dir / "ei_final")
    print(f"\nExpert iteration complete! Final model saved to {checkpoint_dir / 'ei_final'}")

    wandb.finish()
    return model, tokenizer


def _run_single_gpu_mode(tokenizer, formatted_prompts, raw_questions, raw_answers,
                         prompts_valid, ground_truths_valid, current_checkpoint):
    """
    Single GPU mode: sequential loading/freeing to fit in memory.
    Only one model (vLLM or training) loaded at a time.
    """
    # Track optimizer state on CPU to persist across model reloads
    optimizer_state = None

    for ei_step in range(n_ei_steps):
        print(f"\n{'='*60}")
        print(f"Starting EI step {ei_step}")
        print(f"{'='*60}")

        # === PHASE 1: Generation with vLLM ===
        print(f"\n[EI Step {ei_step}] Loading vLLM for generation...")
        vllm_instance = init_vllm(current_checkpoint, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=VLLM_GPU_MEMORY)

        current_batch_prompts = formatted_prompts[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_raw_answers = raw_answers[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_questions = raw_questions[ei_step*batch_size:(ei_step+1)*batch_size]

        print(f"Generating {len(current_batch_prompts)} x {rollout_count} = {len(current_batch_prompts) * rollout_count} rollouts...")
        current_batch_generations = vllm_instance.generate(current_batch_prompts, sampling_params)

        # Log first 2 prompts and answers
        for log_idx in range(min(2, len(current_batch_prompts))):
            print(f"\n--- Sample {log_idx + 1} ---")
            print(f"Question: {current_batch_questions[log_idx][:200]}...")
            print(f"Ground truth answer: {current_batch_raw_answers[log_idx][:200]}...")

        # Filter successful generations
        current_batch_sft_prompts = []
        current_batch_sft_answers = []
        for i in range(len(current_batch_prompts)):
            for j in range(len(current_batch_generations[i].outputs)):
                current_generated_text = current_batch_generations[i].outputs[j].text
                scores = r1_zero_reward_fn(current_generated_text, current_batch_raw_answers[i])

                # Log generations and rewards for first 2 prompts
                if i < 2:
                    print(f"  Prompt {i} Gen {j}: {current_generated_text[:150]}...")
                    print(f"    Rewards: format={scores['format_reward']}, answer={scores['answer_reward']}")

                if scores["answer_reward"] > 0 and scores["format_reward"] > 0:
                    current_batch_sft_prompts.append(current_batch_prompts[i])
                    current_batch_sft_answers.append(current_generated_text)

        print(f"\n{'='*60}\n")
        print(f"EI step {ei_step}: {len(current_batch_sft_prompts)} successful generations out of {len(current_batch_prompts) * rollout_count} total")
        wandb.log({"ei/successful_generations": len(current_batch_sft_prompts), "ei_step": ei_step})

        # Free vLLM before training
        free_vllm(vllm_instance)
        vllm_instance = None

        # === PHASE 2: Training ===
        if len(current_batch_sft_prompts) > 0:
            print(f"\n[EI Step {ei_step}] Loading model for training...")
            model = AutoModelForCausalLM.from_pretrained(current_checkpoint).to(TRAIN_DEVICE)

            # Restore optimizer state if we have one
            optimizer = AdamW(model.parameters(), lr=1e-5)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)

            # Train (pass None for vllm_instance since we're not doing eval during training)
            model, _, optimizer = train(
                model, None, current_batch_sft_prompts, current_batch_sft_answers,
                optimizer=optimizer,
                checkpoint_prefix=f"ei_step_{ei_step}",
                epoch_count=1,
                init_wandb=False,
                run_eval=False
            )

            # Save checkpoint for next iteration
            next_checkpoint = f"checkpoints/ei_step_{ei_step}_1"
            current_checkpoint = next_checkpoint

            # Save optimizer state to CPU
            optimizer_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                             for k, v in optimizer.state_dict().items()}

            # Free training model before evaluation
            free_model(model)
            model = None
        else:
            print(f"EI step {ei_step}: No successful generations, skipping training")

        # === PHASE 3: Evaluation ===
        print(f"\n[EI Step {ei_step}] Loading vLLM for evaluation...")
        vllm_instance = init_vllm(current_checkpoint, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=VLLM_GPU_MEMORY)

        run_async_evaluation(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, eval_sampling_params, ei_step)

        # Free vLLM after evaluation
        free_vllm(vllm_instance)
        vllm_instance = None

    # Save final model
    print(f"\n[Final] Loading model to save final checkpoint...")
    model = AutoModelForCausalLM.from_pretrained(current_checkpoint)
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint_dir / "ei_final")
    tokenizer.save_pretrained(checkpoint_dir / "ei_final")
    print(f"\nExpert iteration complete! Final model saved to {checkpoint_dir / 'ei_final'}")

    wandb.finish()
    return model, tokenizer


if __name__ == "__main__":
    expert_iteration()
