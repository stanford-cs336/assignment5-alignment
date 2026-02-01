from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
import torch
import wandb
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from pathlib import Path

import threading
from cs336_alignment.impl.sft_train import init_vllm, load_r1_zero_prompt_template, \
    train, load_policy_into_vllm_instance, run_async_evaluation, sampling_params as eval_sampling_params

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


def expert_iteration():
    checkpoint_path = Path("checkpoints/epoch_2")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    vllm_instance = init_vllm("checkpoints/epoch_2", seed=42, device="cuda:1")

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize wandb once for the entire EI run
    wandb.init(project="expert-iteration")
    wandb.define_metric("ei_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    # Create optimizer once to maintain momentum across EI steps
    optimizer = AdamW(model.parameters(), lr=1e-5)

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
    eval_thread = None  # Track running evaluation thread

    for ei_step in range(n_ei_steps):
        print(f"\n{'='*60}")
        print(f"Starting EI step {ei_step}")
        print(f"{'='*60}")

        current_batch_prompts = formatted_prompts[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_raw_answers = raw_answers[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_questions = raw_questions[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_generations = vllm_instance.generate(current_batch_prompts, sampling_params)

        # Log first 2 prompts and answers
        for log_idx in range(min(2, len(current_batch_prompts))):
            print(f"\n--- Sample {log_idx + 1} ---")
            print(f"Question: {current_batch_questions[log_idx][:200]}...")
            print(f"Ground truth answer: {current_batch_raw_answers[log_idx][:200]}...")

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

        if len(current_batch_sft_prompts) > 0:
            model, _, optimizer = train(
                model, vllm_instance, current_batch_sft_prompts, current_batch_sft_answers,
                optimizer=optimizer,
                checkpoint_prefix=f"ei_step_{ei_step}",
                epoch_count=1,
                init_wandb=False,
                run_eval=False
            )
        else:
            print(f"EI step {ei_step}: No successful generations, skipping training")

        # Run async evaluation after each EI step
        if eval_thread is not None and eval_thread.is_alive():
            eval_thread.join()
        load_policy_into_vllm_instance(model, vllm_instance)
        eval_thread = threading.Thread(
            target=run_async_evaluation,
            args=(vllm_instance, r1_zero_reward_fn, prompts_valid, ground_truths_valid, eval_sampling_params, ei_step)
        )
        eval_thread.start()

    # Wait for final evaluation to complete
    if eval_thread is not None and eval_thread.is_alive():
        print("Waiting for final evaluation to complete...")
        eval_thread.join()

    # Save final model after all EI steps
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint_dir / "ei_final")
    tokenizer.save_pretrained(checkpoint_dir / "ei_final")
    print(f"\nExpert iteration complete! Final model saved to {checkpoint_dir / 'ei_final'}")

    wandb.finish()
    return model, tokenizer


if __name__ == "__main__":
    expert_iteration()


