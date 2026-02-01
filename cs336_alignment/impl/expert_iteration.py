from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from pathlib import Path

from cs336_alignment.impl.sft_train import init_vllm, load_r1_zero_prompt_template, format_gsm8k_answer_for_r1_zero, \
    train, load_policy_into_vllm_instance

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
    vllm_instance = init_vllm("checkpoints/epoch_2", seed=42, device="cuda:1")  # need to update to from the model above

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompt template
    prompt_template = load_r1_zero_prompt_template()

    # Load and format dataset
    ds = load_dataset("gsm8k", "main")

    # Format training data with r1_zero template
    raw_questions = [example['question'] for example in ds['train']]
    raw_answers = [example['answer'] for example in ds['train']]

    # Apply r1_zero formatting to prompts and answers
    formatted_prompts = [prompt_template.format(question=q) for q in raw_questions]
    formatted_answers = [format_gsm8k_answer_for_r1_zero(a) for a in raw_answers]

    for ei_step in range(n_ei_steps):
        print(f"\n{'='*60}")
        print(f"Starting EI step {ei_step}")
        print(f"{'='*60}")

        current_batch_prompts = formatted_prompts[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_answers = formatted_answers[ei_step*batch_size:(ei_step+1)*batch_size]
        current_batch_generations = vllm_instance.generate(current_batch_prompts, sampling_params)

        # Log first 2 prompts and answers
        for log_idx in range(min(2, len(current_batch_prompts))):
            print(f"\n--- Sample {log_idx + 1} ---")
            print(f"Prompt: {current_batch_prompts[log_idx][:200]}...")
            print(f"Ground truth answer: {current_batch_answers[log_idx][:200]}...")

        current_batch_sft_prompts = []
        current_batch_sft_answers = []
        for i in range(len(current_batch_prompts)):
            for j in range(len(current_batch_generations[i].outputs)):
                current_generated_text = current_batch_generations[i].outputs[j].text
                scores = r1_zero_reward_fn(current_generated_text, current_batch_answers[i])

                # Log generations and rewards for first 2 prompts
                if i < 2:
                    print(f"  Prompt {i} Gen {j}: {current_generated_text[:150]}...")
                    print(f"    Rewards: format={scores['format_reward']}, answer={scores['answer_reward']}")

                if scores["answer_reward"] > 0 and scores["format_reward"] > 0:
                    current_batch_sft_prompts.append(current_batch_prompts[i])
                    current_batch_sft_answers.append(current_generated_text)

        print(f"\n{'='*60}\n")

        print(f"EI step {ei_step}: {len(current_batch_sft_prompts)} successful generations out of {len(current_batch_prompts) * rollout_count} total")
        if len(current_batch_sft_prompts) > 0:
            train(model, vllm_instance, current_batch_sft_prompts, current_batch_sft_answers)
            load_policy_into_vllm_instance(model, vllm_instance)
        else:
            print(f"EI step {ei_step}: No successful generations, skipping training")






