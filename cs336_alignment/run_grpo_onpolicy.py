## GRPO training scripts
import os
import gc
import random
from pathlib import Path
import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s [%(levelname)s] %(message)s',
    filename = "grpo_training_onpolicy.log",
    filemode = "a"
)

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.prompting import get_prompt, get_grade
from cs336_alignment.grpo import grpo_train_step, save_checkpoint, load_checkpoint

from datasets import load_dataset

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")

## wrapper of grading function using R1-zero three shot prompting format
def reward_fn(response: str, ground_truth: str) -> dict:
    return get_grade(response=response, ground_truth=ground_truth, prompt_key="three_shot")

## quick transformation of train_step to train_footprint
def train_footprint(train_step: int) -> str:
    if train_step < 10:
        return "000" + str(train_step)
    elif train_step < 100:
        return "00" + str(train_step)
    elif train_step < 1000:
        return "0" + str(train_step)
    else:
        return str(train_step)

n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 4
gradient_accumulation_steps = 64
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0
assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"

with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero_three_shot_gsm8k.prompt"), "r") as fin:
    prompt_template = fin.read()

sampling_params = {
    "n": group_size,
    "seed": 42,
    "temperature": sampling_temperature,
    "top_p": 1.0,
    "max_tokens": sampling_max_tokens
}
r1_zero_params = {
    "stop": ["</answer>"],
    "include_stop_str_in_output": True
}

ds = load_dataset("openai/gsm8k", "main")
ds_train_len = len(ds["train"])

continue_from = 30
max_iter = 50
if continue_from >= 0:
    continue_pt = train_footprint(continue_from)
    if not Path(os.path.join(CHECKPOINT_DIR, f"train_step_{continue_pt}")).is_dir():
        raise FileNotFoundError(f"Checkpoint file {os.path.join(CHECKPOINT_DIR, f'train_step_{continue_pt}')} not found.")
train_step = continue_from if continue_from >= 0 else -1

while train_step < max_iter:
    train_step += 1
    train_pt = train_footprint(train_step)
    output_dir = os.path.join(CHECKPOINT_DIR, f"train_step_{train_pt}")
    os.makedirs(output_dir, exist_ok=True)
    # logging.info(f"Starting training step {train_step}, output directory: {output_dir}")

    ## draw samples from the GSM8K dataset and transform to R1-zero format, get completions
    if train_step == 0:
        vllm_server = VLLMServer(model_id = "allenai/OLMo-2-0425-1B", gpu = 0, startup_timeout = 600, logging_level = "ERROR")
    else:
        continue_pt = train_footprint(train_step - 1)
        vllm_server = VLLMServer(model_id = os.path.join(CHECKPOINT_DIR, f"train_step_{continue_pt}"), gpu = 0, startup_timeout = 600, logging_level = "ERROR")
        # logging.info(f"VLLM server continue from {continue_pt}")
    vllm_server.start()
    try:
        prompts = list()
        ground_truths = list()
        rollout_responses = list()

        sample_size = rollout_batch_size // group_size
        sample_indices = [x for x in range(ds_train_len)]
        random.shuffle(sample_indices)
        sample_indices = sample_indices[:sample_size]
        for index in sample_indices:
            sample = ds["train"][index]
            prompt, question, reasoning, answer = get_prompt(sample, prompt_template)
            prompts.append(prompt)
            ground_truths.append(answer)
        completions = vllm_server.generate_completions(prompts = prompts, sampling_params = sampling_params | r1_zero_params)
        rollout_responses += [x.text.replace("\n", " ").strip() for x in completions]
        repeated_prompts = [x for x in prompts for _ in range(group_size)]
        repeated_ground_truths = [x for x in ground_truths for _ in range(group_size)]
        assert len(repeated_prompts) == len(repeated_ground_truths) == len(rollout_responses) == rollout_batch_size
        del prompts, ground_truths
        gc.collect()
    finally:
        vllm_server.stop()

    if train_step == 0:
        model, tokenizer = get_model_and_tokenizer(model_id_or_dir = "allenai/OLMo-2-0425-1B", device = "cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.95), weight_decay = 0.0)
    else:
        continue_pt = train_footprint(train_step - 1)
        model, tokenizer = get_model_and_tokenizer(model_id_or_dir = os.path.join(CHECKPOINT_DIR, f"train_step_{continue_pt}"), device = "cuda:0")
        optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.95), weight_decay = 0.0)
        load_checkpoint(os.path.join(CHECKPOINT_DIR, f"train_step_{continue_pt}/checkpoint.pt"), optimizer)
        # logging.info(f"Model and optimizer continue from {continue_pt}")

    model.train()

    loss, metadata = grpo_train_step(
        model = model,
        tokenizer = tokenizer,
        optimizer = optimizer,
        gradient_accumulation_steps = gradient_accumulation_steps,
        max_grad_norm = max_grad_norm,
        reward_fn = reward_fn,
        repeated_prompts = repeated_prompts,
        rollout_responses = rollout_responses,
        repeated_ground_truths = repeated_ground_truths,
        group_size = group_size,
        baseline = "mean",
        advantage_eps = 1e-6,
        advantage_normalizer = "std",
        importance_reweighting_method = "none",
        old_log_probs = None,
        cliprange = None,
        loss_normalization = "sequence",
        normalization_constant = None,
    )
    assert torch.isfinite(loss).item(), "Loss is NaN or infinite"
    # logging.info(f"Training loss at step {train_step}: {loss.item()}")
    logging.info(f"Training metadata at step {train_step}: {metadata}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    save_checkpoint(optimizer, train_step, os.path.join(output_dir, "checkpoint.pt"))
    del model, tokenizer, optimizer
    gc.collect()
    torch.cuda.empty_cache()


## validation step
sample_size = n_val_examples
sample_indices = [x for x in range(len(ds["test"]))]
random.seed(42)
random.shuffle(sample_indices)
sample_indices = sample_indices[:sample_size]
microbatch_size = 32
accumulation_steps = sample_size // microbatch_size

sampling_params = {
    "n": 1,
    "seed": 42,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": sampling_max_tokens
}
r1_zero_params = {
    "stop": ["</answer>"],
    "include_stop_str_in_output": True
}

for train_step in range(0, max_iter + 1, 5):
    train_pt = train_footprint(train_step)
    output_dir = os.path.join(CHECKPOINT_DIR, f"train_step_{train_pt}")
    if not Path(output_dir).is_dir():
        logging.warning(f"Checkpoint directory {output_dir} does not exist. Skipping validation for this step.")
        continue
    vllm_server = VLLMServer(model_id = output_dir, gpu = 0, startup_timeout = 600, logging_level = "ERROR")
    vllm_server.start()
    try:
        rewards = list()

        for step in range(accumulation_steps):
            prompts = list()
            ground_truths = list()
            lo, hi = step * microbatch_size, (step + 1) * microbatch_size
            for index in sample_indices[lo:hi]:
                sample = ds["test"][index]
                prompt, question, reasoning, answer = get_prompt(sample, prompt_template)
                prompts.append(prompt)
                ground_truths.append(answer)
            completions = vllm_server.generate_completions(prompts = prompts, sampling_params = sampling_params | r1_zero_params)
            val_responses = [x.text.replace("\n", " ").strip() for x in completions]
            assert len(prompts) == len(ground_truths) == len(val_responses) == microbatch_size
            rewards.extend([reward_fn(response=response, ground_truth=ground_truth) for response, ground_truth in zip(val_responses, ground_truths)])
            del prompts
            gc.collect()
    finally:
        vllm_server.stop()

    tot_reward = sum([r["reward"] for r in rewards]) / len(rewards)
    format_reward = sum([r["format_reward"] for r in rewards]) / len(rewards)
    answer_reward = sum([r["answer_reward"] for r in rewards]) / len(rewards)
    logging.info(f"Validation average reward at step {train_step}: {tot_reward}, format reward: {format_reward}, answer reward: {answer_reward}")