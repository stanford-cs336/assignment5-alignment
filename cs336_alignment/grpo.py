from __future__ import annotations

import os
from typing import Any, Callable, Literal

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase
) -> dict[str, Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask aligned with
    labels that is 1 for response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizerBase
            Tokenizer to use for tokenization.

    Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    assert len(prompt_strs) == len(output_strs), "prompt_strs and output_strs must have the same length"

    prompt_tokens = [tokenizer(x, add_special_tokens=False) for x in prompt_strs]
    output_tokens = [tokenizer(x, add_special_tokens=True) for x in output_strs]
    combine_tokens = tokenizer([x + y if y[0] == " " else x + " " + y for x, y in zip(prompt_strs, output_strs)], add_special_tokens=True, padding=True, return_tensors="pt")

    input_ids = combine_tokens["input_ids"][:, :-1]
    labels = combine_tokens["input_ids"][:, 1:]

    response_mask = torch.zeros_like(labels)
    for i, (prompt, output) in enumerate(zip(prompt_tokens, output_tokens)):
        prompt_len = len(prompt["input_ids"])
        output_len = len(output["input_ids"])
        response_mask[i, (prompt_len - 1): (prompt_len + output_len - 1)] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask
    }

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    outputs = model(input_ids=input_ids)
    log_probs = F.log_softmax(outputs.logits, dim = -1)
    res = {
        "log_probs": torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    }
    if return_token_entropy:
        res["token_entropy"] = Categorical(logits=outputs.logits).entropy()
    return res

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    assert len(rollout_responses) > 0, "rollout_responses must be non-empty"
    assert len(rollout_responses) == len(repeated_ground_truths), "rollout_responses and repeated_ground_truths must have the same length"
    rewards = [reward_fn(x, y) for x, y in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward["reward"] for reward in rewards], dtype=torch.float32)
    metadata = {key: np.mean([reward[key] for reward in rewards]) for key in rewards[0].keys()}
    return raw_rewards, metadata

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            For this problem, support mean, which subtracts the per-group mean
            reward. Later, none will mean no baseline subtraction.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            For this problem, support std, which divides by the per-group
            standard deviation. Later, none will mean no normalization and
            mean will mean divide by the per-group mean reward.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    if baseline not in ["mean", "none"]:
        raise NotImplementedError(f"baseline {baseline} not supported")
    if advantage_normalizer not in ["std", "none", "mean"]:
        raise NotImplementedError(f"advantage_normalizer {advantage_normalizer} not supported")
    rollout_batch_size = raw_rewards.shape[0]
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_groups = rollout_batch_size // group_size
    grouped_rewards = raw_rewards.view(n_groups, group_size)
    grouped_mean = torch.mean(grouped_rewards, dim = 1, keepdim = True)
    grouped_std = torch.std(grouped_rewards, dim = 1, keepdim = True)
    if baseline == "mean":
        grouped_rewards = grouped_rewards - grouped_mean
    if advantage_normalizer == "std":
        grouped_rewards = grouped_rewards / (grouped_std + advantage_eps)
    elif advantage_normalizer == "mean":
        grouped_rewards = grouped_rewards / (grouped_mean + advantage_eps)
    advantages = torch.flatten(grouped_rewards)
    metadata = {
        "mean": torch.mean(raw_rewards).item(),
        "std": torch.std(raw_rewards).item(),
        "max": torch.max(raw_rewards).item(),
        "min": torch.min(raw_rewards).item()
    }
    return advantages, metadata

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if importance_reweighting_method not in ["none", "noclip", "grpo", "gspo"]:
        raise NotImplementedError(f"importance_reweighting_method {importance_reweighting_method} not supported")
    if raw_rewards_or_advantages.ndim == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)
    per_token_policy_gradient_loss = -raw_rewards_or_advantages * policy_log_probs
    metadata = {}
    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if loss_normalization not in ["sequence", "constant"]:
        raise NotImplementedError(f"loss_normalization {loss_normalization} not supported")
    if loss_normalization == "constant" and normalization_constant is None:
        raise ValueError("normalization_constant must be provided when loss_normalization is 'constant'")
    if loss_normalization == "sequence":
        sequence_loss = torch.sum(per_token_policy_gradient_loss * mask, dim = 1) / torch.sum(mask, dim = 1)
        loss = torch.mean(sequence_loss)
    elif loss_normalization == "constant":
        loss = torch.sum(per_token_policy_gradient_loss * mask) / normalization_constant
    return loss

def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    rollout_batch_size = len(rollout_responses)
    assert gradient_accumulation_steps > 0, "gradient_accumulation_steps must be positive"
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    assert rollout_batch_size == len(repeated_prompts) == len(repeated_ground_truths), "rollout_responses, repeated_prompts, and repeated_ground_truths must have the same length"
    assert rollout_batch_size % gradient_accumulation_steps == 0, "rollout_batch_size must be divisible by gradient_accumulation_steps"
    assert (rollout_batch_size // gradient_accumulation_steps) % group_size == 0, "batch_size per optimizer step must be divisible by group_size"

    device = next(model.parameters()).device

    ## compute raw rewards and group-normalized advantages
    rewards = [reward_fn(x, y) for x, y in zip(rollout_responses, repeated_ground_truths)]
    raw_rewards = torch.tensor([reward["reward"] for reward in rewards], dtype = torch.float32)
    advantages, advantages_metadata = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer
    )
    advantages = advantages.to(device)

    optimizer.zero_grad()

    ## microbatching
    microbatch_size = rollout_batch_size // gradient_accumulation_steps
    total_loss = torch.zeros(1).to(device)
    for i in range(0, rollout_batch_size, microbatch_size):
        inputs_microbatch = repeated_prompts[i:i + microbatch_size]
        labels_microbatch = rollout_responses[i: i + microbatch_size]

        tokenized_microbatch = tokenize_prompt_and_output(
            prompt_strs = inputs_microbatch,
            output_strs = labels_microbatch,
            tokenizer = tokenizer
        )
        for key, val in tokenized_microbatch.items():
            tokenized_microbatch[key] = val.to(device)

        log_probs = get_response_log_probs(
            model = model,
            input_ids = tokenized_microbatch["input_ids"],
            labels = tokenized_microbatch["labels"],
            return_token_entropy = False,
        )

        gradient_loss, gradient_loss_metadata = compute_policy_gradient_loss(
            raw_rewards_or_advantages = advantages[i: i + microbatch_size],
            policy_log_probs = log_probs["log_probs"],
            importance_reweighting_method = importance_reweighting_method,
            # old_log_probs = old_log_probs[i: i + microbatch_size, :],
            cliprange = cliprange,
            response_mask = tokenized_microbatch["response_mask"]
        )

        loss_microbatch = aggregate_loss_across_microbatch(
            per_token_policy_gradient_loss = gradient_loss,
            mask = tokenized_microbatch["response_mask"],
            loss_normalization = loss_normalization,
            normalization_constant = normalization_constant
        )
        loss_microbatch = loss_microbatch / gradient_accumulation_steps
        loss_microbatch.backward()
        total_loss = total_loss + loss_microbatch.detach()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none = True)

    metadata = {} | advantages_metadata
    return total_loss, metadata

def save_checkpoint(optimizer: torch.optim.Optimizer, iteration: int, out: str):
    torch.save({
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)


def load_checkpoint(src: str, optimizer: torch.optim.Optimizer = None):
    checkpoint = torch.load(src)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

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

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.prompting import get_prompt, get_grade

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

if __name__ == "__main__":
    n_train_examples = 6400
    n_val_examples = 1024
    num_rollout_steps = 200
    learning_rate = 1e-5
    rollout_batch_size = train_batch_size = 32
    group_size = 4
    gradient_accumulation_steps = 8
    sampling_temperature = 1.0
    sampling_max_tokens = 512
    max_grad_norm = 1.0
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"

    ds = load_dataset("openai/gsm8k", "main")
    ds_train_len = len(ds["train"])

    continue_from = -1
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
        sampling_params = {
            "n": group_size,
            "seed": 42,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512
        }
        r1_zero_params = {
            "stop": ["</answer>"],
            "include_stop_str_in_output": True
        }
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
            with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero_three_shot_gsm8k.prompt"), "r") as fin:
                prompt_template = fin.read()

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