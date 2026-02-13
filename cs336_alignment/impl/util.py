from typing import Literal

from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F

def tokenize_prompt_and_output(prompt_strs: list[str],
                               output_strs: list[str],
                               tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    tokenized_prompts = tokenizer(prompt_strs)['input_ids']
    tokenized_outputs = tokenizer(output_strs)['input_ids']
    combined = [torch.tensor(tokenized_prompts[i] + tokenized_outputs[i]) for i in range(len(tokenized_prompts))]

    # Use eos_token_id as padding if pad_token_id is not available
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    combined_padded = pad_sequence(combined, batch_first=True, padding_value=pad_token_id)
    input_ids = combined_padded[:,:-1]
    labels = combined_padded[:, 1:]
    response_mask = [torch.tensor([False for i in range(len(tokenized_prompts[j]) - 1)]) for j in range(len(labels))]
    response_mask_padded = [ F.pad(response_mask[i], (0, len(labels[i]) - len(response_mask[i])), value=True) for i in range(len(response_mask))]
    for i in range(len(response_mask_padded)):
        item = response_mask_padded[i]
        for j in range(len(item)):
            if j >= len(combined[i]) - 1:
                item[j] = False

    return dict(input_ids=input_ids, labels=labels, response_mask=torch.stack(response_mask_padded))

#logits (batch_size, sequence_length, vocab_size)
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    # m = torch.max(logits, dim=-1, keepdim=True).values # batch_size, sequence_length
    # shifted = logits - m # batch_size, sequence_length, vocab_size
    # sum = torch.sum(torch.exp(shifted), dim=-1, keepdim=True) # batch_size, sequence_length
    # prob = torch.exp(logits - m) / sum # batch_size, sequence_length, vocab_size
    # log_prob = logits - m - torch.log(torch.sum(torch.exp(logits - m), dim=-1, keepdim=True)) # batch_size, sequence_length, vocab_size
    # entropy = -torch.sum(prob * log_prob, dim=-1, keepdim=False)
    # return entropy # batch_size, sequence_length

    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq, vocab)
    probs = F.softmax(logits, dim=-1)          # (batch, seq, vocab)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor, #(batch_size, sequence_length)
    labels: torch.Tensor, #(batch_size, sequence_length)
    return_token_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
    result = {}
    logits = model(input_ids).logits #(batch_size, sequence_length, vocab_size)
    m = torch.max(logits, dim=-1, keepdim=True).values  # batch_size, sequence_length
    shifted = logits - m  # batch_size, sequence_length, vocab_size
    sum = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)  # batch_size, sequence_length
    prob = torch.exp(logits - m) / sum  # batch_size, sequence_length, vocab_size

    if return_token_entropy:
        result['token_entropy'] = compute_entropy(logits) #(batch_size, sequence_length)

    # need to index into logits to find the index for label and get #(batch_size, sequence_length) and then take log
    log_probs = torch.log(torch.gather(prob, dim=-1 ,index=labels.unsqueeze(-1))).squeeze(-1)
    result['log_probs'] = log_probs # batch_size, sequence_length, vocab_size
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor, # has the same shape as tensor
    normalize_constant: float,
    dim: int | None= None,
    ) -> torch.Tensor:
    tensor  = tensor * mask
    return torch.sum(tensor, dim=dim, keepdim=False) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    batch_size = policy_log_probs.shape[0]
    loss = - masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=None)
    loss = loss / batch_size  # Average over responses
    loss /= gradient_accumulation_steps
    loss.backward()

    return loss, {}


def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn,
    max_new_tokens: int = 512,
) -> dict:
    """Log model generations with rewards and statistics."""
    model.eval()

    logs = []
    correct_lengths = []
    incorrect_lengths = []

    with torch.no_grad():
        for prompt, ground_truth in zip(prompts, ground_truths):
            # Generate
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Decode response
            response = tokenizer.decode(output.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)

            # Get rewards
            rewards = reward_fn(response, ground_truth)

            # Compute entropy
            logits = torch.stack(output.scores, dim=0).unsqueeze(0)  # (1, seq_len, vocab_size)
            avg_entropy = compute_entropy(logits).mean().item()

            # Track lengths
            length = len(output.sequences[0]) - input_ids.shape[1]
            if rewards["answer_reward"] > 0:
                correct_lengths.append(length)
            else:
                incorrect_lengths.append(length)

            logs.append({
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": rewards["reward"],
                "format_reward": rewards["format_reward"],
                "answer_reward": rewards["answer_reward"],
                "avg_entropy": avg_entropy,
                "length": length,
            })

    # Aggregate statistics
    all_lengths = [log["length"] for log in logs]
    return {
        "logs": logs,
        "avg_length": sum(all_lengths) / len(all_lengths),
        "avg_correct_length": sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0,
        "avg_incorrect_length": sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0,
    }

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):
    raw_rewards = torch.tensor([reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"] for i in range(len(rollout_responses))])
    raw_rewards_by_group = raw_rewards.view(-1, group_size)
    raw_rewards_group_mean = raw_rewards_by_group.mean(dim = 1, keepdim = True)
    raw_rewards_by_group_normalized = raw_rewards_by_group - raw_rewards_group_mean
    if normalize_by_std:
        raw_rewards_group_std = raw_rewards_by_group.std(dim = 1, keepdim = True)
        raw_rewards_by_group_normalized = raw_rewards_by_group_normalized / (raw_rewards_group_std + advantage_eps)
    advantage = raw_rewards_by_group_normalized.view(len(repeated_ground_truths))
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "std_reward": raw_rewards.std().item(),
    }
    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor, # (batch_size, 1)
    policy_log_probs: torch.Tensor, # (batch_size, sequence_length)
    ) -> torch.Tensor:
    return torch.neg(raw_rewards_or_advantages * policy_log_probs)



def compute_grpo_clip_loss(
    advantages: torch.Tensor, #(batch_size, 1)
    policy_log_probs: torch.Tensor, #(batch_size, sequence_length)
    old_log_probs: torch.Tensor, #(batch_size, sequence_length)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, min=1-cliprange, max=1+cliprange)
    is_ratio_clipped_by_max =  torch.greater(ratio, clipped_ratio)
    is_ratio_clipped_by_min =  torch.less(ratio, clipped_ratio)

    grpo_clip_ross = torch.neg(torch.min(ratio * advantages, clipped_ratio * advantages))
    return grpo_clip_ross, dict(is_ratio_clipped_by_min=is_ratio_clipped_by_min, is_ratio_clipped_by_max=is_ratio_clipped_by_max)

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs),  {}
    else:
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)



def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
    ) -> torch.Tensor:
    masked = tensor * mask
    count = torch.sum(mask, dim = dim)
    masked_sum = torch.sum(masked, dim = dim)
    return masked_sum/count


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    loss /= gradient_accumulation_steps
    loss = masked_mean(loss, response_mask, dim=-1)
    loss = torch.mean(loss, dim = 0)
    loss.backward()
    return loss, metadata







