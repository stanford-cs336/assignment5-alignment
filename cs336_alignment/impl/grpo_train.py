import torch
import logging
from typing import Literal

from datasets import load_dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cs336_alignment.impl.mathBaseline import evaluate_vllm
from cs336_alignment.impl.sft_train import init_vllm, load_r1_zero_prompt_template, load_policy_into_vllm_instance
from cs336_alignment.impl.util import get_response_log_probs, tokenize_prompt_and_output, \
    compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

TRAIN_DEVICE = "cuda:0"
VLLM_DEVICE = "cuda:1"
EVALUATION_STEP = 10
EVAL_SAMPLING_PARAM = SamplingParams(
        temperature=0.7,  # Lower temperature for more focused outputs
        top_p=0.95,       # Slightly lower top_p
        max_tokens=2048,  # More tokens for reasoning
        stop=["</answer>", "</answer >", "User:", "\n\n\n"],  # Multiple stop sequences
        include_stop_str_in_output=True
    )


def grpo_train( policy: PreTrainedModel,
                optimizer: Optimizer,
                vllm_instance: LLM,
                n_grpo_steps: int = 200,
                advantage_eps: float = 1e-6,
                rollout_batch_size: int = 256,
                group_size: int = 8,
                sampling_temperature: float = 1.0,
                sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses
                sampling_max_tokens: int = 1024,
                epochs_per_rollout_batch: int = 1, # On-policy
                train_batch_size: int = 256, # On-policy
                gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100
                clip_range = 0.2,
                loss_type: Literal[
                "no_baseline",
                "reinforce_with_baseline",
                "grpo_clip",
                ] = "reinforce_with_baseline",
                use_std_normalization: bool = True,
                ):
    policy.train()  # Set to training mode
    if torch.cuda.is_available():
        policy = policy.to(TRAIN_DEVICE)
    ds = load_dataset("gsm8k", "main")
    prompt_template = load_r1_zero_prompt_template()
    valid_prompts = [prompt_template.format(question=example['question']) for example in ds['test']]
    valid_ground_truths = [example['answer'] for example in ds['test']]
    '''assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size'''
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    # Format training data with r1_zero template
    raw_questions = [example['question'] for example in ds['train']]
    raw_answers = [example['answer'] for example in ds['train']]


    # Apply r1_zero formatting to prompts only (answers used raw for reward function)
    formatted_prompts = [prompt_template.format(question=q) for q in raw_questions]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        n=group_size,  # Generate group_size outputs per prompt
    )
    global_step = 0

    logger.info(f"Starting GRPO training with {n_grpo_steps} steps")
    logger.info(f"Loss type: {loss_type}, rollout_batch_size: {rollout_batch_size}, group_size: {group_size}")
    logger.info(f"train_batch_size: {train_batch_size}, micro_batch_size: {micro_train_batch_size}, grad_accum_steps: {gradient_accumulation_steps}")

    for i in range(n_grpo_steps):
        load_policy_into_vllm_instance(policy, vllm_instance)
        current_batch_prompts = formatted_prompts[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        current_batch_answer = raw_answers[i * n_prompts_per_rollout_batch: (i + 1) * n_prompts_per_rollout_batch]
        repeated_batch_prompt = []
        repeated_ground_truths = []
        for j in range(n_prompts_per_rollout_batch):
            repeated_ground_truths+=[current_batch_answer[j]]* group_size
            repeated_batch_prompt+=[current_batch_prompts[j]]* group_size

        vllm_outputs = vllm_instance.generate(current_batch_prompts, sampling_params)
        rollout_responses = []
        for request_output in vllm_outputs:  # One per input prompt
            for completion in request_output.outputs:  # group_size completions
                rollout_responses.append(completion.text)
        advantage, raw_rewards, metadata = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization)
        logger.info(f"Rollout batch {i}: mean_reward={metadata['mean_reward']:.4f}, std_reward={metadata['std_reward']:.4f}")
        advantage = advantage.unsqueeze(-1).to(TRAIN_DEVICE)
        raw_rewards = raw_rewards.unsqueeze(-1).to(TRAIN_DEVICE)

        tokenized = tokenize_prompt_and_output(repeated_batch_prompt, rollout_responses, tokenizer) #batch_size seq_len
        input_ids = tokenized['input_ids'].to(TRAIN_DEVICE) #batch_size seq_len
        labels = tokenized['labels'].to(TRAIN_DEVICE) #batch_size seq_len
        response_mask = tokenized['response_mask'].to(TRAIN_DEVICE)

        policy.eval()
        with torch.no_grad():
            old_log_probs = get_response_log_probs(policy, input_ids, labels, True)['log_probs']
            # old_log_probs is already on TRAIN_DEVICE since input_ids/labels are

        policy.train()
        for _ in range(epochs_per_rollout_batch):
            accumulated_loss = 0.0
            for idx, train_step in enumerate(range(0, train_batch_size, micro_train_batch_size)):
                new_log_prob = get_response_log_probs(policy, input_ids[train_step:train_step+micro_train_batch_size], labels[train_step:train_step+micro_train_batch_size], True)['log_probs']
                loss, loss_metadata = grpo_microbatch_train_step(new_log_prob, response_mask[train_step:train_step+micro_train_batch_size],
                                           gradient_accumulation_steps, loss_type, raw_rewards[train_step:train_step+micro_train_batch_size],
                                           advantage[train_step:train_step+micro_train_batch_size],
                                           old_log_probs[train_step:train_step+micro_train_batch_size], clip_range)
                accumulated_loss += loss.item()
                if (idx + 1) % gradient_accumulation_steps == 0:
                    clip_grad_norm_(policy.parameters(), 1.0)
                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    logger.info(f"Step {global_step}: loss={accumulated_loss:.4f}")
                    accumulated_loss = 0.0
                    if global_step % EVALUATION_STEP == 0:
                        logger.info(f"Running evaluation at step {global_step}...")
                        load_policy_into_vllm_instance(policy, vllm_instance)
                        eval_metrics = evaluate_vllm(vllm_instance, r1_zero_reward_fn, valid_prompts, valid_ground_truths, EVAL_SAMPLING_PARAM, step_number=global_step)
                        logger.info(f"Evaluation results - accuracy: {eval_metrics['accuracy']:.4f}, format_reward: {eval_metrics['format_reward']:.4f}")





if __name__ == "__main__":
    current_checkpoint = "TODO"
    policy = AutoModelForCausalLM.from_pretrained(current_checkpoint).to(TRAIN_DEVICE)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    vllm_instance = init_vllm(current_checkpoint, seed=42, device=VLLM_DEVICE, gpu_memory_utilization=0.85)
    grpo_train(policy, optimizer, vllm_instance)

