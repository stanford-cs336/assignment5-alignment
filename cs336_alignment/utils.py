import logging
from collections.abc import Callable
from unittest.mock import patch

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_randomseed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_entropy(logits: Float[Tensor, "B S V"]) -> Float[Tensor, "B S"]:
    probs = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probs + 1e-10)  # Adding a small value to avoid log(0)
    entropy = -torch.sum(probs * log_probabilities, dim=-1)
    return entropy


def get_reponse_log_probs(
    model: torch.nn.Module,
    input_ids: Float[Tensor, "B S"],
    labels: Float[Tensor, "B S"],
    return_token_entropy: bool = False,
) -> dict[str, Float[Tensor, "B S"]]:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    softmax_logits: Float[Tensor, "B S V"] = torch.log_softmax(logits, dim=-1)
    log_probs: Float[Tensor, "B S"] = softmax_logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    return {"log_probs": log_probs}


def masked_normalize(tensor: Tensor, mask: Tensor, normalize_constant: float, dim: int) -> Tensor:
    return ((tensor * mask).sum(dim=dim)) / normalize_constant


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_randomseed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,  # type: ignore
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def save_pretrained(model, tokenizer, output_dir):
    """Save the model and tokenizer to the specified directory."""
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


def load_pretrained(model_name):
    """Load the model and tokenizer from the specified model name."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def log_generation(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    # TODO: log average token entropy of the response
    # average response length for correct responses, and for incorrect responses
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    for i, (output, ground_truth) in enumerate(zip(outputs, ground_truths)):
        response = output.outputs[0].text
        reward = reward_fn(response, ground_truth)
        logger.info(f"Prompt {i}: {prompts[i]}")
        logger.info(f"Response {i}: {response}")
        logger.info(f"Ground Truth {i}: {ground_truth}")
        logger.info(f"Format reward {i}: {reward['format_reward']}")
        logger.info(f"Answer reward {i}: {reward['answer_reward']}")
        logger.info(f"Total reward {i}: {reward['reward']}")


if __name__ == "__main__":
    # Example usage
    dim = 2
    tensor = torch.randn(2, 5, 10)
    mask = torch.ones_like(tensor, dtype=torch.bool)
    normalize_constant = 2
    masked_normalized_tensor = masked_normalize(tensor, mask, normalize_constant, dim)
