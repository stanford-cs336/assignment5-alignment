import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.utils as nn_utils
import wandb
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams

from cs336_alignment.sft_dataset import SFTDataset, gsm8k_reward_fn, make_collate_fn
from cs336_alignment.utils import (
    get_reponse_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
    load_pretrained,
    masked_normalize,
)

"""
Section 4: Supervised Finetuning for MATH 
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Model
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-1
    gradient_accumulation_steps: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vllm_device: str = "cuda:1" if device == "cuda" else "cpu"
    gpu_memory_utilization: float = 0.85
    vllm_seed: int = 42

    # Data
    batch_size: int = 8

    # Training loop
    eval_interval: int = 100

    # Logging
    use_wandb: bool = True
    log_interval: int = 10
    project: str = "sft-gsm8k"


class SFTTrainer:
    def __init__(
        self,
        config: TrainConfig,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.weight_decay
        )

        self._build_dataloader(train_dataset, val_dataset)

        # For evaluating policy model
        self.vllm_model = init_vllm(
            config.model_id, config.vllm_device, config.vllm_seed, config.gpu_memory_utilization
        )
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True,
        )

    def _build_dataloader(self, train_ds, val_ds):
        self.train_dataset, self.val_dataset = train_ds, val_ds
        collate_fn = make_collate_fn(self.tokenizer)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def train(self):
        config = self.config
        self.model.train()
        self.optimizer.zero_grad()
        for i, inputs in tqdm(enumerate(self.train_loader)):
            input_ids = inputs["input_ids"].to(config.device)
            labels = inputs["labels"].to(config.device)
            response_mask = inputs["response_mask"].to(config.device)
            out = get_reponse_log_probs(self.model, input_ids, labels, return_token_entropy=False)
            policy_log_probs = out["log_probs"]
            loss, _ = sft_microbatch_train_step(
                policy_log_probs=policy_log_probs,
                response_mask=response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                normalize_constant=1.0,
            )

            if (i + 1) % config.log_interval == 0:  # light logging
                if config.use_wandb:
                    wandb.log({"train/loss": loss.item(), "train_step": i + 1})
                print(f"step {i + 1}: loss={loss.item():.4f}")

            if (i + 1) % config.gradient_accumulation_steps == 0:
                nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if (i + 1) % config.eval_interval == 0:
                logger.info(f"Evaluating policy at step {i + 1}")
                load_policy_into_vllm_instance(self.model, self.vllm_model)
                self.evaluate_policy(i + 1)

    def _evaluate_vllm(
        self, reward_fn: Callable[[str, str], int], prompts: list[str], ground_truths: list[str]
    ) -> tuple[list[dict[str, str | int]], int]:
        results = []
        num_correct = 0
        outputs = self.vllm_model.generate(prompts, self.sampling_params)
        for output, ground_truth in zip(outputs, ground_truths):
            reward = reward_fn(output.outputs[0].text, ground_truth)
            num_correct += 1 if reward == 1 else 0
            result = {
                "response": output.outputs[0].text,
                "ground_truth": ground_truth,
                "reward": reward,
            }
            results.append(result)
        return results, num_correct

    @torch.no_grad()
    def evaluate_policy(self, step: int) -> None:
        # TODO: evaluate vllm
        self.model.eval()
        results = []
        num_correct = 0
        for inputs in self.val_loader:
            prompts = inputs["prompts"]
            answers = inputs["responses"]
            batch_results, batch_correct = self._evaluate_vllm(gsm8k_reward_fn, prompts, answers)
            num_correct += batch_correct
            results.extend(batch_results)

        val_acc = num_correct / len(self.val_dataset)
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        if self.config.use_wandb:
            wandb.log({"eval/accuracy": val_acc, "eval_step": step})

        os.makedirs("cs336_alignment/results", exist_ok=True)
        json.dump(results, open(f"cs336_alignment/results/sft_gsm8k_step_{step}.json", "w"), indent=4)
        self.model.train()


def sft_microbatch_train_step(
    policy_log_probs: Float[Tensor, "B S"],
    response_mask: Float[Tensor, "B S"],
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    nll = -policy_log_probs
    per_ex: Float[Tensor, " B"] = masked_normalize(nll, response_mask, normalize_constant, dim=1)
    loss = per_ex.mean() / gradient_accumulation_steps
    loss.backward()
    return loss, {}


if __name__ == "__main__":
    config = TrainConfig()
    model, tokenizer = load_pretrained(config.model_id)
    model.to(config.device)

    if config.use_wandb:
        # --- W&B init + metric axes
        wandb.init(project=config.project, config=asdict(config))
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    train_ds = SFTDataset(config, tokenizer, "data/gsm8k/train.jsonl")
    val_ds = SFTDataset(config, tokenizer, "data/gsm8k/test.jsonl")
    trainer = SFTTrainer(config, tokenizer, model, train_ds, val_ds)
    trainer.train()
