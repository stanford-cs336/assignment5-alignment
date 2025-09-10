import json
import os
from collections.abc import Callable

from datasets import load_dataset
from drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams

"""
Section 3.2 : Zero-shot MATH Baseline
"""

TEMPLATE_PATH = "cs336_alignment/prompts/r1_zero.prompt"


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    results = []
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    num_correct = 0
    num_formated_but_not_correct = 0
    num_unformated = 0
    for output, ground_truth in zip(outputs, ground_truths):
        reward = reward_fn(output.outputs[0].text, ground_truth)
        result = {
            "response": output.outputs[0].text,
            "ground_truth": ground_truth,
            "reward": reward,
        }
        if reward["reward"] == 1.0:
            num_correct += 1
        elif reward["format_reward"] == 1.0:
            num_formated_but_not_correct += 1
        else:
            num_unformated += 1
        results.append(result)

    print(f"Number of correct responses: {num_correct}/{len(prompts)}")
    print(f"Number of formatted but incorrect responses: {num_formated_but_not_correct}/{len(prompts)}")
    print(f"Number of unformatted responses: {num_unformated}/{len(prompts)}")

    os.makedirs("cs336_alignment/results", exist_ok=True)
    json.dump(results, open("cs336_alignment/results/r1_zero.json", "w"), indent=4)


model_name = "Qwen/Qwen2.5-Math-1.5B"
ds = load_dataset("hiyouga/math12k", split="test")
template = open(TEMPLATE_PATH).read()
formatted_prompts = [template.format(question=problem) for problem in ds["problem"]]
ground_truths = list(ds["answer"])
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    stop=["</answer>"],
    include_stop_str_in_output=True,
)
llm = LLM(model=model_name)
evaluate_vllm(
    vllm_model=llm,
    reward_fn=r1_zero_reward_fn,
    prompts=formatted_prompts,
    ground_truths=ground_truths,
    eval_sampling_params=sampling_params,
)
