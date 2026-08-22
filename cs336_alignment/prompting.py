import os
from collections import defaultdict

import numpy as np
import pandas as pd

from datasets import load_dataset
from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

templates = dict()
with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero.prompt"), "r") as fin:
    templates["r1_zero"] = fin.read()
with open(os.path.join(SCRIPT_DIR, "prompts/r1_zero_three_shot_gsm8k.prompt"), "r") as fin:
    templates["three_shot"] = fin.read()
with open(os.path.join(SCRIPT_DIR, "prompts/question_only.prompt"), "r") as fin:
    templates["question_only"] = fin.read()

## transform GSM8K dataset to R1-zero format
def get_prompt(sample: dict, template: str) -> tuple[str, str, str, str]:
    question = sample["question"].replace("\n", " ").strip()
    response = sample["answer"].replace("\n", " ").split("####")
    assert len(response) == 2, f"Expected only one delimiter ####, got {len(response) - 1}"
    reasoning = response[0].strip()
    answer = response[1].strip()
    prompt = template.format(question=question)
    return prompt, question, reasoning, answer

def get_grade(response: str, ground_truth: str, prompt_key: str) -> dict:
    if prompt_key in ["r1_zero", "three_shot"]:
        return r1_zero_reward_fn(response=response, ground_truth=ground_truth)
    elif prompt_key == "question_only":
        return question_only_reward_fn(response=response, ground_truth=ground_truth)
    else:
        raise ValueError(f"Unknown prompt_key: {prompt_key}")

## adhoc script for sanity check
if __name__ == "__main__":
    ds = load_dataset("openai/gsm8k", "main")
    vllm_server = VLLMServer(model_id = "allenai/OLMo-2-0425-1B", gpu = 0, startup_timeout = 600)
    vllm_server.start()

    try:
        index_list = [i for i in range(100)]
        sampling_params = {
            "n": 1,
            "seed": 42,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 512
        }
        r1_zero_params = {
            "stop": ["</answer>"],
            "include_stop_str_in_output": True
        }
        for prompt_key in templates.keys():
            prompts = defaultdict(list)
            for index in index_list:
                sample = ds["train"][index]
                prompt, question, reasoning, answer = get_prompt(sample, templates[prompt_key])
                prompts["prompt"].append(prompt)
                prompts["question"].append(question)
                prompts["reasoning"].append(reasoning)
                prompts["answer"].append(answer)

            reward_output = defaultdict(list)
            with open(f"./{prompt_key}_completions.txt", "w") as fout:
                if prompt_key in ["r1_zero", "three_shot"]:
                    completions = vllm_server.generate_completions(prompts = prompts["prompt"], sampling_params = sampling_params | r1_zero_params)
                else:
                    completions = vllm_server.generate_completions(prompts = prompts["prompt"], sampling_params = sampling_params)
                for i, completion in enumerate(completions):
                    question, answer = prompts["question"][i], prompts["answer"][i]
                    response = completion.text.replace("\n", " ").strip()
                    fout.write(f"Question: {question}\n")
                    fout.write(f"Answer: {answer}\n")
                    fout.write(f"Response: {response}\n")
                    for key, value in get_grade(response=response, ground_truth=answer, prompt_key=prompt_key).items():
                        reward_output[key].append(value)
                        fout.write(f"{key}: {value} ")
                    fout.write("\n\n")
                pd.DataFrame(reward_output).to_csv(f"./{prompt_key}_reward.csv", index = False)
    finally:
        vllm_server.stop()