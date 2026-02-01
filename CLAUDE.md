# CS336 Assignment 5: Alignment

Stanford CS336 Spring 2025 assignment on LLM alignment techniques.

## Git Remotes

- **origin** (primary): `https://github.com/Kangaroo-Yuchi/assignment5-alignment.git` - personal fork for pushing work
- **upstream**: `https://github.com/stanford-cs336/assignment5-alignment.git` - original course repo

```bash
git push origin main          # Push your changes
git fetch upstream && git merge upstream/main  # Sync course updates
```

## Project Structure

### Core Implementation (`cs336_alignment/impl/`)

| File | Purpose |
|------|---------|
| `util.py` | Core utilities: tokenization (`tokenize_prompt_and_output`), log prob computation (`get_response_log_probs`), entropy calculation, masked operations, SFT training step |
| `sft_train.py` | SFT training loop using Qwen2.5-Math-1.5B on GSM8K with vLLM evaluation |
| `mathBaseline.py` | vLLM-based evaluation harness for math problems (GSM8K) |

### Grading & Rewards (`cs336_alignment/`)

| File | Purpose |
|------|---------|
| `drgrpo_grader.py` | Math answer grading with high recall (from R1-Zero). Handles LaTeX parsing, numeric comparison, symbolic equality. Main functions: `r1_zero_reward_fn`, `grade` |

### Test Adapters (`tests/adapters.py`)

Functions to implement for the assignment:
- **Tokenization**: `run_tokenize_prompt_and_output`
- **Policy Gradient**: `run_compute_naive_policy_gradient_loss`, `run_compute_grpo_clip_loss`, `run_compute_group_normalized_rewards`
- **DPO**: `run_compute_per_instance_dpo_loss`
- **SFT**: `get_packed_sft_dataset`, `run_iterate_batches`
- **Evaluation**: `run_parse_mmlu_response`, `run_parse_gsm8k_response`
- **Utilities**: `run_masked_mean`, `run_masked_normalize`, `run_get_response_log_probs`, `run_compute_entropy`

## Quick Commands

```bash
uv sync --no-install-package flash-attn && uv sync  # Setup
uv run pytest                                        # Run tests
uv run python cs336_alignment/impl/sft_train.py     # Run SFT training
```

## Dependency Management

When fixing environment or dependency issues, **always update `pyproject.toml`** rather than using one-time `uv pip install` commands. This ensures changes are preserved for future deployments.

### Guidelines

1. **Version constraints**: Add minimum version constraints when a specific version is required for compatibility
   ```toml
   "datasets>=3.0.0",      # Required for huggingface_hub compatibility
   "fsspec>=2024.2.0",     # Fixes glob pattern bug
   ```

2. **After updating pyproject.toml**, run:
   ```bash
   uv sync
   ```

3. **Common compatibility issues encountered**:
   - `fsspec` + `huggingface_hub` + `datasets`: Ensure all three are recent versions. Old `datasets` (2.x) doesn't work with new `fsspec`/`huggingface_hub`
   - FlashInfer vs Flash Attention: Set `VLLM_ATTENTION_BACKEND` appropriately. Use `FLASH_ATTN` if FlashInfer isn't installed
   - Blackwell GPUs (sm_120): Require PyTorch nightly or future stable releases; current stable PyTorch only supports up to sm_90

