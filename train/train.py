"""
train.py
========
Full-parameter fine-tuning of Qwen3-VL (Merger + LLM) with pre-extracted ViT features.
Frozen ViT → Cached pre_merger_embeds → Trainable Merger → Trainable LLM

Architecture:
    [Cached .safetensors] → [Trainable Merger] → [Trainable LLM] → autoregressive loss

Usage:
    # Single node, 8x A100 (run from project root)
    deepspeed --num_gpus=8 train/train.py --config train/config.yaml

Requirements:
    - transformers==4.57.1
    - torch>=2.6.0
    - deepspeed>=0.16.0
    - safetensors
    - pyyaml
    - flash-attn (for flash_attention_2)
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import prompt/compose.py
# Project structure:
#   project_root/
#   ├── prompt/
#   │   └── compose.py          (PREFIX_PROMPT, SUFFIX_PROMPT)
#   └── train/
#       ├── train.py            (this file)
#       ├── dataset.py
#       ├── config.yaml
#       └── ds_config.json
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent           # train/
_PROJECT_ROOT = _THIS_DIR.parent                       # project_root/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from prompt.compose import PREFIX_PROMPT, SUFFIX_PROMPT  # noqa: E402
from dataset import CachedFeatureDataset, DataCollatorForCachedFeatures  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


# ================================================================
# Configuration
# ================================================================
@dataclass
class ModelConfig:
    """Model and feature configuration."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    # Processor settings (must match extraction config)
    min_pixels: int = 200704   # 256 * 28 * 28
    max_pixels: int = 1003520  # 1280 * 28 * 28


@dataclass
class DataConfig:
    """Data paths and processing configuration."""
    input_json: str = ""             # Path to input.json
    feature_root: str = ""           # Root dir for .safetensors features
    max_frames_per_video: int = 8    # Max frames per video; uniform sample if exceeded
    max_seq_length: int = 8192       # Max token sequence length


@dataclass
class PromptConfig:
    """
    Prompt templates loaded from prompt/compose.py.
    Placeholders <video_material>, <text_material>, <instruction>
    will be replaced at runtime with actual sample data.
    """
    prefix_prompt: str = PREFIX_PROMPT
    suffix_prompt: str = SUFFIX_PROMPT


def load_config(config_path: str) -> dict:
    """Load YAML config file and return as nested dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_config_from_dict(cfg: dict):
    """
    Build dataclass configs from YAML dict.
    Prompt config uses defaults from prompt/compose.py;
    YAML overrides are applied if present.
    """
    model_cfg = ModelConfig(**cfg.get("model", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))

    # PromptConfig defaults come from prompt/compose.py.
    # Allow optional overrides from YAML (partial or full).
    prompt_overrides = cfg.get("prompt", {})
    prompt_cfg = PromptConfig(
        prefix_prompt=prompt_overrides.get("prefix_prompt", PREFIX_PROMPT),
        suffix_prompt=prompt_overrides.get("suffix_prompt", SUFFIX_PROMPT),
    )

    return model_cfg, data_cfg, prompt_cfg


# ================================================================
# Model Setup: Freeze ViT, keep Merger + LLM trainable
# ================================================================
def setup_model(model_cfg: ModelConfig):
    """
    Load Qwen3-VL model and freeze the ViT (visual encoder blocks).
    Only Merger + LLM remain trainable.

    Frozen modules:
        - model.visual.patch_embed
        - model.visual.blocks (all ViT transformer blocks)
        - model.visual.rotary_pos_emb

    Trainable modules:
        - model.visual.merger (PatchMerger)
        - model.model (LLM decoder)
        - model.lm_head
    """
    dtype = getattr(torch, model_cfg.torch_dtype)

    logger.info(f"Loading model: {model_cfg.model_name}")
    logger.info(f"dtype={model_cfg.torch_dtype}, attn={model_cfg.attn_implementation}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_cfg.model_name,
        torch_dtype=dtype,
        attn_implementation=model_cfg.attn_implementation,
    )

    # ---- Freeze ViT encoder (keep Merger trainable) ----
    frozen_params = 0
    trainable_params = 0

    freeze_prefixes = [
        "visual.patch_embed",
        "visual.blocks",
        "visual.rotary_pos_emb",
    ]

    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(prefix) for prefix in freeze_prefixes)
        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    total_params = frozen_params + trainable_params
    logger.info(
        f"Parameter summary:"
        f"  Total:     {total_params / 1e9:.2f}B"
        f"  Trainable: {trainable_params / 1e9:.2f}B "
        f"({100 * trainable_params / total_params:.1f}%)"
        f"  Frozen:    {frozen_params / 1e9:.2f}B "
        f"({100 * frozen_params / total_params:.1f}%)"
    )

    # ---- Verify Merger is trainable ----
    merger_trainable = any(
        p.requires_grad for n, p in model.named_parameters() if "visual.merger" in n
    )
    assert merger_trainable, "Merger must be trainable!"
    logger.info("Verified: visual.merger is trainable")

    # ---- Enable gradient checkpointing for memory efficiency ----
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    return model


def setup_processor(model_cfg: ModelConfig):
    """Load processor (tokenizer + image processor)."""
    processor = AutoProcessor.from_pretrained(
        model_cfg.model_name,
        min_pixels=model_cfg.min_pixels,
        max_pixels=model_cfg.max_pixels,
    )
    return processor


# ================================================================
# Custom Trainer: Override visual forward to use cached features
# ================================================================
class CachedFeatureTrainer(Trainer):
    """
    Custom Trainer that bypasses ViT forward pass by using cached
    pre_merger_embeds. The model.visual.forward() is monkey-patched
    at each step to skip ViT blocks and directly feed cached features
    to the trainable Merger.

    Data flow during training:
        1. DataCollator builds input_ids with <|image_pad|> placeholders
        2. DataCollator attaches cached pre_merger_embeds
        3. In compute_loss, we monkey-patch model.visual to use cached features
        4. model.forward() → get_image_features() → patched visual() → Merger only
        5. LLM processes merged visual tokens + text tokens → autoregressive loss
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to inject cached pre_merger_embeds.

        Qwen3-VL forward does:
            if pixel_values is not None:
                image_embeds = get_image_features(pixel_values, image_grid_thw)
                inputs_embeds.masked_scatter(image_mask, image_embeds)

        get_image_features() calls self.visual(pixel_values, grid_thw=...)
        which internally: patch_embed → blocks → merger → return

        We monkey-patch visual.forward() to skip patch_embed + blocks,
        directly passing cached features to merger.
        """
        # Pop cached features from inputs (set by DataCollator)
        cached_pre_merger = inputs.pop("cached_pre_merger_embeds", None)
        cached_grid_thw = inputs.pop("cached_image_grid_thw", None)

        if cached_pre_merger is not None:
            # Get the visual module (handle DDP wrapping)
            visual_module = (
                model.module.visual if hasattr(model, "module") else model.visual
            )
            original_forward = visual_module.forward

            def patched_visual_forward(hidden_states, grid_thw=None, **kwargs):
                """
                Patched forward: skip ViT blocks, use cached pre_merger_embeds.
                Only runs the trainable Merger on cached features.

                Original flow: patch_embed → blocks → merger → return
                Patched flow:  cached_features → merger → return
                """
                merger_device = next(visual_module.merger.parameters()).device
                merger_dtype = next(visual_module.merger.parameters()).dtype

                pre_merger = cached_pre_merger.to(
                    device=merger_device, dtype=merger_dtype,
                )

                # Run merger on cached features
                merged = visual_module.merger(pre_merger, grid_thw=grid_thw)

                # Match return format of original visual forward
                from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                    BaseModelOutputWithDeepstackFeatures,
                )
                return BaseModelOutputWithDeepstackFeatures(
                    last_hidden_state=merged,
                    deepstack_features=None,
                )

            # Monkey-patch, run forward, restore
            visual_module.forward = patched_visual_forward
            try:
                # Ensure pixel_values is not None to trigger visual branch
                if "pixel_values" not in inputs or inputs["pixel_values"] is None:
                    inputs["pixel_values"] = torch.zeros(
                        1, dtype=torch.float16,
                        device=next(visual_module.parameters()).device,
                    )
                if cached_grid_thw is not None:
                    inputs["image_grid_thw"] = cached_grid_thw

                loss = super().compute_loss(
                    model, inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                visual_module.forward = original_forward

            return loss
        else:
            return super().compute_loss(
                model, inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )


# ================================================================
# Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-VL with cached ViT features"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--deepspeed", type=str, default=None,
        help="Path to DeepSpeed config JSON (overrides config.yaml)"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank for distributed training (set by torchrun/deepspeed)"
    )
    args, remaining_args = parser.parse_known_args()

    # ---- Load config ----
    cfg = load_config(args.config)
    model_cfg, data_cfg, prompt_cfg = build_config_from_dict(cfg)

    # Log which prompts are being used
    logger.info(
        f"Prompt source: prompt/compose.py"
        f"  PREFIX_PROMPT length: {len(prompt_cfg.prefix_prompt)} chars"
        f"  SUFFIX_PROMPT length: {len(prompt_cfg.suffix_prompt)} chars"
    )

    # ---- Training arguments ----
    train_args_dict = cfg.get("training", {})
    if args.deepspeed:
        train_args_dict["deepspeed"] = args.deepspeed
    training_args = TrainingArguments(**train_args_dict)

    # ---- Seed ----
    set_seed(training_args.seed)
    logger.info(f"Seed: {training_args.seed}")

    # ---- Load model & processor ----
    model = setup_model(model_cfg)
    processor = setup_processor(model_cfg)
    tokenizer = processor.tokenizer

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token_id = eos_token_id = {tokenizer.eos_token_id}")

    # ---- Build dataset ----
    logger.info(f"Loading dataset from: {data_cfg.input_json}")
    logger.info(f"Feature root: {data_cfg.feature_root}")
    logger.info(f"Max frames per video: {data_cfg.max_frames_per_video}")

    train_dataset = CachedFeatureDataset(
        input_json=data_cfg.input_json,
        feature_root=data_cfg.feature_root,
        processor=processor,
        prompt_cfg=prompt_cfg,
        max_frames_per_video=data_cfg.max_frames_per_video,
        max_seq_length=data_cfg.max_seq_length,
    )
    logger.info(f"Dataset size: {len(train_dataset)} samples")

    # ---- Data collator ----
    data_collator = DataCollatorForCachedFeatures(
        tokenizer=tokenizer,
        max_seq_length=data_cfg.max_seq_length,
    )

    # ---- Resume from checkpoint ----
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")

    # ---- Trainer ----
    trainer = CachedFeatureTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ---- Train ----
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- Save ----
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
