"""
dataset.py
==========
Dataset and DataCollator for training Qwen3-VL with pre-extracted ViT features.

Key design:
    - Each sample loads pre_merger_embeds from .safetensors files
    - Builds interleaved text + vision prompt following Qwen3-VL format
    - Uses <|vision_start|><|image_pad|>...<|vision_end|> as visual placeholders
    - The number of <|image_pad|> tokens matches the pre_merger_embeds token count
      AFTER merger (spatial_merge_size=2 → tokens reduced by 4x)

Data flow:
    input.json sample → build prompt text → tokenize → attach cached features → collate

Directory structure expected:
    {feature_root}/
    ├── {sample_id}/
    │   ├── features/
    │   │   ├── 1_1/000001/features.safetensors
    │   │   └── ...
    │   └── gt_script.json
    └── ...
"""

import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from torch.utils.data import Dataset
from safetensors.torch import load_file

logger = logging.getLogger("dataset")


# ================================================================
# Qwen3-VL Special Token IDs
# ================================================================
# These are fixed in Qwen3-VL tokenizer:
#   <|vision_start|>  = 151652
#   <|vision_end|>    = 151653
#   <|vision_pad|>    = 151654
#   <|image_pad|>     = 151655
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"


def uniform_sample_indices(total: int, max_count: int) -> List[int]:
    """
    Uniformly sample `max_count` indices from range(total).
    Always includes first and last index.

    Example:
        uniform_sample_indices(14, 8) → [0, 2, 4, 6, 8, 10, 12, 13]
    """
    if total <= max_count:
        return list(range(total))

    indices = []
    for i in range(max_count):
        idx = round(i * (total - 1) / (max_count - 1))
        indices.append(idx)

    # Deduplicate while preserving order
    seen = set()
    result = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            result.append(idx)
    return result


def compute_post_merger_tokens(pre_merger_tokens: int, spatial_merge_size: int = 2) -> int:
    """
    Compute the number of tokens after Merger.
    Merger reduces spatial tokens by spatial_merge_size^2 (default: 4x).

    For Qwen3-VL: each image's pre_merger has N tokens,
    after merger it becomes N // (spatial_merge_size^2).
    """
    merge_factor = spatial_merge_size ** 2
    return pre_merger_tokens // merge_factor


# ================================================================
# Dataset
# ================================================================
class CachedFeatureDataset(Dataset):
    """
    Dataset that loads pre-extracted ViT features and builds training samples.

    Each sample contains:
        - input_ids: tokenized prompt with <|image_pad|> placeholders
        - labels: same as input_ids but with prompt tokens masked (-100)
        - cached_pre_merger_embeds: concatenated pre-merger features [N_total, D]
        - cached_image_grid_thw: stacked grid info [N_images, 3]

    The prompt format:
        <|im_start|>system
You are a helpful assistant.<|im_end|>
        <|im_start|>user
{prefix_prompt}
        1.mp4:
        <|vision_start|><|image_pad|>...<|vision_end|>
        <|vision_start|><|image_pad|>...<|vision_end|>
        ...
        2.mp4:
        <|vision_start|><|image_pad|>...<|vision_end|>
        ...
        {suffix_prompt}<|im_end|>
        <|im_start|>assistant
{gt_script_json}<|im_end|>
    """

    def __init__(
        self,
        input_json: str,
        feature_root: str,
        processor,
        prompt_cfg,
        max_frames_per_video: int = 8,
        max_seq_length: int = 8192,
        spatial_merge_size: int = 2,
    ):
        """
        Args:
            input_json: Path to input.json containing all samples
            feature_root: Root directory for feature files
            processor: Qwen3-VL processor (tokenizer + image processor)
            prompt_cfg: PromptConfig dataclass with prefix/suffix prompts
            max_frames_per_video: Max frames per video clip; uniform sample if exceeded
            max_seq_length: Maximum token sequence length
            spatial_merge_size: Merger spatial merge size (default 2 for Qwen3-VL)
        """
        self.feature_root = Path(feature_root)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt_cfg = prompt_cfg
        self.max_frames_per_video = max_frames_per_video
        self.max_seq_length = max_seq_length
        self.spatial_merge_size = spatial_merge_size

        # Load input.json
        with open(input_json, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Build sample list: [(sample_id, sample_data), ...]
        self.samples = []
        skipped = 0
        for sample_id, sample_data in raw_data.items():
            gt_script_path = self.feature_root / sample_id / "gt_script.json"
            if not gt_script_path.exists():
                logger.warning(f"gt_script.json not found for {sample_id}, skipping")
                skipped += 1
                continue
            self.samples.append((sample_id, sample_data))

        logger.info(
            f"Loaded {len(self.samples)} samples from {input_json} "
            f"(skipped {skipped} without gt_script.json)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_name_image_list(
        self, name_image_list: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse name_image_list into structured video groups.

        Input format: ["1.mp4", "path/to/feat1.safetensors", ..., "2.mp4", ...]

        Returns:
            List of dicts: [
                {"video_id": "1.mp4", "feature_paths": ["path1", "path2", ...]},
                {"video_id": "2.mp4", "feature_paths": [...]},
                ...
            ]
        """
        groups = []
        current_group = None

        for item in name_image_list:
            if item.endswith(".mp4"):
                if current_group is not None:
                    groups.append(current_group)
                current_group = {"video_id": item, "feature_paths": []}
            elif item.endswith(".safetensors"):
                if current_group is not None:
                    current_group["feature_paths"].append(item)
            else:
                logger.warning(f"Unknown item in name_image_list: {item}")

        if current_group is not None:
            groups.append(current_group)

        return groups

    def _load_features_for_video(
        self, feature_paths: List[str]
    ) -> tuple:
        """
        Load and optionally subsample features for a single video.

        Returns:
            pre_merger_list: list of [N_i, D] tensors (one per frame)
            grid_thw_list: list of [1, 3] tensors (one per frame)
        """
        # Uniform sampling if too many frames
        if len(feature_paths) > self.max_frames_per_video:
            indices = uniform_sample_indices(
                len(feature_paths), self.max_frames_per_video
            )
            feature_paths = [feature_paths[i] for i in indices]

        pre_merger_list = []
        grid_thw_list = []

        for fp in feature_paths:
            full_path = self.feature_root / fp
            if not full_path.exists():
                logger.warning(f"Feature file not found: {full_path}, skipping frame")
                continue

            data = load_file(str(full_path))
            # pre_merger_embeds: [1, N_tokens, D_vit] → squeeze to [N_tokens, D_vit]
            pre_merger = data["pre_merger_embeds"].squeeze(0)
            grid_thw = data["image_grid_thw"]  # [1, 3]

            pre_merger_list.append(pre_merger)
            grid_thw_list.append(grid_thw)

        return pre_merger_list, grid_thw_list

    def _build_prompt_text(
        self,
        sample_id: str,
        sample_data: dict,
        video_groups: List[Dict],
        post_merger_token_counts: List[List[int]],
    ) -> tuple:
        """
        Build the full prompt text with <|image_pad|> placeholders.

        Returns:
            user_text: The user message text (with image placeholders)
            assistant_text: The target text (gt_script JSON)
        """
        # ---- Replace placeholders in prefix prompt ----
        prefix = self.prompt_cfg.prefix_prompt
        prefix = prefix.replace("<video_material>", sample_data.get("video_material", ""))
        prefix = prefix.replace("<text_material>", sample_data.get("text_material", ""))
        prefix = prefix.replace("<instruction>", sample_data.get("instruction", ""))

        # ---- Build interleaved video frames section ----
        video_sections = []
        for group_idx, group in enumerate(video_groups):
            video_id = group["video_id"]
            section_lines = [f"{video_id}:"]

            token_counts = post_merger_token_counts[group_idx]
            for count in token_counts:
                # Each frame: <|vision_start|><|image_pad|>*N<|vision_end|>
                image_placeholder = (
                    VISION_START_TOKEN
                    + IMAGE_PAD_TOKEN * count
                    + VISION_END_TOKEN
                )
                section_lines.append(image_placeholder)

            video_sections.append("".join(section_lines))

        video_text = "".join(video_sections)

        # ---- Compose user message ----
        user_text = prefix + video_text + self.prompt_cfg.suffix_prompt

        # ---- Load gt_script as assistant response ----
        gt_script_path = self.feature_root / sample_id / "gt_script.json"
        with open(gt_script_path, "r", encoding="utf-8") as f:
            gt_script = json.load(f)
        assistant_text = json.dumps(gt_script, ensure_ascii=False, indent=2)

        return user_text, assistant_text

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Build a single training sample.

        Returns dict with:
            - input_ids: [seq_len] int64
            - attention_mask: [seq_len] int64
            - labels: [seq_len] int64 (prompt tokens masked as -100)
            - cached_pre_merger_embeds: [N_total_tokens, D_vit]
            - cached_image_grid_thw: [N_total_images, 3]
        """
        sample_id, sample_data = self.samples[idx]

        # ---- Parse video groups from name_image_list ----
        video_groups = self._parse_name_image_list(
            sample_data["name_image_list"]
        )

        # ---- Load all features ----
        all_pre_merger = []     # flat list of [N_i, D] tensors
        all_grid_thw = []       # flat list of [1, 3] tensors
        post_merger_counts = [] # nested: [[count_frame1, count_frame2, ...], ...]

        for group in video_groups:
            pre_merger_list, grid_thw_list = self._load_features_for_video(
                group["feature_paths"]
            )
            group_counts = []
            for pm, gthw in zip(pre_merger_list, grid_thw_list):
                n_tokens = pm.shape[0]
                post_count = compute_post_merger_tokens(
                    n_tokens, self.spatial_merge_size
                )
                group_counts.append(post_count)
                all_pre_merger.append(pm)
                all_grid_thw.append(gthw)

            post_merger_counts.append(group_counts)

        # ---- Build prompt text ----
        user_text, assistant_text = self._build_prompt_text(
            sample_id, sample_data, video_groups, post_merger_counts
        )

        # ---- Construct chat messages for tokenizer ----
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

        # Apply chat template to get full text
        # We need to tokenize user + assistant together, then mask user part in labels
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        input_ids = encoding["input_ids"].squeeze(0)       # [seq_len]
        attention_mask = encoding["attention_mask"].squeeze(0)

        # ---- Build labels: mask prompt tokens with -100 ----
        # Strategy: tokenize user part (up to assistant response), mask those positions
        prompt_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt tokens (only train on response)

        # ---- Concatenate cached features ----
        if len(all_pre_merger) > 0:
            cached_pre_merger = torch.cat(all_pre_merger, dim=0)  # [N_total, D]
            cached_grid_thw = torch.cat(all_grid_thw, dim=0)      # [N_images, 3]
        else:
            cached_pre_merger = None
            cached_grid_thw = None

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if cached_pre_merger is not None:
            result["cached_pre_merger_embeds"] = cached_pre_merger
            result["cached_image_grid_thw"] = cached_grid_thw

        return result


# ================================================================
# Data Collator
# ================================================================
class DataCollatorForCachedFeatures:
    """
    Collator that pads input_ids/labels/attention_mask to batch max length,
    and concatenates cached visual features across the batch.

    For cached_pre_merger_embeds, since each sample may have different
    numbers of visual tokens, we concatenate them and let the model's
    internal logic handle splitting via image_grid_thw.
    """

    def __init__(self, tokenizer, max_seq_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Handles:
            - Left-padding for input_ids, attention_mask
            - Left-padding for labels (with -100)
            - Concatenation of cached features across batch
        """
        batch_size = len(features)

        # ---- Pad text tensors ----
        max_len = min(
            max(f["input_ids"].shape[0] for f in features),
            self.max_seq_length,
        )

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            if seq_len > max_len:
                # Truncate from right
                input_ids_batch.append(f["input_ids"][:max_len])
                attention_mask_batch.append(f["attention_mask"][:max_len])
                labels_batch.append(f["labels"][:max_len])
            elif seq_len < max_len:
                # Pad from left (standard for causal LM)
                pad_len = max_len - seq_len
                input_ids_batch.append(
                    torch.cat([
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                        f["input_ids"],
                    ])
                )
                attention_mask_batch.append(
                    torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        f["attention_mask"],
                    ])
                )
                labels_batch.append(
                    torch.cat([
                        torch.full((pad_len,), -100, dtype=torch.long),
                        f["labels"],
                    ])
                )
            else:
                input_ids_batch.append(f["input_ids"])
                attention_mask_batch.append(f["attention_mask"])
                labels_batch.append(f["labels"])

        batch = {
            "input_ids": torch.stack(input_ids_batch),           # [B, max_len]
            "attention_mask": torch.stack(attention_mask_batch),  # [B, max_len]
            "labels": torch.stack(labels_batch),                 # [B, max_len]
        }

        # ---- Concatenate cached visual features ----
        # Qwen3-VL expects all image features concatenated, with image_grid_thw
        # recording each image's grid shape for splitting inside the model.
        all_pre_merger = []
        all_grid_thw = []
        has_features = False

        for f in features:
            if "cached_pre_merger_embeds" in f and f["cached_pre_merger_embeds"] is not None:
                all_pre_merger.append(f["cached_pre_merger_embeds"])
                all_grid_thw.append(f["cached_image_grid_thw"])
                has_features = True

        if has_features:
            batch["cached_pre_merger_embeds"] = torch.cat(all_pre_merger, dim=0)
            batch["cached_image_grid_thw"] = torch.cat(all_grid_thw, dim=0)
            # image_grid_thw is needed by model.forward() for position computation
            batch["image_grid_thw"] = batch["cached_image_grid_thw"]

        return batch
