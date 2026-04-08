"""
inference_with_features.py
==========================
Load pre-extracted visual features (post_merger + deepstack) from safetensors,
bypass the Vision Encoder, and directly feed them into Qwen3-VL's Language Model
for inference. Supports: multi-video, multi-frame, image-text interleaved input.

The prompt is constructed by:
1. Loading PREFIX_PROMPT and SUFFIX_PROMPT from prompt/compose.py
2. Replacing <video_material>, <text_material>, <instruction> placeholders
   with actual content from input.json
3. Appending interleaved video frames (clip_name + frame features)
4. Appending SUFFIX_PROMPT

Usage:
    python scripts/inference_with_features.py \
        --video_id 286638572610 \
        --features_root ./data/In-Domain_test-features \
        --all_input_json ./data/In-Domain_test/input.json \
        --model_name Qwen/Qwen3-VL-8B-Instruct \
        --max_new_tokens 2048
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Make project root importable so we can do `from prompt.compose import ...`
# Project structure:
#   project_root/
#     prompt/
#       compose.py          (contains PREFIX_PROMPT, SUFFIX_PROMPT)
#     scripts/
#       inference_with_features.py   (this file)
#     data/
#       ...
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompt.compose import PREFIX_PROMPT, SUFFIX_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("inference")


# ================================================================
# Feature Loader
# ================================================================
class FeatureLoader:
    """Load pre-extracted visual features from safetensors files."""

    @staticmethod
    def load_single_feature(
        feature_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """
        Load a single frame's features from a safetensors file.

        Returns:
            dict with keys:
                - "post_merger_embeds": Tensor[N, hidden_size]
                - "pre_merger_embeds": Tensor[M, vision_dim] (optional)
                - "deepstack_features": list[Tensor]
                - "image_grid_thw": Tensor[1, 3]
        """
        tensors = load_file(feature_path)
        result = {
            "post_merger_embeds": tensors["post_merger_embeds"].to(device, dtype=dtype),
            "image_grid_thw": tensors["image_grid_thw"].to(device),
        }

        if "pre_merger_embeds" in tensors:
            result["pre_merger_embeds"] = tensors["pre_merger_embeds"].to(device, dtype=dtype)

        ds_features = []
        i = 0
        while f"deepstack_feature_{i:02d}" in tensors:
            ds_features.append(tensors[f"deepstack_feature_{i:02d}"].to(device, dtype=dtype))
            i += 1
        result["deepstack_features"] = ds_features

        return result

    @staticmethod
    def load_from_name_image_list(
        name_image_list: list[str],
        features_root: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[dict]:
        """
        Load features according to name_image_list from input.json.

        The list interleaves video clip names (e.g. "1.mp4") and feature file
        paths (e.g. "286638572610/features/1_2/000001/features.safetensors").

        Returns:
            list[dict]: Each item is either:
                - {"type": "clip_name", "name": "1.mp4"}
                - {"type": "feature", "name": "...", "data": <feature_dict>}
        """
        items = []
        features_root = Path(features_root)

        for entry in name_image_list:
            if entry.endswith(".mp4"):
                items.append({"type": "clip_name", "name": entry})
            else:
                full_path = features_root / entry
                if not full_path.exists():
                    logger.warning(f"Feature file not found, skipping: {full_path}")
                    continue
                feat = FeatureLoader.load_single_feature(
                    str(full_path), device=device, dtype=dtype
                )
                items.append({"type": "feature", "name": entry, "data": feat})

        n_clips = sum(1 for item in items if item["type"] == "clip_name")
        n_frames = sum(1 for item in items if item["type"] == "feature")
        logger.info(f"Loaded {n_clips} clip markers and {n_frames} frames")

        return items


# ================================================================
# Prompt Builder
# ================================================================
def build_prompt_from_input(video_info: dict) -> tuple[str, str]:
    """
    Build the prefix and suffix prompts by filling placeholders in
    PREFIX_PROMPT / SUFFIX_PROMPT with fields from input.json.

    Placeholders replaced in PREFIX_PROMPT:
        <video_material>  -> video_info["video_material"]
        <text_material>   -> video_info["text_material"]
        <instruction>     -> video_info["instruction"]

    The interleaved video frames are NOT part of the prefix string; they are
    injected separately between the prefix and suffix as image tokens.

    Returns:
        (prefix_prompt, suffix_prompt)
    """
    prefix = PREFIX_PROMPT
    prefix = prefix.replace("<video_material>", video_info.get("video_material", ""))
    prefix = prefix.replace("<text_material>", video_info.get("text_material", ""))
    prefix = prefix.replace("<instruction>", video_info.get("instruction", ""))

    suffix = SUFFIX_PROMPT

    return prefix, suffix


# ================================================================
# Inference Engine
# ================================================================
class Qwen3VLFeatureInference:
    """
    Inference with pre-extracted visual features on Qwen3-VL.

    Prompt layout fed to the model:
        <prefix_prompt>
        1.mp4:
        <image><image><image>
        2.mp4:
        <image><image>...
        ...
        <suffix_prompt>
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = torch_dtype

        logger.info(f"Loading model: {model_name}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer

        # Special token IDs
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.hidden_size = self.model.config.text_config.hidden_size

        logger.info(f"Model loaded. hidden_size={self.hidden_size}")

    def _build_interleaved_prompt(
        self,
        interleaved_items: list[dict],
        prefix_prompt: str = "",
        suffix_prompt: str = "",
    ) -> tuple:
        """
        Build input_ids and merged features from interleaved items.

        Final message content layout:
            [prefix_prompt text]
            [clip_name_1 text] [image] [image] ...
            [clip_name_2 text] [image] [image] ...
            ...
            [suffix_prompt text]

        Returns:
            (input_ids, all_image_embeds, merged_grid_thw, all_deepstack_features)
        """
        content_parts = []
        all_image_embeds = []
        all_grid_thw = []
        all_deepstack_features = []

        # --- Prefix prompt (with placeholders already filled) ---
        if prefix_prompt:
            content_parts.append({"type": "text", "text": prefix_prompt})

        # --- Interleaved clip names and frame features ---
        for item in interleaved_items:
            if item["type"] == "clip_name":
                content_parts.append({
                    "type": "text",
                    "text": f"\n{item['name']}:\n",
                })
            elif item["type"] == "feature":
                content_parts.append({"type": "image"})
                feat = item["data"]
                all_image_embeds.append(feat["post_merger_embeds"])
                all_grid_thw.append(feat["image_grid_thw"])
                all_deepstack_features.append(feat.get("deepstack_features", []))

        # --- Suffix prompt ---
        if suffix_prompt:
            content_parts.append({"type": "text", "text": f"\n\n{suffix_prompt}"})

        # Wrap as a chat message and apply chat template
        messages = [{"role": "user", "content": content_parts}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Replace each <|vision_start|><|image_pad|><|vision_end|> placeholder
        # with the correct number of <|image_pad|> tokens matching the actual
        # feature sequence length of each frame.
        vision_start_token = "<|vision_start|>"
        vision_end_token = "<|vision_end|>"
        image_pad_token = "<|image_pad|>"

        rebuilt_parts = []
        remaining = text
        embed_idx = 0

        while True:
            vs_pos = remaining.find(vision_start_token)
            if vs_pos == -1:
                rebuilt_parts.append(remaining)
                break
            ve_pos = remaining.find(vision_end_token)
            if ve_pos == -1:
                rebuilt_parts.append(remaining)
                break

            # Text before this vision segment
            rebuilt_parts.append(remaining[:vs_pos])

            # Build the vision segment with the correct token count
            num_tokens = all_image_embeds[embed_idx].shape[0]
            vision_section = (
                vision_start_token
                + image_pad_token * num_tokens
                + vision_end_token
            )
            rebuilt_parts.append(vision_section)
            embed_idx += 1

            remaining = remaining[ve_pos + len(vision_end_token):]

        text_rebuilt = "".join(rebuilt_parts)

        # Tokenize the reconstructed text
        input_ids = self.tokenizer.encode(text_rebuilt, return_tensors="pt").to(self.device)

        # Merge grid_thw: each is (1, 3) -> (num_images, 3)
        merged_grid_thw = torch.cat(all_grid_thw, dim=0)

        return input_ids, all_image_embeds, merged_grid_thw, all_deepstack_features

    @torch.no_grad()
    def generate(
        self,
        interleaved_items: list[dict],
        prefix_prompt: str = "",
        suffix_prompt: str = "",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True,
    ) -> str:
        """
        Run inference with interleaved image-text features.

        Args:
            interleaved_items: Output of FeatureLoader.load_from_name_image_list().
            prefix_prompt: Filled prefix prompt string.
            suffix_prompt: Suffix prompt string.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling threshold.
            do_sample: Whether to use sampling or greedy decoding.

        Returns:
            Generated text string.
        """
        # 1. Build input_ids and features
        input_ids, all_image_embeds, image_grid_thw, all_deepstack = \
            self._build_interleaved_prompt(interleaved_items, prefix_prompt, suffix_prompt)

        attention_mask = torch.ones_like(input_ids)

        # 2. Compute M-RoPE position_ids
        position_ids, rope_deltas = self.model.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        # 3. Build inputs_embeds and inject post_merger features
        def _get_embed_tokens(model):
            for name, module in model.named_modules():
                if name.endswith("embed_tokens") and hasattr(module, "weight"):
                    return module
            raise RuntimeError("Cannot find embed_tokens module")

        embed_tokens = _get_embed_tokens(self.model)
        inputs_embeds = embed_tokens(input_ids)

        # Concatenate all frames' post_merger_embeds into one tensor
        all_embeds_cat = torch.cat(all_image_embeds, dim=0)  # (total_tokens, hidden_size)

        # Replace all image_pad positions with visual embeddings
        image_mask = (input_ids == self.image_pad_id)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, all_embeds_cat)

        # 4. Merge DeepStack features across frames
        # all_deepstack: list[list[Tensor]], outer=frames, inner=layers
        num_ds_layers = max((len(ds) for ds in all_deepstack), default=0)
        merged_deepstack = None
        if num_ds_layers > 0:
            merged_deepstack = []
            for layer_idx in range(num_ds_layers):
                layer_features = []
                for frame_ds in all_deepstack:
                    if layer_idx < len(frame_ds):
                        layer_features.append(frame_ds[layer_idx])
                if layer_features:
                    merged_deepstack.append(torch.cat(layer_features, dim=0))

        # 5. Generate
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        outputs = self.model.generate(input_ids=input_ids, **gen_kwargs)

        # Extract only generated tokens (skip the input portion)
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response


# ================================================================
# Main
# ================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-video feature-based inference with Qwen3-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/inference_with_features.py \\
        --video_id 286638572610 \\
        --features_root ./data/In-Domain_test-features \\
        --all_input_json ./data/In-Domain_test/input.json \\
        --model_name Qwen/Qwen3-VL-8B-Instruct \\
        --max_new_tokens 2048
        """,
    )

    parser.add_argument(
        "--video_id", type=str, required=True,
        help="Video ID to process, e.g., 286638572610",
    )
    parser.add_argument(
        "--features_root", type=str, required=True,
        help="Root directory of pre-extracted features",
    )
    parser.add_argument(
        "--all_input_json", type=str, required=True,
        help="Path to input.json containing video metadata and name_image_list",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048,
        help="Maximum number of tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.8,
        help="Top-p (nucleus) sampling threshold (default: 0.8)",
    )
    parser.add_argument(
        "--no_sample", action="store_true",
        help="Use greedy decoding instead of sampling",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save result JSON. Default: <features_root>/<video_id>/inference_result.json",
    )

    args = parser.parse_args()

    # ---- 1. Load input.json and locate the target video ----
    logger.info(f"Loading input.json from: {args.all_input_json}")
    with open(args.all_input_json, "r", encoding="utf-8") as f:
        all_input = json.load(f)

    video_id = args.video_id
    if video_id not in all_input:
        logger.error(
            f"Video ID '{video_id}' not found in input.json. "
            f"Available IDs (first 10): {list(all_input.keys())[:10]}..."
        )
        return

    video_info = all_input[video_id]
    name_image_list = video_info["name_image_list"]
    logger.info(f"Video ID: {video_id}")
    logger.info(f"  name_image_list has {len(name_image_list)} entries")
    logger.info(f"  instruction: {video_info.get('instruction', 'N/A')[:80]}...")
    logger.info(f"  text_material: {video_info.get('text_material', 'N/A')[:80]}...")
    logger.info(f"  video_material: {video_info.get('video_material', 'N/A')[:80]}...")

    # ---- 2. Build prompts by filling placeholders ----
    prefix_prompt, suffix_prompt = build_prompt_from_input(video_info)
    logger.info(f"Prefix prompt length: {len(prefix_prompt)} chars")
    logger.info(f"Suffix prompt length: {len(suffix_prompt)} chars")

    # ---- 3. Load pre-extracted features ----
    interleaved_items = FeatureLoader.load_from_name_image_list(
        name_image_list=name_image_list,
        features_root=args.features_root,
        device=args.device,
        dtype=torch.bfloat16,
    )

    if not any(item["type"] == "feature" for item in interleaved_items):
        logger.error("No feature files loaded! Check --features_root path.")
        return

    n_frames = sum(1 for item in interleaved_items if item["type"] == "feature")
    n_tokens = sum(
        item["data"]["post_merger_embeds"].shape[0]
        for item in interleaved_items
        if item["type"] == "feature"
    )
    logger.info(f"Loaded {n_frames} frames, {n_tokens} visual tokens total")

    # ---- 4. Run inference ----
    inferencer = Qwen3VLFeatureInference(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )

    response = inferencer.generate(
        interleaved_items=interleaved_items,
        prefix_prompt=prefix_prompt,
        suffix_prompt=suffix_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    # ---- 5. Output results ----
    print("\n" + "=" * 80)
    print("Inference Result")
    print("=" * 80)
    print(response)
    print("=" * 80)

    # Save result to JSON
    if args.output_path:
        result_path = Path(args.output_path)
    else:
        result_path = Path(args.features_root) / video_id / "inference_result.json"

    os.makedirs(result_path.parent, exist_ok=True)
    result = {
        "video_id": video_id,
        "total_frames": n_frames,
        "total_visual_tokens": n_tokens,
        "prefix_prompt": prefix_prompt,
        "suffix_prompt": suffix_prompt,
        "response": response,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
