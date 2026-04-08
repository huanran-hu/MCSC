"""
eval.py
=======
Evaluate a generated script using pre-extracted visual features and Qwen2.5-VL.

This script loads:
  1. Pre-extracted post_merger features from safetensors files
  2. A generated script (JSON list) from --script_path
  3. Prompt templates from prompt/eval.py (PREFIX_PROMPT + 6 evaluation prompts)

It runs 6 independent evaluation passes (one per evaluation dimension),
parses the <score> from each response, and saves all results to a single JSON.

Usage:
    python scripts/eval.py \
        --video_id 286638572610 \
        --features_root ./MCSC \
        --all_input_json ./In-Domain_test/input.json \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --max_new_tokens 4096 \
        --script_path ./path/to/script.json \
        --output_path ./path/to/eval_result.json \
        --device cuda
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ---------------------------------------------------------------------------
# Make project root importable so we can do `from prompt.eval import ...`
# Project structure:
#   project_root/
#     prompt/
#       eval.py             (contains PREFIX_PROMPT, single_prompt dict)
#     scripts/
#       eval.py             (this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompt.eval import PREFIX_PROMPT, single_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("eval")


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

        Expected keys in the safetensors file:
          - "post_merger_embeds": Tensor[N, hidden_size]
          - "image_grid_thw": Tensor[1, 3]

        Args:
            feature_path: Path to features.safetensors file.
            device: Target device.
            dtype: Target dtype for float tensors.

        Returns:
            dict with "post_merger_embeds" and "image_grid_thw".
        """
        tensors = load_file(feature_path)
        result = {
            "post_merger_embeds": tensors["post_merger_embeds"].to(device, dtype=dtype),
            "image_grid_thw": tensors["image_grid_thw"].to(device),
        }
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
        paths. This method preserves the interleaved structure.

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
def build_prefix_prompt(video_info: dict) -> str:
    """
    Build the prefix prompt by replacing placeholders in PREFIX_PROMPT
    with actual content from input.json.

    Placeholders replaced:
        <video_material>  -> video_info["video_material"]
        <text_material>   -> video_info["text_material"]
        <instruction>     -> video_info["instruction"]

    Args:
        video_info: Dict for one video_id from input.json.

    Returns:
        Filled prefix prompt string.
    """
    prefix = PREFIX_PROMPT
    prefix = prefix.replace("<video_material>", video_info.get("video_material", ""))
    prefix = prefix.replace("<text_material>", video_info.get("text_material", ""))
    prefix = prefix.replace("<instruction>", video_info.get("instruction", ""))
    return prefix


def build_suffix_prompt(eval_prompt_template: str, script_json: list) -> str:
    """
    Build a suffix prompt by inserting the script JSON into the
    evaluation prompt template.

    The <script> placeholder in the template is replaced with the
    pretty-printed JSON string of the script.

    Note: <video_material> in the suffix templates is intentionally kept
    as-is (it serves as a reference label, not a placeholder to fill).

    Args:
        eval_prompt_template: One of the EVAL_*_PROMPT strings.
        script_json: The script as a Python list of dicts.

    Returns:
        Filled suffix prompt string.
    """
    script_str = json.dumps(script_json, indent=2, ensure_ascii=False)
    return eval_prompt_template.replace("<script>", script_str)


def parse_score(response: str) -> int:
    """
    Parse the evaluation score from model response.

    Looks for the pattern `<score>: X` or `<score>:X` where X is an integer.

    Args:
        response: Raw model response string.

    Returns:
        Parsed integer score (1-5), or -1 if parsing fails.
    """
    # Try to find <score>: followed by a digit
    match = re.search(r"<score>\s*[:：]\s*(\d+)", response)
    if match:
        score = int(match.group(1))
        if 1 <= score <= 5:
            return score
        logger.warning(f"Parsed score {score} is out of range [1, 5]")
        return score

    # Fallback: look for just <score> followed by a digit somewhere
    match = re.search(r"<score>\s*(\d+)", response)
    if match:
        score = int(match.group(1))
        return score

    logger.warning(f"Failed to parse score from response: ...{response[-200:]}")
    return -1


# ================================================================
# Inference Engine (Qwen2.5-VL)
# ================================================================
class Qwen25VLFeatureInference:
    """
    Inference with pre-extracted visual features on Qwen2.5-VL.

    This is the same engine as in inference_with_features.py, adapted for
    evaluation (no DeepStack, Qwen2.5-VL-specific RoPE index).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = torch_dtype

        logger.info(f"Loading model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
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

        # Qwen2.5-VL: hidden_size at config.hidden_size (not config.text_config)
        self.hidden_size = self.model.config.hidden_size
        logger.info(f"Model loaded. hidden_size={self.hidden_size}")

    def _build_interleaved_prompt(
        self,
        interleaved_items: list[dict],
        prefix_prompt: str = "",
        suffix_prompt: str = "",
    ) -> tuple:
        """
        Build input_ids and merged features from interleaved items.

        Prompt layout:
            [prefix_prompt]
            [clip1_name + frame1 + frame2 + ...]
            [clip2_name + frame1 + ...]
            ...
            [suffix_prompt]

        Returns:
            (input_ids, all_image_embeds, merged_grid_thw)
        """
        content_parts = []
        all_image_embeds = []
        all_grid_thw = []

        # Prefix prompt (with placeholders already filled)
        if prefix_prompt:
            content_parts.append({"type": "text", "text": prefix_prompt})

        # Interleaved clip names and frame features
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

        # Suffix prompt
        if suffix_prompt:
            content_parts.append({"type": "text", "text": f"\n\n{suffix_prompt}"})

        messages = [{"role": "user", "content": content_parts}]

        # Apply chat template to get the full text with vision placeholders
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Replace each vision placeholder segment with the correct number
        # of <|image_pad|> tokens matching actual feature lengths
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

            rebuilt_parts.append(remaining[:vs_pos])

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

        # Tokenize
        input_ids = self.tokenizer.encode(
            text_rebuilt, return_tensors="pt"
        ).to(self.device)

        # Merge grid_thw: (1, 3) per frame -> (num_images, 3)
        merged_grid_thw = torch.cat(all_grid_thw, dim=0)

        return input_ids, all_image_embeds, merged_grid_thw

    @torch.no_grad()
    def generate(
        self,
        interleaved_items: list[dict],
        prefix_prompt: str = "",
        suffix_prompt: str = "",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.8,
        do_sample: bool = True,
    ) -> str:
        """
        Run inference with interleaved image-text features.

        Args:
            interleaved_items: Output of FeatureLoader.load_from_name_image_list().
            prefix_prompt: Filled prefix prompt string.
            suffix_prompt: Filled suffix prompt string (with script inserted).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p (nucleus) sampling threshold.
            do_sample: Whether to use sampling or greedy decoding.

        Returns:
            Generated text string.
        """
        # 1. Build input_ids and features
        input_ids, all_image_embeds, image_grid_thw = \
            self._build_interleaved_prompt(
                interleaved_items, prefix_prompt, suffix_prompt
            )

        attention_mask = torch.ones_like(input_ids)

        # 2. Compute M-RoPE position_ids
        # Qwen2.5-VL requires an extra `second_per_grid_ts` parameter
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        # 3. Build inputs_embeds and inject post_merger features
        def _get_embed_tokens(model):
            """Locate the text embedding layer in the model."""
            for name, module in model.named_modules():
                if name.endswith("embed_tokens") and hasattr(module, "weight"):
                    return module
            raise RuntimeError("Cannot find embed_tokens module")

        embed_tokens = _get_embed_tokens(self.model)
        inputs_embeds = embed_tokens(input_ids)

        # Concatenate all frames' embeddings
        all_embeds_cat = torch.cat(all_image_embeds, dim=0)

        # Replace image_pad positions with visual embeddings
        image_mask = (input_ids == self.image_pad_id)
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask_expanded, all_embeds_cat
        )

        # 4. Generate
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

        # Extract only generated tokens
        generated_ids = outputs[0][input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response


# ================================================================
# Main
# ================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a generated script across 6 dimensions using Qwen2.5-VL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/eval.py \\
        --video_id 286638572610 \\
        --features_root ./MCSC \\
        --all_input_json ./In-Domain_test/input.json \\
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \\
        --max_new_tokens 4096 \\
        --script_path ./results/script.json \\
        --output_path ./results/eval_result.json \\
        --device cuda
        """,
    )

    parser.add_argument(
        "--video_id", type=str, required=True,
        help="Video ID to evaluate, e.g., 286638572610",
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
        "--script_path", type=str, required=True,
        help="Path to the generated script JSON file (a JSON list of shot dicts)",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model name or local path (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum number of tokens to generate (default: 4096)",
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
        help="Path to save evaluation result JSON. "
             "Default: <features_root>/<video_id>/eval_result.json",
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
    logger.info(f"  instruction: {video_info.get('instruction', 'N/A')[:80]}...")
    logger.info(f"  text_material: {video_info.get('text_material', 'N/A')[:80]}...")
    logger.info(f"  video_material: {video_info.get('video_material', 'N/A')[:80]}...")

    # ---- 2. Load the generated script ----
    logger.info(f"Loading script from: {args.script_path}")
    with open(args.script_path, "r", encoding="utf-8") as f:
        script_json = json.load(f)

    if not isinstance(script_json, list):
        logger.error("Script file must contain a JSON list of shot dicts.")
        return

    logger.info(f"Script contains {len(script_json)} shots")

    # ---- 3. Build the prefix prompt (shared across all 6 evaluations) ----
    prefix_prompt = build_prefix_prompt(video_info)
    logger.info(f"Prefix prompt length: {len(prefix_prompt)} chars")

    # ---- 4. Load pre-extracted features ----
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

    # ---- 5. Load model ----
    inferencer = Qwen25VLFeatureInference(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=torch.bfloat16,
    )

    # ---- 6. Run 6 evaluation passes ----
    eval_results = {}
    dimension_names = list(single_prompt.keys())

    for dim_idx, dim_name in enumerate(dimension_names, 1):
        logger.info(
            f"[{dim_idx}/{len(dimension_names)}] Evaluating dimension: {dim_name}"
        )

        # Build suffix prompt with script inserted
        eval_template = single_prompt[dim_name]
        suffix_prompt = build_suffix_prompt(eval_template, script_json)

        # Run inference
        response = inferencer.generate(
            interleaved_items=interleaved_items,
            prefix_prompt=prefix_prompt,
            suffix_prompt=suffix_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.no_sample,
        )

        # Parse score from response
        score = parse_score(response)
        logger.info(f"  {dim_name}: score={score}")

        eval_results[dim_name] = {
            "response": response,
            "score": score,
        }

    # ---- 7. Save results ----
    if args.output_path:
        result_path = Path(args.output_path)
    else:
        result_path = Path(args.features_root) / video_id / "eval_result.json"

    os.makedirs(result_path.parent, exist_ok=True)

    result = {
        "video_id": video_id,
        "model_name": args.model_name,
        "script_path": args.script_path,
        "total_frames": n_frames,
        "total_visual_tokens": n_tokens,
        "scores": {
            dim_name: eval_results[dim_name]["score"]
            for dim_name in dimension_names
        },
        "details": eval_results,
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- 8. Print summary ----
    print("\n" + "=" * 80)
    print(f"Evaluation Results for video_id={video_id}")
    print("=" * 80)
    for dim_name in dimension_names:
        score = eval_results[dim_name]["score"]
        print(f"  {dim_name:30s} : {score}")
    valid_scores = [
        eval_results[d]["score"]
        for d in dimension_names
        if eval_results[d]["score"] > 0
    ]
    if valid_scores:
        avg = sum(valid_scores) / len(valid_scores)
        print(f"  {'average':30s} : {avg:.2f}")
    print("=" * 80)

    logger.info(f"Result saved to: {result_path}")


if __name__ == "__main__":
    main()
