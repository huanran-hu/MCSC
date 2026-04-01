# MCScript: Benchmarking Realistic Video Script Creation from Multimodal Long Contexts

**Huanran Hu, Zihui Ren, Dingyi Yang, Liangyu Chen, Qixiang Gao, Tiezheng Ge, Qin Jin**

## Overview
For more details on dataset contruction, Evaluator–Human Agreement, etc, please refer to [Additional Main Results](#additional-main-results).
For more details on dataset annotation, human evaluation, additional case studies, etc, please refer to [supplementary material](supplementary.pdf).


## MCSCript

### MCSC-ZH

We provides pre-extracted Qwen3-VL visual features including two formats. Features are stored in safetensors format, enabling inference without raw video files or the Vision Encoder.

#### Data Description

`all_input.json` contains all sample entries. Each item includes `instruction` (user instruction), `text_material` (text reference material), `video_material` (video clip inventory with durations), and `name_image_list` (interleaved video clip IDs and feature file paths). When constructing model input, follow `name_image_list` order to build an image-text interleaved sequence: clip IDs (e.g., `"1.mp4"`) serve as text markers and feature paths are loaded as visual embeddings.

Download the feature archive and unzip it. The directory is organized by sample ID, with each sample containing a `features/` directory. Inside, video clips are identified by sub-directory names (e.g., `1_1`, `1_2`), and each clip contains numbered frame directories (e.g., `000001`, `000002`). Each frame directory holds three files:

- **`features.safetensors`** — Contains the following tensors:
  - `post_merger_embeds`: Output of the Vision Encoder (ViT + Merger), shape `[N, hidden_size]`. This is the primary visual representation ready for direct injection into the language model.
  - `image_grid_thw`: Patch grid dimensions `[1, 3]` (temporal, height, width), required for M-RoPE position encoding.
  - `pre_merger_embeds` (optional): Raw ViT output before the Merger, shape `[M, vision_dim]`.
  - `deepstack_feature_00`, `deepstack_feature_01`, ... : Multi-level intermediate ViT features used by Qwen3-VL's DeepStack mechanism for fine-grained visual injection into LLM layers.

- **`metadata.json`** — Records the source image path, MD5 hash, original resolution, tensor shapes, and extraction device/dtype for traceability.

- **`feature_card.json`** — Full reproducibility card including model configuration, extraction parameters, library versions, and feature dimension descriptions.

Features were extracted using Qwen3-VL-8B-Instruct (transformers 4.57.1, torch 2.6.0, bfloat16). See `scripts/extract_all_features.py` for the extraction code.


#### Quick Start

**1. Clone the repository**

**2. Install dependencies**

We recommend Python ≥ 3.10 and CUDA ≥ 12.1.

```bash
# Create a virtual environment (recommended)
conda create -n mcsc python=3.10 -y
conda activate mcsc

# Install PyTorch (adjust for your CUDA version, see https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (requires CUDA toolkit)
pip install flash-attn --no-build-isolation

# Install other dependencies
pip install -r requirements.txt
```

3. Download the data

Download the pre-extracted features from the link below and unzip:
https://g20.alicdn.com/zhonghong/MCSC-ZH_backup/

4. Run inference
You can customize prefix_prompt and suffix_prompt, using video_material, instruction, and text_material in MCSC-ZH/input.json.
```bash
python scripts/inference_with_features.py \
    --video_id 286638572610 \
    --features_root ./data/MCSC-ZH-features \
    --all_input_json ./MCSC-ZH/input.json \
    --prefix_prompt "..." \
    --suffix_prompt "..." \
    --max_new_tokens 4096
```

### MCSC-GEN
MCSC-GEN is designed for **direct inference with any multimodal large language model** without pre-extracted features. Each sample contains frames from multiple video clips along with structured textual inputs. Unzip `MCSC-GEN/frames.zip` and unzip it to the `frames/` directory. `iMCSC-GEN/nput.json` contains all samples, where each item includes a `name_image_list` (interleaved video clip IDs and frame paths), `video_material` (video clip inventory with durations), `text_material` (text reference material), and `instruction` (user instruction). When constructing model input, follow the `name_image_list` order to build an **image-text interleaved** sequence: clip IDs (e.g., `"1.mp4"`) serve as text markers and frame paths are loaded as images. `MCSC-GEN/metadata.json` provides additional annotations: `distractor` indicates which clips are irrelevant distractor material, and `duration` specifies the target video length in seconds.

## Additional Main Results

### Data Construction Pipeline

Overview of the MCScript dataset construction. Video materials are drawn from a large video pool.

![pipeline](images/pipeline.png)

### Dataset Statistics

(a) Distribution of total video duration. (b) Distribution of shot duration. (c) Word cloud of video types in MCSC-Bench.

![stat](images/stat.png)

### Multi-Dimensional Evaluation

Multi-dimensional evaluation on MCSC-Bench (rescaled by maximum and minimum for better visualization) shows a clear performance ladder across models.

![radar](images/radar.png)

### Full Results on MCSC-GEN

Due to page limits in the main paper, we only report partial MCSC-GEN results. Below we list the complete performance of all evaluated models on MCSC-GEN.

![English](images/English.png)

### Long-Context Stress Test

To examine model robustness under flexible demands, we conduct a comprehensive long-context stress test from both input and output perspectives. Since ads provide sufficient available material, this stress test is specifically evaluated on MCSC-ZH.

**Input-side settings:**
- **Input ×2:** Increases the average number of shots to 12.43 while maintaining the 4:1 Available-to-Distractor ratio.
- **Noise 1:1:** Increases distractor materials to match the number of available materials.

**Output-side settings:**
- Models are required to produce scripts with **Duration ×2** and **Duration ×4** relative to the target length.

To provide a holistic assessment and discourage degenerate strategies (e.g., trivially short outputs yielding low error rates), we define an Overall Score:

$$\text{Overall} = (1 - Err) \times (1 - Rep) \times \frac{1}{1 + \Delta T}$$

which jointly penalizes material misuse, repetition, and duration deviation. The continuous penalty term 1/(1+ΔT) prevents the factor from collapsing to zero when ΔT fluctuates significantly, while Err and Rep are guaranteed to remain positive by their respective definitions.

**Analysis.** Performance decreases under most stress settings. Qwen3-VL-8B exhibits notable sensitivity to both input noise and output length. Qwen2.5-VL-72B is relatively robust to increased input noise but degrades substantially when longer outputs are required. In contrast, Gemini-2.5-Pro shows more stable performance across all dimensions. Overall, sustaining effective material selection and planning over extended input and output horizons remains challenging for current MLLMs.

![long_context](images/long_context.png)

### Video Generation Case Study

Qualitative analysis of downstream video generation from model-produced scripts.

![generation](images/generation.png)


## License, Ethics, and Access

By downloading or using the MCScript dataset, you agree to all the following terms.

### Academic Use Only
This dataset is available for academic research purposes only. Any commercial use is strictly prohibited.

### No Redistribution
You may not redistribute the dataset in any form without prior written consent from the authors.

### Privacy Protection
Chinese data is derived from e-commerce videos under authorized institutional access. All visual content is released exclusively as de-identified features extracted via the Qwen3-VL-8B vision encoder; no raw images or videos are distributed for privacy reasons. Researchers requiring features from alternative encoders (e.g., Qwen2.5-VL) may contact us at [huanranhu@ruc.edu.cn] for assistance.

### Copyright and Takedown Policy
MCSC-GEN contains sampled frames from publicly available YouTube and TikTok videos. We reference the Vript dataset for video selection; all video content is  sourced from public platforms. We respect the privacy of personal information of the original source. If you are a copyright holder and believe any content infringes your rights, please contact [huanranhu@ruc.edu.cn].

### Disclaimer
You are solely responsible for legal liability arising from your use of this dataset. The authors reserve the right to modify or terminate access at any time and shall not be liable for any damages arising from its use.

