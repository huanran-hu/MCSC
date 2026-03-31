# MCScript: Benchmarking Realistic Video Script Creation from Multimodal Long Contexts

**Huanran Hu, Zihui Ren, Dingyi Yang, Liangyu Chen, Qixiang Gao, Tiezheng Ge, Qin Jin**

## Overview
For more details on dataset contruction, Evaluator–Human Agreement, etc, please refer to [Additional Main Results](## Additional Main Results).
For more details on dataset annotation, human evaluation, additional case studies, etc, please refer to [supplementary material](supplementary.pdf).


## MCSC-GEN

### MCSC-ZH

We provides pre-extracted Qwen3-VL visual features for multi-video script generation research. Features are stored in safetensors format, enabling inference **without** raw video files or the Vision Encoder.

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
每个sample包括来自多个视频片段的帧，输入视频素材清单、文本素材、用户指令，进行推理
MCSC-GEN/frames.zip，包含所有sample的图片帧数据，请将其解压到frames目录下
MCSC-GEN/input.json中，每个item代表一个sample，name_image_list包含该sample的视频id与视频帧路径。video_material，text_material，instruction分别表示视频素材清单，文本素材，用户指令，推理时一并输入模型。注意视频id和视频帧要使用图文交错的形式，即name_image_list
MCSC-GEN/metadata.json中，每个item有distractor，和duration，字段，标明了干扰素材




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

### Evaluator–Human Agreement

We validate the reliability of our Evaluator Model by measuring agreement with human judgments using Spearman's ρ (rank), Kendall's τ (rank), and Pearson's r (value). \* denotes p-value < 0.001, indicating statistical significance.

![human](images/human.png)

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

