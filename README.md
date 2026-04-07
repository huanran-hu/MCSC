# MCSC-Bench: Multimodal Context-to-Script Creation for Realistic Video Production

**Huanran Hu, Zihui Ren, Dingyi Yang, Liangyu Chen, Qixiang Gao, Tiezheng Ge, Qin Jin**
Renmin University of China, Alibaba Group, Nanyang Technological University

##  Supplementary Material
For more details on dataset annotation, human evaluation, additional case studies, etc, please refer to [supplementary material](supplementary.pdf).

## MCSC-Bench Release :loudspeaker:

Download link: https://huggingface.co/datasets/huanranhu-ruc/MCSC.

### In-Domain Test

We provides pre-extracted **Qwen3-VL-8B** visual features including two formats: (ViT output before the Merger and output after the Merger),  and **Qwen2.5-VL-7B** (ViT output). Features are stored in safetensors format, enabling inference without raw video files or the Vision Encoder.

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

3. Download the data: Download the pre-extracted features from the link below and unzip: https://huggingface.co/datasets/huanranhu-ruc/MCSC/In-Domain_test.

4. Run inference
You can customize prefix_prompt and suffix_prompt to compose, using video_material, instruction, and text_material in `In-Domain_test/input.json`. `scripts/inference_with_features.py` is for performing a sample inference, using **Qwen3-VL-8B**  Merger output:

```bash
python scripts/inference_with_features.py \
    --video_id 286638572610 \
    --features_root ./MCSC \
    --all_input_json ./In-Domain_test/input.json \
    --prefix_prompt "..." \
    --suffix_prompt "..." \
    --max_new_tokens 4096
```

### Out-Of-Domain Test
In Out-Of-Domain_test, we provide general OOD test set. It is designed for direct inference. Each sample contains frames from multiple video clips along with structured textual inputs. Unzip [Out-Of-Domain_test/frames.zip](https://huggingface.co/datasets/huanranhu-ruc/MCSC/blob/main/Out-Of-Domain_test/frames.zip) and unzip it to the frames/ directory. 
You can also customize prefix_prompt and suffix_prompt.

### Train
https://huggingface.co/datasets/huanranhu-ruc/MCSC/tree/main/train

### Eval




## Models

Our Evaluator Model: https://huggingface.co/huanranhu-ruc/MCSC_evaluator

Our Trained Model on the MCSC-Bench train set: MCSC-8B: https://huggingface.co/huanranhu-ruc/MCSC-8B



## Additional Main Results

### Data Construction Pipeline

Overview of the MCSC-Bench dataset construction. Video materials are drawn from a large video pool.

![pipeline](images/pipeline.png)

### Multi-Dimensional Evaluation

Multi-dimensional evaluation on MCSC-Bench (rescaled by maximum and minimum for better visualization) shows a clear performance ladder across models.

![radar](images/radar.png)

### Full Results on Out-of-Domain Test

Due to page limits in the main paper, we only report partial results. Below we list the complete performance of all evaluated models.

![English](images/English.png)


## License, Ethics, and Access

This dataset is released under the CC BY-NC-ND 4.0 License, with additional restrictions. Specifically: (1) Attribution — proper credit must be given when using this dataset; (2) NonCommercial — only academic and research use is permitted; (3) NoDerivatives & No Redistribution — the dataset may not be redistributed, remixed, or adapted without prior written consent. We adopt this license to protect source data privacy and comply with upstream platform terms of service. The accompanying source code is released under the MIT License.

This research was conducted in strict adherence to the Code of Ethics and Professional Conduct. All data used in this work derived from publicly available websites and does not contain personally identifiable information or offensive content. For human evaluation, the annotators we recruited possess a high level of education. They were fairly compensated for their time and effort in rating the generated scripts according to our multi-dimensional evaluation criteria.
By downloading or using the MCSC-Bench dataset, you agree to all the following terms.

### Academic Use Only
This dataset is available for academic research purposes only. Any commercial use is strictly prohibited.

### No Redistribution
You may not redistribute the dataset in any form without prior written consent from the authors.

### Privacy Protection
Chinese data is derived from e-commerce videos under authorized institutional access. All visual content is released exclusively as de-identified features extracted via the frequently-used vision encoders (e.g., Qwen3-VL-8B, Qwen2.5-VL-7B); no raw images or videos are distributed for privacy reasons. Researchers requiring features from alternative encoders may contact us at [huanranhu@ruc.edu.cn] for assistance.

### Copyright and Takedown Policy
Out-Of-Domain test set contains sampled frames from publicly available YouTube and TikTok videos. We reference the Vript dataset for video selection; all video content is  sourced from public platforms. We respect the privacy of personal information of the original source. If you are a copyright holder and believe any content infringes your rights, please contact [huanranhu@ruc.edu.cn].

### Disclaimer
You are solely responsible for legal liability arising from your use of this dataset. The authors reserve the right to modify or terminate access at any time and shall not be liable for any damages arising from its use.




