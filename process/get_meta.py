import os
import json
import re

ROOT_DIR = "general_pair_new"

def extract_duration(instruction):
    """从 instruction 中提取时长，支持 target duration 和 total duration 两种格式"""
    match = re.match(r"(?:target|total)\s+duration:\s*(\d+)s", instruction)
    if match:
        return int(match.group(1))
    return None

def process_sample(sample_dir):
    shuffle_path = os.path.join(sample_dir, "input", "shuffle_mask_name.json")
    input_path = os.path.join(sample_dir, "input", "input.json")

    if not os.path.exists(shuffle_path) or not os.path.exists(input_path):
        return None

    with open(shuffle_path, "r", encoding="utf-8") as f:
        shuffle_map = json.load(f)

    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 找出干扰素材：原始 id 以 3_ 开头，取映射后的数字编号
    distractor = []
    for orig_id, mapped_id in shuffle_map.items():
        if orig_id.startswith("3_"):
            distractor.append(int(mapped_id))
    distractor.sort()

    # 提取时长
    duration = extract_duration(input_data.get("instruction", ""))

    return {
        "distractor": distractor,
        "duration": duration
    }

def main():
    metadata = {}

    for sample_name in sorted(os.listdir(ROOT_DIR)):
        sample_dir = os.path.join(ROOT_DIR, sample_name)
        if not os.path.isdir(sample_dir):
            continue

        result = process_sample(sample_dir)
        if result is None:
            print(f"[SKIP] {sample_name}: missing files")
            continue
        if result["duration"] is None:
            print(f"[WARN] {sample_name}: failed to extract duration")

        metadata[sample_name] = result

    # 写入 metadata.json
    output_path = os.path.join(ROOT_DIR, "metadata.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Done. Total samples: {len(metadata)}, saved to {output_path}")

if __name__ == "__main__":
    main()
