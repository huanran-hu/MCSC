import os
import json
import random
import re

ROOT_DIR = "general_pair_new"

def needs_shuffle(shuffle_map):
    """检查是否所有 key == value（即未打乱）"""
    return all(str(k) == str(v) for k, v in shuffle_map.items())

def process_sample(sample_dir):
    shuffle_path = os.path.join(sample_dir, "input", "shuffle_mask_name.json")
    input_path = os.path.join(sample_dir, "input", "input.json")

    if not os.path.exists(shuffle_path) or not os.path.exists(input_path):
        return False

    with open(shuffle_path, "r", encoding="utf-8") as f:
        shuffle_map = json.load(f)

    if not needs_shuffle(shuffle_map):
        return False

    # ---- Step 1: 生成新的随机映射 ----
    original_ids = list(shuffle_map.keys())  # e.g., ["1_1", "1_2", "1_3", "1_4", "3_1"]
    new_numbers = list(range(1, len(original_ids) + 1))  # [1, 2, 3, 4, 5]
    random.shuffle(new_numbers)

    # 新映射: {"1_1": 3, "1_2": 5, "1_3": 1, "1_4": 2, "3_1": 4}
    new_shuffle_map = {oid: new_numbers[i] for i, oid in enumerate(original_ids)}

    # 反向映射: 原始id -> 新编号, 用于替换 input.json
    # e.g., "1_1" -> 3, 即 "1_1.mp4" -> "3.mp4"
    id_to_new = {oid: str(num) for oid, num in new_shuffle_map.items()}

    # ---- Step 2: 更新 input.json ----
    with open(input_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # 处理 name_image_list
    new_name_image_list = []
    for item in input_data["name_image_list"]:
        if item.endswith(".mp4"):
            # 提取原始 video_id, e.g., "1_1.mp4" -> "1_1"
            orig_id = item.replace(".mp4", "")
            if orig_id in id_to_new:
                new_name_image_list.append(f"{id_to_new[orig_id]}.mp4")
            else:
                # 不在映射中，保持原样
                new_name_image_list.append(item)
        else:
            # 图片路径，保持不变
            new_name_image_list.append(item)
    input_data["name_image_list"] = new_name_image_list

    # 处理 video_material
    # e.g., "1_1.mp4: 6s, 1_2.mp4: 2s, ..." -> "3.mp4: 6s, 5.mp4: 2s, ..."
    vm = input_data["video_material"]
    parts = [p.strip() for p in vm.split(",")]
    new_parts = []
    for part in parts:
        match = re.match(r"(.+)\.mp4:\s*(.+)", part)
        if match:
            orig_id = match.group(1)
            duration = match.group(2)
            if orig_id in id_to_new:
                new_parts.append(f"{id_to_new[orig_id]}.mp4: {duration}")
            else:
                new_parts.append(part)
        else:
            new_parts.append(part)
    input_data["video_material"] = ", ".join(new_parts)

    # ---- Step 3: 写回文件 ----
    with open(shuffle_path, "w", encoding="utf-8") as f:
        json.dump(new_shuffle_map, f, ensure_ascii=False, indent=2)

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(input_data, f, ensure_ascii=False, indent=2)

    return True

def main():
    processed = 0
    skipped = 0

    for sample_name in sorted(os.listdir(ROOT_DIR)):
        sample_dir = os.path.join(ROOT_DIR, sample_name)
        if not os.path.isdir(sample_dir):
            continue

        if process_sample(sample_dir):
            processed += 1
            print(f"[PROCESSED] {sample_name}")
        else:
            skipped += 1

    print(f"Done. Processed: {processed}, Skipped (already shuffled or missing files): {skipped}")

if __name__ == "__main__":
    random.seed(42)
    main()
