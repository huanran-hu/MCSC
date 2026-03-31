import os
import json
import shutil
import random

# 缺少 3_ 开头子目录的列表
missing_list = [

    "I-nikl0IQYg"
]

base_dir = "general_pair_new"

def get_donor_dirs(base_dir, missing_set):
    """获取所有不在 missing_list 中的子目录（作为捐赠源）"""
    donors = []
    for d in os.listdir(base_dir):
        d_path = os.path.join(base_dir, d)
        if not os.path.isdir(d_path):
            continue
        if d in missing_set:
            continue
        # 确认该目录下有 frames/1_1
        frames_1_1 = os.path.join(d_path, "frames", "1_1")
        if os.path.isdir(frames_1_1):
            # 确认里面有 jpg
            jpgs = [f for f in os.listdir(frames_1_1) if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                donors.append(d)
    return donors


def process(base_dir, missing_list):
    missing_set = set(missing_list)
    donors = get_donor_dirs(base_dir, missing_set)
    
    if not donors:
        print("没有可用的捐赠目录!")
        return
    
    print(f"可用捐赠目录数量: {len(donors)}")
    
    for target_name in missing_list:
        target_path = os.path.join(base_dir, target_name)
        if not os.path.isdir(target_path):
            print(f"[跳过] 目标目录不存在: {target_path}")
            continue
        
        # =====================
        # 1. 从其他子目录复制 frames/1_1 -> 目标的 frames/3_1
        # =====================
        donor_name = random.choice(donors)
        src_path = os.path.join(base_dir, donor_name, "frames", "1_1")
        dst_path = os.path.join(target_path, "frames", "3_1")
        
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        
        shutil.copytree(src_path, dst_path)
        
        # 获取复制后 3_1 下的所有 jpg，排序
        new_jpgs = sorted([f for f in os.listdir(dst_path) if f.lower().endswith(".jpg")])
        num_frames = len(new_jpgs)
        
        print(f"[复制] {src_path} -> {dst_path}  ({num_frames} 帧, 来源: {donor_name})")
        
        # =====================
        # 2. 修改 input/input.json
        # =====================
        input_json_path = os.path.join(target_path, "input", "input.json")
        if not os.path.isfile(input_json_path):
            print(f"  [警告] input.json 不存在: {input_json_path}")
            continue
        
        with open(input_json_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        
        # name_image_list 追加 "2.mp4" + 所有新帧路径
        name_image_list = input_data.get("name_image_list", [])
        name_image_list.append("2.mp4")
        for jpg in new_jpgs:
            name_image_list.append(f"frames/{target_name}/3_1/{jpg}")
        input_data["name_image_list"] = name_image_list
        
        # video_material 追加 ", 2.mp4: xs"
        video_material = input_data.get("video_material", "")
        video_material = video_material.rstrip().rstrip(",")
        video_material += f", 2.mp4: {num_frames}s"
        input_data["video_material"] = video_material
        
        with open(input_json_path, "w", encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False, indent=2)
        
        print(f"  [更新] input.json: 添加 2.mp4 + {num_frames} 帧路径")
        
        # =====================
        # 3. 修改 input/shuffle_mask_name.json
        # =====================
        shuffle_json_path = os.path.join(target_path, "input", "shuffle_mask_name.json")
        if os.path.isfile(shuffle_json_path):
            with open(shuffle_json_path, "r", encoding="utf-8") as f:
                shuffle_data = json.load(f)
        else:
            shuffle_data = {}
            print(f"  [警告] shuffle_mask_name.json 不存在, 将新建: {shuffle_json_path}")
        
        shuffle_data["3_1"] = 2
        
        with open(shuffle_json_path, "w", encoding="utf-8") as f:
            json.dump(shuffle_data, f, ensure_ascii=False, indent=2)
        
        print(f"  [更新] shuffle_mask_name.json: 添加 '3_1': 2")
    
    print("全部处理完成!")


if __name__ == "__main__":
    process(base_dir, missing_list)
