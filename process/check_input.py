import os
import json

def check_all(base_dir="general_pair_new"):
    error_count = 0
    
    for sub_dir in sorted(os.listdir(base_dir)):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        
        errors = []
        
        # ========================
        # 检查1: frames 目录结构
        # ========================
        frames_path = os.path.join(sub_dir_path, "frames")
        if not os.path.isdir(frames_path):
            errors.append("frames 目录不存在")
        else:
            child_dirs = [d for d in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, d))]
            
            # 1a: 是否有 1_ 开头子目录
            has_1 = any(d.startswith("1_") for d in child_dirs)
            if not has_1:
                errors.append("frames 缺少 1_ 开头子目录")
            
            # 1b: 是否有 3_ 开头子目录
            has_3 = any(d.startswith("3_") for d in child_dirs)
            if not has_3:
                errors.append("frames 缺少 3_ 开头子目录")
            
            # 1c: 每个子目录至少有一个 .jpg
            for child_dir in sorted(child_dirs):
                child_dir_path = os.path.join(frames_path, child_dir)
                jpgs = [f for f in os.listdir(child_dir_path) if f.lower().endswith(".jpg")]
                if len(jpgs) == 0:
                    errors.append(f"frames/{child_dir} 无 .jpg 文件")
        
        # ========================
        # 检查2 & 3: input/input.json
        # ========================
        input_json_path = os.path.join(sub_dir_path, "input", "input.json")
        if not os.path.isfile(input_json_path):
            errors.append("input/input.json 不存在")
        else:
            try:
                with open(input_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 2: name_image_list 中至少两个含 .mp4 的 item
                name_image_list = data.get("name_image_list", [])
                mp4_items = [item for item in name_image_list if isinstance(item, str) and ".mp4" in item]
                if len(mp4_items) < 2:
                    errors.append(f"name_image_list 含 .mp4 的 item 仅 {len(mp4_items)} 个 (需 ≥2)")
                
                # 3: video_material 中至少两个含 .mp4 的字符串
                video_material = data.get("video_material", "")
                mp4_count = video_material.count(".mp4")
                if mp4_count < 2:
                    errors.append(f"video_material 含 .mp4 仅 {mp4_count} 次 (需 ≥2)")
                    
            except json.JSONDecodeError as e:
                errors.append(f"input.json 解析失败: {e}")
        
        # ========================
        # 输出不满足的子目录
        # ========================
        if errors:
            error_count += 1
            print(f"❌ {sub_dir}")
            for err in errors:
                print(f"   - {err}")
    
    # 汇总
    total = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"{'='*50}")
    print(f"总计: {total} 个子目录, 其中 {error_count} 个不满足要求, {total - error_count} 个正常")


if __name__ == "__main__":
    check_all()
