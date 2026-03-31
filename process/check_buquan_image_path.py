import os
import json
import re

def fix_input_jsons(base_dir="general_pair_new"):
    fix_count = 0

    for sub_dir in sorted(os.listdir(base_dir)):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        input_json_path = os.path.join(sub_dir_path, "input", "input.json")
        if not os.path.isfile(input_json_path):
            continue

        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name_image_list = data.get("name_image_list", [])
        if not name_image_list:
            continue

        modified = False
        new_list = []

        # 按 .mp4 分组，每个 mp4 后面跟着它的帧路径
        groups = []  # [(mp4_name, [frame_paths]), ...]
        current_mp4 = None
        current_frames = []

        for item in name_image_list:
            if ".mp4" in item:
                if current_mp4 is not None:
                    groups.append((current_mp4, current_frames))
                current_mp4 = item
                current_frames = []
            else:
                current_frames.append(item)
        if current_mp4 is not None:
            groups.append((current_mp4, current_frames))

        for mp4_name, frame_paths in groups:
            new_list.append(mp4_name)

            if not frame_paths:
                continue

            # ========================
            # 修复1: 去除路径中多余的 000xxx.jpg/ 段
            # 例如 frames/000001.jpg/6897940245858258177/1_1/000001.jpg
            #   -> frames/6897940245858258177/1_1/000001.jpg
            # ========================
            cleaned_paths = []
            for p in frame_paths:
                # 检测模式: frames/000xxx.jpg/... 中间多了一段
                fixed = re.sub(r'^(frames)/\d{6}\.jpg/', r'\1/', p)
                if fixed != p:
                    if not modified:
                        print(f"🔧 {sub_dir}")
                    print(f"   [路径修复] {p} -> {fixed}")
                    modified = True
                cleaned_paths.append(fixed)

            # ========================
            # 修复2: 补全中间缺失的帧
            # 从已有帧中推断目录和帧号范围，补全所有中间帧
            # ========================
            # 提取目录和帧号
            # 格式: frames/{id}/{x_y}/000014.jpg
            dir_frame_map = {}  # dir_path -> [frame_numbers]
            for p in cleaned_paths:
                match = re.match(r'^(frames/.+/)(\d{6})\.jpg$', p)
                if match:
                    dir_path = match.group(1)  # e.g. frames/xxx/3_1/
                    frame_num = int(match.group(2))
                    if dir_path not in dir_frame_map:
                        dir_frame_map[dir_path] = []
                    dir_frame_map[dir_path].append(frame_num)

            # 对每个目录，从最小帧号到最大帧号补全
            expanded_paths = []
            for dir_path in dir_frame_map:
                nums = sorted(dir_frame_map[dir_path])
                min_num = nums[0]
                max_num = nums[-1]
                full_nums = list(range(min_num, max_num + 1))

                if len(full_nums) != len(nums):
                    if not modified:
                        print(f"🔧 {sub_dir}")
                    missing = set(full_nums) - set(nums)
                    print(f"   [帧补全] {dir_path}: 原有 {len(nums)} 帧, 补全至 {len(full_nums)} 帧, 补充帧号: {sorted(missing)}")
                    modified = True

                for n in full_nums:
                    expanded_paths.append(f"{dir_path}{n:06d}.jpg")

            new_list.extend(expanded_paths)

        # 检查是否有修改
        if modified:
            data["name_image_list"] = new_list
            with open(input_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            fix_count += 1
            print(f"   ✅ 已保存修复后的 input.json")

    # 汇总
    total = len([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    print(f"{'='*60}")
    print(f"总计: {total} 个子目录, 其中 {fix_count} 个被修复")


if __name__ == "__main__":
    fix_input_jsons()
