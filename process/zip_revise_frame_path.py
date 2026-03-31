import os
import json
import zipfile

ROOT_DIR = "general_pair_new"
ZIP_PATH = "frames.zip"

def main():
    missing_frames = []
    processed = 0

    # ---- Step 1: 创建压缩包并修改路径 ----
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_STORED) as zf:
        for sample_name in sorted(os.listdir(ROOT_DIR)):
            sample_dir = os.path.join(ROOT_DIR, sample_name)
            if not os.path.isdir(sample_dir):
                continue

            frames_dir = os.path.join(sample_dir, "frames")
            input_path = os.path.join(sample_dir, "input", "input.json")

            # 检查 frames 目录是否存在
            if not os.path.exists(frames_dir) or not os.path.isdir(frames_dir):
                missing_frames.append(sample_name)
                print(f"[WARN] {sample_name}: frames directory missing!")
                continue

            # 将 frames 下所有图片加入压缩包
            for video_id in sorted(os.listdir(frames_dir)):
                video_dir = os.path.join(frames_dir, video_id)
                if not os.path.isdir(video_dir):
                    continue
                for img_name in sorted(os.listdir(video_dir)):
                    img_path = os.path.join(video_dir, img_name)
                    if not os.path.isfile(img_path):
                        continue
                    # 压缩包内路径: sample_name/video_id/img_name
                    arcname = os.path.join(sample_name, video_id, img_name)
                    zf.write(img_path, arcname)

            # 修改 input/input.json 中的图片路径
            if not os.path.exists(input_path):
                print(f"[WARN] {sample_name}: input/input.json missing!")
                continue

            with open(input_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)

            new_name_image_list = []
            for item in input_data["name_image_list"]:
                if item.endswith(".jpg") or item.endswith(".png"):
                    # e.g., "general_pair_new/_ODoC7Xw5QY/frames/3_3/000001.jpg"
                    #     -> "frames/_ODoC7Xw5QY/3_3/000001.jpg"
                    if item.startswith("frames/"):
                        new_name_image_list.append(item)
                    else:
                        parts = item.split("/")
                        # 找到 "frames" 的位置
                        if "frames" in parts:
                            frames_idx = parts.index("frames")
                            # sample_name 在 frames 前一位
                            sample = parts[frames_idx - 1]
                            # frames 之后的部分: video_id/img_name
                            rest = "/".join(parts[frames_idx + 1:])
                            new_path = f"frames/{sample}/{rest}"
                            new_name_image_list.append(new_path)
                        else:
                            new_name_image_list.append(item)
                else:
                    new_name_image_list.append(item)

            input_data["name_image_list"] = new_name_image_list

            with open(input_path, "w", encoding="utf-8") as f:
                json.dump(input_data, f, ensure_ascii=False, indent=2)

            processed += 1

    # ---- Step 2: 报告 ----
    print(f"Done. Processed: {processed}")
    print(f"ZIP saved to: {ZIP_PATH}")

    if missing_frames:
        print(f"[ALERT] {len(missing_frames)} samples missing frames directory:")
        for name in missing_frames:
            print(f"  - {name}")
    else:
        print("All samples have frames directory.")

if __name__ == "__main__":
    main()
