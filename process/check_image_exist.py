import os
import json

ROOT_DIR = "general_pair_new"
FRAMES_DIR = "frames"

def main():
    missing_files = []
    total_images = 0
    total_samples = 0

    for sample_name in sorted(os.listdir(ROOT_DIR)):
        sample_dir = os.path.join(ROOT_DIR, sample_name)
        if not os.path.isdir(sample_dir):
            continue

        input_path = os.path.join(sample_dir, "input", "input.json")
        if not os.path.exists(input_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        total_samples += 1

        for item in input_data.get("name_image_list", []):
            if item.endswith(".jpg") or item.endswith(".png"):
                total_images += 1
                if not os.path.exists(item):
                    missing_files.append((sample_name, item))

    # 报告
    print(f"Total samples checked: {total_samples}")
    print(f"Total image paths checked: {total_images}")

    if missing_files:
        print(f"[ERROR] {len(missing_files)} missing files:")
        for sample, path in missing_files:
            print(f"  [{sample}] {path}")
    else:
        print("All image paths exist. ✅")

if __name__ == "__main__":
    main()
