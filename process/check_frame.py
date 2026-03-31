import os

def check_frames(base_dir="frames"):
    for sub_dir in sorted(os.listdir(base_dir)):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue

        # 获取所有子目录
        child_dirs = [d for d in os.listdir(sub_dir_path) if os.path.isdir(os.path.join(sub_dir_path, d))]

        # 检查1: 是否有 1_ 开头和 3_ 开头的子目录
        has_1 = any(d.startswith("1_") for d in child_dirs)
        has_3 = any(d.startswith("3_") for d in child_dirs)

        if not has_1:
            print(f"[缺少 1_ 开头子目录] {sub_dir_path}")
        if not has_3:
            print(f"[缺少 3_ 开头子目录] {sub_dir_path}")

        # 检查2: 每个子目录下是否至少有一个 .jpg 文件
        for child_dir in sorted(child_dirs):
            child_dir_path = os.path.join(sub_dir_path, child_dir)
            jpg_files = [f for f in os.listdir(child_dir_path) if f.lower().endswith(".jpg")]
            if len(jpg_files) == 0:
                print(f"[无 .jpg 文件] {child_dir_path}")

if __name__ == "__main__":
    check_frames()
