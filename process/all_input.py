import os
import json

def merge_input_jsons(base_dir="general_pair_new"):
    result = {}
    
    # 遍历 base_dir 下的所有直接子目录
    for sub_dir in sorted(os.listdir(base_dir)):
        sub_dir_path = os.path.join(base_dir, sub_dir)
        
        # 确保是目录
        if not os.path.isdir(sub_dir_path):
            continue
        
        # 构造 input.json 的路径
        input_json_path = os.path.join(sub_dir_path, "input", "input.json")
        
        # 检查文件是否存在
        if not os.path.isfile(input_json_path):
            print(f"[跳过] 文件不存在: {input_json_path}")
            continue
        
        # 读取 JSON 内容
        try:
            with open(input_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 以子目录名作为 key
            result[sub_dir] = data
            print(f"[成功] 已读取: {input_json_path}")
        except json.JSONDecodeError as e:
            print(f"[错误] JSON解析失败: {input_json_path}, 错误: {e}")
        except Exception as e:
            print(f"[错误] 读取失败: {input_json_path}, 错误: {e}")
    
    # 写入合并后的 input.json
    output_path = "input.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"合并完成! 共处理 {len(result)} 个子目录, 输出文件: {output_path}")

if __name__ == "__main__":
    merge_input_jsons()
