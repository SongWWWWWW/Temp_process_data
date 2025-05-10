import os
import json
from collections import defaultdict
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_file(filename, input_dir):
    if not filename.endswith(".jsonl"):
        print(f"{filename} is not a .jsonl file, skipping.")
        return None

    model_name = filename.replace(".jsonl", "")
    filepath = os.path.join(input_dir, filename)
    acc_details = []

    try:
        with open(filepath, "r") as f:
            for idx, line in enumerate(f):
                try:
                    entry = ast.literal_eval(line)
                    acc_details.append({
                        f"q_{idx}": 1 if entry["acc"]==True or entry["acc"]=='True' else 0
                    })
                except Exception as e:
                    print(f"parsing {filename} error: {str(e)}")
                    continue
    except Exception as e:
        print(f"opening {filename} error: {str(e)}")
        return None

    if not acc_details:
        return None

    total = len(acc_details)
    acc_count = sum(1 for d in acc_details if list(d.values())[0] == 1)
    acc_rate = round(acc_count / total, 4)

    return {
        "model_name": model_name,
        "acc": acc_rate,
        "detail": acc_details
    }

def process_files(input_dir="data_collect_all", output_file="all_models_results_detailed.jsonl", max_workers=25):
    if os.path.exists(output_file):
        print(f"{output_file} exists")
        return
    results = []
    file_list = [f for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    # 使用线程池处理文件
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_file, filename, input_dir): filename
                  for filename in file_list}

        # 使用tqdm显示进度
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                finally:
                    pbar.update(1)

    # 按模型名称排序结果
    results.sort(key=lambda x: x["acc"], reverse=True)

    # 写入汇总文件
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    # 统计分布
    stats = defaultdict(int)
    for item in results:
        rate = item["acc"]
        key = f"{int(rate*100)//10*10}%-{(int(rate*100)//10+1)*10}%"
        stats[key] += 1

    print("\nAcc Distribution:")
    for k, v in sorted(stats.items()):
        print(f"{k}: {v} models")
def get_jsonl_dirs():
    base_path = "./"
    folders = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and name.endswith("_jsonl")
    ]
    return folders
if __name__ == "__main__":
    input_dir_list = get_jsonl_dirs()
    output_file_list = [f"{i[:-6]}.jsonl" for i in input_dir_list]
    for input_dir , output_file in zip(input_dir_list,output_file_list):
        process_files(input_dir=input_dir, output_file=output_file)