import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库

# 定义基础目录

# 查找所有包含"gsm8k"的.parquet文件
def find_parquet_files(directory,dataset_name):
    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.parquet') and dataset_name.lower() in file.lower():
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# 解析模型名称和日期
def parse_model_and_date(file_path):
    parts = Path(file_path).parts
    model_name = None
    date_str = None

    # 遍历路径部分，找到模型名称和日期
    for part in reversed(parts):  # 从后往前检查以找到最近的日期和模型名
        # 修改模型名称提取逻辑
        if 'details_' in part and '__' in part:
            try:
                # 处理包含双下划线的目录名 (如 datasets--...__Yi-34B)
                model_part = part.split('__')[-1]
                if model_part.startswith("details_"):
                    model_name = model_part.split('_', 1)[1]  # 处理 details_01-ai__Yi-34B 格式
                else:
                    model_name = model_part
                break  # 找到模型名后立即停止
            except IndexError:
                print(f"name parse error: {parts}")
                continue

    # 单独遍历查找日期（避免与模型名查找冲突）
    for part in reversed(parts):
        # 增强日期格式匹配（支持 2023-12-05T03-47-25.491369 格式）
        if part.startswith("20") and "T" in part and "-" in part:
            date_str = part
            break

    return model_name, date_str

def has_multiple_jsonl_files(directory, min_count=100):
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                count += 1
                if count >= min_count:
                    return True
    return False
# 主函数
def data_2_jsonl(data_dir,output_path,dataset_name):
    # 查找所有符合条件的 .parquet 文件
    print(f"finding {dataset_name} parquet...")
    parquet_files = find_parquet_files(data_dir,dataset_name)

    # 如果没有找到任何文件，直接退出
    if not parquet_files:
        print("no parquet")
        return

    if has_multiple_jsonl_files(output_path):
        print(f"{output_path} has multi jsonl file , skip")
        return

    model_to_latest_file = {}

    # 使用 tqdm 显示进度条，筛选每个模型最新的文件
    print("looking for the newest file...")
    for file in tqdm(parquet_files, desc="looking for ", unit="file"):
        model_name, date_str = parse_model_and_date(file)
        if model_name is None or date_str is None:
            print(f"invalid file: {file}")
            continue
        if model_name not in model_to_latest_file or date_str > model_to_latest_file[model_name][1]:
            model_to_latest_file[model_name] = (file, date_str)

    # 对于每个模型，处理其对应的最新文件
    print("preprocessing data...")
    for model_name, (file, _) in tqdm(model_to_latest_file.items(), desc="preprocessing data", unit="model"):
        print(f"preprocecssing: {file}")
        df = pd.read_parquet(file)
        output_file = f"{output_path}/{model_name}.jsonl"

        # 检查是否需要从嵌套的 metrics 中提取 acc
        if 'metrics' in df.columns:
            df['acc'] = df['metrics'].apply(lambda x: x.get('acc', None))  # 提取 metrics 中的 acc

        # 添加列名兼容性处理
        column_mapping = {
            'acc': ['acc', 'accuracy'],  # 可能的列名变体
            'example': ['example', 'problem']
        }
        selected_columns = {}
        for target_col, variants in column_mapping.items():
            for var in variants:
                if var in df.columns:
                    selected_columns[target_col] = var
                    break
            else:
                raise KeyError(f"cannot find a replacing name for {target_col} in file: {file}")

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for index, row in df[list(selected_columns.values())].iterrows():
                record = {
                    "acc": row[selected_columns['acc']],
                    "example": row[selected_columns['example']]
                }
                outfile.write(f"{record}\n")
        print(f"files are saved in : {output_file}")
def get_base_dir():
    base_path = "./"
    folders = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and name.startswith("data")
    ]
    return folders


if __name__ == "__main__":
    base_dir_list = []
    output_path_list = [f"{i}_jsonl" for i in base_dir_list]
    dataset_name = "abstract_algebra"
    name_file_path = "mmlu_name.txt"
    for base_dir, output_path in zip(base_dir_list,output_path_list):

        with open(name_file_path, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]
        os.makedirs(output_path, exist_ok=True)

        data_2_jsonl(data_dir=base_dir,output_path=output_path, dataset_name=dataset_name)