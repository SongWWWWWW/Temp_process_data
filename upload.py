from huggingface_hub import HfApi, login
import argparse
import os
# arg = argparse.ArgumentParser()
# arg.add_argument("--path", type=str, default="text.txt", help="文件路径")
# args = arg.parse_args()
def get_data_jsonl_files():
    base_path = "./"
    files = [
        name for name in os.listdir(base_path)
        if os.path.isfile(os.path.join(base_path, name))
        and name.startswith("data") and name.endswith(".jsonl")
    ]
    return files
path = get_data_jsonl_files()
tok_first = "hf"
tok_ind = "BfjKuTrxkCgkcmgSkYw"
tok_end = "OaptiGTagVYVaqU"
# 登录
login(token=f"{tok_first}_{tok_ind}{tok_end}")

# 创建一个新 repo
api = HfApi()
# api.create_repo(repo_id="SongWWWWWW/eval_pj", repo_type="dataset")

# 上传本地文件
api.upload_file(
    path_or_fileobj=path,  # 本地文件路径
    path_in_repo=f"MMLU/{path}",  # 存储在仓库中的子路径（"" 表示根目录）
    repo_id="SongWWWWWW/eval_pj",
    repo_type="dataset"
)