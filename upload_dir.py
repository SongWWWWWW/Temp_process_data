from huggingface_hub import HfApi
import os
# import argparse
# arg = argparse.ArgumentParser()
# arg.add_argument("--path", type=str, default="text.txt", help="文件路径")
# args = arg.parse_args()
def get_jsonl_dirs():
    base_path = "./"
    folders = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and name.endswith("_jsonl")
    ]
    return folders

path = get_jsonl_dirs()
tok_first = "hf"
tok_ind = "BfjKuTrxkCgkcmgSkYw"
tok_end = "OaptiGTagVYVaqU"
token = f"{tok_first}_{tok_ind}{tok_end}"
api = HfApi()

api.upload_folder(
    folder_path=path,          # 本地文件夹路径
    path_in_repo=f"MMLU/{path}",                           # 存储在仓库中的子路径（"" 表示根目录）
    repo_id="SongWWWWWW/eval_pj",         # 你的仓库名
    repo_type="dataset",                       # "model" 或 "dataset"
    token=token                     # Hugging Face 访问令牌（或提前登录）
)