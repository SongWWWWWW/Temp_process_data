from huggingface_hub import HfApi, hf_hub_download
import os
import concurrent.futures
from tqdm import tqdm
import threading
import argparse

def download_file(api, dataset_id, file_path, cache_dir, progress):
    try:
        local_path = hf_hub_download(
            repo_id=dataset_id,
            filename=file_path,
            repo_type="dataset",
            cache_dir=cache_dir
        )
        progress.update(1)
    except Exception as e:
        print(f"download: {dataset_id}/{file_path} error: {e}")

def count_subdirs(path):
    return len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def process_dataset(api, dataset_id, dataset_name, cache_dir, master_progress):
    try:
        file_list = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")
        target_files = [f for f in file_list if dataset_name.lower() in f.lower()]

        if not target_files:
            print(f"cannot find {dataset_name}, skipping {dataset_id}")
            master_progress.update(1)
            return

        with tqdm(total=len(target_files), desc=f"📥 {dataset_id}", position=1, leave=False) as progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(download_file, api, dataset_id, f, cache_dir, progress)
                    for f in target_files
                ]
                concurrent.futures.wait(futures)

        master_progress.update(1)

    except Exception as e:
        print(f"handle: {dataset_id} error: {e}")
        master_progress.update(1)

def download_data(data_file_path, cache_dir, dataset_name):
    # api = HfApi()

    with open(data_file_path, "r", encoding="utf-8") as file:
        dataset_ids = [line.strip() for line in file if line.strip()]

    existing_dirs = count_subdirs(cache_dir)
    if existing_dirs >= len(dataset_ids):
        print(f"there are already {existing_dirs} directory")
        return

    print(f"start downloading {len(dataset_ids)} datasets...\n")

    with tqdm(total=len(dataset_ids), desc="downloading", position=0) as master_progress:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as dataset_executor:
            futures = [
                dataset_executor.submit(process_dataset, api, dataset_id, dataset_name, cache_dir, master_progress)
                for dataset_id in dataset_ids
            ]
            concurrent.futures.wait(futures)


if __name__ == "__main__":
    # 初始化 Hugging Face API 客户端
    tok_first = "hf"
    tok_ind = "BfjKuTrxkCgkcmgSkYw"
    tok_end = "OaptiGTagVYVaqU"
    api = HfApi(token=f"{tok_first}_{tok_ind}{tok_end}")

    parser = argparse.ArgumentParser(description="Distributed HF dataset downloader by machine index")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    # 读取 data.txt 文件中的数据集标识符
    data_file_path = "log.txt"  # 确保该文件路径正确
    with open(data_file_path, "r", encoding="utf-8") as file:
        dataset_ids = [line.strip() for line in file if line.strip()]

    name_file_path = "mmlu_name.txt"
    with open(name_file_path, "r", encoding="utf-8") as file:
        names = [line.strip() for line in file if line.strip()]
    
    for name in names[args.start:args.end]:
        print(f"finding: {name}")
        # 遍历每个数据集标识符并下载文件
        for dataset_id in dataset_ids:
            try:
                print(f"\nhandling: {dataset_id}")

                # 列出数据集中的所有文件
                # print("listing...")
                file_list = api.list_repo_files(repo_id=dataset_id, repo_type="dataset")

                # 如果需要筛选特定文件类型或名称，可以在这里添加过滤逻辑
                # 示例：筛选包含 "gsm8k" 的文件
                # print('file list')
                # print(file_list)
                target_files = [f for f in file_list if name.lower() in f.lower()]
                # print('target files')
                # print(target_files)

                if not target_files:
                    print(f"cannot find {name} in: {dataset_id}")
                    continue
                dir_name = f"./data_{name}"

                # 下载目标文件
                for file_path in target_files:
                    print(f"downloading: {file_path}")
                    local_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename=file_path,
                        repo_type="dataset",
                        cache_dir=dir_name  # 修改为你希望保存的目录
                    )
                    print(f"now in: {local_path}")

            except Exception as e:
                print(f"Error when downloading {dataset_id}: {e}")
                continue

    print("\nFinish")