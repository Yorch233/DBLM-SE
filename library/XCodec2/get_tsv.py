import os
from multiprocessing import Pool, cpu_count

import torch
import torchaudio
from tqdm import tqdm


def process_file(args):
    file_path, root_dir = args  # 解包传入的参数
    try:
        rel_path = os.path.relpath(file_path, start=root_dir)
        waveform, sample_rate = torchaudio.load(file_path)
        nsample = waveform.shape[1]
        batch_size = 10000  # 选择适合的批处理大小
        for start in range(0, waveform.numel(), batch_size):
            end = min(start + batch_size, waveform.numel())
            if torch.isnan(waveform.view(-1)[start:end]).any():
                print(rel_path)
                return None  # 如果包含nan，则不处理这个文件

        if nsample == 0:
            return None
        return f"{rel_path}\t{nsample}\n"
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def list_audio_files(root_dir, output_file, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        # 排除指定的子文件夹
        dirs[:] = [
            d for d in dirs if os.path.join(root, d) not in exclude_dirs
        ]

        for filename in files:
            if filename.endswith(('.wav', '.flac', '.mp3')):
                file_path = os.path.join(root, filename)
                audio_files.append((file_path, root_dir))  # 将root_dir与文件路径一起打包

    # 按文件名排序
    audio_files.sort(key=lambda x: x[0])

    # 使用多进程处理文件
    pool = Pool(processes=int(cpu_count() / 2))  # 使用一半的CPU核心
    results = list(
        tqdm(pool.imap(process_file, audio_files),
             total=len(audio_files),
             desc="Processing audio files"))

    # 写入结果到文件
    with open(output_file, 'w') as file:
        file.write(f"{root_dir}\n")
        for result in results:
            if result:  # 只有当result不为None时才写入文件
                file.write(result)


# 示例使用
root_directory = '/aifs4su/data/zheny/data/data_8_21_2'
output_tsv = '/aifs4su/data/zheny/data/data_8_21_2/mls_all_audio_path_higher_quality.txt'
exclude_folders = ['/aifs4su/data/zheny/data/data_8_21_2/test-clean']

list_audio_files(root_directory, output_tsv, exclude_dirs=exclude_folders)
