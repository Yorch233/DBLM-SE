import os
from functools import partial
from multiprocessing import Pool, cpu_count
from os.path import isdir, join
from pathlib import Path

from tqdm import tqdm


def find_files_in_subdir(subdir, extension):
    """
    在给定的子目录中查找具有特定扩展名的文件。
    
    Args:
        subdir (str): 子目录路径。
        extension (str): 文件扩展名（例如 ".txt"）。
    
    Returns:
        list: 符合条件的文件列表，每个元素是 [文件名（无扩展名），文件路径]。
    """
    out = []
    try:
        for root, dirs, filenames in os.walk(subdir):
            for f in filenames:
                if f.endswith(extension):
                    out.append([str(Path(f).stem), os.path.join(root, f)])
    except Exception as e:
        print(f"Error processing {subdir}: {e}")
    return out


def get_second_level_subdirs(path_dir):
    """
    获取指定目录下的所有二级子目录。
    如果一级子目录没有二级子目录，则将一级子目录本身作为处理单元。
    
    Args:
        path_dir (str): 目标目录路径。
    
    Returns:
        list: 所有二级子目录路径列表。
    """
    second_level_subdirs = []
    try:
        first_level = [
            join(path_dir, d) for d in os.listdir(path_dir)
            if isdir(join(path_dir, d))
        ]
        for first_subdir in first_level:
            second_subdirs = [
                join(first_subdir, sd) for sd in os.listdir(first_subdir)
                if isdir(join(first_subdir, sd))
            ]
            if second_subdirs:
                second_level_subdirs.extend(second_subdirs)
            else:
                # 如果一级子目录没有二级子目录，则将一级子目录本身作为处理单元
                second_level_subdirs.append(first_subdir)
    except Exception as e:
        print(f"Error accessing directories in {path_dir}: {e}")
    return second_level_subdirs


def find_all_files(path_dir, extension):
    """
    使用多进程查找指定目录及其二级子目录中所有具有特定扩展名的文件。
    
    Args:
        path_dir (str): 目标目录路径。
        extension (str): 文件扩展名（例如 ".txt"）。
    
    Returns:
        list: 所有符合条件的文件列表。
    """
    out = []
    subdirs = get_second_level_subdirs(path_dir)

    if not subdirs:
        subdirs = [path_dir]

    func = partial(find_files_in_subdir, extension=extension)

    with Pool(processes=100) as pool:
        for result in tqdm(pool.imap(func, subdirs),
                           total=len(subdirs),
                           desc="Processing subdirectories"):
            out.extend(result)

    return out


def read_filelist(path, delimiter='|'):
    """
    读取文件列表，每行使用指定的分隔符分割。
    
    Args:
        path (str): 文件路径。
        delimiter (str): 分隔符，默认为 '|'.
    
    Returns:
        list: 分割后的文件列表，每个元素是一个列表。
    """
    with open(path, encoding='utf8') as f:
        lines = [line.strip().split(delimiter) for line in f if line.strip()]
    return lines


def write_filelist(filelists, path, delimiter='|'):
    """
    将文件列表写入指定路径，每行使用指定的分隔符连接。
    
    Args:
        filelists (list): 文件列表，每个元素是一个列表。
        path (str): 输出文件路径。
        delimiter (str): 分隔符，默认为 '|'.
    """
    with open(path, 'w', encoding='utf8') as f:
        for line in filelists:
            f.write(delimiter.join(line) + '\n')


# 示例用法
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多进程查找文件并显示进度条（按二级目录）")
    parser.add_argument("directory", help="目标目录路径")
    parser.add_argument("extension", help="目标文件扩展名，例如 .txt")
    parser.add_argument("output", help="输出文件列表的路径")
    args = parser.parse_args()

    directory = args.directory
    extension = args.extension
    output_path = args.output

    all_files = find_all_files(directory, extension)
    write_filelist(all_files, output_path)

    print(f"查找完成，共找到 {len(all_files)} 个文件。结果已写入 {output_path}。")
