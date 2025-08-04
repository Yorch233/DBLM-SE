import os
from os.path import basename, exists, expanduser, join

import hydra
import librosa
import utils
from tqdm import tqdm
from utils import find_all_files, read_filelist, write_filelist


@hydra.main(version_base=None, config_path='config', config_name='default')
def preprocess(cfg):
    os.makedirs('filelists', exist_ok=True)
    # train
    root = cfg.preprocess.datasets.LibriSpeech.root_val
    root = expanduser(root)
    trainfiles = []
    print(f'Root: {root}')
    for subset in cfg.preprocess.datasets.LibriSpeech.testsets:
        files = find_all_files(join(root, subset), '.flac')
        print(f'Found {len(files)} flac files in {subset}')
        for i in range(len(files)):
            files[i][1] = files[i][1].replace(root, '').lstrip('/')
        trainfiles.extend(files)

    print(f'Write train filelist to {cfg.preprocess.view.test_filelist}')
    os.makedirs('filelists', exist_ok=True)
    utils.write_filelist(trainfiles, cfg.preprocess.view.test_filelist)


if __name__ == '__main__':
    preprocess()
