import os
import random
import re
from os.path import basename, exists, join

import hydra
import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import utils
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoFeatureExtractor


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        ocwd = hydra.utils.get_original_cwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        ds = FSDataset(phase, self.cfg)
        # ds = FSDataset_add_STFT(phase, self.cfg)
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=28,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)

        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        pass


class FSDataset(Dataset):
    """Dataset batching wav, mel 
    and other acoustic features

    Args:
        phase: train, val, test
        cfg: hydra config
    """

    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        self.ocwd = hydra.utils.get_original_cwd()

        self.sr = cfg.preprocess.audio.sr

        # self.filelist = utils.read_filelist(join(self.ocwd, self.phase_cfg.filelist))
        self.filelist = self.get_filelist(self.phase_cfg.filelist)
        self.min_audio_length = cfg.dataset.min_audio_length
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0")

    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    def get_filelist(self, fpath):
        with open(fpath, 'r') as f:
            # flist = [l.strip() for l in f if l.strip()]
            flist = [l.strip().split('\t')[0] for l in f if l.strip()]
        return flist

    def __getitem__(self, idx):
        # (  wavpath,fid) = self.filelist[idx]
        wavpath = self.filelist[idx]
        wavpath_full = join(self.cfg.preprocess.datasets.LibriSpeech.root,
                            wavpath)
        # wav = self.load_wav(wavpath)
        # wav = torch.from_numpy(wav)

        wav, sr = torchaudio.load(wavpath_full)

        if sr != 16000:
            wav = Resample(sr, 16000)(wav)
        wav = wav[0, :]
        length = wav.shape[0]
        # length = wav.shape[1]
        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]
        i = random.randint(0, length - self.min_audio_length)
        wav = wav[i:i + self.min_audio_length]

        wav_pad = F.pad(wav, (160, 160))
        feat = self.feature_extractor(
            wav_pad, sampling_rate=16000,
            return_tensors="pt").data['input_features']
        out = {
            'wav': wav,
            'feat': feat,
            # 'paths': wavpath_full
        }

        return out

    def collate_fn(self, bs):

        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        feats = [b['feat'] for b in bs]
        feats = torch.stack(feats)
        out = {
            'wav': wavs,
            'feats': feats,
            # 'paths': [b['paths'] for b in bs]
        }
        return out


@hydra.main(config_path='config', config_name='default', version_base=None)
def main(cfg):

    # 初始化DataModule
    data_module = DataModule(cfg)

    # 获取训练数据加载器
    train_loader = data_module.val_dataloader()

    # 用于保存无错误的音频路径
    valid_filelist = []

    # 遍历数据集
    for batch_idx, batch in enumerate(
            tqdm(train_loader, desc="Processing batches", unit="batch")):
        # try:
        wavs = batch['wav']
        # paths = batch['paths']
        # print(f"Loaded batch {batch_idx + 1} with shape: {wavs.shape}")

        # 将没有问题的路径保存到 valid_filelist
        #     valid_filelist.extend(paths)
        # except Exception as e:
        #     # 如果遇到问题，打印错误信息和文件路径
        #     print(f"Error in batch {batch_idx + 1}: {e}")
        #     print(f"Paths in this batch: {batch['paths']}")

        # 保存没有问题的音频文件列表到新的文件中
        # with open('/aifs4su/data/zheny/data/data_8_21_2/mls_all_audio_path2.txt', 'w') as f:
        #     for item in valid_filelist:
        #         f.write(f"{item}\n")
        c = 1
    print(f"Successfully saved valid filelist to 'valid_filelist.txt'")


if __name__ == "__main__":
    main()
