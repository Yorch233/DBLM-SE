import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torchaudio
from safetensors.torch import load_model
from tabulate import tabulate
from torch.utils.data import Dataset
from tqdm import tqdm

from library.register import ModelRegister
from src.common.config import read_config_from_yaml
from src.modeling_dblm import DBLMForSE
from src.sde import DDBM_VPSDE


def parse_args():
    """
    Parse command-line arguments for audio enhancement configuration.
    """
    parser = argparse.ArgumentParser(description="Enhance audio files using diffusion-based language model.")

    parser.add_argument('--audio_path', type=str,
                        help='Directory path to input noisy audio files (WAV format)')
    parser.add_argument('--output_path', type=str, default='output',
                        help='Directory path to save enhanced audio files')
    parser.add_argument('--run_path', type=str, default='checkpoints',
                        help='Path to directory containing config.yml and model checkpoint')
    parser.add_argument('--num_steps', type=int, default=3,
                        help='Number of sampling steps for the SDE solver (higher = better quality but slower)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='GPU device index to use for inference (default: 0)')

    return parser.parse_args()


def setup_models(config, device):
    """
    Setup and return the required models and SDE process.
    
    Args:
        config: Configuration object containing model parameters
        device: torch device to load models onto
    
    Returns:
        tuple: (audio_encoder, audio_codec, alm, sde)
    """
    # Load audio encoder and codec from registry
    audio_encoder = ModelRegister.fetch(config.audio_encoder)(device=device)
    audio_codec = ModelRegister.fetch(config.audio_codec)(device=device)

    # Initialize the diffusion language model
    alm = DBLMForSE(
        in_dim=config.latent_dim,
        d_model=config.model_dim,
        nhead=config.num_head,
        num_layers=config.num_layers
    )
    alm = torch.compile(alm)
    load_model(alm, os.path.join(config.run_path, 'model.safetensors'))
    alm.to(device)
    alm.eval()

    total_params = sum(p.numel() for p in alm.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params / 1e9:.2f} B")

    # Setup SDE process
    sde = DDBM_VPSDE(
        beta=config.beta,
        t_max=config.t_max,
        t_min=0,
        device=device,
    )

    return audio_encoder, audio_codec, alm, sde


class AudioDataset(Dataset):
    """
    Dataset class to handle loading and preprocessing of audio files.
    """
    def __init__(self, audio_path, sample_rate=16000, return_path=False, reverse=False):
        self.reverse = reverse
        self.return_path = return_path

        if os.path.isdir(audio_path):
            self.audio_paths = sorted(glob(f"{audio_path}/*.wav"))
            self.sample_rate = sample_rate
        else:
            raise NotImplementedError("Only directory is supported.")

        self.segment_length = 6 * 16000  # 6 seconds at 16kHz

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        if self.reverse:
            index = len(self) - index - 1

        if hasattr(self, 'audio_paths'):
            audio_path = self.audio_paths[index]
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            return (audio, os.path.basename(audio_path)[:-4]) if self.return_path else audio
        else:
            audio = self.audio_files["audio"][index]
            return (audio, self.audio_files["path"][index]) if self.return_path else audio

    def get_data_loader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, reverse=None):
        if reverse is not None:
            self.reverse = reverse
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )


@torch.no_grad()
def predict_x0(xt, timestep, alm, sde):
    """
    Predict x0 given xt and timestep using the diffusion model.
    
    Args:
        xt: Current state tensor
        timestep: Current timestep
        alm: Diffusion language model
        sde: SDE process object
    
    Returns:
        Predicted initial state x0
    """
    timestep = torch.full((xt.shape[0],), timestep, device=xt.device, dtype=torch.float32)
    output = alm(xt, timestep, return_logits=False)
    return output


def inference():
    """
    Main function to perform audio enhancement on a dataset of noisy audios.
    """
    args = parse_args()

    # Set device based on local_rank
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config = read_config_from_yaml(os.path.join(args.run_path, 'config.yml'))
    config.run_path = args.run_path

    # Setup models
    audio_encoder, audio_codec, alm, sde = setup_models(config, device)

    # Prepare output directory
    output_path = os.path.join(args.output_path, f"steps={args.num_steps}_gpu{args.local_rank}")
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    noisy_audio_dataset = AudioDataset(args.audio_path, sample_rate=16000, return_path=True)
    dataloader = noisy_audio_dataset.get_data_loader()

    # Configure SDE solver
    solver = sde.get_sde_solver(model_fn=lambda xt, timestep: predict_x0(xt, timestep, alm, sde))

    with torch.no_grad():
        for audio, audio_name in tqdm(dataloader, desc=f"Enhancing Audios (GPU {args.local_rank})"):
            noisy_audio, audio_name = audio.to(device), audio_name[0]
            output_audio_path = os.path.join(output_path, f"{audio_name}_enhanced.wav")

            # Normalize audio to [-1, 1]
            max_val = noisy_audio.abs().max()
            if max_val > 0:
                scale_factor = 1.0 / max_val
                noisy_audio = noisy_audio * scale_factor
            else:
                scale_factor = 1.0

            # Encode noisy audio to latent space
            y = audio_encoder(noisy_audio)
            x = y

            # Run reverse SDE sampling
            x_0 = solver.sampling(x=x, num_steps=args.num_steps)

            # Decode back to waveform
            logits = alm.get_logits(x_0)
            _, max_indices = torch.max(logits, dim=1)
            enhanced_audio = audio_codec.token2audio(max_indices.unsqueeze(0))

            # Denormalize
            enhanced_audio = enhanced_audio / scale_factor

            # Save result
            torchaudio.save(
                output_audio_path,
                enhanced_audio.type(torch.float32).cpu().squeeze().unsqueeze(0),
                16000
            )


if __name__ == "__main__":
    inference()