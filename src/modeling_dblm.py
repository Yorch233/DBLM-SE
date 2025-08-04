"""
Author: Yorch233
GitHub: https://github.com/Yorch233/DBLM-SE
Email: qyao@stmail.ujs.edu.cn
Date: 2025-08-05
"""

import warnings
from typing import Any, Callable, Dict, List, Union


class Register(dict):
    """
    A custom dictionary subclass designed to register and manage artifacts.
    Artifacts can be registered by type and name, allowing for easy retrieval.
    """

    def __init__(self, types_list: List[str] = None) -> None:
        """
        Initializes the ArtifactRegister object. Optionally initializes with a list of types.

        Parameters:
            types_list (List[str], optional): A list of types to initialize the register with. Defaults to None.
        """
        super().__init__()
        self._dict: Dict[str, Dict[str, Any]] = {}

        if types_list is not None:
            for artifact_type in types_list:
                self._dict[artifact_type] = {}

    def register(self, artifact_name: str) -> Callable[[Any], Any]:
        """
        Decorator method to register an artifact by its type and name.

        Parameters:
            artifact_name (str): The name of the artifact.

        Returns:
            Callable[[Any], Any]: A decorator function.
        """

        def decorator(artifact: Any) -> Any:
            self._dict[artifact_name] = artifact
            return artifact

        return decorator

    def fetch(self, name_or_name_list: Union[str, List[str]]) -> Any:
        """
        Retrieves a registered artifact by its type and name.

        Parameters:
            artifact_type (str): The type of the artifact.
            artifact_name (str): The name of the artifact.

        Returns:
            Any: The registered artifact.

        Raises:
            KeyError: If the artifact type or name is not registered.
        """
        if isinstance(name_or_name_list, list):
            artifacts = {}
            for name in name_or_name_list:
                if name in self._dict:
                    artifacts[name] = self._dict[name]
                else:
                    warnings.warn(
                        f"Unregistered artifact_name: '{name}'. Ignoring this artifact."
                    )
            if len(artifacts) == 0:
                raise ValueError("No registered artifacts found.")
            return artifacts
        elif isinstance(name_or_name_list, str):
            if name_or_name_list in self._dict:
                return self._dict[name_or_name_list]
            else:
                raise KeyError(
                    f"Unregistered artifact_name: '{name_or_name_list}'.")


import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional, Union

from transformers import LlamaConfig, LlamaForCausalLM

NUM_AUDIO_TOKENS = 65536  # Codebook size for discrete audio tokens


class SinusPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time or sequence positions.
    Generates fixed sinusoidal embeddings based on position indices.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """
    Embeds continuous time steps into a high-dimensional vector space using sinusoidal embeddings,
    followed by a small MLP to project into the model's dimension.
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


class ConvPositionEmbedding(nn.Module):
    """
    Applies convolutional position embedding to enhance local context awareness.
    Uses two 1D conv layers with Mish activation.
    """

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 != 0, "Kernel size must be odd."
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class InputEmbedding(nn.Module):
    """
    Combines input features with time embeddings and applies convolutional position encoding.
    Projects input to model dimension and enhances with temporal context.
    """

    def __init__(self, in_dim: int, dim: int, out_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        self.out_proj = nn.Linear(dim * 2, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Broadcast time embedding across sequence length
        time_emb = time_emb.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = self.in_proj(x)
        x = self.out_proj(torch.cat((x, time_emb), dim=-1))
        x = self.conv_pos_embed(x) + x  # Residual connection
        return x


class DBLMForSE(nn.Module):
    """
    Diffusion Bridge Language Model for Speech Enhancement (DBLM-SE).
    
    This model performs joint speech denoising and dereverberation by:
    - Using diffusion in latent space
    - Predicting discrete acoustic tokens via a language modeling head
    - Leveraging LLaMA-based transformer architecture
    """

    def __init__(
        self,
        in_dim: int = 1024,
        d_model: int = 2048,
        nhead: int = 16,
        num_layers: int = 16
    ):
        super().__init__()
        self.d_model = d_model

        # Time embedding module
        self.time_embed = TimestepEmbedding(d_model)

        # Input embedding with time conditioning
        self.input_embed = InputEmbedding(in_dim, d_model, d_model)

        # Main LLaMA-based transformer backbone
        self.Llama_config = LlamaConfig(
            hidden_size=d_model,
            intermediate_size=d_model * 2,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            dropout_rate=0.1,
            attention_dropout=0.1,
            is_decoder=True,
            use_cache=True
        )
        self.llama = LlamaForCausalLM(config=self.Llama_config)

        # Output projection from transformer hidden state to input dimension
        self.out_proj = nn.Linear(d_model, in_dim)

        # Prediction layer for discrete audio token logits
        self.predict_layer_config = LlamaConfig(
            hidden_size=in_dim,
            intermediate_size=2048,
            num_attention_heads=8,
            num_hidden_layers=8,
            dropout_rate=0.1,
            attention_dropout=0.1,
            is_decoder=True,
            use_cache=True,
            vocab_size=NUM_AUDIO_TOKENS
        )
        self.predict_layer = LlamaForCausalLM(config=self.predict_layer_config)

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts logits over discrete audio tokens from the latent representation.

        Args:
            x (torch.Tensor): Latent representations of shape [B, T, D].

        Returns:
            torch.Tensor: Logits of shape [B, NUM_AUDIO_TOKENS, T].
        """
        logits = self.predict_layer(inputs_embeds=x).logits  # [B, T, NUM_AUDIO_TOKENS]
        logits = logits.transpose(-1, -2)  # [B, NUM_AUDIO_TOKENS, T]
        return logits

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        return_logits: bool = True
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Noisy latent input of shape [B, T, D].
            time (torch.Tensor): Diffusion time steps of shape [B].
            return_logits (bool): Whether to compute and return logits.

        Returns:
            torch.Tensor or tuple:
                - If return_logits is True: (denoised_latents, logits)
                - Else: denoised_latents
        """
        batch, seq_len = x.shape[0], x.shape[1]

        # Embed time step
        t = self.time_embed(time)

        # Embed input with time condition
        x = self.input_embed(x, t)

        # Pass through main transformer
        outputs = self.llama(inputs_embeds=x, output_hidden_states=True)
        x = outputs.hidden_states[-1]

        # Project to original input dimension
        x = self.out_proj(x)

        if return_logits:
            logits = self.get_logits(x)
            return x, logits
        else:
            return x