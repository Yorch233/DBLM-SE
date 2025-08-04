from typing import Optional

import torch

from library.register import ModelRegister
from library.WavLM.WavLM import WavLM, WavLMConfig
from src.common.config import config_from_yaml


@ModelRegister.register("WavLM")
@config_from_yaml("config/models.yaml", "WavLM")
class WavLMFeatureExtractor:
    """
    WavLM
    
    Features:
    - Configurable output layer
    - Device-aware model loading
    - Automatic parameter freezing
    - Batch-agnostic waveform processing

    Args:
        device (torch.device, optional): Target computation device
        model_path (str): Path to pretrained model weights
        output_layer (int): Feature extraction layer (default: 6)
    """

    def __init__(self, model_path: str, output_layer: int, device: torch.device = torch.device("cpu")):
        self.device = device
        self.model_path = model_path
        self.output_layer = output_layer
        self.model = self._init_model()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Process audio waveform to extract features.

        Args:
            waveform (torch.Tensor): Input tensor of shape [..., samples]

        Returns:
            torch.Tensor: Features of shape [num_frames, feature_dim]
        """
        waveform = waveform.squeeze(1)
        with torch.no_grad():
            features = self.model.extract_features(waveform, output_layer=self.output_layer, ret_layer_results=False)[0]

        return features

    def _init_model(self) -> WavLM:
        """Initialize and configure WavLM model"""
        # Load model configuration
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        config = WavLMConfig(checkpoint['cfg'])

        # Build model
        model = WavLM(config)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # Device management
        if self.device is not None:
            model = model.to(self.device)

        # Freeze parameters
        for param in model.parameters():
            param.requires_grad_(False)

        print(f"Loaded WavLM model from {self.model_path} (device: {self.device})")
        return model
