from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

from library.register import ModelRegister
from library.XCodec2.vq import (CodecDecoderVocos, CodecEncoder_Transformer,
                                SemanticEncoder)
from src.common.config import config_from_yaml


@ModelRegister.register("XCodec2")
@config_from_yaml("config/models.yaml", "XCodec2")
class XCodec2(nn.Module):
    """
    XCodec2
    
    Features:
    - Unified device management
    - Configurable components
    - Automatic parameter freezing
    - Safe tensor operations
    
    Args:
        model_paths (dict): Path configurations for pretrained components
        codec_dim (int): Dimension of codec features
        semantic_dim (int): Dimension of semantic features
        device: torch.device,
    """

    def __init__(
        self, model_paths: dict, codec_dim: int, semantic_dim: int, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.model_paths = model_paths
        self.codec_dim = codec_dim
        self.semantic_dim = semantic_dim

        # Component initialization
        self.feature_extractor = self._init_feature_extractor()
        self.semantic_model = self._init_semantic_model()
        self.codec_components = self._init_codec()

        # Load codec parameters
        self._load_codec_weights()

        # Freeze parameters
        self._freeze_parameters()

    def _init_feature_extractor(self) -> AutoFeatureExtractor:
        """Initialize audio feature extractor with HF Auto API"""
        return AutoFeatureExtractor.from_pretrained(self.model_paths["wav2vec"])

    def _init_semantic_model(self) -> Wav2Vec2BertModel:
        """Initialize and configure semantic model"""
        model = Wav2Vec2BertModel.from_pretrained(self.model_paths["wav2vec"], output_hidden_states=True)
        return model.eval().to(self.device)

    def _init_codec(self) -> nn.ModuleDict:
        """Initialize codec components with placeholder structure"""
        return nn.ModuleDict(
            {
                "SemanticEncoder_module": SemanticEncoder(self.semantic_dim, self.semantic_dim, self.semantic_dim),
                "encoder": CodecEncoder_Transformer(),
                "decoder": CodecDecoderVocos(),
                "fc_prior": nn.Linear(self.codec_dim, self.codec_dim),
                "fc_post_a": nn.Linear(self.codec_dim, self.semantic_dim)
            }).to(self.device).eval()

    def _load_codec_weights(self):
        """Load trained codec weights using safetensors"""
        load_model(self.codec_components, self.model_paths["codec"], device=str(self.device))
        print(f"Loaded codec weights from {self.model_paths['codec']}")

    def _freeze_parameters(self):
        """Immobilize all model parameters"""
        for param in self.parameters():
            param.requires_grad_(False)
        print(f"Model parameters frozen on {self.device}")

    def extract_features(self, wav_batch: torch.Tensor, pad: tuple = (0, 0)) -> torch.Tensor:
        padded_wavs = torch.stack([F.pad(wav, pad) for wav in wav_batch])
        batch_feats = []

        for wav in padded_wavs:
            feat = self.feature_extractor(
                wav.cpu().numpy(), sampling_rate=16000, return_tensors="pt").data['input_features']
            batch_feats.append(feat)
        feat_batch = torch.concat(batch_feats, dim=0).to(self.device)
        return feat_batch

    @torch.no_grad()
    def audio2embed(self, waveform: torch.Tensor, pad: tuple = (0, 0)) -> torch.Tensor:
        """
        Extract combined audio features [B, codec_dim + semantic_dim, T]
        
        Args:
            waveform (torch.Tensor): Input audio tensor [B, T]
        """
        waveform = waveform.squeeze(0)

        waveform = waveform.detach().cpu()
        feature = self.extract_features(waveform, pad=pad)
        feature = feature.to(self.device)

        waveform = waveform.to(self.device)

        codec_feat = self._extract_codec_features(waveform)
        semantic_feat = self._extract_semantic_features(feature)

        if codec_feat.shape[-1] != semantic_feat.shape[-1]:
            min_len = min(codec_feat.shape[-1], semantic_feat.shape[-1])
            codec_feat = codec_feat[:, :, :min_len]
            semantic_feat = semantic_feat[:, :, :min_len]
        return torch.cat([semantic_feat, codec_feat], dim=1)

    def _extract_codec_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract processed codec features"""
        # wav = torch.nn.functional.pad(wav, (0, (200 - (wav.shape[1] % 200))))

        vq_emb = self.codec_components["encoder"](wav)
        return vq_emb.transpose(1, 2)

    def _extract_semantic_features(self, feature: torch.Tensor) -> torch.Tensor:
        """Extract and process semantic features"""

        # Semantic processing
        hidden_states = self.semantic_model(feature).hidden_states[16]
        return self.codec_components["SemanticEncoder_module"](hidden_states.transpose(1, 2))

    @torch.no_grad()
    def embed2token(self, features: torch.Tensor) -> torch.LongTensor:
        """
        Convert features to discrete tokens [B, T]
        
        Args:
            features (torch.Tensor): Combined features [B, D, T]
        """
        features.to(self.device)
        # Prior projection
        projected = self.codec_components["fc_prior"](features.transpose(1, 2)).transpose(1, 2)

        # Vector quantization
        return self.codec_components["decoder"](projected, vq=True)[1]

    @torch.no_grad()
    def audio2token(self, wav: torch.Tensor, pad: tuple = (0, 0)) -> torch.LongTensor:
        """
        Convert audio to discrete tokens [B, T]
        
        Args:
            wav (torch.Tensor): Input audio tensor [B, T]
        """
        wav.to(self.device)
        return self.embed2token(self.audio2embed(wav, pad))

    @torch.no_grad()
    def token2audio(self, codes: torch.LongTensor) -> torch.Tensor:
        """
        Reconstruct waveform from discrete codes [B, T]
        
        Args:
            codes (torch.LongTensor): Discrete code indices [B, T]
        """
        codes.to(self.device)
        # Code to embeddings
        embeddings = self.codec_components["decoder"].quantizer.get_output_from_indices(codes.transpose(1,
                                                                                                        2)).transpose(
                                                                                                            1, 2)

        # Post processing
        post_processed = self.codec_components["fc_post_a"](embeddings.transpose(1, 2)).transpose(1, 2)

        # Waveform reconstruction
        return self.codec_components["decoder"](post_processed.transpose(1, 2), vq=False)[0].squeeze(1)
