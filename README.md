
# Diffusion Bridge Language Model for Speech Enhancement

**Model Hub:** [Yorch233/DBLM-SE-1B](https://huggingface.co/Yorch233/DBLM-SE-1B)  
**Code Repository:** [GitHub - Yorch233/DBLM-SE](https://github.com/Yorch233/DBLM-SE)

**DBLM-SE** is a LLaMA-based **Diffusion Bridge Language Model (DBLM)** designed for high-fidelity **speech enhancement**. By modeling the restoration process as a *diffusion bridge* in the latent space, the model leverages an LLM-style architecture to map corrupted speech signals to clean acoustic representations, enabling **joint denoising and dereverberation** in a unified framework.

---

## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Yorch233/DBLM-SE.git
cd DBLM-SE
pip install -r requirements.txt
```

### Download Checkpoint

Download the pretrained model of **DBLM-SE-1B** (with about 1 billion parameter) from Hugging Face:

```bash
huggingface-cli download Yorch233/DBLM-SE-1B --local-dir checkpoints
```

### Inference

Enhance noisy audio using the provided inference script:

```bash
python inference.py \
  --audio_path /path/to/noisy_audio.wav \
  --output_path output \
  --run_path checkpoints \
  --num_steps 10 \
  --local_rank 0
```

This will generate enhanced audio in the specified output directory.

---

## 📋 Model Overview

DBLM-SE-1B enhances noisy and reverberant speech through a hybrid, latent-space processing pipeline:

1. **WavLM Encoder**  
   → Extracts continuous latent representations from input audio (T × 1024).

2. **Diffusion Bridge Language Model (DBLM)**  
   → Iteratively restores clean latents via backward SDE/ODE sampling.  
   → Utilizes causal attention and diffusion conditioning to ensure temporal coherence and stability.

3. **Latent-to-Discrete Translator**  
   → Converts clean continuous latents into **XCodec2 discrete acoustic token IDs** (multi-codebook format).

4. **XCodec2 Decoder**  
   → Synthesizes high-fidelity, clean speech from the restored discrete tokens.

This repository includes the complete DBLM model for latent restoration and acoustic token generation.

---

## ✅ Key Capabilities

- **Speech Denoising**  
  Effectively removes background noise (e.g., babble, traffic, machinery).
  
- **Dereverberation**  
  Suppresses room reverberation, significantly improving speech clarity and ASR performance.

- **Latent Diffusion Bridge**  
  Enables stable, iterative enhancement in a compressed latent space for efficient and high-quality restoration.

- **XCodec2-Compatible Output**  
  Directly generates discrete acoustic tokens compatible with neural vocoders for end-to-end speech reconstruction.

---

## 📦 Checkpoint Details

| Attribute         | Description |
|------------------|-----------|
| **Model Name**   | DBLM-SE-1B |
| **Architecture** | LLaMA-style Transformer |
| **Parameters**   | ~1.0 Billion |
| **Input**        | WavLM-Large continuous latents (T × 1024) |
| **Output**       | XCodec2 discrete acoustic token IDs (multi-codebook) |

The training data for **DBLM-SE-1B** is constructed following the **DNS Challenge** methodology ([Microsoft DNS Challenge](https://github.com/microsoft/DNS-Challenge )), ensuring realistic and diverse noisy-reverberant conditions for robust speech enhancement.

- **Clean Speech**: LibriVox, VCTK
- **Noise**: AudioSet, Freesound, DEMAND  
- **Room Impulse Responses (RIRs)**: OpenSLR26, OpenSLR28

To improve generalization and prevent overfitting, noisy and reverberant training samples are generated **on the fly** during training with the following augmentation strategy:
- **90% probability** of adding background noise (SNR: [-5, 20] dB)
- **50% probability** of applying reverberation via RIR convolution

This dynamic mixing pipeline enables the model to learn **joint denoising and dereverberation** in a realistic and data-efficient manner.

## 🚧 Upcoming Features

- [ ] **Docker Support** – Containerized deployment for easy integration  
- [ ] **Training Code** – Full training pipeline and configuration release  

---

## 📄 License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](LICENSE) file for more details.

---
