from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16000


def load_audio(file_path: str, target_sample_rate: int = TARGET_SAMPLE_RATE):
    """
    Load an audio file and convert it to mono 16 kHz audio.
    """

    waveform, sample_rate = torchaudio.load(file_path)

    # Convert stereo/multi-channel audio to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)

    return waveform, target_sample_rate


def normalize_audio(waveform: torch.Tensor):
    """
    Normalize waveform amplitude.
    """

    max_value = waveform.abs().max()

    if max_value > 0:
        waveform = waveform / max_value

    return waveform


def load_and_preprocess(file_path: str):
    """
    Complete audio loading + preprocessing pipeline.
    """

    waveform, sample_rate = load_audio(file_path)

    waveform = normalize_audio(waveform)

    return waveform, sample_rate


def save_waveform(waveform, sample_rate, output_path):
    """
    Save a processed waveform.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(
        str(output_path),
        waveform.unsqueeze(0),
        sample_rate,
    )