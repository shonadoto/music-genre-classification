import librosa
import numpy as np
import torch


def load_audio(file_path, sample_rate=22050):
    audio, _ = librosa.load(str(file_path), sr=sample_rate, mono=True)
    return audio


def audio_to_mel_spectrogram(
    audio,
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=11025.0,
):
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def normalize_spectrogram(spectrogram):
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    if max_val - min_val > 0:
        normalized = (spectrogram - min_val) / (max_val - min_val)
    else:
        normalized = spectrogram
    return normalized


def prepare_audio_for_model(
    audio_path,
    sample_rate=22050,
    n_fft=2048,
    hop_length=512,
    n_mels=128,
    fmin=0.0,
    fmax=11025.0,
):
    audio = load_audio(audio_path, sample_rate)
    mel_spec = audio_to_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_spec = normalize_spectrogram(mel_spec)
    mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0)
    return mel_spec
