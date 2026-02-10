from pathlib import Path

import torch

from music_genre_classification.data.data_manager import ensure_data_available
from music_genre_classification.data.dataset import GENRES
from music_genre_classification.models.lightning_module import MusicGenreLightningModule
from music_genre_classification.utils.audio_processing import prepare_audio_for_model


def infer(model_path, audio_path, output_path=None):
    ensure_data_available("data/dataset")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MusicGenreLightningModule.load_from_checkpoint(
        model_path, map_location=device
    )
    model.to(device)
    model.eval()

    spectrogram = prepare_audio_for_model(
        Path(audio_path),
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
    )

    with torch.no_grad():
        logits = model(spectrogram.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_genre = GENRES[predicted_class]

    print(f"Predicted genre: {predicted_genre}")
    print("Probabilities:")
    for i, genre in enumerate(GENRES):
        print(f"  {genre}: {probabilities[0][i].item():.4f}")

    if output_path:
        with open(output_path, "w") as f:
            f.write(f"Predicted genre: {predicted_genre}\n")
            f.write("Probabilities:\n")
            for i, genre in enumerate(GENRES):
                f.write(f"  {genre}: {probabilities[0][i].item():.4f}\n")

    return predicted_genre, probabilities[0].numpy()
