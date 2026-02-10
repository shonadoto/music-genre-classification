from pathlib import Path

import torch
from torch.utils.data import Dataset

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


class GTZANDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = Path(data_dir).resolve()
        self.split = split
        self.spectrogram_files = []
        self.labels = []

        spectrograms_dir = self.data_dir / "spectrograms"
        if not spectrograms_dir.exists():
            raise RuntimeError(
                "Precomputed spectrograms were not found in data_dir/spectrograms"
            )

        for genre_idx, genre in enumerate(GENRES):
            genre_path = spectrograms_dir / genre
            if genre_path.exists():
                for spec_file in genre_path.glob("*.pt"):
                    self.spectrogram_files.append(spec_file.resolve())
                    self.labels.append(genre_idx)

    def __len__(self):
        return len(self.spectrogram_files)

    def __getitem__(self, idx):
        spec_path = self.spectrogram_files[idx]
        label = self.labels[idx]

        spectrogram = torch.load(spec_path, map_location="cpu")

        return {
            "spectrogram": spectrogram,
            "label": torch.tensor(label, dtype=torch.long),
        }
