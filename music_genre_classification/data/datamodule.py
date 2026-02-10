from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from music_genre_classification.data.data_manager import ensure_data_available
from music_genre_classification.data.dataset import GTZANDataset


class GTZANDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir="data/dataset",
        batch_size=32,
        num_workers=4,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
        sample_rate=22050,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        clip_duration=30,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_duration = clip_duration

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        ensure_data_available(
            self.data_dir,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            clip_duration=self.clip_duration,
        )

    def setup(self, stage=None):
        def build_stratified_subsets(dataset):
            indices = list(range(len(dataset)))
            labels = dataset.labels

            train_indices, temp_indices = train_test_split(
                indices,
                test_size=self.val_split + self.test_split,
                random_state=self.random_seed,
                stratify=labels,
            )

            temp_labels = [labels[i] for i in temp_indices]
            test_ratio_in_temp = self.test_split / (self.val_split + self.test_split)

            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=test_ratio_in_temp,
                random_state=self.random_seed,
                stratify=temp_labels,
            )

            return (
                Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices),
            )

        if stage == "fit" or stage is None:
            full_dataset = GTZANDataset(
                self.data_dir,
                split="train",
            )

            train_dataset, val_dataset, test_dataset = build_stratified_subsets(
                full_dataset
            )

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

        if stage == "test" or stage is None:
            if self.test_dataset is None:
                full_dataset = GTZANDataset(
                    self.data_dir,
                    split="test",
                )

                _, _, test_dataset = build_stratified_subsets(full_dataset)
                self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
