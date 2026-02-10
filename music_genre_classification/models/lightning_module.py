import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from music_genre_classification.models.baseline import BaselineModel
from music_genre_classification.models.cnn import CNNSpectrogramModel
from music_genre_classification.utils.metrics import calculate_metrics


class MusicGenreLightningModule(LightningModule):
    def __init__(
        self,
        model_name="cnn",
        num_classes=10,
        learning_rate=1e-4,
        weight_decay=1e-5,
        dropout=0.3,
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if model_name == "baseline":
            self.model = BaselineModel(
                num_classes=num_classes, dropout=dropout, **model_kwargs
            )
        elif model_name == "cnn":
            self.model = CNNSpectrogramModel(
                num_classes=num_classes, dropout=dropout, **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["spectrogram"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        metrics = calculate_metrics(logits, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/accuracy",
            metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/f1_macro",
            metrics["f1_macro"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["spectrogram"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        metrics = calculate_metrics(logits, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/accuracy",
            metrics["accuracy"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/f1_macro",
            metrics["f1_macro"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        x, y = batch["spectrogram"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)

        metrics = calculate_metrics(logits, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", metrics["accuracy"], on_step=False, on_epoch=True)
        self.log("test/f1_macro", metrics["f1_macro"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
