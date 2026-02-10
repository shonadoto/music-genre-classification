from pathlib import Path

import git
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from music_genre_classification.data.datamodule import GTZANDataModule
from music_genre_classification.models.lightning_module import MusicGenreLightningModule


def get_git_commit_id():
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha[:7]


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def train(cfg: DictConfig):
    data_module = GTZANDataModule(**cfg.datamodule.datamodule)
    data_module.prepare_data()
    data_module.setup()

    model = MusicGenreLightningModule(**cfg.model.model)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logger.logger.experiment_name,
        tracking_uri=cfg.logger.logger.tracking_uri,
    )

    mlflow_logger.log_hyperparams(cfg)
    mlflow_logger.log_hyperparams({"git_commit_id": get_git_commit_id()})

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(cfg.trainer.default_root_dir) / "checkpoints",
        filename="epoch_{epoch:02d}-val_loss_{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=cfg.trainer.default_root_dir,
    )

    trainer.fit(model, data_module)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        output_dir = Path(cfg.trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        symlink_path = output_dir / "model_full.ckpt"
        if symlink_path.exists():
            symlink_path.unlink()
        symlink_path.symlink_to(Path(best_model_path).absolute())


if __name__ == "__main__":
    train()
