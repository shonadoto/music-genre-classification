from pathlib import Path

from hydra import compose, initialize_config_dir
from pytorch_lightning import Trainer

from music_genre_classification.data.datamodule import GTZANDataModule
from music_genre_classification.models.lightning_module import MusicGenreLightningModule


def run_test(model_path: str):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train")

    data_module = GTZANDataModule(**cfg.datamodule.datamodule)
    data_module.prepare_data()
    data_module.setup(stage="test")

    model = MusicGenreLightningModule.load_from_checkpoint(model_path)

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        default_root_dir=cfg.trainer.default_root_dir,
    )
    return trainer.test(model, datamodule=data_module)
