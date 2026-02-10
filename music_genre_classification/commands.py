import subprocess
import sys

import click

from music_genre_classification.deployment.telegram_bot import run_bot
from music_genre_classification.inference.infer import infer as run_infer
from music_genre_classification.training.test import run_test


@click.group()
def cli():
    pass


@cli.command()
@click.argument("overrides", nargs=-1)
def train(overrides):
    subprocess.run(
        [sys.executable, "-m", "music_genre_classification.training.train", *overrides]
    )


@cli.command()
@click.option("--model_path", required=True)
@click.option("--audio_path", required=True)
@click.option("--output_path", default=None)
def infer(model_path, audio_path, output_path):
    run_infer(model_path, audio_path, output_path)


@cli.command()
@click.argument("model_path")
@click.option("--token", default=None, help="Telegram bot token")
def bot(model_path, token):
    run_bot(model_path, token)


@cli.command()
@click.argument("model_path")
def test(model_path):
    run_test(model_path)


if __name__ == "__main__":
    cli()
