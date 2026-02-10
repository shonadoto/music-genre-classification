import logging
import os
import tempfile
from pathlib import Path

import soundfile as sf
import torch
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from music_genre_classification.data.dataset import GENRES
from music_genre_classification.models.lightning_module import MusicGenreLightningModule
from music_genre_classification.utils.audio_processing import (
    audio_to_mel_spectrogram,
    load_audio,
    normalize_spectrogram,
)

logger = logging.getLogger(__name__)


class GenreBot:
    def __init__(self, model_path: str):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = MusicGenreLightningModule.load_from_checkpoint(
            model_path, map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded from %s on device=%s", model_path, self.device)
        self.sample_rate = 22050
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.chunk_duration_sec = 30
        self.target_frames = (
            int(self.chunk_duration_sec * self.sample_rate / self.hop_length) + 1
        )

    def _predict_chunk(self, audio_chunk):
        mel_spec = audio_to_mel_spectrogram(
            audio_chunk,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec = normalize_spectrogram(mel_spec)
        spectrogram = torch.from_numpy(mel_spec).float().unsqueeze(0)

        frames = spectrogram.shape[-1]
        if frames > self.target_frames:
            spectrogram = spectrogram[:, :, : self.target_frames]
        elif frames < self.target_frames:
            pad_frames = self.target_frames - frames
            spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_frames))

        logits = self.model(spectrogram.unsqueeze(0).to(self.device))
        return torch.softmax(logits, dim=1)[0].detach().cpu()

    def _convert_to_wav_if_needed(self, audio_path: Path):
        if audio_path.suffix.lower() == ".wav":
            logger.info("Input is already WAV: %s", audio_path.name)
            return audio_path, False

        converted_path = audio_path.with_suffix(".wav")
        logger.info("Converting %s -> %s", audio_path.name, converted_path.name)
        audio = load_audio(audio_path, sample_rate=self.sample_rate)
        sf.write(str(converted_path), audio, self.sample_rate)
        return converted_path, True

    def predict(self, audio_path: Path):
        logger.info("Starting prediction for file: %s", audio_path.name)
        wav_path, is_temp_wav = self._convert_to_wav_if_needed(audio_path)
        audio = load_audio(wav_path, sample_rate=self.sample_rate)
        chunk_size = self.chunk_duration_sec * self.sample_rate
        total_chunks = max(1, (len(audio) + chunk_size - 1) // chunk_size)
        logger.info(
            "Audio loaded: samples=%d, duration=%.2fs, chunks=%d",
            len(audio),
            len(audio) / self.sample_rate,
            total_chunks,
        )

        try:
            with torch.no_grad():
                chunk_probabilities = []
                for chunk_idx, start in enumerate(
                    range(0, len(audio), chunk_size), start=1
                ):
                    chunk = audio[start : start + chunk_size]
                    if len(chunk) == 0:
                        continue
                    logger.info("Processing chunk %d/%d", chunk_idx, total_chunks)
                    chunk_probabilities.append(self._predict_chunk(chunk))

                if not chunk_probabilities:
                    raise RuntimeError("Audio could not be split into valid chunks")

                probabilities = torch.stack(chunk_probabilities, dim=0).mean(dim=0)
                predicted_class = torch.argmax(probabilities).item()
                predicted_genre = GENRES[predicted_class]
                logger.info(
                    "Prediction completed: genre=%s, confidence=%.4f",
                    predicted_genre,
                    probabilities[predicted_class].item(),
                )
        finally:
            if is_temp_wav and wav_path.exists():
                wav_path.unlink()
                logger.info("Temporary converted file removed: %s", wav_path.name)

        return predicted_genre, probabilities


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me an audio file (wav/mp3/ogg/m4a/flac). "
        "Long audio is split into 30-second chunks automatically."
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    if message is None:
        return

    file_to_download = None
    filename = ""

    if message.document:
        file_to_download = await message.document.get_file()
        filename = message.document.file_name or "audio.bin"
        logger.info("Received document: %s", filename)
    elif message.audio:
        file_to_download = await message.audio.get_file()
        filename = message.audio.file_name or "audio.bin"
        logger.info("Received audio: %s", filename)
    elif message.voice:
        file_to_download = await message.voice.get_file()
        filename = "voice.ogg"
        logger.info("Received voice message")
    else:
        await message.reply_text("Please send audio as file/document.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_audio_path = Path(tmp_dir) / filename
        await file_to_download.download_to_drive(str(tmp_audio_path))
        logger.info("Downloaded file to: %s", tmp_audio_path)

        bot_model: GenreBot = context.application.bot_data["genre_model"]
        try:
            genre, probs = bot_model.predict(tmp_audio_path)
        except Exception as exc:
            logger.exception("Failed to process audio")
            await message.reply_text(f"Failed to process audio: {exc}")
            return

        lines = [f"Predicted genre: {genre}", "", "Probabilities:"]
        for i, genre_name in enumerate(GENRES):
            lines.append(f"{genre_name}: {probs[i].item():.4f}")

        await message.reply_text("\n".join(lines))
        logger.info("Result sent to user")


def run_bot(model_path: str, token: str | None = None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    bot_token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        raise RuntimeError("Provide Telegram token via --token or TELEGRAM_BOT_TOKEN")

    logger.info("Starting Telegram bot")
    app = ApplicationBuilder().token(bot_token).build()
    app.bot_data["genre_model"] = GenreBot(model_path)

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(
        MessageHandler(
            filters.Document.ALL | filters.AUDIO | filters.VOICE, handle_audio
        )
    )

    logger.info("Bot polling started")
    app.run_polling()
