import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import gdown
import torch
from dvc.repo import Repo
from tqdm import tqdm

from music_genre_classification.data.dataset import GENRES
from music_genre_classification.utils.audio_processing import prepare_audio_for_model

GDRIVE_DATASET_URL = (
    "https://drive.google.com/file/d/1azKoUx8MiwB7OLY_qaxUD3T5VHsQQACJ/view?usp=sharing"
)


def _extract_archive(archive_path: Path, extract_dir: Path):
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return


def _find_genres_root(extract_dir: Path) -> Path:
    direct = extract_dir / "genres_original"
    if direct.exists():
        return direct

    nested = extract_dir / "Data" / "genres_original"
    if nested.exists():
        return nested

    for candidate in extract_dir.rglob("genres_original"):
        if candidate.is_dir():
            return candidate

    raise RuntimeError("genres_original was not found after archive extraction")


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Project root was not found")


def _ensure_dvc_initialized(project_root: Path):
    if (project_root / ".dvc").exists():
        return
    subprocess.run(
        [sys.executable, "-m", "dvc", "init", "--no-scm"],
        cwd=project_root,
        check=True,
    )


def _find_repo_root() -> Path:
    project_root = _find_project_root()
    _ensure_dvc_initialized(project_root)
    return project_root


def _ensure_local_remote(repo_root: Path):
    storage_path = repo_root / "dvc_storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "remote",
            "add",
            "-f",
            "-d",
            "local_storage",
            "dvc_storage",
        ],
        cwd=repo_root,
        check=True,
    )


def download_data(data_dir: str = "data/dataset"):
    repo_root = _find_repo_root()
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    target_directory = data_path / "genres_original"
    download_dir = data_path / "_downloads"
    extract_dir = data_path / "_extracted"
    archive_path = download_dir / "dataset_archive"

    if target_directory.exists():
        shutil.rmtree(target_directory)
    if download_dir.exists():
        shutil.rmtree(download_dir)
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    downloaded_path = gdown.download(
        url=GDRIVE_DATASET_URL,
        output=str(archive_path),
        quiet=False,
        fuzzy=True,
    )
    if downloaded_path is None:
        raise RuntimeError("Failed to download dataset from Google Drive")

    real_archive_path = Path(downloaded_path)
    _extract_archive(real_archive_path, extract_dir)
    genres_original_path = _find_genres_root(extract_dir)

    shutil.copytree(str(genres_original_path), str(target_directory))
    shutil.rmtree(download_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    return target_directory


def _build_spectrograms(
    data_path: Path,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    clip_duration: int,
):
    genres_path = data_path / "genres_original"
    if not genres_path.exists():
        raise RuntimeError("genres_original is missing")

    spectrograms_path = data_path / "spectrograms"
    if spectrograms_path.exists():
        shutil.rmtree(spectrograms_path)
    spectrograms_path.mkdir(parents=True, exist_ok=True)

    target_frames = int(clip_duration * sample_rate / hop_length) + 1
    saved_count = 0
    skipped_count = 0
    all_wav_files = []

    for genre in GENRES:
        src_genre_path = genres_path / genre
        if src_genre_path.exists():
            all_wav_files.extend(list(src_genre_path.glob("*.wav")))

    progress = tqdm(all_wav_files, desc="Building spectrograms", unit="file")

    for wav_file in progress:
        genre = wav_file.parent.name
        progress.set_postfix_str(genre)
        dst_genre_path = spectrograms_path / genre
        dst_genre_path.mkdir(parents=True, exist_ok=True)

        try:
            spectrogram = prepare_audio_for_model(
                wav_file,
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )

            frames = spectrogram.shape[-1]
            if frames > target_frames:
                spectrogram = spectrogram[:, :, :target_frames]
            elif frames < target_frames:
                pad_frames = target_frames - frames
                spectrogram = torch.nn.functional.pad(spectrogram, (0, pad_frames))

            output_file = dst_genre_path / f"{wav_file.stem}.pt"
            torch.save(spectrogram, output_file)
            saved_count += 1
        except Exception as exc:
            skipped_count += 1
            print(f"[WARN] Failed to process {wav_file.name}: {exc}")

    if saved_count == 0:
        raise RuntimeError("No valid audio files were converted to spectrograms")
    if skipped_count > 0:
        print(
            f"""[INFO] Spectrograms built:
{saved_count}, skipped broken files: {skipped_count}"""
        )

    return spectrograms_path


def ensure_data_available(
    data_dir: str = "data/dataset",
    sample_rate: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    clip_duration: int = 30,
):
    repo_root = _find_repo_root()
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = repo_root / data_path

    genres_path = data_path / "genres_original"
    spectrograms_path = data_path / "spectrograms"
    _ensure_local_remote(repo_root)
    repo = Repo(str(repo_root))
    spectrograms_dvc = data_path / "spectrograms.dvc"

    if spectrograms_dvc.exists():
        try:
            repo.pull(targets=[str(spectrograms_dvc.relative_to(repo_root))])
        except Exception:
            pass
        if spectrograms_path.exists() and any(spectrograms_path.rglob("*.pt")):
            return spectrograms_path

    if not genres_path.exists() or not any(genres_path.rglob("*.wav")):
        genres_dvc = data_path / "genres_original.dvc"
        if genres_dvc.exists():
            try:
                repo.pull(targets=[str(genres_dvc.relative_to(repo_root))])
            except Exception:
                pass

    if not genres_path.exists() or not any(genres_path.rglob("*.wav")):
        download_data(str(data_path))

    spectrograms_path = _build_spectrograms(
        data_path=data_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        clip_duration=clip_duration,
    )

    repo.add(targets=[str(spectrograms_path.relative_to(repo_root))])
    try:
        repo.push(
            targets=[str((data_path / "spectrograms.dvc").relative_to(repo_root))]
        )
    except Exception:
        pass

    return spectrograms_path
