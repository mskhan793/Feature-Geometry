"""End-to-end extraction through Trainer.predict on the fixture HDF5 with a tiny fixture encoder (CPU)."""

import numpy as np
import torch
from torch import nn
from torchvision import transforms as T

from src.data.aquamonitor_manifest import build_manifest
from src.data.hdf5_images import Hdf5ImageDataset
from src.data.image_source import image_dataset
from src.features import chunk_store
from src.features.extract import extract_chunks
from src.lightning_utils.trainer import build_trainer

INFO = {"encoder": "fixture", "model_id": "fixture", "transform": "ToTensor"}


class MeanColour(nn.Module):
    """Fixture encoder: per-channel mean and std of the image (6 features)."""

    def forward(self, x):
        return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)


def run(aqua_files, cache_dir, chunk_size=2):
    m = build_manifest(aqua_files["parquet"], aqua_files["hdf5"])
    trainer = build_trainer([], max_epochs=1, accelerator="cpu", deterministic="warn", enable_progress_bar=False,
                            enable_model_summary=False)
    written = extract_chunks(trainer, MeanColour(), INFO, m,
                             lambda rows: image_dataset(rows, T.ToTensor(), hdf5=str(aqua_files["hdf5"])), cache_dir,
                             chunk_size, range(chunk_store.n_chunks(len(m), chunk_size)), batch_size=2, num_workers=0,
                             log=lambda s: None)
    return m, written


def load(cache_dir, n_rows, chunk_size=2):
    return np.concatenate([np.load(chunk_store.paths(cache_dir, k)[0]) for k in range(chunk_store.n_chunks(n_rows, chunk_size))])


def test_features_match_each_named_image(aqua_files, tmp_path):
    m, written = run(aqua_files, tmp_path)
    assert len(written) == chunk_store.n_chunks(len(m), 2)
    features = load(tmp_path, len(m))
    ds = Hdf5ImageDataset(str(aqua_files["hdf5"]), m["img"].tolist(), m["hdf5_row"].tolist(), T.ToTensor())
    expected = torch.stack([MeanColour()(ds[i][0][None])[0] for i in range(len(ds))]).numpy()
    np.testing.assert_allclose(features, expected, rtol=1e-6)


def test_resume_skips_done_chunks(aqua_files, tmp_path):
    m, _ = run(aqua_files, tmp_path)
    stamp = {k: chunk_store.paths(tmp_path, k)[0].stat().st_mtime_ns for k in range(chunk_store.n_chunks(len(m), 2))}
    _, written = run(aqua_files, tmp_path)
    assert written == []
    assert stamp == {k: chunk_store.paths(tmp_path, k)[0].stat().st_mtime_ns for k in stamp}


def test_repeated_extraction_is_identical(aqua_files, tmp_path):
    m, _ = run(aqua_files, tmp_path / "a")
    run(aqua_files, tmp_path / "b")
    np.testing.assert_array_equal(load(tmp_path / "a", len(m)), load(tmp_path / "b", len(m)))
