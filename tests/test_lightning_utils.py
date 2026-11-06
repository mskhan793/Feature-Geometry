"""Shared Lightning utilities: one real train/validate/test cycle on seeded synthetic data (CPU, a few seconds)."""

import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

from src.lightning_utils.callbacks import standard_callbacks
from src.lightning_utils.run_dir import make_run_dir, write_resolved_config
from src.lightning_utils.seeding import seed_everything
from src.lightning_utils.trainer import build_trainer


class TwoGaussians(pl.LightningDataModule):
    """Test fixture: two 2D Gaussian classes, seeded, separate non-overlapping train/val/test draws."""

    def __init__(self, n=64, seed=0):
        super().__init__()
        self.n, self.seed = n, seed

    def _draw(self, offset):
        g = torch.Generator().manual_seed(self.seed + offset)
        y = torch.arange(self.n) % 2
        x = torch.randn(self.n, 2, generator=g) + 6.0 * y[:, None].float()
        return TensorDataset(x, y)

    def setup(self, stage=None):
        self.train, self.val, self.test = self._draw(0), self._draw(1), self._draw(2)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32)


class Linear(pl.LightningModule):
    """Test fixture: logistic regression logging loss and accuracy per split."""

    def __init__(self, lr=0.5):
        super().__init__()
        self.lr = lr
        self.net = torch.nn.Linear(2, 2)

    def _step(self, batch, split):
        x, y = batch
        logits = self.net(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        self.log(f"{split}_loss", loss)
        self.log(f"{split}_acc", (logits.argmax(1) == y).float().mean())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def _fit(tmp_path, seed=0, lr=0.5, max_epochs=3, patience=None):
    seed_everything(seed)
    run_dir = make_run_dir(tmp_path, "smoke", timestamp=f"seed{seed}_lr{lr}")
    dm, model = TwoGaussians(), Linear(lr)
    trainer = build_trainer(
        standard_callbacks(run_dir, monitor="val_acc", patience=patience),
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=dm)
    return trainer, model, dm, run_dir


def test_train_validate_test_cycle_writes_checkpoints_and_metrics(tmp_path):
    trainer, model, dm, run_dir = _fit(tmp_path)
    assert (run_dir / "last.ckpt").is_file()
    assert len(list(run_dir.glob("*_best.ckpt"))) == 1
    rows = [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert all({"train_loss", "val_acc", "lr"} <= r.keys() for r in rows)
    (result,) = trainer.test(model, datamodule=dm, ckpt_path="best", verbose=False)
    assert result["test_acc"] > 0.9  # separable classes: a failed fit would sit near 0.5


def test_same_seed_gives_identical_weights(tmp_path):
    _, a, _, _ = _fit(tmp_path / "a", seed=3)
    _, b, _, _ = _fit(tmp_path / "b", seed=3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_early_stopping_only_when_patience_given(tmp_path):
    assert not any(isinstance(c, EarlyStopping) for c in standard_callbacks(tmp_path, "val_acc"))
    trainer, _, _, _ = _fit(tmp_path, lr=0.0, max_epochs=10, patience=1)  # lr 0: val_acc never improves
    assert trainer.current_epoch == 2


def test_trainer_defaults_and_overrides():
    trainer = build_trainer([], max_epochs=1, accelerator="cpu")
    assert trainer.gradient_clip_val == 1.0 and trainer.logger is None
    assert build_trainer([], max_epochs=1, accelerator="cpu", gradient_clip_val=None).gradient_clip_val is None


def test_resolved_config_written_by_rank_zero_only(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv("RANK", "1")
    assert write_resolved_config(tmp_path, {"seed": 0}) is None
    monkeypatch.setenv("RANK", "0")
    path = write_resolved_config(tmp_path, {"seed": 0, "run_dir": tmp_path})
    assert json.loads(path.read_text()) == {"run_dir": str(tmp_path), "seed": 0}
