"""I-JEPA adapter (software test, random weights): the authors' unmodified module loads with hash checks; the pooled
feature is the mean of the encoder's final patch tokens; it inherits the final LayerNorm's linear constraint
sum_j (f_j - b_j) / w_j = 0; ViT-H/14 has the expected token grid and width; a changed source file is refused; the
multipart-ETag helper matches its definition."""

import hashlib

import pytest
import torch

from src.encoders import ijepa


def test_pooled_feature_and_layernorm_constraint():
    torch.manual_seed(0)
    vit = ijepa.official_module().vit_tiny(img_size=[224], patch_size=14).eval()
    with torch.no_grad():
        vit.norm.weight.uniform_(0.5, 1.5)
        vit.norm.bias.normal_()
        x = torch.randn(3, 3, 224, 224)
        tokens = vit(x)
        f = ijepa.MeanPatchTokens(vit)(x)
    assert tokens.shape == (3, 256, 192) and f.shape == (3, 192)
    assert torch.equal(f, tokens.mean(dim=1))
    c = ((f - vit.norm.bias) / vit.norm.weight).sum(1)
    assert c.abs().max() < 1e-3


def test_vith14_shape_and_source_guard(monkeypatch, tmp_path):
    vit = ijepa.build_vith14()
    assert vit.patch_embed.num_patches == 256 and vit.embed_dim == 1280 and len(vit.blocks) == 32
    bad = tmp_path / "utils"
    bad.mkdir()
    (bad / "tensors.py").write_text("# changed\n")
    monkeypatch.setattr(ijepa, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="differs"):
        ijepa._source("src.utils.tensors")


def test_s3_etag(tmp_path):
    from scripts.benchmark.ijepa_target_weights import s3_etag

    data = bytes(range(256)) * 100
    (tmp_path / "f").write_bytes(data)
    parts = [data[i:i + 1000] for i in range(0, len(data), 1000)]
    expected = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in parts)).hexdigest() + f"-{len(parts)}"
    assert s3_etag(tmp_path / "f", part=1000) == expected
