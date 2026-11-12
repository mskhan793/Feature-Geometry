"""Benchmark manifests: derived validation (stratified, units whole, train-only, deterministic), content-hash units,
CIFAR pickle decoding and duplicates, the publisher-file builders, support summaries (software fixtures only)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.data import benchmark_manifests as bm
from src.data.cifar_images import CifarImageDataset


def test_derived_validation_is_stratified_whole_units_and_deterministic():
    train = pd.DataFrame({"unit_id": [f"u{i // 2}" for i in range(200)], "label": ["a"] * 120 + ["b"] * 80})
    v1, v2 = bm.derive_validation(train, seed=1), bm.derive_validation(train, seed=1)
    assert v1.equals(v2) and not v1.equals(bm.derive_validation(train, seed=2))
    units = train.assign(v=v1).groupby("unit_id").v.nunique()
    assert (units == 1).all()                                  # a unit is never split
    per = train.assign(v=v1).drop_duplicates("unit_id").groupby("label").v.sum()
    assert per.to_dict() == {"a": 6, "b": 4}                   # round(0.1 x 60), round(0.1 x 40) units


def write_cifar(folder, train_rows, test_rows):
    meta = {"fine_label_names": ["x", "y"], "coarse_label_names": ["c"]}
    pickle.dump(meta, open(folder / "meta", "wb"))
    for part, rows in (("train", train_rows), ("test", test_rows)):
        pickle.dump({"filenames": [f"f{i}.png" for i in range(len(rows))], "fine_labels": [r[1] for r in rows],
                     "coarse_labels": [0] * len(rows), "data": np.stack([r[0] for r in rows]).astype(np.uint8)},
                    open(folder / part, "wb"))


def test_cifar_builder_units_duplicates_and_decoding(tmp_path):
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, 3072) for _ in range(40)]
    train = [(imgs[i], i % 2) for i in range(40)] + [(imgs[0], 1)]        # a duplicate of image 0, other label
    write_cifar(tmp_path, train, [(imgs[1], 1), (rng.integers(0, 256, 3072), 0)])  # test repeats train image 1
    m = bm.cifar100(tmp_path)
    assert m.source_id.is_unique and set(m.split) == {"train", "val", "test"} and (m.val_origin == "derived").all()
    s = bm.support(m, "split")
    assert s["images_sharing_a_unit"] == 4 and s["units_with_several_labels"] == 1 and s["units_in_several_splits"] == 1
    assert m.loc[m.source_id == "train[0]", "split"].item() == m.loc[m.source_id == "train[40]", "split"].item()
    ds = CifarImageDataset(tmp_path, ["train[3]"], lambda im: np.asarray(im))
    img, _ = ds[0]
    assert img.shape == (32, 32, 3) and np.array_equal(img.transpose(2, 0, 1).ravel(), imgs[3])
    with pytest.raises(AttributeError):
        CifarImageDataset(tmp_path, ["val[0]"], lambda im: im)


def test_file_builders_and_content_units(tmp_path):
    meta = tmp_path / "food" / "meta"
    meta.mkdir(parents=True)
    (meta / "train.txt").write_text("\n".join(f"apple/{i}" for i in range(30)) + "\n")
    (meta / "test.txt").write_text("apple/100\napple/101\n")
    content = pd.Series({f"food-101/images/apple/{i}.jpg": f"h{min(i, 28)}" for i in [*range(30), 100, 101]})
    f = bm.food101(meta, content)
    assert len(f) == 32 and f.available.all() and f[f.split == "test"].shape[0] == 2
    assert f.loc[f.source_id == "apple/29", "unit_id"].item() == "h28"     # identical bytes share a unit
    with pytest.raises(AssertionError):
        bm.food101(meta, content.drop("food-101/images/apple/3.jpg"))     # a listed image missing from the archive
    ann = tmp_path / "ann"
    ann.mkdir()
    (ann / "trainval.txt").write_text("\n".join(f"Abyssinian_{i} 1 1 1" for i in range(20)) + "\n")
    (ann / "test.txt").write_text("Abyssinian_99 1 1 1\n")
    p = bm.oxford_pets(ann, pd.Series({f"images/Abyssinian_{i}.jpg": f"p{i}" for i in [*range(20), 99]}))
    assert set(p.label) == {"Abyssinian"} and set(p.coarse_label) == {"cat"} and (p.split == "val").sum() == 2
    data = tmp_path / "air"
    data.mkdir()
    for s, ids in (("train", ["1", "2"]), ("val", ["3"]), ("test", ["4"])):
        (data / f"images_variant_{s}.txt").write_text("".join(f"{i} 707-320\n" for i in ids))
        (data / f"images_family_{s}.txt").write_text("".join(f"{i} Boeing 707\n" for i in ids))
    a = bm.fgvc_aircraft(data, pd.Series({f"fgvc-aircraft-2013b/data/images/{i}.jpg": f"a{i}" for i in "1234"}))
    assert a.split.tolist() == ["train", "train", "val", "test"] and (a.val_origin == "official").all()
    assert set(a.label) == {"707-320"} and set(a.coarse_label) == {"Boeing 707"}


def test_pack_archive_hdf5_tar_and_zip(tmp_path):
    import hashlib
    import tarfile
    import zipfile

    import h5py

    from tests.test_class_geometry_pilot import run
    src = tmp_path / "src" / "ds" / "images"
    src.mkdir(parents=True)
    for n in "abc":
        (src / f"{n}.jpg").write_bytes(n.encode() * 10)
    with tarfile.open(tmp_path / "t.tar.gz", "w:gz") as t:
        t.add(tmp_path / "src" / "ds", arcname="ds")
    refs = ["ds/images/c.jpg", "ds/images/a.jpg"]
    m = pd.DataFrame({"source_id": ["c", "a"], "image_ref": refs, "unit_id": ["u1", "u2"], "label": ["x", "y"], "split": ["train", "val"]})
    m.to_parquet(tmp_path / "m.parquet", index=False)
    (tmp_path / "c.tsv").write_text("".join(f"{r}\t{hashlib.sha1((r[-5] * 10).encode()).hexdigest()}\n" for r in refs))
    run(["scripts/benchmark/pack_archive_hdf5.py", "--manifest", str(tmp_path / "m.parquet"), "--archives", str(tmp_path / "t.tar.gz"),
         "--content", str(tmp_path / "c.tsv"), "--out", str(tmp_path / "o.h5"), "--out-manifest", str(tmp_path / "x.parquet")])
    with h5py.File(tmp_path / "o.h5") as h:
        assert [v.decode() for v in h["metadata/img"][:]] == ["c", "a"] and h["images"][0].tobytes() == b"c" * 10
    assert pd.read_parquet(tmp_path / "x.parquet").hdf5_row.tolist() == [0, 1]
    (tmp_path / "zips").mkdir()
    with zipfile.ZipFile(tmp_path / "zips" / "p.zip", "w") as z:
        z.writestr("img/q.jpg", b"qq")
    pd.DataFrame({"source_id": ["q"], "image_ref": ["p.zip:img/q.jpg"], "unit_id": ["q"], "label": ["x"],
                  "split": ["train"]}).to_parquet(tmp_path / "z.parquet", index=False)
    run(["scripts/benchmark/pack_archive_hdf5.py", "--manifest", str(tmp_path / "z.parquet"), "--archives", str(tmp_path / "zips"),
         "--out", str(tmp_path / "z.h5"), "--out-manifest", str(tmp_path / "zx.parquet")])
    with h5py.File(tmp_path / "z.h5") as h:
        assert h["images"][0].tobytes() == b"qq"


def test_siglip2_encoder_wiring(monkeypatch):
    """The siglip2 branch: pinned revision, the checkpoint's own processor, pooled vision output (tiny random model)."""
    import transformers
    from PIL import Image

    from src.encoders import frozen_encoders as fe
    calls = {}
    cfg = transformers.SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                          image_size=32, patch_size=16)
    proc = transformers.SiglipImageProcessor(size={"height": 32, "width": 32})

    def model(repo, revision):
        calls["model"] = (repo, revision)
        return transformers.SiglipVisionModel(cfg)

    def processor(repo, revision):
        calls["processor"] = (repo, revision)
        return proc
    monkeypatch.setattr(transformers.SiglipVisionModel, "from_pretrained", staticmethod(model))
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", staticmethod(processor))
    enc, transform, info = fe.load_encoder("siglip2")
    x = transform(Image.new("RGB", (50, 40), (255, 0, 0)))
    assert x.shape == (3, 32, 32) and torch_allclose(x[0], 1.0) and torch_allclose(x[1], -1.0)  # rescale + mean/std 0.5
    out = enc(x[None])
    assert out.shape == (1, 32) and calls["model"] == calls["processor"] == ("google/siglip2-so400m-patch16-256", fe.REVISIONS["siglip2"])
    assert "e8708ab7" in info["library"] and not enc.training


def torch_allclose(t, v):
    import torch
    return bool(torch.allclose(t, torch.full_like(t, v), atol=1e-6))
