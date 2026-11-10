"""FIN-Benthic2 rows, file-based images and source-free selection (synthetic files; unit tests only)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms as T

from src.data.expanded_cohort import select_frames, select_specimens
from src.data.file_images import FileImageDataset
from src.data.finbenthic2_cohort import finbenthic2_frames
from src.data.image_source import image_dataset


def name(cam, spec, run, sec, frac):
    return f"{cam}-{spec}.{run}.2016-08-48-{sec:02d}-{frac:012d}-utc.png"


@pytest.fixture
def fb2(tmp_path):
    rows = []
    for k, (spec, taxon) in enumerate([("Agapetus1", "Agapetus_sp"), ("Agapetus2", "Agapetus_sp"),
                                       ("Elmis1", "Elmis_aenea"), ("Elmis2", "Elmis_aenea")]):
        (tmp_path / spec).mkdir()
        for cam in (0, 1):
            for j in range(4):
                img = name(cam, spec, 1 + (j == 3), 30 + j, 10 ** 11 + j)
                Image.fromarray(np.full((8, 6 + k, 3), 40 * k + j, np.uint8)).save(tmp_path / spec / img)
                rows.append({"individual": spec, "taxon": taxon, "img": img,
                             **{f"fold{f}": ("test" if f == k % 5 else "val" if (f + 1) % 5 == k % 5 else "train")
                                for f in range(5)}})
    csv = tmp_path / "folds.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, tmp_path


def test_rows_parse_camera_run_and_path(fb2):
    csv, _ = fb2
    f = finbenthic2_frames(csv)
    assert set(f.camera) == {0, 1} and f.taxon_group.nunique() == 2
    assert (f.path == f.individual + "/" + f.img).all()
    assert set(f.imaging_run[f.individual == "Agapetus1"]) == {"Agapetus1.1", "Agapetus1.2"}


def test_bad_name_or_wrong_specimen_raises(fb2, tmp_path):
    csv, _ = fb2
    df = pd.read_csv(csv)
    df.loc[0, "img"] = "photo.png"
    df.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="expected form"):
        finbenthic2_frames(tmp_path / "bad.csv")
    df = pd.read_csv(csv)
    df.loc[0, "individual"] = "Elmis1"
    df.to_csv(tmp_path / "bad2.csv", index=False)
    with pytest.raises(ValueError, match="specimen id"):
        finbenthic2_frames(tmp_path / "bad2.csv")


def test_file_dataset_returns_the_named_image_and_raises_on_missing(fb2):
    csv, root = fb2
    f = finbenthic2_frames(csv)
    ds = image_dataset(f, T.PILToTensor(), image_root=str(root))
    x, i = ds[5]
    assert i == 5 and x.shape[1:] == (8, 6) and int(x[0, 0, 0]) == 5 % 4
    with pytest.raises(FileNotFoundError):
        FileImageDataset(root, ["Agapetus1/missing.png"], T.PILToTensor())[0]
    with pytest.raises(ValueError, match="exactly one"):
        image_dataset(f, T.PILToTensor())


def test_packed_hdf5_holds_the_files_and_cohort_positions(fb2, tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    csv, root = fb2
    f = finbenthic2_frames(csv)
    cohort = f.iloc[[5, 2, 30]].reset_index(drop=True)
    cohort.to_parquet(tmp_path / "cohort.parquet")
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "scripts/pack_finbenthic2_hdf5.py", "--csv", str(csv), "--images", str(root),
                        "--out", str(tmp_path / "fb2.h5"), "--cohort", str(tmp_path / "cohort.parquet")],
                       cwd=repo, env={"PYTHONPATH": str(repo), "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = pd.read_parquet(tmp_path / "cohort_hdf5.parquet")
    assert rows.img.tolist() == cohort.img.tolist() and rows.hdf5_row.tolist() == [5, 2, 30]
    from_hdf5 = image_dataset(rows, T.PILToTensor(), hdf5=str(tmp_path / "fb2.h5"))
    from_files = image_dataset(rows, T.PILToTensor(), image_root=str(root))
    assert all(np.array_equal(from_hdf5[i][0], from_files[i][0]) for i in range(len(rows)))


def test_source_free_selection_is_capped_and_deterministic(fb2):
    csv, _ = fb2
    f = finbenthic2_frames(csv)
    a = select_specimens(f, 1, 0, seed=7, source=None)
    assert len(a) == 2 and a.equals(select_specimens(f, 1, 0, seed=7, source=None))
    assert "dataset" not in a.columns
    frames = select_frames(f[f.individual.isin(a.individual)], 2)
    assert (frames.groupby(["individual", "camera"]).size() == 2).all()
    assert frames.equals(select_frames(f[f.individual.isin(a.individual)], 2))
