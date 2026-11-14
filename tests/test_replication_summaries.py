"""Replication summaries (software test, fixture logs only): Slurm-log provenance parsing (which run wrote an output,
billed project), per-arm pair counts, headline-table formatting."""

import pandas as pd

from scripts.replication import completion_matrix as cm
from scripts.replication import headline_tables as ht


def _log(path, job, task, tail):
    path.write_text(f"start 2026-11-13T20:00:40+02:00 job {job} node nid1 code ab8cdb78185d6af0\n"
                    f"task {task} -> /x\n{tail}\n")


def test_provenance_outcomes_and_accounts(tmp_path):
    _log(tmp_path / "fg_repl_b1_11_1.out", "11_1", "dinov2_l df24 official lw", "end 2026-11-13T21:00:00+02:00")
    _log(tmp_path / "fg_repl_b2_12_1.out", "12_1", "dinov2_l df24 official lw", "already done")
    _log(tmp_path / "fg_repl_b2_12_2.out", "12_2", "bioclip2 df24 official fda", "")
    (tmp_path / "fg_repl_b1_11_1.err").write_text("ignored")
    sacct = tmp_path / "sacct.txt"
    pd.DataFrame({"JobID": ["11_1", "12_1", "12_2"], "Account": ["p534", "p824", "p824"]}).to_csv(sacct, sep="|", index=False)
    p = cm.provenance(tmp_path, [sacct]).set_index("job")
    assert p.loc["11_1", "outcome"] == "wrote output" and p.loc["11_1", "account"] == "p534"
    assert p.loc["12_1", "outcome"] == "skipped (already done)" and p.loc["12_2", "outcome"] == "did not finish"
    assert p.loc["11_1", "code"] == "ab8cdb7" and p.loc["11_1", "batch"] == "fg_repl_b1"
    assert p.loc["12_2", "analysis"] == "fda" and p.loc["12_2", "encoder"] == "bioclip2"


def test_pair_arms_split_valid_and_not_eligible():
    forms = {"full": {"pairs_valid": 6}, "diagonal": {"pairs_valid_train": 10}, "spherical": {"pairs_valid": 10}}
    rows = list(cm.arms_pairs(None, [("pcs64", forms)], 10))
    assert ("pcs64", "full", "ok", 6) in rows and ("pcs64", "full", "not eligible", 4) in rows
    assert ("pcs64", "diagonal", "ok", 10) in rows and len(rows) == 4


def test_headline_formatting():
    t = pd.DataFrame({"a": ["1.0"], "b": ["n/a"]}, index=["ds"])
    assert ht.markdown(t).splitlines() == ["| dataset | a | b |", "| --- | --- | --- |", "| ds | 1.0 | n/a |"]
    r = {"mean": 0.1234, "ci": True, "partitions_ci_above_0": 3.0, "partitions_ci_below_0": 0.0}
    assert ht.cell(r, ".3f") == "0.123 (+3/-0)"
    assert ht.cell({**r, "ci": False}, ".2e") == "1.23e-01"
