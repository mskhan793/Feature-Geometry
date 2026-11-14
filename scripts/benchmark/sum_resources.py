"""Resources of saved LUMI job listings (task.md d57a630): ``sacct -X -P -o
JobID,JobName,Partition,State,Start,End,ElapsedRaw,AllocCPUS,AllocTRES`` -> per-job rows and totals per job-name
prefix. GPU jobs: GPU-hours = elapsed hours x allocated GCDs / 2 (LUMI counts one MI250X = 2 GCDs as one GPU). CPU
jobs: allocated core-hours = elapsed hours x allocated cores; billed CPU core-hours = elapsed hours x the ``billing``
TRES (LUMI bills the larger of the core and the memory share). Energy = the ``energy`` TRES (joules, as recorded by
Slurm; absent for some jobs) in kWh. Allocation-based, not LUMI's accounting database.

Usage: sum_resources.py --sacct FILE [FILE ...] --out FILE.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

GROUPS = {"fg_c0_dl": "C0 download", "fg_pack": "C0 packing", "fg_c1_": "C1 pilots", "fg_c1_checks": "C1 checks", "fg_c2_": "C2 GPU extraction",
          "fg_c2chk_": "C2 CPU checks", "fg_c3_": "C3 records",
          "fg_repl_": "geometry replication", "fg_ijepa_": "I-JEPA gates", "fg_gbcm": "matched GBC"}


def tres(s: str, key: str) -> int:
    """Value of one AllocTRES entry; the key may carry a type suffix (``gres/gpu:mi250=8`` for ``gres/gpu``)."""
    return next((int(x.split("=")[1]) for x in s.split(",") if x.split("=")[0].split(":")[0] == key), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sacct", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    t = pd.concat([pd.read_csv(f, sep="|", dtype=str) for f in a.sacct]).drop_duplicates("JobID")
    t["hours"] = t.ElapsedRaw.astype(int) / 3600
    t["gcds"] = t.AllocTRES.fillna("").map(lambda s: tres(s, "gres/gpu"))
    t["gpu_h"] = t.hours * t.gcds / 2
    t["core_h"] = (t.hours * t.AllocCPUS.astype(int)).where(t.gcds == 0, 0)
    t["billed_cpu_core_h"] = (t.hours * t.AllocTRES.fillna("").map(lambda s: tres(s, "billing"))).where(t.gcds == 0, 0)
    t["energy_kwh"] = t.AllocTRES.fillna("").map(lambda s: tres(s, "energy")) / 3.6e6
    t["group"] = t.JobName.map(lambda n: next((g for p, g in sorted(GROUPS.items(), key=lambda kv: -len(kv[0]))
                                               if n.startswith(p)), "other"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.out, index=False)
    tot = t.groupby("group")[["hours", "gpu_h", "core_h", "billed_cpu_core_h", "energy_kwh"]].sum().round(2)
    tot["jobs"] = t.groupby("group").size()
    tot["states"] = t.groupby("group").State.agg(lambda s: dict(s.value_counts()))
    summary = {"per_group": tot.to_dict(orient="index"),
               "total": tot[["gpu_h", "core_h", "billed_cpu_core_h", "energy_kwh"]].sum().round(2).to_dict()}
    if "Account" in t:  # listings saved with the Account field: totals per billed project
        acc = t.groupby("Account")[["gpu_h", "core_h", "billed_cpu_core_h", "energy_kwh"]].sum().round(2)
        summary["per_account"] = acc.to_dict(orient="index")
        print(acc.to_string())
    a.out.with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    print(tot.to_string())
    print("total", tot[["gpu_h", "core_h", "billed_cpu_core_h", "energy_kwh"]].sum().round(2).to_dict())


if __name__ == "__main__":
    main()
