"""Write the file manifest of the AquaMonitor release on Hugging Face (``mikkoim/aquamonitor``) at one pinned revision.

One TSV row per file: path, size, and the official checksum: sha256 for large-file (LFS) entries, the git blob sha1
for small files. ``thumbnail/`` (resized preview images) is left out. The download job verifies every file
against this manifest.

Usage: aquamonitor_release_manifest.py --revision SHA --out MANIFEST.tsv
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/datasets/mikkoim/aquamonitor/tree/{rev}?recursive=true"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    with urllib.request.urlopen(API.format(rev=a.revision)) as r:
        entries = json.load(r)
    rows = []
    for e in sorted(entries, key=lambda e: e["path"]):
        if e["type"] != "file" or e["path"].startswith("thumbnail/"):
            continue
        if "lfs" in e:
            rows.append((e["path"], e["lfs"]["size"], "sha256", e["lfs"]["oid"]))
        else:
            rows.append((e["path"], e["size"], "git_sha1", e["oid"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("path\tsize\talgorithm\tchecksum\n" + "".join(f"{p}\t{s}\t{k}\t{c}\n" for p, s, k, c in rows))
    print(f"{len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB -> {a.out}")


if __name__ == "__main__":
    main()
