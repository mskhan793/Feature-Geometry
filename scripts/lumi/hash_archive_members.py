"""Stream a .tar.gz once: write ``member<TAB>sha1`` of every image member (file bytes) to --content, and extract the
members matching --extract (small split / annotation files) into --extract-to. Read-only on the archive.

Usage: hash_archive_members.py ARCHIVE --content FILE [--image-suffix .jpg] [--extract REGEX] [--extract-to DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("--content", required=True, type=Path)
    ap.add_argument("--image-suffix", default=".jpg")
    ap.add_argument("--extract", default=None)
    ap.add_argument("--extract-to", type=Path)
    a = ap.parse_args()
    pattern = re.compile(a.extract) if a.extract else None
    n_img = n_ext = 0
    with tarfile.open(a.archive, "r|gz") as tar, open(a.content, "w") as out:
        for m in tar:
            if not m.isfile():
                continue
            if m.name.lower().endswith(a.image_suffix):
                out.write(f"{m.name}\t{hashlib.sha1(tar.extractfile(m).read()).hexdigest()}\n")
                n_img += 1
            elif pattern and pattern.search(m.name):
                target = a.extract_to / m.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(m).read())
                n_ext += 1
    print(f"{a.archive.name}: {n_img} images hashed, {n_ext} files extracted")


if __name__ == "__main__":
    main()
