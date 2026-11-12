"""Transcribe the FIN-Benthic2 release's 'Table of classes.pdf' (Table I: Taxa, Species, Genus, Family, Order,
#specimens, #images) into a CSV, verbatim (read-only; task.md f54f534).

Names are copied as printed, including the release's typos (e.g. 'Perlodiae', 'Taenioptegyridae', 'Rhycophila',
'Sphaeridae'); nothing is corrected or looked up elsewhere. '-' means no entry at that rank. The leading class numbers
are kept in separate columns (they are not unique in the release: taxa number 36 and family number 8 occur twice), so
grouping must use the names. Requires `pdftotext` (poppler).

Usage: make_finbenthic2_taxonomy.py --pdf "Table of classes.pdf" --out finbenthic2_table_of_classes.csv
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd

RANKS = ["taxa", "species", "genus", "family", "order"]
ROW = re.compile(r"^\s*(\d+-\S.*)$")


def split_entry(entry: str) -> tuple[int | None, str | None]:
    if entry == "-":
        return None, None
    number, name = entry.split("-", 1)
    return int(number), name.strip()


def parse(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if not ROW.match(line):
            continue
        cells = re.split(r"\s{2,}", line.strip())
        assert len(cells) == 7, cells
        row = {}
        for rank, entry in zip(RANKS, cells[:5]):
            row[f"{rank}_number"], row[rank] = split_entry(entry)
        row["specimens"], row["images"] = int(cells[5]), int(cells[6])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    text = subprocess.run(["pdftotext", "-layout", str(a.pdf), "-"], check=True, capture_output=True, text=True).stdout
    table = parse(text)
    assert len(table) == 39 and table.taxa.is_unique, (len(table), table.taxa.duplicated().sum())
    table["source_pdf_sha256"] = hashlib.sha256(a.pdf.read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
