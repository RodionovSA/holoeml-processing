"""Convert the Thorlabs protected-silver (-P01) reflectance workbook to .npy files.

Writes one array per angle-of-incidence sheet to ``processing/refdata/``:
``thorlabs_p01_silver_aoi{12,45}.npy``, shape ``(P, 4)``, columns
``wavelength [nm], R_p, R_s, R_unpol`` with reflectance as fractions (0-1),
sorted by wavelength, duplicates removed.

    uv run --with openpyxl python scripts/extract_mirror_reference.py thorlabs_protected_silver_coating.xlsx
"""

import re
import sys
from pathlib import Path

import numpy as np
import openpyxl

OUT = Path(__file__).resolve().parents[1] / "processing" / "refdata"


def main(xlsx: str) -> None:
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for ws in wb.worksheets:
        aoi = int(re.match(r"\s*(\d+)", ws.title).group(1))
        rows = [r[2:6] for r in ws.iter_rows(values_only=True)
                if all(isinstance(v, (int, float)) for v in r[2:6])]
        a = np.array(rows, dtype=float)
        a[:, 0] *= 1000.0        # um -> nm
        a[:, 1:] /= 100.0        # % -> fraction
        a = a[np.argsort(a[:, 0], kind="stable")]
        a = a[np.concatenate([[True], np.diff(a[:, 0]) > 0])]
        path = OUT / f"thorlabs_p01_silver_aoi{aoi}.npy"
        np.save(path, a)
        print(f"{ws.title}: {len(a)} points, {a[0, 0]:.0f}-{a[-1, 0]:.0f} nm -> {path}")


if __name__ == "__main__":
    main(sys.argv[1])
