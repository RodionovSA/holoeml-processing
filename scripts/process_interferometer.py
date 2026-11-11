"""Process an interferometer scan folder into ``<out>_phi.h5`` and ``<out>_T.h5``.

One wavelength file is loaded at a time: black subtraction, bad-pixel median fill,
phase solve (sample minus reference, aligned to the first wavelength's sign branch),
and T = I_sample / I_ref. Re-running resumes an interrupted run; see
:func:`processing.interferometer.process_scan`.

    uv run python scripts/process_interferometer.py \\
        ".../HoloEML281026/interferometer/SiO2" \\
        --out ".../HoloEML281026/interferometer/HoloEML281026_SiO2_aia"
    # -> HoloEML281026_SiO2_aia_phi.h5, HoloEML281026_SiO2_aia_T.h5

    # different phase settings, T already computed:
    uv run python scripts/process_interferometer.py ".../SiO2" \\
        --out ".../HoloEML281026_SiO2_vp_aia" --config my_phase.yaml --no-T

    # aligned to the layout (orientation, drift, glass plane, structure masks):
    uv run python scripts/process_interferometer.py ".../SiO2" \\
        --out ".../HoloEML281026_SiO2_aia" --gds ".../Phase_calib_261028.GDS"

Write the output to a local disk, not into Dropbox/Google Drive: ~10 GB per sample.
"""

import argparse
from pathlib import Path

from processing.interferometer import process_scan, scan_folder
from processing.phase import PhasePipeline

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="scan folder with *_black_*, *_<wvl>nm_* .npz files")
    ap.add_argument("--out", required=True,
                    help="output prefix: writes <out>_phi.h5 and <out>_T.h5; "
                         "its folder must exist")
    ap.add_argument("--no-T", dest="transmission", action="store_false",
                    help="skip transmission; write only <out>_phi.h5")
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase.yaml"),
                    help="PhasePipeline YAML (default: configs/phase.yaml)")
    ap.add_argument("--overwrite", action="store_true",
                    help="start over instead of resuming existing output files")
    ap.add_argument("--k-offset", type=float, default=10.0,
                    help="hot pixel: |black - median| > k * MAD (default 10)")
    ap.add_argument("--k-std", type=float, default=5.0,
                    help="noisy pixel: black_std > k * median(black_std) (default 5)")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"],
                    help="HDF5 filter for float maps (default gzip level 1)")
    ap.add_argument("--gds", default=None,
                    help="layout GDS: align the output to it (overrides layout.gds and "
                         "enables the layout section of the config)")
    ap.add_argument("--keep-unaligned", action="store_true",
                    help="with a layout, keep the unaligned *_unaligned.h5 files")
    args = ap.parse_args()

    scan = scan_folder(args.folder)
    print(f"{scan.name}: {len(scan.files)} wavelengths "
          f"{scan.wavelengths.min():g}-{scan.wavelengths.max():g} nm in "
          f"{len(scan.blocks)} blocks ({', '.join(b.name for b in scan.blocks)})")
    pipe = PhasePipeline.from_yaml(args.config)
    if args.keep_unaligned:
        pipe.config.layout.keep_unaligned = True
    paths = process_scan(scan, pipe, args.out,
                         transmission=args.transmission, overwrite=args.overwrite, k_offset=args.k_offset, k_std=args.k_std,
                         compression=None if args.compression == "none" else args.compression,
                         gds=args.gds)
    for p in paths:
        if p is not None:
            print("wrote", p)


if __name__ == "__main__":
    main()
