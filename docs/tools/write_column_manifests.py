"""Write the column manifests of the published database tables.

Run at each table release, against the directory the release tables were downloaded
to. The manifests in `docs/_data/columns/` record which columns each table carries and
in which order; the Database columns page renders them, and
`tests/database/test_column_docs.py` fails when a column has no description in
`spherical.database.column_docs`.

    pixi run -e dev python docs/tools/write_column_manifests.py ~/data/sphere/database
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
FITS_TABLES = ("observations", "targets", "other_observations")
MODES = ("ifs", "irdis")
VARIANTS = ("sam", "polarimetry")


def fits_columns(path: Path) -> list[tuple[str, str]]:
    """(name, unit) of every column of the first table extension, read from the header only."""
    header = fits.getheader(path, 1)
    return [(header[f"TTYPE{i}"], header.get(f"TUNIT{i}", "")) for i in range(1, header["TFIELDS"] + 1)]


def csv_columns(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def table_path(database_dir: Path, table_type: str, mode: str) -> Path:
    suffix = "csv" if table_type == "files" else "fits"
    return database_dir / f"table_of_{table_type}_{mode}.{suffix}"


def write_manifests(database_dir: Path, out_dir: Path) -> None:
    missing = [
        str(table_path(database_dir, table_type, mode))
        for table_type in (*FITS_TABLES, "files")
        for mode in MODES
        if not table_path(database_dir, table_type, mode).exists()
    ]
    if missing:
        sys.exit("missing tables:\n  " + "\n  ".join(missing))

    out_dir.mkdir(parents=True, exist_ok=True)
    units: dict[str, str] = {}
    for table_type in FITS_TABLES:
        for mode in MODES:
            columns = fits_columns(table_path(database_dir, table_type, mode))
            units.update({name: unit for name, unit in columns if unit})
            names = [name for name, _ in columns]
            (out_dir / f"table_columns_{table_type}_{mode}.txt").write_text("\n".join(names) + "\n")
            for variant in VARIANTS:
                path = database_dir / f"table_of_{table_type}_{mode}_{variant}.fits"
                if path.exists() and [name for name, _ in fits_columns(path)] != names:
                    print(f"note: {path.name} has different columns from table_of_{table_type}_{mode}.fits")
    for mode in MODES:
        names = csv_columns(table_path(database_dir, "files", mode))
        (out_dir / f"table_columns_files_{mode}.txt").write_text("\n".join(names) + "\n")
    (out_dir / "table_units.txt").write_text("".join(f"{name}\t{unit}\n" for name, unit in sorted(units.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("database_dir", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "_data" / "columns")
    args = parser.parse_args()
    write_manifests(args.database_dir.expanduser(), args.out)


if __name__ == "__main__":
    main()
