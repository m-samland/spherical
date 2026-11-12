"""Stream gzip and legacy Unix-compress FITS files from the ESO archive."""

import gzip
import shutil
from pathlib import Path
from uuid import uuid4

from astropy.io import fits

FITS_BLOCK = 2880


def is_truncated_fits(path: Path) -> bool:
    """Check the block length and the end offset of the last FITS HDU."""
    size = path.stat().st_size
    if size == 0 or size % FITS_BLOCK:
        return True
    try:
        with fits.open(path) as hdul:
            info = hdul[-1].fileinfo()
            return info is not None and info["datLoc"] + info["datSpan"] > size
    except Exception:
        # A complete block sequence that cannot be opened is reported by its reader.
        return False


def decompress_fits_z(source: Path, output: Path) -> None:
    """Stream a compressed FITS file and publish the output only on success."""
    partial = output.with_name(f".{output.name}.{uuid4().hex}.part")
    try:
        with source.open("rb") as compressed, partial.open("xb") as decoded:
            is_lzw = compressed.read(2) == b"\x1f\x9d"
            compressed.seek(0)
            if is_lzw:
                from uncompresspy import LZWFile

                try:
                    with LZWFile(compressed, save_checkpoints=False) as reader:
                        shutil.copyfileobj(reader, decoded)
                except ValueError as exc:
                    raise EOFError(f"Invalid LZW stream: {exc}") from exc
            else:
                with gzip.GzipFile(fileobj=compressed, mode="rb") as reader:
                    shutil.copyfileobj(reader, decoded)
        # An LZW stream can end early without raising. Check before publishing it.
        if is_lzw and is_truncated_fits(partial):
            raise EOFError("Truncated FITS after LZW decompression")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
