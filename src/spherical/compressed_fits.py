"""Decompress the ``.fits.Z`` files delivered by the ESO archive.

The archive compresses newer files with gzip and older ones with Unix ``compress`` (LZW), and names both
``.fits.Z``. LZW files are read with the pure-Python ``uncompresspy`` package, so no system ``uncompress`` is
needed.
"""

from __future__ import annotations

import gzip
import shutil
import warnings
from pathlib import Path

from uncompresspy import LZWFile

LZW_MAGIC = b"\x1f\x9d"


def decompress_fits_z(source: Path, target: Path) -> None:
    """Write the decompressed content of ``source`` to ``target``.

    The output goes to ``<target>.part`` first and is renamed when complete, so a failure never leaves ``target``.

    Args:
        source: A gzip or LZW compressed file.
        target: Path of the decompressed file.

    Raises:
        EOFError: The compressed stream ends early.
        ValueError: The LZW stream is corrupt.
        gzip.BadGzipFile: The file is neither LZW nor gzip.
    """
    partial = target.with_name(target.name + ".part")
    try:
        with open(source, "rb") as fin, open(partial, "wb") as fout:
            is_lzw = fin.read(2) == LZW_MAGIC
            fin.seek(0)
            if is_lzw:
                with warnings.catch_warnings():
                    # uncompresspy only warns when the stream stops inside a code, which is a truncated download.
                    warnings.filterwarnings("error", message=".*truncated", category=RuntimeWarning)
                    try:
                        with LZWFile(fin, save_checkpoints=False) as reader:
                            shutil.copyfileobj(reader, fout)
                    except RuntimeWarning as exc:
                        raise EOFError(str(exc)) from None
            else:
                with gzip.open(fin) as reader:
                    shutil.copyfileobj(reader, fout)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
