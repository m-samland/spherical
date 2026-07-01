from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_DOI = "10.5281/zenodo.15147730"
DEFAULT_TIMEOUT = 120
MANIFEST_NAME = ".zenodo_manifest.json"


def _http_get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "spherical-zenodo-sync/1.0",
        },
    )
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def _resolve_zenodo_record_id(doi_or_record: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Accepts:
      - DOI: 10.5281/zenodo.15147730
      - DOI URL: https://doi.org/10.5281/zenodo.15147730
      - record id: 15459438
      - Zenodo record URL: https://zenodo.org/records/15459438
    Returns the concrete Zenodo record id as a string.
    """
    doi_or_record = doi_or_record.strip()

    if doi_or_record.isdigit():
        return doi_or_record

    record_match = re.search(r"zenodo\.org/records/(\d+)", doi_or_record)
    if record_match:
        return record_match.group(1)

    doi = doi_or_record
    if doi.startswith("https://doi.org/"):
        doi = doi.removeprefix("https://doi.org/")
    elif doi.startswith("http://doi.org/"):
        doi = doi.removeprefix("http://doi.org/")

    # Resolve DOI redirect to latest Zenodo landing page.
    doi_url = f"https://doi.org/{doi}"
    req = Request(
        doi_url,
        headers={"User-Agent": "spherical-zenodo-sync/1.0"},
    )
    with urlopen(req, timeout=timeout) as response:
        final_url = response.geturl()

    record_match = re.search(r"zenodo\.org/records/(\d+)", final_url)
    if not record_match:
        raise RuntimeError(f"Could not resolve DOI '{doi}' to a Zenodo record URL. Final URL was: {final_url}")
    return record_match.group(1)


def _load_record(record_id: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    return _http_get_json(f"https://zenodo.org/api/records/{record_id}", timeout=timeout)


def _extract_version(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata", {})
    version = metadata.get("version")
    if version:
        return str(version)

    # Fall back to publication date if version not set.
    for key in ("publication_date", "created", "updated", "modified"):
        value = record.get(key) or metadata.get(key)
        if value:
            return str(value)
    return None


def _normalize_file_entry(file_entry: dict[str, Any]) -> dict[str, Any]:
    """
    Zenodo file metadata has changed over time. Handle common shapes defensively.
    """
    key = file_entry.get("key") or file_entry.get("filename") or file_entry.get("name")
    if not key:
        raise RuntimeError(f"Could not determine file name from entry: {file_entry}")

    links = file_entry.get("links", {})
    download_url = links.get("self") or links.get("content") or file_entry.get("self") or file_entry.get("download")
    if not download_url:
        # Common fallback for newer Zenodo API payloads.
        record_id = file_entry.get("record_id")
        if record_id:
            download_url = f"https://zenodo.org/records/{record_id}/files/{key}"

    checksum = file_entry.get("checksum")
    size = file_entry.get("size")

    return {
        "key": key,
        "download_url": download_url,
        "checksum": checksum,
        "size": size,
    }


def _iter_record_files(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_files = record.get("files", [])
    normalized = [_normalize_file_entry(entry) for entry in raw_files]
    return normalized


def _wanted_filenames(instrument: str, include_polarimetry: bool) -> set[str]:
    wanted: set[str] = set()

    if instrument in {"ifs", "all"}:
        wanted.add("table_of_files_ifs.csv")
        wanted.add("table_of_observations_ifs.fits")

    if instrument in {"irdis", "all"}:
        wanted.add("table_of_files_irdis.csv")
        wanted.add("table_of_observations_irdis.fits")
        if include_polarimetry:
            wanted.add("table_of_observations_irdis_polarimetry.fits")

    return wanted


def _md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()


def _checksum_matches(path: Path, zenodo_checksum: str | None) -> bool:
    if not path.exists() or not zenodo_checksum:
        return False
    if not zenodo_checksum.startswith("md5:"):
        return False
    expected = zenodo_checksum.split(":", 1)[1]
    return _md5sum(path) == expected


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _download_file(url: str, destination: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        req = Request(url, headers={"User-Agent": "spherical-zenodo-sync/1.0"})
        with urlopen(req, timeout=timeout) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _print_status(action: str, name: str, detail: str | None = None) -> None:
    prefix = f"{action}:"
    if detail:
        print(f"{prefix:<12} {name:<50} ({detail})")
    else:
        print(f"{prefix:<12} {name}")


def sync_tables(
    doi_or_record: str,
    dest: Path,
    instrument: str,
    include_polarimetry: bool,
    timeout: int,
    force: bool,
) -> int:
    record_id = _resolve_zenodo_record_id(doi_or_record, timeout=timeout)
    record = _load_record(record_id, timeout=timeout)
    version = _extract_version(record)

    files = _iter_record_files(record)
    wanted = _wanted_filenames(instrument=instrument, include_polarimetry=include_polarimetry)
    selected = [f for f in files if f["key"] in wanted]

    missing_from_record = sorted(wanted - {f["key"] for f in selected})
    if missing_from_record:
        raise RuntimeError("The Zenodo record does not contain the expected files: " + ", ".join(missing_from_record))

    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)

    old_record_id = manifest.get("record_id")
    old_version = manifest.get("version")
    record_changed = (old_record_id != record_id) or (old_version != version)

    print(f"Resolved Zenodo record: {record_id}")
    if version:
        print(f"Record version marker: {version}")
    if old_record_id or old_version:
        print(f"Previous local record: {old_record_id or '-'} / {old_version or '-'}")
    print(f"Destination: {dest}")

    downloads = 0
    skipped = 0
    file_manifest: dict[str, Any] = manifest.get("files", {})

    for file_info in selected:
        name = file_info["key"]
        path = dest / name
        checksum = file_info.get("checksum")
        size = file_info.get("size")
        url = file_info.get("download_url")

        if not url:
            raise RuntimeError(f"No download URL found in Zenodo metadata for file '{name}'")

        local_exists = path.exists()
        local_matches_zenodo = _checksum_matches(path, checksum)

        if local_exists and not local_matches_zenodo and not force:
            _print_status(
                "Skipped",
                name,
                "local file differs from Zenodo; use --force to overwrite",
            )
            skipped += 1
            continue

        needs_download = force or record_changed or not local_exists

        if needs_download:
            _print_status("Downloading", name)
            _download_file(url, path, timeout=timeout)

            if checksum and checksum.startswith("md5:"):
                local_md5 = _md5sum(path)
                expected_md5 = checksum.split(":", 1)[1]
                if local_md5 != expected_md5:
                    raise RuntimeError(f"Checksum mismatch for {name}: expected {expected_md5}, got {local_md5}")

            file_manifest[name] = {
                "record_id": record_id,
                "checksum": checksum,
                "size": size,
                "downloaded_at_unix": int(time.time()),
            }
            downloads += 1
        else:
            _print_status("Up to date", name)
            skipped += 1

    manifest.update(
        {
            "doi_or_record": doi_or_record,
            "record_id": record_id,
            "version": version,
            "files": file_manifest,
        }
    )
    _save_manifest(manifest_path, manifest)

    print()
    print(f"Done. Downloaded: {downloads}, skipped: {skipped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync the latest spherical database tables from Zenodo.")
    parser.add_argument(
        "--doi",
        default=DEFAULT_DOI,
        help=(f"Zenodo DOI, DOI URL, record URL, or numeric record id. Default: {DEFAULT_DOI}"),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Directory where the Zenodo tables should be stored.",
    )
    parser.add_argument(
        "--instrument",
        choices=("ifs", "irdis", "all"),
        default="all",
        help="Which instrument tables to sync.",
    )
    parser.add_argument(
        "--include-polarimetry",
        action="store_true",
        help="Also download table_of_observations_irdis_polarimetry.fits.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds. Default: {DEFAULT_TIMEOUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the local files appear current.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        return sync_tables(
            doi_or_record=args.doi,
            dest=args.dest,
            instrument=args.instrument,
            include_polarimetry=args.include_polarimetry,
            timeout=args.timeout,
            force=args.force,
        )
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
