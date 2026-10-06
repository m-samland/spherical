"""Record where and with what a tutorial was run, and stamp it into the notebook.

A reduction tutorial's run script calls `write_run_provenance` before it starts. The
`docs-tutorials` pixi task then executes and cleans the notebook and runs this file,
which stores the run's provenance (or, for a notebook without a run, the current
environment's) in the notebook metadata. A kernel cell cannot write notebook metadata,
so this is a separate step. The docs render it with the `tutorial-stamp` directive.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "docs" / "tutorials"))
from tutorial_helpers import tutorial_dir  # noqa: E402

METADATA_KEY = "spherical"
RUN_PROVENANCE = "run_provenance.json"
WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
PROVENANCE_KEYS = ("spherical", "charis", "trap", "date", "cpu", "cores", "ram_gb", "os", "ncpu")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return out.stdout.strip()


def _git_describe() -> str | None:
    """`git describe` of the checkout; an editable install's metadata version is stale.

    `-dirty` is added for uncommitted changes to tracked files other than the tutorial
    notebooks, which rendering itself rewrites.
    """
    described = _git("describe", "--tags")
    if not described:
        return None
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", ".",
                   ":(exclude)docs/tutorials/*.ipynb")
    return f"{described}-dirty" if changed else described


def _cpu_model() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _usable_cores() -> int:
    """Cores this process may use; Slurm and cgroups restrict it below os.cpu_count()."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def collect_provenance(ncpu: int | None, today: datetime.date) -> dict:
    """Versions and machine description. No hostname and no user name."""
    return {
        "spherical": _git_describe() or _package_version("spherical"),
        "charis": _package_version("charis"),
        "trap": _package_version("trap-hci"),
        "date": today.isoformat(),
        "cpu": _cpu_model(),
        "cores": _usable_cores(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "ncpu": ncpu,
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def write_run_provenance(path: Path, ncpu: int, extra: dict | None = None) -> dict:
    """Write a run's provenance, or record a resume of the run already described there.

    A resumed run keeps the original description and appends to `resumes`. Resuming with
    a different spherical version raises, since the products would then come from two
    versions of the code.
    """
    today = datetime.date.today()
    current = collect_provenance(ncpu, today) | (extra or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("spherical") != current["spherical"]:
            raise ValueError(f"{path} describes a run with spherical {data.get('spherical')}; "
                             f"resuming it with {current['spherical']} would mix versions")
        data.setdefault("resumes", []).append({"date": current["date"], "spherical": current["spherical"]})
    else:
        data = current
    _write_json(path, data)
    return data


def stamp(path: Path, run: dict, rendered: datetime.date) -> None:
    """Store the run's provenance and the render date in the notebook metadata.

    Also drops widget outputs: `tqdm.notebook` ignores TQDM_DISABLE, so progress bars
    from library code arrive as widgets that render as nothing in the docs.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = [o for o in cell["outputs"] if WIDGET_MIME not in o.get("data", {})]
    nb.setdefault("metadata", {}).pop("widgets", None)
    nb["metadata"][METADATA_KEY] = {"run": run, "rendered": rendered.isoformat()}
    _write_json(path, nb)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args(argv)
    today = datetime.date.today()
    run_file = tutorial_dir() / args.notebook.stem / RUN_PROVENANCE
    if run_file.exists():
        run = json.loads(run_file.read_text(encoding="utf-8"))
    else:
        run = collect_provenance(None, today)
    stamp(args.notebook, run, today)
    print(f"Stamped {args.notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
