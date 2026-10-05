"""Every spherical module is listed exactly once in the API reference pages.

The subpackages have no ``__init__.py`` (implicit namespace packages), so
autosummary cannot discover modules recursively and the pages list them by hand.
This test is what keeps that hand-kept list complete.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "spherical"
API_DIR = REPO / "docs" / "reference" / "api"

EXCLUDED_FILENAMES = {"__init__.py", "_version.py"}
# Body of a ``.. autosummary::`` directive inside an ```{eval-rst}``` fence: its indented and blank lines.
AUTOSUMMARY_BLOCK = re.compile(r"^\.\. autosummary::[ \t]*\n((?:[ \t]+\S.*\n|[ \t]*\n)*)", re.MULTILINE)


def package_modules(src: Path = SRC) -> set[str]:
    """Dotted names of all modules under ``src``, excluding tests and packaging files."""
    modules = set()
    for path in src.rglob("*.py"):
        relative = path.relative_to(src)
        if path.name in EXCLUDED_FILENAMES or "tests" in relative.parts or "__pycache__" in relative.parts:
            continue
        modules.add(".".join(("spherical", *relative.with_suffix("").parts)))
    return modules


def autosummary_blocks(api_dir: Path = API_DIR) -> list[tuple[bool, list[str]]]:
    """``(has_toctree, entries)`` for every autosummary block in the API pages."""
    blocks = []
    for page in sorted(api_dir.glob("*.md")):
        for body in AUTOSUMMARY_BLOCK.findall(page.read_text()):
            lines = [line.strip() for line in body.splitlines() if line.strip()]
            has_toctree = any(line.startswith(":toctree:") for line in lines)
            entries = [line for line in lines if not line.startswith(":")]
            blocks.append((has_toctree, entries))
    return blocks


def module_entries(api_dir: Path = API_DIR) -> list[str]:
    return [entry for has_toctree, entries in autosummary_blocks(api_dir) if has_toctree for entry in entries]


def object_entries(api_dir: Path = API_DIR) -> list[str]:
    return [entry for has_toctree, entries in autosummary_blocks(api_dir) if not has_toctree for entry in entries]


def test_every_module_is_listed():
    missing = package_modules() - set(module_entries())
    assert not missing, f"Add these modules to a page in docs/reference/api/: {sorted(missing)}"


def test_no_module_listed_twice():
    entries = module_entries()
    duplicates = sorted({entry for entry in entries if entries.count(entry) > 1})
    assert not duplicates, f"Listed on more than one API page: {duplicates}"


def test_no_stale_entries():
    modules = package_modules()
    stale_modules = sorted(set(module_entries()) - modules)
    stale_objects = sorted(
        entry for entry in object_entries() if not any(entry.startswith(module + ".") for module in modules)
    )
    assert not stale_modules, f"API pages list modules that no longer exist: {stale_modules}"
    assert not stale_objects, f"API pages list objects outside any module: {stale_objects}"


def test_excludes_tests_and_version(tmp_path):
    src = tmp_path / "spherical"
    for relative in ["__init__.py", "_version.py", "a.py", "sub/b.py", "sub/tests/test_b.py"]:
        (src / relative).parent.mkdir(parents=True, exist_ok=True)
        (src / relative).write_text("")
    assert package_modules(src) == {"spherical.a", "spherical.sub.b"}


def test_new_module_would_be_detected(tmp_path):
    src = tmp_path / "spherical"
    (src / "newpkg").mkdir(parents=True)
    (src / "newpkg" / "thing.py").write_text("")
    api_dir = tmp_path / "api"
    api_dir.mkdir()
    (api_dir / "page.md").write_text("```{eval-rst}\n.. autosummary::\n   :toctree: generated\n\n   spherical.other\n```\n")
    assert package_modules(src) - set(module_entries(api_dir)) == {"spherical.newpkg.thing"}
