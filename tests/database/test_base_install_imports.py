"""The database half must work on a base install, without the ``pipeline`` extra.

The test extra installs scipy & co. so the pipeline tests run everywhere, which means no
test environment lacks them and an accidental import would go unnoticed. Walk the imports
statically instead: everything reachable from ``spherical`` and ``spherical.database``,
following ``spherical.*`` modules transitively and including imports inside functions.
"""
from __future__ import annotations

import ast
from pathlib import Path

import spherical

# Import names of the `pipeline` extra in pyproject.toml, which core does not provide.
PIPELINE_ONLY = {"scipy", "skimage", "photutils", "dill", "charis", "trap"}

PACKAGE_ROOT = Path(spherical.__file__).parent


def _module_name(path: Path) -> str:
    parts = path.relative_to(PACKAGE_ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _module_file(module: str) -> Path | None:
    """Source file of a ``spherical.*`` module, or None if the name is not a module."""
    path = PACKAGE_ROOT.joinpath(*module.split(".")[1:])
    for candidate in (path / "__init__.py", path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _imports(path: Path) -> set[str]:
    """Absolute names of every module imported anywhere in ``path``."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level:
                base = package.rsplit(".", node.level - 1)[0]
                source = f"{base}.{source}" if source else base
            names.add(source)
            # `from pkg import name` may import the submodule pkg.name.
            names |= {f"{source}.{alias.name}" for alias in node.names}
    return names


def _walk_base_imports() -> tuple[set[str], dict[str, set[str]]]:
    """Spherical modules reachable from the base surface, and external roots -> importers."""
    todo = [PACKAGE_ROOT / "__init__.py", *(PACKAGE_ROOT / "database").glob("*.py")]
    seen: set[str] = set()
    external: dict[str, set[str]] = {}
    while todo:
        path = todo.pop()
        module = _module_name(path)
        if module in seen:
            continue
        seen.add(module)
        for name in _imports(path):
            root = name.split(".")[0]
            if root != "spherical":
                external.setdefault(root, set()).add(module)
            elif (target := _module_file(name)) is not None:
                todo.append(target)
    return seen, external


def test_database_never_imports_pipeline_only_packages():
    _, external = _walk_base_imports()
    offending = {root: sorted(external[root]) for root in PIPELINE_ONLY & external.keys()}
    assert not offending, (
        f"the base install surface imports pipeline-only packages: {offending}"
    )


def test_database_never_imports_spherical_pipeline():
    modules, _ = _walk_base_imports()
    reached = sorted(m for m in modules if m.startswith("spherical.pipeline"))
    assert not reached, f"the base install surface reaches {reached}"
