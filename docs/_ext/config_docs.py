"""Read and render the ``#:`` doc-comments of spherical's configuration dataclasses.

Stdlib only: the docs guard in ``tests/docs`` imports it without Sphinx, and the
``config-table`` directive in ``spherical_docs`` renders from the same functions,
so the test and the published page read the comments the same way.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import typing


def field_doc_comments(source: str, class_name: str) -> dict[str, str]:
    """Doc-comments of the annotated fields of ``class_name`` in ``source``.

    A doc-comment is a run of ``#:`` lines directly above the field, as Sphinx
    reads them. ``#:`` alone is a paragraph break.
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    cls = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name)
    docs: dict[str, str] = {}
    for node in cls.body:
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        comment: list[str] = []
        index = node.lineno - 2
        while index >= 0 and lines[index].strip().startswith("#:"):
            comment.insert(0, lines[index].strip()[2:].strip())
            index -= 1
        if comment:
            docs[node.target.id] = "\n".join(comment)
    return docs


def _sub_config_type(cls: type, field: dataclasses.Field) -> type | None:
    hint = typing.get_type_hints(cls).get(field.name)
    return hint if dataclasses.is_dataclass(hint) else None


def config_classes(root: type) -> list[type]:
    """``root`` and every dataclass reachable through its field types, each once."""
    seen: list[type] = []

    def visit(cls: type) -> None:
        if cls in seen:
            return
        seen.append(cls)
        for field in dataclasses.fields(cls):
            sub = _sub_config_type(cls, field)
            if sub is not None:
                visit(sub)

    visit(root)
    return seen


def undocumented_fields(module, roots: list[type]) -> dict[str, list[str]]:
    """Class name to the fields that have no doc-comment, over all reachable classes."""
    source = inspect.getsource(module)
    missing: dict[str, list[str]] = {}
    for root in roots:
        for cls in config_classes(root):
            docs = field_doc_comments(source, cls.__name__)
            names = [field.name for field in dataclasses.fields(cls) if field.name not in docs]
            if names:
                missing[cls.__name__] = names
    return missing


#: Defaults that depend on the machine building the docs, shown as written in the source.
LITERAL_DEFAULTS: dict[tuple[str, str], str] = {
    ("DirectoryConfig", "base_path"): "~/data/sphere",
    ("DirectoryConfig", "raw_directory"): 'base_path / "data"',
    ("DirectoryConfig", "reduction_directory"): 'base_path / "reduction"',
}


def applies_to(description: str) -> str:
    """Instrument a field applies to, from its doc-comment's leading marker."""
    if description.startswith("IFS only."):
        return "IFS"
    if description.startswith("IRDIS only."):
        return "IRDIS"
    return "Both"


def _default(cls: type, field: dataclasses.Field) -> str:
    literal = LITERAL_DEFAULTS.get((cls.__name__, field.name))
    if literal is not None:
        return f"``{literal}``"
    if field.default is not dataclasses.MISSING:
        return f"``{field.default!r}``"
    return f"``{field.default_factory()!r}``"


def _cell(text: str) -> list[str]:
    """Indent a possibly multi-paragraph cell for a list-table."""
    lines = text.splitlines() or [""]
    return [lines[0]] + [f"       {line}" if line else "" for line in lines[1:]]


def _table(rows: list[tuple[str, str, str, str, str]]) -> list[str]:
    """Four columns: the type sits under the field name to leave room for the description."""
    out = [".. list-table::", "   :header-rows: 1", "   :widths: 28 16 8 48", "   :class: config-table", ""]
    out += ["   * - Field", "     - Default", "     - For", "     - Description"]
    for name, type_, default, applies, description in rows:
        out += [f"   * - {name}", "", f"       {type_}"]
        for cell in (default, applies, description):
            cell_lines = _cell(cell)
            out.append(f"     - {cell_lines[0]}")
            out.extend(cell_lines[1:])
    out.append("")
    return out


def _resolve_common(root: type, path: str) -> tuple[type, dataclasses.Field]:
    cls = root
    *parents, name = path.split(".")
    for parent in parents:
        field = next((f for f in dataclasses.fields(cls) if f.name == parent), None)
        sub = _sub_config_type(cls, field) if field else None
        if sub is None:
            raise ValueError(f"Unknown config field in :common: {path!r}")
        cls = sub
    field = next((f for f in dataclasses.fields(cls) if f.name == name), None)
    if field is None or _sub_config_type(cls, field) is not None:
        raise ValueError(f"Unknown config field in :common: {path!r}")
    return cls, field


def _parent_field_doc(source: str, attribute_of: dict[type, str], cls: type) -> str:
    """Doc-comment of the composite field that holds ``cls`` (shown under its rubric)."""
    path = attribute_of[cls]
    if path == "config":
        return ""
    *_, attribute = path.split(".")
    for parent, parent_path in attribute_of.items():
        if f"{parent_path}.{attribute}" == path:
            return field_doc_comments(source, parent.__name__).get(attribute, "")
    return ""


def _applies(cls: type, description: str) -> str:
    """A field's own marker, else the marker that opens its class docstring."""
    own = applies_to(description)
    return own if own != "Both" else applies_to(inspect.getdoc(cls) or "")


def render_config_rst(root: type, common: list[str]) -> str:
    """RST for one top-level config: commonly changed fields, then one table per class."""
    source = inspect.getsource(inspect.getmodule(root))
    out: list[str] = []
    if common:
        rows = []
        for path in common:
            cls, field = _resolve_common(root, path)
            doc = field_doc_comments(source, cls.__name__).get(field.name, "")
            rows.append((f"``config.{path}``", f"``{field.type}``", _default(cls, field), _applies(cls, doc), doc))
        out += [".. rubric:: Commonly changed", ""] + _table(rows)

    attribute_of = {root: "config"}
    for cls in config_classes(root):
        for field in dataclasses.fields(cls):
            sub = _sub_config_type(cls, field)
            if sub is not None and sub not in attribute_of:
                attribute_of[sub] = f"{attribute_of[cls]}.{field.name}"
    for cls in config_classes(root):
        docs = field_doc_comments(source, cls.__name__)
        rows = [
            (
                f"``{field.name}``",
                f"``{field.type}``",
                _default(cls, field),
                _applies(cls, docs.get(field.name, "")),
                docs.get(field.name, ""),
            )
            for field in dataclasses.fields(cls)
            if _sub_config_type(cls, field) is None
        ]
        if rows:
            out += [f".. rubric:: ``{attribute_of[cls]}`` ({cls.__name__})", ""]
            intro = _parent_field_doc(source, attribute_of, cls)
            if intro:
                out += [intro, ""]
            out += _table(rows)
    return "\n".join(out)
