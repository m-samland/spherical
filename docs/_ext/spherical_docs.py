"""Sphinx directives that render spherical's reference tables from the code."""

from __future__ import annotations

import importlib

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective

import config_docs


def _import(dotted: str):
    module_name, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


def _parse(directive: SphinxDirective, rst: str) -> list[nodes.Node]:
    container = nodes.container()
    directive.state.nested_parse(StringList(rst.splitlines(), source=directive.get_source_info()[0]), 0, container)
    # In a MyST fence nested_parse reads the RST as Markdown and silently emits text, not tables.
    if not any(True for child in container.children for _ in child.findall(nodes.table)):
        raise directive.error(f"{directive.name} must be written inside an {{eval-rst}} block")
    return container.children


class ConfigTable(SphinxDirective):
    """Field tables for a top-level reduction config and all its sub-configs."""

    required_arguments = 1
    option_spec = {"common": directives.unchanged}

    def run(self):
        try:
            root = _import(self.arguments[0])
        except (ImportError, AttributeError) as error:
            raise self.error(f"cannot import {self.arguments[0]}: {error}") from error
        common = [name.strip() for name in self.options.get("common", "").split(",") if name.strip()]
        try:
            rst = config_docs.render_config_rst(root, common)
        except ValueError as error:
            raise self.error(str(error)) from error
        self.env.note_dependency(importlib.import_module(root.__module__).__file__)
        return _parse(self, rst)


class StepTable(SphinxDirective):
    """Ordered step table for one instrument, from the step registry."""

    required_arguments = 1

    def run(self):
        from spherical.pipeline import pipeline_config, step_registry

        instrument = self.arguments[0].lower()
        if instrument not in ("ifs", "irdis"):
            raise self.error("step-table takes 'ifs' or 'irdis'")
        registry = step_registry.STEP_REGISTRY if instrument == "ifs" else step_registry.IRDIS_STEP_REGISTRY
        summaries = step_registry.IFS_STEP_SUMMARIES if instrument == "ifs" else step_registry.IRDIS_STEP_SUMMARIES
        defaults = pipeline_config.PipelineStepsConfig()
        rows = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 4 30 10 56",
            "   :class: step-table",
            "",
            "   * - #",
            "     - Switch",
            "     - On by default",
            "     - What it does",
        ]
        for number, (name, spec) in enumerate(registry.items(), start=1):
            notes = []
            if spec.is_trap:
                notes.append("TRAP post-processing.")
            if spec.leaf:
                notes.append("Nothing downstream reads its output.")
            if spec.internal_guard:
                notes.append("Skips work already done on its own.")
            rows += [
                f"   * - {number}",
                f"     - ``config.steps.{name}``",
                f"     - {'yes' if getattr(defaults, name) else 'no'}",
                f"     - {' '.join([summaries[name], *notes])}",
            ]
        self.env.note_dependency(step_registry.__file__)
        return _parse(self, "\n".join(rows))


def setup(app):
    app.add_directive("config-table", ConfigTable)
    app.add_directive("step-table", StepTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
