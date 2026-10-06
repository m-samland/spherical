"""Sphinx directives: reference tables generated from the code, the diagrams, the landing-page strip and the callouts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import config_docs
import pipeline_map
import sequence_strip
import step_diagram
import tutorial_stamp
from docutils import nodes
from docutils.parsers.rst import directives
from docutils.parsers.rst.directives.admonitions import BaseAdmonition
from docutils.statemachine import StringList
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import relative_uri


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


def _stage_hrefs(directive: SphinxDirective) -> dict[str, str]:
    """Resolve one docname option per strip stage to a relative href, failing on unknown documents."""
    hrefs = {}
    for key, _ in sequence_strip.STAGES:
        docname = directive.options.get(key)
        if docname is None:
            raise directive.error(f"{directive.name} needs the :{key}: option")
        if docname not in directive.env.found_docs:
            raise directive.error(f"{directive.name} :{key}: names an unknown document {docname!r}")
        # Re-read this page when a target is renamed, so incremental builds catch it too.
        directive.env.note_dependency(str(directive.env.doc2path(docname)))
        # Computed here rather than through the builder: env.app is deprecated in Sphinx 9.
        suffix = directive.config.html_file_suffix or ".html"
        hrefs[key] = relative_uri(directive.env.docname + suffix, docname + suffix)
    return hrefs


class SequenceStrip(SphinxDirective):
    """Landing-page strip; each option names the document its stage links to."""

    option_spec = {key: directives.unchanged_required for key, _ in sequence_strip.STAGES}

    def run(self):
        return [nodes.raw("", sequence_strip.render_sequence_strip(_stage_hrefs(self)), format="html")]


class PipelineMap(SphinxDirective):
    """Big-picture map; same stage options as the strip."""

    option_spec = {key: directives.unchanged_required for key, _ in sequence_strip.STAGES}

    def run(self):
        return [nodes.raw("", pipeline_map.render_pipeline_map(_stage_hrefs(self)), format="html")]


class StepDiagram(SphinxDirective):
    """Anatomy step diagram, generated from both step registries."""

    def run(self):
        from spherical.pipeline import pipeline_config, step_registry

        try:
            rows = step_diagram.build_phase_rows(step_registry.STEP_ORDER, step_registry.IRDIS_STEP_ORDER)
        except ValueError as error:
            raise self.error(str(error)) from error
        defaults = pipeline_config.PipelineStepsConfig()
        steps = step_registry.STEP_ORDER + step_registry.IRDIS_STEP_ORDER
        optional = {name for name in steps if not getattr(defaults, name)}
        self.env.note_dependency(step_registry.__file__)
        return [nodes.raw("", step_diagram.render_step_diagram(rows, optional), format="html")]


def _callout(title: str, css_class: str) -> type[BaseAdmonition]:
    """An admonition with a fixed title and class, so authors cannot misspell either."""

    class Callout(BaseAdmonition):
        node_class = nodes.admonition
        required_arguments = 0
        optional_arguments = 0
        has_content = True

        def run(self):
            self.arguments = [title]
            self.options["class"] = [css_class]
            return super().run()

    Callout.__name__ = "".join(part.capitalize() for part in css_class.split("-"))
    return Callout


CALLOUTS = {
    "expected-result": "Expected result",
    "instrument-background": "Instrument background",
    "common-mistake": "Common mistake",
}


class TutorialStamp(SphinxDirective):
    """The provenance line of a tutorial notebook, read from the notebook's own metadata."""

    def run(self):
        source = Path(self.env.doc2path(self.env.docname))
        if source.suffix != ".ipynb":
            raise self.error("tutorial-stamp only works in a notebook")
        metadata = json.loads(source.read_text(encoding="utf-8")).get("metadata", {})
        try:
            text = tutorial_stamp.format_stamp(metadata.get(tutorial_stamp.METADATA_KEY))
        except ValueError as error:
            raise self.error(str(error)) from error
        return [nodes.paragraph(text, text, classes=["tutorial-stamp"])]


def setup(app):
    app.add_directive("config-table", ConfigTable)
    app.add_directive("step-table", StepTable)
    app.add_directive("sequence-strip", SequenceStrip)
    app.add_directive("pipeline-map", PipelineMap)
    app.add_directive("step-diagram", StepDiagram)
    app.add_directive("tutorial-stamp", TutorialStamp)
    for name, title in CALLOUTS.items():
        app.add_directive(name, _callout(title, name))
    return {"parallel_read_safe": True, "parallel_write_safe": True}
