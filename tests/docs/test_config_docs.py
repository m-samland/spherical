"""Every configuration field carries a ``#:`` doc-comment, which the docs render."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_ext"))

import config_docs  # noqa: E402

from spherical.pipeline import pipeline_config  # noqa: E402

ROOTS = [pipeline_config.IFSReductionConfig, pipeline_config.IRDISReductionConfig]


def test_every_config_field_is_documented():
    missing = config_docs.undocumented_fields(pipeline_config, ROOTS)
    assert not missing, f"Add a '#:' comment above these fields in pipeline_config.py: {missing}"


def test_config_classes_reaches_all_subconfigs():
    names = {cls.__name__ for cls in config_docs.config_classes(pipeline_config.IRDISReductionConfig)}
    assert {"IRDISReductionConfig", "PreprocConfig", "IRDISCalibrationConfig", "IRDISPreprocessConfig"} <= names


def test_comment_must_touch_field():
    source = (
        "class C:\n"
        "    #: documented\n"
        "    a: int = 1\n"
        "    #: detached\n"
        "\n"
        "    b: int = 2\n"
        "    # plain comment\n"
        "    c: int = 3\n"
    )
    assert config_docs.field_doc_comments(source, "C") == {"a": "documented"}


def test_multiline_comment_keeps_paragraphs():
    source = "class C:\n    #: First line\n    #: continues.\n    #:\n    #: New paragraph.\n    a: int = 1\n"
    assert config_docs.field_doc_comments(source, "C")["a"] == "First line\ncontinues.\n\nNew paragraph."


def test_new_subconfig_field_without_doc_is_reported(tmp_path, monkeypatch):
    module_file = tmp_path / "fake_config.py"
    module_file.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass, field\n\n"
        "@dataclass\n"
        "class Sub:\n"
        "    #: documented\n"
        "    a: int = 1\n"
        "    b: int = 2\n\n"
        "@dataclass\n"
        "class Root:\n"
        "    #: the sub-config\n"
        "    sub: Sub = field(default_factory=Sub)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import fake_config

    assert config_docs.undocumented_fields(fake_config, [fake_config.Root]) == {"Sub": ["b"]}


def test_base_path_default_renders_literal():
    rst = config_docs.render_config_rst(pipeline_config.IFSReductionConfig, common=[])
    assert "``~/data/sphere``" in rst
    assert str(Path.home()) not in rst


def test_unknown_common_field_raises():
    with pytest.raises(ValueError, match="directories.base_pth"):
        config_docs.render_config_rst(pipeline_config.IFSReductionConfig, common=["directories.base_pth"])


def test_applies_to_reads_leading_marker():
    assert config_docs.applies_to("IFS only. Something.") == "IFS"
    assert config_docs.applies_to("IRDIS only. Something.") == "IRDIS"
    assert config_docs.applies_to("Something for both.") == "Both"


def test_render_lists_each_subconfig_once_with_its_path():
    rst = config_docs.render_config_rst(pipeline_config.IFSReductionConfig, common=[])
    assert rst.count(".. rubric:: ``config.extraction``") == 1
    assert "``config.directories.base_path``" not in rst  # rows show the bare field name


def test_class_level_marker_applies_to_its_fields():
    rst = config_docs.render_config_rst(pipeline_config.IFSReductionConfig, common=[])
    extraction = rst.split(".. rubric:: ``config.extraction``")[1].split(".. rubric::")[0]
    assert "- Both" not in extraction and "- IFS" in extraction


def test_irdis_subconfigs_are_marked_irdis():
    rst = config_docs.render_config_rst(pipeline_config.IRDISReductionConfig, common=[])
    for attribute in ("calibration", "irdis_preprocessing"):
        section = rst.split(f".. rubric:: ``config.{attribute}``")[1].split(".. rubric::")[0]
        assert "- Both" not in section and "- IRDIS" in section, attribute
