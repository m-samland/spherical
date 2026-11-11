"""The README is the PyPI long description, so its links and images must be absolute."""

import re
from pathlib import Path

README = Path(__file__).parents[2] / "README.md"


def relative_targets(text: str) -> list[str]:
    targets = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'<img[^>]+src="([^"]+)"', text)
    return [target for target in targets if not target.startswith(("https://", "http://", "mailto:", "#"))]


def test_readme_links_are_absolute():
    assert relative_targets(README.read_text()) == []


def test_relative_link_is_reported():
    assert relative_targets("See [x](docs/x.md) and ![b](assets/b.png).") == ["docs/x.md", "assets/b.png"]


def test_readme_links_the_docs_site():
    assert "https://spherical-hci.readthedocs.io" in README.read_text()
