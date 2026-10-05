"""Hand-written docs pages and the README follow the writing rules (docs/contributing/writing-docs.md).

Checked: no em-dashes, none of the listed hype words. Code fences, inline code,
URLs, MyST directive options and HTML are ignored. "robust" is allowed (it has a
statistical meaning).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BANNED = [
    "seamless", "seamlessly", "powerful", "effortless", "effortlessly", "comprehensive", "cutting-edge",
    "state-of-the-art", "leverage", "leverages", "unlock", "unlocks", "delve", "simply", "just",
    "easily", "intuitive", "intuitively",
]
_WORD = re.compile(r"(?<![\w-])(" + "|".join(re.escape(word) for word in BANNED) + r")(?![\w-])", re.IGNORECASE)


def pages() -> list[Path]:
    docs = ROOT / "docs"
    found = [
        path for path in docs.rglob("*.md")
        if not {"superpowers", "_build", "generated", "jupyter_execute"} & set(path.relative_to(docs).parts)
        and not path.name.startswith(("PR_release_", "_scratch"))
    ]
    return sorted(found) + [ROOT / "README.md"]


def _prose(text: str) -> str:
    text = re.sub(r"^[ \t]*(`{3,}|~{3,}).*?^[ \t]*\1\s*$", "", text, flags=re.MULTILINE | re.DOTALL)  # fences, also indented
    text = re.sub(r"`[^`\n]+`", "", text)  # inline code
    text = re.sub(r"\]\([^)]*\)|<[^>\n]+>|https?://\S+", "", text)  # link targets, HTML, bare URLs
    text = re.sub(r"^:[\w-]+:.*$", "", text, flags=re.MULTILINE)  # directive options
    return text


def prose_violations(text: str) -> list[str]:
    found = []
    for number, line in enumerate(_prose(text).splitlines(), start=1):
        if "\u2014" in line:
            found.append(f"line {number}: em-dash")
        found += [f"line {number}: '{match.group(1)}'" for match in _WORD.finditer(line)]
    return found


@pytest.mark.parametrize("path", pages(), ids=lambda path: str(path.relative_to(ROOT)))
def test_page_follows_prose_rules(path):
    assert prose_violations(path.read_text()) == []


def test_prose_violation_is_reported():
    assert prose_violations("You can simply run it \u2014 done.") == ["line 1: em-dash", "line 1: 'simply'"]


def test_code_and_urls_are_ignored():
    text = "Run `just build`.\n\n```bash\necho simply\n```\n\nSee [docs](https://x.org/seamless) and <https://x.org/easily>.\n"
    assert prose_violations(text) == []


def test_indented_fence_is_ignored():
    assert prose_violations("1. Example\n\n   ```bash\n   echo just\n   ```\n") == []


def test_robust_is_allowed():
    assert prose_violations("A robust estimator of the noise.") == []
