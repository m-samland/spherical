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


# A fence, also indented: opening marker, info string, body, matching closing marker.
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)\n(.*?)^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
# Directive fences whose body is code or reStructuredText, not prose.
_CODE_DIRECTIVES = {"code", "code-block", "code-cell", "eval-rst", "toctree"}


def _strip_fence(match: re.Match) -> str:
    """Drop code fences; keep the body of prose directives such as callouts and the glossary."""
    info = match.group(2).strip()
    name = info[1:].split("}", 1)[0] if info.startswith("{") else None
    if name is None or name in _CODE_DIRECTIVES:
        return ""
    return _FENCE.sub(_strip_fence, match.group(3))  # nested fences inside, e.g. a code sample in a callout


def _prose(text: str) -> str:
    text = _FENCE.sub(_strip_fence, text)
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


def test_directive_fence_body_is_checked():
    """Callouts and the glossary are prose inside a fence; only code fences are skipped."""
    text = "```{common-mistake}\nYou can simply do it.\n```\n\n```{glossary}\nTerm\n  A seamless thing.\n```\n"
    assert [v.split(": ", 1)[1] for v in prose_violations(text)] == ["'simply'", "'seamless'"]


def test_code_directive_fences_are_ignored():
    text = "```{eval-rst}\n.. just a directive\n```\n\n```{code-block} python\nsimply()\n```\n"
    assert prose_violations(text) == []


def test_robust_is_allowed():
    assert prose_violations("A robust estimator of the noise.") == []
