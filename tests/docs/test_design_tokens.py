"""Every colour pair the site uses meets WCAG AA, in light and dark mode.

Tokens live in docs/_static/custom.css as --spherical-* custom properties inside
html[data-theme="light"] and html[data-theme="dark"] blocks.
"""

import re
from pathlib import Path

import pytest

CSS = Path(__file__).parents[2] / "docs" / "_static" / "custom.css"

# (foreground, background, minimum ratio). 4.5 for text, 3.0 for non-text marks (WCAG 1.4.11).
PAIRS = [
    ("ink", "paper", 4.5),
    ("night", "paper", 4.5),  # headings and links in light mode; checked in dark only as a mark
    ("jupiter-text", "paper", 4.5),
    ("ink", "haze", 4.5),
    ("on-night", "night", 4.5),
    ("primary", "paper", 4.5),  # pydata uses primary as text (active tab, TOC) in both modes
    ("primary-text", "primary", 4.5),
    ("primary-text", "primary-highlight", 4.5),
    ("jupiter", "paper", 3.0),
    ("gold", "paper", 3.0),
    ("green", "paper", 3.0),
    ("amber", "paper", 3.0),
]
# Night is the header colour in both modes; in dark mode headings use ink, so night/paper is not a text pair there.
SKIP = {("dark", "night", "paper")}


def parse_tokens(css: str) -> dict[str, dict[str, str]]:
    tokens = {}
    for mode, body in re.findall(r'html\[data-theme="(light|dark)"\]\s*\{([^}]*)\}', css):
        found = dict(re.findall(r"--spherical-([a-z-]+)\s*:\s*(#[0-9A-Fa-f]{6})\s*;", body))
        tokens.setdefault(mode, {}).update(found)
    return tokens


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def failing_pairs(tokens: dict[str, dict[str, str]]) -> list[str]:
    failures = []
    for mode, values in tokens.items():
        for fg, bg, minimum in PAIRS:
            if (mode, fg, bg) in SKIP:
                continue
            ratio = contrast(values[fg], values[bg])
            if ratio < minimum:
                failures.append(f"{mode}: {fg} on {bg} is {ratio:.2f}, needs {minimum}")
    return failures


def test_both_modes_define_every_token():
    tokens = parse_tokens(CSS.read_text())
    names = {name for pair in PAIRS for name in pair[:2]}
    assert set(tokens) == {"light", "dark"}
    for mode in ("light", "dark"):
        assert names <= set(tokens[mode]), f"{mode} lacks {sorted(names - set(tokens[mode]))}"


def test_token_pairs_meet_wcag_aa():
    tokens = parse_tokens(CSS.read_text())
    assert set(tokens) == {"light", "dark"}, "no token blocks found; an empty set would pass vacuously"
    assert failing_pairs(tokens) == []


def test_low_contrast_pair_is_reported():
    css = 'html[data-theme="light"] { --spherical-ink: #DDDDDD; --spherical-paper: #FFFFFF; }'
    tokens = parse_tokens(css)
    tokens["light"].update({name: "#000000" for pair in PAIRS for name in pair[:2] if name not in ("ink", "paper")})
    tokens["light"]["paper"] = "#FFFFFF"
    assert any("ink on paper" in failure for failure in failing_pairs(tokens))


@pytest.mark.parametrize(("a", "b", "expected"), [("#FFFFFF", "#000000", 21.0), ("#D9702A", "#FFFFFF", 3.33)])
def test_contrast_formula(a, b, expected):
    assert contrast(a, b) == pytest.approx(expected, abs=0.01)
