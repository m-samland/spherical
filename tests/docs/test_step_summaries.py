"""Every pipeline step has a one-sentence summary for the steps reference page."""

import pytest

from spherical.pipeline import step_registry


@pytest.mark.parametrize(
    ("registry", "summaries"),
    [
        (step_registry.STEP_REGISTRY, "IFS_STEP_SUMMARIES"),
        (step_registry.IRDIS_STEP_REGISTRY, "IRDIS_STEP_SUMMARIES"),
    ],
)
def test_summaries_match_registry(registry, summaries):
    summary = getattr(step_registry, summaries)
    assert set(summary) == set(registry), (
        f"{summaries} keys differ from the registry: "
        f"missing {sorted(set(registry) - set(summary))}, extra {sorted(set(summary) - set(registry))}"
    )
    assert all(text.strip() for text in summary.values())
