"""Layout regression tests for the TRAP result mosaics (GitHub issue #146).

Panel titles used to be scaled with the size of the whole figure while each
panel shrinks as more observations are added, so with ~50 data sets the titles
dwarfed the detection maps. These tests render small and large mosaics and
check that the title stays a similar fraction of its panel.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402

from spherical.pipeline.visualize import mosaic  # noqa: E402


def _make_results_tree(base: Path, n_observations: int) -> Table:
    """Write detection maps and candidate tables for n observations."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_observations):
        target, obs_mode, date = f"HD_{1000 + i}", "OBS_H", f"2020-01-{i % 28 + 1:02d}"
        leaf = base / target / obs_mode / date
        fits_path = leaf / mosaic.TEMPLATE_PATTERNS["flat"]
        fits_path.parent.mkdir(parents=True)
        fits.writeto(fits_path, rng.normal(size=(64, 64)).astype("f4"))
        pd.DataFrame({
            "candidate_id": [0, 0],
            "x": [20.0, 20.0],
            "y": [30.0, 30.0],
            "norm_snr_fit_free": [6.0, 6.0],
            "wavelength": [1.0, 1.5],
            "contrast": [1e-5, 2e-5],
            "uncertainty": [1e-6, 1e-6],
        }).to_csv(leaf / mosaic.CANDIDATE_PATTERNS["flat"], index=False)
        rows.append((target.replace("_", " "), obs_mode, date, 60.0, 30.0, 0.8))
    return Table(
        rows=rows,
        names=["MAIN_ID", "FILTER", "NIGHT_START", "TOTAL_EXPTIME_SCI", "ROTATION", "MEAN_FWHM"],
    )


def _title_to_panel_ratio(tmp_path: Path, plot_function, n_observations: int) -> float:
    """Height of the first panel title divided by the height of its panel content."""
    base = tmp_path / f"n{n_observations}"
    table = _make_results_tree(base, n_observations)
    fig = plot_function(base, observation_table=table, dpi=20)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[0]
        images = ax.get_images()
        content = images[0] if images else ax
        return (
            ax.title.get_window_extent(renderer).height
            / content.get_window_extent(renderer).height
        )
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "plot_function",
    [mosaic.plot_detection_mosaic, mosaic.plot_spectrum_mosaic],
    ids=["detection", "spectrum"],
)
def test_title_size_relative_to_panel_does_not_grow_with_observations(tmp_path, plot_function):
    small = _title_to_panel_ratio(tmp_path, plot_function, 4)
    large = _title_to_panel_ratio(tmp_path, plot_function, 50)

    assert large < 1.5 * small, (
        f"title/panel height ratio grew from {small:.2f} (N=4) to {large:.2f} (N=50)"
    )


def test_candidate_labels_stay_legible_in_large_detection_mosaic(tmp_path):
    table = _make_results_tree(tmp_path, 50)
    fig = mosaic.plot_detection_mosaic(tmp_path, observation_table=table, dpi=20)
    try:
        label_sizes = [text.get_fontsize() for ax in fig.axes for text in ax.texts]
    finally:
        plt.close(fig)

    assert label_sizes, "expected SNR labels next to the candidates"
    assert min(label_sizes) >= 8
