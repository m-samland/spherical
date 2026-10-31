import numpy as np
import pandas as pd

from spherical.pipeline.steps.flux_psf_calibration import build_nd_attenuation

# transmission_nd takes nanometres; wavelengths outside its tables come back NaN.
WAVELENGTHS_NM = np.array([1000.0, 1200.0])


def test_uniform_nd_gives_one_column_per_frame():
    frames = pd.DataFrame({"INS4 FILT2 NAME": ["ND_1.0", "ND_1.0", "ND_1.0"]})
    att = build_nd_attenuation(frames, wavelengths=WAVELENGTHS_NM)
    assert att.shape == (2, 3)
    assert np.all(np.isfinite(att))
    assert np.allclose(att[:, 0], att[:, 1])
    assert np.allclose(att[:, 0], att[:, 2])


def test_mixed_nd_does_not_raise_and_differs_per_frame():
    """Regression: used to raise ValueError('Non-unique ND filters in sequence.')."""
    frames = pd.DataFrame({"INS4 FILT2 NAME": ["ND_2.0", "ND_1.0"]})
    att = build_nd_attenuation(frames, wavelengths=WAVELENGTHS_NM)
    assert att.shape == (2, 2)
    assert np.all(np.isfinite(att))
    # ND_2.0 attenuates more than ND_1.0 at every wavelength.
    assert np.all(att[:, 0] < att[:, 1])
