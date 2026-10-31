"""The rotation sense of DEROT ANGLE, as TRAP uses it for SPHERE.

spherical never rotates images. TRAP places the companion model in each frame
with ``yx_position_in_cube`` and the angles spherical writes as ``DEROT ANGLE``
(``run_trap.py``). The Conventions page states the resulting rule; these tests
pin it. The end-to-end check against GRAVITY astrometry is
``tests/regression/test_51eri_astrometry_regression.py``.
"""

import numpy as np
import pytest
from scipy import ndimage

makesource = pytest.importorskip("trap.makesource")
parameters = pytest.importorskip("trap.parameters")

FRAME = (101, 101)  # centre pixel (50, 50), arrays indexed (y, x)


def _detector_yx(sky_yx, angles):
    """Detector (y, x) of a source at ``sky_yx`` relative to the star in a North-up, East-left image."""
    return makesource.yx_position_in_cube(FRAME, sky_yx, np.asarray(angles, dtype=float), right_handed=False)


@pytest.mark.parametrize("preset", ["trap_config_for_ifs", "trap_config_for_irdis"])
def test_sphere_presets_are_not_right_handed(preset):
    assert getattr(parameters, preset)().reduction.right_handed is False


def test_sky_position_appears_rotated_clockwise_by_derot_angle():
    """North (+y with the origin at the lower left) moves towards +x as the angle grows."""
    coords = _detector_yx((10.0, 0.0), [0.0, 30.0, 90.0])
    np.testing.assert_allclose(coords[0], [60.0, 50.0], atol=1e-9)
    np.testing.assert_allclose(coords[1], [50.0 + 10.0 * np.cos(np.radians(30.0)), 55.0], atol=1e-9)
    np.testing.assert_allclose(coords[2], [50.0, 60.0], atol=1e-9)


def test_east_rotates_the_same_way():
    """East (-x) moves to +y after 90 degrees: the whole field turns clockwise."""
    np.testing.assert_allclose(_detector_yx((0.0, -10.0), [90.0])[0], [60.0, 50.0], atol=1e-9)


def test_rule_holds_for_arbitrary_positions_and_angles():
    rng = np.random.default_rng(196)
    for _ in range(20):
        y, x = rng.uniform(-40, 40, size=2)
        theta = rng.uniform(-180, 180)
        t = np.radians(theta)
        expected = [50.0 - x * np.sin(t) + y * np.cos(t), 50.0 + x * np.cos(t) + y * np.sin(t)]
        np.testing.assert_allclose(_detector_yx((y, x), [theta])[0], expected, atol=1e-9)


def test_ndimage_rotate_by_minus_derot_angle_puts_north_up():
    """The one-liner on the Conventions page undoes the rotation."""
    theta = 90.0
    y, x = _detector_yx((10.0, 0.0), [theta])[0]
    frame = np.zeros(FRAME)
    frame[int(round(y)), int(round(x))] = 1.0
    derotated = ndimage.rotate(frame, -theta, reshape=False, order=1)
    assert np.unravel_index(derotated.argmax(), derotated.shape) == (60, 50)
