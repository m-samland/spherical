"""spot_trace: the per-frame spot amplitude passed to TRAP follows the star only (#271).

A synthetic sequence with a flat transmission must give a flat trace through a waffle amplitude change,
a DIT change, a frame whose spots fail in every channel and a frame whose spots fail in one channel.
"""
from __future__ import annotations

import numpy as np
import pytest

from spherical.pipeline.steps.spot_to_flux import spot_trace

N_WAVE, N_FRAMES = 6, 40


def _spots(transmission=None, seed=0, noise=0.02):
    """Spot fluxes (n_wave, n_frames, 4) with 2% spot-to-spot scatter, unit waffle amplitude and DIT."""
    rng = np.random.default_rng(seed)
    t = np.ones(N_FRAMES) if transmission is None else np.asarray(transmission, dtype=float)
    level = np.linspace(1.0, 2.0, N_WAVE)[:, None, None]
    return level * t[None, :, None] * (1 + rng.normal(0, noise, (N_WAVE, N_FRAMES, 4)))


def _trace(spots, waffle=None, dit=None):
    waffle = np.full(N_FRAMES, 0.05) if waffle is None else waffle
    dit = np.full(N_FRAMES, 4.0) if dit is None else dit
    return spot_trace(spots, waffle, dit)


def test_flat_sequence_gives_a_flat_trace():
    result = _trace(_spots())
    np.testing.assert_allclose(result.variation, 1.0, atol=0.05)
    assert not result.failed_frames.any()


def test_waffle_amplitude_change_is_removed():
    waffle = np.full(N_FRAMES, 0.05)
    waffle[10:19] = 0.025
    spots = _spots()
    spots[:, 10:19] *= (0.025 / 0.05) ** 2
    np.testing.assert_allclose(_trace(spots, waffle=waffle).variation, 1.0, atol=0.05)


def test_dit_change_is_removed():
    dit = np.full(N_FRAMES, 4.0)
    dit[30:] = 8.0
    spots = _spots()
    spots[:, 30:] *= 2.0
    np.testing.assert_allclose(_trace(spots, dit=dit).variation, 1.0, atol=0.05)


def test_real_transmission_change_is_kept():
    transmission = np.ones(N_FRAMES)
    transmission[20:] = 0.6
    variation = _trace(_spots(transmission)).variation
    np.testing.assert_allclose(variation[:, 25] / variation[:, 5], 0.6, rtol=0.03)


def test_frame_with_spots_failing_in_most_channels_is_nan_and_failed():
    spots = _spots()
    spots[:, 7] = [20.0, 0.7, 0.05, 1.0]
    result = _trace(spots)
    assert np.flatnonzero(result.failed_frames).tolist() == [7]
    assert np.all(np.isnan(result.variation[:, 7]))
    np.testing.assert_allclose(np.delete(result.variation, 7, axis=1), 1.0, atol=0.05)


def test_single_failed_channel_is_filled_from_the_other_channels_of_the_frame():
    transmission = np.ones(N_FRAMES)
    transmission[12] = 0.7
    spots = _spots(transmission)
    spots[0, 12] = [20.0, 0.7, 0.05, 1.0]
    spots[5, 12] = np.nan
    result = _trace(spots)
    assert not result.failed_frames.any()
    assert result.filled[0, 12] and result.filled[5, 12]
    np.testing.assert_allclose(result.variation[[0, 5], 12], np.median(result.variation[1:5, 12]), rtol=1e-12)
    np.testing.assert_allclose(result.variation[:, 12], 0.7, rtol=0.05)


def test_noisy_channel_is_judged_against_its_own_scatter():
    spots = _spots()
    spots[-1] = _spots(seed=1, noise=0.4)[-1]
    result = _trace(spots)
    assert not result.failed_frames.any()
    assert result.filled[-1].mean() < 0.1


def test_spot_flux_is_returned_on_the_common_amplitude_and_dit():
    waffle = np.full(N_FRAMES, 0.05)
    waffle[:5] = 0.1
    spots = _spots()
    spots[:, :5] *= 4.0
    result = _trace(spots, waffle=waffle)
    np.testing.assert_allclose(result.spot_flux[:, :5] / result.spot_flux[:, 5:10].mean(axis=1, keepdims=True), 1.0,
                               atol=0.05)


def test_failed_trace_frames_become_bad_frames_merged_with_the_center_outliers():
    pytest.importorskip("trap")
    from spherical.pipeline.run_trap import drop_failed_trace_frames

    trace = np.ones((2, 6))
    trace[:, 4] = np.nan
    amplitude, bad_frames, failed = drop_failed_trace_frames(trace, [1, 4])
    assert failed == [4]
    assert bad_frames == [1, 4]
    assert np.all(np.isfinite(amplitude))

    _, bad_frames, failed = drop_failed_trace_frames(np.ones((2, 6)), None)
    assert failed == [] and bad_frames is None
