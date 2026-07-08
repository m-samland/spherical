"""Tests for the packaged coronagraph-transmission wiring in run_trap.py."""

import numpy as np
import pytest

# trap is only installed in the dev env (pipeline deps validated locally);
# skip cleanly in the CI `test` env instead of erroring at collection.
pytest.importorskip("trap")

from trap.parameters import trap_config_for_ifs

from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.run_trap import _load_coronagraph_transmission, _resolve_coronagraph_transmission


def test_load_ifs_transmission_shape_and_range():
    table = _load_coronagraph_transmission("IFS")
    assert table.ndim == 2 and table.shape[1] == 2
    separation, throughput = table[:, 0], table[:, 1]
    assert separation[0] == pytest.approx(0.0)
    assert separation[-1] == pytest.approx(200.0)
    assert np.all(np.diff(separation) > 0)          # strictly ascending
    assert np.all((throughput >= 0.0) & (throughput <= 1.0))


def test_load_irdis_transmission_loads():
    table = _load_coronagraph_transmission("IRDIS")
    assert table.ndim == 2 and table.shape[1] == 2


def test_load_unknown_instrument_raises():
    with pytest.raises(KeyError):
        _load_coronagraph_transmission("NIRPS")


def test_default_config_toggle_is_true():
    assert IFSReductionConfig().apply_coronagraph_transmission is True


def test_resolve_injects_default_when_toggle_on_and_unset():
    reduction_config = IFSReductionConfig()
    trap_reduction_config = trap_config_for_ifs().reduction  # coronagraph_transmission defaults None
    assert trap_reduction_config.coronagraph_transmission is None
    table = _resolve_coronagraph_transmission(reduction_config, trap_reduction_config)
    assert isinstance(table, np.ndarray) and table.shape[1] == 2


def test_resolve_none_when_toggle_off():
    reduction_config = IFSReductionConfig()
    reduction_config.apply_coronagraph_transmission = False  # mutable dataclass
    trap_reduction_config = trap_config_for_ifs().reduction
    assert _resolve_coronagraph_transmission(reduction_config, trap_reduction_config) is None


def test_resolve_respects_explicit_user_table():
    reduction_config = IFSReductionConfig()
    user_table = np.array([[0.0, 0.5], [100.0, 1.0]])
    trap_reduction_config = trap_config_for_ifs().reduction.merge(
        coronagraph_transmission=user_table)
    # explicit table wins -> resolve returns None (no change)
    assert _resolve_coronagraph_transmission(reduction_config, trap_reduction_config) is None


def test_resolve_none_when_field_unsupported():
    from types import SimpleNamespace
    reduction_config = IFSReductionConfig()
    unsupported = SimpleNamespace()  # no coronagraph_transmission attribute
    assert _resolve_coronagraph_transmission(reduction_config, unsupported) is None
