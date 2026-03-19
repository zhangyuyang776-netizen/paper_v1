from __future__ import annotations

import numpy as np
import pytest

from core.types import InterfaceState, State


def test_state_single_component_liquid_valid() -> None:
    state = State(
        Tl=np.array([300.0, 305.0], dtype=np.float64),
        Yl_full=np.array([[1.0], [1.0]], dtype=np.float64),
        Tg=np.array([320.0, 330.0], dtype=np.float64),
        Yg_full=np.array([[0.1, 0.21, 0.69], [0.12, 0.20, 0.68]], dtype=np.float64),
        interface=InterfaceState(
            Ts=310.0,
            mpp=0.01,
            Ys_g_full=np.array([0.15, 0.20, 0.65], dtype=np.float64),
            Ys_l_full=np.array([1.0], dtype=np.float64),
        ),
    )
    assert state.n_liq_species_full == 1
    assert state.n_gas_species_full == 3


def test_state_multicomponent_liquid_valid() -> None:
    state = State(
        Tl=np.array([300.0, 305.0], dtype=np.float64),
        Yl_full=np.array([[0.6, 0.4], [0.55, 0.45]], dtype=np.float64),
        Tg=np.array([320.0, 330.0], dtype=np.float64),
        Yg_full=np.array([[0.1, 0.21, 0.69], [0.12, 0.20, 0.68]], dtype=np.float64),
        interface=InterfaceState(
            Ts=310.0,
            mpp=0.01,
            Ys_g_full=np.array([0.15, 0.20, 0.65], dtype=np.float64),
            Ys_l_full=np.array([0.5, 0.5], dtype=np.float64),
        ),
    )
    assert state.n_liq_species_full == 2


def test_state_liquid_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        State(
            Tl=np.array([300.0, 305.0], dtype=np.float64),
            Yl_full=np.array([[1.0]], dtype=np.float64),
            Tg=np.array([320.0], dtype=np.float64),
            Yg_full=np.array([[0.2, 0.8]], dtype=np.float64),
            interface=InterfaceState(
                Ts=310.0,
                mpp=0.01,
                Ys_g_full=np.array([0.2, 0.8], dtype=np.float64),
                Ys_l_full=np.array([1.0], dtype=np.float64),
            ),
        )


def test_state_gas_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        State(
            Tl=np.array([300.0], dtype=np.float64),
            Yl_full=np.array([[1.0]], dtype=np.float64),
            Tg=np.array([320.0, 330.0], dtype=np.float64),
            Yg_full=np.array([[0.2, 0.8]], dtype=np.float64),
            interface=InterfaceState(
                Ts=310.0,
                mpp=0.01,
                Ys_g_full=np.array([0.2, 0.8], dtype=np.float64),
                Ys_l_full=np.array([1.0], dtype=np.float64),
            ),
        )


def test_interface_state_single_component_liquid_kept() -> None:
    interface = InterfaceState(
        Ts=310.0,
        mpp=0.02,
        Ys_g_full=np.array([0.2, 0.8], dtype=np.float64),
        Ys_l_full=np.array([1.0], dtype=np.float64),
    )
    assert interface.Ys_l_full.shape == (1,)
