from __future__ import annotations

import pytest

from core.types import RecoveryConfig


def make_valid_recovery_config(**overrides: object) -> RecoveryConfig:
    defaults: dict[str, object] = dict(
        rho_min=1.0e-12,
        m_min=1.0e-20,
        species_recovery_eps_abs=1.0e-14,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
        h_abs_tol=1.0e-8,
        h_rel_tol=1.0e-10,
        h_check_tol=1.0e-8,
        T_step_tol=1.0e-8,
        T_min_l=250.0,
        T_max_l=900.0,
        T_min_g=200.0,
        T_max_g=4000.0,
        liquid_h_inv_max_iter=50,
        cp_min=1.0,
        gas_h_inv_max_iter=50,
        use_cantera_hpy_first=True,
    )
    defaults.update(overrides)
    return RecoveryConfig(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Case 1: valid config constructs successfully and aliases work
# ---------------------------------------------------------------------------

def test_valid_config_constructs() -> None:
    cfg = make_valid_recovery_config()
    assert cfg.rho_min == 1.0e-12
    assert cfg.h_abs_tol == 1.0e-8
    assert cfg.liquid_h_inv_max_iter == 50
    assert cfg.gas_h_inv_max_iter == 50
    assert cfg.use_cantera_hpy_first is True


def test_backward_compat_aliases() -> None:
    cfg = make_valid_recovery_config(h_abs_tol=1.0e-9, liquid_h_inv_max_iter=75)
    # liq_h_inv_tol / gas_h_inv_tol → h_abs_tol
    assert cfg.liq_h_inv_tol == cfg.h_abs_tol == 1.0e-9
    assert cfg.gas_h_inv_tol == cfg.h_abs_tol == 1.0e-9
    # liq_h_inv_max_iter → liquid_h_inv_max_iter
    assert cfg.liq_h_inv_max_iter == cfg.liquid_h_inv_max_iter == 75
    # gas_h_inv_max_iter is a direct field
    assert cfg.gas_h_inv_max_iter == 50


# ---------------------------------------------------------------------------
# Case 2: use_cantera_hpy_first=False raises ValueError
# ---------------------------------------------------------------------------

def test_use_cantera_hpy_first_false_raises() -> None:
    with pytest.raises(ValueError, match="use_cantera_hpy_first"):
        make_valid_recovery_config(use_cantera_hpy_first=False)


# ---------------------------------------------------------------------------
# Case 3: strictly positive threshold fields reject zero and negative values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field",
    [
        "rho_min",
        "m_min",
        "species_recovery_eps_abs",
        "Y_sum_tol",
        "Y_hard_tol",
        "h_abs_tol",
        "h_rel_tol",
        "h_check_tol",
        "T_step_tol",
        "cp_min",
    ],
)
def test_strictly_positive_field_zero_raises(field: str) -> None:
    with pytest.raises(ValueError):
        make_valid_recovery_config(**{field: 0.0})


@pytest.mark.parametrize("field", ["rho_min", "h_abs_tol", "cp_min"])
def test_strictly_positive_field_negative_raises(field: str) -> None:
    with pytest.raises(ValueError):
        make_valid_recovery_config(**{field: -1.0e-10})


def test_T_min_l_zero_raises() -> None:
    with pytest.raises(ValueError):
        make_valid_recovery_config(T_min_l=0.0)


def test_T_min_g_zero_raises() -> None:
    with pytest.raises(ValueError):
        make_valid_recovery_config(T_min_g=0.0)


# ---------------------------------------------------------------------------
# Case 4: temperature bound inversions raise ValueError
# ---------------------------------------------------------------------------

def test_T_max_l_leq_T_min_l_raises() -> None:
    with pytest.raises(ValueError, match="T_max_l"):
        make_valid_recovery_config(T_min_l=500.0, T_max_l=400.0)


def test_T_max_l_equal_T_min_l_raises() -> None:
    with pytest.raises(ValueError, match="T_max_l"):
        make_valid_recovery_config(T_min_l=400.0, T_max_l=400.0)


def test_T_max_g_leq_T_min_g_raises() -> None:
    with pytest.raises(ValueError, match="T_max_g"):
        make_valid_recovery_config(T_min_g=3000.0, T_max_g=2000.0)


# ---------------------------------------------------------------------------
# Case 5: iteration count < 1 raises ValueError
# ---------------------------------------------------------------------------

def test_liquid_h_inv_max_iter_zero_raises() -> None:
    with pytest.raises(ValueError, match="liquid_h_inv_max_iter"):
        make_valid_recovery_config(liquid_h_inv_max_iter=0)


def test_gas_h_inv_max_iter_zero_raises() -> None:
    with pytest.raises(ValueError, match="gas_h_inv_max_iter"):
        make_valid_recovery_config(gas_h_inv_max_iter=0)
