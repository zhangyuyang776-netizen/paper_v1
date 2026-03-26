from __future__ import annotations

import numpy as np
import pytest

from core.state_recovery import (
    StateRecoveryError,
    _invert_liquid_h_to_T_safeguarded,
    _recover_liquid_phase_state_with_diagnostics,
    validate_recovered_state_postchecks,
)
from core.types import (
    ConservativeContents,
    RecoveryConfig,
    RecoveryTemperatureSeeds,
    RegionSlices,
    SpeciesMaps,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeLinearLiquidThermo:
    """h(T, Y) = h_ref + cp * T  (pressure-independent)."""

    def __init__(self, cp: float = 2000.0, h_ref: float = 0.0) -> None:
        self.cp = float(cp)
        self.h_ref = float(h_ref)

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        return self.h_ref + self.cp * float(T)

    def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
        return self.cp


def make_recovery_cfg(**overrides: object) -> RecoveryConfig:
    defaults: dict[str, object] = dict(
        rho_min=1.0e-12,
        m_min=1.0e-20,
        species_recovery_eps_abs=1.0e-14,
        Y_sum_tol=1.0e-10,
        Y_hard_tol=1.0e-6,
        h_abs_tol=1.0e-10,
        h_rel_tol=1.0e-10,
        h_check_tol=1.0e-6,
        T_step_tol=1.0e-8,
        T_min_l=200.0,
        T_max_l=800.0,
        T_min_g=200.0,
        T_max_g=4000.0,
        liquid_h_inv_max_iter=100,
        cp_min=1.0,
        gas_h_inv_max_iter=100,
        use_cantera_hpy_first=True,
    )
    defaults.update(overrides)
    return RecoveryConfig(**defaults)  # type: ignore[arg-type]


def make_mesh_liq(n_liq: int = 3):
    """Build a mesh with n_liq liquid cells and 1 dummy gas cell."""
    from core.types import Mesh1D
    n_gas = 1
    n_cells = n_liq + n_gas
    r_faces = np.linspace(0.0, float(n_cells), n_cells + 1, dtype=np.float64)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    volumes = np.ones(n_cells, dtype=np.float64)
    face_areas = np.ones(n_cells + 1, dtype=np.float64)
    dr = np.ones(n_cells, dtype=np.float64)
    return Mesh1D(
        r_faces=r_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=RegionSlices(
            liq=slice(0, n_liq),
            gas_near=slice(n_liq, n_cells),
            gas_far=slice(n_cells, n_cells),
            gas_all=slice(n_liq, n_cells),
        ),
        interface_face_index=n_liq,
        interface_cell_liq=n_liq - 1,
        interface_cell_gas=n_liq,
    )


def make_species_maps_single_liq() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("N2", "O2"),
        gas_active_names=("O2",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0], dtype=np.int64),
        gas_reduced_to_full=np.array([1], dtype=np.int64),
        liq_full_to_gas_full=np.array([0], dtype=np.int64),
    )


def make_liq_contents(n_liq: int, cp: float, T_ref: np.ndarray) -> ConservativeContents:
    rho_l = np.full(n_liq, 700.0, dtype=np.float64)
    mass_l = rho_l * 1.0  # volumes = 1
    species_mass_l = mass_l[:, None] * np.ones((n_liq, 1))
    enthalpy_l = mass_l * (cp * T_ref)
    # 1 dummy gas cell (make_mesh_liq always adds 1 gas cell)
    n_gas = 1
    n_gas_full = 2
    mass_g = np.ones(n_gas, dtype=np.float64)
    Yg = np.full((n_gas, n_gas_full), [0.79, 0.21])
    species_mass_g = mass_g[:, None] * Yg
    enthalpy_g = mass_g * 1005.0 * 900.0
    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )


# ---------------------------------------------------------------------------
# Case 1: basic liquid enthalpy inversion with forward check passes
# ---------------------------------------------------------------------------

def test_liquid_inversion_forward_check_passes() -> None:
    cfg = make_recovery_cfg()
    thermo = FakeLinearLiquidThermo(cp=2000.0)
    T_target = 350.0
    h_target = thermo.enthalpy_mass(T_target, np.array([1.0]))
    y_full = np.array([1.0])

    T, mode, bounds, h_fwd_err, _ = _invert_liquid_h_to_T_safeguarded(
        target_h=h_target,
        y_full=y_full,
        recovery_cfg=cfg,
        liquid_thermo=thermo,
    )
    assert abs(T - T_target) < 1e-6
    assert h_fwd_err < 1.0


# ---------------------------------------------------------------------------
# Case 2: forward check rejects a deliberately wrong T (mock that lies)
# ---------------------------------------------------------------------------

def test_liquid_inversion_forward_check_raises_on_bad_result() -> None:
    """Forward check rejects a T that satisfies h_abs_tol but not h_check_tol.

    Strategy: set h_abs_tol extremely large so Newton stops at the starting
    midpoint T=500 K (well away from T_target=350 K).  The forward check with
    the default tight h_check_tol then rejects that result.
    """
    cfg = make_recovery_cfg(h_abs_tol=1.0e10, h_check_tol=1.0e-6)
    thermo = FakeLinearLiquidThermo(cp=2000.0)
    T_target = 350.0
    # h(T_target) = 700,000 J/kg; bracket is [200, 800] → midpoint = 500 K
    # Newton stops at 500 K (|h(500) - 700000| = 300000 <= 1e10)
    # Absolute forward check: 300000 > h_check_tol=1e-6 → raises
    h_target = thermo.enthalpy_mass(T_target, np.array([1.0]))
    y_full = np.array([1.0])

    with pytest.raises(StateRecoveryError, match="forward enthalpy check failed"):
        _invert_liquid_h_to_T_safeguarded(
            target_h=h_target,
            y_full=y_full,
            recovery_cfg=cfg,
            liquid_thermo=thermo,
        )


# ---------------------------------------------------------------------------
# Case 3: temperature seeds (S-2) accelerate convergence — seed is used
# ---------------------------------------------------------------------------

def test_liquid_phase_recovery_with_seeds() -> None:
    n_liq = 3
    T_true = np.array([300.0, 320.0, 340.0])
    cp = 2000.0
    mesh = make_mesh_liq(n_liq)
    species_maps = make_species_maps_single_liq()
    contents = make_liq_contents(n_liq, cp, T_true)
    cfg = make_recovery_cfg()
    thermo = FakeLinearLiquidThermo(cp=cp)

    # Provide perfect seeds (as if from previous timestep).
    seeds = RecoveryTemperatureSeeds(T_l=T_true.copy())

    rho_l, Yl_full, hl, Tl, diag = _recover_liquid_phase_state_with_diagnostics(
        contents, mesh, species_maps, cfg, thermo,
        temperature_seeds=seeds,
    )
    np.testing.assert_allclose(Tl, T_true, atol=1e-6)
    # Convergence is achieved (mode must be a non-empty string).
    assert isinstance(diag["liq_inversion_mode"], str)


def test_liquid_phase_recovery_without_seeds_also_passes() -> None:
    n_liq = 3
    T_true = np.array([300.0, 320.0, 340.0])
    cp = 2000.0
    mesh = make_mesh_liq(n_liq)
    species_maps = make_species_maps_single_liq()
    contents = make_liq_contents(n_liq, cp, T_true)
    cfg = make_recovery_cfg()
    thermo = FakeLinearLiquidThermo(cp=cp)

    rho_l, Yl_full, hl, Tl, diag = _recover_liquid_phase_state_with_diagnostics(
        contents, mesh, species_maps, cfg, thermo,
    )
    np.testing.assert_allclose(Tl, T_true, atol=1e-6)


# ---------------------------------------------------------------------------
# Case 4: NaN seed falls back to rolling hint (no crash)
# ---------------------------------------------------------------------------

def test_liquid_nan_seed_falls_back_to_rolling_hint() -> None:
    n_liq = 2
    T_true = np.array([300.0, 320.0])
    cp = 2000.0
    mesh = make_mesh_liq(n_liq)
    species_maps = make_species_maps_single_liq()
    contents = make_liq_contents(n_liq, cp, T_true)
    cfg = make_recovery_cfg()
    thermo = FakeLinearLiquidThermo(cp=cp)

    seeds = RecoveryTemperatureSeeds(T_l=np.array([float("nan"), float("nan")]))
    rho_l, Yl_full, hl, Tl, diag = _recover_liquid_phase_state_with_diagnostics(
        contents, mesh, species_maps, cfg, thermo,
        temperature_seeds=seeds,
    )
    np.testing.assert_allclose(Tl, T_true, atol=1e-6)


# ---------------------------------------------------------------------------
# Case 5: liquid phase diagnostics contain expected S-7 keys
# ---------------------------------------------------------------------------

def test_liquid_phase_diagnostics_contain_s7_keys() -> None:
    n_liq = 2
    T_true = np.array([300.0, 320.0])
    cp = 2000.0
    mesh = make_mesh_liq(n_liq)
    species_maps = make_species_maps_single_liq()
    contents = make_liq_contents(n_liq, cp, T_true)
    cfg = make_recovery_cfg()
    thermo = FakeLinearLiquidThermo(cp=cp)

    rho_l, Yl_full, hl, Tl, diag = _recover_liquid_phase_state_with_diagnostics(
        contents, mesh, species_maps, cfg, thermo,
    )
    for key in ("liq_recovery_success", "liq_h_fwd_check_max_err",
                "liq_Y_num_minor_fixes", "liq_Y_max_sum_err",
                "liq_Y_max_negative_species_mass"):
        assert key in diag, f"Missing key: {key!r}"
    assert diag["liq_recovery_success"] is True
    assert diag["liq_h_fwd_check_max_err"] >= 0.0
    assert diag["liq_Y_num_minor_fixes"] == 0


# ---------------------------------------------------------------------------
# Case 6: validate_recovered_state_postchecks raises on rho below rho_min
# ---------------------------------------------------------------------------

def test_postchecks_raise_on_rho_below_rho_min() -> None:
    from core.state_recovery import StateRecoveryError, validate_recovered_state_postchecks
    from core.types import InterfaceState, State

    cfg = make_recovery_cfg(rho_min=100.0)  # very high rho_min to force failure
    state = State(
        Tl=np.array([300.0]),
        Yl_full=np.ones((1, 1)),
        Tg=np.array([900.0]),
        Yg_full=np.ones((1, 1)),
        interface=InterfaceState(
            Ts=300.0, mpp=0.0,
            Ys_g_full=np.array([1.0]),
            Ys_l_full=np.array([1.0]),
        ),
        rho_l=np.array([1.0]),   # far below rho_min=100
        rho_g=np.array([1.0]),
        hl=np.array([600000.0]),
        hg=np.array([904500.0]),
    )
    with pytest.raises(StateRecoveryError, match="rho_min"):
        validate_recovered_state_postchecks(state, cfg)
