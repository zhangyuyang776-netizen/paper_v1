from __future__ import annotations

import numpy as np
import pytest

from core.state_recovery import (
    StateRecoveryError,
    _invert_gas_h_to_T_hpy_first,
    _select_initial_temperature_guess,
    recover_state_from_contents,
    recover_state_from_contents_detailed,
    summarize_recovery_diagnostics,
)
from core.types import (
    ConservativeContents,
    InterfaceState,
    RecoveryConfig,
    RecoveryTemperatureSeeds,
    RegionSlices,
    SpeciesMaps,
    StateRecoveryResult,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class FakeLinearGasThermo:
    """h(T, Y, P) = h_ref + cp * T  (pressure-independent, ignores Y/P)."""

    def __init__(self, cp: float = 1005.0, h_ref: float = 0.0, ref_pressure: float = 101325.0) -> None:
        self.cp = float(cp)
        self.h_ref = float(h_ref)
        self.reference_pressure = ref_pressure

    def enthalpy_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float:
        return self.h_ref + self.cp * float(T)

    def cp_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float:
        return self.cp

    def valid_temperature_range(self, Y_full: np.ndarray | None = None) -> tuple[float, float]:
        return (200.0, 4000.0)


class HPYThermo(FakeLinearGasThermo):
    """Gas thermo that also provides temperature_from_hpy (exact inversion)."""

    def temperature_from_hpy(self, h: float, Y_full: np.ndarray, pressure: float) -> float:
        return (float(h) - self.h_ref) / self.cp


class MoleFractionThermo(HPYThermo):
    """Also provides mole_fractions_from_mass (returns a mock mole fraction matrix)."""

    def mole_fractions_from_mass(self, Y_full: np.ndarray) -> np.ndarray:
        # Trivial: return Y as-is (valid for equal-MW mixture).
        return np.asarray(Y_full, dtype=np.float64)


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


def make_mesh(n_liq: int, n_gas: int):
    from core.types import Mesh1D
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


def make_species_maps() -> SpeciesMaps:
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


def make_contents(n_liq: int, n_gas: int, cp_l: float, cp_g: float,
                  T_l: np.ndarray, T_g: np.ndarray) -> ConservativeContents:
    mass_l = np.full(n_liq, 700.0)
    species_mass_l = mass_l[:, None] * np.ones((n_liq, 1))
    enthalpy_l = mass_l * (cp_l * T_l)
    mass_g = np.full(n_gas, 1.2)
    Yg = np.full((n_gas, 2), [0.79, 0.21])
    species_mass_g = mass_g[:, None] * Yg
    enthalpy_g = mass_g * (cp_g * T_g)
    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )


def make_interface(species_maps: SpeciesMaps) -> InterfaceState:
    return InterfaceState(
        Ts=340.0,
        mpp=0.01,
        Ys_g_full=np.full(species_maps.n_gas_full, 1.0 / species_maps.n_gas_full),
        Ys_l_full=np.full(species_maps.n_liq_full, 1.0),
    )


# ---------------------------------------------------------------------------
# Case 1: HPY-first path is taken and forward check passes
# ---------------------------------------------------------------------------

def test_gas_hpy_path_taken_and_forward_check_passes() -> None:
    cfg = make_recovery_cfg()
    thermo = HPYThermo(cp=1005.0)
    T_target = 900.0
    h_target = thermo.enthalpy_mass(T_target, np.array([0.79, 0.21]))
    y_full = np.array([0.79, 0.21])

    T, mode, bounds, skipped, h_fwd_err = _invert_gas_h_to_T_hpy_first(
        target_h=h_target,
        y_full=y_full,
        recovery_cfg=cfg,
        gas_thermo=thermo,
        pressure=101325.0,
    )
    assert mode == "hpy"
    assert skipped is None
    assert abs(T - T_target) < 1e-6
    assert h_fwd_err < 1.0


# ---------------------------------------------------------------------------
# Case 2: HPY fails forward check → falls back to Newton which succeeds
# ---------------------------------------------------------------------------

class SlightlyOffHPYThermo(FakeLinearGasThermo):
    """temperature_from_hpy returns T slightly perturbed from exact answer."""

    def __init__(self, cp: float, offset: float = 10.0) -> None:
        super().__init__(cp=cp)
        self.offset = offset

    def temperature_from_hpy(self, h: float, Y_full: np.ndarray, pressure: float) -> float:
        exact = (float(h) - self.h_ref) / self.cp
        return exact + self.offset  # perturbed result


def test_gas_hpy_check_failure_falls_back_to_newton() -> None:
    """HPY returns a T offset by 10 K; forward check catches it and Newton is used."""
    # offset = 10 K → |h_fwd - target| = cp * 10 = 10050 J/kg
    # ref = h_target ≈ 904500 J/kg
    # h_check_tol = 1e-4 → threshold = 90.45 J/kg  → 10050 >> 90 → HPY check fails
    cfg = make_recovery_cfg(h_check_tol=1.0e-4)
    thermo = SlightlyOffHPYThermo(cp=1005.0, offset=10.0)
    T_target = 900.0
    h_target = thermo.enthalpy_mass(T_target, np.array([0.79, 0.21]))
    y_full = np.array([0.79, 0.21])

    T, mode, bounds, skipped, h_fwd_err = _invert_gas_h_to_T_hpy_first(
        target_h=h_target,
        y_full=y_full,
        recovery_cfg=cfg,
        gas_thermo=thermo,
        pressure=101325.0,
    )
    assert skipped == "hpy_check_failed"
    assert mode == "fallback_scalar"
    assert abs(T - T_target) < 1e-4
    assert h_fwd_err < 1.0


# ---------------------------------------------------------------------------
# Case 3: missing pressure → skipped_reason = "missing_reference_pressure",
#          Newton fallback used
# ---------------------------------------------------------------------------

def test_gas_missing_pressure_falls_back_to_newton() -> None:
    cfg = make_recovery_cfg()
    thermo = HPYThermo(cp=1005.0)
    T_target = 900.0
    h_target = thermo.enthalpy_mass(T_target, np.array([0.79, 0.21]))
    y_full = np.array([0.79, 0.21])

    T, mode, bounds, skipped, h_fwd_err = _invert_gas_h_to_T_hpy_first(
        target_h=h_target,
        y_full=y_full,
        recovery_cfg=cfg,
        gas_thermo=thermo,
        pressure=None,
    )
    assert mode == "fallback_scalar"
    assert skipped == "missing_reference_pressure"
    assert abs(T - T_target) < 1e-6
    assert h_fwd_err < 1.0


# ---------------------------------------------------------------------------
# Case 4: recover_state_from_contents_detailed returns StateRecoveryResult
# ---------------------------------------------------------------------------

def test_recover_detailed_returns_state_recovery_result() -> None:
    n_liq, n_gas = 2, 3
    T_l = np.array([300.0, 320.0])
    T_g = np.array([900.0, 920.0, 940.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    liq_thermo = HPYThermo(cp=cp_l)  # linear, has enthalpy_mass
    gas_thermo = HPYThermo(cp=cp_g)
    interface = make_interface(species_maps)

    from core.state_recovery import LiquidThermoProtocol

    # Patch liq_thermo to match LiquidThermoProtocol (no pressure arg)
    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp

        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)

        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

        def valid_temperature_range(self) -> tuple[float, float]:
            return (200.0, 800.0)

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=interface,
        gas_pressure=101325.0,
    )

    assert isinstance(result, StateRecoveryResult)
    assert result.state is not None
    assert "liq_inversion_mode" in result.diagnostics
    assert "gas_inversion_mode" in result.diagnostics
    np.testing.assert_allclose(result.state.Tl, T_l, atol=1e-5)
    np.testing.assert_allclose(result.state.Tg, T_g, atol=1e-5)


# ---------------------------------------------------------------------------
# Case 5: Xg_full is populated when mole_fractions_from_mass is available (S-5)
# ---------------------------------------------------------------------------

def test_xg_full_populated_when_method_available() -> None:
    n_liq, n_gas = 1, 2
    T_l = np.array([300.0])
    T_g = np.array([900.0, 920.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    gas_thermo = MoleFractionThermo(cp=cp_g)

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp

        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)

        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
    )
    assert result.state.Xg_full is not None
    assert result.state.Xg_full.shape == (n_gas, species_maps.n_gas_full)


def test_xg_full_is_none_when_method_unavailable() -> None:
    n_liq, n_gas = 1, 2
    T_l = np.array([300.0])
    T_g = np.array([900.0, 920.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    gas_thermo = HPYThermo(cp=cp_g)  # no mole_fractions_from_mass

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp

        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)

        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
    )
    assert result.state.Xg_full is None


# ---------------------------------------------------------------------------
# Case 6: summarize_recovery_diagnostics accepts both State and StateRecoveryResult
# ---------------------------------------------------------------------------

def test_summarize_accepts_state_recovery_result() -> None:
    n_liq, n_gas = 1, 2
    T_l = np.array([300.0])
    T_g = np.array([900.0, 920.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    gas_thermo = HPYThermo(cp=cp_g)

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp

        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)

        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
    )

    summary = summarize_recovery_diagnostics(result)
    assert "min_Tl" in summary
    assert "min_Tg" in summary
    assert "gas_inversion_mode" in summary

    # Also works with bare State
    summary2 = summarize_recovery_diagnostics(result.state)
    assert "min_Tl" in summary2
    assert "gas_inversion_mode" not in summary2


# ---------------------------------------------------------------------------
# Case 7: temperature seeds passed through detailed API (S-2 end-to-end)
# ---------------------------------------------------------------------------

def test_detailed_api_with_temperature_seeds() -> None:
    n_liq, n_gas = 2, 2
    T_l = np.array([300.0, 310.0])
    T_g = np.array([900.0, 910.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    gas_thermo = HPYThermo(cp=cp_g)

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp

        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)

        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    seeds = RecoveryTemperatureSeeds(T_l=T_l.copy(), T_g=T_g.copy())
    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
        temperature_seeds=seeds,
    )
    np.testing.assert_allclose(result.state.Tl, T_l, atol=1e-5)
    np.testing.assert_allclose(result.state.Tg, T_g, atol=1e-5)


# ---------------------------------------------------------------------------
# Case 8: S-1 strict gas_pressure validation
# ---------------------------------------------------------------------------

def test_detailed_api_rejects_nonfinite_gas_pressure() -> None:
    n_liq, n_gas = 1, 1
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, 2000.0, 1005.0,
                             np.array([300.0]), np.array([900.0]))
    cfg = make_recovery_cfg()

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp
        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)
        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    with pytest.raises(StateRecoveryError, match="gas_pressure"):
        recover_state_from_contents_detailed(
            contents=contents,
            mesh=mesh,
            species_maps=species_maps,
            recovery_cfg=cfg,
            liquid_thermo=SimpleLiqThermo(2000.0),
            gas_thermo=HPYThermo(cp=1005.0),
            interface_seed=make_interface(species_maps),
            gas_pressure=float("nan"),
        )


def test_detailed_api_rejects_negative_gas_pressure() -> None:
    n_liq, n_gas = 1, 1
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, 2000.0, 1005.0,
                             np.array([300.0]), np.array([900.0]))
    cfg = make_recovery_cfg()

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp
        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)
        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    with pytest.raises(StateRecoveryError, match="gas_pressure"):
        recover_state_from_contents_detailed(
            contents=contents,
            mesh=mesh,
            species_maps=species_maps,
            recovery_cfg=cfg,
            liquid_thermo=SimpleLiqThermo(2000.0),
            gas_thermo=HPYThermo(cp=1005.0),
            interface_seed=make_interface(species_maps),
            gas_pressure=-1.0,
        )


# ---------------------------------------------------------------------------
# Case 9: S-5 MW conversion fallback populates Xg_full
# ---------------------------------------------------------------------------

class MWConversionThermo(FakeLinearGasThermo):
    """Gas thermo with molecular_weights but no mole_fractions_from_mass."""

    def __init__(self, cp: float, molecular_weights: list[float]) -> None:
        super().__init__(cp=cp)
        self._mw = np.array(molecular_weights, dtype=np.float64)

    @property
    def molecular_weights(self) -> np.ndarray:
        return self._mw

    def temperature_from_hpy(self, h: float, Y_full: np.ndarray, pressure: float) -> float:
        return (float(h) - self.h_ref) / self.cp


def test_xg_full_populated_via_mw_conversion() -> None:
    """When mole_fractions_from_mass is absent but molecular_weights exists, use MW fallback."""
    n_liq, n_gas = 1, 2
    T_l = np.array([300.0])
    T_g = np.array([900.0, 920.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()  # 2 gas species: N2=28, O2=32
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    # MW of N2=28, O2=32 g/mol
    gas_thermo = MWConversionThermo(cp=cp_g, molecular_weights=[28.0, 32.0])

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp
        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)
        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
    )
    assert result.state.Xg_full is not None
    assert result.state.Xg_full.shape == (n_gas, species_maps.n_gas_full)
    # Row sums should be ≈ 1.
    np.testing.assert_allclose(np.sum(result.state.Xg_full, axis=1), np.ones(n_gas), atol=1e-12)
    assert result.diagnostics["Xg_full_recovery_status"] == "mw_conversion"


# ---------------------------------------------------------------------------
# Case 10: S-7 full diagnostics key set is present
# ---------------------------------------------------------------------------

def test_detailed_result_contains_full_s7_diagnostic_keys() -> None:
    n_liq, n_gas = 2, 2
    T_l = np.array([300.0, 310.0])
    T_g = np.array([900.0, 910.0])
    cp_l, cp_g = 2000.0, 1005.0
    mesh = make_mesh(n_liq, n_gas)
    species_maps = make_species_maps()
    contents = make_contents(n_liq, n_gas, cp_l, cp_g, T_l, T_g)
    cfg = make_recovery_cfg()
    gas_thermo = HPYThermo(cp=cp_g)

    class SimpleLiqThermo:
        def __init__(self, cp: float) -> None:
            self.cp = cp
        def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp * float(T)
        def cp_mass(self, T: float, Y_full: np.ndarray) -> float:
            return self.cp

    result = recover_state_from_contents_detailed(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=SimpleLiqThermo(cp_l),
        gas_thermo=gas_thermo,
        interface_seed=make_interface(species_maps),
        gas_pressure=101325.0,
    )
    diag = result.diagnostics
    required_keys = [
        "liq_recovery_success",
        "gas_recovery_success",
        "liq_h_fwd_check_max_err",
        "gas_h_fwd_check_max_err",
        "liq_Y_num_minor_fixes",
        "liq_Y_max_sum_err",
        "liq_Y_max_negative_species_mass",
        "gas_Y_num_minor_fixes",
        "gas_Y_max_sum_err",
        "gas_Y_max_negative_species_mass",
        "gas_recovery_used_HPY",
        "gas_recovery_used_fallback",
        "gas_pressure_source",
        "Xg_full_recovery_status",
        "postchecks_passed",
        "min_rho_l",
        "min_rho_g",
        "n_liq_cells",
        "n_gas_cells",
        "liq_inversion_mode",
        "gas_inversion_mode",
        "gas_hpy_skipped_reason",
    ]
    for key in required_keys:
        assert key in diag, f"Missing diagnostics key: {key!r}"
    assert diag["gas_pressure_source"] == "explicit"
    assert diag["postchecks_passed"] is True
    assert diag["liq_recovery_success"] is True
    assert diag["gas_recovery_success"] is True


# ---------------------------------------------------------------------------
# Case 11: S-2 cp_min linear estimate provides T_hint
# ---------------------------------------------------------------------------

def test_cp_min_linear_estimate_used_as_t_hint() -> None:
    """cp_min linear estimate T=h/cp_min should be used when no seed or rolling hint."""
    from core.state_recovery import _select_initial_temperature_guess
    cfg = make_recovery_cfg(cp_min=1005.0)
    T_low, T_high = 200.0, 4000.0
    # h ≈ 1005 * 900 = 904500 → T_est = 904500 / 1005 = 900 K (in bounds)
    h_target = 1005.0 * 900.0
    T_hint = _select_initial_temperature_guess(
        target_h=h_target,
        bounds=(T_low, T_high),
        recovery_cfg=cfg,
        seed=None,
        rolling_hint=None,
    )
    assert T_hint is not None
    assert abs(T_hint - 900.0) < 1.0


def test_cp_min_level2_skipped_when_seed_present() -> None:
    """Level 1 seed takes priority over cp_min estimate."""
    from core.state_recovery import _select_initial_temperature_guess
    cfg = make_recovery_cfg(cp_min=1005.0)
    T_hint = _select_initial_temperature_guess(
        target_h=1005.0 * 900.0,
        bounds=(200.0, 4000.0),
        recovery_cfg=cfg,
        seed=850.0,
        rolling_hint=None,
    )
    # Should use seed=850, not cp_min estimate=900
    assert T_hint == pytest.approx(850.0)
