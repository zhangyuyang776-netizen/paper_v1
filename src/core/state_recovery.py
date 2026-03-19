from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import ConservativeContents, InterfaceState, Mesh1D, RecoveryConfig, SpeciesMaps, State


RECOVERY_MASS_TOL = 1.0e-12


class StateRecoveryError(ValueError):
    """Raised when a conservative state cannot be recovered into a valid primitive state."""


class LiquidThermoProtocol(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...


class GasThermoProtocol(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...


def _recover_density(mass: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    mass = np.asarray(mass, dtype=np.float64)
    volumes = np.asarray(volumes, dtype=np.float64)
    if mass.shape != volumes.shape:
        raise StateRecoveryError("mass and volumes must have the same shape")
    if np.any(volumes <= 0.0):
        raise StateRecoveryError("volumes must be strictly positive for density recovery")
    if np.any(mass <= 0.0):
        raise StateRecoveryError("mass must be strictly positive for density recovery")
    return mass / volumes


def _recover_full_mass_fractions(
    species_mass: np.ndarray,
    mass: np.ndarray,
    *,
    n_full: int,
    single_component_name: str | None = None,
) -> np.ndarray:
    species_mass = np.asarray(species_mass, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1:
        raise StateRecoveryError("mass must be one-dimensional")
    if species_mass.ndim != 2:
        raise StateRecoveryError("species_mass must be two-dimensional")
    if species_mass.shape != (mass.shape[0], n_full):
        raise StateRecoveryError("species_mass shape must be (n_cells, n_full)")
    if np.any(mass <= 0.0):
        raise StateRecoveryError("mass must be strictly positive for species recovery")
    if np.any(species_mass < 0.0):
        raise StateRecoveryError("species_mass must be non-negative")

    species_sum = np.sum(species_mass, axis=1)
    diff = np.abs(species_sum - mass)
    tol = RECOVERY_MASS_TOL * np.maximum(mass, 1.0)
    if np.any(diff > tol):
        raise StateRecoveryError("species_mass sums must match mass within tolerance")

    if n_full == 1:
        # Formal paper_v1 rule:
        # single-component liquid is treated as a no-reduced-species case and recovers Y_full = [1.0].
        return np.ones((mass.shape[0], 1), dtype=np.float64)

    return species_mass / mass[:, None]


def _recover_specific_enthalpy(enthalpy: np.ndarray, mass: np.ndarray) -> np.ndarray:
    enthalpy = np.asarray(enthalpy, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if enthalpy.shape != mass.shape:
        raise StateRecoveryError("enthalpy and mass must have the same shape")
    if np.any(mass <= 0.0):
        raise StateRecoveryError("mass must be strictly positive for specific enthalpy recovery")
    return enthalpy / mass


def _invert_temperature_monotone_bisection(
    *,
    target_h: float,
    y_full: np.ndarray,
    thermo: LiquidThermoProtocol | GasThermoProtocol,
    T_low: float,
    T_high: float,
    tol: float,
    max_iter: int,
) -> float:
    if not np.isfinite(target_h):
        raise StateRecoveryError("target_h must be finite")
    if not np.isfinite(T_low) or not np.isfinite(T_high) or T_high <= T_low:
        raise StateRecoveryError("Require finite T_low, T_high and T_high > T_low")
    if tol <= 0.0:
        raise StateRecoveryError("tol must be > 0")
    if max_iter < 1:
        raise StateRecoveryError("max_iter must be >= 1")

    h_low = float(thermo.enthalpy_mass(T_low, y_full))
    h_high = float(thermo.enthalpy_mass(T_high, y_full))
    if not (h_low <= target_h <= h_high):
        raise StateRecoveryError("target_h must be bracketed by enthalpy values at T_low and T_high")

    left = float(T_low)
    right = float(T_high)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        h_mid = float(thermo.enthalpy_mass(mid, y_full))
        if abs(h_mid - target_h) <= tol:
            return mid
        if h_mid < target_h:
            left = mid
        else:
            right = mid

    raise StateRecoveryError("enthalpy inversion exceeded max_iter without convergence")


def _invert_liquid_h_to_T(
    h: np.ndarray,
    Y_full: np.ndarray,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    Y_full = np.asarray(Y_full, dtype=np.float64)
    if Y_full.shape[0] != h.shape[0]:
        raise StateRecoveryError("liquid Y_full row count must match h length")
    return np.array(
        [
            _invert_temperature_monotone_bisection(
                target_h=float(h_i),
                y_full=Y_full[i, :],
                thermo=liquid_thermo,
                T_low=recovery_cfg.T_min_l,
                T_high=recovery_cfg.T_max_l,
                tol=recovery_cfg.liq_h_inv_tol,
                max_iter=recovery_cfg.liq_h_inv_max_iter,
            )
            for i, h_i in enumerate(h)
        ],
        dtype=np.float64,
    )


def _invert_gas_h_to_T(
    h: np.ndarray,
    Y_full: np.ndarray,
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    Y_full = np.asarray(Y_full, dtype=np.float64)
    if Y_full.shape[0] != h.shape[0]:
        raise StateRecoveryError("gas Y_full row count must match h length")
    return np.array(
        [
            _invert_temperature_monotone_bisection(
                target_h=float(h_i),
                y_full=Y_full[i, :],
                thermo=gas_thermo,
                T_low=recovery_cfg.T_min_g,
                T_high=recovery_cfg.T_max_g,
                tol=recovery_cfg.gas_h_inv_tol,
                max_iter=recovery_cfg.gas_h_inv_max_iter,
            )
            for i, h_i in enumerate(h)
        ],
        dtype=np.float64,
    )


def _recover_liquid_phase_state(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    liq_volumes = mesh.volumes[mesh.region_slices.liq]
    rho_l = _recover_density(contents.mass_l, liq_volumes)
    Yl_full = _recover_full_mass_fractions(
        contents.species_mass_l,
        contents.mass_l,
        n_full=species_maps.n_liq_full,
        single_component_name=species_maps.liq_full_names[0] if species_maps.n_liq_full == 1 else None,
    )
    hl = _recover_specific_enthalpy(contents.enthalpy_l, contents.mass_l)
    Tl = _invert_liquid_h_to_T(hl, Yl_full, recovery_cfg, liquid_thermo)
    return rho_l, Yl_full, hl, Tl


def _recover_gas_phase_state(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gas_volumes = mesh.volumes[mesh.region_slices.gas_all]
    rho_g = _recover_density(contents.mass_g, gas_volumes)
    Yg_full = _recover_full_mass_fractions(
        contents.species_mass_g,
        contents.mass_g,
        n_full=species_maps.n_gas_full,
    )
    hg = _recover_specific_enthalpy(contents.enthalpy_g, contents.mass_g)
    Tg = _invert_gas_h_to_T(hg, Yg_full, recovery_cfg, gas_thermo)
    return rho_g, Yg_full, hg, Tg


def _copy_interface_state(interface_seed: InterfaceState) -> InterfaceState:
    return InterfaceState(
        Ts=float(interface_seed.Ts),
        mpp=float(interface_seed.mpp),
        Ys_g_full=np.array(interface_seed.Ys_g_full, dtype=np.float64, copy=True),
        Ys_l_full=np.array(interface_seed.Ys_l_full, dtype=np.float64, copy=True),
    )


def validate_recovered_state_bounds(state: State, recovery_cfg: RecoveryConfig) -> None:
    if np.any(state.Tl < recovery_cfg.T_min_l) or np.any(state.Tl > recovery_cfg.T_max_l):
        raise StateRecoveryError("recovered liquid temperature lies outside recovery bounds")
    if np.any(state.Tg < recovery_cfg.T_min_g) or np.any(state.Tg > recovery_cfg.T_max_g):
        raise StateRecoveryError("recovered gas temperature lies outside recovery bounds")


def recover_state_from_contents(
    *,
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig | None = None,
    recovery_config: RecoveryConfig | None = None,
    liquid_thermo: LiquidThermoProtocol | None = None,
    gas_thermo: GasThermoProtocol | None = None,
    interface_seed: InterfaceState | None = None,
    time: float | None = None,
    state_id: str | None = None,
) -> State:
    # Formal paper_v1 rule:
    # recovery proceeds from conservative cell contents to rho, Y, h, and only then to T.
    # Recovery does not renormalize, clip, or silently correct mass fractions or temperatures.
    # Invalid conservative states must raise an explicit StateRecoveryError.
    cfg = recovery_cfg if recovery_cfg is not None else recovery_config
    if cfg is None:
        raise StateRecoveryError("recover_state_from_contents requires recovery_cfg")
    if liquid_thermo is None:
        raise StateRecoveryError("recover_state_from_contents requires liquid_thermo")
    if gas_thermo is None:
        raise StateRecoveryError("recover_state_from_contents requires gas_thermo")
    if interface_seed is None:
        raise StateRecoveryError("recover_state_from_contents requires interface_seed")

    rho_l, Yl_full, hl, Tl = _recover_liquid_phase_state(contents, mesh, species_maps, cfg, liquid_thermo)
    rho_g, Yg_full, hg, Tg = _recover_gas_phase_state(contents, mesh, species_maps, cfg, gas_thermo)

    state = State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=_copy_interface_state(interface_seed),
        rho_l=rho_l,
        rho_g=rho_g,
        hl=hl,
        hg=hg,
        Xg_full=None,
        time=time,
        state_id=state_id,
    )
    validate_recovered_state_bounds(state, cfg)
    return state


def summarize_recovery_diagnostics(state: State) -> dict[str, float]:
    return {
        "min_Tl": float(np.min(state.Tl)),
        "max_Tl": float(np.max(state.Tl)),
        "min_Tg": float(np.min(state.Tg)),
        "max_Tg": float(np.max(state.Tg)),
        "min_rho_l": float(np.min(state.rho_l)) if state.rho_l is not None else float("nan"),
        "min_rho_g": float(np.min(state.rho_g)) if state.rho_g is not None else float("nan"),
    }


__all__ = [
    "GasThermoProtocol",
    "LiquidThermoProtocol",
    "StateRecoveryError",
    "recover_state_from_contents",
    "summarize_recovery_diagnostics",
]
