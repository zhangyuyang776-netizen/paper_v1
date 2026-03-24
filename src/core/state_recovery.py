from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import ConservativeContents, InterfaceState, Mesh1D, RecoveryConfig, SpeciesMaps, State


RECOVERY_MASS_TOL = 1.0e-12


class StateRecoveryError(ValueError):
    """Raised when a conservative state cannot be recovered into a valid primitive state."""


class LiquidThermoProtocol(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...

    def cp_mass(self, T: float, Y_full: np.ndarray) -> float: ...

    def valid_temperature_range(self, species_subset: tuple[str, ...] | None = None) -> tuple[float, float]: ...


class GasThermoProtocol(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float: ...

    def cp_mass(self, T: float, Y_full: np.ndarray, P: float | None = None) -> float: ...

    def valid_temperature_range(self, Y_full: np.ndarray | None = None) -> tuple[float, float]: ...

    def temperature_from_hpy(self, h: float, Y_full: np.ndarray, pressure: float) -> float: ...


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
        _ = single_component_name
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


def _validate_effective_temperature_bracket(T_low: float, T_high: float, *, label: str) -> tuple[float, float]:
    low = float(T_low)
    high = float(T_high)
    if not np.isfinite(low) or not np.isfinite(high):
        raise StateRecoveryError(f"{label} effective bracket must be finite")
    if high <= low:
        raise StateRecoveryError(f"{label} effective bracket is empty or invalid")
    return low, high


def _query_valid_temperature_range(
    thermo: object,
    *,
    y_full: np.ndarray | None = None,
) -> tuple[float, float] | None:
    valid_range = getattr(thermo, "valid_temperature_range", None)
    if not callable(valid_range):
        return None
    attempts: list[tuple[object, ...]] = []
    if y_full is not None:
        attempts.append((y_full,))
        attempts.append(tuple())
    else:
        attempts.append(tuple())
    for args in attempts:
        try:
            result = valid_range(*args)
        except TypeError:
            continue
        return (float(result[0]), float(result[1]))
    return None


def _get_effective_liquid_temperature_bounds(
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
    Y_full_row: np.ndarray | None = None,
) -> tuple[float, float]:
    cfg_low = float(recovery_cfg.T_min_l)
    cfg_high = float(recovery_cfg.T_max_l)
    thermo_range = _query_valid_temperature_range(liquid_thermo, y_full=Y_full_row)
    if thermo_range is None:
        return _validate_effective_temperature_bracket(cfg_low, cfg_high, label="liquid")
    low = max(cfg_low, float(thermo_range[0]))
    high = min(cfg_high, float(thermo_range[1]))
    try:
        return _validate_effective_temperature_bracket(low, high, label="liquid")
    except StateRecoveryError as exc:
        raise StateRecoveryError(
            "liquid effective bracket is empty; recovery_cfg is incompatible with thermo range"
        ) from exc


def _get_effective_gas_temperature_bounds(
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
    Y_full_row: np.ndarray | None = None,
) -> tuple[float, float]:
    cfg_low = float(recovery_cfg.T_min_g)
    cfg_high = float(recovery_cfg.T_max_g)
    thermo_range = _query_valid_temperature_range(gas_thermo, y_full=Y_full_row)
    if thermo_range is None:
        return _validate_effective_temperature_bracket(cfg_low, cfg_high, label="gas")
    low = max(cfg_low, float(thermo_range[0]))
    high = min(cfg_high, float(thermo_range[1]))
    try:
        return _validate_effective_temperature_bracket(low, high, label="gas")
    except StateRecoveryError as exc:
        raise StateRecoveryError(
            "gas effective bracket is empty; recovery_cfg is incompatible with thermo range"
        ) from exc


def _call_thermo_enthalpy(
    thermo: LiquidThermoProtocol | GasThermoProtocol,
    *,
    T: float,
    y_full: np.ndarray,
    pressure: float | None = None,
) -> float:
    try:
        return float(thermo.enthalpy_mass(T, y_full, pressure))  # type: ignore[misc]
    except TypeError:
        return float(thermo.enthalpy_mass(T, y_full))  # type: ignore[misc]


def _call_thermo_cp(
    thermo: LiquidThermoProtocol | GasThermoProtocol,
    *,
    T: float,
    y_full: np.ndarray,
    pressure: float | None = None,
) -> float:
    cp_fn = getattr(thermo, "cp_mass", None)
    if not callable(cp_fn):
        raise StateRecoveryError("thermo cp_mass is unavailable for Newton recovery")
    try:
        return float(cp_fn(T, y_full, pressure))
    except TypeError:
        return float(cp_fn(T, y_full))


def _infer_gas_recovery_pressure(gas_thermo: GasThermoProtocol) -> float | None:
    reference_pressure = getattr(gas_thermo, "reference_pressure", None)
    if reference_pressure is None:
        return None
    pressure = float(reference_pressure)
    if not np.isfinite(pressure) or pressure <= 0.0:
        return None
    return pressure


def _call_temperature_from_hpy(
    gas_thermo: GasThermoProtocol,
    *,
    h: float,
    y_full: np.ndarray,
    pressure: float,
) -> float:
    for name in ("temperature_from_hpy", "invert_h_to_T_hpy"):
        fn = getattr(gas_thermo, name, None)
        if not callable(fn):
            continue
        return float(fn(h, y_full, pressure))
    raise StateRecoveryError("gas thermo does not provide temperature_from_hpy/invert_h_to_T_hpy")


def _invert_temperature_monotone_bisection(
    *,
    target_h: float,
    y_full: np.ndarray,
    thermo: LiquidThermoProtocol | GasThermoProtocol,
    T_low: float,
    T_high: float,
    tol: float,
    max_iter: int,
    pressure: float | None = None,
) -> float:
    if not np.isfinite(target_h):
        raise StateRecoveryError("target_h must be finite")
    _validate_effective_temperature_bracket(T_low, T_high, label="scalar inversion")
    if tol <= 0.0:
        raise StateRecoveryError("tol must be > 0")
    if max_iter < 1:
        raise StateRecoveryError("max_iter must be >= 1")

    h_low = _call_thermo_enthalpy(thermo, T=float(T_low), y_full=y_full, pressure=pressure)
    h_high = _call_thermo_enthalpy(thermo, T=float(T_high), y_full=y_full, pressure=pressure)
    if not (h_low <= target_h <= h_high):
        raise StateRecoveryError("target_h must be bracketed by enthalpy values at T_low and T_high")

    left = float(T_low)
    right = float(T_high)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        h_mid = _call_thermo_enthalpy(thermo, T=mid, y_full=y_full, pressure=pressure)
        if abs(h_mid - target_h) <= tol:
            return mid
        if h_mid < target_h:
            left = mid
        else:
            right = mid

    raise StateRecoveryError("enthalpy inversion exceeded max_iter without convergence")


def _invert_temperature_safeguarded_newton(
    *,
    target_h: float,
    y_full: np.ndarray,
    thermo: LiquidThermoProtocol | GasThermoProtocol,
    T_low: float,
    T_high: float,
    tol: float,
    max_iter: int,
    pressure: float | None = None,
    T_hint: float | None = None,
) -> tuple[float, str]:
    if not np.isfinite(target_h):
        raise StateRecoveryError("target_h must be finite")
    left, right = _validate_effective_temperature_bracket(T_low, T_high, label="safeguarded inversion")
    if tol <= 0.0:
        raise StateRecoveryError("tol must be > 0")
    if max_iter < 1:
        raise StateRecoveryError("max_iter must be >= 1")

    h_left = _call_thermo_enthalpy(thermo, T=left, y_full=y_full, pressure=pressure)
    h_right = _call_thermo_enthalpy(thermo, T=right, y_full=y_full, pressure=pressure)
    if not (h_left <= target_h <= h_right):
        raise StateRecoveryError("target_h must be bracketed by enthalpy values at T_low and T_high")

    if T_hint is None or not np.isfinite(float(T_hint)):
        T_curr = 0.5 * (left + right)
    else:
        T_curr = min(max(float(T_hint), left), right)
    used_newton = False
    used_bisection = False

    for _ in range(max_iter):
        h_curr = _call_thermo_enthalpy(thermo, T=T_curr, y_full=y_full, pressure=pressure)
        residual = h_curr - target_h
        if abs(residual) <= tol:
            if used_newton and used_bisection:
                return T_curr, "newton+bisection"
            if used_newton:
                return T_curr, "newton"
            return T_curr, "bisection"

        if residual < 0.0:
            left = T_curr
        else:
            right = T_curr

        T_next: float | None = None
        try:
            cp_curr = _call_thermo_cp(thermo, T=T_curr, y_full=y_full, pressure=pressure)
        except Exception:
            cp_curr = float("nan")
        if np.isfinite(cp_curr) and cp_curr > 0.0:
            trial = T_curr - residual / cp_curr
            if np.isfinite(trial) and left < trial < right:
                T_next = trial
                used_newton = True
        if T_next is None:
            T_next = 0.5 * (left + right)
            used_bisection = True
        T_curr = T_next

    raise StateRecoveryError("enthalpy inversion exceeded max_iter without convergence")


def _invert_liquid_h_to_T_safeguarded(
    *,
    target_h: float,
    y_full: np.ndarray,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
    T_hint: float | None = None,
) -> tuple[float, str, tuple[float, float]]:
    bounds = _get_effective_liquid_temperature_bounds(recovery_cfg, liquid_thermo, y_full)
    T, mode = _invert_temperature_safeguarded_newton(
        target_h=target_h,
        y_full=y_full,
        thermo=liquid_thermo,
        T_low=bounds[0],
        T_high=bounds[1],
        tol=float(recovery_cfg.liq_h_inv_tol),
        max_iter=int(recovery_cfg.liq_h_inv_max_iter),
        T_hint=T_hint,
    )
    return T, mode, bounds


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
            _invert_liquid_h_to_T_safeguarded(
                target_h=float(h_i),
                y_full=Y_full[i, :],
                recovery_cfg=recovery_cfg,
                liquid_thermo=liquid_thermo,
            )[0]
            for i, h_i in enumerate(h)
        ],
        dtype=np.float64,
    )


def _invert_gas_h_to_T_hpy_first(
    *,
    target_h: float,
    y_full: np.ndarray,
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
    pressure: float | None,
    T_hint: float | None = None,
) -> tuple[float, str, tuple[float, float], str | None]:
    bounds = _get_effective_gas_temperature_bounds(recovery_cfg, gas_thermo, y_full)
    skipped_reason: str | None = None
    if recovery_cfg.use_cantera_hpy_first and pressure is not None:
        try:
            T_hpy = _call_temperature_from_hpy(
                gas_thermo,
                h=target_h,
                y_full=y_full,
                pressure=pressure,
            )
            if np.isfinite(T_hpy) and bounds[0] <= T_hpy <= bounds[1]:
                return float(T_hpy), "hpy", bounds, None
        except Exception:
            skipped_reason = "hpy_call_failed"
    elif recovery_cfg.use_cantera_hpy_first and pressure is None:
        skipped_reason = "missing_reference_pressure"
    T, _mode = _invert_temperature_safeguarded_newton(
        target_h=target_h,
        y_full=y_full,
        thermo=gas_thermo,
        T_low=bounds[0],
        T_high=bounds[1],
        tol=float(recovery_cfg.gas_h_inv_tol),
        max_iter=int(recovery_cfg.gas_h_inv_max_iter),
        pressure=pressure,
        T_hint=T_hint,
    )
    return T, "fallback_scalar", bounds, skipped_reason


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
    pressure = _infer_gas_recovery_pressure(gas_thermo)
    return np.array(
        [
            _invert_gas_h_to_T_hpy_first(
                target_h=float(h_i),
                y_full=Y_full[i, :],
                recovery_cfg=recovery_cfg,
                gas_thermo=gas_thermo,
                pressure=pressure,
            )[0]
            for i, h_i in enumerate(h)
        ],
        dtype=np.float64,
    )


def _recover_liquid_phase_state_with_diagnostics(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    liq_volumes = mesh.volumes[mesh.region_slices.liq]
    rho_l = _recover_density(contents.mass_l, liq_volumes)
    Yl_full = _recover_full_mass_fractions(
        contents.species_mass_l,
        contents.mass_l,
        n_full=species_maps.n_liq_full,
        single_component_name=species_maps.liq_full_names[0] if species_maps.n_liq_full == 1 else None,
    )
    hl = _recover_specific_enthalpy(contents.enthalpy_l, contents.mass_l)

    Tl = np.zeros_like(hl)
    inversion_modes: list[str] = []
    bounds_list: list[tuple[float, float]] = []
    T_hint: float | None = None
    for i, h_i in enumerate(hl):
        T_i, mode_i, bounds_i = _invert_liquid_h_to_T_safeguarded(
            target_h=float(h_i),
            y_full=Yl_full[i, :],
            recovery_cfg=recovery_cfg,
            liquid_thermo=liquid_thermo,
            T_hint=T_hint,
        )
        Tl[i] = T_i
        T_hint = T_i
        inversion_modes.append(mode_i)
        bounds_list.append(bounds_i)
    diag = {
        "liq_T_bounds_effective": bounds_list,
        "liq_inversion_mode": inversion_modes[0] if len(set(inversion_modes)) == 1 else inversion_modes,
        "n_liq_cells": int(mesh.n_liq),
        "min_Tl": float(np.min(Tl)) if Tl.size else float("nan"),
        "max_Tl": float(np.max(Tl)) if Tl.size else float("nan"),
    }
    return rho_l, Yl_full, hl, Tl, diag


def _recover_liquid_phase_state(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_l, Yl_full, hl, Tl, _ = _recover_liquid_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        liquid_thermo,
    )
    return rho_l, Yl_full, hl, Tl


def _recover_gas_phase_state_with_diagnostics(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gas_volumes = mesh.volumes[mesh.region_slices.gas_all]
    rho_g = _recover_density(contents.mass_g, gas_volumes)
    Yg_full = _recover_full_mass_fractions(
        contents.species_mass_g,
        contents.mass_g,
        n_full=species_maps.n_gas_full,
    )
    hg = _recover_specific_enthalpy(contents.enthalpy_g, contents.mass_g)

    Tg = np.zeros_like(hg)
    inversion_modes: list[str] = []
    bounds_list: list[tuple[float, float]] = []
    skipped_reasons: list[str] = []
    pressure = _infer_gas_recovery_pressure(gas_thermo)
    T_hint: float | None = None
    for i, h_i in enumerate(hg):
        T_i, mode_i, bounds_i, skipped_reason_i = _invert_gas_h_to_T_hpy_first(
            target_h=float(h_i),
            y_full=Yg_full[i, :],
            recovery_cfg=recovery_cfg,
            gas_thermo=gas_thermo,
            pressure=pressure,
            T_hint=T_hint,
        )
        Tg[i] = T_i
        T_hint = T_i
        inversion_modes.append(mode_i)
        bounds_list.append(bounds_i)
        if skipped_reason_i is not None:
            skipped_reasons.append(skipped_reason_i)
    diag = {
        "gas_T_bounds_effective": bounds_list,
        "gas_inversion_mode": inversion_modes[0] if len(set(inversion_modes)) == 1 else inversion_modes,
        "n_gas_cells": int(mesh.n_gas),
        "min_Tg": float(np.min(Tg)) if Tg.size else float("nan"),
        "max_Tg": float(np.max(Tg)) if Tg.size else float("nan"),
    }
    if skipped_reasons:
        diag["gas_hpy_skipped_reason"] = skipped_reasons[0] if len(set(skipped_reasons)) == 1 else skipped_reasons
    return rho_g, Yg_full, hg, Tg, diag


def _recover_gas_phase_state(
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    gas_thermo: GasThermoProtocol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho_g, Yg_full, hg, Tg, _ = _recover_gas_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        gas_thermo,
    )
    return rho_g, Yg_full, hg, Tg


def _copy_interface_state(interface_seed: InterfaceState) -> InterfaceState:
    return InterfaceState(
        Ts=float(interface_seed.Ts),
        mpp=float(interface_seed.mpp),
        Ys_g_full=np.array(interface_seed.Ys_g_full, dtype=np.float64, copy=True),
        Ys_l_full=np.array(interface_seed.Ys_l_full, dtype=np.float64, copy=True),
    )


def validate_recovered_state_bounds(
    state: State,
    recovery_cfg: RecoveryConfig,
    *,
    diagnostics: dict[str, object] | None = None,
) -> None:
    liq_bounds = diagnostics.get("liq_T_bounds_effective") if diagnostics is not None else None
    gas_bounds = diagnostics.get("gas_T_bounds_effective") if diagnostics is not None else None

    if liq_bounds:
        for T_i, bounds_i in zip(state.Tl, liq_bounds):
            if float(T_i) < float(bounds_i[0]) or float(T_i) > float(bounds_i[1]):
                raise StateRecoveryError("recovered state outside effective thermo bounds for liquid phase")
    elif np.any(state.Tl < recovery_cfg.T_min_l) or np.any(state.Tl > recovery_cfg.T_max_l):
        raise StateRecoveryError("recovered liquid temperature lies outside recovery bounds")

    if gas_bounds:
        for T_i, bounds_i in zip(state.Tg, gas_bounds):
            if float(T_i) < float(bounds_i[0]) or float(T_i) > float(bounds_i[1]):
                raise StateRecoveryError("recovered state outside effective thermo bounds for gas phase")
    elif np.any(state.Tg < recovery_cfg.T_min_g) or np.any(state.Tg > recovery_cfg.T_max_g):
        raise StateRecoveryError("recovered gas temperature lies outside recovery bounds")


def _recover_state_from_contents_internal(
    *,
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
    interface_seed: InterfaceState,
    time: float | None = None,
    state_id: str | None = None,
) -> tuple[State, dict[str, object]]:
    rho_l, Yl_full, hl, Tl, liq_diag = _recover_liquid_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        liquid_thermo,
    )
    rho_g, Yg_full, hg, Tg, gas_diag = _recover_gas_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        gas_thermo,
    )

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
    diagnostics = {
        **liq_diag,
        **gas_diag,
    }
    validate_recovered_state_bounds(state, recovery_cfg, diagnostics=diagnostics)
    return state, diagnostics


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
    cfg = recovery_cfg if recovery_cfg is not None else recovery_config
    if cfg is None:
        raise StateRecoveryError("recover_state_from_contents requires recovery_cfg")
    if liquid_thermo is None:
        raise StateRecoveryError("recover_state_from_contents requires liquid_thermo")
    if gas_thermo is None:
        raise StateRecoveryError("recover_state_from_contents requires gas_thermo")
    if interface_seed is None:
        raise StateRecoveryError("recover_state_from_contents requires interface_seed")

    state, _ = _recover_state_from_contents_internal(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=cfg,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        interface_seed=interface_seed,
        time=time,
        state_id=state_id,
    )
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
