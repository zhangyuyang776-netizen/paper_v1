from __future__ import annotations

from typing import Protocol

import numpy as np

from .types import (
    ConservativeContents,
    InterfaceState,
    Mesh1D,
    RecoveryConfig,
    RecoveryTemperatureSeeds,
    SpeciesMaps,
    State,
    StateRecoveryResult,
)


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
    species_recovery_eps_abs: float,
    m_min: float,
    Y_sum_tol: float,
    Y_hard_tol: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover full mass fractions from conservative species masses.

    S-3 formal contract: minor-fix / hard-fail for negative species, cfg-aware
    sum tolerance, post-norm Y range check.  Returns (Y_full, diagnostics).
    """
    species_mass = np.asarray(species_mass, dtype=np.float64).copy()
    mass = np.asarray(mass, dtype=np.float64)
    if mass.ndim != 1:
        raise StateRecoveryError("mass must be one-dimensional")
    if species_mass.ndim != 2:
        raise StateRecoveryError("species_mass must be two-dimensional")
    if species_mass.shape != (mass.shape[0], n_full):
        raise StateRecoveryError("species_mass shape must be (n_cells, n_full)")
    if np.any(mass < m_min):
        raise StateRecoveryError("mass must be >= m_min for species recovery")

    # S-3: minor-fix / hard-fail for negative species masses.
    num_minor_fixes = 0
    max_negative_species_mass = 0.0
    for i in range(species_mass.shape[0]):
        row = species_mass[i, :]
        neg_mask = row < 0.0
        if np.any(neg_mask):
            worst_neg = float(np.min(row[neg_mask]))
            if worst_neg < max_negative_species_mass:
                max_negative_species_mass = worst_neg
            if abs(worst_neg) > mass[i] * species_recovery_eps_abs:
                raise StateRecoveryError(
                    f"species_mass[{i}] has negative values exceeding "
                    f"species_recovery_eps_abs tolerance (hard-fail)"
                )
            # Minor fix: clip and renormalize so the row sum equals mass[i].
            row = np.clip(row, 0.0, None)
            row_sum = float(np.sum(row))
            if row_sum <= 0.0:
                raise StateRecoveryError(
                    f"species_mass[{i}] is all zero after minor-fix clip"
                )
            species_mass[i, :] = row * (mass[i] / row_sum)
            num_minor_fixes += 1

    species_sum = np.sum(species_mass, axis=1)
    diff = np.abs(species_sum - mass)
    max_Y_sum_err = float(np.max(diff / np.maximum(np.abs(mass), 1.0)))
    if np.any(diff > Y_sum_tol * np.maximum(np.abs(mass), 1.0)):
        raise StateRecoveryError("species_mass sums must match mass within tolerance")

    if n_full == 1:
        _ = single_component_name
        Y = np.ones((mass.shape[0], 1), dtype=np.float64)
    else:
        Y = species_mass / mass[:, None]

    # Post-norm Y range check.
    if np.any(Y < -Y_hard_tol) or np.any(Y > 1.0 + Y_hard_tol):
        raise StateRecoveryError(
            "recovered Y_full contains values outside [-Y_hard_tol, 1+Y_hard_tol]"
        )

    diag: dict[str, object] = {
        "num_minor_fixes": num_minor_fixes,
        "max_Y_sum_err": max_Y_sum_err,
        "max_negative_species_mass": max_negative_species_mass,
    }
    return Y, diag


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


def _select_initial_temperature_guess(
    *,
    target_h: float,
    bounds: tuple[float, float],
    recovery_cfg: RecoveryConfig,
    seed: float | None,
    rolling_hint: float | None,
) -> float | None:
    """Return the best T_hint using a three-level priority chain.

    Level 1: external seed (clamped to bounds if valid).
    Level 2: cp_min linear estimate T = h / cp_min (only if in-bounds).
    Level 3: rolling hint from previous cell (clamped to bounds if valid).
    Returns None when all levels are unusable (Newton will use midpoint).
    """
    T_low, T_high = bounds
    # Level 1: external seed.
    if seed is not None and np.isfinite(float(seed)):
        return min(max(float(seed), T_low), T_high)
    # Level 2: cp_min linear estimate.
    cp_min = float(recovery_cfg.cp_min)
    if np.isfinite(target_h) and cp_min > 0.0 and target_h > 0.0:
        T_est = target_h / cp_min
        if np.isfinite(T_est) and T_low <= T_est <= T_high:
            return T_est
    # Level 3: rolling hint.
    if rolling_hint is not None and np.isfinite(float(rolling_hint)):
        return min(max(float(rolling_hint), T_low), T_high)
    return None


def _recover_xg_full_best_effort(
    Yg_full: np.ndarray,
    gas_thermo: GasThermoProtocol,
) -> tuple[np.ndarray | None, str]:
    """Recover mole fractions with two-tier fallback strategy (S-5).

    Try 1: mole_fractions_from_mass (preferred, thermo-correct).
    Try 2: analytic MW conversion if molecular_weights attribute available.
    Returns (Xg_full, status) where status ∈
        {"mole_fractions_from_mass", "mw_conversion", "failed"}.
    """
    mole_fractions_fn = getattr(gas_thermo, "mole_fractions_from_mass", None)
    if callable(mole_fractions_fn):
        try:
            Xg = np.asarray(mole_fractions_fn(Yg_full), dtype=np.float64)
            if Xg.shape == Yg_full.shape and np.all(np.isfinite(Xg)) and np.all(Xg >= 0.0):
                return Xg, "mole_fractions_from_mass"
        except Exception:
            pass

    mw_attr = getattr(gas_thermo, "molecular_weights", None)
    if mw_attr is not None:
        try:
            MW = np.asarray(mw_attr() if callable(mw_attr) else mw_attr, dtype=np.float64)
            if (
                MW.ndim == 1
                and MW.shape[0] == Yg_full.shape[1]
                and np.all(MW > 0.0)
            ):
                X_unnorm = Yg_full / MW[None, :]
                X_sum = np.sum(X_unnorm, axis=1, keepdims=True)
                X_sum = np.where(X_sum <= 0.0, 1.0, X_sum)
                Xg = X_unnorm / X_sum
                if np.all(np.isfinite(Xg)) and np.all(Xg >= 0.0):
                    return Xg, "mw_conversion"
        except Exception:
            pass

    return None, "failed"


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
) -> tuple[float, str, int]:
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

    for iter_num in range(max_iter):
        h_curr = _call_thermo_enthalpy(thermo, T=T_curr, y_full=y_full, pressure=pressure)
        residual = h_curr - target_h
        if abs(residual) <= tol:
            niter = iter_num + 1
            if used_newton and used_bisection:
                return T_curr, "newton+bisection", niter
            if used_newton:
                return T_curr, "newton", niter
            return T_curr, "bisection", niter

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
) -> tuple[float, str, tuple[float, float], float, int]:
    bounds = _get_effective_liquid_temperature_bounds(recovery_cfg, liquid_thermo, y_full)
    T, mode, niter = _invert_temperature_safeguarded_newton(
        target_h=target_h,
        y_full=y_full,
        thermo=liquid_thermo,
        T_low=bounds[0],
        T_high=bounds[1],
        tol=float(recovery_cfg.h_abs_tol),
        max_iter=int(recovery_cfg.liquid_h_inv_max_iter),
        T_hint=T_hint,
    )
    # S-4: absolute forward enthalpy consistency check.
    h_fwd = _call_thermo_enthalpy(liquid_thermo, T=T, y_full=y_full)
    h_fwd_err = abs(h_fwd - target_h)
    if h_fwd_err > float(recovery_cfg.h_check_tol):
        raise StateRecoveryError(
            f"liquid forward enthalpy check failed: |h_fwd - h_target| = {h_fwd_err:.3e} "
            f"> h_check_tol = {float(recovery_cfg.h_check_tol):.3e}"
        )
    return T, mode, bounds, h_fwd_err, niter


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
) -> tuple[float, str, tuple[float, float], str | None, float, int]:
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
                # S-4: absolute forward check before accepting the HPY result.
                h_fwd = _call_thermo_enthalpy(gas_thermo, T=T_hpy, y_full=y_full, pressure=pressure)
                h_fwd_err_hpy = abs(h_fwd - target_h)
                if h_fwd_err_hpy <= float(recovery_cfg.h_check_tol):
                    return float(T_hpy), "hpy", bounds, None, h_fwd_err_hpy, 0
                else:
                    skipped_reason = "hpy_check_failed"
        except Exception:
            skipped_reason = "hpy_call_failed"
    elif recovery_cfg.use_cantera_hpy_first and pressure is None:
        skipped_reason = "missing_reference_pressure"

    # Newton fallback path.
    T, _mode, niter = _invert_temperature_safeguarded_newton(
        target_h=target_h,
        y_full=y_full,
        thermo=gas_thermo,
        T_low=bounds[0],
        T_high=bounds[1],
        tol=float(recovery_cfg.h_abs_tol),
        max_iter=int(recovery_cfg.gas_h_inv_max_iter),
        pressure=pressure,
        T_hint=T_hint,
    )
    # S-4: absolute forward enthalpy consistency check for Newton path.
    h_fwd = _call_thermo_enthalpy(gas_thermo, T=T, y_full=y_full, pressure=pressure)
    h_fwd_err = abs(h_fwd - target_h)
    if h_fwd_err > float(recovery_cfg.h_check_tol):
        raise StateRecoveryError(
            f"gas forward enthalpy check failed: |h_fwd - h_target| = {h_fwd_err:.3e} "
            f"> h_check_tol = {float(recovery_cfg.h_check_tol):.3e}"
        )
    return T, "fallback_scalar", bounds, skipped_reason, h_fwd_err, niter


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
    *,
    temperature_seeds: RecoveryTemperatureSeeds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    liq_volumes = mesh.volumes[mesh.region_slices.liq]
    rho_l = _recover_density(contents.mass_l, liq_volumes)
    Yl_full, Y_diag = _recover_full_mass_fractions(
        contents.species_mass_l,
        contents.mass_l,
        n_full=species_maps.n_liq_full,
        single_component_name=species_maps.liq_full_names[0] if species_maps.n_liq_full == 1 else None,
        species_recovery_eps_abs=float(recovery_cfg.species_recovery_eps_abs),
        m_min=float(recovery_cfg.m_min),
        Y_sum_tol=float(recovery_cfg.Y_sum_tol),
        Y_hard_tol=float(recovery_cfg.Y_hard_tol),
    )
    hl = _recover_specific_enthalpy(contents.enthalpy_l, contents.mass_l)

    Tl = np.zeros_like(hl)
    inversion_modes: list[str] = []
    bounds_list: list[tuple[float, float]] = []
    h_fwd_errs: list[float] = []
    niters: list[int] = []
    # S-2: T_hint three-level priority chain (seed > cp_min estimate > rolling).
    seed_T_l = temperature_seeds.T_l if temperature_seeds is not None else None
    T_hint_rolling: float | None = None
    for i, h_i in enumerate(hl):
        bounds_for_hint = _get_effective_liquid_temperature_bounds(
            recovery_cfg, liquid_thermo, Yl_full[i, :]
        )
        seed_i = float(seed_T_l[i]) if (seed_T_l is not None and i < len(seed_T_l)) else None
        T_hint = _select_initial_temperature_guess(
            target_h=float(h_i),
            bounds=bounds_for_hint,
            recovery_cfg=recovery_cfg,
            seed=seed_i,
            rolling_hint=T_hint_rolling,
        )
        T_i, mode_i, bounds_i, h_fwd_err_i, niter_i = _invert_liquid_h_to_T_safeguarded(
            target_h=float(h_i),
            y_full=Yl_full[i, :],
            recovery_cfg=recovery_cfg,
            liquid_thermo=liquid_thermo,
            T_hint=T_hint,
        )
        Tl[i] = T_i
        T_hint_rolling = T_i
        inversion_modes.append(mode_i)
        bounds_list.append(bounds_i)
        h_fwd_errs.append(h_fwd_err_i)
        niters.append(niter_i)
    diag: dict[str, object] = {
        "liq_recovery_success": True,
        "liq_T_bounds_effective": bounds_list,
        "liq_inversion_mode": inversion_modes[0] if len(set(inversion_modes)) == 1 else inversion_modes,
        "n_liq_cells": int(mesh.n_liq),
        "min_Tl": float(np.min(Tl)) if Tl.size else float("nan"),
        "max_Tl": float(np.max(Tl)) if Tl.size else float("nan"),
        "liq_h_fwd_check_max_err": float(max(h_fwd_errs)) if h_fwd_errs else float("nan"),
        "liq_h_inv_max_niter": int(max(niters)) if niters else 0,
        "liq_h_inv_total_niter": int(sum(niters)),
        "liq_Y_num_minor_fixes": int(Y_diag["num_minor_fixes"]),
        "liq_Y_max_sum_err": float(Y_diag["max_Y_sum_err"]),
        "liq_Y_max_negative_species_mass": float(Y_diag["max_negative_species_mass"]),
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
    *,
    gas_pressure: float | None = None,
    temperature_seeds: RecoveryTemperatureSeeds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    gas_volumes = mesh.volumes[mesh.region_slices.gas_all]
    rho_g = _recover_density(contents.mass_g, gas_volumes)
    Yg_full, Y_diag = _recover_full_mass_fractions(
        contents.species_mass_g,
        contents.mass_g,
        n_full=species_maps.n_gas_full,
        species_recovery_eps_abs=float(recovery_cfg.species_recovery_eps_abs),
        m_min=float(recovery_cfg.m_min),
        Y_sum_tol=float(recovery_cfg.Y_sum_tol),
        Y_hard_tol=float(recovery_cfg.Y_hard_tol),
    )
    hg = _recover_specific_enthalpy(contents.enthalpy_g, contents.mass_g)

    # S-5: best-effort Xg_full recovery with two-tier fallback.
    Xg_full, xg_status = _recover_xg_full_best_effort(Yg_full, gas_thermo)

    # X-Y consistency: when Xg_full is available, verify it is consistent with
    # Yg_full via the MW transformation X_i = (Y_i/MW_i) / sum(Y_j/MW_j).
    # Requires molecular_weights on gas_thermo; NaN if unavailable.
    gas_XY_consistency_err = float("nan")
    if Xg_full is not None:
        mw_attr = getattr(gas_thermo, "molecular_weights", None)
        if mw_attr is not None:
            try:
                MW = np.asarray(mw_attr() if callable(mw_attr) else mw_attr, dtype=np.float64)
                if MW.ndim == 1 and MW.shape[0] == Yg_full.shape[1] and np.all(MW > 0.0):
                    X_from_Y = Yg_full / MW[None, :]
                    X_sum = np.sum(X_from_Y, axis=1, keepdims=True)
                    X_sum = np.where(X_sum <= 0.0, 1.0, X_sum)
                    X_from_Y = X_from_Y / X_sum
                    gas_XY_consistency_err = float(np.max(np.abs(Xg_full - X_from_Y)))
            except Exception:
                pass

    Tg = np.zeros_like(hg)
    inversion_modes: list[str] = []
    bounds_list: list[tuple[float, float]] = []
    skipped_reasons: list[str | None] = []
    h_fwd_errs: list[float] = []
    niters: list[int] = []
    # S-1: use explicit gas_pressure; fall back to inference for legacy callers.
    if gas_pressure is not None and np.isfinite(gas_pressure) and gas_pressure > 0.0:
        pressure: float | None = gas_pressure
        gas_pressure_source = "explicit"
    else:
        pressure = _infer_gas_recovery_pressure(gas_thermo)
        gas_pressure_source = "inferred" if pressure is not None else "none"
    # S-2: T_hint three-level priority chain (seed > cp_min estimate > rolling).
    seed_T_g = temperature_seeds.T_g if temperature_seeds is not None else None
    T_hint_rolling: float | None = None
    for i, h_i in enumerate(hg):
        bounds_for_hint = _get_effective_gas_temperature_bounds(
            recovery_cfg, gas_thermo, Yg_full[i, :]
        )
        seed_i = float(seed_T_g[i]) if (seed_T_g is not None and i < len(seed_T_g)) else None
        T_hint = _select_initial_temperature_guess(
            target_h=float(h_i),
            bounds=bounds_for_hint,
            recovery_cfg=recovery_cfg,
            seed=seed_i,
            rolling_hint=T_hint_rolling,
        )
        T_i, mode_i, bounds_i, skipped_reason_i, h_fwd_err_i, niter_i = _invert_gas_h_to_T_hpy_first(
            target_h=float(h_i),
            y_full=Yg_full[i, :],
            recovery_cfg=recovery_cfg,
            gas_thermo=gas_thermo,
            pressure=pressure,
            T_hint=T_hint,
        )
        Tg[i] = T_i
        T_hint_rolling = T_i
        inversion_modes.append(mode_i)
        bounds_list.append(bounds_i)
        skipped_reasons.append(skipped_reason_i)
        h_fwd_errs.append(h_fwd_err_i)
        niters.append(niter_i)

    non_none_skips = [r for r in skipped_reasons if r is not None]
    diag: dict[str, object] = {
        "gas_recovery_success": True,
        "gas_T_bounds_effective": bounds_list,
        "gas_inversion_mode": inversion_modes[0] if len(set(inversion_modes)) == 1 else inversion_modes,
        "n_gas_cells": int(mesh.n_gas),
        "min_Tg": float(np.min(Tg)) if Tg.size else float("nan"),
        "max_Tg": float(np.max(Tg)) if Tg.size else float("nan"),
        "gas_h_fwd_check_max_err": float(max(h_fwd_errs)) if h_fwd_errs else float("nan"),
        "gas_h_inv_max_niter": int(max(niters)) if niters else 0,
        "gas_h_inv_total_niter": int(sum(niters)),
        "gas_Y_num_minor_fixes": int(Y_diag["num_minor_fixes"]),
        "gas_Y_max_sum_err": float(Y_diag["max_Y_sum_err"]),
        "gas_Y_max_negative_species_mass": float(Y_diag["max_negative_species_mass"]),
        "gas_recovery_used_HPY": any(m == "hpy" for m in inversion_modes),
        "gas_recovery_used_fallback": any(m != "hpy" for m in inversion_modes),
        "gas_pressure_source": gas_pressure_source,
        "Xg_full_recovery_status": xg_status,
        "gas_XY_consistency_err": gas_XY_consistency_err,
        "Xg_full": Xg_full,
        "gas_hpy_skipped_reason": (
            non_none_skips[0] if len(set(non_none_skips)) == 1 else (non_none_skips if non_none_skips else None)
        ),
    }
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


def validate_recovered_state_postchecks(
    state: State,
    recovery_cfg: RecoveryConfig,
    *,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Unified post-check: T bounds, rho_min, Y range/sum, Xg validity.

    Raises StateRecoveryError on any violation; returns a dict of post-check
    diagnostics that the caller should merge into the main diagnostics dict.
    """
    liq_bounds = diagnostics.get("liq_T_bounds_effective") if diagnostics is not None else None
    gas_bounds = diagnostics.get("gas_T_bounds_effective") if diagnostics is not None else None

    # T bounds.
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

    rho_min = float(recovery_cfg.rho_min)
    Y_hard_tol = float(recovery_cfg.Y_hard_tol)
    Y_sum_tol = float(recovery_cfg.Y_sum_tol)

    # rho_min check.
    min_rho_l = float(np.min(state.rho_l)) if state.rho_l is not None else float("nan")
    min_rho_g = float(np.min(state.rho_g)) if state.rho_g is not None else float("nan")
    if state.rho_l is not None and np.any(state.rho_l < rho_min):
        raise StateRecoveryError(f"recovered liquid density below rho_min ({rho_min:.3e})")
    if state.rho_g is not None and np.any(state.rho_g < rho_min):
        raise StateRecoveryError(f"recovered gas density below rho_min ({rho_min:.3e})")

    # Y range and row-sum checks.
    liq_Y_max_sum_err = float("nan")
    gas_Y_max_sum_err = float("nan")
    gas_X_max_sum_err = float("nan")

    if state.Yl_full is not None:
        if np.any(state.Yl_full < -Y_hard_tol) or np.any(state.Yl_full > 1.0 + Y_hard_tol):
            raise StateRecoveryError("recovered Yl_full contains values outside Y_hard_tol bounds")
        Y_sums = np.sum(state.Yl_full, axis=1)
        liq_Y_max_sum_err = float(np.max(np.abs(Y_sums - 1.0)))
        if liq_Y_max_sum_err > Y_sum_tol:
            raise StateRecoveryError(
                f"recovered Yl_full row sums deviate from 1 by {liq_Y_max_sum_err:.3e}"
            )

    if state.Yg_full is not None:
        if np.any(state.Yg_full < -Y_hard_tol) or np.any(state.Yg_full > 1.0 + Y_hard_tol):
            raise StateRecoveryError("recovered Yg_full contains values outside Y_hard_tol bounds")
        Y_sums = np.sum(state.Yg_full, axis=1)
        gas_Y_max_sum_err = float(np.max(np.abs(Y_sums - 1.0)))
        if gas_Y_max_sum_err > Y_sum_tol:
            raise StateRecoveryError(
                f"recovered Yg_full row sums deviate from 1 by {gas_Y_max_sum_err:.3e}"
            )

    # Xg_full validity (finite, non-negative, sum ≈ 1).
    if state.Xg_full is not None:
        if not np.all(np.isfinite(state.Xg_full)):
            raise StateRecoveryError("recovered Xg_full contains non-finite values")
        if np.any(state.Xg_full < 0.0):
            raise StateRecoveryError("recovered Xg_full contains negative values")
        X_sums = np.sum(state.Xg_full, axis=1)
        gas_X_max_sum_err = float(np.max(np.abs(X_sums - 1.0)))
        if gas_X_max_sum_err > Y_sum_tol:
            raise StateRecoveryError(
                f"recovered Xg_full row sums deviate from 1 by {gas_X_max_sum_err:.3e}"
            )

    # X-Y consistency check: Xg_full must agree with Yg_full via MW transformation.
    postcheck_gas_XY_consistency_err = float("nan")
    if diagnostics is not None:
        xy_err_raw = diagnostics.get("gas_XY_consistency_err")
        if xy_err_raw is not None:
            try:
                xy_val = float(xy_err_raw)
                if not np.isnan(xy_val):
                    postcheck_gas_XY_consistency_err = xy_val
                    if xy_val > Y_sum_tol:
                        raise StateRecoveryError(
                            f"Xg_full / Yg_full X-Y consistency error {xy_val:.3e} "
                            f"exceeds Y_sum_tol {Y_sum_tol:.3e}"
                        )
            except (TypeError, ValueError):
                pass

    return {
        "postchecks_passed": True,
        "min_rho_l": min_rho_l,
        "min_rho_g": min_rho_g,
        "liq_Y_max_sum_err": liq_Y_max_sum_err,
        "gas_Y_max_sum_err": gas_Y_max_sum_err,
        "gas_X_max_sum_err": gas_X_max_sum_err,
        "postcheck_gas_XY_consistency_err": postcheck_gas_XY_consistency_err,
    }


def _recover_state_from_contents_internal(
    *,
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
    interface_seed: InterfaceState,
    gas_pressure: float | None = None,
    temperature_seeds: RecoveryTemperatureSeeds | None = None,
    time: float | None = None,
    state_id: str | None = None,
) -> tuple[State, dict[str, object]]:
    rho_l, Yl_full, hl, Tl, liq_diag = _recover_liquid_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        liquid_thermo,
        temperature_seeds=temperature_seeds,
    )
    rho_g, Yg_full, hg, Tg, gas_diag = _recover_gas_phase_state_with_diagnostics(
        contents,
        mesh,
        species_maps,
        recovery_cfg,
        gas_thermo,
        gas_pressure=gas_pressure,
        temperature_seeds=temperature_seeds,
    )

    # Extract Xg_full from gas diagnostics; keep status in diag.
    Xg_full: np.ndarray | None = gas_diag.pop("Xg_full", None)  # type: ignore[assignment]

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
        Xg_full=Xg_full,
        time=time,
        state_id=state_id,
    )
    diagnostics: dict[str, object] = {
        **liq_diag,
        **gas_diag,
    }
    post_diag = validate_recovered_state_postchecks(state, recovery_cfg, diagnostics=diagnostics)
    diagnostics.update(post_diag)
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


def recover_state_from_contents_detailed(
    *,
    contents: ConservativeContents,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    recovery_cfg: RecoveryConfig,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
    interface_seed: InterfaceState,
    gas_pressure: float,
    temperature_seeds: RecoveryTemperatureSeeds | None = None,
    time: float | None = None,
    state_id: str | None = None,
) -> StateRecoveryResult:
    """Recover primitive state and return full per-phase diagnostics.

    Unlike :func:`recover_state_from_contents`, this function requires an
    explicit ``gas_pressure`` (S-1) and returns a :class:`StateRecoveryResult`
    that exposes the full diagnostics dict alongside the recovered state.
    ``temperature_seeds`` may be supplied to seed the enthalpy inversion with
    temperature hints from the previous timestep (S-2).
    """
    # S-1: strict validation — gas_pressure must be finite and positive.
    if not np.isfinite(float(gas_pressure)) or float(gas_pressure) <= 0.0:
        raise StateRecoveryError(
            f"gas_pressure must be finite and positive, got {gas_pressure!r}"
        )
    state, diagnostics = _recover_state_from_contents_internal(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=recovery_cfg,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        interface_seed=interface_seed,
        gas_pressure=gas_pressure,
        temperature_seeds=temperature_seeds,
        time=time,
        state_id=state_id,
    )
    # S-5: Xg_full must be recoverable; silent failure is not allowed here.
    if state.Xg_full is None:
        raise StateRecoveryError(
            "Xg_full could not be recovered from gas_thermo; "
            "gas_thermo must provide mole_fractions_from_mass or molecular_weights"
        )
    return StateRecoveryResult(state=state, diagnostics=diagnostics)


def summarize_recovery_diagnostics(
    state_or_result: "State | StateRecoveryResult",
) -> dict[str, object]:
    """Return a flat summary dict from a recovered state or detailed result.

    When passed a :class:`StateRecoveryResult` the summary is enriched with
    inversion-mode and HPY diagnostics available from the detailed recovery.
    """
    if isinstance(state_or_result, StateRecoveryResult):
        state = state_or_result.state
        diag: dict[str, object] = state_or_result.diagnostics
    else:
        state = state_or_result
        diag = {}
    result: dict[str, object] = {
        "min_Tl": float(np.min(state.Tl)),
        "max_Tl": float(np.max(state.Tl)),
        "min_Tg": float(np.min(state.Tg)),
        "max_Tg": float(np.max(state.Tg)),
        "min_rho_l": float(np.min(state.rho_l)) if state.rho_l is not None else float("nan"),
        "min_rho_g": float(np.min(state.rho_g)) if state.rho_g is not None else float("nan"),
    }
    # Pass through all formal S-7 diagnostic keys when available (detailed recovery only).
    _passthrough_keys = (
        # Phase recovery success flags.
        "liq_recovery_success",
        "gas_recovery_success",
        # Forward enthalpy check.
        "liq_h_fwd_check_max_err",
        "gas_h_fwd_check_max_err",
        # Species mass fraction diagnostics.
        "liq_Y_num_minor_fixes",
        "liq_Y_max_sum_err",
        "liq_Y_max_negative_species_mass",
        "gas_Y_num_minor_fixes",
        "gas_Y_max_sum_err",
        "gas_Y_max_negative_species_mass",
        # Inversion mode and HPY path.
        "liq_inversion_mode",
        "gas_inversion_mode",
        "gas_recovery_used_HPY",
        "gas_recovery_used_fallback",
        "gas_hpy_skipped_reason",
        # Pressure and mole-fraction recovery.
        "gas_pressure_source",
        "Xg_full_recovery_status",
        # Enthalpy inversion iteration counts.
        "liq_h_inv_max_niter",
        "liq_h_inv_total_niter",
        "gas_h_inv_max_niter",
        "gas_h_inv_total_niter",
        # X-Y consistency error.
        "gas_XY_consistency_err",
        # Post-check results.
        "postchecks_passed",
        # Cell counts.
        "n_liq_cells",
        "n_gas_cells",
    )
    for key in _passthrough_keys:
        if key in diag:
            result[key] = diag[key]
    return result


__all__ = [
    "GasThermoProtocol",
    "LiquidThermoProtocol",
    "StateRecoveryError",
    "recover_state_from_contents",
    "recover_state_from_contents_detailed",
    "summarize_recovery_diagnostics",
]
