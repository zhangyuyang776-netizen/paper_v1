from __future__ import annotations

"""Unique interface-face truth source for paper_v1.

This module constructs the single InterfaceFacePackage consumed by
interface_mass.py / interface_energy.py. It may evaluate interface properties,
one-sided gradients, diffusive fluxes, total species fluxes, heat fluxes, and
total energy fluxes, but it must not form interface residual rows or update
dot_a / interface unknowns.

Forbidden here:
- forming Eq.(2.15)~Eq.(2.19) residuals
- hard-pinning Ts / mpp / Ys
- treating the interface as an ordinary internal face
- dropping the species-diffusion enthalpy term
- using reduced species instead of full-order gas composition
- updating dot_a or introducing regime switching
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import Mesh1D, RunConfig, State
from physics.energy_flux import compute_total_energy_flux_density
from properties.equilibrium import compute_interface_equilibrium


class InterfaceFaceError(ValueError):
    """Raised when the interface-face package cannot be built consistently."""


@dataclass(frozen=True)
class InterfaceFacePackage:
    r_s: float
    area_s: float
    dr_l_s: float
    dr_g_s: float
    dot_a_frozen: float

    Tl_last: float
    Tg_first: float
    Yl_last_full: np.ndarray
    Yg_first_full: np.ndarray
    Xg_first_full: np.ndarray

    Ts: float
    Ys_l_full: np.ndarray
    Ys_g_full: np.ndarray
    Xs_g_full: np.ndarray
    mpp: float

    rho_s_l: float
    rho_s_g: float
    h_s_l: float
    h_s_g: float
    h_liq_species_s_full: np.ndarray
    h_gas_species_s_full: np.ndarray
    k_s_l: float
    k_s_g: float
    D_s_l_full: np.ndarray
    D_s_g_full: np.ndarray

    dTdr_l_s: float
    dTdr_g_s: float
    dYdr_l_s_full: np.ndarray
    dXdr_g_s_full: np.ndarray

    J_l_full: np.ndarray
    J_g_full: np.ndarray
    Vd0_g_full: np.ndarray
    Vcd_g: float

    N_l_full: np.ndarray
    N_g_full: np.ndarray

    q_l_s: float
    q_g_s: float
    q_l_cond: float
    q_g_cond: float
    q_l_species_diff: float
    q_g_species_diff: float

    E_l_s: float
    E_g_s: float

    G_g_if_abs: float

    Yeq_g_cond_full: np.ndarray | None
    # liquid full-species ordering, not gas full ordering
    gamma_cond_full: np.ndarray | None
    eps_mass_gas_kinematic: float


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise InterfaceFaceError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise InterfaceFaceError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise InterfaceFaceError(f"{name} must have length {expected_size}")
    return arr


def _validate_positive_scalar(name: str, value: float, *, allow_zero: bool = False) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise InterfaceFaceError(f"{name} must be finite")
    if allow_zero:
        if scalar < 0.0:
            raise InterfaceFaceError(f"{name} must be >= 0")
    else:
        if scalar <= 0.0:
            raise InterfaceFaceError(f"{name} must be > 0")
    return scalar


def _validate_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise InterfaceFaceError(f"{name} must be finite")
    return scalar


def _validate_mass_fraction_vector(name: str, value: Any, expected_size: int) -> np.ndarray:
    arr = _as_1d_float_array(name, value, expected_size=expected_size)
    if np.any(arr < 0.0):
        raise InterfaceFaceError(f"{name} must be non-negative")
    if not np.isclose(float(np.sum(arr)), 1.0, rtol=0.0, atol=1.0e-12):
        raise InterfaceFaceError(f"{name} must sum to 1 within tolerance")
    return arr


def _mass_to_mole_fractions_gas(Y_full: np.ndarray, gas_props: Any) -> np.ndarray:
    mw = _as_1d_float_array("gas molecular weights", gas_props.molecular_weights, expected_size=Y_full.shape[0])
    if np.any(mw <= 0.0):
        raise InterfaceFaceError("gas molecular weights must be strictly positive")
    mole_basis = Y_full / mw
    denom = float(np.sum(mole_basis))
    if not np.isfinite(denom) or denom <= 0.0:
        raise InterfaceFaceError("gas mole-fraction denominator must be finite and strictly positive")
    X = mole_basis / denom
    if not np.isclose(float(np.sum(X)), 1.0, rtol=0.0, atol=1.0e-12):
        raise InterfaceFaceError("gas mole fractions must sum to 1")
    return X


def _extract_interface_geometry(
    mesh: Mesh1D,
    dot_a_frozen: float,
) -> tuple[float, float, float, float]:
    _validate_finite_scalar("dot_a_frozen", dot_a_frozen)
    if mesh.interface_face_index is None or mesh.interface_cell_liq is None or mesh.interface_cell_gas is None:
        raise InterfaceFaceError("mesh must define interface face and neighboring cell indices")
    r_s = float(mesh.r_faces[mesh.interface_face_index])
    area_s = float(mesh.face_areas[mesh.interface_face_index])
    dr_l_s = r_s - float(mesh.r_centers[mesh.interface_cell_liq])
    dr_g_s = float(mesh.r_centers[mesh.interface_cell_gas]) - r_s
    _validate_positive_scalar("r_s", r_s)
    _validate_positive_scalar("area_s", area_s)
    _validate_positive_scalar("dr_l_s", dr_l_s)
    _validate_positive_scalar("dr_g_s", dr_g_s)
    return r_s, area_s, dr_l_s, dr_g_s


def _validate_interface_state_shapes(state: State, run_cfg: RunConfig) -> None:
    if state.n_liq_species_full != run_cfg.species_maps.n_liq_full:
        raise InterfaceFaceError("state liquid full-species dimension must match species_maps")
    if state.n_gas_species_full != run_cfg.species_maps.n_gas_full:
        raise InterfaceFaceError("state gas full-species dimension must match species_maps")
    _validate_mass_fraction_vector("state.interface.Ys_l_full", state.interface.Ys_l_full, state.n_liq_species_full)
    _validate_mass_fraction_vector("state.interface.Ys_g_full", state.interface.Ys_g_full, state.n_gas_species_full)
    _validate_mass_fraction_vector("state.Yl_full[last]", state.Yl_full[-1, :], state.n_liq_species_full)
    _validate_mass_fraction_vector("state.Yg_full[first]", state.Yg_full[0, :], state.n_gas_species_full)


def _build_interface_liquid_properties(
    Ts: float,
    Ys_l_full: np.ndarray,
    liquid_props: Any,
) -> tuple[float, float, np.ndarray, float, np.ndarray]:
    rho_s_l = float(liquid_props.density_mass(Ts, Ys_l_full))
    h_s_l = float(liquid_props.enthalpy_mass(Ts, Ys_l_full))
    h_liq_species_s_full = _as_1d_float_array(
        "h_liq_species_s_full",
        liquid_props.pure_enthalpy_vector(Ts),
        expected_size=Ys_l_full.shape[0],
    )
    k_s_l = float(liquid_props.conductivity(Ts, Ys_l_full))
    D_l = liquid_props.diffusivity(Ts, Ys_l_full)
    if D_l is None:
        if Ys_l_full.shape[0] == 1:
            D_s_l_full = np.zeros(1, dtype=np.float64)
        else:
            raise InterfaceFaceError("multi-component liquid requires interface diffusivity from liquid_props")
    else:
        D_s_l_full = _as_1d_float_array("D_s_l_full", D_l, expected_size=Ys_l_full.shape[0])
    _validate_positive_scalar("rho_s_l", rho_s_l)
    _validate_positive_scalar("k_s_l", k_s_l)
    return rho_s_l, h_s_l, h_liq_species_s_full, k_s_l, D_s_l_full


def _build_interface_gas_properties(
    Ts: float,
    Ys_g_full: np.ndarray,
    gas_props: Any,
    p_inf: float,
) -> tuple[float, float, np.ndarray, float, np.ndarray]:
    rho_s_g = float(gas_props.density_mass(Ts, Ys_g_full, p_inf))
    h_s_g = float(gas_props.enthalpy_mass(Ts, Ys_g_full, p_inf))
    h_gas_species_s_full = _as_1d_float_array(
        "h_gas_species_s_full",
        gas_props.species_enthalpies_mass(Ts),
        expected_size=Ys_g_full.shape[0],
    )
    k_s_g = float(gas_props.conductivity(Ts, Ys_g_full, p_inf))
    D_s_g_full = _as_1d_float_array(
        "D_s_g_full",
        gas_props.diffusivity(Ts, Ys_g_full, p_inf),
        expected_size=Ys_g_full.shape[0],
    )
    _validate_positive_scalar("rho_s_g", rho_s_g)
    _validate_positive_scalar("k_s_g", k_s_g)
    if np.any(D_s_g_full <= 0.0):
        raise InterfaceFaceError("D_s_g_full must be strictly positive")
    return rho_s_g, h_s_g, h_gas_species_s_full, k_s_g, D_s_g_full


def _build_interface_one_sided_gradients(
    Ts: float,
    Tl_last: float,
    Tg_first: float,
    Ys_l_full: np.ndarray,
    Yl_last_full: np.ndarray,
    Xs_g_full: np.ndarray,
    Xg_first_full: np.ndarray,
    dr_l_s: float,
    dr_g_s: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    dTdr_l_s = (Ts - Tl_last) / dr_l_s
    dTdr_g_s = (Tg_first - Ts) / dr_g_s
    dYdr_l_s_full = (Ys_l_full - Yl_last_full) / dr_l_s
    dXdr_g_s_full = (Xg_first_full - Xs_g_full) / dr_g_s
    return dTdr_l_s, dTdr_g_s, dYdr_l_s_full, dXdr_g_s_full


def _build_interface_liquid_diffusive_flux(
    rho_s_l: float,
    D_s_l_full: np.ndarray,
    dYdr_l_s_full: np.ndarray,
    liquid_closure_index: int | None,
) -> np.ndarray:
    n_spec = D_s_l_full.shape[0]
    if n_spec == 1:
        return np.zeros(1, dtype=np.float64)
    if liquid_closure_index is None or not (0 <= liquid_closure_index < n_spec):
        raise InterfaceFaceError("multi-component liquid interface flux requires a valid closure index")
    nonclosure = [idx for idx in range(n_spec) if idx != liquid_closure_index]
    J = np.zeros(n_spec, dtype=np.float64)
    J[nonclosure] = -rho_s_l * D_s_l_full[nonclosure] * dYdr_l_s_full[nonclosure]
    J[liquid_closure_index] = -float(np.sum(J[nonclosure]))
    return J


def _build_interface_gas_diffusive_flux(
    rho_s_g: float,
    Ys_g_full: np.ndarray,
    Xs_g_full: np.ndarray,
    D_s_g_full: np.ndarray,
    dXdr_g_s_full: np.ndarray,
    x_eps: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_safe = np.maximum(Xs_g_full, x_eps)
    Vd0_g_full = -(D_s_g_full / x_safe) * dXdr_g_s_full
    Vcd_g = -float(np.sum(Ys_g_full * Vd0_g_full))
    J_g_full = -rho_s_g * Ys_g_full * (Vd0_g_full + Vcd_g)
    return J_g_full, Vd0_g_full, Vcd_g


def _build_total_species_fluxes(
    mpp: float,
    Ys_l_full: np.ndarray,
    Ys_g_full: np.ndarray,
    J_l_full: np.ndarray,
    J_g_full: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    N_l_full = -mpp * Ys_l_full + J_l_full
    N_g_full = -mpp * Ys_g_full + J_g_full
    return N_l_full, N_g_full


def _build_interface_heat_fluxes(
    k_s_l: float,
    k_s_g: float,
    dTdr_l_s: float,
    dTdr_g_s: float,
    J_l_full: np.ndarray,
    J_g_full: np.ndarray,
    h_liq_species_s_full: np.ndarray,
    h_gas_species_s_full: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    q_l_cond, q_l_species_diff, q_l_total = compute_total_energy_flux_density(
        np.array([k_s_l], dtype=np.float64),
        np.array([dTdr_l_s], dtype=np.float64),
        J_l_full[None, :],
        h_liq_species_s_full[None, :],
    )
    q_g_cond, q_g_species_diff, q_g_total = compute_total_energy_flux_density(
        np.array([k_s_g], dtype=np.float64),
        np.array([dTdr_g_s], dtype=np.float64),
        J_g_full[None, :],
        h_gas_species_s_full[None, :],
    )
    return (
        float(q_l_total[0]),
        float(q_g_total[0]),
        float(q_l_cond[0]),
        float(q_g_cond[0]),
        float(q_l_species_diff[0]),
        float(q_g_species_diff[0]),
    )


def _build_interface_total_energy_fluxes(
    mpp: float,
    h_s_l: float,
    h_s_g: float,
    q_l_s: float,
    q_g_s: float,
) -> tuple[float, float]:
    return (-mpp * h_s_l + q_l_s, -mpp * h_s_g + q_g_s)


def _build_gas_interface_absolute_mass_flux(
    rho_s_g: float,
    dot_a_frozen: float,
    mpp: float,
    area_s: float,
) -> float:
    return (rho_s_g * dot_a_frozen - mpp) * area_s


def _build_equilibrium_diagnostics(
    Ts: float,
    Ys_l_full: np.ndarray,
    equilibrium_model: Any,
    p_inf: float,
    n_gas_full: int,
    n_liq_full: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if equilibrium_model is None:
        return None, None
    eq_result = compute_interface_equilibrium(
        equilibrium_model,
        Ts=Ts,
        P=p_inf,
        Yl_if_full=Ys_l_full,
    )
    Yeq_g_cond_full = _as_1d_float_array("Yeq_g_cond_full", eq_result.Yg_eq_full, expected_size=n_gas_full).copy()
    gamma_cond_full = np.zeros(n_liq_full, dtype=np.float64)
    if hasattr(equilibrium_model, "liquid_cond_indices"):
        gamma_cond_full[np.asarray(equilibrium_model.liquid_cond_indices, dtype=np.int64)] = _as_1d_float_array(
            "gamma_cond",
            eq_result.gamma_cond,
        )
    else:
        gamma = _as_1d_float_array("gamma_cond", eq_result.gamma_cond, expected_size=n_liq_full)
        gamma_cond_full[:] = gamma
    return Yeq_g_cond_full, gamma_cond_full


def build_interface_face_package(
    run_cfg: RunConfig,
    mesh: Mesh1D,
    state: State,
    gas_props: Any,
    liquid_props: Any,
    equilibrium_model: Any,
    dot_a_frozen: float,
) -> InterfaceFacePackage:
    _validate_interface_state_shapes(state, run_cfg)
    # RunConfig.pressure is the normalized core alias for initialization.gas_pressure.
    p_inf = float(run_cfg.pressure)
    Ts = _validate_positive_scalar("state.interface.Ts", state.interface.Ts)
    mpp = float(state.interface.mpp)
    if not np.isfinite(mpp):
        raise InterfaceFaceError("state.interface.mpp must be finite")

    r_s, area_s, dr_l_s, dr_g_s = _extract_interface_geometry(mesh, dot_a_frozen)
    Tl_last = _validate_positive_scalar("Tl_last", state.Tl[-1])
    Tg_first = _validate_positive_scalar("Tg_first", state.Tg[0])
    Yl_last_full = _validate_mass_fraction_vector("Yl_last_full", state.Yl_full[-1, :], state.n_liq_species_full)
    Yg_first_full = _validate_mass_fraction_vector("Yg_first_full", state.Yg_full[0, :], state.n_gas_species_full)
    Ys_l_full = _validate_mass_fraction_vector("Ys_l_full", state.interface.Ys_l_full, state.n_liq_species_full)
    Ys_g_full = _validate_mass_fraction_vector("Ys_g_full", state.interface.Ys_g_full, state.n_gas_species_full)
    Xg_first_full = _mass_to_mole_fractions_gas(Yg_first_full, gas_props)
    Xs_g_full = _mass_to_mole_fractions_gas(Ys_g_full, gas_props)

    rho_s_l, h_s_l, h_liq_species_s_full, k_s_l, D_s_l_full = _build_interface_liquid_properties(
        Ts, Ys_l_full, liquid_props
    )
    rho_s_g, h_s_g, h_gas_species_s_full, k_s_g, D_s_g_full = _build_interface_gas_properties(
        Ts, Ys_g_full, gas_props, p_inf
    )

    dTdr_l_s, dTdr_g_s, dYdr_l_s_full, dXdr_g_s_full = _build_interface_one_sided_gradients(
        Ts=Ts,
        Tl_last=Tl_last,
        Tg_first=Tg_first,
        Ys_l_full=Ys_l_full,
        Yl_last_full=Yl_last_full,
        Xs_g_full=Xs_g_full,
        Xg_first_full=Xg_first_full,
        dr_l_s=dr_l_s,
        dr_g_s=dr_g_s,
    )

    liquid_closure_index = None
    if run_cfg.species_maps.liq_closure_name is not None:
        liquid_closure_index = run_cfg.species_maps.liq_full_names.index(run_cfg.species_maps.liq_closure_name)
    J_l_full = _build_interface_liquid_diffusive_flux(
        rho_s_l=rho_s_l,
        D_s_l_full=D_s_l_full,
        dYdr_l_s_full=dYdr_l_s_full,
        liquid_closure_index=liquid_closure_index,
    )
    J_g_full, Vd0_g_full, Vcd_g = _build_interface_gas_diffusive_flux(
        rho_s_g=rho_s_g,
        Ys_g_full=Ys_g_full,
        Xs_g_full=Xs_g_full,
        D_s_g_full=D_s_g_full,
        dXdr_g_s_full=dXdr_g_s_full,
    )
    if not np.allclose(np.sum(J_l_full), 0.0, rtol=0.0, atol=1.0e-10):
        raise InterfaceFaceError("liquid interface diffusive flux must satisfy sum_i J_l_i = 0")
    if not np.allclose(np.sum(J_g_full), 0.0, rtol=0.0, atol=1.0e-10):
        raise InterfaceFaceError("gas interface diffusive flux must satisfy sum_i J_g_i = 0")

    N_l_full, N_g_full = _build_total_species_fluxes(
        mpp=mpp,
        Ys_l_full=Ys_l_full,
        Ys_g_full=Ys_g_full,
        J_l_full=J_l_full,
        J_g_full=J_g_full,
    )
    q_l_s, q_g_s, q_l_cond, q_g_cond, q_l_species_diff, q_g_species_diff = _build_interface_heat_fluxes(
        k_s_l=k_s_l,
        k_s_g=k_s_g,
        dTdr_l_s=dTdr_l_s,
        dTdr_g_s=dTdr_g_s,
        J_l_full=J_l_full,
        J_g_full=J_g_full,
        h_liq_species_s_full=h_liq_species_s_full,
        h_gas_species_s_full=h_gas_species_s_full,
    )
    E_l_s, E_g_s = _build_interface_total_energy_fluxes(
        mpp=mpp,
        h_s_l=h_s_l,
        h_s_g=h_s_g,
        q_l_s=q_l_s,
        q_g_s=q_g_s,
    )
    G_g_if_abs = _build_gas_interface_absolute_mass_flux(
        rho_s_g=rho_s_g,
        dot_a_frozen=float(dot_a_frozen),
        mpp=mpp,
        area_s=area_s,
    )
    Yeq_g_cond_full, gamma_cond_full = _build_equilibrium_diagnostics(
        Ts=Ts,
        Ys_l_full=Ys_l_full,
        equilibrium_model=equilibrium_model,
        p_inf=p_inf,
        n_gas_full=state.n_gas_species_full,
        n_liq_full=state.n_liq_species_full,
    )
    eps_mass_gas_kinematic = float(np.sum(N_g_full) + mpp)

    return InterfaceFacePackage(
        r_s=r_s,
        area_s=area_s,
        dr_l_s=dr_l_s,
        dr_g_s=dr_g_s,
        dot_a_frozen=float(dot_a_frozen),
        Tl_last=Tl_last,
        Tg_first=Tg_first,
        Yl_last_full=Yl_last_full.copy(),
        Yg_first_full=Yg_first_full.copy(),
        Xg_first_full=Xg_first_full.copy(),
        Ts=Ts,
        Ys_l_full=Ys_l_full.copy(),
        Ys_g_full=Ys_g_full.copy(),
        Xs_g_full=Xs_g_full.copy(),
        mpp=mpp,
        rho_s_l=rho_s_l,
        rho_s_g=rho_s_g,
        h_s_l=h_s_l,
        h_s_g=h_s_g,
        h_liq_species_s_full=h_liq_species_s_full.copy(),
        h_gas_species_s_full=h_gas_species_s_full.copy(),
        k_s_l=k_s_l,
        k_s_g=k_s_g,
        D_s_l_full=D_s_l_full.copy(),
        D_s_g_full=D_s_g_full.copy(),
        dTdr_l_s=dTdr_l_s,
        dTdr_g_s=dTdr_g_s,
        dYdr_l_s_full=dYdr_l_s_full.copy(),
        dXdr_g_s_full=dXdr_g_s_full.copy(),
        J_l_full=J_l_full.copy(),
        J_g_full=J_g_full.copy(),
        Vd0_g_full=Vd0_g_full.copy(),
        Vcd_g=Vcd_g,
        N_l_full=N_l_full.copy(),
        N_g_full=N_g_full.copy(),
        q_l_s=q_l_s,
        q_g_s=q_g_s,
        q_l_cond=q_l_cond,
        q_g_cond=q_g_cond,
        q_l_species_diff=q_l_species_diff,
        q_g_species_diff=q_g_species_diff,
        E_l_s=E_l_s,
        E_g_s=E_g_s,
        G_g_if_abs=G_g_if_abs,
        Yeq_g_cond_full=None if Yeq_g_cond_full is None else Yeq_g_cond_full.copy(),
        gamma_cond_full=None if gamma_cond_full is None else gamma_cond_full.copy(),
        eps_mass_gas_kinematic=eps_mass_gas_kinematic,
    )


__all__ = [
    "InterfaceFaceError",
    "InterfaceFacePackage",
    "build_interface_face_package",
]
