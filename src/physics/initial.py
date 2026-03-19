from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Any

import numpy as np

from core.layout import UnknownLayout
from core.types import InterfaceState, Mesh1D, RunConfig, State


class InitialStateError(ValueError):
    """Raised when the project-rule initial state cannot be constructed safely."""


@dataclass(frozen=True)
class InitialBuildInfo:
    t0_smooth: float
    rho_g_inf: float
    cp_g_inf: float
    k_g_inf: float
    D_T_g_inf: float
    Yg_inf_full: np.ndarray
    Ys_g_full0: np.ndarray
    notes: tuple[str, ...]


@dataclass(frozen=True)
class InitialStateBundle:
    state0: State
    dot_a0: float
    info: InitialBuildInfo


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise InitialStateError(f"{name} must be a 1D float array")
    if arr.size == 0:
        raise InitialStateError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise InitialStateError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise InitialStateError(f"{name} must have length {expected_size}")
    return arr


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise InitialStateError(f"{name} must be finite and > 0")
    return scalar


def _validate_mass_fraction_vector(
    name: str,
    value: np.ndarray,
    *,
    expected_size: int,
    atol: float = 1.0e-12,
) -> np.ndarray:
    arr = _as_1d_float_array(name, value, expected_size=expected_size)
    if np.any(arr < -atol):
        raise InitialStateError(f"{name} must be non-negative")
    total = float(np.sum(arr))
    if not np.isclose(total, 1.0, rtol=0.0, atol=atol):
        raise InitialStateError(f"{name} must sum to 1 within tolerance")
    if np.any(arr < 0.0):
        raise InitialStateError(f"{name} contains negative entries below numerical tolerance")
    return arr


def _validate_seed_vector(
    name: str,
    value: np.ndarray,
    *,
    expected_size: int,
    atol: float = 1.0e-12,
) -> np.ndarray:
    arr = _as_1d_float_array(name, value, expected_size=expected_size)
    if np.any(arr < -atol):
        raise InitialStateError(f"{name} must be non-negative")
    total = float(np.sum(arr))
    if total >= 1.0 - atol:
        raise InitialStateError(f"{name} sum must be strictly less than 1")
    if np.any(arr < 0.0):
        raise InitialStateError(f"{name} contains negative entries below numerical tolerance")
    return arr


def _validate_layout_compatibility(run_cfg: RunConfig, mesh: Mesh1D, layout: UnknownLayout | None) -> None:
    if layout is None:
        return
    if layout.n_liq_cells != mesh.n_liq:
        raise InitialStateError("layout.n_liq_cells must match mesh.n_liq")
    if layout.n_gas_cells != mesh.n_gas:
        raise InitialStateError("layout.n_gas_cells must match mesh.n_gas")
    if layout.unknowns_profile != run_cfg.unknowns_profile:
        raise InitialStateError("layout.unknowns_profile must match run_cfg.unknowns_profile")


def _validate_initialization_inputs(
    run_cfg: RunConfig,
    mesh: Mesh1D,
    gas_props: Any,
    liquid_props: Any,
    layout: UnknownLayout | None,
) -> None:
    init = run_cfg.initialization
    species_maps = run_cfg.species_maps
    _validate_layout_compatibility(run_cfg, mesh, layout)

    interface_radius = float(mesh.r_faces[mesh.interface_face_index])
    if not np.isclose(interface_radius, run_cfg.a0, rtol=0.0, atol=1.0e-12 * max(1.0, run_cfg.a0)):
        raise InitialStateError("initial mesh interface radius must match run_cfg.a0")

    _validate_positive_scalar("initialization.t_init_T", init.t_init_T)
    _validate_positive_scalar("initialization.gas_pressure", init.gas_pressure)
    _validate_positive_scalar("initialization.gas_temperature", init.gas_temperature)
    _validate_positive_scalar("initialization.liquid_temperature", init.liquid_temperature)

    gas_y = _validate_mass_fraction_vector(
        "initialization.gas_y_full_0",
        init.gas_y_full_0,
        expected_size=species_maps.n_gas_full,
    )
    liquid_y = _validate_mass_fraction_vector(
        "initialization.liquid_y_full_0",
        init.liquid_y_full_0,
        expected_size=species_maps.n_liq_full,
    )
    y_vap_if0 = _validate_seed_vector(
        "initialization.y_vap_if0_gas_full",
        init.y_vap_if0_gas_full,
        expected_size=species_maps.n_gas_full,
    )

    mapped_gas_indices = np.asarray(species_maps.liq_full_to_gas_full, dtype=np.int64)
    if mapped_gas_indices.ndim != 1 or mapped_gas_indices.shape[0] != species_maps.n_liq_full:
        raise InitialStateError("species_maps.liq_full_to_gas_full must be a 1D mapping aligned with liquid full species")
    if np.any(mapped_gas_indices < 0) or np.any(mapped_gas_indices >= species_maps.n_gas_full):
        raise InitialStateError("species_maps.liq_full_to_gas_full contains out-of-range gas indices")
    if len(np.unique(mapped_gas_indices)) != mapped_gas_indices.shape[0]:
        raise InitialStateError("species_maps.liq_full_to_gas_full must be one-to-one")

    non_cond_mask = np.ones(species_maps.n_gas_full, dtype=bool)
    non_cond_mask[mapped_gas_indices] = False
    if np.any(np.abs(y_vap_if0[non_cond_mask]) > 1.0e-12):
        raise InitialStateError("initialization.y_vap_if0_gas_full must be zero outside mapped vapor gas species")

    if not tuple(gas_props.species_names) == species_maps.gas_full_names:
        raise InitialStateError("gas_props species order must match run_cfg.species_maps.gas_full_names")
    if not tuple(liquid_props.species_names) == species_maps.liq_full_names:
        raise InitialStateError("liquid_props species order must match run_cfg.species_maps.liq_full_names")

    liq_T_range = liquid_props.valid_temperature_range()
    if not (liq_T_range[0] <= init.liquid_temperature <= liq_T_range[1]):
        raise InitialStateError("initial liquid_temperature must lie inside liquid thermo valid_temperature_range")

    gas_cond_inf = float(np.sum(gas_y[mapped_gas_indices]))
    if gas_cond_inf >= 1.0 - 1.0e-12:
        raise InitialStateError("far-field gas composition must retain a non-condensable remainder")

    _ = liquid_y


def _compute_far_field_thermal_diffusivity(
    T_inf: float,
    P_inf: float,
    Yg_inf_full: np.ndarray,
    gas_props: Any,
) -> tuple[float, float, float, float]:
    rho_g_inf = float(gas_props.density_mass(T_inf, Yg_inf_full, P_inf))
    cp_g_inf = float(gas_props.cp_mass(T_inf, Yg_inf_full, P_inf))
    k_g_inf = float(gas_props.conductivity(T_inf, Yg_inf_full, P_inf))
    for name, value in (
        ("rho_g_inf", rho_g_inf),
        ("cp_g_inf", cp_g_inf),
        ("k_g_inf", k_g_inf),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise InitialStateError(f"{name} must be finite and strictly positive")
    D_T_g_inf = k_g_inf / (cp_g_inf * rho_g_inf)
    if not np.isfinite(D_T_g_inf) or D_T_g_inf <= 0.0:
        raise InitialStateError("D_T_g_inf must be finite and strictly positive")
    return rho_g_inf, cp_g_inf, k_g_inf, D_T_g_inf


def _build_initial_interface_gas_composition(
    Yg_inf_full: np.ndarray,
    condensable_gas_indices: np.ndarray,
    noncond_gas_indices: np.ndarray,
    Y_vap_if0_full_cond: np.ndarray,
    *,
    atol: float = 1.0e-12,
) -> np.ndarray:
    Yg_inf = _as_1d_float_array("Yg_inf_full", Yg_inf_full)
    Y_seed = _as_1d_float_array("Y_vap_if0_full_cond", Y_vap_if0_full_cond, expected_size=Yg_inf.shape[0])
    cond_idx = np.asarray(condensable_gas_indices, dtype=np.int64)
    noncond_idx = np.asarray(noncond_gas_indices, dtype=np.int64)

    sum_cond_seed = float(np.sum(Y_seed[cond_idx]))
    sum_cond_inf = float(np.sum(Yg_inf[cond_idx]))
    remainder_inf = 1.0 - sum_cond_inf
    remainder_seed = 1.0 - sum_cond_seed

    if remainder_inf <= atol:
        raise InitialStateError("far-field gas state must include a non-condensable remainder")
    if remainder_seed <= atol:
        raise InitialStateError("initial interface vapor seed must leave a non-condensable remainder")

    alpha = remainder_seed / remainder_inf
    if not np.isfinite(alpha) or alpha < 0.0:
        raise InitialStateError("initial non-condensable gas scaling factor must be finite and non-negative")

    Ys_g_full0 = Y_seed.copy()
    Ys_g_full0[noncond_idx] = alpha * Yg_inf[noncond_idx]

    if np.any(Ys_g_full0 < -atol):
        raise InitialStateError("constructed interface gas seed must be non-negative")
    if not np.isclose(float(np.sum(Ys_g_full0)), 1.0, rtol=0.0, atol=atol):
        raise InitialStateError("constructed interface gas seed must sum to 1")
    return Ys_g_full0


def _profile_multiplier(
    r_gas_cell: np.ndarray,
    *,
    a0: float,
    diffusivity: float,
    t0_smooth: float,
) -> np.ndarray:
    r = _as_1d_float_array("r_gas_cell", r_gas_cell)
    a = _validate_positive_scalar("a0", a0)
    D = _validate_positive_scalar("diffusivity", diffusivity)
    t0 = _validate_positive_scalar("t0_smooth", t0_smooth)
    if np.any(r <= a):
        raise InitialStateError("gas cell centers must lie strictly outside the interface radius")

    sqrt_term = 2.0 * sqrt(D * t0)
    arg = (r - a) / sqrt_term
    erfc_values = np.array([erfc(float(item)) for item in arg], dtype=np.float64)
    return (a / r) * erfc_values


def _build_initial_gas_temperature_profile(
    r_gas_cell: np.ndarray,
    a0: float,
    Td0: float,
    T_inf: float,
    D_T_g_inf: float,
    t0_smooth: float,
) -> np.ndarray:
    multiplier = _profile_multiplier(
        r_gas_cell,
        a0=a0,
        diffusivity=D_T_g_inf,
        t0_smooth=t0_smooth,
    )
    Tg0 = float(T_inf) + multiplier * (float(Td0) - float(T_inf))
    if not np.all(np.isfinite(Tg0)):
        raise InitialStateError("initial gas temperature profile must be finite")
    return Tg0


def _build_initial_gas_composition_profile(
    r_gas_cell: np.ndarray,
    a0: float,
    Yg_inf_full: np.ndarray,
    Ys_g_full0: np.ndarray,
    D_T_g_inf: float,
    t0_smooth: float,
) -> np.ndarray:
    Yg_inf = _as_1d_float_array("Yg_inf_full", Yg_inf_full)
    Ys_g = _validate_mass_fraction_vector(
        "Ys_g_full0",
        Ys_g_full0,
        expected_size=Yg_inf.shape[0],
    )
    multiplier = _profile_multiplier(
        r_gas_cell,
        a0=a0,
        diffusivity=D_T_g_inf,
        t0_smooth=t0_smooth,
    )[:, None]
    Yg_full0 = Yg_inf[None, :] + multiplier * (Ys_g[None, :] - Yg_inf[None, :])
    if not np.all(np.isfinite(Yg_full0)):
        raise InitialStateError("initial gas composition profile must be finite")
    if np.any(Yg_full0 < -1.0e-12):
        raise InitialStateError("initial gas composition profile must remain non-negative")
    row_sums = np.sum(Yg_full0, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-12):
        raise InitialStateError("initial gas composition profile must sum to 1 in every cell")
    return Yg_full0


def _build_initial_liquid_fields(
    n_liq: int,
    T_liq0_scalar: float,
    Yl_full0_scalar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Yl_scalar = _as_1d_float_array("Yl_full0_scalar", Yl_full0_scalar)
    Tl0 = np.full(n_liq, float(T_liq0_scalar), dtype=np.float64)
    Yl_full0 = np.tile(Yl_scalar[None, :], (n_liq, 1))
    return Tl0, Yl_full0


def _build_liquid_derived_state(
    Tl0: np.ndarray,
    Yl_full0: np.ndarray,
    liquid_props: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rho_l = np.asarray(liquid_props.density_mass_batch(Tl0, Yl_full0), dtype=np.float64)
    hl = np.asarray(liquid_props.enthalpy_mass_batch(Tl0, Yl_full0), dtype=np.float64)
    if rho_l.ndim != 1 or rho_l.shape[0] != Tl0.shape[0]:
        raise InitialStateError("liquid density batch output has inconsistent shape")
    if hl.ndim != 1 or hl.shape[0] != Tl0.shape[0]:
        raise InitialStateError("liquid enthalpy batch output has inconsistent shape")
    if not np.all(np.isfinite(rho_l)) or np.any(rho_l <= 0.0):
        raise InitialStateError("initial liquid density field must be finite and strictly positive")
    if not np.all(np.isfinite(hl)):
        raise InitialStateError("initial liquid enthalpy field must be finite")
    return rho_l, hl


def _build_gas_derived_state(
    Tg0: np.ndarray,
    Yg_full0: np.ndarray,
    P_inf: float,
    gas_props: Any,
) -> tuple[np.ndarray, np.ndarray]:
    rho_g = np.asarray(gas_props.density_mass_batch(Tg0, Yg_full0, P_inf), dtype=np.float64)
    hg = np.asarray(gas_props.enthalpy_mass_batch(Tg0, Yg_full0, P_inf), dtype=np.float64)
    if rho_g.ndim != 1 or rho_g.shape[0] != Tg0.shape[0]:
        raise InitialStateError("gas density batch output has inconsistent shape")
    if hg.ndim != 1 or hg.shape[0] != Tg0.shape[0]:
        raise InitialStateError("gas enthalpy batch output has inconsistent shape")
    if not np.all(np.isfinite(rho_g)) or np.any(rho_g <= 0.0):
        raise InitialStateError("initial gas density field must be finite and strictly positive")
    if not np.all(np.isfinite(hg)):
        raise InitialStateError("initial gas enthalpy field must be finite")
    return rho_g, hg


def _assemble_initial_state(
    *,
    Tl0: np.ndarray,
    Yl_full0: np.ndarray,
    rho_l: np.ndarray,
    hl: np.ndarray,
    Tg0: np.ndarray,
    Yg_full0: np.ndarray,
    rho_g: np.ndarray,
    hg: np.ndarray,
    Ts0: float,
    Ys_l_full0: np.ndarray,
    Ys_g_full0: np.ndarray,
    mpp0: float,
    time: float,
) -> State:
    return State(
        Tl=Tl0,
        Yl_full=Yl_full0,
        Tg=Tg0,
        Yg_full=Yg_full0,
        interface=InterfaceState(
            Ts=float(Ts0),
            mpp=float(mpp0),
            Ys_g_full=Ys_g_full0.copy(),
            Ys_l_full=Ys_l_full0.copy(),
        ),
        rho_l=rho_l,
        rho_g=rho_g,
        hl=hl,
        hg=hg,
        # State does not require pre-populated Xg_full.
        # Downstream code must construct mole fractions from Yg_full when needed.
        Xg_full=None,
        time=float(time),
        state_id="initial_state",
    )


def build_initial_state_bundle(
    run_cfg: RunConfig,
    mesh: Mesh1D,
    gas_props: Any,
    liquid_props: Any,
    layout: UnknownLayout | None = None,
) -> InitialStateBundle:
    # Formal project rule:
    # initial interface gas composition is a seeded Eq.(2.21) value, not an equilibrium solve.
    # Formal project rule:
    # initial interface mass flux is fixed to mpp0 = 0.0.
    # Formal project rule:
    # initialization uses the same t0 smoothing parameter for Eq.(2.20) and Eq.(2.21).
    _validate_initialization_inputs(run_cfg, mesh, gas_props, liquid_props, layout)

    init = run_cfg.initialization
    species_maps = run_cfg.species_maps
    T_inf = float(init.gas_temperature)
    P_inf = float(init.gas_pressure)
    Td0 = float(init.liquid_temperature)
    Yg_inf_full = init.gas_y_full_0.astype(np.float64, copy=True)
    Yl0_full = init.liquid_y_full_0.astype(np.float64, copy=True)
    Y_vap_if0_full = init.y_vap_if0_gas_full.astype(np.float64, copy=True)
    t0_smooth = float(init.t_init_T)

    rho_g_inf, cp_g_inf, k_g_inf, D_T_g_inf = _compute_far_field_thermal_diffusivity(
        T_inf=T_inf,
        P_inf=P_inf,
        Yg_inf_full=Yg_inf_full,
        gas_props=gas_props,
    )

    condensable_gas_indices = np.asarray(species_maps.liq_full_to_gas_full, dtype=np.int64)
    all_gas_indices = np.arange(species_maps.n_gas_full, dtype=np.int64)
    noncond_gas_indices = all_gas_indices[~np.isin(all_gas_indices, condensable_gas_indices)]

    Ys_g_full0 = _build_initial_interface_gas_composition(
        Yg_inf_full=Yg_inf_full,
        condensable_gas_indices=condensable_gas_indices,
        noncond_gas_indices=noncond_gas_indices,
        Y_vap_if0_full_cond=Y_vap_if0_full,
    )

    r_gas = mesh.r_centers[mesh.gas_slice]
    Tg0 = _build_initial_gas_temperature_profile(
        r_gas_cell=r_gas,
        a0=float(run_cfg.a0),
        Td0=Td0,
        T_inf=T_inf,
        D_T_g_inf=D_T_g_inf,
        t0_smooth=t0_smooth,
    )
    Yg_full0 = _build_initial_gas_composition_profile(
        r_gas_cell=r_gas,
        a0=float(run_cfg.a0),
        Yg_inf_full=Yg_inf_full,
        Ys_g_full0=Ys_g_full0,
        D_T_g_inf=D_T_g_inf,
        t0_smooth=t0_smooth,
    )

    Tl0, Yl_full0 = _build_initial_liquid_fields(
        n_liq=mesh.n_liq,
        T_liq0_scalar=Td0,
        Yl_full0_scalar=Yl0_full,
    )
    rho_l, hl = _build_liquid_derived_state(
        Tl0=Tl0,
        Yl_full0=Yl_full0,
        liquid_props=liquid_props,
    )
    rho_g, hg = _build_gas_derived_state(
        Tg0=Tg0,
        Yg_full0=Yg_full0,
        P_inf=P_inf,
        gas_props=gas_props,
    )

    state0 = _assemble_initial_state(
        Tl0=Tl0,
        Yl_full0=Yl_full0,
        rho_l=rho_l,
        hl=hl,
        Tg0=Tg0,
        Yg_full0=Yg_full0,
        rho_g=rho_g,
        hg=hg,
        Ts0=Td0,
        Ys_l_full0=Yl0_full,
        Ys_g_full0=Ys_g_full0,
        mpp0=0.0,
        time=0.0,
    )

    info = InitialBuildInfo(
        t0_smooth=t0_smooth,
        rho_g_inf=rho_g_inf,
        cp_g_inf=cp_g_inf,
        k_g_inf=k_g_inf,
        D_T_g_inf=D_T_g_inf,
        Yg_inf_full=Yg_inf_full.copy(),
        Ys_g_full0=Ys_g_full0.copy(),
        notes=(
            "solver time starts at t=0",
            "t0_smooth is shared by Eq.(2.20) and Eq.(2.21)",
            "Ys_g_full0 is a seeded initialization profile value, not equilibrium composition",
            "mpp0=0.0 and dot_a0=0.0 by project rule",
        ),
    )
    return InitialStateBundle(state0=state0, dot_a0=0.0, info=info)


__all__ = [
    "InitialBuildInfo",
    "InitialStateBundle",
    "InitialStateError",
    "build_initial_state_bundle",
]
