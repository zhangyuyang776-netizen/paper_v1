from __future__ import annotations

"""Semantic interface-mass residual builder for paper_v1.

This module consumes the unique InterfaceFacePackage and interface velocities
recovered elsewhere, then translates those already-built interface quantities
into mass-related residual terms. It must not recompute interface properties,
gradients, diffusive fluxes, total species fluxes, or equilibrium states.

Forbidden here:
- recomputing J / N / q / E or interface equilibrium
- recovering u_l_if_abs / u_g_if_abs locally
- adding a second gas-side Eq.(2.18) residual row
- assembling into a global residual vector
- treating iface_pkg.Yeq_g_cond_full as a condensable-only subset vector
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.types import RunConfig, SpeciesMaps
from physics.interface_face import InterfaceFacePackage


class InterfaceMassError(ValueError):
    """Raised when interface-mass residual inputs are inconsistent."""


@dataclass(frozen=True)
class InterfaceMassResidualPackage:
    eq215_liq_full_indices: np.ndarray
    eq215_gas_full_indices: np.ndarray
    eq215_values: np.ndarray

    eq216_gas_full_indices: np.ndarray
    eq216_values: np.ndarray

    eq219_gas_full_indices: np.ndarray
    eq219_values: np.ndarray

    mpp_residual: float

    gas_eq18_diag: float | None
    gas_eq18_from_G_diag: float | None
    G_g_if_abs: float
    diagnostics: dict[str, Any]


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise InterfaceMassError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise InterfaceMassError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise InterfaceMassError(f"{name} must have length {expected_size}")
    return arr


def _as_1d_int_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise InterfaceMassError(f"{name} must be a 1D integer array")
    return arr


def _validate_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise InterfaceMassError(f"{name} must be finite")
    return scalar


def _validate_positive_scalar(name: str, value: float) -> float:
    scalar = _validate_finite_scalar(name, value)
    if scalar <= 0.0:
        raise InterfaceMassError(f"{name} must be > 0")
    return scalar


def _require_species_maps(run_cfg: RunConfig) -> SpeciesMaps:
    species_maps = getattr(run_cfg, "species_maps", None)
    if not isinstance(species_maps, SpeciesMaps):
        raise InterfaceMassError("run_cfg.species_maps must be a SpeciesMaps instance")
    return species_maps


def _validate_interface_face_inputs(species_maps: SpeciesMaps, iface_pkg: InterfaceFacePackage) -> None:
    n_liq_full = species_maps.n_liq_full
    n_gas_full = species_maps.n_gas_full

    _as_1d_float_array("iface_pkg.N_l_full", iface_pkg.N_l_full, expected_size=n_liq_full)
    _as_1d_float_array("iface_pkg.N_g_full", iface_pkg.N_g_full, expected_size=n_gas_full)
    _as_1d_float_array("iface_pkg.Ys_g_full", iface_pkg.Ys_g_full, expected_size=n_gas_full)
    if iface_pkg.Yeq_g_cond_full is not None:
        _as_1d_float_array("iface_pkg.Yeq_g_cond_full", iface_pkg.Yeq_g_cond_full, expected_size=n_gas_full)

    _validate_finite_scalar("iface_pkg.mpp", iface_pkg.mpp)
    _validate_positive_scalar("iface_pkg.rho_s_l", iface_pkg.rho_s_l)
    _validate_positive_scalar("iface_pkg.rho_s_g", iface_pkg.rho_s_g)
    _validate_positive_scalar("iface_pkg.area_s", iface_pkg.area_s)
    _validate_finite_scalar("iface_pkg.dot_a_frozen", iface_pkg.dot_a_frozen)
    _validate_finite_scalar("iface_pkg.G_g_if_abs", iface_pkg.G_g_if_abs)


def _closure_indices(species_maps: SpeciesMaps) -> tuple[int | None, int]:
    liq_closure_index = None
    if species_maps.liq_closure_name is not None:
        liq_closure_index = species_maps.liq_full_names.index(species_maps.liq_closure_name)
    gas_closure_index = species_maps.gas_full_names.index(species_maps.gas_closure_name)
    return liq_closure_index, gas_closure_index


def _build_eq215_active_index_pairs(
    run_cfg: RunConfig,
    iface_pkg: InterfaceFacePackage,
) -> tuple[np.ndarray, np.ndarray]:
    del iface_pkg
    species_maps = _require_species_maps(run_cfg)
    liq_closure_index, gas_closure_index = _closure_indices(species_maps)
    liq_to_gas = _as_1d_int_array("species_maps.liq_full_to_gas_full", species_maps.liq_full_to_gas_full)

    liq_indices: list[int] = []
    gas_indices: list[int] = []
    for liq_idx, gas_idx in enumerate(liq_to_gas.tolist()):
        if liq_closure_index is not None and liq_idx == liq_closure_index:
            continue
        if gas_idx == gas_closure_index:
            continue
        liq_indices.append(liq_idx)
        gas_indices.append(gas_idx)

    if len(set(gas_indices)) != len(gas_indices):
        raise InterfaceMassError("Eq.(2.15) active gas indices must be unique")

    return np.asarray(liq_indices, dtype=np.int64), np.asarray(gas_indices, dtype=np.int64)


def _build_eq216_active_gas_indices(
    run_cfg: RunConfig,
    iface_pkg: InterfaceFacePackage,
) -> np.ndarray:
    del iface_pkg
    species_maps = _require_species_maps(run_cfg)
    _, gas_closure_index = _closure_indices(species_maps)
    condensable_gas = set(np.asarray(species_maps.liq_full_to_gas_full, dtype=np.int64).tolist())

    active = [
        gas_idx
        for gas_idx in range(species_maps.n_gas_full)
        if gas_idx not in condensable_gas and gas_idx != gas_closure_index
    ]
    return np.asarray(active, dtype=np.int64)


def _build_eq219_active_gas_indices(
    run_cfg: RunConfig,
    iface_pkg: InterfaceFacePackage,
) -> np.ndarray:
    del iface_pkg
    species_maps = _require_species_maps(run_cfg)
    _, gas_closure_index = _closure_indices(species_maps)
    condensable_gas = np.unique(np.asarray(species_maps.liq_full_to_gas_full, dtype=np.int64))
    active = [int(gas_idx) for gas_idx in condensable_gas.tolist() if int(gas_idx) != gas_closure_index]
    return np.asarray(active, dtype=np.int64)


def _validate_active_index_sets(
    species_maps: SpeciesMaps,
    eq215_liq: np.ndarray,
    eq215_gas: np.ndarray,
    eq216_gas: np.ndarray,
    eq219_gas: np.ndarray,
) -> None:
    for name, arr, upper in (
        ("eq215_liq", eq215_liq, species_maps.n_liq_full),
        ("eq215_gas", eq215_gas, species_maps.n_gas_full),
        ("eq216_gas", eq216_gas, species_maps.n_gas_full),
        ("eq219_gas", eq219_gas, species_maps.n_gas_full),
    ):
        if np.any(arr < 0) or np.any(arr >= upper):
            raise InterfaceMassError(f"{name} contains out-of-range full-species indices")
        if len(np.unique(arr)) != arr.shape[0]:
            raise InterfaceMassError(f"{name} must not contain duplicate indices")

    if set(eq216_gas.tolist()) & set(eq219_gas.tolist()):
        raise InterfaceMassError("Eq.(2.16) and Eq.(2.19) active gas index sets must be disjoint")


def _build_mpp_residual(
    iface_pkg: InterfaceFacePackage,
    u_l_if_abs: float,
) -> float:
    u_l_if = _validate_finite_scalar("u_l_if_abs", u_l_if_abs)
    return float(iface_pkg.mpp + iface_pkg.rho_s_l * (u_l_if - iface_pkg.dot_a_frozen))


def _build_gas_eq18_diagnostic(
    iface_pkg: InterfaceFacePackage,
    u_g_if_abs: float | None,
) -> tuple[float | None, float | None]:
    gas_eq18_diag: float | None = None
    if u_g_if_abs is not None:
        u_g_if = _validate_finite_scalar("u_g_if_abs", u_g_if_abs)
        gas_eq18_diag = float(iface_pkg.mpp + iface_pkg.rho_s_g * (u_g_if - iface_pkg.dot_a_frozen))

    u_g_from_G = float(iface_pkg.G_g_if_abs / (iface_pkg.rho_s_g * iface_pkg.area_s))
    gas_eq18_from_G_diag = float(
        iface_pkg.mpp + iface_pkg.rho_s_g * (u_g_from_G - iface_pkg.dot_a_frozen)
    )
    return gas_eq18_diag, gas_eq18_from_G_diag


def build_interface_mass_residual_package(
    run_cfg: RunConfig,
    iface_pkg: InterfaceFacePackage,
    u_l_if_abs: float,
    u_g_if_abs: float | None = None,
) -> InterfaceMassResidualPackage:
    """Build semantic mass-related interface residual terms.

    This module only translates already-built interface-face quantities into
    Eq.(2.15), Eq.(2.16), Eq.(2.18), and Eq.(2.19) residual expressions.
    """

    species_maps = _require_species_maps(run_cfg)
    _validate_interface_face_inputs(species_maps, iface_pkg)

    eq215_liq, eq215_gas = _build_eq215_active_index_pairs(run_cfg, iface_pkg)
    eq216_gas = _build_eq216_active_gas_indices(run_cfg, iface_pkg)
    eq219_gas = _build_eq219_active_gas_indices(run_cfg, iface_pkg)
    _validate_active_index_sets(species_maps, eq215_liq, eq215_gas, eq216_gas, eq219_gas)

    N_l_full = _as_1d_float_array("iface_pkg.N_l_full", iface_pkg.N_l_full, expected_size=species_maps.n_liq_full)
    N_g_full = _as_1d_float_array("iface_pkg.N_g_full", iface_pkg.N_g_full, expected_size=species_maps.n_gas_full)
    Ys_g_full = _as_1d_float_array("iface_pkg.Ys_g_full", iface_pkg.Ys_g_full, expected_size=species_maps.n_gas_full)

    eq215_values = N_g_full[eq215_gas] - N_l_full[eq215_liq]
    eq216_values = N_g_full[eq216_gas]

    # iface_pkg.Yeq_g_cond_full is currently a full gas equilibrium mass-fraction vector.
    Yeq_g_full = iface_pkg.Yeq_g_cond_full
    if eq219_gas.size > 0 and Yeq_g_full is None:
        raise InterfaceMassError("Eq.(2.19) requires iface_pkg.Yeq_g_cond_full when condensable gas species exist")
    if Yeq_g_full is None:
        eq219_values = np.zeros(0, dtype=np.float64)
    else:
        Yeq_g_full = _as_1d_float_array("iface_pkg.Yeq_g_cond_full", Yeq_g_full, expected_size=species_maps.n_gas_full)
        eq219_values = Ys_g_full[eq219_gas] - Yeq_g_full[eq219_gas]

    mpp_residual = _build_mpp_residual(iface_pkg, u_l_if_abs)
    gas_eq18_diag, gas_eq18_from_G_diag = _build_gas_eq18_diagnostic(iface_pkg, u_g_if_abs)

    diagnostics = {
        "n_eq215": int(eq215_values.shape[0]),
        "n_eq216": int(eq216_values.shape[0]),
        "n_eq219": int(eq219_values.shape[0]),
        "eq215_liq_full_indices": tuple(int(idx) for idx in eq215_liq.tolist()),
        "eq215_gas_full_indices": tuple(int(idx) for idx in eq215_gas.tolist()),
        "eq216_gas_full_indices": tuple(int(idx) for idx in eq216_gas.tolist()),
        "eq219_gas_full_indices": tuple(int(idx) for idx in eq219_gas.tolist()),
        "u_l_if_abs": float(u_l_if_abs),
        "u_g_if_abs": None if u_g_if_abs is None else float(u_g_if_abs),
    }

    return InterfaceMassResidualPackage(
        eq215_liq_full_indices=eq215_liq.copy(),
        eq215_gas_full_indices=eq215_gas.copy(),
        eq215_values=eq215_values.copy(),
        eq216_gas_full_indices=eq216_gas.copy(),
        eq216_values=eq216_values.copy(),
        eq219_gas_full_indices=eq219_gas.copy(),
        eq219_values=eq219_values.copy(),
        mpp_residual=float(mpp_residual),
        gas_eq18_diag=None if gas_eq18_diag is None else float(gas_eq18_diag),
        gas_eq18_from_G_diag=None if gas_eq18_from_G_diag is None else float(gas_eq18_from_G_diag),
        G_g_if_abs=float(iface_pkg.G_g_if_abs),
        diagnostics=diagnostics,
    )
