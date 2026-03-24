from __future__ import annotations

"""Bulk property aggregation for Phase 2.

This module intentionally aggregates only bulk liquid/gas properties from a
full-order State, Mesh1D geometry, and the already-built thermo backends.
It does not construct interface packages, equilibrium closures, fluxes, or
any residual/Jacobian contributions.
"""

from typing import Any

import numpy as np

from core.types import Mesh1D, Props, State
from properties.gas import GasThermoModel
from properties.liquid import LiquidThermoModel


class AggregatorError(Exception):
    """Base error for bulk property aggregation."""


class AggregatorValidationError(AggregatorError):
    """Raised when State/Grid/backends cannot be aggregated consistently."""


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise AggregatorValidationError(f"{name} must be a one-dimensional float array")
    if arr.size == 0:
        raise AggregatorValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise AggregatorValidationError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise AggregatorValidationError(f"{name} must have length {expected_size}")
    return arr


def _as_2d_float_array(
    name: str,
    value: Any,
    *,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise AggregatorValidationError(f"{name} must be a two-dimensional float array")
    if arr.size == 0:
        raise AggregatorValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise AggregatorValidationError(f"{name} must contain only finite values")
    if expected_rows is not None and arr.shape[0] != expected_rows:
        raise AggregatorValidationError(f"{name} must have row count {expected_rows}")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise AggregatorValidationError(f"{name} must have column count {expected_cols}")
    return arr


def _require_positive_array(name: str, arr: np.ndarray) -> np.ndarray:
    if np.any(arr <= 0.0):
        raise AggregatorValidationError(f"{name} must be strictly positive everywhere")
    return arr


def _normalize_gas_pressure(gas_pressure: float | np.ndarray | None, n_gas_cells: int, default_pressure: float) -> np.ndarray:
    if gas_pressure is None:
        return np.full(n_gas_cells, float(default_pressure), dtype=np.float64)
    if np.isscalar(gas_pressure):
        value = float(gas_pressure)
        if not np.isfinite(value) or value <= 0.0:
            raise AggregatorValidationError("gas_pressure must be finite and strictly positive")
        return np.full(n_gas_cells, value, dtype=np.float64)
    arr = _as_1d_float_array("gas_pressure", gas_pressure, expected_size=n_gas_cells)
    if np.any(arr <= 0.0):
        raise AggregatorValidationError("gas_pressure must be strictly positive")
    return arr


def _normalize_liquid_diffusivity(diff_value: np.ndarray | None, *, n_species: int) -> np.ndarray | None:
    if diff_value is None:
        return None
    arr = _as_1d_float_array("liquid diffusivity vector", diff_value, expected_size=n_species)
    if np.any(arr <= 0.0):
        raise AggregatorValidationError("liquid diffusivity vector must be strictly positive")
    return arr


def validate_state_grid_compatibility(state: State, grid: Mesh1D) -> None:
    if grid.n_liq != state.n_liq_cells:
        raise AggregatorValidationError("grid.n_liq must match state.n_liq_cells")
    if grid.n_gas != state.n_gas_cells:
        raise AggregatorValidationError("grid.n_gas must match state.n_gas_cells")
    if state.Tl.shape != (grid.n_liq,):
        raise AggregatorValidationError("state.Tl shape must match liquid cell count")
    if state.Yl_full.shape[0] != grid.n_liq:
        raise AggregatorValidationError("state.Yl_full row count must match liquid cell count")
    if state.Tg.shape != (grid.n_gas,):
        raise AggregatorValidationError("state.Tg shape must match gas cell count")
    if state.Yg_full.shape[0] != grid.n_gas:
        raise AggregatorValidationError("state.Yg_full row count must match gas cell count")
    if grid.liq_slice != grid.region_slices.liq:
        raise AggregatorValidationError("grid.liq_slice must match region_slices.liq")
    if grid.gas_slice != grid.region_slices.gas_all:
        raise AggregatorValidationError("grid.gas_slice must match region_slices.gas_all")


def build_bulk_props(
    *,
    state: State,
    grid: Mesh1D,
    liquid_thermo: LiquidThermoModel,
    gas_thermo: GasThermoModel,
    gas_pressure: float | np.ndarray | None = None,
) -> Props:
    validate_state_grid_compatibility(state, grid)

    if state.n_liq_species_full != liquid_thermo.n_species:
        raise AggregatorValidationError("state liquid species dimension must match liquid_thermo.n_species")
    if state.n_gas_species_full != gas_thermo.n_species:
        raise AggregatorValidationError("state gas species dimension must match gas_thermo.n_species")

    gas_pressure_arr = _normalize_gas_pressure(gas_pressure, state.n_gas_cells, gas_thermo.reference_pressure)

    rho_l = liquid_thermo.density_mass_batch(state.Tl, state.Yl_full)
    cp_l = liquid_thermo.cp_mass_batch(state.Tl, state.Yl_full)
    h_l = liquid_thermo.enthalpy_mass_batch(state.Tl, state.Yl_full)
    k_l = np.array(
        [liquid_thermo.conductivity(float(state.Tl[i]), state.Yl_full[i, :]) for i in range(state.n_liq_cells)],
        dtype=np.float64,
    )
    mu_l = np.array(
        [liquid_thermo.viscosity(float(state.Tl[i]), state.Yl_full[i, :]) for i in range(state.n_liq_cells)],
        dtype=np.float64,
    )
    liquid_diff_rows = [
        _normalize_liquid_diffusivity(
            liquid_thermo.diffusivity(float(state.Tl[i]), state.Yl_full[i, :]),
            n_species=liquid_thermo.n_species,
        )
        for i in range(state.n_liq_cells)
    ]
    if all(row is None for row in liquid_diff_rows):
        D_l = None
    elif any(row is None for row in liquid_diff_rows):
        raise AggregatorValidationError("liquid diffusivity rows must be consistently None or vectors")
    else:
        D_l = np.vstack(liquid_diff_rows)

    rho_g = gas_thermo.density_mass_batch(state.Tg, state.Yg_full, gas_pressure_arr)
    cp_g = gas_thermo.cp_mass_batch(state.Tg, state.Yg_full, gas_pressure_arr)
    h_g = gas_thermo.enthalpy_mass_batch(state.Tg, state.Yg_full, gas_pressure_arr)
    k_g = np.array(
        [
            gas_thermo.conductivity(float(state.Tg[i]), state.Yg_full[i, :], float(gas_pressure_arr[i]))
            for i in range(state.n_gas_cells)
        ],
        dtype=np.float64,
    )
    mu_g = np.array(
        [
            gas_thermo.viscosity(float(state.Tg[i]), state.Yg_full[i, :], float(gas_pressure_arr[i]))
            for i in range(state.n_gas_cells)
        ],
        dtype=np.float64,
    )
    D_g = np.vstack(
        [
            gas_thermo.diffusivity(float(state.Tg[i]), state.Yg_full[i, :], float(gas_pressure_arr[i]))
            for i in range(state.n_gas_cells)
        ]
    )
    if D_l is not None and D_l.shape != (state.n_liq_cells, state.n_liq_species_full):
        raise AggregatorValidationError("D_l shape must be (n_liq_cells, n_liq_species_full)")
    if D_g.shape != (state.n_gas_cells, state.n_gas_species_full):
        raise AggregatorValidationError("D_g shape must be (n_gas_cells, n_gas_species_full)")

    finite_arrays: list[np.ndarray] = [rho_l, cp_l, h_l, k_l, mu_l, rho_g, cp_g, h_g, k_g, mu_g, D_g]
    if D_l is not None:
        finite_arrays.append(D_l)
    diagnostics = {
        "n_liq_cells": state.n_liq_cells,
        "n_gas_cells": state.n_gas_cells,
        "liquid_species_count": state.n_liq_species_full,
        "gas_species_count": state.n_gas_species_full,
        "all_finite": bool(all(np.all(np.isfinite(arr)) for arr in finite_arrays)),
        "min_rho_l": float(np.min(rho_l)),
        "min_rho_g": float(np.min(rho_g)),
        "min_cp_l": float(np.min(cp_l)),
        "min_cp_g": float(np.min(cp_g)),
        "min_gas_pressure": float(np.min(gas_pressure_arr)),
        "max_gas_pressure": float(np.max(gas_pressure_arr)),
    }

    return Props(
        rho_l=rho_l,
        cp_l=cp_l,
        h_l=h_l,
        k_l=k_l,
        mu_l=mu_l,
        D_l=D_l,
        rho_g=rho_g,
        cp_g=cp_g,
        h_g=h_g,
        k_g=k_g,
        mu_g=mu_g,
        D_g=D_g,
        diagnostics=diagnostics,
    )


BulkProps = Props


__all__ = [
    "AggregatorError",
    "AggregatorValidationError",
    "BulkProps",
    "build_bulk_props",
    "validate_state_grid_compatibility",
]
