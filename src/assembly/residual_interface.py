from __future__ import annotations

"""Interface-block residual assembly from semantic interface packages.

This module translates already-built interface mass/energy residual packages
into global interface-block rows. It must not recompute interface properties,
equilibrium, heat fluxes, total species fluxes, or any other interface truth.

Forbidden here:
- adding a second gas-side Eq.(2.18) residual row
- recomputing Y_eq, N_l/N_g, E_l/E_g, or any interface transport quantity
- introducing time terms into the interface block
- bypassing layout accessors / interface slices with hard-coded offsets
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.layout import UnknownLayout
from core.types import SpeciesMaps
from physics.interface_energy import InterfaceEnergyResidualPackage
from physics.interface_face import InterfaceFacePackage
from physics.interface_mass import InterfaceMassResidualPackage


class InterfaceResidualAssemblyError(ValueError):
    """Raised when interface residual assembly inputs are inconsistent."""


@dataclass(frozen=True)
class InterfaceResidualResult:
    rows_global: np.ndarray
    values: np.ndarray

    liq_species_rows_global: np.ndarray
    liq_species_values: np.ndarray

    Ts_row_global: np.ndarray
    Ts_value: np.ndarray

    gas_species_rows_global: np.ndarray
    gas_species_values: np.ndarray

    mpp_row_global: np.ndarray
    mpp_value: np.ndarray

    gas_species_row_kind: np.ndarray
    diagnostics: dict[str, Any]


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise InterfaceResidualAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise InterfaceResidualAssemblyError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise InterfaceResidualAssemblyError(f"{name} must have length {expected_size}")
    return arr

def _as_1d_int_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise InterfaceResidualAssemblyError(f"{name} must be a 1D integer array")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise InterfaceResidualAssemblyError(f"{name} must have length {expected_size}")
    return arr


def _validate_index_array(name: str, value: Any, *, upper: int) -> np.ndarray:
    arr = _as_1d_int_array(name, value)
    if np.any(arr < 0) or np.any(arr >= upper):
        raise InterfaceResidualAssemblyError(f"{name} contains out-of-range full-species indices")
    return arr


def _empty_interface_result() -> InterfaceResidualResult:
    empty_i = np.zeros(0, dtype=np.int64)
    empty_f = np.zeros(0, dtype=np.float64)
    empty_s = np.zeros(0, dtype="<U8")
    return InterfaceResidualResult(
        rows_global=empty_i,
        values=empty_f,
        liq_species_rows_global=empty_i,
        liq_species_values=empty_f,
        Ts_row_global=empty_i,
        Ts_value=empty_f,
        gas_species_rows_global=empty_i,
        gas_species_values=empty_f,
        mpp_row_global=empty_i,
        mpp_value=empty_f,
        gas_species_row_kind=empty_s,
        diagnostics={},
    )


def _validate_interface_inputs(
    *,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    iface_face_pkg: InterfaceFacePackage,
    iface_mass_pkg: InterfaceMassResidualPackage,
    iface_energy_pkg: InterfaceEnergyResidualPackage,
) -> None:
    if layout.n_liq_red != species_maps.n_liq_red:
        raise InterfaceResidualAssemblyError("layout.n_liq_red must match species_maps.n_liq_red")
    if layout.n_gas_red != species_maps.n_gas_red:
        raise InterfaceResidualAssemblyError("layout.n_gas_red must match species_maps.n_gas_red")

    if layout.if_temperature_slice.stop - layout.if_temperature_slice.start != 1:
        raise InterfaceResidualAssemblyError("layout.if_temperature_slice must contain exactly one row")
    if layout.if_mpp_slice.stop - layout.if_mpp_slice.start != 1:
        raise InterfaceResidualAssemblyError("layout.if_mpp_slice must contain exactly one row")
    if layout.if_liq_species_slice.stop - layout.if_liq_species_slice.start != layout.n_liq_red:
        raise InterfaceResidualAssemblyError("layout.if_liq_species_slice length must match layout.n_liq_red")
    if layout.if_gas_species_slice.stop - layout.if_gas_species_slice.start != layout.n_gas_red:
        raise InterfaceResidualAssemblyError("layout.if_gas_species_slice length must match layout.n_gas_red")

    eq215_liq = _validate_index_array("iface_mass_pkg.eq215_liq_full_indices", iface_mass_pkg.eq215_liq_full_indices, upper=species_maps.n_liq_full)
    eq215_gas = _validate_index_array("iface_mass_pkg.eq215_gas_full_indices", iface_mass_pkg.eq215_gas_full_indices, upper=species_maps.n_gas_full)
    eq216_gas = _validate_index_array("iface_mass_pkg.eq216_gas_full_indices", iface_mass_pkg.eq216_gas_full_indices, upper=species_maps.n_gas_full)
    eq219_gas = _validate_index_array("iface_mass_pkg.eq219_gas_full_indices", iface_mass_pkg.eq219_gas_full_indices, upper=species_maps.n_gas_full)
    _as_1d_float_array("iface_mass_pkg.eq215_values", iface_mass_pkg.eq215_values, expected_size=eq215_liq.shape[0])
    _as_1d_float_array("iface_mass_pkg.eq216_values", iface_mass_pkg.eq216_values, expected_size=eq216_gas.shape[0])
    _as_1d_float_array("iface_mass_pkg.eq219_values", iface_mass_pkg.eq219_values, expected_size=eq219_gas.shape[0])
    if not np.isfinite(iface_mass_pkg.mpp_residual):
        raise InterfaceResidualAssemblyError("iface_mass_pkg.mpp_residual must be finite")
    if not np.isfinite(iface_energy_pkg.energy_residual):
        raise InterfaceResidualAssemblyError("iface_energy_pkg.energy_residual must be finite")
    if eq215_liq.shape[0] != eq215_gas.shape[0]:
        raise InterfaceResidualAssemblyError("Eq.(2.15) liquid/gas full-index arrays must have equal length")
    if eq215_liq.shape[0] != layout.n_liq_red:
        raise InterfaceResidualAssemblyError("Eq.(2.15) active liquid species count must match layout.n_liq_red")
    if eq216_gas.shape[0] + eq219_gas.shape[0] != layout.n_gas_red:
        raise InterfaceResidualAssemblyError("Eq.(2.16)+Eq.(2.19) gas row counts must match layout.n_gas_red")
    full_to_reduced_liq = _as_1d_int_array("species_maps.liq_full_to_reduced", species_maps.liq_full_to_reduced, expected_size=species_maps.n_liq_full)
    full_to_reduced_gas = _as_1d_int_array("species_maps.gas_full_to_reduced", species_maps.gas_full_to_reduced, expected_size=species_maps.n_gas_full)
    _ = iface_face_pkg
    _ = full_to_reduced_liq
    _ = full_to_reduced_gas


def _validate_package_consistency(
    *,
    species_maps: SpeciesMaps,
    iface_face_pkg: InterfaceFacePackage,
    iface_mass_pkg: InterfaceMassResidualPackage,
    iface_energy_pkg: InterfaceEnergyResidualPackage,
) -> None:
    if not np.isclose(float(iface_energy_pkg.E_l_s), float(iface_face_pkg.E_l_s), rtol=0.0, atol=1.0e-12):
        raise InterfaceResidualAssemblyError("iface_energy_pkg.E_l_s must match iface_face_pkg.E_l_s")
    if not np.isclose(float(iface_energy_pkg.E_g_s), float(iface_face_pkg.E_g_s), rtol=0.0, atol=1.0e-12):
        raise InterfaceResidualAssemblyError("iface_energy_pkg.E_g_s must match iface_face_pkg.E_g_s")

    eq215_liq = np.asarray(iface_mass_pkg.eq215_liq_full_indices, dtype=np.int64)
    eq215_gas = np.asarray(iface_mass_pkg.eq215_gas_full_indices, dtype=np.int64)
    eq216_gas = np.asarray(iface_mass_pkg.eq216_gas_full_indices, dtype=np.int64)
    eq219_gas = np.asarray(iface_mass_pkg.eq219_gas_full_indices, dtype=np.int64)
    N_l_full = _as_1d_float_array("iface_face_pkg.N_l_full", iface_face_pkg.N_l_full, expected_size=species_maps.n_liq_full)
    N_g_full = _as_1d_float_array("iface_face_pkg.N_g_full", iface_face_pkg.N_g_full, expected_size=species_maps.n_gas_full)
    Ys_g_full = _as_1d_float_array("iface_face_pkg.Ys_g_full", iface_face_pkg.Ys_g_full, expected_size=species_maps.n_gas_full)

    expected_eq215 = N_g_full[eq215_gas] - N_l_full[eq215_liq]
    if not np.allclose(expected_eq215, iface_mass_pkg.eq215_values, rtol=0.0, atol=1.0e-12):
        raise InterfaceResidualAssemblyError("iface_mass_pkg.eq215_values must match iface_face_pkg total species fluxes")

    expected_eq216 = N_g_full[eq216_gas]
    if not np.allclose(expected_eq216, iface_mass_pkg.eq216_values, rtol=0.0, atol=1.0e-12):
        raise InterfaceResidualAssemblyError("iface_mass_pkg.eq216_values must match iface_face_pkg total species fluxes")

    Yeq_g_full = iface_face_pkg.Yeq_g_cond_full
    if eq219_gas.size > 0:
        if Yeq_g_full is None:
            raise InterfaceResidualAssemblyError("iface_face_pkg.Yeq_g_cond_full is required for Eq.(2.19) rows")
        Yeq_g_full = _as_1d_float_array("iface_face_pkg.Yeq_g_cond_full", Yeq_g_full, expected_size=species_maps.n_gas_full)
        expected_eq219 = Ys_g_full[eq219_gas] - Yeq_g_full[eq219_gas]
        if not np.allclose(expected_eq219, iface_mass_pkg.eq219_values, rtol=0.0, atol=1.0e-12):
            raise InterfaceResidualAssemblyError("iface_mass_pkg.eq219_values must match iface_face_pkg equilibrium closure")


def _assemble_interface_liq_species_rows(
    *,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    iface_mass_pkg: InterfaceMassResidualPackage,
) -> tuple[np.ndarray, np.ndarray]:
    if layout.n_liq_red == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

    liq_slice = layout.if_liq_species_slice
    rows = np.arange(liq_slice.start, liq_slice.stop, dtype=np.int64)
    values = np.full(layout.n_liq_red, np.nan, dtype=np.float64)
    filled = np.zeros(layout.n_liq_red, dtype=bool)
    full_to_reduced = _as_1d_int_array("species_maps.liq_full_to_reduced", species_maps.liq_full_to_reduced, expected_size=species_maps.n_liq_full)

    for full_idx, value in zip(
        _validate_index_array("iface_mass_pkg.eq215_liq_full_indices", iface_mass_pkg.eq215_liq_full_indices, upper=species_maps.n_liq_full).tolist(),
        np.asarray(iface_mass_pkg.eq215_values, dtype=np.float64).tolist(),
    ):
        red_idx = int(full_to_reduced[full_idx])
        if red_idx < 0:
            raise InterfaceResidualAssemblyError("Eq.(2.15) liquid full species must map to a reduced interface row")
        if red_idx >= layout.n_liq_red:
            raise InterfaceResidualAssemblyError("Eq.(2.15) liquid reduced index exceeds layout.n_liq_red")
        if filled[red_idx]:
            raise InterfaceResidualAssemblyError("Eq.(2.15) liquid reduced row must be filled exactly once")
        values[red_idx] = float(value)
        filled[red_idx] = True

    if not np.all(filled):
        raise InterfaceResidualAssemblyError("every liquid interface reduced row must be mapped exactly once")

    return rows, values


def _assemble_interface_temperature_row(
    *,
    layout: UnknownLayout,
    iface_energy_pkg: InterfaceEnergyResidualPackage,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([layout.if_temperature_index], dtype=np.int64),
        np.asarray([float(iface_energy_pkg.energy_residual)], dtype=np.float64),
    )


def _assemble_interface_gas_species_rows(
    *,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    iface_mass_pkg: InterfaceMassResidualPackage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gas_slice = layout.if_gas_species_slice
    rows = np.arange(gas_slice.start, gas_slice.stop, dtype=np.int64)
    values = np.full(layout.n_gas_red, np.nan, dtype=np.float64)
    kind = np.full(layout.n_gas_red, "UNSET", dtype="<U8")
    filled = np.zeros(layout.n_gas_red, dtype=bool)
    full_to_reduced = _as_1d_int_array("species_maps.gas_full_to_reduced", species_maps.gas_full_to_reduced, expected_size=species_maps.n_gas_full)

    for full_idx, value in zip(
        _validate_index_array("iface_mass_pkg.eq216_gas_full_indices", iface_mass_pkg.eq216_gas_full_indices, upper=species_maps.n_gas_full).tolist(),
        np.asarray(iface_mass_pkg.eq216_values, dtype=np.float64).tolist(),
    ):
        red_idx = int(full_to_reduced[full_idx])
        if red_idx < 0:
            raise InterfaceResidualAssemblyError("Eq.(2.16) gas full species must map to a reduced interface row")
        if red_idx >= layout.n_gas_red:
            raise InterfaceResidualAssemblyError("Eq.(2.16) gas reduced index exceeds layout.n_gas_red")
        if filled[red_idx]:
            raise InterfaceResidualAssemblyError("each gas reduced interface row must be filled exactly once")
        values[red_idx] = float(value)
        kind[red_idx] = "eq16"
        filled[red_idx] = True

    for full_idx, value in zip(
        _validate_index_array("iface_mass_pkg.eq219_gas_full_indices", iface_mass_pkg.eq219_gas_full_indices, upper=species_maps.n_gas_full).tolist(),
        np.asarray(iface_mass_pkg.eq219_values, dtype=np.float64).tolist(),
    ):
        red_idx = int(full_to_reduced[full_idx])
        if red_idx < 0:
            raise InterfaceResidualAssemblyError("Eq.(2.19) gas full species must map to a reduced interface row")
        if red_idx >= layout.n_gas_red:
            raise InterfaceResidualAssemblyError("Eq.(2.19) gas reduced index exceeds layout.n_gas_red")
        if filled[red_idx]:
            raise InterfaceResidualAssemblyError("each gas reduced interface row must be filled exactly once")
        values[red_idx] = float(value)
        kind[red_idx] = "eq19"
        filled[red_idx] = True

    if not np.all(filled) or not np.all(np.isfinite(values)) or np.any(kind == "UNSET"):
        raise InterfaceResidualAssemblyError("every gas reduced interface species row must map exactly once to eq16 or eq19")

    return rows, values, kind


def _assemble_interface_mpp_row(
    *,
    layout: UnknownLayout,
    iface_mass_pkg: InterfaceMassResidualPackage,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([layout.if_mpp_index], dtype=np.int64),
        np.asarray([float(iface_mass_pkg.mpp_residual)], dtype=np.float64),
    )


def assemble_interface_residual(
    *,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    iface_face_pkg: InterfaceFacePackage,
    iface_mass_pkg: InterfaceMassResidualPackage,
    iface_energy_pkg: InterfaceEnergyResidualPackage,
    owner_active: bool = True,
) -> InterfaceResidualResult:
    """Assemble interface-block residual rows from semantic interface packages."""

    if not owner_active:
        return _empty_interface_result()

    _validate_interface_inputs(
        layout=layout,
        species_maps=species_maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )
    _validate_package_consistency(
        species_maps=species_maps,
        iface_face_pkg=iface_face_pkg,
        iface_mass_pkg=iface_mass_pkg,
        iface_energy_pkg=iface_energy_pkg,
    )

    liq_rows, liq_vals = _assemble_interface_liq_species_rows(
        layout=layout,
        species_maps=species_maps,
        iface_mass_pkg=iface_mass_pkg,
    )
    Ts_row, Ts_val = _assemble_interface_temperature_row(
        layout=layout,
        iface_energy_pkg=iface_energy_pkg,
    )
    gas_rows, gas_vals, gas_kind = _assemble_interface_gas_species_rows(
        layout=layout,
        species_maps=species_maps,
        iface_mass_pkg=iface_mass_pkg,
    )
    mpp_row, mpp_val = _assemble_interface_mpp_row(
        layout=layout,
        iface_mass_pkg=iface_mass_pkg,
    )

    if layout.n_liq_red > 0:
        rows_global = np.concatenate([liq_rows, Ts_row, gas_rows, mpp_row])
        values = np.concatenate([liq_vals, Ts_val, gas_vals, mpp_val])
    else:
        rows_global = np.concatenate([Ts_row, gas_rows, mpp_row])
        values = np.concatenate([Ts_val, gas_vals, mpp_val])

    diagnostics = {
        "gas_species_row_kind": tuple(str(item) for item in gas_kind.tolist()),
        "gas_eq18_diag": iface_mass_pkg.gas_eq18_diag,
        "gas_eq18_from_G_diag": iface_mass_pkg.gas_eq18_from_G_diag,
        "E_l_s": float(iface_energy_pkg.E_l_s),
        "E_g_s": float(iface_energy_pkg.E_g_s),
        "G_g_if_abs": float(iface_mass_pkg.G_g_if_abs),
    }

    return InterfaceResidualResult(
        rows_global=rows_global,
        values=values,
        liq_species_rows_global=liq_rows,
        liq_species_values=liq_vals,
        Ts_row_global=Ts_row,
        Ts_value=Ts_val,
        gas_species_rows_global=gas_rows,
        gas_species_values=gas_vals,
        mpp_row_global=mpp_row,
        mpp_value=mpp_val,
        gas_species_row_kind=gas_kind,
        diagnostics=diagnostics,
    )


__all__ = [
    "InterfaceResidualAssemblyError",
    "InterfaceResidualResult",
    "assemble_interface_residual",
]
