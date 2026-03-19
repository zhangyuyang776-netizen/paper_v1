from __future__ import annotations

"""Global Jacobian sparsity-pattern builder for the fixed-geometry inner solve.

This module defines an explicit, conservative CSR sparsity pattern aligned with:
- UnknownLayout
- residual_liquid / residual_interface / residual_gas
- single-mpp interface strategy
- shared InterfaceFacePackage
- continuity-based velocity-recovery coupling

Forbidden here:
- adding Rd back into the inner pattern
- introducing a second gas-side mpp row
- inferring pattern from runtime residual evaluations
- using state- or property-dependent numeric heuristics
- hard-coding global offsets outside layout accessors
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.layout import UnknownLayout
from core.types import Mesh1D, SpeciesMaps


class JacobianPatternBuildError(ValueError):
    """Raised when the Jacobian sparsity pattern cannot be built consistently."""


@dataclass(frozen=True)
class JacobianPattern:
    indptr: np.ndarray
    indices: np.ndarray
    shape: tuple[int, int]
    row_nnz: np.ndarray
    diagnostics: dict[str, Any]


def _empty_int_array() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


def _as_sorted_unique_int_array(values: set[int] | list[int] | np.ndarray) -> np.ndarray:
    if isinstance(values, set):
        if not values:
            return _empty_int_array()
        return np.asarray(sorted(values), dtype=np.int64)
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return _empty_int_array()
    return np.unique(arr)


def _validate_pattern_inputs(*, mesh: Mesh1D, layout: UnknownLayout, species_maps: SpeciesMaps) -> None:
    if layout.n_liq_cells != mesh.n_liq:
        raise JacobianPatternBuildError("layout.n_liq_cells must match mesh.n_liq")
    if layout.n_gas_cells != mesh.n_gas:
        raise JacobianPatternBuildError("layout.n_gas_cells must match mesh.n_gas")
    if layout.n_liq_red != species_maps.n_liq_red:
        raise JacobianPatternBuildError("layout.n_liq_red must match species_maps.n_liq_red")
    if layout.n_gas_red != species_maps.n_gas_red:
        raise JacobianPatternBuildError("layout.n_gas_red must match species_maps.n_gas_red")
    if layout.total_size <= 0:
        raise JacobianPatternBuildError("layout.total_size must be positive")
    if mesh.n_liq <= 0:
        raise JacobianPatternBuildError("mesh.n_liq must be positive")
    if mesh.n_gas <= 0:
        raise JacobianPatternBuildError("mesh.n_gas must be positive")


def _cell_liq_cols(layout: UnknownLayout, cell_id: int) -> np.ndarray:
    cols = [layout.liq_temperature_index(cell_id)]
    species_slice = layout.liq_species_slice_for_cell(cell_id)
    cols.extend(range(species_slice.start, species_slice.stop))
    return np.asarray(cols, dtype=np.int64)


def _cell_gas_cols(layout: UnknownLayout, cell_id: int) -> np.ndarray:
    cols = [layout.gas_temperature_index(cell_id)]
    species_slice = layout.gas_species_slice_for_cell(cell_id)
    cols.extend(range(species_slice.start, species_slice.stop))
    return np.asarray(cols, dtype=np.int64)


def _interface_cols(layout: UnknownLayout) -> np.ndarray:
    return np.arange(layout.if_block.start, layout.if_block.stop, dtype=np.int64)


def _interface_rows(layout: UnknownLayout) -> np.ndarray:
    return np.arange(layout.if_block.start, layout.if_block.stop, dtype=np.int64)


def _add_row_cols(row_sets: list[set[int]], row: int, cols: set[int] | list[int] | np.ndarray) -> None:
    if not (0 <= row < len(row_sets)):
        raise JacobianPatternBuildError(f"row {row} out of range for pattern build")
    row_sets[row].update(int(col) for col in _as_sorted_unique_int_array(cols).tolist())


def _add_self_coupling(row_sets: list[set[int]], n_dof: int) -> None:
    for i in range(n_dof):
        row_sets[i].add(i)


def _build_liquid_row_pattern(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    row_sets: list[set[int]],
    iface_cols: np.ndarray,
) -> None:
    liq_prefix_cols: set[int] = set()
    gas0_cols = _cell_gas_cols(layout, 0) if mesh.n_gas > 0 else _empty_int_array()
    last_liq = mesh.n_liq - 1

    for n in range(mesh.n_liq):
        liq_prefix_cols.update(int(col) for col in _cell_liq_cols(layout, n).tolist())
        cols = set(liq_prefix_cols)
        if n + 1 < mesh.n_liq:
            cols.update(int(col) for col in _cell_liq_cols(layout, n + 1).tolist())
        if n == last_liq:
            cols.update(int(col) for col in iface_cols.tolist())
            cols.update(int(col) for col in gas0_cols.tolist())

        _add_row_cols(row_sets, layout.liq_temperature_index(n), cols)
        species_slice = layout.liq_species_slice_for_cell(n)
        for row in range(species_slice.start, species_slice.stop):
            _add_row_cols(row_sets, row, cols)


def _build_interface_row_pattern(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    row_sets: list[set[int]],
    iface_cols: np.ndarray,
) -> None:
    cols = set(int(col) for col in iface_cols.tolist())
    if mesh.n_liq > 0:
        for liq_cell in range(mesh.n_liq):
            cols.update(int(col) for col in _cell_liq_cols(layout, liq_cell).tolist())
    if mesh.n_gas > 0:
        cols.update(int(col) for col in _cell_gas_cols(layout, 0).tolist())
    for row in _interface_rows(layout).tolist():
        _add_row_cols(row_sets, int(row), cols)


def _build_gas_row_pattern(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    row_sets: list[set[int]],
    iface_cols: np.ndarray,
) -> None:
    gas_prefix_cols: set[int] = set()
    last_liq_cols = _cell_liq_cols(layout, mesh.n_liq - 1) if mesh.n_liq > 0 else _empty_int_array()

    for n in range(mesh.n_gas):
        gas_prefix_cols.update(int(col) for col in _cell_gas_cols(layout, n).tolist())
        cols = set(gas_prefix_cols)
        cols.update(int(col) for col in iface_cols.tolist())
        if n + 1 < mesh.n_gas:
            cols.update(int(col) for col in _cell_gas_cols(layout, n + 1).tolist())
        if n == 0:
            cols.update(int(col) for col in last_liq_cols.tolist())

        _add_row_cols(row_sets, layout.gas_temperature_index(n), cols)
        species_slice = layout.gas_species_slice_for_cell(n)
        for row in range(species_slice.start, species_slice.stop):
            _add_row_cols(row_sets, row, cols)


def _row_sets_to_csr(row_sets: list[set[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(row_sets)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    row_nnz = np.zeros(n_rows, dtype=np.int64)
    indices_parts: list[np.ndarray] = []

    cursor = 0
    for row, cols in enumerate(row_sets):
        row_indices = _as_sorted_unique_int_array(cols)
        row_nnz[row] = row_indices.size
        indices_parts.append(row_indices)
        cursor += row_indices.size
        indptr[row + 1] = cursor

    indices = np.concatenate(indices_parts, dtype=np.int64) if indices_parts else _empty_int_array()
    return indptr, indices, row_nnz


def _count_cross_block_edges(
    row_sets: list[set[int]],
    *,
    row_block: slice,
    col_block: slice,
) -> int:
    count = 0
    col_range = range(col_block.start, col_block.stop)
    col_set = set(col_range)
    for row in range(row_block.start, row_block.stop):
        count += sum(1 for col in row_sets[row] if col in col_set)
    return count


def _build_pattern_diagnostics(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    row_sets: list[set[int]],
    row_nnz: np.ndarray,
    indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "n_dof": int(layout.total_size),
        "nnz_total": int(indices.size),
        "nnz_max_row": int(np.max(row_nnz)) if row_nnz.size else 0,
        "nnz_avg": float(np.mean(row_nnz)) if row_nnz.size else 0.0,
        "block_sizes": {
            "liq": int(layout.liq_block.stop - layout.liq_block.start),
            "iface": int(layout.if_block.stop - layout.if_block.start),
            "gas": int(layout.gas_block.stop - layout.gas_block.start),
        },
        "cross_block_edges": {
            "liq_to_iface": _count_cross_block_edges(row_sets, row_block=layout.liq_block, col_block=layout.if_block),
            "iface_to_liq": _count_cross_block_edges(row_sets, row_block=layout.if_block, col_block=layout.liq_block),
            "iface_to_gas": _count_cross_block_edges(row_sets, row_block=layout.if_block, col_block=layout.gas_block),
            "gas_to_iface": _count_cross_block_edges(row_sets, row_block=layout.gas_block, col_block=layout.if_block),
            "liq_to_gas": _count_cross_block_edges(row_sets, row_block=layout.liq_block, col_block=layout.gas_block),
            "gas_to_liq": _count_cross_block_edges(row_sets, row_block=layout.gas_block, col_block=layout.liq_block),
        },
        "has_liq_species": bool(layout.has_liq_species_bulk),
        "has_iface_liq_species": bool(layout.has_liq_species_interface),
        "has_gas_species": bool(layout.has_gas_species),
        "mesh_cells": {
            "n_liq": int(mesh.n_liq),
            "n_gas": int(mesh.n_gas),
        },
    }


def build_jacobian_pattern(
    *,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
) -> JacobianPattern:
    _validate_pattern_inputs(mesh=mesh, layout=layout, species_maps=species_maps)

    n_dof = int(layout.total_size)
    row_sets = [set() for _ in range(n_dof)]
    _add_self_coupling(row_sets, n_dof)

    iface_cols = _interface_cols(layout)

    _build_liquid_row_pattern(mesh=mesh, layout=layout, row_sets=row_sets, iface_cols=iface_cols)
    _build_interface_row_pattern(mesh=mesh, layout=layout, row_sets=row_sets, iface_cols=iface_cols)
    _build_gas_row_pattern(mesh=mesh, layout=layout, row_sets=row_sets, iface_cols=iface_cols)

    indptr, indices, row_nnz = _row_sets_to_csr(row_sets)
    diagnostics = _build_pattern_diagnostics(
        mesh=mesh,
        layout=layout,
        row_sets=row_sets,
        row_nnz=row_nnz,
        indices=indices,
    )
    return JacobianPattern(
        indptr=indptr,
        indices=indices,
        shape=(n_dof, n_dof),
        row_nnz=row_nnz,
        diagnostics=diagnostics,
    )


__all__ = [
    "JacobianPattern",
    "JacobianPatternBuildError",
    "build_jacobian_pattern",
]
