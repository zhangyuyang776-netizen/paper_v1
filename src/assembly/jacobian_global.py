from __future__ import annotations

"""Global Jacobian orchestration for the fixed-geometry inner solve.

This module closes the frozen Jacobian assembly pipeline by:
- reusing the existing liquid / interface / gas block Jacobian assemblers
- sharing one upstream residual-evaluation pipeline across all blocks
- validating merged layout-global COO triplets against JacobianPattern
- optionally inserting triplets into an already preallocated PETSc Mat

Forbidden here:
- rebuilding an alternative physics / interface / velocity / far-field chain
- bypassing the block Jacobian assemblers and hand-writing triplets
- inventing a second layout-to-PETSc permutation in this module
- silently coalescing duplicate triplets to hide block-boundary defects
- writing pattern-external slots or relying on dynamic new-nonzero insertion
- introducing Rd or a second gas-side mpp row
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from assembly.jacobian_gas import (
    GasJacobianFDOptions,
    GasJacobianResult,
    assemble_gas_jacobian,
)
from assembly.jacobian_interface import (
    InterfaceJacobianFDOptions,
    InterfaceJacobianResult,
    assemble_interface_jacobian,
)
from assembly.jacobian_liquid import (
    LiquidJacobianFDOptions,
    LiquidJacobianResult,
    assemble_liquid_jacobian,
)
from assembly.jacobian_pattern import JacobianPattern
from assembly.residual_gas import GasResidualResult
from assembly.residual_interface import InterfaceResidualResult
from assembly.residual_liquid import LiquidResidualResult
from core.layout import UnknownLayout
from core.types import Mesh1D, SpeciesMaps


class GlobalJacobianAssemblyError(ValueError):
    """Raised when global Jacobian orchestration inputs are inconsistent."""


@dataclass(frozen=True)
class JacobianOwnership:
    owned_liq_cells: np.ndarray
    owned_gas_cells: np.ndarray
    interface_owner_active: bool


@dataclass(frozen=True)
class GlobalJacobianFDOptions:
    liquid: LiquidJacobianFDOptions = field(default_factory=LiquidJacobianFDOptions)
    interface: InterfaceJacobianFDOptions = field(default_factory=InterfaceJacobianFDOptions)
    gas: GasJacobianFDOptions = field(default_factory=GasJacobianFDOptions)
    cache_size: int = 32


@dataclass(frozen=True)
class GlobalJacobianResult:
    rows_global: np.ndarray
    cols_global: np.ndarray
    values: np.ndarray
    liquid: LiquidJacobianResult
    interface: InterfaceJacobianResult
    gas: GasJacobianResult
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PETScGlobalJacobianResult:
    mat: Any
    triplets: GlobalJacobianResult
    diagnostics: dict[str, Any]


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GlobalJacobianAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise GlobalJacobianAssemblyError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise GlobalJacobianAssemblyError(f"{name} must have length {expected_size}")
    return arr


def _as_1d_int_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise GlobalJacobianAssemblyError(f"{name} must be a 1D integer array")
    return arr


def _validate_global_jacobian_inputs(
    *,
    u_trial_layout: np.ndarray,
    layout: UnknownLayout,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    ownership: JacobianOwnership,
    pattern: JacobianPattern,
    build_all_residual_blocks_from_layout_vector: Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
    fd_options: GlobalJacobianFDOptions,
) -> JacobianOwnership:
    trial = _as_1d_float_array("u_trial_layout", u_trial_layout, expected_size=layout.total_size)
    if pattern.shape != (layout.total_size, layout.total_size):
        raise GlobalJacobianAssemblyError("pattern.shape must match layout.total_size")
    if layout.n_liq_cells != mesh.n_liq or layout.n_gas_cells != mesh.n_gas:
        raise GlobalJacobianAssemblyError("layout cell counts must match mesh cell counts")
    if layout.n_liq_red != species_maps.n_liq_red or layout.n_gas_red != species_maps.n_gas_red:
        raise GlobalJacobianAssemblyError("layout reduced-species sizes must match species_maps")
    if not callable(build_all_residual_blocks_from_layout_vector):
        raise GlobalJacobianAssemblyError("build_all_residual_blocks_from_layout_vector must be callable")
    if not isinstance(fd_options, GlobalJacobianFDOptions):
        raise GlobalJacobianAssemblyError("fd_options must be a GlobalJacobianFDOptions instance")
    if not isinstance(fd_options.cache_size, (int, np.integer)) or int(fd_options.cache_size) < 0:
        raise GlobalJacobianAssemblyError("fd_options.cache_size must be an integer >= 0")

    owned_liq = _as_1d_int_array("ownership.owned_liq_cells", ownership.owned_liq_cells)
    owned_gas = _as_1d_int_array("ownership.owned_gas_cells", ownership.owned_gas_cells)
    if owned_liq.size > 0:
        if np.any(owned_liq < 0) or np.any(owned_liq >= mesh.n_liq):
            raise GlobalJacobianAssemblyError("ownership.owned_liq_cells contains out-of-range liquid cell indices")
        if np.unique(owned_liq).size != owned_liq.size:
            raise GlobalJacobianAssemblyError("ownership.owned_liq_cells must not contain duplicates")
    if owned_gas.size > 0:
        if np.any(owned_gas < 0) or np.any(owned_gas >= mesh.n_gas):
            raise GlobalJacobianAssemblyError("ownership.owned_gas_cells contains out-of-range gas cell indices")
        if np.unique(owned_gas).size != owned_gas.size:
            raise GlobalJacobianAssemblyError("ownership.owned_gas_cells must not contain duplicates")

    return JacobianOwnership(
        owned_liq_cells=owned_liq.copy(),
        owned_gas_cells=owned_gas.copy(),
        interface_owner_active=bool(ownership.interface_owner_active),
    )


def _validate_layout_to_petsc_mapping(
    layout_to_petsc: np.ndarray | None,
    *,
    n_global: int,
) -> np.ndarray:
    if layout_to_petsc is None:
        return np.arange(n_global, dtype=np.int64)
    perm = np.asarray(layout_to_petsc, dtype=np.int64)
    if perm.ndim != 1 or perm.shape[0] != n_global:
        raise GlobalJacobianAssemblyError("layout_to_petsc must be a 1D permutation of length n_global")
    if np.any(perm < 0) or np.any(perm >= n_global):
        raise GlobalJacobianAssemblyError("layout_to_petsc must contain indices in [0, n_global)")
    if np.unique(perm).size != n_global:
        raise GlobalJacobianAssemblyError("layout_to_petsc must be a permutation of 0..n_global-1")
    return perm


def _make_shared_block_residual_evaluator(
    *,
    build_all_residual_blocks_from_layout_vector: Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
    n_global: int,
    cache_size: int,
) -> tuple[
    Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
    dict[str, int],
]:
    cache: OrderedDict[bytes, tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]] = OrderedDict()
    stats = {"hits": 0, "misses": 0}

    def _evaluate_all(u_layout: np.ndarray) -> tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]:
        trial = _as_1d_float_array("u_layout", u_layout, expected_size=n_global)
        key_array = np.ascontiguousarray(trial, dtype=np.float64)
        key = key_array.tobytes()
        if key in cache:
            stats["hits"] += 1
            cache.move_to_end(key)
            return cache[key]

        stats["misses"] += 1
        result = build_all_residual_blocks_from_layout_vector(np.array(key_array, dtype=np.float64, copy=True))
        if not isinstance(result, tuple) or len(result) != 3:
            raise GlobalJacobianAssemblyError(
                "build_all_residual_blocks_from_layout_vector must return a 3-tuple "
                "(liquid, interface, gas)"
            )
        if cache_size > 0:
            cache[key] = result
            if len(cache) > cache_size:
                cache.popitem(last=False)
        return result

    return _evaluate_all, stats


def _make_block_callbacks(
    evaluate_all: Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
) -> tuple[
    Callable[[np.ndarray], LiquidResidualResult],
    Callable[[np.ndarray], InterfaceResidualResult],
    Callable[[np.ndarray], GasResidualResult],
]:
    def _eval_liquid(u_layout: np.ndarray) -> LiquidResidualResult:
        return evaluate_all(u_layout)[0]

    def _eval_interface(u_layout: np.ndarray) -> InterfaceResidualResult:
        return evaluate_all(u_layout)[1]

    def _eval_gas(u_layout: np.ndarray) -> GasResidualResult:
        return evaluate_all(u_layout)[2]

    return _eval_liquid, _eval_interface, _eval_gas


def _coalesce_or_raise_duplicate_triplets(
    *,
    rows_global: np.ndarray,
    cols_global: np.ndarray,
) -> None:
    if rows_global.size != cols_global.size:
        raise GlobalJacobianAssemblyError("rows_global and cols_global must have the same length")
    if rows_global.size == 0:
        return
    pairs = np.column_stack((rows_global, cols_global))
    if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
        raise GlobalJacobianAssemblyError("global Jacobian triplets contain duplicate (row, col) entries")


def _validate_global_triplets_against_pattern(
    *,
    rows_global: np.ndarray,
    cols_global: np.ndarray,
    values: np.ndarray,
    pattern: JacobianPattern,
    expected_rows_global: np.ndarray,
) -> None:
    rows = np.asarray(rows_global, dtype=np.int64)
    cols = np.asarray(cols_global, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    n_global = int(pattern.shape[0])
    if rows.ndim != 1 or cols.ndim != 1 or vals.ndim != 1:
        raise GlobalJacobianAssemblyError("rows_global, cols_global, and values must all be 1D arrays")
    if rows.size != cols.size or rows.size != vals.size:
        raise GlobalJacobianAssemblyError("rows_global, cols_global, and values must have the same length")
    if np.any(~np.isfinite(vals)):
        raise GlobalJacobianAssemblyError("global Jacobian values must be finite")
    if np.any(rows < 0) or np.any(rows >= n_global) or np.any(cols < 0) or np.any(cols >= n_global):
        raise GlobalJacobianAssemblyError("global Jacobian triplets contain out-of-range row/col indices")
    _coalesce_or_raise_duplicate_triplets(rows_global=rows, cols_global=cols)

    for row, col in zip(rows.tolist(), cols.tolist()):
        row_cols = np.asarray(pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]], dtype=np.int64)
        if not np.any(row_cols == col):
            raise GlobalJacobianAssemblyError(
                f"global Jacobian emitted pattern-external triplet ({row}, {col})"
            )

    expected_rows = np.asarray(expected_rows_global, dtype=np.int64)
    if expected_rows.size != np.unique(expected_rows).size:
        raise GlobalJacobianAssemblyError("expected global row set must not contain duplicates")
    if set(rows.tolist()) != set(expected_rows.tolist()):
        raise GlobalJacobianAssemblyError("global Jacobian row set must exactly match the owned block-row union")


def _build_global_jacobian_diagnostics(
    *,
    liquid: LiquidJacobianResult,
    interface: InterfaceJacobianResult,
    gas: GasJacobianResult,
    values: np.ndarray,
    cache_size: int,
    cache_stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "n_triplets_total": int(values.size),
        "n_rows_liquid": int(liquid.owned_liq_rows_global.size),
        "n_rows_interface": int(interface.owned_interface_rows_global.size),
        "n_rows_gas": int(gas.owned_gas_rows_global.size),
        "nnz_liquid": int(liquid.values.size),
        "nnz_interface": int(interface.values.size),
        "nnz_gas": int(gas.values.size),
        "cache_size": int(cache_size),
        "cache_hits": int(cache_stats["hits"]),
        "cache_misses": int(cache_stats["misses"]),
        "max_abs_value": float(np.max(np.abs(values))) if values.size else 0.0,
    }


def assemble_global_jacobian(
    *,
    u_trial_layout: np.ndarray,
    layout: UnknownLayout,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    ownership: JacobianOwnership,
    pattern: JacobianPattern,
    build_all_residual_blocks_from_layout_vector: Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
    fd_options: GlobalJacobianFDOptions | None = None,
) -> GlobalJacobianResult:
    """Assemble layout-global Jacobian COO triplets from frozen block Jacobians."""

    options = fd_options if fd_options is not None else GlobalJacobianFDOptions()
    normalized_ownership = _validate_global_jacobian_inputs(
        u_trial_layout=u_trial_layout,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=build_all_residual_blocks_from_layout_vector,
        fd_options=options,
    )
    trial = _as_1d_float_array("u_trial_layout", u_trial_layout, expected_size=layout.total_size)

    evaluate_all, cache_stats = _make_shared_block_residual_evaluator(
        build_all_residual_blocks_from_layout_vector=build_all_residual_blocks_from_layout_vector,
        n_global=layout.total_size,
        cache_size=int(options.cache_size),
    )
    eval_liquid, eval_interface, eval_gas = _make_block_callbacks(evaluate_all)

    liquid = assemble_liquid_jacobian(
        u_trial_layout=trial,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_liquid_residual_from_layout_vector=eval_liquid,
        owned_liq_cells=normalized_ownership.owned_liq_cells,
        fd_options=options.liquid,
    )
    interface = assemble_interface_jacobian(
        u_trial_layout=trial,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=eval_interface,
        owner_active=normalized_ownership.interface_owner_active,
        fd_options=options.interface,
    )
    gas = assemble_gas_jacobian(
        u_trial_layout=trial,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_gas_residual_from_layout_vector=eval_gas,
        owned_gas_cells=normalized_ownership.owned_gas_cells,
        fd_options=options.gas,
    )

    rows_global = np.concatenate([liquid.rows_global, interface.rows_global, gas.rows_global])
    cols_global = np.concatenate([liquid.cols_global, interface.cols_global, gas.cols_global])
    values = np.concatenate([liquid.values, interface.values, gas.values])
    expected_rows_global = np.concatenate(
        [
            liquid.owned_liq_rows_global,
            interface.owned_interface_rows_global,
            gas.owned_gas_rows_global,
        ]
    )
    _validate_global_triplets_against_pattern(
        rows_global=rows_global,
        cols_global=cols_global,
        values=values,
        pattern=pattern,
        expected_rows_global=expected_rows_global,
    )

    diagnostics = _build_global_jacobian_diagnostics(
        liquid=liquid,
        interface=interface,
        gas=gas,
        values=values,
        cache_size=int(options.cache_size),
        cache_stats=cache_stats,
    )
    return GlobalJacobianResult(
        rows_global=rows_global,
        cols_global=cols_global,
        values=values,
        liquid=liquid,
        interface=interface,
        gas=gas,
        diagnostics=diagnostics,
    )


def _insert_triplets_into_petsc_mat(
    *,
    mat: Any,
    rows_global: np.ndarray,
    cols_global: np.ndarray,
    values: np.ndarray,
    layout_to_petsc_perm: np.ndarray,
    zero_before_insert: bool,
    assembly_flush: bool,
) -> None:
    perm = np.asarray(layout_to_petsc_perm, dtype=np.int64)
    if perm.ndim != 1:
        raise GlobalJacobianAssemblyError("layout_to_petsc_perm must be a validated 1D permutation")
    rows_petsc = perm[rows_global] if rows_global.size else np.zeros(0, dtype=np.int64)
    cols_petsc = perm[cols_global] if cols_global.size else np.zeros(0, dtype=np.int64)

    if zero_before_insert and hasattr(mat, "zeroEntries"):
        mat.zeroEntries()

    if rows_petsc.size == 0:
        if assembly_flush and hasattr(mat, "assemblyBegin") and hasattr(mat, "assemblyEnd"):
            mat.assemblyBegin()
            mat.assemblyEnd()
        return

    if hasattr(mat, "setValues"):
        for row in np.unique(rows_petsc):
            mask = rows_petsc == row
            mat.setValues(int(row), cols_petsc[mask], values[mask])
    elif hasattr(mat, "setValue"):
        for row, col, val in zip(rows_petsc.tolist(), cols_petsc.tolist(), values.tolist()):
            mat.setValue(int(row), int(col), float(val))
    else:
        raise GlobalJacobianAssemblyError("PETSc Mat must support setValue or setValues for Jacobian insertion")

    if assembly_flush and hasattr(mat, "assemblyBegin") and hasattr(mat, "assemblyEnd"):
        mat.assemblyBegin()
        mat.assemblyEnd()


def assemble_and_insert_global_jacobian(
    *,
    u_trial_layout: np.ndarray,
    layout: UnknownLayout,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    ownership: JacobianOwnership,
    pattern: JacobianPattern,
    build_all_residual_blocks_from_layout_vector: Callable[[np.ndarray], tuple[LiquidResidualResult, InterfaceResidualResult, GasResidualResult]],
    mat: Any,
    PETSc,
    layout_to_petsc: np.ndarray | None = None,
    fd_options: GlobalJacobianFDOptions | None = None,
    zero_before_insert: bool = True,
    assembly_flush: bool = True,
) -> PETScGlobalJacobianResult:
    """Assemble global Jacobian triplets and insert them into a preallocated PETSc Mat."""

    _ = PETSc
    triplets = assemble_global_jacobian(
        u_trial_layout=u_trial_layout,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=build_all_residual_blocks_from_layout_vector,
        fd_options=fd_options,
    )
    perm = _validate_layout_to_petsc_mapping(layout_to_petsc, n_global=layout.total_size)
    _insert_triplets_into_petsc_mat(
        mat=mat,
        rows_global=triplets.rows_global,
        cols_global=triplets.cols_global,
        values=triplets.values,
        layout_to_petsc_perm=perm,
        zero_before_insert=bool(zero_before_insert),
        assembly_flush=bool(assembly_flush),
    )
    diagnostics = dict(triplets.diagnostics)
    diagnostics["identity_mapping"] = bool(np.array_equal(perm, np.arange(layout.total_size, dtype=np.int64)))
    diagnostics["zero_before_insert"] = bool(zero_before_insert)
    diagnostics["assembly_flush"] = bool(assembly_flush)
    return PETScGlobalJacobianResult(mat=mat, triplets=triplets, diagnostics=diagnostics)


__all__ = [
    "GlobalJacobianAssemblyError",
    "GlobalJacobianFDOptions",
    "GlobalJacobianResult",
    "JacobianOwnership",
    "PETScGlobalJacobianResult",
    "assemble_and_insert_global_jacobian",
    "assemble_global_jacobian",
]
