from __future__ import annotations

"""Sparse finite-difference Jacobian assembly for gas bulk residual rows.

This module assembles layout-global COO triplets for gas residual rows only.
It follows the exact current-geometry gas residual evaluation chain via a
caller-provided callback, and restricts numerical differentiation to the
supports allowed by the frozen JacobianPattern.

Forbidden here:
- re-deriving an alternative gas / interface / far-field evaluation chain
- writing directly to PETSc Mat
- introducing Rd or a second gas-side mpp row
- emitting triplets outside the JacobianPattern support
- introducing far-field pseudo-unknown columns
"""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from assembly.jacobian_pattern import JacobianPattern
from assembly.residual_gas import GasResidualResult
from core.layout import UnknownLayout
from core.types import Mesh1D, SpeciesMaps


class GasJacobianAssemblyError(ValueError):
    """Raised when gas Jacobian assembly inputs are inconsistent."""


@dataclass(frozen=True)
class GasJacobianFDOptions:
    rel_step: float = 1.0e-7
    abs_step_temperature: float = 1.0e-8
    abs_step_species: float = 1.0e-10
    abs_step_interface: float = 1.0e-10
    abs_step_bulk: float = 1.0e-10
    abs_step_mpp: float = 1.0e-10
    max_shrink: int = 8
    shrink_factor: float = 0.5
    use_central: bool = False


@dataclass(frozen=True)
class GasJacobianColumnPlan:
    col_global: int
    affected_rows_global: np.ndarray
    step: float
    variable_kind: str


@dataclass(frozen=True)
class GasJacobianResult:
    rows_global: np.ndarray
    cols_global: np.ndarray
    values: np.ndarray
    owned_gas_rows_global: np.ndarray
    candidate_cols_global: np.ndarray
    diagnostics: dict[str, Any]


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise GasJacobianAssemblyError(f"{name} must be a 1D float array")
    if not np.all(np.isfinite(arr)):
        raise GasJacobianAssemblyError(f"{name} must contain only finite values")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise GasJacobianAssemblyError(f"{name} must have length {expected_size}")
    return arr


def _validate_fd_options(fd_options: GasJacobianFDOptions) -> None:
    if not np.isfinite(fd_options.rel_step) or fd_options.rel_step <= 0.0:
        raise GasJacobianAssemblyError("fd_options.rel_step must be finite and > 0")
    if not np.isfinite(fd_options.abs_step_temperature) or fd_options.abs_step_temperature <= 0.0:
        raise GasJacobianAssemblyError("fd_options.abs_step_temperature must be finite and > 0")
    if not np.isfinite(fd_options.abs_step_species) or fd_options.abs_step_species <= 0.0:
        raise GasJacobianAssemblyError("fd_options.abs_step_species must be finite and > 0")
    if not np.isfinite(fd_options.abs_step_interface) or fd_options.abs_step_interface <= 0.0:
        raise GasJacobianAssemblyError("fd_options.abs_step_interface must be finite and > 0")
    if not np.isfinite(fd_options.abs_step_bulk) or fd_options.abs_step_bulk <= 0.0:
        raise GasJacobianAssemblyError("fd_options.abs_step_bulk must be finite and > 0")
    if not np.isfinite(fd_options.abs_step_mpp) or fd_options.abs_step_mpp <= 0.0:
        raise GasJacobianAssemblyError("fd_options.abs_step_mpp must be finite and > 0")
    if not isinstance(fd_options.max_shrink, (int, np.integer)) or fd_options.max_shrink < 0:
        raise GasJacobianAssemblyError("fd_options.max_shrink must be an integer >= 0")
    if not np.isfinite(fd_options.shrink_factor) or not (0.0 < fd_options.shrink_factor < 1.0):
        raise GasJacobianAssemblyError("fd_options.shrink_factor must be finite and lie in (0, 1)")


def _normalize_owned_gas_cells(mesh: Mesh1D, owned_gas_cells: Any | None) -> np.ndarray:
    if owned_gas_cells is None:
        return np.arange(mesh.n_gas, dtype=np.int64)
    owned = np.asarray(owned_gas_cells, dtype=np.int64)
    if owned.ndim != 1:
        raise GasJacobianAssemblyError("owned_gas_cells must be a 1D integer array")
    if owned.size == 0:
        return np.zeros(0, dtype=np.int64)
    if np.any(owned < 0) or np.any(owned >= mesh.n_gas):
        raise GasJacobianAssemblyError("owned_gas_cells contains out-of-range gas cell indices")
    if np.unique(owned).size != owned.size:
        raise GasJacobianAssemblyError("owned_gas_cells must not contain duplicates")
    return owned


def _owned_gas_rows_from_cells(layout: UnknownLayout, owned_gas_cells: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    for cell in owned_gas_cells.tolist():
        rows.append(layout.gas_temperature_index(int(cell)))
        species_slice = layout.gas_species_slice_for_cell(int(cell))
        rows.extend(range(species_slice.start, species_slice.stop))
    return np.asarray(rows, dtype=np.int64)


def _extract_gas_column_support_from_pattern(
    *,
    pattern: JacobianPattern,
    owned_gas_rows_global: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    n_dof = int(pattern.shape[0])
    candidate_cols: set[int] = set()
    affected_rows: dict[int, list[int]] = {}
    for row in owned_gas_rows_global.tolist():
        if not (0 <= row < n_dof):
            raise GasJacobianAssemblyError("owned gas row lies outside JacobianPattern shape")
        cols = np.asarray(pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]], dtype=np.int64)
        for col in cols.tolist():
            candidate_cols.add(int(col))
            affected_rows.setdefault(int(col), []).append(int(row))
    candidate_cols_global = np.asarray(sorted(candidate_cols), dtype=np.int64)
    affected_rows_global = {
        col: np.asarray(sorted(set(rows)), dtype=np.int64) for col, rows in affected_rows.items()
    }
    return candidate_cols_global, affected_rows_global


def _classify_gas_jacobian_column(
    *,
    col_global: int,
    layout: UnknownLayout,
    mesh: Mesh1D,
) -> str:
    for gas_cell in range(mesh.n_gas):
        if col_global == layout.gas_temperature_index(gas_cell):
            return "gas_temperature"
        gas_slice = layout.gas_species_slice_for_cell(gas_cell)
        if gas_slice.start <= col_global < gas_slice.stop:
            return "gas_species"

    if layout.if_temperature_slice.start <= col_global < layout.if_temperature_slice.stop:
        return "if_temperature"
    if layout.if_liq_species_slice.start <= col_global < layout.if_liq_species_slice.stop:
        return "if_liq_species"
    if layout.if_gas_species_slice.start <= col_global < layout.if_gas_species_slice.stop:
        return "if_gas_species"
    if layout.if_mpp_slice.start <= col_global < layout.if_mpp_slice.stop:
        return "if_mpp"

    if mesh.n_liq > 0:
        liq_last = mesh.n_liq - 1
        if col_global == layout.liq_temperature_index(liq_last):
            return "liq_last_temperature"
        liq_last_slice = layout.liq_species_slice_for_cell(liq_last)
        if liq_last_slice.start <= col_global < liq_last_slice.stop:
            return "liq_last_species"

    if layout.liq_block.start <= col_global < layout.liq_block.stop:
        raise GasJacobianAssemblyError("gas Jacobian does not allow liq_other bulk columns")
    raise GasJacobianAssemblyError(f"column {col_global} is outside all recognized layout blocks")


def _choose_fd_step(
    *,
    x_value: float,
    variable_kind: str,
    fd_options: GasJacobianFDOptions,
) -> float:
    scale = max(1.0, abs(float(x_value)))
    if variable_kind in {"gas_temperature", "if_temperature", "liq_last_temperature"}:
        return max(fd_options.abs_step_temperature, fd_options.rel_step * scale)
    if variable_kind == "if_mpp":
        return max(fd_options.abs_step_mpp, fd_options.rel_step * scale)
    if variable_kind in {"if_liq_species", "if_gas_species"}:
        return max(fd_options.abs_step_interface, fd_options.rel_step * scale)
    return max(fd_options.abs_step_species if variable_kind == "gas_species" else fd_options.abs_step_bulk, fd_options.rel_step * scale)


def _residual_rows_to_map(
    result: GasResidualResult,
    *,
    expected_rows: np.ndarray,
) -> dict[int, float]:
    rows = np.asarray(result.rows_global, dtype=np.int64)
    values = _as_1d_float_array("gas residual values", result.values, expected_size=rows.size)
    if rows.size != np.unique(rows).size:
        raise GasJacobianAssemblyError("gas residual callback returned duplicate row indices")
    row_map = {int(row): float(val) for row, val in zip(rows.tolist(), values.tolist())}
    if set(row_map.keys()) != set(expected_rows.tolist()):
        raise GasJacobianAssemblyError("gas residual callback must return exactly the requested owned gas rows")
    return row_map


def _safe_perturb_and_evaluate(
    *,
    u_trial_layout: np.ndarray,
    col_global: int,
    step: float,
    preferred_sign: float,
    allow_sign_fallback: bool,
    evaluate_gas_residual_from_layout_vector: Callable[[np.ndarray], GasResidualResult],
    expected_rows: np.ndarray,
    fd_options: GasJacobianFDOptions,
) -> tuple[GasResidualResult, float]:
    if step <= 0.0 or not np.isfinite(step):
        raise GasJacobianAssemblyError("finite-difference step must be finite and > 0")
    if preferred_sign not in (-1.0, 1.0):
        raise GasJacobianAssemblyError("preferred_sign must be either +1.0 or -1.0")

    last_error: Exception | None = None
    trial_step = float(step)
    n_shrink = int(fd_options.max_shrink)
    for _ in range(n_shrink + 1):
        signs = (preferred_sign, -preferred_sign) if allow_sign_fallback else (preferred_sign,)
        for sign in signs:
            u_pert = np.array(u_trial_layout, dtype=np.float64, copy=True)
            u_pert[col_global] = float(u_pert[col_global] + sign * trial_step)
            try:
                result = evaluate_gas_residual_from_layout_vector(u_pert)
                _residual_rows_to_map(result, expected_rows=expected_rows)
                if not np.all(np.isfinite(np.asarray(result.values, dtype=np.float64))):
                    raise GasJacobianAssemblyError("perturbed gas residual contains non-finite values")
                return result, sign * trial_step
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        trial_step *= float(fd_options.shrink_factor)

    raise GasJacobianAssemblyError(
        f"failed to evaluate perturbed gas residual for column {col_global}"
    ) from last_error


def _assemble_gas_jacobian_fd(
    *,
    u_trial_layout: np.ndarray,
    layout: UnknownLayout,
    mesh: Mesh1D,
    evaluate_gas_residual_from_layout_vector: Callable[[np.ndarray], GasResidualResult],
    owned_gas_rows_global: np.ndarray,
    candidate_cols_global: np.ndarray,
    affected_rows_by_col: dict[int, np.ndarray],
    fd_options: GasJacobianFDOptions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[GasJacobianColumnPlan], dict[int, dict[str, Any]]]:
    baseline_result = evaluate_gas_residual_from_layout_vector(np.array(u_trial_layout, dtype=np.float64, copy=True))
    baseline_map = _residual_rows_to_map(baseline_result, expected_rows=owned_gas_rows_global)

    rows_out: list[int] = []
    cols_out: list[int] = []
    values_out: list[float] = []
    plans: list[GasJacobianColumnPlan] = []
    step_stats: dict[int, dict[str, Any]] = {}

    for col in candidate_cols_global.tolist():
        variable_kind = _classify_gas_jacobian_column(col_global=int(col), layout=layout, mesh=mesh)
        step = _choose_fd_step(x_value=float(u_trial_layout[col]), variable_kind=variable_kind, fd_options=fd_options)
        affected_rows = affected_rows_by_col[int(col)]
        plans.append(
            GasJacobianColumnPlan(
                col_global=int(col),
                affected_rows_global=affected_rows.copy(),
                step=float(step),
                variable_kind=variable_kind,
            )
        )

        if fd_options.use_central:
            plus_result, plus_h = _safe_perturb_and_evaluate(
                u_trial_layout=u_trial_layout,
                col_global=int(col),
                step=float(step),
                preferred_sign=1.0,
                allow_sign_fallback=False,
                evaluate_gas_residual_from_layout_vector=evaluate_gas_residual_from_layout_vector,
                expected_rows=owned_gas_rows_global,
                fd_options=fd_options,
            )
            minus_result, minus_h = _safe_perturb_and_evaluate(
                u_trial_layout=u_trial_layout,
                col_global=int(col),
                step=float(step),
                preferred_sign=-1.0,
                allow_sign_fallback=False,
                evaluate_gas_residual_from_layout_vector=evaluate_gas_residual_from_layout_vector,
                expected_rows=owned_gas_rows_global,
                fd_options=fd_options,
            )
            plus_map = _residual_rows_to_map(plus_result, expected_rows=owned_gas_rows_global)
            minus_map = _residual_rows_to_map(minus_result, expected_rows=owned_gas_rows_global)
            denom = plus_h - minus_h
            if denom == 0.0:
                raise GasJacobianAssemblyError(f"central-difference denominator vanished for column {col}")
            for row in affected_rows.tolist():
                deriv = (plus_map[int(row)] - minus_map[int(row)]) / denom
                if not np.isfinite(deriv):
                    raise GasJacobianAssemblyError(
                        f"non-finite gas Jacobian entry produced for row {row}, column {col}"
                    )
                rows_out.append(int(row))
                cols_out.append(int(col))
                values_out.append(float(deriv))
            step_stats[int(col)] = {
                "step": float(max(abs(plus_h), abs(minus_h))),
                "kind": variable_kind,
                "plus_step": float(plus_h),
                "minus_step": float(minus_h),
            }
        else:
            pert_result, actual_h = _safe_perturb_and_evaluate(
                u_trial_layout=u_trial_layout,
                col_global=int(col),
                step=float(step),
                preferred_sign=1.0,
                allow_sign_fallback=True,
                evaluate_gas_residual_from_layout_vector=evaluate_gas_residual_from_layout_vector,
                expected_rows=owned_gas_rows_global,
                fd_options=fd_options,
            )
            pert_map = _residual_rows_to_map(pert_result, expected_rows=owned_gas_rows_global)
            for row in affected_rows.tolist():
                deriv = (pert_map[int(row)] - baseline_map[int(row)]) / float(actual_h)
                if not np.isfinite(deriv):
                    raise GasJacobianAssemblyError(
                        f"non-finite gas Jacobian entry produced for row {row}, column {col}"
                    )
                rows_out.append(int(row))
                cols_out.append(int(col))
                values_out.append(float(deriv))
            step_stats[int(col)] = {"step": float(abs(actual_h)), "kind": variable_kind}

    return (
        np.asarray(rows_out, dtype=np.int64),
        np.asarray(cols_out, dtype=np.int64),
        np.asarray(values_out, dtype=np.float64),
        plans,
        step_stats,
    )


def assemble_gas_jacobian(
    *,
    u_trial_layout: np.ndarray,
    layout: UnknownLayout,
    mesh: Mesh1D,
    species_maps: SpeciesMaps,
    pattern: JacobianPattern,
    evaluate_gas_residual_from_layout_vector: Callable[[np.ndarray], GasResidualResult],
    owned_gas_cells: Any | None = None,
    fd_options: GasJacobianFDOptions | None = None,
) -> GasJacobianResult:
    """Assemble sparse FD Jacobian triplets for gas residual rows."""

    trial = _as_1d_float_array("u_trial_layout", u_trial_layout, expected_size=layout.total_size)
    if pattern.shape != (layout.total_size, layout.total_size):
        raise GasJacobianAssemblyError("pattern.shape must match layout.total_size")
    if layout.n_liq_cells != mesh.n_liq or layout.n_gas_cells != mesh.n_gas:
        raise GasJacobianAssemblyError("layout cell counts must match mesh cell counts")
    if layout.n_liq_red != species_maps.n_liq_red or layout.n_gas_red != species_maps.n_gas_red:
        raise GasJacobianAssemblyError("layout reduced-species sizes must match species_maps")

    options = fd_options if fd_options is not None else GasJacobianFDOptions()
    _validate_fd_options(options)
    owned_cells = _normalize_owned_gas_cells(mesh, owned_gas_cells)
    owned_rows = _owned_gas_rows_from_cells(layout, owned_cells)
    candidate_cols, affected_rows_by_col = _extract_gas_column_support_from_pattern(
        pattern=pattern,
        owned_gas_rows_global=owned_rows,
    )

    rows_global, cols_global, values, plans, step_stats = _assemble_gas_jacobian_fd(
        u_trial_layout=trial,
        layout=layout,
        mesh=mesh,
        evaluate_gas_residual_from_layout_vector=evaluate_gas_residual_from_layout_vector,
        owned_gas_rows_global=owned_rows,
        candidate_cols_global=candidate_cols,
        affected_rows_by_col=affected_rows_by_col,
        fd_options=options,
    )

    for row, col in zip(rows_global.tolist(), cols_global.tolist()):
        row_cols = np.asarray(pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]], dtype=np.int64)
        if not np.any(row_cols == col):
            raise GasJacobianAssemblyError("gas Jacobian emitted a triplet outside the JacobianPattern support")

    diagnostics = {
        "n_owned_gas_rows": int(owned_rows.size),
        "n_candidate_cols": int(candidate_cols.size),
        "nnz_triplets": int(values.size),
        "fd_scheme": "central" if options.use_central else "forward",
        "failed_columns": [],
        "column_step_stats": step_stats,
        "max_abs_value": float(np.max(np.abs(values))) if values.size else 0.0,
        "candidate_cols_by_kind": {
            kind: [int(plan.col_global) for plan in plans if plan.variable_kind == kind]
            for kind in sorted({plan.variable_kind for plan in plans})
        },
        "owned_gas_cells": tuple(int(cell) for cell in owned_cells.tolist()),
    }
    return GasJacobianResult(
        rows_global=rows_global,
        cols_global=cols_global,
        values=values,
        owned_gas_rows_global=owned_rows,
        candidate_cols_global=candidate_cols,
        diagnostics=diagnostics,
    )


__all__ = [
    "GasJacobianAssemblyError",
    "GasJacobianColumnPlan",
    "GasJacobianFDOptions",
    "GasJacobianResult",
    "assemble_gas_jacobian",
]
