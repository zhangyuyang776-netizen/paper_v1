from __future__ import annotations

"""PETSc AIJ/MPIAIJ preallocation from a frozen global Jacobian pattern.

This module translates the explicit CSR sparsity graph from jacobian_pattern.py
into per-row diagonal/off-diagonal nonzero counts for PETSc matrix
preallocation. It must not infer sparsity from runtime residual evaluations or
rebuild a second pattern truth source.

Forbidden here:
- rebuilding sparsity from residual/state/property values
- introducing Rd or a second gas-side mpp row
- creating a second layout ordering distinct from JacobianPattern
- allowing dynamic new nonzero allocation to hide pattern defects
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from assembly.jacobian_pattern import JacobianPattern


class PETScPreallocError(ValueError):
    """Raised when PETSc preallocation inputs are inconsistent."""


@dataclass(frozen=True)
class PreallocCounts:
    n_global: int
    n_local_rows: int
    local_rows_layout: np.ndarray
    local_rows_petsc: np.ndarray
    d_nz: np.ndarray
    o_nz: np.ndarray
    local_row_cols_layout: tuple[np.ndarray, ...]
    local_row_cols_petsc: tuple[np.ndarray, ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PETScPreallocResult:
    mat: Any
    counts: PreallocCounts
    diagnostics: dict[str, Any]


def _as_1d_int_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim != 1:
        raise PETScPreallocError(f"{name} must be a 1D integer array")
    if expected_size is not None and arr.shape[0] != expected_size:
        raise PETScPreallocError(f"{name} must have length {expected_size}")
    return arr


def _normalize_ownership_ranges(
    ownership_ranges: Any,
    *,
    n_global: int,
) -> np.ndarray:
    ranges = np.asarray(ownership_ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise PETScPreallocError("ownership_ranges must have shape (nranks, 2)")
    if ranges.shape[0] == 0:
        raise PETScPreallocError("ownership_ranges must not be empty")
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] < 0):
        raise PETScPreallocError("ownership_ranges must be non-negative")
    if np.any(ranges[:, 1] < ranges[:, 0]):
        raise PETScPreallocError("ownership_ranges must satisfy stop >= start")
    if int(ranges[0, 0]) != 0:
        raise PETScPreallocError("ownership_ranges must start at row 0")
    if int(ranges[-1, 1]) != int(n_global):
        raise PETScPreallocError("ownership_ranges must end at n_global")
    if np.any(ranges[1:, 0] != ranges[:-1, 1]):
        raise PETScPreallocError("ownership_ranges must be contiguous and non-overlapping")
    return ranges


def _build_owner_map_from_ranges(
    ownership_ranges: np.ndarray,
    *,
    n_global: int,
) -> np.ndarray:
    owner_map = np.full(n_global, -1, dtype=np.int64)
    for rank, (start, stop) in enumerate(ownership_ranges.tolist()):
        owner_map[start:stop] = rank
    if np.any(owner_map < 0):
        raise PETScPreallocError("ownership_ranges must cover all global rows exactly once")
    return owner_map


def _identity_permutation(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def _validate_mapping(
    layout_to_petsc: Any | None,
    *,
    n_global: int,
) -> np.ndarray:
    if layout_to_petsc is None:
        return _identity_permutation(n_global)
    perm = _as_1d_int_array("layout_to_petsc", layout_to_petsc, expected_size=n_global)
    if np.any(perm < 0) or np.any(perm >= n_global):
        raise PETScPreallocError("layout_to_petsc must contain indices in [0, n_global)")
    if len(np.unique(perm)) != n_global:
        raise PETScPreallocError("layout_to_petsc must be a permutation of 0..n_global-1")
    return perm


def _extract_local_pattern_rows(
    *,
    pattern: JacobianPattern,
    ownership_range: tuple[int, int],
    layout_to_petsc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    rstart, rend = int(ownership_range[0]), int(ownership_range[1])
    n_global = int(pattern.shape[0])
    if not (0 <= rstart <= rend <= n_global):
        raise PETScPreallocError("ownership_range must satisfy 0 <= start <= stop <= n_global")

    local_rows_petsc = np.arange(rstart, rend, dtype=np.int64)
    petsc_to_layout = np.empty(n_global, dtype=np.int64)
    petsc_to_layout[layout_to_petsc] = np.arange(n_global, dtype=np.int64)
    local_rows_layout = petsc_to_layout[local_rows_petsc]

    local_row_cols_layout: list[np.ndarray] = []
    local_row_cols_petsc: list[np.ndarray] = []
    for row_layout in local_rows_layout.tolist():
        cols_layout = np.asarray(
            pattern.indices[pattern.indptr[row_layout] : pattern.indptr[row_layout + 1]],
            dtype=np.int64,
        )
        if not np.any(cols_layout == row_layout):
            cols_layout = np.unique(np.concatenate([cols_layout, np.asarray([row_layout], dtype=np.int64)]))
        else:
            cols_layout = np.unique(cols_layout)
        cols_petsc = np.unique(layout_to_petsc[cols_layout])
        local_row_cols_layout.append(cols_layout)
        local_row_cols_petsc.append(cols_petsc)

    return (
        local_rows_layout,
        local_rows_petsc,
        tuple(local_row_cols_layout),
        tuple(local_row_cols_petsc),
    )


def _count_diag_offdiag_nnz(
    *,
    local_row_cols_petsc: tuple[np.ndarray, ...],
    owner_map: np.ndarray,
    my_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_local = len(local_row_cols_petsc)
    d_nz = np.zeros(n_local, dtype=np.int64)
    o_nz = np.zeros(n_local, dtype=np.int64)
    for i, cols in enumerate(local_row_cols_petsc):
        col_owners = owner_map[cols]
        d_nz[i] = int(np.count_nonzero(col_owners == my_rank))
        o_nz[i] = int(cols.size - d_nz[i])
    return d_nz, o_nz


def _build_prealloc_diagnostics(
    *,
    n_global: int,
    ownership_range: tuple[int, int],
    d_nz: np.ndarray,
    o_nz: np.ndarray,
    layout_to_petsc: np.ndarray,
) -> dict[str, Any]:
    return {
        "n_global": int(n_global),
        "n_local_rows": int(d_nz.size),
        "ownership_range": (int(ownership_range[0]), int(ownership_range[1])),
        "nnz_local_total": int(np.sum(d_nz) + np.sum(o_nz)),
        "diag_nnz_local": int(np.sum(d_nz)),
        "offdiag_nnz_local": int(np.sum(o_nz)),
        "max_d_nz_row": int(np.max(d_nz)) if d_nz.size else 0,
        "max_o_nz_row": int(np.max(o_nz)) if o_nz.size else 0,
        "identity_mapping": bool(np.array_equal(layout_to_petsc, _identity_permutation(n_global))),
    }


def _validate_existing_mat(*, mat: Any, counts: PreallocCounts) -> None:
    if not hasattr(mat, "getSize"):
        raise PETScPreallocError("existing PETSc Mat must support getSize for reuse validation")
    global_size = tuple(int(v) for v in mat.getSize())
    expected_global = (int(counts.n_global), int(counts.n_global))
    if global_size != expected_global:
        raise PETScPreallocError(
            f"existing PETSc Mat global size {global_size} does not match expected {expected_global}"
        )

    if hasattr(mat, "getLocalSize"):
        local_size = tuple(int(v) for v in mat.getLocalSize())
        if local_size[0] != int(counts.n_local_rows):
            raise PETScPreallocError(
                "existing PETSc Mat local row size does not match preallocation counts"
            )

    if hasattr(mat, "getOwnershipRange"):
        ownership = tuple(int(v) for v in mat.getOwnershipRange())
        expected_ownership = tuple(int(v) for v in counts.diagnostics["ownership_range"])
        if ownership != expected_ownership:
            raise PETScPreallocError(
                f"existing PETSc Mat ownership range {ownership} does not match expected {expected_ownership}"
            )


def build_petsc_prealloc_counts(
    *,
    pattern: JacobianPattern,
    ownership_range: tuple[int, int],
    ownership_ranges: np.ndarray,
    layout_to_petsc: np.ndarray | None = None,
) -> PreallocCounts:
    if pattern.shape[0] != pattern.shape[1]:
        raise PETScPreallocError("pattern must be square")
    n_global = int(pattern.shape[0])
    if pattern.indptr.shape != (n_global + 1,):
        raise PETScPreallocError("pattern.indptr must have shape (n_global + 1,)")
    if np.any(pattern.indptr[1:] < pattern.indptr[:-1]):
        raise PETScPreallocError("pattern.indptr must be nondecreasing")
    if int(pattern.indptr[-1]) != int(pattern.indices.size):
        raise PETScPreallocError("pattern.indptr[-1] must equal pattern.indices.size")
    if np.any(pattern.indices < 0) or np.any(pattern.indices >= n_global):
        raise PETScPreallocError("pattern.indices must lie in [0, n_global)")

    ranges = _normalize_ownership_ranges(ownership_ranges, n_global=n_global)
    rstart, rend = int(ownership_range[0]), int(ownership_range[1])
    if not np.any((ranges[:, 0] == rstart) & (ranges[:, 1] == rend)):
        raise PETScPreallocError("ownership_range must match one row in ownership_ranges")
    matches = np.flatnonzero((ranges[:, 0] == rstart) & (ranges[:, 1] == rend))
    if matches.size != 1:
        raise PETScPreallocError("ownership_range must match exactly one row in ownership_ranges")
    my_rank = int(matches[0])

    perm = _validate_mapping(layout_to_petsc, n_global=n_global)
    owner_map = _build_owner_map_from_ranges(ranges, n_global=n_global)
    local_rows_layout, local_rows_petsc, local_row_cols_layout, local_row_cols_petsc = _extract_local_pattern_rows(
        pattern=pattern,
        ownership_range=(rstart, rend),
        layout_to_petsc=perm,
    )
    d_nz, o_nz = _count_diag_offdiag_nnz(
        local_row_cols_petsc=local_row_cols_petsc,
        owner_map=owner_map,
        my_rank=my_rank,
    )
    diagnostics = _build_prealloc_diagnostics(
        n_global=n_global,
        ownership_range=(rstart, rend),
        d_nz=d_nz,
        o_nz=o_nz,
        layout_to_petsc=perm,
    )
    return PreallocCounts(
        n_global=n_global,
        n_local_rows=int(local_rows_layout.size),
        local_rows_layout=local_rows_layout,
        local_rows_petsc=local_rows_petsc,
        d_nz=d_nz,
        o_nz=o_nz,
        local_row_cols_layout=local_row_cols_layout,
        local_row_cols_petsc=local_row_cols_petsc,
        diagnostics=diagnostics,
    )


def create_or_preallocate_petsc_mat(
    *,
    counts: PreallocCounts,
    PETSc,
    comm,
    mat: Any | None = None,
    new_nonzero_allocation_err: bool = True,
) -> PETScPreallocResult:
    if counts.d_nz.shape != (counts.n_local_rows,) or counts.o_nz.shape != (counts.n_local_rows,):
        raise PETScPreallocError("counts.d_nz and counts.o_nz must have shape (n_local_rows,)")

    option = PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR
    mat_was_none = mat is None
    if mat is None:
        mat = PETSc.Mat().createAIJ(
            size=((counts.n_local_rows, counts.n_global), (counts.n_local_rows, counts.n_global)),
            nnz=(counts.d_nz, counts.o_nz),
            comm=comm,
        )
    else:
        _validate_existing_mat(mat=mat, counts=counts)
        if hasattr(mat, "zeroEntries"):
            mat.zeroEntries()
        if not hasattr(mat, "setPreallocationNNZ"):
            raise PETScPreallocError("existing PETSc Mat must support setPreallocationNNZ for reuse")
        mat.setPreallocationNNZ((counts.d_nz, counts.o_nz))
        if hasattr(mat, "setUp"):
            mat.setUp()

    if hasattr(mat, "_set_allowed_slots"):
        mat._set_allowed_slots(counts.local_rows_petsc, counts.local_row_cols_petsc)
    if hasattr(mat, "setOption"):
        mat.setOption(option, bool(new_nonzero_allocation_err))
    if hasattr(mat, "setUp"):
        mat.setUp()

    diagnostics = dict(counts.diagnostics)
    diagnostics["new_matrix_created"] = bool(mat_was_none)
    return PETScPreallocResult(mat=mat, counts=counts, diagnostics=diagnostics)


def build_petsc_prealloc(
    *,
    pattern: JacobianPattern,
    ownership_range: tuple[int, int],
    ownership_ranges: np.ndarray,
    PETSc,
    comm,
    mat: Any | None = None,
    layout_to_petsc: np.ndarray | None = None,
) -> PETScPreallocResult:
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=ownership_range,
        ownership_ranges=ownership_ranges,
        layout_to_petsc=layout_to_petsc,
    )
    return create_or_preallocate_petsc_mat(
        counts=counts,
        PETSc=PETSc,
        comm=comm,
        mat=mat,
        new_nonzero_allocation_err=True,
    )


__all__ = [
    "PETScPreallocError",
    "PETScPreallocResult",
    "PreallocCounts",
    "build_petsc_prealloc",
    "build_petsc_prealloc_counts",
    "create_or_preallocate_petsc_mat",
]
