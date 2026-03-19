from __future__ import annotations

import numpy as np

from assembly.jacobian_pattern import JacobianPattern
from assembly.petsc_prealloc import (
    PETScPreallocError,
    build_petsc_prealloc,
    build_petsc_prealloc_counts,
    create_or_preallocate_petsc_mat,
)


def _make_pattern() -> JacobianPattern:
    row_sets = [
        [0, 1, 4],
        [0, 1, 2],
        [1, 2, 3, 5],
        [2, 3, 4],
        [0, 3, 4, 5],
        [2, 4, 5],
    ]
    indptr = [0]
    indices = []
    row_nnz = []
    for cols in row_sets:
        cols_sorted = sorted(set(cols))
        indices.extend(cols_sorted)
        row_nnz.append(len(cols_sorted))
        indptr.append(len(indices))
    return JacobianPattern(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        shape=(6, 6),
        row_nnz=np.asarray(row_nnz, dtype=np.int64),
        diagnostics={},
    )


class _FakeMat:
    class Option:
        NEW_NONZERO_ALLOCATION_ERR = "NEW_NONZERO_ALLOCATION_ERR"

    def __init__(self) -> None:
        self.size = None
        self.nnz = None
        self.comm = None
        self.options = {}
        self.allowed_slots: dict[int, set[int]] = {}
        self.reused = False
        self.values: dict[tuple[int, int], float] = {}
        self.ownership_range = (0, 0)

    def createAIJ(self, *, size, nnz, comm=None):
        self.size = size
        self.nnz = nnz
        self.comm = comm
        return self

    def setPreallocationNNZ(self, nnz):
        self.nnz = nnz
        self.reused = True

    def zeroEntries(self):
        self.values.clear()

    def setUp(self):
        return None

    def setOption(self, option, value):
        self.options[option] = bool(value)

    def getSize(self):
        if self.size is None:
            raise RuntimeError("size not set")
        rows = self.size[0][1]
        cols = self.size[1][1]
        return (rows, cols)

    def getLocalSize(self):
        if self.size is None:
            raise RuntimeError("size not set")
        return (int(self.size[0][0]), int(self.size[1][0]))

    def getOwnershipRange(self):
        return tuple(int(v) for v in self.ownership_range)

    def _set_allowed_slots(self, rows, row_cols):
        self.allowed_slots = {int(r): set(np.asarray(c, dtype=np.int64).tolist()) for r, c in zip(rows, row_cols)}
        if len(rows) == 0:
            self.ownership_range = (0, 0)
        else:
            rows_arr = np.asarray(rows, dtype=np.int64)
            self.ownership_range = (int(np.min(rows_arr)), int(np.max(rows_arr)) + 1)

    def setValue(self, row: int, col: int, value: float):
        allowed = self.allowed_slots.get(int(row), set())
        err_on_new = self.options.get(self.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        if err_on_new and int(col) not in allowed:
            raise RuntimeError("new nonzero allocation error")
        self.values[(int(row), int(col))] = float(value)


class _FakePETSc:
    Mat = _FakeMat


def test_serial_identity_mapping_counts_match_pattern_row_nnz() -> None:
    pattern = _make_pattern()
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=(0, 6),
        ownership_ranges=np.asarray([[0, 6]], dtype=np.int64),
        layout_to_petsc=None,
    )

    assert counts.n_local_rows == 6
    assert np.array_equal(counts.local_rows_layout, np.arange(6, dtype=np.int64))
    assert np.array_equal(counts.local_rows_petsc, np.arange(6, dtype=np.int64))
    assert np.array_equal(counts.d_nz, pattern.row_nnz)
    assert np.array_equal(counts.o_nz, np.zeros(6, dtype=np.int64))
    assert int(np.sum(counts.d_nz)) == int(pattern.indices.size)


def test_mpi_like_ownership_counts_diag_and_offdiag_correctly() -> None:
    pattern = _make_pattern()
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=(2, 4),
        ownership_ranges=np.asarray([[0, 2], [2, 4], [4, 6]], dtype=np.int64),
        layout_to_petsc=None,
    )

    assert np.array_equal(counts.local_rows_petsc, np.asarray([2, 3], dtype=np.int64))
    assert np.array_equal(counts.local_rows_layout, np.asarray([2, 3], dtype=np.int64))
    assert np.array_equal(counts.d_nz, np.asarray([2, 2], dtype=np.int64))
    assert np.array_equal(counts.o_nz, np.asarray([2, 1], dtype=np.int64))
    assert np.array_equal(counts.d_nz + counts.o_nz, np.asarray([4, 3], dtype=np.int64))


def test_non_identity_permutation_maps_local_columns_correctly() -> None:
    pattern = _make_pattern()
    perm = np.asarray([3, 4, 0, 1, 5, 2], dtype=np.int64)
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=(0, 3),
        ownership_ranges=np.asarray([[0, 3], [3, 6]], dtype=np.int64),
        layout_to_petsc=perm,
    )

    assert np.array_equal(counts.local_rows_petsc, np.asarray([0, 1, 2], dtype=np.int64))
    assert np.array_equal(counts.local_rows_layout, np.asarray([2, 3, 5], dtype=np.int64))
    for cols_layout, cols_petsc in zip(counts.local_row_cols_layout, counts.local_row_cols_petsc):
        assert np.array_equal(cols_petsc, np.unique(perm[cols_layout]))
    assert np.all(counts.d_nz + counts.o_nz == np.asarray([4, 3, 3], dtype=np.int64))
    assert np.any(counts.o_nz > 0)


def test_new_matrix_path_and_reuse_path_work() -> None:
    pattern = _make_pattern()
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=(0, 6),
        ownership_ranges=np.asarray([[0, 6]], dtype=np.int64),
        layout_to_petsc=None,
    )

    result_new = create_or_preallocate_petsc_mat(
        counts=counts,
        PETSc=_FakePETSc,
        comm=None,
        mat=None,
        new_nonzero_allocation_err=True,
    )
    assert result_new.diagnostics["new_matrix_created"] is True
    assert result_new.mat.getSize() == (6, 6)
    assert result_new.mat.options[_FakeMat.Option.NEW_NONZERO_ALLOCATION_ERR] is True

    result_reuse = create_or_preallocate_petsc_mat(
        counts=counts,
        PETSc=_FakePETSc,
        comm=None,
        mat=result_new.mat,
        new_nonzero_allocation_err=True,
    )
    assert result_reuse.diagnostics["new_matrix_created"] is False
    assert result_reuse.mat.reused is True


def test_reuse_path_rejects_mismatched_existing_matrix() -> None:
    pattern = _make_pattern()
    counts = build_petsc_prealloc_counts(
        pattern=pattern,
        ownership_range=(0, 6),
        ownership_ranges=np.asarray([[0, 6]], dtype=np.int64),
        layout_to_petsc=None,
    )
    bad_mat = _FakePETSc.Mat().createAIJ(size=((5, 5), (5, 5)), nnz=(np.ones(5, dtype=np.int64), np.zeros(5, dtype=np.int64)), comm=None)

    try:
        create_or_preallocate_petsc_mat(
            counts=counts,
            PETSc=_FakePETSc,
            comm=None,
            mat=bad_mat,
            new_nonzero_allocation_err=True,
        )
    except PETScPreallocError as exc:
        assert "global size" in str(exc)
    else:
        raise AssertionError("mismatched existing matrix must raise PETScPreallocError")


def test_fake_fill_respects_preallocated_pattern_slots() -> None:
    pattern = _make_pattern()
    result = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=(0, 6),
        ownership_ranges=np.asarray([[0, 6]], dtype=np.int64),
        PETSc=_FakePETSc,
        comm=None,
        mat=None,
        layout_to_petsc=None,
    )

    for row, cols in zip(result.counts.local_rows_petsc.tolist(), result.counts.local_row_cols_petsc):
        for col in cols.tolist():
            result.mat.setValue(row, int(col), 1.0)

    outside = next(col for col in range(pattern.shape[0]) if col not in set(result.counts.local_row_cols_petsc[0].tolist()))
    try:
        result.mat.setValue(int(result.counts.local_rows_petsc[0]), int(outside), 1.0)
    except RuntimeError as exc:
        assert "new nonzero allocation error" in str(exc)
    else:
        raise AssertionError("writing outside the preallocated pattern must fail")


def test_invalid_mapping_raises_clean_error() -> None:
    pattern = _make_pattern()
    bad_perm = np.asarray([0, 1, 1, 3, 4, 5], dtype=np.int64)
    try:
        build_petsc_prealloc_counts(
            pattern=pattern,
            ownership_range=(0, 6),
            ownership_ranges=np.asarray([[0, 6]], dtype=np.int64),
            layout_to_petsc=bad_perm,
        )
    except PETScPreallocError as exc:
        assert "permutation" in str(exc)
    else:
        raise AssertionError("invalid permutation must raise PETScPreallocError")
