from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.layout import UnknownLayout
from solvers.nonlinear_context import build_nonlinear_context
from solvers.petsc_linear import (
    apply_structured_pc,
    build_fieldsplit_specs,
    build_linear_config_view,
)


def _make_layout() -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=2,
        n_gas_cells=3,
        n_liq_red=1,
        n_gas_red=2,
        liq_cell_width=2,
        gas_cell_width=3,
        n_if_unknowns=5,
        liq_block=slice(0, 4),
        if_block=slice(4, 9),
        gas_block=slice(9, 18),
        total_size=18,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 7),
        if_mpp_slice=slice(7, 8),
        if_liq_species_slice=slice(8, 9),
        gas_temperature_slice=slice(9, 18, 3),
        gas_species_local_slice=slice(1, 3),
    )


class _FakeISObj:
    def __init__(self, indices) -> None:
        self.indices = tuple(int(idx) for idx in indices)


class _FakeISFactory:
    def createGeneral(self, indices):
        return _FakeISObj(indices)


class _FakePETSc:
    class IS(_FakeISFactory):
        pass

    class PC:
        class SchurFactType:
            FULL = "FULL"
            DIAG = "DIAG"
            LOWER = "LOWER"
            UPPER = "UPPER"

        class SchurPreType:
            A11 = "A11"
            SELF = "SELF"
            SELFP = "SELFP"
            USER = "USER"


class _FakePC:
    def __init__(self) -> None:
        self.pc_type = ""
        self.fieldsplit_type = None
        self.schur_fact_type = None
        self.schur_precondition = None
        self.fieldsplit_is: list[tuple[str, object]] = []
        self.sub_ksps: list[_FakeKSP] = []
        self.asm_overlap = 0
        self.asm_subksps: list[_FakeKSP] = []

    def setType(self, pc_type: str) -> None:
        self.pc_type = str(pc_type)

    def getType(self) -> str:
        return self.pc_type

    def setFieldSplitType(self, fieldsplit_type: str) -> None:
        self.fieldsplit_type = str(fieldsplit_type)

    def getFieldSplitType(self):
        return self.fieldsplit_type

    def setFieldSplitIS(self, name: str, is_obj: object) -> None:
        self.fieldsplit_is.append((str(name), is_obj))

    def setFieldSplitSchurFactType(self, schur_fact_type) -> None:
        self.schur_fact_type = schur_fact_type

    def setFieldSplitSchurPre(self, schur_precondition) -> None:
        self.schur_precondition = schur_precondition

    def setUp(self) -> None:
        if self.pc_type == "fieldsplit" and not self.sub_ksps:
            self.sub_ksps = [_FakeKSP() for _ in self.fieldsplit_is]

    def getFieldSplitSubKSP(self):
        return tuple(self.sub_ksps)

    def setASMOverlap(self, overlap: int) -> None:
        self.asm_overlap = int(overlap)

    def getASMSubKSP(self):
        if not self.asm_subksps:
            self.asm_subksps = [_FakeKSP(), _FakeKSP()]
        return tuple(self.asm_subksps)


class _FakeKSP:
    def __init__(self) -> None:
        self.ksp_type = ""
        self.pc = _FakePC()
        self.operators = None
        self.tolerances = {}
        self.initial_guess_nonzero = None
        self.restart = None
        self.modified_gram_schmidt = None
        self.preallocate_vectors = None

    def setType(self, ksp_type: str) -> None:
        self.ksp_type = str(ksp_type)

    def getType(self) -> str:
        return self.ksp_type

    def getPC(self) -> _FakePC:
        return self.pc

    def setOperators(self, A, P) -> None:
        self.operators = (A, P)

    def setTolerances(self, *, rtol: float, atol: float, max_it: int) -> None:
        self.tolerances = {"rtol": float(rtol), "atol": float(atol), "max_it": int(max_it)}

    def setInitialGuessNonzero(self, enabled: bool) -> None:
        self.initial_guess_nonzero = bool(enabled)

    def setGMRESRestart(self, restart: int) -> None:
        self.restart = int(restart)

    def setGMRESModifiedGramSchmidt(self, enabled: bool) -> None:
        self.modified_gram_schmidt = bool(enabled)

    def setGMRESPreAllocateVectors(self, enabled: bool) -> None:
        self.preallocate_vectors = bool(enabled)


def _make_ctx(*, solver_inner_petsc: object) -> object:
    return build_nonlinear_context(
        cfg=SimpleNamespace(solver_inner_petsc=solver_inner_petsc),
        layout=_make_layout(),
        grid=SimpleNamespace(geometry_current=SimpleNamespace(name="geom")),
        t_old=0.0,
        dt=1.0,
        a_current=1.0,
        dot_a_frozen=0.0,
        state_guess=object(),
        accepted_state_old=object(),
        old_state_on_current_geometry=object(),
        old_mass_on_current_geometry=object(),
        props_current=object(),
        parallel_handles={"PETSc": _FakePETSc},
    )


def test_linear_cfg_defaults() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())

    cfg = build_linear_config_view(ctx)

    assert cfg.ksp_type == "fgmres"
    assert cfg.pc_type == "fieldsplit"
    assert cfg.fieldsplit_type == "schur"
    assert cfg.fieldsplit_scheme == "bulk_iface"


def test_linear_cfg_reads_normalized_inner_solver() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())
    ctx.cfg = SimpleNamespace(
        inner_solver=SimpleNamespace(
            ksp_type="gmres",
            pc_type="jacobi",
            ksp_rtol=1.0e-7,
            ksp_atol=1.0e-11,
            ksp_max_it=120,
        )
    )

    cfg = build_linear_config_view(ctx)

    assert cfg.ksp_type == "gmres"
    assert cfg.pc_type == "jacobi"
    assert cfg.max_it == 120


def test_linear_cfg_rejects_conflicting_solver_sources() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace(ksp_type="fgmres"))
    ctx.cfg = SimpleNamespace(
        inner_solver=SimpleNamespace(ksp_type="gmres"),
        solver_inner_petsc=SimpleNamespace(ksp_type="fgmres"),
    )

    with pytest.raises(ValueError, match="conflicting inner_solver and solver_inner_petsc"):
        build_linear_config_view(ctx)


def test_linear_cfg_invalid_pc_type() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace(pc_type="sor"))

    with pytest.raises(ValueError, match="unsupported PC type"):
        build_linear_config_view(ctx)


def test_linear_cfg_invalid_fieldsplit_scheme() -> None:
    ctx = _make_ctx(
        solver_inner_petsc=SimpleNamespace(
            fieldsplit=SimpleNamespace(scheme="three_way"),
        )
    )

    with pytest.raises(ValueError, match="unsupported fieldsplit scheme"):
        build_linear_config_view(ctx)


def test_linear_cfg_parses_string_booleans_explicitly() -> None:
    ctx = _make_ctx(
        solver_inner_petsc=SimpleNamespace(
            gmres_modified_gram_schmidt="false",
            gmres_preallocate="true",
            fieldsplit=SimpleNamespace(
                bulk=SimpleNamespace(asm_overlap=1),
            ),
        )
    )

    cfg = build_linear_config_view(ctx)

    assert cfg.gmres_modified_gram_schmidt is False
    assert cfg.gmres_preallocate is True


def test_fieldsplit_specs_from_layout_bulk_iface() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())

    specs = build_fieldsplit_specs(ctx, plan="bulk_iface")

    assert [spec.name for spec in specs] == ["bulk", "iface"]
    assert all(spec.index_count > 0 for spec in specs)


def test_fieldsplit_specs_forbid_rd_in_iface() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())

    class _BadLayout:
        def __init__(self, base) -> None:
            self._base = base

        def __getattr__(self, name: str):
            return getattr(self._base, name)

        def default_fieldsplit_plan(self, plan: str = "bulk_iface"):
            _ = plan
            return (
                {"name": "bulk", "blocks": ("liq", "gas"), "policy": "concat"},
                {"name": "iface", "blocks": ("if", "Rd"), "policy": "contiguous"},
            )

    ctx.layout = _BadLayout(ctx.layout)
    with pytest.raises(ValueError, match="Rd"):
        build_fieldsplit_specs(ctx, plan="bulk_iface")


def test_fieldsplit_specs_forbid_layout_rd_block_even_without_rd_label() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())

    class _BadLayout:
        def __init__(self, base) -> None:
            self._base = base

        def __getattr__(self, name: str):
            return getattr(self._base, name)

        def has_block(self, block_name: str) -> bool:
            return block_name == "Rd"

    ctx.layout = _BadLayout(ctx.layout)
    with pytest.raises(ValueError, match="Rd"):
        build_fieldsplit_specs(ctx, plan="bulk_iface")


def test_fieldsplit_specs_raise_clean_error_when_layout_is_map_missing_split() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())

    class _BadLayout:
        def __init__(self, base) -> None:
            self._base = base

        def __getattr__(self, name: str):
            return getattr(self._base, name)

        def build_is_petsc(self, *, PETSc, plan="bulk_iface", ownership_range=None):
            _ = (PETSc, plan, ownership_range)
            return {"bulk": _FakeISObj((0, 1, 2))}

    ctx.layout = _BadLayout(ctx.layout)
    with pytest.raises(ValueError, match="missing split 'iface'"):
        build_fieldsplit_specs(ctx, plan="bulk_iface")


def test_apply_structured_pc_fieldsplit_schur() -> None:
    solver_cfg = SimpleNamespace(
        ksp_type="fgmres",
        pc_type="fieldsplit",
        ksp_rtol=1.0e-8,
        ksp_atol=1.0e-12,
        ksp_max_it=200,
        restart=40,
        fieldsplit=SimpleNamespace(
            scheme="bulk_iface",
            type="schur",
            schur_fact_type="full",
            schur_precondition="a11",
        ),
    )
    ctx = _make_ctx(solver_inner_petsc=solver_cfg)
    ksp = _FakeKSP()

    diag = apply_structured_pc(ksp=ksp, ctx=ctx, A="A", P="P")

    assert ksp.getType() == "fgmres"
    assert ksp.getPC().getType() == "fieldsplit"
    assert ksp.getPC().getFieldSplitType() == "schur"
    assert diag.ksp_type == "fgmres"
    assert diag.pc_type == "fieldsplit"
    assert [item["name"] for item in diag.splits] == ["bulk", "iface"]


def test_apply_structured_pc_non_fieldsplit() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace(pc_type="jacobi", ksp_type="gmres"))
    ksp = _FakeKSP()

    diag = apply_structured_pc(ksp=ksp, ctx=ctx, A="A", P=None)

    assert ksp.getType() == "gmres"
    assert ksp.getPC().getType() == "jacobi"
    assert diag.pc_type == "jacobi"
    assert diag.splits == []


def test_apply_structured_pc_fieldsplit_requires_operator_matrix() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())
    ksp = _FakeKSP()

    with pytest.raises(ValueError, match="requires a non-null operator matrix A"):
        apply_structured_pc(ksp=ksp, ctx=ctx, A=None, P=None)


def test_apply_structured_pc_non_fieldsplit_requires_operator_matrix() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace(pc_type="jacobi"))
    ksp = _FakeKSP()

    with pytest.raises(ValueError, match="requires a non-null operator matrix A"):
        apply_structured_pc(ksp=ksp, ctx=ctx, A=None, P=None)


def test_subksp_defaults_bulk_iface() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())
    ksp = _FakeKSP()

    diag = apply_structured_pc(ksp=ksp, ctx=ctx, A="A", P=None)
    bulk_ksp, iface_ksp = ksp.getPC().getFieldSplitSubKSP()

    assert diag.subksp_configured is True
    assert bulk_ksp.getType() == "fgmres"
    assert bulk_ksp.getPC().getType() == "asm"
    assert bulk_ksp.getPC().asm_overlap == 1
    for asm_ksp in bulk_ksp.getPC().getASMSubKSP():
        assert asm_ksp.getType() == "preonly"
        assert asm_ksp.getPC().getType() == "ilu"
    assert iface_ksp.getType() == "preonly"
    assert iface_ksp.getPC().getType() == "lu"


def test_layout_build_is_petsc_used() -> None:
    ctx = _make_ctx(solver_inner_petsc=SimpleNamespace())
    ksp = _FakeKSP()

    diag = apply_structured_pc(ksp=ksp, ctx=ctx, A="A", P=None)

    assert diag.used_layout_build_is is True
