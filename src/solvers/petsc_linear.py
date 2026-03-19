from __future__ import annotations

"""PETSc linear-layer configuration helpers for the fixed-geometry inner solve.

This module only configures KSP / PC / fieldsplit / Schur / sub-KSP structure.
It does not create SNES objects, assemble residuals/Jacobians, or touch outer
predictor-corrector / accept-reject logic.
"""

from dataclasses import dataclass, field, replace
from typing import Any

from .nonlinear_context import NonlinearContext


def _get_petsc():
    """Import PETSc lazily so tests can run without petsc4py at import time."""

    try:
        from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc
    except ImportError:
        bootstrap_mpi_before_petsc = None

    if bootstrap_mpi_before_petsc is not None:
        bootstrap_mpi_before_petsc()
    from petsc4py import PETSc

    return PETSc


@dataclass(slots=True, kw_only=True)
class FieldSplitSpec:
    """One layout-derived fieldsplit definition consumed by PETSc PCFIELDSPLIT."""

    name: str
    blocks: tuple[str, ...]
    policy: str
    is_obj: object | None = None
    index_count: int = 0


@dataclass(slots=True, kw_only=True)
class LinearConfigView:
    """Normalized linear-solver configuration view read from solver_inner_petsc."""

    ksp_type: str = "fgmres"
    pc_type: str = "fieldsplit"
    fieldsplit_scheme: str = "bulk_iface"
    fieldsplit_type: str = "schur"
    schur_fact_type: str = "full"
    schur_precondition: str = "a11"
    rtol: float = 1.0e-8
    atol: float = 1.0e-12
    max_it: int = 200
    restart: int = 50
    gmres_modified_gram_schmidt: bool = True
    gmres_preallocate: bool = True
    bulk_ksp_type: str = "fgmres"
    bulk_pc_type: str = "asm"
    bulk_sub_ksp_type: str = "preonly"
    bulk_sub_pc_type: str = "ilu"
    bulk_asm_overlap: int = 1
    iface_ksp_type: str = "preonly"
    iface_pc_type: str = "lu"


@dataclass(slots=True, kw_only=True)
class LinearPCDiagnostics:
    """Structured diagnostics emitted after configuring the PETSc linear layer."""

    ksp_type: str = ""
    pc_type: str = ""
    fieldsplit_type: str = ""
    schur_fact_type: str = ""
    schur_precondition: str = ""
    restart: int | None = None
    ew_enabled: bool | None = None
    splits: list[dict[str, object]] = field(default_factory=list)
    used_layout_build_is: bool = False
    subksp_configured: bool = False
    from_options_applied: bool = False
    meta: dict[str, object] = field(default_factory=dict)


def _get_nested(source: object, *path: str, default: object = None) -> object:
    current: object = source
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def _to_float(name: str, value: object, default: float) -> float:
    if value is None:
        return float(default)
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _to_positive_int(name: str, value: object, default: int) -> int:
    if value is None:
        return int(default)
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be > 0")
    return number


def _to_nonnegative_int(name: str, value: object, default: int) -> int:
    if value is None:
        return int(default)
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be >= 0")
    return number


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _normalize_ksp_type(ksp_type: str) -> str:
    """Normalize and validate supported PETSc KSP types."""

    normalized = str(ksp_type).strip().lower()
    if normalized not in {"fgmres", "gmres", "preonly"}:
        raise ValueError(f"unsupported KSP type: {ksp_type}")
    return normalized


def _normalize_pc_type(pc_type: str) -> str:
    """Normalize and validate supported PETSc PC types."""

    normalized = str(pc_type).strip().lower()
    if normalized not in {"fieldsplit", "asm", "ilu", "lu", "jacobi", "none"}:
        raise ValueError(f"unsupported PC type: {pc_type}")
    return normalized


def _normalize_fieldsplit_type(fs_type: str) -> str:
    """Normalize and validate supported PETSc PCFIELDSPLIT coupling types."""

    normalized = str(fs_type).strip().lower()
    if normalized not in {"schur", "additive", "multiplicative", "symmetric_multiplicative"}:
        raise ValueError(f"unsupported fieldsplit type: {fs_type}")
    return normalized


def _normalize_fieldsplit_scheme(scheme: str) -> str:
    """Normalize and validate supported layout-driven fieldsplit schemes."""

    normalized = str(scheme).strip().lower()
    if normalized not in {"bulk_iface"}:
        raise ValueError(f"unsupported fieldsplit scheme: {scheme}")
    return normalized


def _normalize_schur_fact_type(fact_type: str) -> str:
    normalized = str(fact_type).strip().lower()
    if normalized not in {"diag", "lower", "upper", "full"}:
        raise ValueError(f"unsupported Schur fact type: {fact_type}")
    return normalized


def _normalize_schur_precondition(precondition: str) -> str:
    normalized = str(precondition).strip().lower()
    if normalized not in {"a11", "self", "selfp", "user"}:
        raise ValueError(f"unsupported Schur precondition: {precondition}")
    return normalized


def build_linear_config_view(ctx: NonlinearContext) -> LinearConfigView:
    """Read and normalize linear-layer configuration from the normalized solver config."""

    normalized_solver_cfg = _get_nested(ctx.cfg, "inner_solver", default=None)
    raw_solver_cfg = _get_nested(ctx.cfg, "solver_inner_petsc", default=None)
    if normalized_solver_cfg is None and raw_solver_cfg is None:
        raise ValueError("NonlinearContext.cfg must provide inner_solver or solver_inner_petsc")
    if normalized_solver_cfg is not None and raw_solver_cfg is not None and normalized_solver_cfg is not raw_solver_cfg:
        raise ValueError(
            "NonlinearContext.cfg must not provide conflicting inner_solver and solver_inner_petsc sources"
        )
    solver_cfg = normalized_solver_cfg if normalized_solver_cfg is not None else raw_solver_cfg

    fieldsplit_cfg = _get_nested(solver_cfg, "fieldsplit", default=None)
    bulk_cfg = _get_nested(fieldsplit_cfg, "bulk", default=None)
    iface_cfg = _get_nested(fieldsplit_cfg, "iface", default=None)

    return LinearConfigView(
        ksp_type=_normalize_ksp_type(str(_get_nested(solver_cfg, "ksp_type", default="fgmres"))),
        pc_type=_normalize_pc_type(str(_get_nested(solver_cfg, "pc_type", default="fieldsplit"))),
        fieldsplit_scheme=_normalize_fieldsplit_scheme(
            str(_get_nested(fieldsplit_cfg, "scheme", default="bulk_iface"))
        ),
        fieldsplit_type=_normalize_fieldsplit_type(str(_get_nested(fieldsplit_cfg, "type", default="schur"))),
        schur_fact_type=_normalize_schur_fact_type(
            str(_get_nested(fieldsplit_cfg, "schur_fact_type", default="full"))
        ),
        schur_precondition=_normalize_schur_precondition(
            str(_get_nested(fieldsplit_cfg, "schur_precondition", default="a11"))
        ),
        rtol=_to_float("ksp_rtol", _get_nested(solver_cfg, "ksp_rtol", default=None), 1.0e-8),
        atol=_to_float("ksp_atol", _get_nested(solver_cfg, "ksp_atol", default=None), 1.0e-12),
        max_it=_to_positive_int("ksp_max_it", _get_nested(solver_cfg, "ksp_max_it", default=None), 200),
        restart=_to_positive_int("restart", _get_nested(solver_cfg, "restart", default=None), 50),
        gmres_modified_gram_schmidt=_to_bool(
            _get_nested(solver_cfg, "gmres_modified_gram_schmidt", default=None), True
        ),
        gmres_preallocate=_to_bool(_get_nested(solver_cfg, "gmres_preallocate", default=None), True),
        bulk_ksp_type=_normalize_ksp_type(str(_get_nested(bulk_cfg, "ksp_type", default="fgmres"))),
        bulk_pc_type=_normalize_pc_type(str(_get_nested(bulk_cfg, "pc_type", default="asm"))),
        bulk_sub_ksp_type=_normalize_ksp_type(str(_get_nested(bulk_cfg, "sub_ksp_type", default="preonly"))),
        bulk_sub_pc_type=_normalize_pc_type(str(_get_nested(bulk_cfg, "sub_pc_type", default="ilu"))),
        bulk_asm_overlap=_to_nonnegative_int(
            "bulk_asm_overlap", _get_nested(bulk_cfg, "asm_overlap", default=None), 1
        ),
        iface_ksp_type=_normalize_ksp_type(str(_get_nested(iface_cfg, "ksp_type", default="preonly"))),
        iface_pc_type=_normalize_pc_type(str(_get_nested(iface_cfg, "pc_type", default="lu"))),
    )


def _resolve_petsc_provider(ctx: NonlinearContext) -> Any:
    provider = ctx.parallel_handles.get("PETSc") if isinstance(ctx.parallel_handles, dict) else None
    return provider if provider is not None else _get_petsc()


def build_fieldsplit_specs(
    ctx: NonlinearContext,
    *,
    plan: str | None = None,
    ownership_range: tuple[int, int] | None = None,
) -> list[FieldSplitSpec]:
    """Build fieldsplit specs from ctx.layout without re-deriving magic indices."""

    layout = ctx.layout
    plan_name = "bulk_iface" if plan is None else str(plan)
    if not hasattr(layout, "default_fieldsplit_plan"):
        raise ValueError("ctx.layout must provide default_fieldsplit_plan(plan)")
    if not hasattr(layout, "build_is_petsc"):
        raise ValueError("ctx.layout must provide build_is_petsc(...)")
    if not hasattr(layout, "describe_fieldsplits"):
        raise ValueError("ctx.layout must provide describe_fieldsplits(plan)")
    has_block = getattr(layout, "has_block", None)
    if callable(has_block):
        try:
            if bool(has_block("Rd")):
                raise ValueError("layout must not expose Rd inside the inner fieldsplit structure")
        except TypeError:
            if bool(has_block(block_name="Rd")):
                raise ValueError("layout must not expose Rd inside the inner fieldsplit structure")

    raw_plan = tuple(layout.default_fieldsplit_plan(plan_name))
    descriptions = {
        str(item["name"]): dict(item) for item in tuple(layout.describe_fieldsplits(plan_name))
    }
    petsc = _resolve_petsc_provider(ctx)
    is_map = layout.build_is_petsc(PETSc=petsc, plan=plan_name, ownership_range=ownership_range)

    specs: list[FieldSplitSpec] = []
    for item in raw_plan:
        name = str(item["name"])
        blocks = tuple(str(block) for block in tuple(item["blocks"]))
        if name == "iface" and any(block.strip().lower() == "rd" for block in blocks):
            raise ValueError("iface fieldsplit must not include Rd")
        if name not in is_map:
            raise ValueError(f"layout.build_is_petsc() missing split '{name}' required by fieldsplit plan")
        desc = descriptions.get(name, {})
        index_count = int(desc.get("index_count", 0))
        specs.append(
            FieldSplitSpec(
                name=name,
                blocks=blocks,
                policy=str(item["policy"]),
                is_obj=is_map[name],
                index_count=index_count,
            )
        )
    return specs


def configure_ksp_tolerances(ksp: object, cfg_lin: LinearConfigView) -> None:
    """Apply top-level KSP tolerances and GMRES/FGMRES options."""

    if hasattr(ksp, "setTolerances"):
        ksp.setTolerances(rtol=cfg_lin.rtol, atol=cfg_lin.atol, max_it=cfg_lin.max_it)
    if hasattr(ksp, "setInitialGuessNonzero"):
        ksp.setInitialGuessNonzero(False)
    if cfg_lin.ksp_type in {"fgmres", "gmres"}:
        if hasattr(ksp, "setGMRESRestart"):
            ksp.setGMRESRestart(cfg_lin.restart)
        if hasattr(ksp, "setGMRESModifiedGramSchmidt"):
            ksp.setGMRESModifiedGramSchmidt(cfg_lin.gmres_modified_gram_schmidt)
        if hasattr(ksp, "setGMRESPreAllocateVectors"):
            ksp.setGMRESPreAllocateVectors(cfg_lin.gmres_preallocate)


def _configure_named_ksp(ksp: object, *, ksp_type: str, pc_type: str, restart: int | None = None) -> None:
    if hasattr(ksp, "setType"):
        ksp.setType(ksp_type)
    if hasattr(ksp, "getPC"):
        pc = ksp.getPC()
        if hasattr(pc, "setType"):
            pc.setType(pc_type)
        if restart is not None and ksp_type in {"fgmres", "gmres"} and hasattr(ksp, "setGMRESRestart"):
            ksp.setGMRESRestart(restart)


def apply_fieldsplit_subksp_defaults(
    *,
    pc: object,
    cfg_lin: LinearConfigView,
    specs: list[FieldSplitSpec],
) -> dict[str, object]:
    """Apply default sub-KSP configuration for bulk/iface fieldsplits."""

    if not hasattr(pc, "getFieldSplitSubKSP"):
        raise ValueError("PC object must provide getFieldSplitSubKSP() after setUp()")
    sub_ksps = list(pc.getFieldSplitSubKSP())
    if len(sub_ksps) != len(specs):
        raise ValueError("number of PETSc sub-KSP objects does not match fieldsplit specs")

    configured_names: list[str] = []
    for spec, sub_ksp in zip(specs, sub_ksps):
        if spec.name == "bulk":
            _configure_named_ksp(
                sub_ksp,
                ksp_type=cfg_lin.bulk_ksp_type,
                pc_type=cfg_lin.bulk_pc_type,
                restart=cfg_lin.restart,
            )
            bulk_pc = sub_ksp.getPC()
            if cfg_lin.bulk_pc_type == "asm" and hasattr(bulk_pc, "setASMOverlap"):
                bulk_pc.setASMOverlap(cfg_lin.bulk_asm_overlap)
            if cfg_lin.bulk_pc_type == "asm" and hasattr(bulk_pc, "getASMSubKSP"):
                for asm_ksp in list(bulk_pc.getASMSubKSP()):
                    _configure_named_ksp(
                        asm_ksp,
                        ksp_type=cfg_lin.bulk_sub_ksp_type,
                        pc_type=cfg_lin.bulk_sub_pc_type,
                    )
            configured_names.append(spec.name)
            continue
        if spec.name == "iface":
            _configure_named_ksp(
                sub_ksp,
                ksp_type=cfg_lin.iface_ksp_type,
                pc_type=cfg_lin.iface_pc_type,
            )
            configured_names.append(spec.name)
            continue
        raise ValueError(f"unsupported fieldsplit name for sub-KSP defaults: {spec.name}")

    return {"configured_names": tuple(configured_names), "subksp_count": len(sub_ksps)}


def _resolve_schur_fact_type(petsc: Any, fact_type: str) -> Any:
    enum_container = getattr(getattr(petsc, "PC", None), "SchurFactType", None)
    if enum_container is None:
        return fact_type
    return getattr(enum_container, fact_type.upper(), fact_type)


def _resolve_schur_precondition(petsc: Any, precondition: str) -> Any:
    enum_container = getattr(getattr(petsc, "PC", None), "SchurPreType", None)
    if enum_container is None:
        return precondition
    return getattr(enum_container, precondition.upper(), precondition)


def apply_structured_pc(
    *,
    ksp: object,
    ctx: NonlinearContext,
    A: object | None,
    P: object | None,
    pc_type_override: str | None = None,
) -> LinearPCDiagnostics:
    """Configure top-level PETSc KSP/PC state for the inner linear solve."""

    cfg_lin = build_linear_config_view(ctx)
    if pc_type_override is not None:
        cfg_lin = replace(cfg_lin, pc_type=_normalize_pc_type(pc_type_override))
    if A is None:
        raise ValueError("apply_structured_pc() requires a non-null operator matrix A")

    if hasattr(ksp, "setType"):
        ksp.setType(cfg_lin.ksp_type)
    if hasattr(ksp, "setOperators"):
        ksp.setOperators(A, A if P is None else P)
    configure_ksp_tolerances(ksp, cfg_lin)

    pc = ksp.getPC()
    if hasattr(pc, "setType"):
        pc.setType(cfg_lin.pc_type)

    diag = LinearPCDiagnostics(
        ksp_type=cfg_lin.ksp_type,
        pc_type=cfg_lin.pc_type,
        fieldsplit_type=cfg_lin.fieldsplit_type if cfg_lin.pc_type == "fieldsplit" else "",
        schur_fact_type=cfg_lin.schur_fact_type if cfg_lin.pc_type == "fieldsplit" else "",
        schur_precondition=cfg_lin.schur_precondition if cfg_lin.pc_type == "fieldsplit" else "",
        restart=cfg_lin.restart,
        ew_enabled=False,
        meta={"fieldsplit_scheme": cfg_lin.fieldsplit_scheme},
    )

    if cfg_lin.pc_type != "fieldsplit":
        return diag

    petsc = _resolve_petsc_provider(ctx)
    ownership_range = (
        ctx.parallel_handles.get("ownership_range")
        if isinstance(ctx.parallel_handles, dict)
        else None
    )
    specs = build_fieldsplit_specs(ctx, plan=cfg_lin.fieldsplit_scheme, ownership_range=ownership_range)
    diag.used_layout_build_is = True
    diag.splits = [
        {
            "name": spec.name,
            "blocks": spec.blocks,
            "policy": spec.policy,
            "index_count": spec.index_count,
        }
        for spec in specs
    ]

    if hasattr(pc, "setFieldSplitType"):
        pc.setFieldSplitType(cfg_lin.fieldsplit_type)
    for spec in specs:
        if hasattr(pc, "setFieldSplitIS"):
            pc.setFieldSplitIS(spec.name, spec.is_obj)
    if cfg_lin.fieldsplit_type == "schur":
        if hasattr(pc, "setFieldSplitSchurFactType"):
            pc.setFieldSplitSchurFactType(_resolve_schur_fact_type(petsc, cfg_lin.schur_fact_type))
        if hasattr(pc, "setFieldSplitSchurPre"):
            pc.setFieldSplitSchurPre(_resolve_schur_precondition(petsc, cfg_lin.schur_precondition))
    if hasattr(pc, "setUp"):
        pc.setUp()
    diag.meta["subksp"] = apply_fieldsplit_subksp_defaults(pc=pc, cfg_lin=cfg_lin, specs=specs)
    diag.subksp_configured = True
    return diag


def finalize_ksp_config(
    ksp: object,
    diag_pc: LinearPCDiagnostics,
    *,
    from_options: bool,
) -> None:
    """Capture final KSP/PC types after optional PETSc options processing."""

    diag_pc.from_options_applied = bool(from_options)
    if hasattr(ksp, "getType"):
        diag_pc.ksp_type = str(ksp.getType())
    if hasattr(ksp, "getPC"):
        pc = ksp.getPC()
        if hasattr(pc, "getType"):
            diag_pc.pc_type = str(pc.getType())
        if hasattr(pc, "getFieldSplitType"):
            fs_type = pc.getFieldSplitType()
            diag_pc.fieldsplit_type = "" if fs_type is None else str(fs_type)


__all__ = [
    "FieldSplitSpec",
    "LinearConfigView",
    "LinearPCDiagnostics",
    "_get_petsc",
    "_normalize_fieldsplit_type",
    "_normalize_ksp_type",
    "_normalize_pc_type",
    "apply_fieldsplit_subksp_defaults",
    "apply_structured_pc",
    "build_fieldsplit_specs",
    "build_linear_config_view",
    "configure_ksp_tolerances",
    "finalize_ksp_config",
]
