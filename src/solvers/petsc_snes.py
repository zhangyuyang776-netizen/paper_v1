from __future__ import annotations

"""PETSc SNES execution for the fixed-geometry inner solve.

This module owns the inner-only PETSc SNES orchestration layer. It does not
perform outer predictor/corrector work, remap, state recovery, or geometry
updates. The residual/Jacobian physics truth must arrive through
``NonlinearContext`` and injected builder hooks.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np

from assembly.jacobian_global import (
    GlobalJacobianFDOptions,
    JacobianOwnership,
    assemble_and_insert_global_jacobian,
)
from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from assembly.petsc_prealloc import build_petsc_prealloc
from assembly.residual_global import GlobalResidualResult, assemble_global_residual_from_trial_view
from core.state_pack import pack_state_to_array
from physics.radius_update import build_radius_update_package
from .linesearch_guards import (
    GuardCheckResult,
    build_linesearch_guard_config,
    make_petsc_linesearch_postcheck,
    make_petsc_linesearch_precheck,
)
from .nonlinear_context import NonlinearContext
from .nonlinear_types import (
    FailureClass,
    FailureInfo,
    InnerSolveResult,
    InnerSolveStats,
    ResidualNorms,
    SolverBackend,
)
from .petsc_linear import LinearPCDiagnostics, apply_structured_pc, finalize_ksp_config


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
class SnesConfigView:
    """Normalized SNES-level configuration extracted from the inner-solver config."""

    snes_type: str = "newtonls"
    linesearch_type: str = "bt"
    snes_rtol: float = 1.0e-8
    snes_atol: float = 1.0e-12
    snes_stol: float = 0.0
    snes_max_it: int = 50
    monitor: bool = False
    options_prefix: str = ""
    use_ew: bool = False
    lag_jacobian: int = -1
    lag_preconditioner: int = -1


@dataclass(slots=True, kw_only=True)
class ResidualEvalCache:
    """Mutable cache of the most recent residual-side diagnostics and snapshot."""

    last_snapshot: object | None = None
    last_diag: dict[str, object] = field(default_factory=dict)
    history_linf: list[float] = field(default_factory=list)
    n_res_eval: int = 0
    n_domain_error: int = 0
    fatal_exception: BaseException | None = None


@dataclass(slots=True, kw_only=True)
class JacobianEvalCache:
    """Mutable cache of Jacobian-side diagnostics."""

    n_jac_eval: int = 0
    last_diag: dict[str, object] = field(default_factory=dict)
    fatal_exception: BaseException | None = None


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


def _to_nonnegative_float(name: str, value: object, default: float) -> float:
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


def _to_int(name: str, value: object, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _ctx_meta_get(ctx: NonlinearContext, key: str, default=None):
    meta = getattr(ctx, "meta", None)
    if not isinstance(meta, dict):
        return default
    return meta.get(key, default)


def _emit_solver_diag(
    ctx: NonlinearContext,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    level: str = "INFO",
) -> None:
    sink = _ctx_meta_get(ctx, "diag_sink", None)
    if sink is None:
        return
    try:
        sink(event=str(event), payload={"level": str(level), **(payload or {})})
    except Exception:
        return


def _solver_base_payload(ctx: NonlinearContext) -> dict[str, Any]:
    return {
        "step_id": _ctx_meta_get(ctx, "step_id", None),
        "outer_iter_index": _ctx_meta_get(ctx, "outer_iter_index", None),
    }


def _safe_int(value: object, default=None):
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: object, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _normalize_snes_type(snes_type: str) -> str:
    """Normalize and validate supported SNES types."""

    normalized = str(snes_type).strip().lower()
    if normalized not in {"newtonls"}:
        raise ValueError(f"unsupported SNES type: {snes_type}")
    return normalized


def _normalize_linesearch_type(linesearch_type: str) -> str:
    """Normalize and validate supported PETSc SNES line-search types."""

    normalized = str(linesearch_type).strip().lower()
    if normalized not in {"bt"}:
        raise ValueError(f"unsupported SNES line-search type: {linesearch_type}")
    return normalized


def build_snes_config_view(ctx: NonlinearContext) -> SnesConfigView:
    """Build a normalized SNES config view from the normalized inner solver config."""

    normalized_solver_cfg = _get_nested(ctx.cfg, "inner_solver", default=None)
    raw_solver_cfg = _get_nested(ctx.cfg, "solver_inner_petsc", default=None)
    if normalized_solver_cfg is None and raw_solver_cfg is None:
        raise ValueError("NonlinearContext.cfg must provide inner_solver or solver_inner_petsc")
    if normalized_solver_cfg is not None and raw_solver_cfg is not None and normalized_solver_cfg is not raw_solver_cfg:
        raise ValueError(
            "NonlinearContext.cfg must not provide conflicting inner_solver and solver_inner_petsc sources"
        )
    solver_cfg = normalized_solver_cfg if normalized_solver_cfg is not None else raw_solver_cfg

    cfg = SnesConfigView(
        snes_type=_normalize_snes_type(str(_get_nested(solver_cfg, "snes_type", default="newtonls"))),
        linesearch_type=_normalize_linesearch_type(
            str(_get_nested(solver_cfg, "linesearch_type", default="bt"))
        ),
        snes_rtol=_to_nonnegative_float("snes_rtol", _get_nested(solver_cfg, "snes_rtol", default=None), 1.0e-8),
        snes_atol=_to_nonnegative_float("snes_atol", _get_nested(solver_cfg, "snes_atol", default=None), 1.0e-12),
        snes_stol=_to_nonnegative_float("snes_stol", _get_nested(solver_cfg, "snes_stol", default=None), 0.0),
        snes_max_it=_to_positive_int(
            "snes_max_it",
            _get_nested(solver_cfg, "snes_max_it", default=_get_nested(solver_cfg, "inner_max_iter", default=None)),
            50,
        ),
        monitor=_to_bool(_get_nested(solver_cfg, "monitor", default=None), False),
        options_prefix=str(_get_nested(solver_cfg, "options_prefix", default="")),
        use_ew=_to_bool(_get_nested(solver_cfg, "use_ew", default=None), False),
        lag_jacobian=_to_int("lag_jacobian", _get_nested(solver_cfg, "lag_jacobian", default=None), -1),
        lag_preconditioner=_to_int(
            "lag_preconditioner", _get_nested(solver_cfg, "lag_preconditioner", default=None), -1
        ),
    )
    if cfg.monitor:
        raise ValueError("petsc_snes.py does not yet support monitor=True")
    if cfg.use_ew:
        raise ValueError("petsc_snes.py does not yet support use_ew=True")
    return cfg


def _comm_size(ctx: NonlinearContext) -> int:
    comm = ctx.parallel_handles.get("comm") if isinstance(ctx.parallel_handles, dict) else None
    if comm is None:
        return 1
    size_attr = getattr(comm, "size", None)
    if size_attr is not None:
        return int(size_attr)
    get_size = getattr(comm, "Get_size", None)
    if callable(get_size):
        return int(get_size())
    return 1


def _require_parallel_metadata_for_multi_rank(ctx: NonlinearContext) -> None:
    if _comm_size(ctx) <= 1:
        return
    missing: list[str] = []
    if "ownership_range" not in ctx.parallel_handles:
        missing.append("parallel_handles.ownership_range")
    if "ownership_ranges" not in ctx.parallel_handles:
        missing.append("parallel_handles.ownership_ranges")
    if "layout_to_petsc" not in ctx.parallel_handles:
        missing.append("parallel_handles.layout_to_petsc")
    if "jacobian_ownership" not in ctx.meta:
        missing.append("meta.jacobian_ownership")
    if missing:
        raise ValueError(
            "multi-rank PETSc SNES solve requires explicit parallel metadata: " + ", ".join(missing)
        )


_CONTROLLED_DOMAIN_EXCEPTION_NAMES = {
    "GlobalResidualAssemblyError",
    "LiquidResidualAssemblyError",
    "InterfaceResidualAssemblyError",
    "GasResidualAssemblyError",
    "StatePackError",
    "StateRecoveryError",
    "EnergyFluxError",
    "ConvectiveFluxError",
    "GasFluxError",
    "LiquidFluxError",
    "InterfaceEnergyError",
    "InterfaceFaceError",
    "InterfaceMassError",
    "VelocityRecoveryError",
    "GlobalJacobianAssemblyError",
    "LiquidJacobianAssemblyError",
    "InterfaceJacobianAssemblyError",
    "GasJacobianAssemblyError",
}


def _is_controlled_domain_exception(exc: BaseException) -> bool:
    return exc.__class__.__name__ in _CONTROLLED_DOMAIN_EXCEPTION_NAMES


def _record_fatal_callback_exception(
    *,
    ctx: NonlinearContext,
    where: str,
    exc: BaseException,
) -> None:
    ctx.meta["fatal_callback_exception"] = {
        "where": str(where),
        "type": type(exc).__name__,
        "message": str(exc),
    }


def _array_view_from_vec(vec: object, writable: bool) -> np.ndarray:
    if hasattr(vec, "getArray"):
        return np.asarray(vec.getArray(readonly=not writable))
    if hasattr(vec, "array"):
        arr = np.asarray(getattr(vec, "array"))
        if writable and not arr.flags.writeable:
            raise ValueError("vector array view must be writable")
        return arr
    arr = np.asarray(vec)
    if writable and not arr.flags.writeable:
        raise ValueError("vector array view must be writable")
    return arr


def _create_vec_from_array(PETSc, array: np.ndarray, *, comm: object | None = None) -> object:
    values = np.asarray(array, dtype=np.float64)
    vec = PETSc.Vec()
    if hasattr(vec, "createWithArray"):
        return vec.createWithArray(values.copy(), comm=comm)
    if hasattr(vec, "createSeq"):
        created = vec.createSeq(int(values.size), comm=comm)
        if hasattr(created, "setArray"):
            created.setArray(values.copy())
        else:
            _array_view_from_vec(created, writable=True)[:] = values
        return created
    raise ValueError("PETSc Vec provider must support createWithArray or createSeq")


def build_initial_state_vec(ctx: NonlinearContext, PETSc) -> object:
    """Build the SNES state vector from ctx.u_guess_vec or ctx.state_guess."""

    if _comm_size(ctx) > 1:
        hook = ctx.meta.get("distributed_state_vec_builder")
        if callable(hook):
            return hook(ctx=ctx, PETSc=PETSc)
        if ctx.u_guess_vec is None:
            raise ValueError(
                "multi-rank PETSc SNES solve requires ctx.u_guess_vec or meta.distributed_state_vec_builder"
            )

    if ctx.u_guess_vec is not None:
        return ctx.u_guess_vec.copy() if hasattr(ctx.u_guess_vec, "copy") else ctx.u_guess_vec

    if "u_trial_layout_guess" in ctx.meta:
        arr = np.asarray(ctx.meta["u_trial_layout_guess"], dtype=np.float64)
        return _create_vec_from_array(PETSc, arr, comm=ctx.parallel_handles.get("comm"))

    species_maps = ctx.meta.get("species_maps")
    if species_maps is None:
        raise ValueError("NonlinearContext.meta must provide species_maps when u_guess_vec is absent")
    arr = pack_state_to_array(ctx.state_guess, ctx.layout, species_maps)
    return _create_vec_from_array(PETSc, arr, comm=ctx.parallel_handles.get("comm"))


def build_residual_vec(ctx: NonlinearContext, PETSc) -> object:
    """Build a residual Vec distributed compatibly with the trial state Vec."""

    if _comm_size(ctx) > 1:
        hook = ctx.meta.get("distributed_residual_vec_builder")
        if callable(hook):
            return hook(ctx=ctx, PETSc=PETSc)
        if ctx.global_vec_template is None and ctx.u_guess_vec is None:
            raise ValueError(
                "multi-rank PETSc SNES solve requires ctx.global_vec_template, ctx.u_guess_vec, or meta.distributed_residual_vec_builder"
            )

    if ctx.global_vec_template is not None and hasattr(ctx.global_vec_template, "duplicate"):
        return ctx.global_vec_template.duplicate()
    if ctx.u_guess_vec is not None and hasattr(ctx.u_guess_vec, "duplicate"):
        return ctx.u_guess_vec.duplicate()
    return _create_vec_from_array(PETSc, np.zeros(int(ctx.layout.total_size), dtype=np.float64), comm=ctx.parallel_handles.get("comm"))


def _default_jacobian_ownership(ctx: NonlinearContext) -> JacobianOwnership:
    return JacobianOwnership(
        owned_liq_cells=np.arange(int(ctx.layout.n_liq_cells), dtype=np.int64),
        owned_gas_cells=np.arange(int(ctx.layout.n_gas_cells), dtype=np.int64),
        interface_owner_active=True,
    )


def build_preallocated_jacobian_mats(ctx: NonlinearContext, PETSc) -> tuple[object, object]:
    """Create preallocated Jacobian/preconditioner matrices from the frozen pattern."""

    _require_parallel_metadata_for_multi_rank(ctx)
    pattern = ctx.meta.get("jacobian_pattern")
    if pattern is None:
        mesh = ctx.meta.get("mesh")
        species_maps = ctx.meta.get("species_maps")
        if mesh is None or species_maps is None:
            raise ValueError("ctx.meta must provide mesh/species_maps or a prebuilt jacobian_pattern")
        pattern = build_jacobian_pattern(mesh=mesh, layout=ctx.layout, species_maps=species_maps)
    if not isinstance(pattern, JacobianPattern):
        raise ValueError("ctx.meta['jacobian_pattern'] must be a JacobianPattern")

    ownership_range = tuple(ctx.parallel_handles.get("ownership_range", (0, int(ctx.layout.total_size))))
    ownership_ranges = ctx.parallel_handles.get(
        "ownership_ranges",
        np.asarray([[0, int(ctx.layout.total_size)]], dtype=np.int64),
    )
    layout_to_petsc = ctx.parallel_handles.get("layout_to_petsc")
    comm = ctx.parallel_handles.get("comm")
    J_result = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=ownership_range,
        ownership_ranges=ownership_ranges,
        PETSc=PETSc,
        comm=comm,
        mat=None,
        layout_to_petsc=layout_to_petsc,
    )
    P_result = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=ownership_range,
        ownership_ranges=ownership_ranges,
        PETSc=PETSc,
        comm=comm,
        mat=None,
        layout_to_petsc=layout_to_petsc,
    )
    ctx.meta["jacobian_pattern"] = pattern
    ctx.meta["jacobian_prealloc_diagnostics"] = {"J": J_result.diagnostics, "P": P_result.diagnostics}
    return J_result.mat, P_result.mat


def _dense_residual_from_sparse(result: GlobalResidualResult, *, n_global: int) -> np.ndarray:
    values = np.zeros(n_global, dtype=np.float64)
    values[np.asarray(result.rows_global, dtype=np.int64)] = np.asarray(result.values, dtype=np.float64)
    return values


def _resolve_residual_builder(ctx: NonlinearContext) -> Callable[[np.ndarray], object]:
    builder = ctx.meta.get("assemble_global_residual_from_layout_vector")
    if callable(builder):
        return builder
    builder = ctx.meta.get("residual_builder")
    if callable(builder):
        return builder
    if _comm_size(ctx) > 1:
        raise ValueError(
            "multi-rank PETSc SNES solve requires an explicit distributed-safe residual builder hook"
        )

    required = (
        "mesh",
        "species_maps",
        "residual_ownership",
        "run_cfg",
        "liquid_thermo",
        "gas_thermo",
        "control_surface_metrics",
        "farfield_bc",
    )
    if all(name in ctx.meta for name in required):
        def _builder(u_layout: np.ndarray) -> GlobalResidualResult:
            return assemble_global_residual_from_trial_view(
                vec_trial=u_layout,
                base_state=ctx.state_guess,
                old_state_current_geom=ctx.old_state_on_current_geometry,
                mesh=ctx.meta["mesh"],
                layout=ctx.layout,
                species_maps=ctx.meta["species_maps"],
                ownership=ctx.meta["residual_ownership"],
                run_cfg=ctx.meta["run_cfg"],
                liquid_thermo=ctx.meta["liquid_thermo"],
                gas_thermo=ctx.meta["gas_thermo"],
                equilibrium_model=ctx.models.equilibrium_model,
                control_surface_metrics=ctx.meta["control_surface_metrics"],
                farfield_bc=ctx.meta["farfield_bc"],
            )

        return _builder
    raise ValueError("NonlinearContext.meta must provide a residual builder hook for PETSc SNES callbacks")


def _resolve_block_residual_builder(ctx: NonlinearContext) -> Callable[[np.ndarray], object]:
    builder = ctx.meta.get("build_all_residual_blocks_from_layout_vector")
    if callable(builder):
        return builder
    raise ValueError("NonlinearContext.meta must provide build_all_residual_blocks_from_layout_vector")


def _resolve_snapshot_from_builder_output(output: object) -> tuple[GlobalResidualResult, object | None]:
    if isinstance(output, tuple) and len(output) == 2:
        result, snapshot = output
        return result, snapshot
    return output, None


def _mark_function_domain_error(
    *,
    snes: object,
    residual_cache: ResidualEvalCache,
    result: GuardCheckResult | FailureInfo | None = None,
) -> None:
    residual_cache.n_domain_error += 1
    if result is not None:
        residual_cache.last_diag["domain_error"] = getattr(result, "failure_reason", None)
    if hasattr(snes, "setFunctionDomainError"):
        snes.setFunctionDomainError()


def _mark_jacobian_domain_error(*, snes: object, jac_cache: JacobianEvalCache, exc: Exception) -> None:
    jac_cache.last_diag["jacobian_error"] = str(exc)
    if hasattr(snes, "setJacobianDomainError"):
        snes.setJacobianDomainError()


def make_residual_callback(
    *,
    ctx: NonlinearContext,
    residual_cache: ResidualEvalCache,
    guard_cfg: object,
    array_view_factory: Callable[[object, bool], np.ndarray],
) -> Callable[[object, object, object], int]:
    """Create the PETSc SNES residual callback bound to one fixed-geometry context."""

    _ = guard_cfg
    residual_builder = _resolve_residual_builder(ctx)
    n_global = int(ctx.layout.total_size)

    def _callback(snes: object, X: object, F: object) -> int:
        trial = np.array(array_view_factory(X, False), dtype=np.float64, copy=True)
        residual_cache.n_res_eval += 1
        try:
            built = residual_builder(trial)
            result, snapshot = _resolve_snapshot_from_builder_output(built)
            if not isinstance(result, GlobalResidualResult):
                raise TypeError("residual builder must return GlobalResidualResult or (GlobalResidualResult, snapshot)")
            dense = _dense_residual_from_sparse(result, n_global=n_global)
            F_view = array_view_factory(F, True)
            F_view[:] = dense
            residual_cache.last_snapshot = snapshot
            residual_cache.last_diag = dict(result.diagnostics)
            residual_cache.last_diag["row_count_total"] = int(result.rows_global.size)
            residual_cache.history_linf.append(float(np.max(np.abs(dense))) if dense.size else 0.0)
            ctx.meta["last_trial_snapshot"] = snapshot
            return 0
        except Exception as exc:
            F_view = array_view_factory(F, True)
            F_view[:] = 0.0
            residual_cache.last_snapshot = None
            if _is_controlled_domain_exception(exc):
                residual_cache.last_diag = {
                    "exception": str(exc),
                    "domain_error": "residual_callback_exception",
                    "exception_type": type(exc).__name__,
                }
                _mark_function_domain_error(snes=snes, residual_cache=residual_cache)
            else:
                residual_cache.fatal_exception = exc
                residual_cache.last_diag = {
                    "fatal_callback_exception": str(exc),
                    "exception_type": type(exc).__name__,
                }
                _record_fatal_callback_exception(ctx=ctx, where="residual_callback", exc=exc)
                _mark_function_domain_error(snes=snes, residual_cache=residual_cache)
            return 0

    return _callback


def make_jacobian_callback(
    *,
    ctx: NonlinearContext,
    jac_cache: JacobianEvalCache,
) -> Callable[[object, object, object, object], int]:
    """Create the PETSc SNES Jacobian callback bound to one fixed-geometry context."""

    _require_parallel_metadata_for_multi_rank(ctx)
    insertion_builder = ctx.meta.get("assemble_and_insert_global_jacobian")
    if not callable(insertion_builder):
        if _comm_size(ctx) > 1:
            raise ValueError(
                "multi-rank PETSc SNES solve requires an explicit distributed-safe Jacobian insertion hook"
            )
        block_builder = _resolve_block_residual_builder(ctx)
        mesh = ctx.meta.get("mesh")
        species_maps = ctx.meta.get("species_maps")
        if mesh is None or species_maps is None:
            raise ValueError("ctx.meta must provide mesh/species_maps for global Jacobian insertion")
        ownership = ctx.meta.get("jacobian_ownership", _default_jacobian_ownership(ctx))
        pattern = ctx.meta.get("jacobian_pattern")
        if pattern is None:
            pattern = build_jacobian_pattern(mesh=mesh, layout=ctx.layout, species_maps=species_maps)
            ctx.meta["jacobian_pattern"] = pattern

        def insertion_builder(*, u_trial_layout: np.ndarray, mat: object, PETSc: object) -> object:
            return assemble_and_insert_global_jacobian(
                u_trial_layout=u_trial_layout,
                layout=ctx.layout,
                mesh=mesh,
                species_maps=species_maps,
                ownership=ownership,
                pattern=pattern,
                build_all_residual_blocks_from_layout_vector=block_builder,
                mat=mat,
                PETSc=PETSc,
                layout_to_petsc=ctx.parallel_handles.get("layout_to_petsc"),
                fd_options=ctx.meta.get("global_jacobian_fd_options", GlobalJacobianFDOptions()),
                zero_before_insert=True,
                assembly_flush=True,
            )

    PETSc_provider = ctx.parallel_handles.get("PETSc")
    if PETSc_provider is None:
        PETSc_provider = _get_petsc()

    def _callback(snes: object, X: object, J: object, P: object) -> int:
        trial = np.array(_array_view_from_vec(X, writable=False), dtype=np.float64, copy=True)
        jac_cache.n_jac_eval += 1
        try:
            if hasattr(J, "zeroEntries"):
                J.zeroEntries()
            if P is not J and hasattr(P, "zeroEntries"):
                P.zeroEntries()
            J_result = insertion_builder(u_trial_layout=trial, mat=J, PETSc=PETSc_provider)
            if P is not J:
                insertion_builder(u_trial_layout=trial, mat=P, PETSc=PETSc_provider)
            jac_cache.last_diag = getattr(J_result, "diagnostics", {})
            return 0
        except Exception as exc:
            if _is_controlled_domain_exception(exc):
                jac_cache.last_diag = {
                    "jacobian_error": str(exc),
                    "domain_error": "jacobian_callback_exception",
                    "exception_type": type(exc).__name__,
                }
                _mark_jacobian_domain_error(snes=snes, jac_cache=jac_cache, exc=exc)
            else:
                jac_cache.fatal_exception = exc
                jac_cache.last_diag = {
                    "fatal_callback_exception": str(exc),
                    "exception_type": type(exc).__name__,
                }
                _record_fatal_callback_exception(ctx=ctx, where="jacobian_callback", exc=exc)
                _mark_jacobian_domain_error(snes=snes, jac_cache=jac_cache, exc=exc)
            return 0

    return _callback


def attach_linesearch_guards(
    *,
    snes: object,
    ctx: NonlinearContext,
    residual_cache: ResidualEvalCache,
    guard_cfg: object,
    array_view_factory: Callable[[object, bool], np.ndarray],
) -> None:
    """Attach pre/post line-search guards to the PETSc line search object."""

    if not hasattr(snes, "getLineSearch"):
        return
    linesearch = snes.getLineSearch()
    if linesearch is None:
        return

    def _on_guard_result(result: GuardCheckResult) -> None:
        ctx.meta["last_guard_result"] = result
        if result.guard_active:
            ctx.diagnostics["guard_active"] = True
            ctx.meta["any_guard_triggered"] = True
        if not result.ok:
            ctx.meta["last_failed_guard_result"] = result

    def _on_domain_error(result: GuardCheckResult) -> None:
        _mark_function_domain_error(snes=snes, residual_cache=residual_cache, result=result)

    pre = make_petsc_linesearch_precheck(
        ctx=ctx,
        cfg_guard=guard_cfg,
        array_view_factory=array_view_factory,
        on_guard_result=_on_guard_result,
        on_domain_error=_on_domain_error,
    )
    post = make_petsc_linesearch_postcheck(
        ctx=ctx,
        cfg_guard=guard_cfg,
        snapshot_getter=lambda: residual_cache.last_snapshot,
        on_guard_result=_on_guard_result,
        on_domain_error=_on_domain_error,
    )
    if hasattr(linesearch, "setPreCheck"):
        linesearch.setPreCheck(pre)
    if hasattr(linesearch, "setPostCheck"):
        linesearch.setPostCheck(post)


def collect_inner_stats(
    *,
    snes: object,
    residual_cache: ResidualEvalCache,
    jac_cache: JacobianEvalCache,
    wall_time_s: float,
    linear_diag: LinearPCDiagnostics | None,
) -> InnerSolveStats:
    """Collect structured PETSc SNES/KSP statistics into InnerSolveStats."""

    ksp = snes.getKSP() if hasattr(snes, "getKSP") else None
    residual_norms = ResidualNorms(
        linf=residual_cache.history_linf[-1] if residual_cache.history_linf else None,
    )
    if hasattr(snes, "getFunctionNorm"):
        residual_norms.l2 = float(snes.getFunctionNorm())
    stats = InnerSolveStats(
        backend=SolverBackend.PETSC_SNES,
        converged=bool(hasattr(snes, "getConvergedReason") and int(snes.getConvergedReason()) > 0),
        snes_reason=int(snes.getConvergedReason()) if hasattr(snes, "getConvergedReason") else None,
        ksp_reason=int(ksp.getConvergedReason()) if ksp is not None and hasattr(ksp, "getConvergedReason") else None,
        inner_iter_count=int(snes.getIterationNumber()) if hasattr(snes, "getIterationNumber") else 0,
        linear_iter_count=int(ksp.getIterationNumber()) if ksp is not None and hasattr(ksp, "getIterationNumber") else 0,
        residual_norms=residual_norms,
        history_linf=list(residual_cache.history_linf),
        line_search_used=True,
        damping_used=False,
        guard_triggered=False,
        wall_time_s=float(wall_time_s),
        message="",
        meta={
            "n_res_eval": int(residual_cache.n_res_eval),
            "n_jac_eval": int(jac_cache.n_jac_eval),
            "n_domain_error": int(residual_cache.n_domain_error),
            "linear_pc": linear_diag.meta if linear_diag is not None else {},
        },
    )
    return stats


def map_snes_failure(
    *,
    ctx: NonlinearContext,
    snes: object,
    residual_cache: ResidualEvalCache,
    jac_cache: JacobianEvalCache,
    exc: Exception | None = None,
) -> FailureInfo:
    """Map SNES/guard/domain failures to the project-level inner failure taxonomy."""

    fatal_callback = ctx.meta.get("fatal_callback_exception")
    if isinstance(fatal_callback, dict):
        exc_type = str(fatal_callback.get("type", "Exception"))
        return FailureInfo(
            failure_class=FailureClass.INNER_FAIL,
            reason_code=f"exception:{exc_type}",
            message=str(fatal_callback.get("message", "")),
            where=str(fatal_callback.get("where", "petsc_snes.solve")),
            recoverable=False,
            rollback_required=False,
            meta={"fatal_callback_exception": dict(fatal_callback)},
        )

    last_guard = ctx.meta.get("last_failed_guard_result")
    if isinstance(last_guard, GuardCheckResult) and not last_guard.ok:
        return FailureInfo(
            failure_class=FailureClass.INNER_FAIL,
            reason_code=f"guard:{last_guard.failure_reason.value}",
            message=last_guard.failure.message or "guard-triggered inner failure",
            where=last_guard.failure.where or "petsc_snes.solve",
            recoverable=bool(last_guard.failure.recoverable),
            rollback_required=False,
            meta={
                "guard_reason": last_guard.failure_reason.value,
                "guard_failure": last_guard.failure.meta,
                "n_domain_error": int(residual_cache.n_domain_error),
            },
        )

    if residual_cache.n_domain_error > 0:
        return FailureInfo(
            failure_class=FailureClass.INNER_FAIL,
            reason_code="snes_diverged_function_domain",
            message="inner SNES solve hit a function-domain error",
            where="petsc_snes.solve",
            recoverable=True,
            rollback_required=False,
            meta={
                "n_domain_error": int(residual_cache.n_domain_error),
                "residual_diag": dict(residual_cache.last_diag),
                "jacobian_diag": dict(jac_cache.last_diag),
            },
        )

    reason = int(snes.getConvergedReason()) if hasattr(snes, "getConvergedReason") else 0
    if exc is not None:
        return FailureInfo(
            failure_class=FailureClass.INNER_FAIL,
            reason_code=f"exception:{type(exc).__name__}",
            message=str(exc),
            where="petsc_snes.solve",
            recoverable=False,
            rollback_required=False,
            meta={"snes_reason": reason},
        )
    return FailureInfo(
        failure_class=FailureClass.INNER_FAIL,
        reason_code=f"snes_reason:{reason}",
        message="PETSc SNES did not converge",
        where="petsc_snes.solve",
        recoverable=True,
        rollback_required=False,
        meta={"snes_reason": reason},
    )


def _compute_dot_a_phys(
    *,
    ctx: NonlinearContext,
    state_vec: object,
    residual_cache: ResidualEvalCache,
) -> float | None:
    getter = ctx.meta.get("dot_a_phys_getter")
    if callable(getter):
        value = getter(ctx=ctx, state_vec=state_vec, residual_cache=residual_cache)
        return None if value is None else float(value)

    snapshot = residual_cache.last_snapshot
    if snapshot is not None and hasattr(snapshot, "meta") and isinstance(snapshot.meta, dict):
        iface_pkg = snapshot.meta.get("iface_pkg")
        vel_pkg = snapshot.meta.get("velocity_pkg")
        if iface_pkg is not None and vel_pkg is not None:
            pkg = build_radius_update_package(iface_pkg, vel_pkg)
            return float(pkg.dot_a_phys)
    return None


def solve_inner_petsc_snes(ctx: NonlinearContext) -> InnerSolveResult:
    """Run one fixed-geometry PETSc SNES solve and return a structured inner result."""

    PETSc = ctx.parallel_handles.get("PETSc") if isinstance(ctx.parallel_handles, dict) and ctx.parallel_handles.get("PETSc") is not None else _get_petsc()
    cfg_snes = build_snes_config_view(ctx)
    ctx.meta.pop("fatal_callback_exception", None)
    ctx.meta.pop("last_guard_result", None)
    ctx.meta.pop("last_failed_guard_result", None)
    ctx.meta.pop("any_guard_triggered", None)
    X = build_initial_state_vec(ctx, PETSc)
    F = build_residual_vec(ctx, PETSc)
    J, P = build_preallocated_jacobian_mats(ctx, PETSc)

    residual_cache = ResidualEvalCache()
    jac_cache = JacobianEvalCache()
    inner_monitor_enabled = bool(_ctx_meta_get(ctx, "runtime_log_inner_monitor", False))
    inner_monitor_stride = max(1, _safe_int(_ctx_meta_get(ctx, "runtime_log_inner_monitor_stride", 1), 1))
    residual_cb = make_residual_callback(
        ctx=ctx,
        residual_cache=residual_cache,
        guard_cfg=None,
        array_view_factory=_array_view_from_vec,
    )
    jacobian_cb = make_jacobian_callback(ctx=ctx, jac_cache=jac_cache)

    comm = ctx.parallel_handles.get("comm")
    snes = PETSc.SNES().create(comm=comm) if hasattr(PETSc.SNES(), "create") else PETSc.SNES()
    if hasattr(snes, "setType"):
        snes.setType(cfg_snes.snes_type)
    if hasattr(snes, "setFunction"):
        snes.setFunction(residual_cb, F)
    if hasattr(snes, "setJacobian"):
        snes.setJacobian(jacobian_cb, J, P)
    if hasattr(snes, "setTolerances"):
        snes.setTolerances(
            rtol=cfg_snes.snes_rtol,
            atol=cfg_snes.snes_atol,
            stol=cfg_snes.snes_stol,
            max_it=cfg_snes.snes_max_it,
        )
    if hasattr(snes, "setOptionsPrefix") and cfg_snes.options_prefix:
        snes.setOptionsPrefix(cfg_snes.options_prefix)
    if hasattr(snes, "setLagJacobian") and cfg_snes.lag_jacobian >= 0:
        snes.setLagJacobian(cfg_snes.lag_jacobian)
    if hasattr(snes, "setLagPreconditioner") and cfg_snes.lag_preconditioner >= 0:
        snes.setLagPreconditioner(cfg_snes.lag_preconditioner)

    ksp = snes.getKSP()
    linear_diag = apply_structured_pc(ksp=ksp, ctx=ctx, A=J, P=P)
    if hasattr(snes, "getLineSearch") and snes.getLineSearch() is not None:
        linesearch = snes.getLineSearch()
        if hasattr(linesearch, "setType"):
            linesearch.setType(cfg_snes.linesearch_type)
    guard_cfg = build_linesearch_guard_config(ctx)
    attach_linesearch_guards(
        snes=snes,
        ctx=ctx,
        residual_cache=residual_cache,
        guard_cfg=guard_cfg,
        array_view_factory=_array_view_from_vec,
    )
    if hasattr(snes, "setFromOptions"):
        snes.setFromOptions()
    finalize_ksp_config(ksp, linear_diag, from_options=True)

    if inner_monitor_enabled and hasattr(snes, "setMonitor"):
        def _snes_monitor(_snes: object, its: int, fnorm: float) -> None:
            iter_idx = int(its) + 1
            if iter_idx % inner_monitor_stride != 0:
                return
            _emit_solver_diag(
                ctx,
                "inner_monitor",
                {
                    **_solver_base_payload(ctx),
                    "snes_iter": iter_idx,
                    "fnorm": _safe_float(fnorm, None),
                    "residual_linf": _safe_float(
                        residual_cache.history_linf[-1] if residual_cache.history_linf else None,
                        None,
                    ),
                },
            )

        snes.setMonitor(_snes_monitor)

    _emit_solver_diag(
        ctx,
        "inner_solve_begin",
        {
            **_solver_base_payload(ctx),
            "parallel_active": bool(_comm_size(ctx) > 1),
            "serial_emulation": bool(_comm_size(ctx) <= 1),
            "comm_size": _safe_int(_comm_size(ctx), None),
            "N": int(ctx.layout.total_size),
            "X_local_size": _safe_int(len(_array_view_from_vec(X, writable=False)), None),
            "jacobian_mode": "assembled",
            "snes_type": getattr(snes, "getType", lambda: cfg_snes.snes_type)(),
            "linesearch_type": getattr(snes.getLineSearch(), "ls_type", cfg_snes.linesearch_type)
            if hasattr(snes, "getLineSearch") and snes.getLineSearch() is not None
            else cfg_snes.linesearch_type,
            "ksp_type": ksp.getType() if hasattr(ksp, "getType") else None,
            "pc_type": ksp.getPC().getType() if hasattr(ksp, "getPC") and ksp.getPC() is not None and hasattr(ksp.getPC(), "getType") else None,
            "f_rtol": _safe_float(cfg_snes.snes_rtol, None),
            "f_atol": _safe_float(cfg_snes.snes_atol, None),
            "max_outer_iter": _safe_int(cfg_snes.snes_max_it, None),
        },
    )

    t_start = perf_counter()
    try:
        snes.solve(None, X)
    except Exception as exc:  # pragma: no cover - exercised by fake PETSc tests
        failure_payload = {
            **_solver_base_payload(ctx),
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
            "snes_reason": _safe_int(snes.getConvergedReason(), None) if hasattr(snes, "getConvergedReason") else None,
            "ksp_reason": _safe_int(ksp.getConvergedReason(), None) if ksp is not None and hasattr(ksp, "getConvergedReason") else None,
            "residual_linf": _safe_float(residual_cache.history_linf[-1] if residual_cache.history_linf else None, None),
            "n_res_eval": _safe_int(residual_cache.n_res_eval, None),
            "n_jac_eval": _safe_int(jac_cache.n_jac_eval, None),
            "time_inner_wall": _safe_float(perf_counter() - t_start, None),
            "jacobian_mode": "assembled",
            "parallel_active": bool(_comm_size(ctx) > 1),
            "serial_emulation": bool(_comm_size(ctx) <= 1),
            "snes_type": getattr(snes, "getType", lambda: cfg_snes.snes_type)(),
            "ksp_type": ksp.getType() if hasattr(ksp, "getType") else None,
            "pc_type": ksp.getPC().getType() if hasattr(ksp, "getPC") and ksp.getPC() is not None and hasattr(ksp.getPC(), "getType") else None,
        }
        _emit_solver_diag(ctx, "inner_solve_failure", failure_payload, level="ERROR")
        raise
    wall_time = perf_counter() - t_start

    stats = collect_inner_stats(
        snes=snes,
        residual_cache=residual_cache,
        jac_cache=jac_cache,
        wall_time_s=wall_time,
        linear_diag=linear_diag,
    )
    stats.guard_triggered = bool(ctx.meta.get("any_guard_triggered", False))
    stats.message = "SNES converged" if stats.converged else "SNES did not converge"

    diagnostics = {
        "linear_pc": linear_diag.meta | {"splits": linear_diag.splits},
        "residual_cache": dict(residual_cache.last_diag),
        "jacobian_cache": dict(jac_cache.last_diag),
        "guard": getattr(ctx.meta.get("last_guard_result"), "diagnostics", {}),
        "parallel_active": bool(_comm_size(ctx) > 1),
        "serial_emulation": bool(_comm_size(ctx) <= 1),
        "inner_monitor_enabled": bool(inner_monitor_enabled),
        "inner_monitor_stride": int(inner_monitor_stride),
        "step_id": _ctx_meta_get(ctx, "step_id", None),
        "outer_iter_index": _ctx_meta_get(ctx, "outer_iter_index", None),
    }

    _emit_solver_diag(
        ctx,
        "inner_solve_end",
        {
            **_solver_base_payload(ctx),
            "converged": bool(stats.converged),
            "snes_reason": _safe_int(stats.snes_reason, None),
            "ksp_reason": _safe_int(stats.ksp_reason, None),
            "inner_iter_count": _safe_int(stats.inner_iter_count, None),
            "linear_iter_count": _safe_int(stats.linear_iter_count, None),
            "residual_l2": _safe_float(stats.residual_norms.l2, None),
            "residual_linf": _safe_float(stats.residual_norms.linf, None),
            "n_res_eval": _safe_int(residual_cache.n_res_eval, None),
            "n_jac_eval": _safe_int(jac_cache.n_jac_eval, None),
            "n_domain_error": _safe_int(residual_cache.n_domain_error, None),
            "time_total": _safe_float(wall_time, None),
            "jacobian_mode": "assembled",
            "snes_type": getattr(snes, "getType", lambda: cfg_snes.snes_type)(),
            "linesearch_type": getattr(snes.getLineSearch(), "ls_type", cfg_snes.linesearch_type)
            if hasattr(snes, "getLineSearch") and snes.getLineSearch() is not None
            else cfg_snes.linesearch_type,
            "ksp_type": ksp.getType() if hasattr(ksp, "getType") else None,
            "pc_type": ksp.getPC().getType() if hasattr(ksp, "getPC") and ksp.getPC() is not None and hasattr(ksp.getPC(), "getType") else None,
        },
        level="INFO" if stats.converged else "WARNING",
    )

    if stats.converged:
        state_vec = X.copy() if hasattr(X, "copy") else X
        dot_a_phys = _compute_dot_a_phys(ctx=ctx, state_vec=state_vec, residual_cache=residual_cache)
        result = InnerSolveResult(
            converged=True,
            state_vec=state_vec,
            dot_a_phys=dot_a_phys,
            old_state_on_current_geometry=ctx.old_state_on_current_geometry,
            stats=stats,
            failure=FailureInfo(),
            diagnostics=diagnostics,
        )
        result.assert_consistent()
        return result

    failure = map_snes_failure(
        ctx=ctx,
        snes=snes,
        residual_cache=residual_cache,
        jac_cache=jac_cache,
    )
    result = InnerSolveResult(
        converged=False,
        state_vec=None,
        dot_a_phys=None,
        old_state_on_current_geometry=ctx.old_state_on_current_geometry,
        stats=stats,
        failure=failure,
        diagnostics=diagnostics,
    )
    result.assert_consistent()
    return result


__all__ = [
    "JacobianEvalCache",
    "ResidualEvalCache",
    "SnesConfigView",
    "_get_petsc",
    "_normalize_linesearch_type",
    "_normalize_snes_type",
    "attach_linesearch_guards",
    "build_initial_state_vec",
    "build_preallocated_jacobian_mats",
    "build_residual_vec",
    "build_snes_config_view",
    "collect_inner_stats",
    "make_jacobian_callback",
    "make_residual_callback",
    "map_snes_failure",
    "solve_inner_petsc_snes",
]
