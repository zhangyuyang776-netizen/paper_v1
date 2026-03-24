from __future__ import annotations

"""Single-step outer/inner orchestration for Phase 5.

This module drives one complete step attempt by composing the already-frozen
predictor, remap/recovery, nonlinear context, inner SNES solve, outer
convergence, outer corrector, and step-acceptance modules. It does not rebuild
an alternative physics / assembly / solver mainline.
"""

from dataclasses import dataclass, replace
from math import isfinite

import numpy as np

from core.grid import build_grid_and_metrics
from core.layout import build_layout
from core.remap import RemapError, build_old_state_on_current_geometry as remap_old_state_on_current_geometry
from core.state_pack import unpack_array_to_state
from core.state_recovery import StateRecoveryError
from core.types import GeometryState, StateTransferRecord
from properties.aggregator import AggregatorError, build_bulk_props
from .nonlinear_context import NonlinearModelHandles, build_nonlinear_context
from .nonlinear_types import (
    FailureClass,
    FailureInfo,
    InnerEntrySource,
    OuterIterationResult,
    StepAction,
    StepAdvanceResult,
)
from .outer_convergence import evaluate_outer_convergence
from .outer_corrector import compute_outer_corrector
from .outer_predictor import compute_outer_predictor
from .petsc_snes import solve_inner_petsc_snes
from .step_acceptance import decide_step_acceptance


@dataclass(slots=True, kw_only=True)
class TimestepperConfigView:
    """Normalized top-level stepper controls used by one step attempt."""

    dt_min: float
    dt_max: float
    max_retries_per_step: int
    outer_max_iter: int


class _OuterStageError(RuntimeError):
    """Private wrapper used to propagate structured outer-stage failures."""

    def __init__(self, failure: FailureInfo):
        super().__init__(failure.message or failure.reason_code or "outer stage failed")
        self.failure = failure


def _validate_positive_finite(name: str, value: float) -> float:
    value_f = float(value)
    if not isfinite(value_f):
        raise ValueError(f"{name} must be finite")
    if value_f <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value_f


def _validate_nonnegative_int(name: str, value: int) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{name} must be >= 0")
    return ivalue


def _as_model_handles(models: object | None) -> NonlinearModelHandles:
    if isinstance(models, NonlinearModelHandles):
        return models
    if models is None:
        return NonlinearModelHandles()
    extra = getattr(models, "extra", None)
    return NonlinearModelHandles(
        gas_model=getattr(models, "gas_model", None),
        liquid_model=getattr(models, "liquid_model", None),
        equilibrium_model=getattr(models, "equilibrium_model", None),
        property_aggregator=getattr(models, "property_aggregator", None),
        liquid_db=getattr(models, "liquid_db", None),
        extra=dict(extra) if isinstance(extra, dict) else {},
    )


def _extra(models: NonlinearModelHandles) -> dict[str, object]:
    return models.extra if isinstance(models.extra, dict) else {}


def _make_default_ownership(*, n_liq: int, n_gas: int) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        owned_liq_cells=list(range(int(n_liq))),
        owned_gas_cells=list(range(int(n_gas))),
        interface_owner_active=True,
    )


def _parallel_size(parallel_handles: dict[str, object] | None) -> int:
    handles = parallel_handles or {}
    comm = handles.get("comm")
    if comm is None:
        return 1
    size_attr = getattr(comm, "size", None)
    if size_attr is not None:
        return int(size_attr)
    get_size = getattr(comm, "Get_size", None)
    if callable(get_size):
        return int(get_size())
    return 1


def _resolve_iteration_ownership(
    *,
    kind: str,
    extra: dict[str, object],
    parallel_handles: dict[str, object] | None,
    n_liq: int,
    n_gas: int,
) -> object:
    explicit = extra.get(kind)
    if explicit is not None:
        return explicit
    if _parallel_size(parallel_handles) > 1:
        raise ValueError(
            f"multi-rank timestepper requires explicit distributed-safe {kind}"
        )
    return _make_default_ownership(n_liq=n_liq, n_gas=n_gas)


def _array_view_from_state_vec(state_vec: object) -> np.ndarray:
    if hasattr(state_vec, "getArray"):
        return np.asarray(state_vec.getArray(readonly=True), dtype=np.float64)
    if hasattr(state_vec, "array"):
        return np.asarray(getattr(state_vec, "array"), dtype=np.float64)
    return np.asarray(state_vec, dtype=np.float64)


def build_timestepper_config_view(cfg: object) -> TimestepperConfigView:
    """Build a normalized timestepper config view from runtime configuration."""

    ts = getattr(cfg, "time_stepper", None)
    outer = getattr(cfg, "outer_stepper", None)

    dt_min = _validate_positive_finite("dt_min", float(getattr(ts, "dt_min", 1.0e-9)))
    dt_max = _validate_positive_finite("dt_max", float(getattr(ts, "dt_max", 1.0e9)))
    if dt_max < dt_min:
        raise ValueError("dt_max must be >= dt_min")
    max_retries = _validate_nonnegative_int(
        "max_retries_per_step",
        int(getattr(ts, "max_retries_per_step", getattr(ts, "retry_max_per_step", 8))),
    )
    outer_max_iter = int(getattr(outer, "outer_max_iter", 1))
    if outer_max_iter <= 0:
        raise ValueError("outer_max_iter must be > 0")
    return TimestepperConfigView(
        dt_min=dt_min,
        dt_max=dt_max,
        max_retries_per_step=max_retries,
        outer_max_iter=outer_max_iter,
    )


def build_current_geometry_from_radius(
    *,
    cfg: object,
    step_id: int,
    t_old: float,
    dt: float,
    a_iter: float,
    dot_a_iter: float,
    outer_iter_index: int,
    accepted_geometry_old: object,
    models: object | None = None,
) -> tuple[object, object, object, object]:
    """Build current geometry/grid/metrics/layout from the current outer iterate."""

    models_h = _as_model_handles(models)
    hook = _extra(models_h).get("build_current_geometry_from_radius")
    if callable(hook):
        return hook(
            cfg=cfg,
            step_id=step_id,
            t_old=t_old,
            dt=dt,
            a_iter=a_iter,
            dot_a_iter=dot_a_iter,
            outer_iter_index=outer_iter_index,
            accepted_geometry_old=accepted_geometry_old,
            models=models_h,
        )

    geometry_current = GeometryState(
        t=float(t_old) + float(dt),
        dt=float(dt),
        a=float(a_iter),
        dot_a=float(dot_a_iter),
        r_end=float(getattr(accepted_geometry_old, "r_end")),
        step_index=int(step_id),
        outer_iter_index=int(outer_iter_index),
    )
    mesh_current, metrics_current = build_grid_and_metrics(cfg, geometry_current)
    layout_current = build_layout(cfg, mesh_current)
    return geometry_current, mesh_current, metrics_current, layout_current


def build_transfer_state_for_next_outer_iter(
    *,
    cfg: object,
    state_converged_k: object,
    mesh_k: object,
    geometry_k: object,
    mesh_k1: object,
    geometry_k1: object,
    outer_iter_index: int | None = None,
    models: object | None = None,
) -> StateTransferRecord:
    """Build the outer transfer record for the next outer iteration only."""

    models_h = _as_model_handles(models)
    hook = _extra(models_h).get("build_transfer_state_for_next_outer_iter")
    if callable(hook):
        return hook(
            cfg=cfg,
            state_converged_k=state_converged_k,
            mesh_k=mesh_k,
            geometry_k=geometry_k,
            mesh_k1=mesh_k1,
            geometry_k1=geometry_k1,
            outer_iter_index=outer_iter_index,
            models=models_h,
        )

    liquid_model = models_h.liquid_model
    gas_model = models_h.gas_model
    if liquid_model is None or gas_model is None:
        raise ValueError("timestepper default remap/recovery path requires liquid_model and gas_model")
    transfer = remap_old_state_on_current_geometry(
        old_state=state_converged_k,
        old_mesh=mesh_k,
        new_mesh=mesh_k1,
        geometry=geometry_k1,
        recovery_config=cfg.recovery,
        species_maps=cfg.species_maps,
        liquid_thermo=liquid_model,
        gas_thermo=gas_model,
    )
    return StateTransferRecord(
        contents=transfer.contents,
        state=transfer.state,
        geometry=transfer.geometry,
        mesh=transfer.mesh,
        source_outer_iter_index=None if outer_iter_index is None else int(outer_iter_index),
        identity_transfer=bool(getattr(transfer, "identity_transfer", False)),
    )


def build_old_state_on_current_geometry(
    *,
    cfg: object,
    accepted_state_old: object,
    accepted_geometry_old: object,
    geometry_current: object,
    mesh_current: object,
    models: object | None = None,
) -> tuple[object, object]:
    """Compatibility wrapper for the superseded pre-inner remap path.

    New timestepper mainline must not use this function for the k=0 outer
    entry. It remains only for legacy callers and dedicated transition tests.
    """

    old_mesh, _ = build_grid_and_metrics(cfg, accepted_geometry_old)
    transfer = build_transfer_state_for_next_outer_iter(
        cfg=cfg,
        state_converged_k=accepted_state_old,
        mesh_k=old_mesh,
        geometry_k=accepted_geometry_old,
        mesh_k1=mesh_current,
        geometry_k1=geometry_current,
        models=models,
    )
    return transfer, transfer.contents


def _build_props_current(
    *,
    cfg: object,
    state_guess: object,
    mesh_current: object,
    models: NonlinearModelHandles,
) -> object:
    hook = _extra(models).get("build_props_current")
    if callable(hook):
        return hook(cfg=cfg, state_guess=state_guess, mesh_current=mesh_current, models=models)

    aggregator = models.property_aggregator
    if callable(aggregator):
        return aggregator(
            state=state_guess,
            grid=mesh_current,
            liquid_thermo=models.liquid_model,
            gas_thermo=models.gas_model,
            gas_pressure=float(getattr(cfg, "pressure")),
        )

    if models.liquid_model is None or models.gas_model is None:
        raise ValueError("timestepper default property path requires liquid_model and gas_model")
    return build_bulk_props(
        state=state_guess,
        grid=mesh_current,
        liquid_thermo=models.liquid_model,
        gas_thermo=models.gas_model,
        gas_pressure=float(getattr(cfg, "pressure")),
    )


def build_state_guess_for_outer_iter(
    *,
    outer_iter_index: int,
    accepted_state_n: object | None = None,
    transfer_in: StateTransferRecord | None = None,
    entry_source: InnerEntrySource | str | None = None,
    previous_inner_result: object | None = None,
    accepted_state_old: object | None = None,
    old_state_on_current_geometry: object | None = None,
    **_: object,
) -> tuple[object, object | None]:
    """Select the current inner-entry state only.

    This function does not perform remap or recovery. For k=0 it returns the
    accepted time-level state. For k>0 it returns ``transfer_in.state``.
    """

    accepted_state_n_eff = accepted_state_n if accepted_state_n is not None else accepted_state_old
    transfer_in_eff = transfer_in if transfer_in is not None else old_state_on_current_geometry
    if entry_source is None:
        raise ValueError("build_state_guess_for_outer_iter requires explicit entry_source")
    entry_source_eff = InnerEntrySource(entry_source)
    if entry_source_eff is InnerEntrySource.ACCEPTED_TIME_LEVEL:
        if accepted_state_n_eff is None:
            raise ValueError("accepted_time_level entry requires accepted_state_n")
        return accepted_state_n_eff, None
    if transfer_in_eff is None:
        raise ValueError("transfer_from_previous_outer entry requires transfer_in")
    previous_vec = None if previous_inner_result is None else getattr(previous_inner_result, "solution_vec", None)
    if previous_vec is None and previous_inner_result is not None:
        previous_vec = getattr(previous_inner_result, "state_vec", None)
    return getattr(transfer_in_eff, "state", transfer_in_eff), previous_vec


def _recover_state_from_inner_solution(
    *,
    inner_result: object,
    layout: object,
    species_maps: object,
    geometry_current: object | None,
    accepted_state_n: object,
    transfer_in: StateTransferRecord | None,
    props_current: object,
    models: NonlinearModelHandles,
    parallel_handles: dict[str, object] | None,
) -> object:
    """Recover a State view from one converged inner solution for transfer."""

    solution_vec = getattr(inner_result, "solution_vec", None)
    if solution_vec is None:
        solution_vec = getattr(inner_result, "state_vec", None)
    if solution_vec is None:
        raise ValueError("inner_result.solution_vec is required for transfer construction")

    extra = _extra(models)
    state_base = transfer_in if transfer_in is not None else accepted_state_n
    hook = extra.get("state_init_from_previous_inner_result")
    if not callable(hook):
        hook = extra.get("state_guess_from_previous_inner_result")
    if _parallel_size(parallel_handles) > 1:
        if not callable(hook):
            raise ValueError(
                "transfer recovery from previous inner result requires explicit state_init_from_previous_inner_result hook in multi-rank mode"
            )
        try:
            state_current = hook(
                previous_inner_result=inner_result,
                reference_state_current_mesh=state_base,
                layout=layout,
                species_maps=species_maps,
                geometry_current=geometry_current,
                parallel_handles=parallel_handles,
                models=models,
            )
        except TypeError:
            state_current = hook(
                previous_inner_result=inner_result,
                old_state_on_current_geometry=state_base,
                layout=layout,
                species_maps=species_maps,
                geometry_current=geometry_current,
                parallel_handles=parallel_handles,
                models=models,
            )
    else:
        vec_view = np.array(_array_view_from_state_vec(solution_vec), copy=True)
        state_seed = getattr(state_base, "state", state_base)
        state_current = unpack_array_to_state(
            vec_view,
            layout,
            species_maps,
            time=getattr(geometry_current, "t", getattr(state_seed, "time", None)),
            state_id=getattr(state_seed, "state_id", None),
        )

    rho_l = getattr(props_current, "rho_l", None)
    rho_g = getattr(props_current, "rho_g", None)
    hl = getattr(props_current, "h_l", getattr(props_current, "hl", None))
    hg = getattr(props_current, "h_g", getattr(props_current, "hg", None))
    if any(value is None for value in (rho_l, rho_g, hl, hg)):
        return state_current
    return replace(
        state_current,
        rho_l=rho_l,
        rho_g=rho_g,
        hl=hl,
        hg=hg,
    )


def _emit_diag(diag_sink: object | None, *, event: str, payload: object) -> None:
    if diag_sink is None:
        return
    if callable(diag_sink):
        diag_sink(event=event, payload=payload)
        return
    write = getattr(diag_sink, "write", None)
    if callable(write):
        write(event=event, payload=payload)


def _failure_class_value(obj: object) -> str | None:
    if obj is None:
        return None
    return getattr(obj, "value", str(obj))


def _predictor_payload(
    *,
    step_id: int,
    t_old: float,
    dt_attempt: float,
    a_old: float,
    dot_a_old: float,
    predictor: object,
) -> dict[str, object]:
    return {
        "step_id": int(step_id),
        "t_old": float(t_old),
        "dt_attempt": float(dt_attempt),
        "a_old": float(a_old),
        "dot_a_old": float(dot_a_old),
        "a_pred": getattr(predictor, "a_pred", None),
        "dot_a_pred": getattr(predictor, "dot_a_pred", None),
        "first_step_special_case": getattr(predictor, "first_step_special_case", None),
    }


def _outer_iter_payload(
    *,
    step_id: int,
    outer_iter_index: int,
    a_iter: float,
    dot_a_iter: float,
    t_old: float | None = None,
    dt: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "step_id": int(step_id),
        "outer_iter_index": int(outer_iter_index),
        "a_iter": float(a_iter),
        "dot_a_iter": float(dot_a_iter),
    }
    if t_old is not None:
        payload["t_old"] = float(t_old)
    if dt is not None:
        payload["dt"] = float(dt)
    return payload


def _step_result_payload(step_result: StepAdvanceResult) -> dict[str, object]:
    last_outer = step_result.outer_iterations[-1] if step_result.outer_iterations else None
    stats = getattr(getattr(last_outer, "inner", None), "stats", None)
    residuals = getattr(stats, "residual_norms", None)
    failure = getattr(step_result, "failure", None)
    acceptance = getattr(step_result, "acceptance", None)
    return {
        "step_id": int(step_result.step_id),
        "accepted": bool(step_result.accepted),
        "dt_attempt": float(step_result.dt_attempt),
        "accepted_geometry": getattr(step_result, "accepted_geometry", None),
        "acceptance": {
            "action": getattr(getattr(acceptance, "action", None), "value", getattr(acceptance, "action", None)),
            "accepted": getattr(acceptance, "accepted", None),
            "dt_current": getattr(acceptance, "dt_current", None),
            "dt_next": getattr(acceptance, "dt_next", None),
            "reject_reason": getattr(acceptance, "reject_reason", None),
        },
        "failure": {
            "failure_class": _failure_class_value(getattr(failure, "failure_class", None)),
            "reason_code": getattr(failure, "reason_code", None),
            "message": getattr(failure, "message", None),
        },
        "diagnostics": {
            "dt_attempt": getattr(step_result, "dt_attempt", None),
            "retries_used": getattr(step_result, "diagnostics", {}).get("retries_used"),
            "outer_iter_count": len(step_result.outer_iterations),
            "outer_converged": getattr(step_result, "diagnostics", {}).get("outer_converged"),
            "reject_reason": getattr(step_result, "diagnostics", {}).get("reject_reason"),
            "inner_converged": getattr(getattr(last_outer, "inner", None), "converged", None),
            "inner_iter_count": getattr(stats, "inner_iter_count", None),
            "linear_iter_count": getattr(stats, "linear_iter_count", None),
            "residual_l2": getattr(residuals, "l2", None),
            "residual_linf": getattr(residuals, "linf", None),
            "snes_reason": getattr(stats, "snes_reason", None),
            "ksp_reason": getattr(stats, "ksp_reason", None),
            "dot_a_phys": getattr(getattr(last_outer, "inner", None), "dot_a_phys", None),
        },
    }


def _wrap_outer_stage_exception(
    exc: Exception,
    *,
    stage: str,
    substage: str | None = None,
    extra_meta: dict[str, object] | None = None,
) -> FailureInfo:
    exc_type = type(exc).__name__
    if isinstance(exc, RemapError):
        failure_class = FailureClass.REMAP_FAIL
    elif isinstance(exc, StateRecoveryError):
        failure_class = FailureClass.RECOVERY_FAIL
    elif isinstance(exc, AggregatorError):
        failure_class = FailureClass.PROPERTY_FAIL
    else:
        failure_class = FailureClass.FATAL_STOP
    reason_code = f"{stage}:{substage}:{exc_type}" if substage else f"{stage}:{exc_type}"
    meta = {
        "stage": stage,
        "substage": substage,
        "exception_type": exc_type,
    }
    if str(exc):
        meta["message"] = str(exc)
    if extra_meta:
        meta.update(extra_meta)
    return FailureInfo(
        failure_class=failure_class,
        reason_code=reason_code,
        message=str(exc),
        where="timestepper",
        recoverable=False,
        rollback_required=True,
        meta=meta,
    )


def run_single_outer_iteration(
    *,
    cfg: object,
    step_id: int,
    outer_iter_index: int,
    t_old: float,
    dt: float,
    a_old: float,
    dot_a_old: float,
    a_iter: float,
    dot_a_iter: float,
    accepted_state_old: object,
    accepted_geometry_old: object,
    entry_source: InnerEntrySource | str,
    transfer_in: StateTransferRecord | None = None,
    models: object | None,
    previous_inner_result: object | None = None,
    previous_eps_dot_a: float | None = None,
    parallel_handles: dict[str, object] | None = None,
    diag_sink: object | None = None,
) -> OuterIterationResult:
    """Run one complete outer iteration up to convergence check / corrector."""

    accepted_state_n = accepted_state_old
    accepted_geometry_n = accepted_geometry_old
    entry_source_eff = InnerEntrySource(entry_source)
    models_h = _as_model_handles(models)
    _emit_diag(
        diag_sink,
        event="outer_iter_begin",
        payload=_outer_iter_payload(
            step_id=step_id,
            outer_iter_index=outer_iter_index,
            a_iter=a_iter,
            dot_a_iter=dot_a_iter,
            t_old=t_old,
            dt=dt,
        ),
    )

    def _raise_stage_error(exc: Exception, *, substage: str) -> None:
        payload = {
            "step_id": int(step_id),
            "outer_iter_index": int(outer_iter_index),
            "stage": substage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        _emit_diag(diag_sink, event="outer_exception", payload=payload)
        raise _OuterStageError(
            _wrap_outer_stage_exception(
                exc,
                stage="outer_iteration",
                substage=substage,
                extra_meta={
                    "step_id": int(step_id),
                    "outer_iter_index": int(outer_iter_index),
                },
            )
        ) from exc

    try:
        geometry_current, mesh_current, metrics_current, layout_current = build_current_geometry_from_radius(
            cfg=cfg,
            step_id=step_id,
            t_old=t_old,
            dt=dt,
            a_iter=a_iter,
            dot_a_iter=dot_a_iter,
            outer_iter_index=outer_iter_index,
            accepted_geometry_old=accepted_geometry_old,
            models=models_h,
        )
    except Exception as exc:
        _raise_stage_error(exc, substage="build_current_geometry_from_radius")

    try:
        state_init, u_guess_vec = build_state_guess_for_outer_iter(
            outer_iter_index=outer_iter_index,
            accepted_state_n=accepted_state_n,
            transfer_in=transfer_in,
            entry_source=entry_source_eff,
            previous_inner_result=previous_inner_result,
        )
    except Exception as exc:
        _raise_stage_error(exc, substage="build_state_guess_for_outer_iter")

    try:
        props_current = _build_props_current(
            cfg=cfg,
            state_guess=state_init,
            mesh_current=mesh_current,
            models=models_h,
        )
    except Exception as exc:
        _raise_stage_error(exc, substage="build_props_current")

    extra = _extra(models_h)
    parallel_meta = dict(parallel_handles or {})
    ctx_meta: dict[str, object] = {
        "mesh": mesh_current,
        "species_maps": cfg.species_maps,
        "run_cfg": cfg,
        "liquid_thermo": models_h.liquid_model,
        "gas_thermo": models_h.gas_model,
        "control_surface_metrics": metrics_current,
        "farfield_bc": extra.get("farfield_bc"),
        "residual_ownership": _resolve_iteration_ownership(
            kind="residual_ownership",
            extra=extra,
            parallel_handles=parallel_handles,
            n_liq=layout_current.n_liq_cells,
            n_gas=layout_current.n_gas_cells,
        ),
        "jacobian_ownership": _resolve_iteration_ownership(
            kind="jacobian_ownership",
            extra=extra,
            parallel_handles=parallel_handles,
            n_liq=layout_current.n_liq_cells,
            n_gas=layout_current.n_gas_cells,
        ),
    }
    for key in (
        "assemble_global_residual_from_layout_vector",
        "build_all_residual_blocks_from_layout_vector",
        "assemble_and_insert_global_jacobian",
        "dot_a_phys_getter",
        "jacobian_pattern",
        "global_jacobian_fd_options",
        "distributed_state_vec_builder",
        "distributed_residual_vec_builder",
        "u_trial_layout_guess",
    ):
        if key in extra:
            ctx_meta[key] = extra[key]

    try:
        ctx = build_nonlinear_context(
            cfg=cfg,
            layout=layout_current,
            grid=mesh_current,
            t_old=t_old,
            dt=dt,
            a_current=a_iter,
            dot_a_frozen=dot_a_iter,
            state_init=state_init,
            accepted_state_n=accepted_state_n,
            entry_source=entry_source_eff,
            transfer_in=transfer_in,
            reference_state_current_mesh=None if transfer_in is None else transfer_in.state,
            reference_contents_current_mesh=None if transfer_in is None else transfer_in.contents,
            props_current=props_current,
            models=models_h,
            step_id=step_id,
            outer_iter_id=outer_iter_index,
            geometry_current=geometry_current,
            u_guess_vec=u_guess_vec,
            dm_composite=parallel_meta.get("dm_composite"),
            global_vec_template=parallel_meta.get("global_vec_template"),
            parallel_handles=parallel_meta,
            meta=ctx_meta,
            diagnostics={
                "step_id": int(step_id),
                "outer_iter_index": int(outer_iter_index),
            },
        )
    except Exception as exc:
        _raise_stage_error(exc, substage="build_nonlinear_context")

    try:
        inner_result = solve_inner_petsc_snes(ctx)
    except Exception as exc:
        _raise_stage_error(exc, substage="solve_inner_petsc_snes")
    if getattr(inner_result, "entry_source", None) is None:
        inner_result.entry_source = entry_source_eff
    _emit_diag(
        diag_sink,
        event="outer_inner_result",
        payload={
            "step_id": int(step_id),
            "outer_iter_index": int(outer_iter_index),
            "inner_converged": bool(inner_result.converged),
            "inner_iter_count": getattr(inner_result.stats, "inner_iter_count", None),
            "linear_iter_count": getattr(inner_result.stats, "linear_iter_count", None),
            "residual_l2": getattr(getattr(inner_result.stats, "residual_norms", None), "l2", None),
            "residual_linf": getattr(getattr(inner_result.stats, "residual_norms", None), "linf", None),
            "snes_reason": getattr(inner_result.stats, "snes_reason", None),
            "ksp_reason": getattr(inner_result.stats, "ksp_reason", None),
            "dot_a_phys": getattr(inner_result, "dot_a_phys", None),
            "entry_source": entry_source_eff.value,
            "used_transfer": bool(transfer_in is not None),
        },
    )
    if not inner_result.converged:
        return OuterIterationResult(
            outer_iter_index=int(outer_iter_index),
            a_iter=float(a_iter),
            dot_a_iter=float(dot_a_iter),
            inner=inner_result,
            entry_source=entry_source_eff,
            used_transfer=transfer_in is not None,
            transfer_identity=None if transfer_in is None else bool(getattr(transfer_in, "identity_transfer", False)),
            corrector=None,
            convergence=None,
            diagnostics={
                "geometry_current": geometry_current,
                "mesh_current": mesh_current,
                "layout_current": layout_current,
                "entry_source": entry_source_eff.value,
                "transfer_identity": None if transfer_in is None else bool(getattr(transfer_in, "identity_transfer", False)),
                "state_init": state_init,
                "transfer_in": transfer_in,
                "props_current": props_current,
            },
        )

    if inner_result.dot_a_phys is None:
        _raise_stage_error(ValueError("inner solve converged without dot_a_phys"), substage="solve_inner_petsc_snes")
    try:
        convergence = evaluate_outer_convergence(
            cfg=cfg,
            dot_a_iter=dot_a_iter,
            dot_a_phys=float(inner_result.dot_a_phys),
            outer_iter_index=outer_iter_index,
            eps_dot_a_prev=previous_eps_dot_a,
        )
    except Exception as exc:
        _raise_stage_error(exc, substage="evaluate_outer_convergence")
    _emit_diag(
        diag_sink,
        event="outer_convergence_result",
        payload={
            "step_id": int(step_id),
            "outer_iter_index": int(outer_iter_index),
            "dot_a_iter": float(dot_a_iter),
            "dot_a_phys": float(inner_result.dot_a_phys),
            "eps_dot_a": getattr(convergence, "eps_dot_a", None),
            "tolerance": getattr(convergence, "tolerance", None),
            "converged": bool(convergence.converged),
        },
    )
    corrector = None
    if not convergence.converged:
        try:
            corrector = compute_outer_corrector(
                cfg=cfg,
                dt=dt,
                a_old=a_old,
                dot_a_old=dot_a_old,
                a_iter=a_iter,
                dot_a_iter=dot_a_iter,
                dot_a_phys=float(inner_result.dot_a_phys),
                outer_iter_index=outer_iter_index,
            )
        except Exception as exc:
            _raise_stage_error(exc, substage="compute_outer_corrector")
        _emit_diag(
            diag_sink,
            event="outer_corrector_result",
            payload={
                "step_id": int(step_id),
                "outer_iter_index": int(outer_iter_index),
                "a_new": getattr(corrector, "a_new", None),
                "dot_a_new": getattr(corrector, "dot_a_new", None),
            },
        )
    return OuterIterationResult(
        outer_iter_index=int(outer_iter_index),
        a_iter=float(a_iter),
        dot_a_iter=float(dot_a_iter),
        inner=inner_result,
        entry_source=entry_source_eff,
        used_transfer=transfer_in is not None,
        transfer_identity=None if transfer_in is None else bool(getattr(transfer_in, "identity_transfer", False)),
        corrector=corrector,
        convergence=convergence,
        diagnostics={
            "geometry_current": geometry_current,
            "mesh_current": mesh_current,
            "layout_current": layout_current,
            "entry_source": entry_source_eff.value,
            "transfer_identity": None if transfer_in is None else bool(getattr(transfer_in, "identity_transfer", False)),
            "state_init": state_init,
            "transfer_in": transfer_in,
            "props_current": props_current,
        },
    )


def advance_one_step(
    *,
    cfg: object,
    step_id: int,
    t_old: float,
    dt: float,
    accepted_state: object | None = None,
    accepted_state_n: object | None = None,
    accepted_geometry: object | None = None,
    accepted_geometry_n: object | None = None,
    accepted_mesh: object | None = None,
    accepted_layout: object | None = None,
    accepted_props: object | None = None,
    accepted_state_old: object | None = None,
    accepted_geometry_old: object | None = None,
    dot_a_old: float,
    models: object | None = None,
    parallel_handles: dict[str, object] | None = None,
    diag_sink: object | None = None,
) -> StepAdvanceResult:
    """Advance one step attempt through predictor, outer loop, and acceptance."""

    cfg_ts = build_timestepper_config_view(cfg)
    t_old_f = float(t_old)
    dt_f = _validate_positive_finite("dt", float(dt))
    if accepted_state is not None and accepted_state_old is not None:
        raise ValueError("Provide either accepted_state or accepted_state_old, not both")
    if accepted_state_n is not None and accepted_state_old is not None:
        raise ValueError("Provide either accepted_state_n or accepted_state_old, not both")
    if accepted_state is not None and accepted_state_n is not None and accepted_state is not accepted_state_n:
        raise ValueError("accepted_state and accepted_state_n must refer to the same object when both are provided")
    if accepted_geometry is not None and accepted_geometry_old is not None:
        raise ValueError("Provide either accepted_geometry or accepted_geometry_old, not both")
    if accepted_geometry_n is not None and accepted_geometry_old is not None:
        raise ValueError("Provide either accepted_geometry_n or accepted_geometry_old, not both")
    if (
        accepted_geometry is not None
        and accepted_geometry_n is not None
        and accepted_geometry is not accepted_geometry_n
    ):
        raise ValueError(
            "accepted_geometry and accepted_geometry_n must refer to the same object when both are provided"
        )
    accepted_state_old = (
        accepted_state
        if accepted_state is not None
        else accepted_state_n
        if accepted_state_n is not None
        else accepted_state_old
    )
    accepted_geometry_old = (
        accepted_geometry
        if accepted_geometry is not None
        else accepted_geometry_n
        if accepted_geometry_n is not None
        else accepted_geometry_old
    )
    if accepted_state_old is None:
        raise ValueError("accepted_state or accepted_state_n must be provided")
    if accepted_geometry_old is None:
        raise ValueError("accepted_geometry or accepted_geometry_n must be provided")
    _ = (accepted_mesh, accepted_layout, accepted_props)
    if not isfinite(float(dot_a_old)):
        raise ValueError("dot_a_old must be finite")
    if getattr(accepted_geometry_old, "a", None) is None or float(getattr(accepted_geometry_old, "a")) <= 0.0:
        raise ValueError("accepted_geometry_old must provide positive radius a")

    dt_attempt = dt_f
    retries_used = 0
    models_h = _as_model_handles(models)

    while True:
        accepted_state_n = accepted_state_old
        accepted_geometry_n = accepted_geometry_old
        predictor = compute_outer_predictor(
            cfg=cfg,
            t_old=t_old_f,
            dt=dt_attempt,
            a_old=float(getattr(accepted_geometry_n, "a")),
            dot_a_old=float(dot_a_old),
            step_id=step_id,
        )
        _emit_diag(
            diag_sink,
            event="predictor_ready",
            payload=_predictor_payload(
                step_id=step_id,
                t_old=t_old_f,
                dt_attempt=dt_attempt,
                a_old=float(getattr(accepted_geometry_n, "a")),
                dot_a_old=float(dot_a_old),
                predictor=predictor,
            ),
        )
        outer_iterations: list[OuterIterationResult] = []
        accepted_state_candidate: object | None = None
        accepted_geometry_candidate: object | None = None
        fatal_failure: FailureInfo | None = None

        a_iter = float(predictor.a_pred)
        dot_a_iter = float(predictor.dot_a_pred)
        previous_inner_result: object | None = None
        previous_eps_dot_a: float | None = None
        entry_source = InnerEntrySource.ACCEPTED_TIME_LEVEL
        transfer_in: StateTransferRecord | None = None

        for outer_iter_index in range(cfg_ts.outer_max_iter):
            try:
                outer_iter = run_single_outer_iteration(
                    cfg=cfg,
                    step_id=step_id,
                    outer_iter_index=outer_iter_index,
                    t_old=t_old_f,
                    dt=dt_attempt,
                    a_old=float(getattr(accepted_geometry_n, "a")),
                    dot_a_old=float(dot_a_old),
                    a_iter=float(a_iter),
                    dot_a_iter=float(dot_a_iter),
                    accepted_state_old=accepted_state_n,
                    accepted_geometry_old=accepted_geometry_n,
                    entry_source=entry_source,
                    transfer_in=transfer_in,
                    models=models_h,
                    previous_inner_result=previous_inner_result,
                    previous_eps_dot_a=previous_eps_dot_a,
                    parallel_handles=parallel_handles,
                    diag_sink=diag_sink,
                )
            except _OuterStageError as exc:
                fatal_failure = exc.failure
                break
            except Exception as exc:
                fatal_failure = _wrap_outer_stage_exception(
                    exc,
                    stage="outer_iteration",
                    substage="unexpected_outer_wrapper",
                )
                break

            outer_iterations.append(outer_iter)
            _emit_diag(diag_sink, event="outer_iteration", payload=outer_iter)
            previous_inner_result = outer_iter.inner
            if not outer_iter.inner.converged:
                break

            if outer_iter.convergence is None:
                raise ValueError("converged inner iteration must provide outer convergence result")
            previous_eps_dot_a = outer_iter.convergence.eps_dot_a
            if outer_iter.convergence.converged:
                accepted_state_candidate = outer_iter.inner.solution_vec
                accepted_geometry_candidate = outer_iter.diagnostics.get("geometry_current")
                break

            if outer_iter.corrector is None:
                raise ValueError("non-converged outer iteration must provide a corrector result")
            geometry_current = outer_iter.diagnostics.get("geometry_current")
            mesh_current = outer_iter.diagnostics.get("mesh_current")
            layout_current = outer_iter.diagnostics.get("layout_current")
            props_current = outer_iter.diagnostics.get("props_current")
            if geometry_current is None or mesh_current is None or layout_current is None or props_current is None:
                raise ValueError(
                    "non-converged outer iteration must retain geometry_current, mesh_current, layout_current, and props_current"
                )
            next_a_iter = float(outer_iter.corrector.a_new)
            next_dot_a_iter = float(outer_iter.corrector.dot_a_new)
            try:
                state_converged_k = _recover_state_from_inner_solution(
                    inner_result=outer_iter.inner,
                    layout=layout_current,
                    species_maps=cfg.species_maps,
                    geometry_current=geometry_current,
                    accepted_state_n=accepted_state_n,
                    transfer_in=transfer_in,
                    props_current=props_current,
                    models=models_h,
                    parallel_handles=parallel_handles,
                )
            except Exception as exc:
                fatal_failure = _wrap_outer_stage_exception(
                    exc,
                    stage="outer_iteration",
                    substage="recover_state_from_inner_solution",
                    extra_meta={
                        "step_id": int(step_id),
                        "outer_iter_index": int(outer_iter_index),
                    },
                )
                break
            try:
                geometry_k1, mesh_k1, _, _ = build_current_geometry_from_radius(
                    cfg=cfg,
                    step_id=step_id,
                    t_old=t_old_f,
                    dt=dt_attempt,
                    a_iter=next_a_iter,
                    dot_a_iter=next_dot_a_iter,
                    outer_iter_index=outer_iter_index + 1,
                    accepted_geometry_old=accepted_geometry_n,
                    models=models_h,
                )
            except Exception as exc:
                fatal_failure = _wrap_outer_stage_exception(
                    exc,
                    stage="outer_iteration",
                    substage="build_current_geometry_from_radius[next]",
                    extra_meta={
                        "step_id": int(step_id),
                        "outer_iter_index": int(outer_iter_index),
                    },
                )
                break
            try:
                transfer_in = build_transfer_state_for_next_outer_iter(
                    cfg=cfg,
                    state_converged_k=state_converged_k,
                    mesh_k=mesh_current,
                    geometry_k=geometry_current,
                    mesh_k1=mesh_k1,
                    geometry_k1=geometry_k1,
                    outer_iter_index=outer_iter_index,
                    models=models_h,
                )
            except Exception as exc:
                fatal_failure = _wrap_outer_stage_exception(
                    exc,
                    stage="outer_iteration",
                    substage="build_transfer_state_for_next_outer_iter",
                    extra_meta={
                        "step_id": int(step_id),
                        "outer_iter_index": int(outer_iter_index),
                    },
                )
                break
            a_iter = next_a_iter
            dot_a_iter = next_dot_a_iter
            entry_source = InnerEntrySource.TRANSFER_FROM_PREVIOUS_OUTER

        last_outer_iter = outer_iterations[-1] if outer_iterations else None
        if fatal_failure is not None:
            decision = decide_step_acceptance(
                cfg=cfg,
                dt_current=dt_attempt,
                retries_used=retries_used,
                inner_result=last_outer_iter.inner if last_outer_iter is not None else None,
                outer_convergence=last_outer_iter.convergence if last_outer_iter is not None else None,
                outer_iter_count=len(outer_iterations),
                outer_iter_limit=cfg_ts.outer_max_iter,
                accepted_state_candidate=None,
                accepted_geometry_candidate=None,
                fatal_failure=fatal_failure,
            )
        elif last_outer_iter is None:
            raise ValueError("advance_one_step produced no outer iteration result")
        elif not last_outer_iter.inner.converged:
            decision = decide_step_acceptance(
                cfg=cfg,
                dt_current=dt_attempt,
                retries_used=retries_used,
                inner_result=last_outer_iter.inner,
                outer_convergence=None,
                outer_iter_count=len(outer_iterations),
                outer_iter_limit=cfg_ts.outer_max_iter,
                accepted_state_candidate=None,
                accepted_geometry_candidate=None,
            )
        elif last_outer_iter.convergence is not None and last_outer_iter.convergence.converged:
            decision = decide_step_acceptance(
                cfg=cfg,
                dt_current=dt_attempt,
                retries_used=retries_used,
                inner_result=last_outer_iter.inner,
                outer_convergence=last_outer_iter.convergence,
                outer_iter_count=len(outer_iterations),
                outer_iter_limit=cfg_ts.outer_max_iter,
                accepted_state_candidate=accepted_state_candidate,
                accepted_geometry_candidate=accepted_geometry_candidate,
            )
        else:
            decision = decide_step_acceptance(
                cfg=cfg,
                dt_current=dt_attempt,
                retries_used=retries_used,
                inner_result=last_outer_iter.inner,
                outer_convergence=last_outer_iter.convergence,
                outer_iter_count=len(outer_iterations),
                outer_iter_limit=cfg_ts.outer_max_iter,
                accepted_state_candidate=None,
                accepted_geometry_candidate=None,
            )
        _emit_diag(
            diag_sink,
            event="step_acceptance_decision",
            payload={
                "step_id": int(step_id),
                "accepted": bool(decision.accepted),
                "action": getattr(decision.action, "value", str(decision.action)),
                "dt_current": float(dt_attempt),
                "dt_next": getattr(decision, "dt_next", None),
                "retries_used": int(retries_used),
                "reject_reason": getattr(decision, "reject_reason", None),
                "failure_class": _failure_class_value(
                    getattr(getattr(decision, "failure", None), "failure_class", None)
                ),
            },
        )

        step_result = StepAdvanceResult(
            step_id=int(step_id),
            t_old=t_old_f,
            t_new_target=t_old_f + dt_attempt,
            dt_attempt=dt_attempt,
            accepted=bool(decision.accepted),
            acceptance=decision,
            predictor=predictor,
            outer_iterations=outer_iterations,
            accepted_solution_vec=accepted_state_candidate if decision.accepted else None,
            accepted_geometry=accepted_geometry_candidate if decision.accepted else None,
            failure=FailureInfo() if decision.accepted else decision.failure,
            diagnostics={
                "dt_attempt": dt_attempt,
                "retries_used": retries_used,
                "outer_iter_count": len(outer_iterations),
                "inner_converged_any": any(it.inner.converged for it in outer_iterations),
                "entry_source_first_outer": (
                    getattr(outer_iterations[0].entry_source, "value", outer_iterations[0].entry_source)
                    if outer_iterations
                    else None
                ),
                "outer_converged": bool(
                    last_outer_iter is not None
                    and last_outer_iter.convergence is not None
                    and last_outer_iter.convergence.converged
                ),
                "reject_reason": decision.reject_reason,
            },
        )
        step_result.assert_consistent()

        if decision.action is StepAction.RETRY_REDUCED_DT:
            retries_used += 1
            dt_attempt = _validate_positive_finite("decision.dt_next", float(decision.dt_next))
            _emit_diag(diag_sink, event="step_retry", payload=_step_result_payload(step_result))
            continue

        _emit_diag(diag_sink, event="step_result", payload=_step_result_payload(step_result))
        return step_result


__all__ = [
    "TimestepperConfigView",
    "advance_one_step",
    "build_current_geometry_from_radius",
    "build_old_state_on_current_geometry",
    "build_transfer_state_for_next_outer_iter",
    "build_state_guess_for_outer_iter",
    "build_timestepper_config_view",
    "run_single_outer_iteration",
]
