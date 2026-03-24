from __future__ import annotations

"""Formal Phase 7 case driver for the evaporation-only mainline."""

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import importlib.util
import math
import sys
import traceback

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from core.config_loader import load_and_validate_config, load_raw_config
from core.grid import build_grid_and_metrics, build_initial_grid
from core.layout import build_layout
from core.preprocess import normalize_config
from core.state_pack import unpack_array_to_state
from core.types import GeometryState
from driver.runtime_evp import (
    RuntimeEVP,
    bind_cfg_output_layout,
    bind_cfg_runtime_public,
    build_run_summary_payload,
    build_runtime_evp,
    commit_accepted_step,
    make_diag_sink,
    note_attempt,
    record_run_end,
    runtime_event,
    runtime_exception,
    runtime_log,
    unbind_cfg_runtime_bridges,
)
from solvers.nonlinear_context import NonlinearModelHandles
from solvers.timestepper import advance_one_step


def _load_module(module_name: str, path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_WRITERS = _load_module("paper_v1_io_writers", SRC_DIR / "io" / "writers.py")
_FAILURE_ARTIFACTS = _load_module("paper_v1_io_failure_artifacts", SRC_DIR / "io" / "failure_artifacts.py")

append_interface_diag_row = _WRITERS.append_interface_diag_row
append_scalars_row = _WRITERS.append_scalars_row
append_step_diag_row = _WRITERS.append_step_diag_row
write_config_echo = _WRITERS.write_config_echo
write_mapping_json = _WRITERS.write_mapping_json
write_metadata_echo = _WRITERS.write_metadata_echo
write_run_summary = _WRITERS.write_run_summary
write_step_snapshot = _WRITERS.write_step_snapshot

write_attempt_failure = _FAILURE_ARTIFACTS.write_attempt_failure


def _parse_cli(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description="Run one evaporation-only paper_v1 case")
    parser.add_argument("config", help="Path to the case YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Build and initialize everything without stepping")
    parser.add_argument("--run-id", default=None, help="Optional explicit run directory token")
    return parser.parse_args(argv)


def _read_backend_hint(config_path: Path) -> dict[str, Any]:
    raw_cfg = load_raw_config(config_path)
    solver_cfg = raw_cfg.get("solver_inner_petsc", {})
    if not isinstance(solver_cfg, Mapping):
        solver_cfg = {}

    enabled_raw = solver_cfg.get("enabled", True)
    enabled = bool(enabled_raw) if not isinstance(enabled_raw, str) else enabled_raw.strip().lower() not in {"0", "false", "no", "off"}
    backend_raw = solver_cfg.get("backend", "petsc" if enabled else "scipy")
    backend_token = str(backend_raw).strip().lower()
    parallel_mode = str(solver_cfg.get("parallel_mode", "")).strip().lower()

    return {
        "enabled": enabled,
        "backend_token": backend_token,
        "parallel_mode_hint": parallel_mode,
    }


def _bootstrap_backend_if_needed(backend_hint: Mapping[str, Any]) -> dict[str, Any]:
    enabled = bool(backend_hint.get("enabled", True))
    backend_token = str(backend_hint.get("backend_token", "petsc")).strip().lower()
    parallel_mode_hint = str(backend_hint.get("parallel_mode_hint", "")).strip().lower()

    if not enabled or backend_token in {"scipy", "numpy"}:
        return {
            "backend": "scipy",
            "petsc_enabled": False,
            "rank": 0,
            "size": 1,
            "comm": None,
            "parallel_mode": "serial",
        }

    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc, get_comm_world, get_rank_size

    bootstrap_mpi_before_petsc()
    comm = get_comm_world()
    rank, size = get_rank_size(comm)

    if backend_token == "petsc_mpi" or size > 1 or parallel_mode_hint in {"mpi", "distributed", "parallel"}:
        backend = "petsc_mpi" if int(size) > 1 else "petsc_serial"
    elif backend_token in {"petsc_serial", "petsc"}:
        backend = "petsc_serial"
    else:
        backend = "petsc_mpi" if int(size) > 1 else "petsc_serial"

    return {
        "backend": backend,
        "petsc_enabled": True,
        "rank": int(rank),
        "size": int(size),
        "comm": comm,
        "parallel_mode": "mpi" if int(size) > 1 else "serial",
    }


def _load_and_preprocess_cfg(config_path: Path):
    raw_cfg = load_and_validate_config(config_path)
    return normalize_config(raw_cfg, source_path=config_path)


def _gas_molecular_weights(cfg) -> np.ndarray:
    import cantera as ct

    gas = ct.Solution(str(cfg.paths.gas_mechanism_path), str(getattr(cfg, "gas_phase_name", "gas")))
    if tuple(gas.species_names) != tuple(cfg.species_maps.gas_full_names):
        raise ValueError("gas mechanism species order must match cfg.species_maps.gas_full_names")
    return np.asarray(gas.molecular_weights, dtype=np.float64) / 1000.0


def _liquid_molecular_weights(liquid_db, liquid_species_full: tuple[str, ...]) -> np.ndarray:
    from properties.liquid_db import get_species_record

    return np.asarray(
        [float(get_species_record(liquid_db, name).molecular_weight) for name in liquid_species_full],
        dtype=np.float64,
    )


def _load_petsc_provider():
    from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc

    bootstrap_mpi_before_petsc()
    from petsc4py import PETSc

    return PETSc


def _build_models(cfg, *, parallel_handles: dict[str, object] | None = None):
    from properties.aggregator import build_bulk_props
    from properties.equilibrium import build_interface_equilibrium_model
    from properties.gas import build_gas_thermo_model
    from properties.liquid import build_liquid_thermo_model
    from properties.liquid_db import build_liquid_database

    liquid_db = build_liquid_database(cfg)
    liquid_model = build_liquid_thermo_model(
        liquid_db=liquid_db,
        liquid_species_full=tuple(cfg.species_maps.liq_full_names),
        molecular_weights=_liquid_molecular_weights(liquid_db, tuple(cfg.species_maps.liq_full_names)),
        p_env=float(cfg.pressure),
    )
    gas_model = build_gas_thermo_model(
        mechanism_path=cfg.paths.gas_mechanism_path,
        gas_species_full=tuple(cfg.species_maps.gas_full_names),
        molecular_weights=_gas_molecular_weights(cfg),
        closure_species=str(cfg.species_maps.gas_closure_name),
    )
    equilibrium_model = build_interface_equilibrium_model(
        liquid_thermo=liquid_model,
        gas_thermo=gas_model,
        liquid_to_gas_species_map=dict(cfg.species.liquid_to_gas_species_map),
        gas_closure_species=str(cfg.species.gas_closure_species),
        reference_pressure=float(cfg.pressure),
    )

    models = NonlinearModelHandles(
        gas_model=gas_model,
        liquid_model=liquid_model,
        equilibrium_model=equilibrium_model,
        property_aggregator=build_bulk_props,
        liquid_db=liquid_db,
        extra={},
    )

    if isinstance(parallel_handles, dict):
        dm_mgr = parallel_handles.get("dm_manager")
        if dm_mgr is not None:
            models.extra["residual_ownership"] = getattr(dm_mgr, "residual_ownership", None)
            models.extra["jacobian_ownership"] = getattr(dm_mgr, "jacobian_ownership", None)
            models.extra = {k: v for k, v in models.extra.items() if v is not None}
    return models


def _build_case_objects(
    cfg,
    *,
    backend_info: Mapping[str, Any],
):
    from physics.initial import build_initial_state_bundle
    from properties.aggregator import build_bulk_props

    geometry0 = GeometryState(
        t=float(cfg.time_stepper.t0),
        dt=float(cfg.time_stepper.dt_start),
        a=float(cfg.a0),
        dot_a=0.0,
        r_end=float(cfg.r_end),
        step_index=0,
        outer_iter_index=0,
    )
    grid0 = build_initial_grid(cfg, geometry0)
    layout0 = build_layout(cfg, grid0)

    models = _build_models(cfg, parallel_handles=None)
    initial_bundle = build_initial_state_bundle(
        run_cfg=cfg,
        mesh=grid0,
        gas_props=models.gas_model,
        liquid_props=models.liquid_model,
        layout=layout0,
    )
    state0 = initial_bundle.state0
    props0 = build_bulk_props(
        state=state0,
        grid=grid0,
        liquid_thermo=models.liquid_model,
        gas_thermo=models.gas_model,
        gas_pressure=float(cfg.pressure),
    )

    parallel_handles: dict[str, object] | None = None
    single_rank_only_boundary_active = False

    if bool(backend_info.get("petsc_enabled", False)):
        from parallel.dm_manager import DMManagerError, build_dm_manager, build_parallel_handles
        from parallel.fieldsplit_is import build_fieldsplit_handles
        from parallel.local_state import build_local_state_hooks

        PETSc = _load_petsc_provider()
        try:
            dm_mgr = build_dm_manager(
                layout=layout0,
                PETSc=PETSc,
                comm=backend_info.get("comm"),
                interface_owner_rank=0,
            )
        except DMManagerError as exc:
            if int(backend_info.get("size", 1)) > 1:
                single_rank_only_boundary_active = True
            else:
                raise
        else:
            parallel_handles = build_parallel_handles(dm_mgr)
            parallel_handles.update(
                build_fieldsplit_handles(
                    layout=layout0,
                    PETSc=PETSc,
                    ownership_range=parallel_handles.get("ownership_range"),
                )
            )
            models.extra["residual_ownership"] = dm_mgr.residual_ownership
            models.extra["jacobian_ownership"] = dm_mgr.jacobian_ownership
            models.extra.update(build_local_state_hooks())

    return {
        "layout": layout0,
        "grid0": grid0,
        "state0": state0,
        "props0": props0,
        "models": models,
        "parallel_handles": parallel_handles,
        "single_rank_only_boundary_active": single_rank_only_boundary_active,
        "geometry0": geometry0,
        "dot_a0": float(initial_bundle.dot_a0),
    }


def _initialize_runtime(
    cfg,
    *,
    build_pack: Mapping[str, Any],
    backend_info: Mapping[str, Any],
    dry_run: bool,
    run_id: str | None,
) -> RuntimeEVP:
    runtime = build_runtime_evp(
        cfg,
        layout=build_pack["layout"],
        grid0=build_pack["grid0"],
        state0=build_pack["state0"],
        props0=build_pack["props0"],
        t0=float(cfg.time_stepper.t0),
        backend=str(backend_info["backend"]),
        petsc_enabled=bool(backend_info["petsc_enabled"]),
        parallel_mode=str(backend_info["parallel_mode"]),
        rank=int(backend_info["rank"]),
        size=int(backend_info["size"]),
        parallel_handles=build_pack.get("parallel_handles"),
        single_rank_only_boundary_active=bool(build_pack.get("single_rank_only_boundary_active", False)),
        run_id=run_id,
    )
    bind_cfg_output_layout(cfg, runtime)
    runtime_log(
        runtime,
        "INFO",
        "run_start",
        config_path=getattr(getattr(cfg, "paths", None), "config_path", None),
        dry_run=bool(dry_run),
        run_id=runtime.run_id,
    )
    runtime_event(
        runtime,
        "run_start",
        {
            "config_path": getattr(getattr(cfg, "paths", None), "config_path", None),
            "dry_run": bool(dry_run),
            "run_id": runtime.run_id,
        },
    )
    runtime_log(
        runtime,
        "INFO",
        "config_ready",
        case_id=runtime.output_layout.case_id,
        backend=runtime.backend,
        petsc_enabled=runtime.petsc_enabled,
        parallel_mode=runtime.parallel_mode,
        rank=runtime.rank,
        size=runtime.size,
    )
    runtime_event(
        runtime,
        "config_ready",
        {
            "case_id": runtime.output_layout.case_id,
            "backend": runtime.backend,
            "petsc_enabled": runtime.petsc_enabled,
            "parallel_mode": runtime.parallel_mode,
            "rank": runtime.rank,
            "size": runtime.size,
        },
    )
    runtime_log(
        runtime,
        "INFO",
        "runtime_initialized",
        run_root=runtime.output_layout.root,
        run_log=runtime.output_layout.run_log,
        run_summary_json=runtime.output_layout.run_summary_json,
        single_rank_only_boundary_active=runtime.single_rank_only_boundary_active,
    )
    runtime_event(
        runtime,
        "runtime_initialized",
        {
            "run_root": runtime.output_layout.root,
            "run_log": runtime.output_layout.run_log,
            "run_summary_json": runtime.output_layout.run_summary_json,
            "single_rank_only_boundary_active": runtime.single_rank_only_boundary_active,
        },
    )
    write_config_echo(cfg, runtime.output_layout, is_root=runtime.is_root)
    write_metadata_echo(runtime, runtime.output_layout, is_root=runtime.is_root)
    bind_cfg_runtime_public(cfg, runtime)
    write_mapping_json(runtime.output_layout, grid=runtime.grid, layout=runtime.layout, cfg=cfg, is_root=runtime.is_root)
    runtime.mapping_written = True
    runtime_log(
        runtime,
        "INFO",
        "mapping_written",
        mapping_json=runtime.output_layout.mapping_json,
        metadata_json=runtime.output_layout.metadata_json,
        config_echo_yaml=runtime.output_layout.config_echo_yaml,
    )
    runtime_event(
        runtime,
        "mapping_written",
        {
            "mapping_json": runtime.output_layout.mapping_json,
            "metadata_json": runtime.output_layout.metadata_json,
            "config_echo_yaml": runtime.output_layout.config_echo_yaml,
        },
    )
    return runtime


def _initial_dt_from_cfg(cfg) -> float:
    return float(getattr(cfg.time_stepper, "dt_start"))


def _remaining_time(cfg, runtime: RuntimeEVP) -> float:
    return max(0.0, float(cfg.time_stepper.t_end) - float(runtime.t))


def _should_write_snapshot(cfg, *, step_id: int) -> bool:
    write_spatial = bool(getattr(cfg.output, "write_spatial_fields", False) or getattr(cfg.output, "write_spatial_species", False))
    cadence = int(getattr(cfg.diagnostics, "output_every_n_steps", 1))
    return write_spatial and cadence > 0 and int(step_id) % cadence == 0


def _state_array_from_handle(state_handle: object) -> np.ndarray:
    if hasattr(state_handle, "getArray"):
        return np.array(state_handle.getArray(readonly=True), dtype=np.float64, copy=True)
    return np.asarray(state_handle, dtype=np.float64).copy()


def _recover_state_from_handle(
    *,
    cfg,
    state_handle: object,
    layout,
    parallel_handles: dict[str, object] | None,
    t_new: float,
    step_id: int,
):
    if state_handle is None:
        raise ValueError("accepted state handle is required")

    if isinstance(parallel_handles, dict) and parallel_handles.get("dm_manager") is not None and hasattr(state_handle, "getArray"):
        from parallel.local_state import recover_state_from_distributed_vec

        return recover_state_from_distributed_vec(
            X_global=state_handle,
            layout=layout,
            species_maps=cfg.species_maps,
            parallel_handles=parallel_handles,
            time=float(t_new),
            state_id=f"step_{int(step_id):06d}",
        )

    return unpack_array_to_state(
        _state_array_from_handle(state_handle),
        layout,
        cfg.species_maps,
        time=float(t_new),
        state_id=f"step_{int(step_id):06d}",
    )


def _step_result_diag(*, step_id: int, t_old: float, dt_used: float, t_new: float, accepted_geometry, state_new, step_result):
    last_outer = step_result.outer_iterations[-1] if step_result.outer_iterations else None
    inner = getattr(last_outer, "inner", None)
    stats = getattr(inner, "stats", None)
    residual_norms = getattr(stats, "residual_norms", None)
    inner_diag = getattr(inner, "diagnostics", {}) if inner is not None else {}
    if not isinstance(inner_diag, Mapping):
        inner_diag = {}
    interface_diag = inner_diag.get("interface_diag", {})
    if not isinstance(interface_diag, Mapping):
        interface_diag = {}

    return SimpleNamespace(
        step=int(step_id),
        step_id=int(step_id),
        t_old=float(t_old),
        t_new=float(t_new),
        dt=float(dt_used),
        linear_converged=getattr(inner, "converged", ""),
        linear_method=getattr(stats, "backend", ""),
        linear_n_iter=getattr(stats, "linear_iter_count", ""),
        linear_residual_norm=getattr(residual_norms, "l2", ""),
        linear_rel_residual=getattr(residual_norms, "scaled_l2", ""),
        nonlinear_converged=getattr(inner, "converged", ""),
        nonlinear_method=getattr(stats, "backend", ""),
        nonlinear_n_iter=getattr(stats, "inner_iter_count", ""),
        nonlinear_residual_norm=getattr(residual_norms, "l2", ""),
        nonlinear_residual_inf=getattr(residual_norms, "linf", ""),
        Ts=float(state_new.interface.Ts),
        Rd=float(getattr(accepted_geometry, "a")),
        mpp=float(state_new.interface.mpp),
        Tg_min=float(np.min(state_new.Tg)),
        Tg_max=float(np.max(state_new.Tg)),
        energy_balance_if=inner_diag.get("energy_balance_if", ""),
        mass_balance_rd=inner_diag.get("mass_balance_rd", ""),
        extra={"interface_diag": dict(interface_diag)},
    )


def _extract_attempt_final_state(
    *,
    cfg,
    step_result,
    parallel_handles: dict[str, object] | None,
):
    if not step_result.outer_iterations:
        return None
    last_outer = step_result.outer_iterations[-1]
    inner = getattr(last_outer, "inner", None)
    state_vec = getattr(inner, "state_vec", None)
    if state_vec is None:
        return None
    geometry_current = None
    diagnostics = getattr(last_outer, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        geometry_current = diagnostics.get("geometry_current")
    if geometry_current is None:
        return None
    mesh_current, _ = build_grid_and_metrics(cfg, geometry_current)
    layout_current = build_layout(cfg, mesh_current)
    try:
        return _recover_state_from_handle(
            cfg=cfg,
            state_handle=state_vec,
            layout=layout_current,
            parallel_handles=parallel_handles,
            t_new=float(getattr(geometry_current, "t")),
            step_id=int(step_result.step_id),
        )
    except Exception:
        return None


def _retry_id_from_step_result(step_result: object) -> int:
    diagnostics = getattr(step_result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        return 0
    try:
        return max(0, int(diagnostics.get("retries_used", 0)))
    except Exception:
        return 0


def _current_interface_log_fields(runtime: RuntimeEVP, *, geometry: object | None = None) -> dict[str, Any]:
    state = getattr(runtime, "state", None)
    interface = getattr(state, "interface", None)
    return {
        "Rd": getattr(geometry, "a", None) if geometry is not None else None,
        "Ts": getattr(interface, "Ts", None) if interface is not None else None,
        "mpp": getattr(interface, "mpp", None) if interface is not None else None,
    }


def _step_result_log_fields(step_result: object) -> dict[str, Any]:
    diagnostics = getattr(step_result, "diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    acceptance = getattr(step_result, "acceptance", None)
    failure = getattr(step_result, "failure", None)
    outer_iterations = getattr(step_result, "outer_iterations", None)
    fields: dict[str, Any] = {
        "retries_used": _retry_id_from_step_result(step_result),
        "outer_iterations": len(outer_iterations or ()),
        "outer_iter_count": diagnostics.get("outer_iter_count"),
        "outer_converged": diagnostics.get("outer_converged"),
        "entry_source_first_outer": diagnostics.get("entry_source_first_outer"),
        "failure_class": getattr(getattr(failure, "failure_class", None), "value", None),
        "reject_reason": getattr(acceptance, "reject_reason", None),
        "nonlinear_n_iter": None,
        "nonlinear_residual_inf": None,
        "snes_reason": None,
        "ksp_reason": None,
    }
    if not outer_iterations:
        return fields
    last_outer = outer_iterations[-1]
    inner = getattr(last_outer, "inner", None)
    stats = getattr(inner, "stats", None)
    residual_norms = getattr(stats, "residual_norms", None)
    fields["nonlinear_n_iter"] = getattr(stats, "inner_iter_count", None)
    fields["nonlinear_residual_inf"] = getattr(residual_norms, "linf", None)
    fields["snes_reason"] = getattr(stats, "snes_reason", None)
    fields["ksp_reason"] = getattr(stats, "ksp_reason", None)
    return fields


def _write_step_failure(
    cfg,
    *,
    runtime: RuntimeEVP,
    accepted_geometry_old,
    dt_try: float,
    step_result,
) -> object | None:
    reject_reason = str(getattr(step_result.acceptance, "reject_reason", "") or getattr(step_result.failure, "message", "") or "step_rejected")
    failure_class = getattr(getattr(step_result, "failure", None), "failure_class", None)
    failure_class_name = getattr(failure_class, "value", str(failure_class or "STEP_REJECTED"))
    ctx_meta = {
        "step_result_diagnostics": getattr(step_result, "diagnostics", {}),
        "failure_meta": getattr(getattr(step_result, "failure", None), "meta", {}),
    }
    last_outer = step_result.outer_iterations[-1] if step_result.outer_iterations else None
    inner = getattr(last_outer, "inner", None)
    u_attempt_final = getattr(inner, "state_vec", None) if inner is not None else None
    retry_id = _retry_id_from_step_result(step_result)
    state_attempt_final = _extract_attempt_final_state(
        cfg=cfg,
        step_result=step_result,
        parallel_handles=runtime.parallel_handles if isinstance(runtime.parallel_handles, dict) else None,
    )

    runtime_log(
        runtime,
        "WARNING",
        "failure_artifact_begin",
        step_id=int(runtime.next_step_id),
        retry_id=retry_id,
    )
    flayout = write_attempt_failure(
        cfg=cfg,
        output_layout=runtime.output_layout,
        step_id=int(runtime.next_step_id),
        retry_id=retry_id,
        t_old=float(runtime.t),
        dt_try=float(dt_try),
        failure_class=failure_class_name,
        reject_reason=reject_reason,
        state_last_good=runtime.state,
        state_attempt_input=runtime.state,
        state_attempt_final=state_attempt_final,
        u_last_good=None,
        u_attempt_input=None,
        u_attempt_final=u_attempt_final,
        diag=_step_result_diag(
            step_id=int(runtime.next_step_id),
            t_old=float(runtime.t),
            dt_used=float(dt_try),
            t_new=float(runtime.t) + float(dt_try),
            accepted_geometry=accepted_geometry_old,
            state_new=runtime.state,
            step_result=step_result,
        ) if runtime.state is not None else None,
        runtime=runtime,
        grid=runtime.grid,
        layout=runtime.layout,
        props_last_good=runtime.props,
        props_attempt_input=runtime.props,
        props_attempt_final=None,
        ctx_meta=ctx_meta,
    )
    if flayout is not None:
        runtime_log(
            runtime,
            "WARNING",
            "failure_artifact_written",
            step_id=int(runtime.next_step_id),
            retry_id=retry_id,
            failure_dir=getattr(flayout, "root", None),
            report_json=getattr(flayout, "report_json", None),
            state_last_good_npz=getattr(flayout, "state_last_good_npz", None),
            state_attempt_input_npz=getattr(flayout, "state_attempt_input_npz", None),
            state_attempt_final_npz=getattr(flayout, "state_attempt_final_npz", None),
        )
    runtime_log(
        runtime,
        "WARNING",
        "step_rejected",
        step_id=int(runtime.next_step_id),
        t_old=float(runtime.t),
        dt_try=float(dt_try),
        failure_dir=getattr(flayout, "root", None) if flayout is not None else None,
        **_step_result_log_fields(step_result),
    )
    return flayout


def _run_step_loop(
    cfg,
    *,
    runtime: RuntimeEVP,
    models,
    accepted_geometry_old,
    dot_a_old: float,
) -> int:
    from properties.aggregator import build_bulk_props

    dt_try = min(_initial_dt_from_cfg(cfg), _remaining_time(cfg, runtime))
    geometry_old = accepted_geometry_old
    dot_a_prev = float(dot_a_old)
    diag_sink = make_diag_sink(runtime)

    while float(runtime.t) < float(cfg.time_stepper.t_end) - 1.0e-15:
        remaining = _remaining_time(cfg, runtime)
        if remaining <= 0.0:
            break
        dt_current = min(float(dt_try), remaining)
        step_id = int(runtime.next_step_id)
        note_attempt(runtime, is_retry=False)
        runtime_log(
            runtime,
            "INFO",
            "step_begin",
            step_id=step_id,
            t_old=float(runtime.t),
            dt_try=float(dt_current),
            dot_a_old=float(dot_a_prev),
            **_current_interface_log_fields(runtime, geometry=geometry_old),
        )
        runtime_event(
            runtime,
            "step_begin",
            {
                "step_id": step_id,
                "t_old": float(runtime.t),
                "dt_try": float(dt_current),
                "dot_a_old": float(dot_a_prev),
                **_current_interface_log_fields(runtime, geometry=geometry_old),
            },
        )

        step_result = advance_one_step(
            cfg=cfg,
            step_id=step_id,
            t_old=float(runtime.t),
            dt=float(dt_current),
            accepted_state=runtime.state,
            accepted_geometry=geometry_old,
            accepted_mesh=runtime.grid,
            accepted_layout=runtime.layout,
            accepted_props=runtime.props,
            dot_a_old=float(dot_a_prev),
            models=models,
            parallel_handles=runtime.parallel_handles if isinstance(runtime.parallel_handles, dict) else None,
            diag_sink=diag_sink,
        )
        runtime.last_step_result = step_result
        runtime.counters.n_retries_total += _retry_id_from_step_result(step_result)

        if not bool(getattr(step_result, "accepted", False)):
            runtime.counters.n_failures_total += 1
            runtime.last_message = str(
                getattr(getattr(step_result, "acceptance", None), "reject_reason", "")
                or getattr(getattr(step_result, "failure", None), "message", "")
                or "step rejected"
            )
            _write_step_failure(
                cfg,
                runtime=runtime,
                accepted_geometry_old=geometry_old,
                dt_try=float(dt_current),
                step_result=step_result,
            )
            return 1

        accepted_geometry = getattr(step_result, "accepted_geometry", None)
        accepted_state_vec = getattr(step_result, "accepted_solution_vec", None)
        if accepted_state_vec is None:
            accepted_state_vec = getattr(step_result, "accepted_state_vec", None)
        if accepted_geometry is None or accepted_state_vec is None:
            raise ValueError("accepted step must provide accepted_geometry and accepted_solution_vec")

        grid_new, _ = build_grid_and_metrics(cfg, accepted_geometry)
        layout_new = build_layout(cfg, grid_new)
        t_new = float(getattr(accepted_geometry, "t"))
        state_new = _recover_state_from_handle(
            cfg=cfg,
            state_handle=accepted_state_vec,
            layout=layout_new,
            parallel_handles=runtime.parallel_handles if isinstance(runtime.parallel_handles, dict) else None,
            t_new=t_new,
            step_id=step_id,
        )
        props_new = build_bulk_props(
            state=state_new,
            grid=grid_new,
            liquid_thermo=models.liquid_model,
            gas_thermo=models.gas_model,
            gas_pressure=float(cfg.pressure),
        )
        diag_row = _step_result_diag(
            step_id=step_id,
            t_old=float(runtime.t),
            dt_used=float(dt_current),
            t_new=t_new,
            accepted_geometry=accepted_geometry,
            state_new=state_new,
            step_result=step_result,
        )

        if bool(getattr(cfg.diagnostics, "write_step_diag", False)):
            append_step_diag_row(runtime.output_layout, step_id=step_id, diag=diag_row, is_root=runtime.is_root)
        if bool(getattr(cfg.output, "write_time_series_scalars", False)):
            append_scalars_row(
                runtime.output_layout,
                step_id=step_id,
                t=t_new,
                state=state_new,
                diag=diag_row,
                cfg=cfg,
                is_root=runtime.is_root,
            )
        if bool(getattr(cfg.diagnostics, "write_interface_diag", False)):
            append_interface_diag_row(runtime.output_layout, step_id=step_id, diag=diag_row, is_root=runtime.is_root)
        snapshot_path = None
        if _should_write_snapshot(cfg, step_id=step_id):
            snapshot_path = write_step_snapshot(
                runtime.output_layout,
                step_id=step_id,
                t=t_new,
                grid=grid_new,
                state=state_new,
                props=props_new,
                layout=layout_new,
                diag=diag_row,
                is_root=runtime.is_root,
            )

        commit_accepted_step(
            runtime,
            step_result=step_result,
            t_new=t_new,
            dt_used=float(dt_current),
            layout_new=layout_new,
            grid_new=grid_new,
            state_new=state_new,
            props_new=props_new,
            last_inner_result=(step_result.outer_iterations[-1].inner if step_result.outer_iterations else None),
        )
        runtime_log(
            runtime,
            "INFO",
            "step_accepted",
            step_id=step_id,
            t_new=t_new,
            dt_used=float(dt_current),
            snapshot_path=snapshot_path,
            Rd=float(getattr(accepted_geometry, "a", math.nan)),
            Ts=float(state_new.interface.Ts),
            mpp=float(state_new.interface.mpp),
            **_step_result_log_fields(step_result),
        )

        geometry_old = accepted_geometry
        dot_a_prev = float(getattr(accepted_geometry, "dot_a"))
        dt_next = getattr(step_result.acceptance, "dt_next", None)
        dt_try = float(dt_next) if dt_next is not None else float(dt_current)

    return 0


def _handle_driver_level_failure(
    cfg,
    *,
    runtime: RuntimeEVP,
    exc: Exception,
    step_id: int,
    dt_try: float | None = None,
    message: str = "",
) -> object | None:
    reject_reason = message or str(exc) or exc.__class__.__name__
    runtime_log(
        runtime,
        "ERROR",
        "failure_artifact_begin",
        step_id=int(step_id),
        retry_id=0,
    )
    flayout = write_attempt_failure(
        cfg=cfg,
        output_layout=runtime.output_layout,
        step_id=int(step_id),
        retry_id=0,
        t_old=float(runtime.t),
        dt_try=float(dt_try if dt_try is not None else _initial_dt_from_cfg(cfg)),
        failure_class=exc.__class__.__name__,
        reject_reason=reject_reason,
        state_last_good=runtime.state,
        state_attempt_input=runtime.state,
        state_attempt_final=None,
        runtime=runtime,
        grid=runtime.grid,
        layout=runtime.layout,
        props_last_good=runtime.props,
        props_attempt_input=runtime.props,
        ctx_meta={
            "message": reject_reason,
            "traceback": traceback.format_exc(),
        },
    )
    if flayout is not None:
        runtime_log(
            runtime,
            "ERROR",
            "failure_artifact_written",
            step_id=int(step_id),
            retry_id=0,
            failure_dir=getattr(flayout, "root", None),
            report_json=getattr(flayout, "report_json", None),
    )
    return flayout


def _best_effort_runtime_failure_side_effects(
    *,
    runtime: RuntimeEVP,
    cfg,
    exc: BaseException,
    config_path: Path,
) -> None:
    side_errors: list[tuple[str, BaseException]] = []
    flayout = None
    runtime.counters.n_failures_total += 1

    try:
        flayout = _handle_driver_level_failure(
            cfg,
            runtime=runtime,
            exc=exc if isinstance(exc, Exception) else Exception(str(exc)),
            step_id=int(runtime.next_step_id),
            dt_try=_initial_dt_from_cfg(cfg) if cfg is not None else None,
            message=str(exc),
        )
    except BaseException as aux:
        side_errors.append(("handle_driver_level_failure", aux))

    try:
        runtime_exception(
            runtime,
            "run_case_top_level",
            exc,
            step_id=int(runtime.next_step_id),
            stage="driver",
            extra={
                "config_path": config_path,
                "failure_dir": getattr(flayout, "root", None) if flayout is not None else None,
                "run_summary_json": runtime.output_layout.run_summary_json,
            },
        )
    except BaseException as aux:
        side_errors.append(("runtime_exception", aux))

    try:
        runtime_log(
            runtime,
            "ERROR",
            "driver_exception",
            config_path=config_path,
            failure_dir=getattr(flayout, "root", None) if flayout is not None else None,
            run_summary_json=runtime.output_layout.run_summary_json,
        )
    except BaseException as aux:
        side_errors.append(("runtime_log", aux))

    try:
        record_run_end(runtime, exit_code=3, ended_by="driver_exception", message=str(exc))
        write_run_summary(runtime.output_layout, summary=build_run_summary_payload(runtime), is_root=runtime.is_root)
    except BaseException as aux:
        side_errors.append(("write_run_summary", aux))

    for where, aux in side_errors:
        print(
            f"[WARNING][best-effort-side-effect-failed] {where}: {aux.__class__.__name__}: {aux}",
            file=sys.stderr,
        )


def run_case(config_path: str | Path, *, dry_run: bool = False, run_id: str | None = None) -> int:
    runtime: RuntimeEVP | None = None
    cfg = None
    config_path = Path(config_path).expanduser().resolve()
    backend_hint = _read_backend_hint(config_path)
    backend_info = _bootstrap_backend_if_needed(backend_hint)
    cfg = _load_and_preprocess_cfg(config_path)
    build_pack = _build_case_objects(cfg, backend_info=backend_info)
    runtime = _initialize_runtime(
        cfg,
        build_pack=build_pack,
        backend_info=backend_info,
        dry_run=bool(dry_run),
        run_id=run_id,
    )

    try:
        if dry_run:
            runtime_log(runtime, "INFO", "dry_run_begin", run_summary_json=runtime.output_layout.run_summary_json)
            record_run_end(runtime, exit_code=0, ended_by="dry_run", message="dry run completed")
            runtime_log(
                runtime,
                "INFO",
                "dry_run_completed",
                run_summary_json=runtime.output_layout.run_summary_json,
                metadata_json=runtime.output_layout.metadata_json,
                mapping_json=runtime.output_layout.mapping_json,
            )
            write_run_summary(runtime.output_layout, summary=build_run_summary_payload(runtime), is_root=runtime.is_root)
            return 0

        if runtime.single_rank_only_boundary_active and runtime.size > 1:
            runtime.counters.n_failures_total += 1
            runtime_log(
                runtime,
                "WARNING",
                "expected_boundary_triggered",
                mpi_size=runtime.size,
                parallel_mode=runtime.parallel_mode,
                failure_class="EXPECTED_EARLY_FAILURE_BOUNDARY",
                reject_reason="interface_replicated_dm_python_api_unavailable",
            )
            flayout = write_attempt_failure(
                cfg=cfg,
                output_layout=runtime.output_layout,
                step_id=int(runtime.next_step_id),
                retry_id=0,
                t_old=float(runtime.t),
                dt_try=float(_initial_dt_from_cfg(cfg)),
                failure_class="EXPECTED_EARLY_FAILURE_BOUNDARY",
                reject_reason="interface_replicated_dm_python_api_unavailable",
                state_last_good=runtime.state,
                state_attempt_input=runtime.state,
                state_attempt_final=None,
                runtime=runtime,
                grid=runtime.grid,
                layout=runtime.layout,
                props_last_good=runtime.props,
                props_attempt_input=runtime.props,
                ctx_meta={"message": "known multi-rank early failure boundary"},
            )
            runtime_log(
                runtime,
                "WARNING",
                "expected_boundary_written",
                failure_dir=getattr(flayout, "root", None) if flayout is not None else None,
                run_summary_json=runtime.output_layout.run_summary_json,
            )
            record_run_end(runtime, exit_code=2, ended_by="expected_boundary", message="known multi-rank early failure boundary")
            write_run_summary(runtime.output_layout, summary=build_run_summary_payload(runtime), is_root=runtime.is_root)
            return 2

        exit_code = _run_step_loop(
            cfg,
            runtime=runtime,
            models=build_pack["models"],
            accepted_geometry_old=build_pack["geometry0"],
            dot_a_old=float(build_pack["dot_a0"]),
        )
        ended_by = "completed" if exit_code == 0 else "step_failure"
        message = runtime.last_message if exit_code != 0 else "run completed"
        record_run_end(runtime, exit_code=exit_code, ended_by=ended_by, message=message)
        if exit_code == 0:
            runtime_log(
                runtime,
                "INFO",
                "run_completed",
                n_steps_attempted=runtime.counters.n_steps_attempted,
                n_steps_accepted=runtime.counters.n_steps_accepted,
                n_retries_total=runtime.counters.n_retries_total,
                t_final=runtime.t,
                dt_last=runtime.last_dt_used,
                run_summary_json=runtime.output_layout.run_summary_json,
        )
        write_run_summary(runtime.output_layout, summary=build_run_summary_payload(runtime), is_root=runtime.is_root)
        return int(exit_code)
    except Exception as exc:
        _best_effort_runtime_failure_side_effects(
            runtime=runtime,
            cfg=cfg,
            exc=exc,
            config_path=config_path,
        )
        raise
    finally:
        if cfg is not None:
            unbind_cfg_runtime_bridges(cfg)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    return run_case(args.config, dry_run=bool(args.dry_run), run_id=args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
