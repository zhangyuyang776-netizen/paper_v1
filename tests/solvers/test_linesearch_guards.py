from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.layout import UnknownLayout
from solvers.linesearch_guards import (
    GuardCheckResult,
    TrialPhysicsSnapshot,
    build_linesearch_guard_config,
    check_trial_snapshot,
    check_trial_vector,
    make_guard_failure,
    make_petsc_linesearch_postcheck,
    make_petsc_linesearch_precheck,
    update_guard_diagnostics,
)
from solvers.nonlinear_context import build_nonlinear_context
from solvers.nonlinear_types import FailureClass, GuardFailureReason


def _make_layout() -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_A",
        n_liq_cells=1,
        n_gas_cells=2,
        n_liq_red=0,
        n_gas_red=1,
        liq_cell_width=1,
        gas_cell_width=2,
        n_if_unknowns=3,
        liq_block=slice(0, 1),
        if_block=slice(1, 4),
        gas_block=slice(4, 8),
        total_size=8,
        liq_temperature_slice=slice(0, 1, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(1, 2),
        if_gas_species_slice=slice(2, 3),
        if_mpp_slice=slice(3, 4),
        if_liq_species_slice=slice(4, 4),
        gas_temperature_slice=slice(4, 8, 2),
        gas_species_local_slice=slice(1, 2),
    )


def _make_ctx():
    cfg = SimpleNamespace(
        recovery=SimpleNamespace(T_min_l=250.0, T_max_l=500.0, T_min_g=200.0, T_max_g=800.0),
        solver_inner_petsc=SimpleNamespace(
            guard=SimpleNamespace(
                allow_small_ts_clip=True,
                ts_clip_abs_tol=1.0e-8,
                allow_small_mpp_clip=False,
            )
        ),
    )
    grid = SimpleNamespace(geometry_current=SimpleNamespace(name="geom"))
    layout = _make_layout()
    return build_nonlinear_context(
        cfg=cfg,
        layout=layout,
        grid=grid,
        t_old=0.0,
        dt=1.0,
        a_current=1.0,
        dot_a_frozen=0.0,
        state_guess=object(),
        accepted_state_old=object(),
        old_state_on_current_geometry=object(),
        old_mass_on_current_geometry=object(),
        props_current=object(),
        diagnostics={},
    )


def _trial_vector(*, Ts: float = 300.0, mpp: float = 0.0) -> np.ndarray:
    vec = np.asarray([300.0, Ts, 0.2, mpp, 300.0, 0.1, 301.0, 0.1], dtype=np.float64)
    return vec


def test_guard_nonfinite_trial_vector() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    trial = _trial_vector()
    trial[0] = np.nan

    result = check_trial_vector(ctx=ctx, u_trial_array=trial, cfg_guard=cfg)

    assert not result.ok
    assert result.guard_active
    assert result.failure_reason is GuardFailureReason.NONFINITE_STATE
    assert result.failure.failure_class is FailureClass.GUARD_FAIL


def test_guard_small_ts_clip_allowed() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    trial = _trial_vector(Ts=199.999999995)

    result = check_trial_vector(ctx=ctx, u_trial_array=trial, cfg_guard=cfg)

    assert result.ok
    assert result.changed
    assert result.clip_ts
    assert trial[ctx.layout.if_temperature_index] == pytest.approx(200.0)


def test_guard_large_ts_violation_fails() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    trial = _trial_vector(Ts=150.0)

    result = check_trial_vector(ctx=ctx, u_trial_array=trial, cfg_guard=cfg)

    assert not result.ok
    assert result.failure_reason is GuardFailureReason.TEMPERATURE_OUT_OF_RANGE


def test_guard_snapshot_density_fail() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    snapshot = TrialPhysicsSnapshot(rho_g_min=0.0)

    result = check_trial_snapshot(ctx=ctx, snapshot=snapshot, cfg_guard=cfg)

    assert not result.ok
    assert result.failure_reason is GuardFailureReason.NEGATIVE_DENSITY


def test_guard_snapshot_property_fail() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    snapshot = TrialPhysicsSnapshot(property_success=False)

    result = check_trial_snapshot(ctx=ctx, snapshot=snapshot, cfg_guard=cfg)

    assert not result.ok
    assert result.failure_reason is GuardFailureReason.PROPERTY_EVAL_FAILED


def test_guard_snapshot_enthalpy_fail() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    snapshot = TrialPhysicsSnapshot(enthalpy_inversion_success=False)

    result = check_trial_snapshot(ctx=ctx, snapshot=snapshot, cfg_guard=cfg)

    assert not result.ok
    assert result.failure_reason is GuardFailureReason.ENTHALPY_INVERSION_FAILED


def test_guard_no_silent_species_projection() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    snapshot = TrialPhysicsSnapshot(y_g_min=-1.0e-4, y_g_max=1.0)

    result = check_trial_snapshot(ctx=ctx, snapshot=snapshot, cfg_guard=cfg)

    assert not result.ok
    assert result.failure_reason is GuardFailureReason.COMPOSITION_OUT_OF_RANGE


def test_guard_failure_info_is_structured() -> None:
    failure = make_guard_failure(
        reason=GuardFailureReason.PROPERTY_EVAL_FAILED,
        where="linesearch_postcheck",
        message="property failure",
        recoverable=True,
        meta={"probe": 1},
    )

    assert failure.failure_class is FailureClass.GUARD_FAIL
    assert failure.where == "linesearch_postcheck"
    assert failure.reason_code == GuardFailureReason.PROPERTY_EVAL_FAILED.value
    assert failure.meta["probe"] == 1


def test_pre_and_post_adapter_update_context_diagnostics() -> None:
    ctx = _make_ctx()
    cfg = build_linesearch_guard_config(ctx)
    pre_results: list[GuardCheckResult] = []
    post_results: list[GuardCheckResult] = []
    domain_failures: list[GuardCheckResult] = []
    pre = make_petsc_linesearch_precheck(
        ctx=ctx,
        cfg_guard=cfg,
        array_view_factory=lambda vec, writable=False: vec,
        on_guard_result=pre_results.append,
        on_domain_error=domain_failures.append,
    )
    trial = _trial_vector(Ts=199.999999995)
    pre_result = pre(None, None, trial)

    assert pre_result is True
    assert pre_results
    assert isinstance(ctx.meta["last_guard_result"], GuardCheckResult)
    assert ctx.meta["last_guard_result"].ok
    assert ctx.diagnostics["guard_active"] is True
    assert ctx.diagnostics["ls_clip_ts"] is True
    assert domain_failures == []

    post = make_petsc_linesearch_postcheck(
        ctx=ctx,
        cfg_guard=cfg,
        snapshot_getter=lambda: TrialPhysicsSnapshot(property_success=False),
        on_guard_result=post_results.append,
        on_domain_error=domain_failures.append,
    )
    post_result = post(None, None, None)

    assert post_result is False
    assert post_results
    assert isinstance(ctx.meta["last_guard_result"], GuardCheckResult)
    assert not ctx.meta["last_guard_result"].ok
    assert domain_failures
    assert domain_failures[-1].failure_reason is GuardFailureReason.PROPERTY_EVAL_FAILED
    assert ctx.diagnostics["guard_reason"] == GuardFailureReason.PROPERTY_EVAL_FAILED.value


def test_update_guard_diagnostics_merges_explicit_counters() -> None:
    result = GuardCheckResult(
        ok=True,
        guard_active=True,
        clip_ts=True,
        diagnostics={"nonfinite_state_detected": False},
    )

    diagnostics = update_guard_diagnostics(
        result=result,
        diagnostics={"existing": 1},
        ls_shrink_count=3,
        line_search_backtracks=2,
    )

    assert diagnostics["existing"] == 1
    assert diagnostics["ls_clip_ts"] is True
    assert diagnostics["ls_shrink_count"] == 3
    assert diagnostics["line_search_backtracks"] == 2
