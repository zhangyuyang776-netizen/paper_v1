from __future__ import annotations

"""Fixed-geometry nonlinear-solve context assembly for Phase 5.

This module packages the frozen-geometry inner-solve environment into one
lightweight context object. It does not perform remap, recovery, property
evaluation, residual/Jacobian assembly, or PETSc solver construction.
"""

from dataclasses import dataclass, field, replace
from math import isclose, isfinite


@dataclass(slots=True, kw_only=True)
class NonlinearModelHandles:
    """Container for runtime model/handle objects used by solver callbacks.

    Produced by higher-level setup/timestep orchestration and consumed by
    residual/Jacobian/global solver callbacks through ``NonlinearContext``.
    """

    gas_model: object | None = None
    liquid_model: object | None = None
    equilibrium_model: object | None = None
    property_aggregator: object | None = None
    liquid_db: object | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class NonlinearContext:
    """Fixed-geometry inner-solve context shared by residual/Jacobian/SNES.

    This object is the single environment entry point for the Phase 5 inner
    solve. It carries the frozen outer iterate, current-geometry old state,
    current state guess, bulk properties, and optional backend-native vector
    handles without introducing a second global numpy truth source.
    """

    cfg: object
    layout: object
    grid: object

    step_id: int | None = None
    outer_iter_id: int = 0

    t_old: float = 0.0
    dt: float = 0.0
    t_new: float = 0.0

    a_current: float = 0.0
    dot_a_frozen: float = 0.0
    geometry_current: object | None = None

    state_guess: object | None = None
    accepted_state_old: object | None = None
    old_state_on_current_geometry: object | None = None
    old_mass_on_current_geometry: object | None = None

    props_current: object | None = None
    models: NonlinearModelHandles = field(default_factory=NonlinearModelHandles)

    u_guess_vec: object | None = None
    dm_composite: object | None = None
    global_vec_template: object | None = None
    parallel_handles: dict[str, object] = field(default_factory=dict)

    meta: dict[str, object] = field(default_factory=dict)
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def t_new_target(self) -> float:
        """Return the current target time for this inner solve."""

        return self.t_new


def _infer_geometry_view(grid: object) -> object | None:
    """Return an existing read-only geometry view from ``grid`` when available."""

    for attr in ("geometry_current", "geometry", "mesh_geometry"):
        if hasattr(grid, attr):
            return getattr(grid, attr)
    return None


def _layout_has_rd_block(layout: object) -> bool:
    """Return whether the provided layout still exposes a forbidden ``Rd`` block."""

    has_block = getattr(layout, "has_block", None)
    if callable(has_block):
        try:
            return bool(has_block("Rd"))
        except TypeError:
            return bool(has_block(block_name="Rd"))
    blocks = getattr(layout, "blocks", None)
    if isinstance(blocks, dict):
        return "Rd" in blocks
    return False


def validate_nonlinear_context(ctx: NonlinearContext) -> None:
    """Validate core fixed-geometry context invariants before inner solve use."""

    for name in (
        "cfg",
        "layout",
        "grid",
        "state_guess",
        "accepted_state_old",
        "old_state_on_current_geometry",
        "old_mass_on_current_geometry",
        "props_current",
    ):
        if getattr(ctx, name) is None:
            raise ValueError(f"NonlinearContext.{name} must be provided")

    if not isfinite(float(ctx.dt)) or float(ctx.dt) <= 0.0:
        raise ValueError("NonlinearContext.dt must be finite and > 0")
    if not isfinite(float(ctx.t_old)):
        raise ValueError("NonlinearContext.t_old must be finite")
    if not isfinite(float(ctx.t_new)):
        raise ValueError("NonlinearContext.t_new must be finite")
    if not isclose(float(ctx.t_new), float(ctx.t_old) + float(ctx.dt), rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("NonlinearContext.t_new must equal t_old + dt within tolerance")
    if not isfinite(float(ctx.a_current)) or float(ctx.a_current) <= 0.0:
        raise ValueError("NonlinearContext.a_current must be finite and > 0")
    if not isfinite(float(ctx.dot_a_frozen)):
        raise ValueError("NonlinearContext.dot_a_frozen must be finite")
    if not isinstance(ctx.outer_iter_id, int) or ctx.outer_iter_id < 0:
        raise ValueError("NonlinearContext.outer_iter_id must be an integer >= 0")

    if _layout_has_rd_block(ctx.layout):
        raise ValueError("NonlinearContext.layout must not contain a forbidden Rd block")

    if ctx.accepted_state_old is ctx.old_state_on_current_geometry:
        raise ValueError(
            "accepted_state_old and old_state_on_current_geometry must remain distinct objects"
        )

    if ctx.geometry_current is None:
        raise ValueError("NonlinearContext.geometry_current must be provided or inferrable from grid")

    if ctx.u_guess_vec is not None and "u_guess_array" in ctx.meta:
        raise ValueError("NonlinearContext must not carry a shadow global numpy truth source in meta")

    if not isinstance(ctx.models, NonlinearModelHandles):
        raise ValueError("NonlinearContext.models must be a NonlinearModelHandles instance")
    if not isinstance(ctx.parallel_handles, dict):
        raise ValueError("NonlinearContext.parallel_handles must be a dictionary")
    if not isinstance(ctx.meta, dict):
        raise ValueError("NonlinearContext.meta must be a dictionary")
    if not isinstance(ctx.diagnostics, dict):
        raise ValueError("NonlinearContext.diagnostics must be a dictionary")


def build_nonlinear_context(
    *,
    cfg: object,
    layout: object,
    grid: object,
    t_old: float,
    dt: float,
    a_current: float,
    dot_a_frozen: float,
    state_guess: object,
    accepted_state_old: object,
    old_state_on_current_geometry: object,
    old_mass_on_current_geometry: object,
    props_current: object,
    models: NonlinearModelHandles | None = None,
    step_id: int | None = None,
    outer_iter_id: int = 0,
    geometry_current: object | None = None,
    u_guess_vec: object | None = None,
    dm_composite: object | None = None,
    global_vec_template: object | None = None,
    parallel_handles: dict[str, object] | None = None,
    meta: dict[str, object] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> NonlinearContext:
    """Build and validate the fixed-geometry inner-solve context."""

    geometry_view = geometry_current if geometry_current is not None else _infer_geometry_view(grid)
    ctx = NonlinearContext(
        cfg=cfg,
        layout=layout,
        grid=grid,
        step_id=step_id,
        outer_iter_id=int(outer_iter_id),
        t_old=float(t_old),
        dt=float(dt),
        t_new=float(t_old) + float(dt),
        a_current=float(a_current),
        dot_a_frozen=float(dot_a_frozen),
        geometry_current=geometry_view,
        state_guess=state_guess,
        accepted_state_old=accepted_state_old,
        old_state_on_current_geometry=old_state_on_current_geometry,
        old_mass_on_current_geometry=old_mass_on_current_geometry,
        props_current=props_current,
        models=models if models is not None else NonlinearModelHandles(),
        u_guess_vec=u_guess_vec,
        dm_composite=dm_composite,
        global_vec_template=global_vec_template,
        parallel_handles=dict(parallel_handles) if parallel_handles is not None else {},
        meta=dict(meta) if meta is not None else {},
        diagnostics=dict(diagnostics) if diagnostics is not None else {},
    )
    validate_nonlinear_context(ctx)
    return ctx


def clone_context_with_state_guess(
    ctx: NonlinearContext,
    *,
    state_guess: object | None = None,
    u_guess_vec: object | None = None,
    props_current: object | None = None,
    old_mass_on_current_geometry: object | None = None,
) -> NonlinearContext:
    """Clone a nonlinear context while replacing the current guess-related fields."""

    cloned = replace(
        ctx,
        state_guess=ctx.state_guess if state_guess is None else state_guess,
        u_guess_vec=ctx.u_guess_vec if u_guess_vec is None else u_guess_vec,
        props_current=ctx.props_current if props_current is None else props_current,
        old_mass_on_current_geometry=(
            ctx.old_mass_on_current_geometry
            if old_mass_on_current_geometry is None
            else old_mass_on_current_geometry
        ),
        parallel_handles=dict(ctx.parallel_handles),
        meta=dict(ctx.meta),
        diagnostics=dict(ctx.diagnostics),
    )
    validate_nonlinear_context(cloned)
    return cloned


__all__ = [
    "NonlinearContext",
    "NonlinearModelHandles",
    "build_nonlinear_context",
    "clone_context_with_state_guess",
    "validate_nonlinear_context",
]
