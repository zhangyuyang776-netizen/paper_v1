from __future__ import annotations

"""Fixed-geometry nonlinear-solve context assembly for Phase 5.

This module packages the frozen-geometry inner-solve environment into one
lightweight context object. It does not perform remap, recovery, property
evaluation, residual/Jacobian assembly, or PETSc solver construction.
"""

from dataclasses import dataclass, field, replace
from math import isclose, isfinite

from .nonlinear_types import InnerEntrySource


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
    solve. The formal contract is expressed in terms of ``state_init``,
    ``accepted_state_n``, ``transfer_in``, and current-mesh reference data.
    Legacy names such as ``state_guess`` and
    ``old_state_on_current_geometry`` remain available through read-only
    compatibility properties during the migration.
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

    entry_source: InnerEntrySource | None = None
    state_init: object | None = None
    accepted_state_n: object | None = None
    transfer_in: object | None = None
    reference_state_current_mesh: object | None = None
    reference_contents_current_mesh: object | None = None

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

    @property
    def state_guess(self) -> object | None:
        """Compatibility alias for the formal ``state_init`` field."""

        return self.state_init

    @property
    def accepted_state_old(self) -> object | None:
        """Compatibility alias for the formal ``accepted_state_n`` field."""

        return self.accepted_state_n

    @property
    def old_state_on_current_geometry(self) -> object | None:
        """Compatibility alias for ``reference_state_current_mesh``."""

        return self.reference_state_current_mesh

    @property
    def old_mass_on_current_geometry(self) -> object | None:
        """Compatibility alias for ``reference_contents_current_mesh``."""

        return self.reference_contents_current_mesh


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


def _state_matches_grid(state: object, grid: object) -> bool:
    n_liq = getattr(state, "n_liq_cells", None)
    n_gas = getattr(state, "n_gas_cells", None)
    grid_n_liq = getattr(grid, "n_liq", None)
    grid_n_gas = getattr(grid, "n_gas", None)
    return (
        n_liq is not None
        and n_gas is not None
        and grid_n_liq is not None
        and grid_n_gas is not None
        and int(n_liq) == int(grid_n_liq)
        and int(n_gas) == int(grid_n_gas)
    )


def _contents_match_grid(contents: object, grid: object) -> bool:
    n_liq = getattr(contents, "n_liq_cells", None)
    n_gas = getattr(contents, "n_gas_cells", None)
    grid_n_liq = getattr(grid, "n_liq", None)
    grid_n_gas = getattr(grid, "n_gas", None)
    return (
        n_liq is not None
        and n_gas is not None
        and grid_n_liq is not None
        and grid_n_gas is not None
        and int(n_liq) == int(grid_n_liq)
        and int(n_gas) == int(grid_n_gas)
    )


def validate_nonlinear_context(ctx: NonlinearContext) -> None:
    """Validate core fixed-geometry context invariants before inner solve use."""

    for name in (
        "cfg",
        "layout",
        "grid",
        "state_init",
        "accepted_state_n",
        "props_current",
        "entry_source",
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
    if not isinstance(ctx.entry_source, InnerEntrySource):
        raise ValueError("NonlinearContext.entry_source must be an InnerEntrySource")

    if _layout_has_rd_block(ctx.layout):
        raise ValueError("NonlinearContext.layout must not contain a forbidden Rd block")

    if ctx.geometry_current is None:
        raise ValueError("NonlinearContext.geometry_current must be provided or inferrable from grid")

    if ctx.entry_source is InnerEntrySource.ACCEPTED_TIME_LEVEL:
        if ctx.transfer_in is not None:
            raise ValueError("accepted_time_level entry_source requires transfer_in == None")
    elif ctx.entry_source is InnerEntrySource.TRANSFER_FROM_PREVIOUS_OUTER:
        if ctx.outer_iter_id < 1:
            raise ValueError("transfer_from_previous_outer entry_source requires outer_iter_id >= 1")
        if ctx.transfer_in is None:
            raise ValueError("transfer_from_previous_outer entry_source requires transfer_in")

    if ctx.reference_state_current_mesh is not None and not _state_matches_grid(ctx.reference_state_current_mesh, ctx.grid):
        raise ValueError("reference_state_current_mesh must match current grid cell counts")
    if (
        ctx.reference_contents_current_mesh is not None
        and not _contents_match_grid(ctx.reference_contents_current_mesh, ctx.grid)
    ):
        raise ValueError("reference_contents_current_mesh must match current grid cell counts")

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
    state_init: object | None = None,
    accepted_state_n: object | None = None,
    entry_source: InnerEntrySource | str | None = None,
    transfer_in: object | None = None,
    reference_state_current_mesh: object | None = None,
    reference_contents_current_mesh: object | None = None,
    state_guess: object | None = None,
    accepted_state_old: object | None = None,
    old_state_on_current_geometry: object | None = None,
    old_mass_on_current_geometry: object | None = None,
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
    """Build and validate the fixed-geometry inner-solve context.

    The formal contract uses the new ``state_init`` / ``accepted_state_n`` /
    ``transfer_in`` language. Legacy parameters remain accepted during the
    migration, but must not be mixed with their new-name equivalents.
    """

    if state_init is not None and state_guess is not None:
        raise ValueError("Provide either state_init or state_guess, not both")
    if accepted_state_n is not None and accepted_state_old is not None:
        raise ValueError("Provide either accepted_state_n or accepted_state_old, not both")
    if reference_state_current_mesh is not None and old_state_on_current_geometry is not None:
        raise ValueError(
            "Provide either reference_state_current_mesh or old_state_on_current_geometry, not both"
        )
    if reference_contents_current_mesh is not None and old_mass_on_current_geometry is not None:
        raise ValueError(
            "Provide either reference_contents_current_mesh or old_mass_on_current_geometry, not both"
        )

    state_init_eff = state_init if state_init is not None else state_guess
    accepted_state_n_eff = accepted_state_n if accepted_state_n is not None else accepted_state_old
    reference_state_eff = (
        reference_state_current_mesh
        if reference_state_current_mesh is not None
        else old_state_on_current_geometry
    )
    reference_contents_eff = (
        reference_contents_current_mesh
        if reference_contents_current_mesh is not None
        else old_mass_on_current_geometry
    )
    if transfer_in is not None and reference_state_eff is None:
        reference_state_eff = getattr(transfer_in, "state", None)
    if transfer_in is not None and reference_contents_eff is None:
        reference_contents_eff = getattr(transfer_in, "contents", None)

    if entry_source is None:
        raise ValueError("Provide entry_source explicitly; silent inference is not allowed")
    entry_source_eff = InnerEntrySource(entry_source)

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
        entry_source=entry_source_eff,
        state_init=state_init_eff,
        accepted_state_n=accepted_state_n_eff,
        transfer_in=transfer_in,
        reference_state_current_mesh=reference_state_eff,
        reference_contents_current_mesh=reference_contents_eff,
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
    """Compatibility wrapper for ``clone_context_with_state_init``."""

    return clone_context_with_state_init(
        ctx,
        state_init=state_guess,
        u_guess_vec=u_guess_vec,
        props_current=props_current,
        reference_contents_current_mesh=old_mass_on_current_geometry,
    )


def clone_context_with_state_init(
    ctx: NonlinearContext,
    *,
    state_init: object | None = None,
    u_guess_vec: object | None = None,
    props_current: object | None = None,
    reference_contents_current_mesh: object | None = None,
    reference_state_current_mesh: object | None = None,
    transfer_in: object | None = None,
) -> NonlinearContext:
    """Clone a nonlinear context while replacing inner-entry-related fields."""

    transfer_in_eff = ctx.transfer_in if transfer_in is None else transfer_in
    reference_contents_eff = reference_contents_current_mesh
    if reference_contents_eff is None and transfer_in is not None:
        reference_contents_eff = getattr(transfer_in, "contents", None)
    if reference_contents_eff is None:
        reference_contents_eff = ctx.reference_contents_current_mesh

    reference_state_eff = reference_state_current_mesh
    if reference_state_eff is None and transfer_in is not None:
        reference_state_eff = getattr(transfer_in, "state", None)
    if reference_state_eff is None:
        reference_state_eff = ctx.reference_state_current_mesh

    cloned = replace(
        ctx,
        state_init=ctx.state_init if state_init is None else state_init,
        u_guess_vec=ctx.u_guess_vec if u_guess_vec is None else u_guess_vec,
        props_current=ctx.props_current if props_current is None else props_current,
        reference_contents_current_mesh=reference_contents_eff,
        reference_state_current_mesh=reference_state_eff,
        transfer_in=transfer_in_eff,
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
    "clone_context_with_state_init",
    "clone_context_with_state_guess",
    "validate_nonlinear_context",
]
