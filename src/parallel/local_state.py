from __future__ import annotations

"""Distributed PETSc Vec access helpers and State recovery from inner/global Vecs."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from core.state_pack import pack_state_to_array, unpack_array_to_state


class LocalStateError(RuntimeError):
    """Raised when distributed local/global state access or conversion fails."""


@dataclass(slots=True)
class LocalSubdomainView:
    """Ghosted local view for one bulk subdomain."""

    local_vec: object
    array_view: np.ndarray
    dof: int
    xs: int
    xm: int
    gxs: int
    gxm: int
    _restore: Callable[[], None] | None = field(default=None, repr=False)

    @property
    def owned_local_slice(self) -> slice:
        start = int(self.xs - self.gxs)
        return slice(start, start + int(self.xm))


@dataclass(slots=True)
class LocalStateView:
    """Composite local ghosted view for liquid / interface / gas state access."""

    dm_mgr: object
    X_global: object
    liq: LocalSubdomainView
    gas: LocalSubdomainView
    if_local_vec: object
    if_array_view: np.ndarray
    interface_owner_active: bool
    _if_restore: Callable[[], None] | None = field(default=None, repr=False)


def _require_dm_manager_from_handles(parallel_handles: dict[str, object]):
    dm_mgr = dict(parallel_handles or {}).get("dm_manager")
    if dm_mgr is None:
        raise LocalStateError("parallel_handles.dm_manager is required")
    return dm_mgr


def _require_global_vec_template(ctx):
    if getattr(ctx, "global_vec_template", None) is not None:
        return ctx.global_vec_template
    handles = getattr(ctx, "parallel_handles", {}) or {}
    if handles.get("global_vec_template") is not None:
        return handles["global_vec_template"]
    dm_mgr = handles.get("dm_manager")
    if dm_mgr is not None and getattr(dm_mgr, "global_vec_template", None) is not None:
        return dm_mgr.global_vec_template
    raise LocalStateError("distributed Vec builder requires a global_vec_template")


def _copy_vec_like(vec: object):
    copy_fn = getattr(vec, "copy", None)
    if callable(copy_fn):
        try:
            return copy_fn()
        except TypeError:
            pass
    duplicate = getattr(vec, "duplicate", None)
    if callable(duplicate):
        out = duplicate()
        src_arr, src_restore = _vec_array_readonly(vec)
        dst_arr, dst_restore = _vec_array_readwrite(out)
        try:
            dst_arr[...] = src_arr
        finally:
            dst_restore()
            src_restore()
        return out
    raise LocalStateError("unable to copy distributed Vec-like object")


def _require_state_init_or_u_guess(ctx) -> None:
    if getattr(ctx, "u_guess_vec", None) is not None:
        return
    if getattr(ctx, "state_init", None) is None:
        raise LocalStateError(
            "distributed state Vec builder requires ctx.state_init when ctx.u_guess_vec is absent"
        )


def _composite_get_access(dm_composite, X_global) -> tuple[object, object, object]:
    get_access = getattr(dm_composite, "getAccess", None)
    if not callable(get_access):
        raise LocalStateError("DMComposite does not expose getAccess")
    try:
        sub_vecs = get_access(X_global)
    except TypeError as exc:
        raise LocalStateError(f"failed to access composite subvectors: {exc}") from exc
    if not isinstance(sub_vecs, (tuple, list)) or len(sub_vecs) != 3:
        raise LocalStateError("DMComposite.getAccess must return 3 subvectors")
    return tuple(sub_vecs)  # type: ignore[return-value]


def _composite_restore_access(dm_composite, X_global, sub_vecs: tuple[object, object, object]) -> None:
    restore = getattr(dm_composite, "restoreAccess", None)
    if not callable(restore):
        return
    try:
        restore(X_global, sub_vecs)
        return
    except TypeError:
        pass
    try:
        restore(X_global, *sub_vecs)
    except TypeError as exc:
        raise LocalStateError(f"failed to restore composite subvector access: {exc}") from exc


def _composite_scatter(dm_composite, X_global, local_vecs: tuple[object, object, object]) -> None:
    scatter = getattr(dm_composite, "scatter", None)
    if not callable(scatter):
        raise LocalStateError("DMComposite does not expose scatter")
    for args in ((X_global, local_vecs), (X_global, list(local_vecs)), (X_global, *local_vecs)):
        try:
            scatter(*args)
            return
        except TypeError:
            continue
        except Exception as exc:
            raise LocalStateError(f"failed to scatter composite global Vec: {exc}") from exc
    raise LocalStateError("failed to scatter composite global Vec due to unsupported DMComposite.scatter signature")


def _composite_gather(dm_composite, X_global, insert_mode, local_vecs: tuple[object, object, object]) -> None:
    gather = getattr(dm_composite, "gather", None)
    if not callable(gather):
        raise LocalStateError("DMComposite does not expose gather")
    for args in (
        (X_global, insert_mode, local_vecs),
        (X_global, insert_mode, list(local_vecs)),
        (X_global, insert_mode, *local_vecs),
    ):
        try:
            gather(*args)
            return
        except TypeError:
            continue
        except Exception as exc:
            raise LocalStateError(f"failed to gather split local Vecs into composite global Vec: {exc}") from exc
    raise LocalStateError("failed to gather split local Vecs due to unsupported DMComposite.gather signature")


def _global_to_local(dm_sub, Xg_sub, Xl_sub) -> None:
    op = getattr(dm_sub, "globalToLocal", None)
    if not callable(op):
        raise LocalStateError("sub-DM does not expose globalToLocal")
    try:
        op(Xg_sub, Xl_sub)
    except TypeError:
        try:
            op(Xg_sub, Xl_sub, None)
        except TypeError as exc:
            raise LocalStateError(f"failed to execute globalToLocal: {exc}") from exc


def _local_to_global(dm_sub, Xl_sub, Xg_sub, insert_mode) -> None:
    op = getattr(dm_sub, "localToGlobal", None)
    if not callable(op):
        raise LocalStateError("sub-DM does not expose localToGlobal")
    try:
        op(Xl_sub, Xg_sub, insert_mode)
    except TypeError:
        try:
            op(Xl_sub, Xg_sub)
        except TypeError as exc:
            raise LocalStateError(f"failed to execute localToGlobal: {exc}") from exc


def _vec_array_readwrite(vec: object) -> tuple[np.ndarray, Callable[[], None]]:
    get_array = getattr(vec, "getArray", None)
    if callable(get_array):
        try:
            arr = np.asarray(get_array(), dtype=np.float64)
        except TypeError:
            arr = np.asarray(get_array(readonly=False), dtype=np.float64)
        restore = getattr(vec, "restoreArray", None)
        return arr, (lambda: restore(arr)) if callable(restore) else (lambda: None)
    raise LocalStateError("Vec-like object does not expose a writable array view")


def _vec_array_readonly(vec: object) -> tuple[np.ndarray, Callable[[], None]]:
    get_array_read = getattr(vec, "getArrayRead", None)
    if callable(get_array_read):
        arr = np.asarray(get_array_read(), dtype=np.float64)
        restore = getattr(vec, "restoreArrayRead", None)
        return arr, (lambda: restore(arr)) if callable(restore) else (lambda: None)
    get_array = getattr(vec, "getArray", None)
    if callable(get_array):
        try:
            arr = np.asarray(get_array(readonly=True), dtype=np.float64)
        except TypeError:
            arr = np.asarray(get_array(), dtype=np.float64)
        restore = getattr(vec, "restoreArray", None)
        return arr, (lambda: restore(arr)) if callable(restore) else (lambda: None)
    raise LocalStateError("Vec-like object does not expose a readable array view")


def _layout_to_petsc_full_array(arr_layout: np.ndarray, layout_to_petsc: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_layout, dtype=np.float64)
    perm = np.asarray(layout_to_petsc, dtype=np.int64)
    if arr.ndim != 1:
        raise LocalStateError("layout array must be one-dimensional")
    if perm.ndim != 1 or perm.shape[0] != arr.shape[0]:
        raise LocalStateError("layout_to_petsc must be a permutation with matching length")
    out = np.empty_like(arr)
    out[perm] = arr
    return out


def _petsc_to_layout_full_array(arr_petsc: np.ndarray, petsc_to_layout: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr_petsc, dtype=np.float64)
    perm = np.asarray(petsc_to_layout, dtype=np.int64)
    if arr.ndim != 1:
        raise LocalStateError("PETSc-order array must be one-dimensional")
    if perm.ndim != 1 or perm.shape[0] != arr.shape[0]:
        raise LocalStateError("petsc_to_layout must be a permutation with matching length")
    return arr[perm]


def _create_global_vec_like(template: object):
    duplicate = getattr(template, "duplicate", None)
    if callable(duplicate):
        return duplicate()
    copy_fn = getattr(template, "copy", None)
    if callable(copy_fn):
        return copy_fn()
    raise LocalStateError("global_vec_template must expose duplicate() or copy()")


def _insert_mode(dm_mgr: object, *, additive: bool):
    petsc = getattr(dm_mgr, "PETSc", None)
    insert_mode = getattr(petsc, "InsertMode", None)
    if insert_mode is not None:
        attr = "ADD_VALUES" if additive else "INSERT_VALUES"
        if hasattr(insert_mode, attr):
            return getattr(insert_mode, attr)
    return "add" if additive else "insert"


def create_distributed_vec_from_layout_array(*, layout_vec, template, parallel_handles: dict[str, object]) -> object:
    """Create one distributed global Vec from a full layout-ordered array."""

    arr = np.asarray(layout_vec, dtype=np.float64)
    if arr.ndim != 1:
        raise LocalStateError("layout_vec must be one-dimensional")
    layout_to_petsc = np.asarray(parallel_handles["layout_to_petsc"], dtype=np.int64)
    if layout_to_petsc.shape[0] != arr.shape[0]:
        raise LocalStateError("layout_to_petsc length must match layout_vec length")
    arr_petsc = _layout_to_petsc_full_array(arr, layout_to_petsc)

    rstart, rend = (int(v) for v in parallel_handles["ownership_range"])
    Xg = _create_global_vec_like(template)
    set_fn = getattr(Xg, "set", None)
    if callable(set_fn):
        set_fn(0.0)

    xloc, restore = _vec_array_readwrite(Xg)
    try:
        if xloc.shape[0] != rend - rstart:
            raise LocalStateError("local Vec array length must match ownership_range span")
        xloc[...] = arr_petsc[rstart:rend]
    finally:
        restore()

    assembly_begin = getattr(Xg, "assemblyBegin", None)
    assembly_end = getattr(Xg, "assemblyEnd", None)
    if callable(assembly_begin) and callable(assembly_end):
        assembly_begin()
        assembly_end()
    return Xg


def distributed_state_vec_builder(*, ctx, PETSc):
    """Build the distributed initial-guess Vec using the same priority as serial mode."""

    _ = PETSc
    if getattr(ctx, "u_guess_vec", None) is not None:
        return _copy_vec_like(ctx.u_guess_vec)

    _require_state_init_or_u_guess(ctx)
    template = _require_global_vec_template(ctx)
    species_maps = getattr(ctx, "meta", {}).get("species_maps")
    if species_maps is None:
        raise LocalStateError(
            "distributed_state_vec_builder requires species_maps when ctx.u_guess_vec is absent"
        )
    u_layout = pack_state_to_array(ctx.state_init, ctx.layout, species_maps)

    return create_distributed_vec_from_layout_array(
        layout_vec=u_layout,
        template=template,
        parallel_handles=ctx.parallel_handles,
    )


def distributed_residual_vec_builder(*, ctx, PETSc):
    """Build a zero-initialized distributed residual Vec."""

    _ = PETSc
    template = _require_global_vec_template(ctx)
    vec = _create_global_vec_like(template)
    set_fn = getattr(vec, "set", None)
    if callable(set_fn):
        set_fn(0.0)
    else:
        arr, restore = _vec_array_readwrite(vec)
        try:
            arr[...] = 0.0
        finally:
            restore()
    return vec


def gather_layout_vector_from_distributed_vec(*, X_global: object, parallel_handles: dict[str, object]) -> np.ndarray:
    """Gather a distributed global Vec into one full layout-ordered array."""

    handles = dict(parallel_handles or {})
    petsc_to_layout = np.asarray(handles["petsc_to_layout"], dtype=np.int64)
    ownership_ranges = np.asarray(handles["ownership_ranges"], dtype=np.int64)
    comm = handles["comm"]

    local_arr, restore = _vec_array_readonly(X_global)
    try:
        local_copy = np.array(local_arr, dtype=np.float64, copy=True)
    finally:
        restore()

    allgather = getattr(comm, "allgather", None)
    if callable(allgather):
        pieces = list(allgather(local_copy))
    else:
        pieces = [local_copy]
    if len(pieces) != ownership_ranges.shape[0]:
        raise LocalStateError("allgather result length must match ownership_ranges row count")

    total_size = int(np.max(ownership_ranges[:, 1])) if ownership_ranges.size else 0
    full_petsc = np.zeros(total_size, dtype=np.float64)
    for idx, piece in enumerate(pieces):
        rstart, rend = (int(v) for v in ownership_ranges[idx])
        piece_arr = np.asarray(piece, dtype=np.float64)
        if piece_arr.ndim != 1 or piece_arr.shape[0] != rend - rstart:
            raise LocalStateError("gathered local piece length must match ownership_ranges")
        full_petsc[rstart:rend] = piece_arr
    return _petsc_to_layout_full_array(full_petsc, petsc_to_layout)


def recover_state_from_distributed_vec(
    *,
    X_global: object,
    layout: object,
    species_maps: object,
    parallel_handles: dict[str, object],
    time: float | None = None,
    state_id: str | None = None,
):
    """Recover one full State from one distributed global Vec."""

    arr = gather_layout_vector_from_distributed_vec(
        X_global=X_global,
        parallel_handles=parallel_handles,
    )
    return unpack_array_to_state(arr, layout, species_maps, time=time, state_id=state_id)


def state_init_from_previous_inner_result(
    *,
    previous_inner_result: object,
    reference_state_current_mesh: object,
    layout: object,
    species_maps: object,
    geometry_current: object | None = None,
    parallel_handles: dict[str, object] | None = None,
    models: object | None = None,
):
    """Recover the next inner entry State from the previous distributed inner Vec."""

    _ = models
    Xg = getattr(previous_inner_result, "solution_vec", None)
    if Xg is None:
        Xg = getattr(previous_inner_result, "state_vec", None)
    if Xg is None:
        raise LocalStateError("previous_inner_result must provide solution_vec or state_vec")
    if parallel_handles is None:
        raise LocalStateError("parallel_handles is required for distributed state recovery")

    state_base = getattr(reference_state_current_mesh, "state", reference_state_current_mesh)
    time = getattr(geometry_current, "t", getattr(state_base, "time", None))
    state_id = getattr(state_base, "state_id", None)
    return recover_state_from_distributed_vec(
        X_global=Xg,
        layout=layout,
        species_maps=species_maps,
        parallel_handles=parallel_handles,
        time=time,
        state_id=state_id,
    )


def state_guess_from_previous_inner_result(
    *,
    previous_inner_result: object,
    old_state_on_current_geometry: object,
    layout: object,
    species_maps: object,
    geometry_current: object | None = None,
    parallel_handles: dict[str, object] | None = None,
    models: object | None = None,
):
    """Compatibility wrapper for ``state_init_from_previous_inner_result``."""

    return state_init_from_previous_inner_result(
        previous_inner_result=previous_inner_result,
        reference_state_current_mesh=old_state_on_current_geometry,
        layout=layout,
        species_maps=species_maps,
        geometry_current=geometry_current,
        parallel_handles=parallel_handles,
        models=models,
    )


def begin_local_state_view(dm_mgr: object, X_global: object) -> LocalStateView:
    """Acquire local ghosted liquid/gas views and local interface data from one global Vec."""

    Xl_liq = dm_mgr.dm_liq.createLocalVec()
    Xl_if = dm_mgr.dm_if.createLocalVec()
    Xl_gas = dm_mgr.dm_gas.createLocalVec()

    _composite_scatter(dm_mgr.dm_composite, X_global, (Xl_liq, Xl_if, Xl_gas))

    liq_arr, liq_restore = _vec_array_readwrite(Xl_liq)
    gas_arr, gas_restore = _vec_array_readwrite(Xl_gas)
    if_arr, if_restore = _vec_array_readwrite(Xl_if)

    liq_gxm = int(dm_mgr.liq_ghost_corners[1])
    gas_gxm = int(dm_mgr.gas_ghost_corners[1])
    liq_arr_np = np.asarray(liq_arr, dtype=np.float64)
    gas_arr_np = np.asarray(gas_arr, dtype=np.float64)
    if_arr_np = np.asarray(if_arr, dtype=np.float64)
    if liq_arr_np.size != liq_gxm * int(dm_mgr.liq_dof):
        raise LocalStateError(
            f"liquid local array size mismatch: {liq_arr_np.size} != {liq_gxm}*{int(dm_mgr.liq_dof)}"
        )
    if gas_arr_np.size != gas_gxm * int(dm_mgr.gas_dof):
        raise LocalStateError(
            f"gas local array size mismatch: {gas_arr_np.size} != {gas_gxm}*{int(dm_mgr.gas_dof)}"
        )
    if if_arr_np.size != int(dm_mgr.n_if_unknowns):
        raise LocalStateError(
            f"interface local array size mismatch: {if_arr_np.size} != {int(dm_mgr.n_if_unknowns)}"
        )
    liq_view = LocalSubdomainView(
        local_vec=Xl_liq,
        array_view=liq_arr_np.reshape(liq_gxm, int(dm_mgr.liq_dof)),
        dof=int(dm_mgr.liq_dof),
        xs=int(dm_mgr.liq_corners[0]),
        xm=int(dm_mgr.liq_corners[1]),
        gxs=int(dm_mgr.liq_ghost_corners[0]),
        gxm=liq_gxm,
        _restore=liq_restore,
    )
    gas_view = LocalSubdomainView(
        local_vec=Xl_gas,
        array_view=gas_arr_np.reshape(gas_gxm, int(dm_mgr.gas_dof)),
        dof=int(dm_mgr.gas_dof),
        xs=int(dm_mgr.gas_corners[0]),
        xm=int(dm_mgr.gas_corners[1]),
        gxs=int(dm_mgr.gas_ghost_corners[0]),
        gxm=gas_gxm,
        _restore=gas_restore,
    )
    return LocalStateView(
        dm_mgr=dm_mgr,
        X_global=X_global,
        liq=liq_view,
        gas=gas_view,
        if_local_vec=Xl_if,
        if_array_view=if_arr_np.reshape(int(dm_mgr.n_if_unknowns)),
        interface_owner_active=bool(dm_mgr.rank == dm_mgr.interface_owner_rank),
        _if_restore=if_restore,
    )


def end_local_state_view(view: LocalStateView) -> None:
    """Release local ghosted Vecs and restore composite access."""

    if view.liq._restore is not None:
        view.liq._restore()
    if view.gas._restore is not None:
        view.gas._restore()
    if view._if_restore is not None:
        view._if_restore()

    for vec in (view.liq.local_vec, view.if_local_vec, view.gas.local_vec):
        destroy = getattr(vec, "destroy", None)
        if callable(destroy):
            destroy()


@contextmanager
def local_state_view(dm_mgr: object, X_global: object):
    """Context-managed local ghosted access to one composite distributed Vec."""

    view = begin_local_state_view(dm_mgr, X_global)
    try:
        yield view
    finally:
        end_local_state_view(view)


def _prepare_output_global_vec(dm_mgr: object, Xg_out: object | None) -> object:
    if Xg_out is not None:
        return Xg_out
    template = getattr(dm_mgr, "global_vec_template", None)
    if template is None:
        create_global_vec = getattr(dm_mgr.dm_composite, "createGlobalVec", None)
        if not callable(create_global_vec):
            raise LocalStateError("dm_manager must expose global_vec_template or dm_composite.createGlobalVec")
        return create_global_vec()
    return _create_global_vec_like(template)


def local_to_global_insert(
    dm_mgr: object,
    Xl_liq: object,
    Xl_if: object,
    Xl_gas: object,
    *,
    Xg_out: object | None = None,
) -> object:
    """Write local vectors into one composite global Vec with INSERT semantics."""

    Xg = _prepare_output_global_vec(dm_mgr, Xg_out)
    set_fn = getattr(Xg, "set", None)
    if callable(set_fn):
        set_fn(0.0)

    _composite_gather(
        dm_mgr.dm_composite,
        Xg,
        _insert_mode(dm_mgr, additive=False),
        (Xl_liq, Xl_if, Xl_gas),
    )

    assembly_begin = getattr(Xg, "assemblyBegin", None)
    assembly_end = getattr(Xg, "assemblyEnd", None)
    if callable(assembly_begin) and callable(assembly_end):
        assembly_begin()
        assembly_end()
    return Xg


def local_to_global_add(
    dm_mgr: object,
    Fl_liq: object,
    Fl_if: object,
    Fl_gas: object,
    *,
    Fg_out: object | None = None,
) -> object:
    """Accumulate local vectors into one composite global Vec with ADD semantics."""

    Fg = _prepare_output_global_vec(dm_mgr, Fg_out)
    _composite_gather(
        dm_mgr.dm_composite,
        Fg,
        _insert_mode(dm_mgr, additive=True),
        (Fl_liq, Fl_if, Fl_gas),
    )

    assembly_begin = getattr(Fg, "assemblyBegin", None)
    assembly_end = getattr(Fg, "assemblyEnd", None)
    if callable(assembly_begin) and callable(assembly_end):
        assembly_begin()
        assembly_end()
    return Fg


def build_local_state_hooks() -> dict[str, object]:
    """Return the distributed-safe builder hooks for solver/timestepper injection."""

    return {
        "distributed_state_vec_builder": distributed_state_vec_builder,
        "distributed_residual_vec_builder": distributed_residual_vec_builder,
        "state_init_from_previous_inner_result": state_init_from_previous_inner_result,
        "state_guess_from_previous_inner_result": state_guess_from_previous_inner_result,
    }


__all__ = [
    "LocalStateError",
    "LocalStateView",
    "LocalSubdomainView",
    "begin_local_state_view",
    "build_local_state_hooks",
    "create_distributed_vec_from_layout_array",
    "distributed_residual_vec_builder",
    "distributed_state_vec_builder",
    "end_local_state_view",
    "gather_layout_vector_from_distributed_vec",
    "local_state_view",
    "local_to_global_add",
    "local_to_global_insert",
    "recover_state_from_distributed_vec",
    "state_init_from_previous_inner_result",
    "state_guess_from_previous_inner_result",
]
