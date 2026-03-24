from __future__ import annotations

"""Distributed DM construction and metadata export for the PETSc inner solve."""

from dataclasses import dataclass

import numpy as np

from parallel.mpi_bootstrap import bootstrap_mpi_before_petsc, get_comm_world, get_rank_size


class DMManagerError(RuntimeError):
    """Raised when DM construction or distributed metadata extraction fails."""


@dataclass(slots=True, frozen=True)
class CellOwnership:
    """Owned cell subsets and interface-row ownership for one rank."""

    owned_liq_cells: np.ndarray
    owned_gas_cells: np.ndarray
    interface_owner_active: bool


@dataclass(slots=True)
class DMManager:
    """All PETSc DM objects and layout-aligned distributed metadata for one solve."""

    PETSc: object
    comm: object
    rank: int
    size: int

    dm_liq: object
    dm_if: object
    dm_gas: object
    dm_composite: object

    global_vec_template: object

    interface_owner_rank: int
    interface_dm_kind: str
    interface_dm_replicated: bool

    n_liq_cells: int
    n_gas_cells: int
    n_if_unknowns: int
    liq_dof: int
    gas_dof: int

    liq_owned_cells: np.ndarray
    gas_owned_cells: np.ndarray

    liq_corners: tuple[int, int]
    gas_corners: tuple[int, int]
    liq_ghost_corners: tuple[int, int]
    gas_ghost_corners: tuple[int, int]

    ownership_range: tuple[int, int]
    ownership_ranges: np.ndarray

    layout_to_petsc: np.ndarray
    petsc_to_layout: np.ndarray

    residual_ownership: CellOwnership
    jacobian_ownership: CellOwnership


def _read_rank_size(comm: object) -> tuple[int, int]:
    try:
        return get_rank_size(comm)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        raise DMManagerError(f"failed to read communicator rank/size: {exc}") from exc


def _get_petsc_provider(PETSc: object | None = None):
    if PETSc is not None:
        return PETSc
    bootstrap_mpi_before_petsc()
    try:
        from petsc4py import PETSc as petsc_provider
    except Exception as exc:  # pragma: no cover - exercised only with petsc4py installed
        raise DMManagerError(f"failed to import petsc4py.PETSc: {exc}") from exc
    return petsc_provider


def _get_comm(PETSc: object, comm: object | None = None):
    target = comm
    if target is None:
        try:
            target = get_comm_world()
        except Exception:
            target = getattr(PETSc, "COMM_WORLD", None)
    if target is None:
        raise DMManagerError("unable to resolve communicator from bootstrap or PETSc provider")
    to_mpi4py = getattr(target, "tompi4py", None)
    if callable(to_mpi4py):
        try:
            converted = to_mpi4py()
        except Exception as exc:
            raise DMManagerError(f"failed to convert communicator to mpi4py comm: {exc}") from exc
        if converted is None:
            raise DMManagerError("communicator.tompi4py() returned None")
        return converted
    return target


def _build_dmda(*, PETSc: object, comm: object, n_cells: int, dof: int):
    factory = getattr(PETSc, "DMDA", None)
    if factory is None:
        raise DMManagerError("PETSc provider does not expose DMDA")
    dm = factory()
    attempts = (
        ((), {"sizes": (int(n_cells),), "dof": int(dof), "stencil_width": 1, "comm": comm}),
        (([int(n_cells)],), {"dof": int(dof), "stencil_width": 1, "comm": comm}),
        ((), {"dim": 1, "sizes": (int(n_cells),), "dof": int(dof), "stencil_width": 1, "comm": comm}),
    )
    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return dm.create(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise DMManagerError(f"failed to create 1D DMDA: {last_error}")


def _build_liq_dmda(*, PETSc: object, comm: object, n_cells: int, dof: int):
    return _build_dmda(PETSc=PETSc, comm=comm, n_cells=n_cells, dof=dof)


def _build_gas_dmda(*, PETSc: object, comm: object, n_cells: int, dof: int):
    return _build_dmda(PETSc=PETSc, comm=comm, n_cells=n_cells, dof=dof)


def _try_build_interface_dm_redundant_explicit(
    *,
    PETSc: object,
    comm: object,
    n_if_unknowns: int,
    owner_rank: int,
):
    factory = getattr(PETSc, "DMRedundant", None)
    if factory is not None:
        dm = factory()
        attempts = (
            ((), {"comm": comm, "n": int(n_if_unknowns), "rank": int(owner_rank)}),
            ((int(n_if_unknowns), int(owner_rank)), {"comm": comm}),
            ((comm, int(owner_rank), int(n_if_unknowns)), {}),
        )
        last_error: Exception | None = None
        for args, kwargs in attempts:
            try:
                return dm.create(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
        raise DMManagerError(f"failed to create interface DMRedundant: {last_error}") from last_error

    generic_factory = getattr(PETSc, "DM", None)
    if generic_factory is None:
        return None

    generic_dm = generic_factory()
    create_redundant = getattr(generic_dm, "createRedundant", None)
    if not callable(create_redundant):
        return None

    attempts = (
        ((), {"comm": comm, "n": int(n_if_unknowns), "rank": int(owner_rank)}),
        ((int(n_if_unknowns), int(owner_rank)), {"comm": comm}),
        ((comm, int(owner_rank), int(n_if_unknowns)), {}),
        ((comm, int(n_if_unknowns), int(owner_rank)), {}),
    )
    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return create_redundant(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise DMManagerError(f"failed to create interface DM via generic createRedundant: {last_error}") from last_error


def _probe_generic_redundant_type(*, PETSc: object, comm: object) -> bool:
    generic_factory = getattr(PETSc, "DM", None)
    if generic_factory is None:
        return False
    dm = generic_factory()
    create = getattr(dm, "create", None)
    target = dm
    if callable(create):
        try:
            created = create(comm=comm)
        except TypeError:
            try:
                created = create()
            except Exception:
                created = None
        except Exception:
            created = None
        if created is not None:
            target = created
    set_type = getattr(target, "setType", None)
    if not callable(set_type):
        return False
    dm_type = getattr(getattr(PETSc, "DM", None), "Type", None)
    redundant_type = getattr(dm_type, "REDUNDANT", None)
    if redundant_type is None:
        return False
    try:
        set_type(redundant_type)
    except Exception:
        return False
    return True


def _build_interface_dmda_single_rank_fallback(*, PETSc: object, comm: object, n_if_unknowns: int):
    factory = getattr(PETSc, "DMDA", None)
    if factory is None:
        raise DMManagerError("single-rank interface fallback requires PETSc.DMDA")
    dm = factory()
    attempts = (
        ((), {"sizes": (1,), "dof": int(n_if_unknowns), "stencil_width": 0, "comm": comm}),
        (([1],), {"dof": int(n_if_unknowns), "stencil_width": 0, "comm": comm}),
        ((), {"dim": 1, "sizes": (1,), "dof": int(n_if_unknowns), "stencil_width": 0, "comm": comm}),
    )
    last_error: Exception | None = None
    for args, kwargs in attempts:
        try:
            return dm.create(*args, **kwargs)
        except TypeError as exc:
            last_error = exc
    raise DMManagerError(
        f"failed to construct single-rank DMDA fallback for interface block: {last_error}"
    ) from last_error


def _build_interface_dm(
    *,
    PETSc: object,
    comm: object,
    n_if_unknowns: int,
    owner_rank: int,
    comm_size: int,
) -> tuple[object, str, bool]:
    explicit = _try_build_interface_dm_redundant_explicit(
        PETSc=PETSc,
        comm=comm,
        n_if_unknowns=n_if_unknowns,
        owner_rank=owner_rank,
    )
    if explicit is not None:
        return explicit, "redundant", True

    redundant_type_supported = _probe_generic_redundant_type(PETSc=PETSc, comm=comm)
    if int(comm_size) == 1:
        fallback = _build_interface_dmda_single_rank_fallback(
            PETSc=PETSc,
            comm=comm,
            n_if_unknowns=n_if_unknowns,
        )
        return fallback, "single_rank_dmda_fallback", False

    if redundant_type_supported:
        raise DMManagerError(
            "multi-rank interface DM requires a usable Python construction/configuration "
            "path for DMREDUNDANT; current petsc4py build recognizes redundant type but "
            "does not expose owner-rank/global-size configuration APIs"
        )
    raise DMManagerError(
        "multi-rank interface DM requires usable DMREDUNDANT support; "
        "current PETSc/petsc4py build does not expose it"
    )


def _compose_dm(*, PETSc: object, comm: object, dm_liq: object, dm_if: object, dm_gas: object):
    factory = getattr(PETSc, "DMComposite", None)
    if factory is None:
        raise DMManagerError("PETSc provider does not expose DMComposite")
    composite = factory()
    try:
        dm = composite.create(comm=comm)
    except TypeError:
        dm = composite.create()
    for child in (dm_liq, dm_if, dm_gas):
        add_dm = getattr(dm, "addDM", None)
        if not callable(add_dm):
            raise DMManagerError("DMComposite object does not expose addDM")
        add_dm(child)
    return dm


def _parse_dmda_1d_corners(raw: object, *, label: str) -> tuple[int, int]:
    """Normalize petsc4py DMDA corner metadata to the 1D ``(xs, xm)`` form."""

    try:
        seq = tuple(raw)
    except TypeError as exc:
        raise DMManagerError(f"unsupported DMDA {label} format: {raw!r}") from exc

    if len(seq) >= 2 and not isinstance(seq[0], (tuple, list)) and not isinstance(seq[1], (tuple, list)):
        return int(seq[0]), int(seq[1])

    if len(seq) >= 2 and isinstance(seq[0], (tuple, list)) and isinstance(seq[1], (tuple, list)):
        starts = tuple(seq[0])
        widths = tuple(seq[1])
        if len(starts) < 1 or len(widths) < 1:
            raise DMManagerError(f"{label} must contain at least one dimension")
        return int(starts[0]), int(widths[0])

    raise DMManagerError(f"unsupported DMDA {label} format: {raw!r}")


def _extract_dmda_cell_ranges(dm) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    get_corners = getattr(dm, "getCorners", None)
    get_ghost = getattr(dm, "getGhostCorners", None)
    if not callable(get_corners) or not callable(get_ghost):
        raise DMManagerError("DMDA object does not expose getCorners/getGhostCorners")

    xs, xm = _parse_dmda_1d_corners(get_corners(), label="corners")
    gxs, gxm = _parse_dmda_1d_corners(get_ghost(), label="ghost corners")
    owned = np.arange(xs, xs + xm, dtype=np.int64)
    return owned, (xs, xm), (gxs, gxm)


def _extract_global_ownership_ranges(vec, comm: object) -> tuple[tuple[int, int], np.ndarray]:
    get_ownership = getattr(vec, "getOwnershipRange", None)
    if not callable(get_ownership):
        raise DMManagerError("global_vec_template does not expose getOwnershipRange")
    local = tuple(int(v) for v in get_ownership())
    if len(local) != 2:
        raise DMManagerError("getOwnershipRange must return a 2-tuple")

    _, size = _read_rank_size(comm)
    allgather = getattr(comm, "allgather", None)
    if callable(allgather):
        gathered = allgather(local)
    elif size == 1:
        gathered = [local]
    else:
        raise DMManagerError("multi-rank communicator must expose allgather for ownership export")
    ranges = np.asarray(gathered, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2:
        raise DMManagerError("ownership_ranges must have shape (size, 2)")
    return (int(local[0]), int(local[1])), ranges


def _build_identity_layout_mapping(layout: object, total_size: int) -> tuple[np.ndarray, np.ndarray]:
    if int(getattr(layout, "total_size")) != int(total_size):
        raise DMManagerError("layout.total_size does not match DMComposite global size")
    perm = np.arange(int(total_size), dtype=np.int64)
    return perm.copy(), perm.copy()


def _validate_layout_matches_dm_blocks(
    *,
    layout: object,
    n_liq_cells: int,
    liq_dof: int,
    n_if_unknowns: int,
    n_gas_cells: int,
    gas_dof: int,
    total_size: int,
) -> None:
    liq_size = int(n_liq_cells) * int(liq_dof)
    if_size = int(n_if_unknowns)
    gas_size = int(n_gas_cells) * int(gas_dof)

    liq_block = getattr(layout, "liq_block")
    if_block = getattr(layout, "if_block")
    gas_block = getattr(layout, "gas_block")

    if int(liq_block.stop) - int(liq_block.start) != liq_size:
        raise DMManagerError("liquid DM block size does not match layout.liq_block")
    if int(if_block.stop) - int(if_block.start) != if_size:
        raise DMManagerError("interface DM block size does not match layout.if_block")
    if int(gas_block.stop) - int(gas_block.start) != gas_size:
        raise DMManagerError("gas DM block size does not match layout.gas_block")
    if int(getattr(layout, "total_size")) != liq_size + if_size + gas_size:
        raise DMManagerError("layout block sizes do not sum to layout.total_size")
    if int(total_size) != int(getattr(layout, "total_size")):
        raise DMManagerError("DMComposite global size does not match layout.total_size")


def build_iteration_ownership(dm_mgr: DMManager) -> CellOwnership:
    """Build the owned-cell view used by residual/Jacobian assembly."""

    return CellOwnership(
        owned_liq_cells=np.asarray(dm_mgr.liq_owned_cells, dtype=np.int64).copy(),
        owned_gas_cells=np.asarray(dm_mgr.gas_owned_cells, dtype=np.int64).copy(),
        interface_owner_active=bool(dm_mgr.rank == dm_mgr.interface_owner_rank),
    )


def validate_dm_manager(dm_mgr: DMManager, layout: object) -> None:
    """Validate block sizes, ownership metadata, and mapping consistency."""

    if dm_mgr.size <= 0:
        raise DMManagerError("communicator size must be >= 1")
    if dm_mgr.rank < 0 or dm_mgr.rank >= dm_mgr.size:
        raise DMManagerError("rank must satisfy 0 <= rank < size")
    if dm_mgr.interface_owner_rank < 0 or dm_mgr.interface_owner_rank >= dm_mgr.size:
        raise DMManagerError("interface_owner_rank must satisfy 0 <= owner < size")
    if dm_mgr.interface_dm_kind not in {"redundant", "single_rank_dmda_fallback"}:
        raise DMManagerError("interface_dm_kind must be one of {'redundant', 'single_rank_dmda_fallback'}")
    if dm_mgr.interface_dm_kind == "single_rank_dmda_fallback":
        if dm_mgr.size != 1:
            raise DMManagerError("single_rank_dmda_fallback is only valid when communicator size == 1")
        if dm_mgr.interface_dm_replicated:
            raise DMManagerError("single-rank interface DMDA fallback must not report replicated interface semantics")
    if dm_mgr.interface_dm_kind == "redundant" and not dm_mgr.interface_dm_replicated:
        raise DMManagerError("redundant interface DM must report replicated semantics")

    total_size = int(dm_mgr.global_vec_template.getSize())
    _validate_layout_matches_dm_blocks(
        layout=layout,
        n_liq_cells=dm_mgr.n_liq_cells,
        liq_dof=dm_mgr.liq_dof,
        n_if_unknowns=dm_mgr.n_if_unknowns,
        n_gas_cells=dm_mgr.n_gas_cells,
        gas_dof=dm_mgr.gas_dof,
        total_size=total_size,
    )

    if dm_mgr.ownership_ranges.shape != (dm_mgr.size, 2):
        raise DMManagerError("ownership_ranges must have shape (size, 2)")
    if dm_mgr.ownership_range[0] < 0 or dm_mgr.ownership_range[1] < dm_mgr.ownership_range[0]:
        raise DMManagerError("ownership_range must be a valid half-open interval")
    if int(dm_mgr.ownership_ranges[0, 0]) != 0:
        raise DMManagerError("ownership_ranges must start at global row 0")
    if int(dm_mgr.ownership_ranges[-1, 1]) != total_size:
        raise DMManagerError("ownership_ranges must end at the composite global size")
    if not np.all(dm_mgr.ownership_ranges[1:, 0] == dm_mgr.ownership_ranges[:-1, 1]):
        raise DMManagerError("ownership_ranges must be contiguous across ranks")
    if tuple(int(v) for v in dm_mgr.ownership_ranges[dm_mgr.rank]) != tuple(int(v) for v in dm_mgr.ownership_range):
        raise DMManagerError("ownership_ranges[rank] must equal ownership_range for the local rank")

    liq_owned = np.asarray(dm_mgr.liq_owned_cells, dtype=np.int64)
    gas_owned = np.asarray(dm_mgr.gas_owned_cells, dtype=np.int64)
    if liq_owned.ndim != 1 or gas_owned.ndim != 1:
        raise DMManagerError("owned cell arrays must be 1D")
    if liq_owned.size > 0 and (np.any(liq_owned < 0) or np.any(liq_owned >= dm_mgr.n_liq_cells)):
        raise DMManagerError("liq_owned_cells contains out-of-range cell indices")
    if gas_owned.size > 0 and (np.any(gas_owned < 0) or np.any(gas_owned >= dm_mgr.n_gas_cells)):
        raise DMManagerError("gas_owned_cells contains out-of-range cell indices")

    if dm_mgr.layout_to_petsc.shape != (total_size,) or dm_mgr.petsc_to_layout.shape != (total_size,):
        raise DMManagerError("layout/PETSc mappings must have length equal to the global size")
    identity = np.arange(total_size, dtype=np.int64)
    if not np.array_equal(dm_mgr.layout_to_petsc, identity):
        raise DMManagerError("Phase 6 DM manager currently requires identity layout_to_petsc")
    if not np.array_equal(dm_mgr.petsc_to_layout, identity):
        raise DMManagerError("Phase 6 DM manager currently requires identity petsc_to_layout")


def build_dm_manager(
    *,
    layout: object,
    PETSc: object | None = None,
    comm: object | None = None,
    interface_owner_rank: int = 0,
) -> DMManager:
    """Build the PETSc DMComposite and all distributed metadata for one layout."""

    petsc = _get_petsc_provider(PETSc)
    communicator = _get_comm(petsc, comm)
    rank, size = _read_rank_size(communicator)
    if int(interface_owner_rank) < 0 or int(interface_owner_rank) >= int(size):
        raise DMManagerError("interface_owner_rank must satisfy 0 <= owner < communicator size")

    dm_liq = _build_liq_dmda(
        PETSc=petsc,
        comm=communicator,
        n_cells=int(getattr(layout, "n_liq_cells")),
        dof=int(getattr(layout, "liq_cell_width")),
    )
    dm_if, interface_dm_kind, interface_dm_replicated = _build_interface_dm(
        PETSc=petsc,
        comm=communicator,
        n_if_unknowns=int(getattr(layout, "n_if_unknowns")),
        owner_rank=int(interface_owner_rank),
        comm_size=int(size),
    )
    dm_gas = _build_gas_dmda(
        PETSc=petsc,
        comm=communicator,
        n_cells=int(getattr(layout, "n_gas_cells")),
        dof=int(getattr(layout, "gas_cell_width")),
    )
    dm_composite = _compose_dm(
        PETSc=petsc,
        comm=communicator,
        dm_liq=dm_liq,
        dm_if=dm_if,
        dm_gas=dm_gas,
    )

    create_global_vec = getattr(dm_composite, "createGlobalVec", None)
    if not callable(create_global_vec):
        raise DMManagerError("DMComposite does not expose createGlobalVec")
    global_vec_template = create_global_vec()

    liq_owned, liq_corners, liq_ghost = _extract_dmda_cell_ranges(dm_liq)
    gas_owned, gas_corners, gas_ghost = _extract_dmda_cell_ranges(dm_gas)
    ownership_range, ownership_ranges = _extract_global_ownership_ranges(global_vec_template, communicator)
    layout_to_petsc, petsc_to_layout = _build_identity_layout_mapping(
        layout,
        int(global_vec_template.getSize()),
    )

    own = CellOwnership(
        owned_liq_cells=liq_owned.copy(),
        owned_gas_cells=gas_owned.copy(),
        interface_owner_active=bool(rank == int(interface_owner_rank)),
    )
    dm_mgr = DMManager(
        PETSc=petsc,
        comm=communicator,
        rank=int(rank),
        size=int(size),
        dm_liq=dm_liq,
        dm_if=dm_if,
        dm_gas=dm_gas,
        dm_composite=dm_composite,
        global_vec_template=global_vec_template,
        interface_owner_rank=int(interface_owner_rank),
        interface_dm_kind=str(interface_dm_kind),
        interface_dm_replicated=bool(interface_dm_replicated),
        n_liq_cells=int(getattr(layout, "n_liq_cells")),
        n_gas_cells=int(getattr(layout, "n_gas_cells")),
        n_if_unknowns=int(getattr(layout, "n_if_unknowns")),
        liq_dof=int(getattr(layout, "liq_cell_width")),
        gas_dof=int(getattr(layout, "gas_cell_width")),
        liq_owned_cells=liq_owned,
        gas_owned_cells=gas_owned,
        liq_corners=liq_corners,
        gas_corners=gas_corners,
        liq_ghost_corners=liq_ghost,
        gas_ghost_corners=gas_ghost,
        ownership_range=ownership_range,
        ownership_ranges=ownership_ranges,
        layout_to_petsc=layout_to_petsc,
        petsc_to_layout=petsc_to_layout,
        residual_ownership=own,
        jacobian_ownership=own,
    )
    validate_dm_manager(dm_mgr, layout)
    return dm_mgr


def build_parallel_handles(dm_mgr: DMManager) -> dict[str, object]:
    """Export solver-facing parallel metadata using existing key contracts."""

    return {
        "PETSc": dm_mgr.PETSc,
        "comm": dm_mgr.comm,
        "rank": dm_mgr.rank,
        "size": dm_mgr.size,
        "dm_manager": dm_mgr,
        "dm_composite": dm_mgr.dm_composite,
        "dm": dm_mgr.dm_composite,
        "global_vec_template": dm_mgr.global_vec_template,
        "ownership_range": dm_mgr.ownership_range,
        "ownership_ranges": dm_mgr.ownership_ranges,
        "layout_to_petsc": dm_mgr.layout_to_petsc,
        "petsc_to_layout": dm_mgr.petsc_to_layout,
        "interface_owner_rank": dm_mgr.interface_owner_rank,
        "interface_dm_kind": dm_mgr.interface_dm_kind,
        "interface_dm_replicated": dm_mgr.interface_dm_replicated,
    }


__all__ = [
    "CellOwnership",
    "DMManager",
    "DMManagerError",
    "build_dm_manager",
    "build_iteration_ownership",
    "build_parallel_handles",
    "validate_dm_manager",
]
