from __future__ import annotations

"""MPI bootstrap helpers used before importing petsc4py."""


_BOOTSTRAPPED: bool = False
_MPI_MODULE: object | None = None
_COMM_WORLD: object | None = None
_BOOTSTRAP_INFO: dict[str, object] | None = None


class MPIBootstrapError(RuntimeError):
    """Raised when MPI bootstrap fails or communicator access is invalid."""


def _read_rank_size_from_comm(comm: object) -> tuple[int, int]:
    """Read rank/size from a communicator or fake communicator test double."""

    if comm is None:
        raise MPIBootstrapError("communicator is None")

    rank = getattr(comm, "rank", None)
    size = getattr(comm, "size", None)
    if rank is not None and size is not None:
        return int(rank), int(size)

    get_rank = getattr(comm, "Get_rank", None)
    get_size = getattr(comm, "Get_size", None)
    if callable(get_rank) and callable(get_size):
        return int(get_rank()), int(get_size())

    raise MPIBootstrapError("communicator does not expose rank/size accessors")


def _build_bootstrap_info(MPI: object, comm: object) -> dict[str, object]:
    """Build the cached bootstrap diagnostics payload."""

    rank, size = _read_rank_size_from_comm(comm)
    is_initialized = getattr(MPI, "Is_initialized", None)
    initialized = bool(is_initialized()) if callable(is_initialized) else True
    return {
        "initialized": initialized,
        "rank": rank,
        "size": size,
        "comm": comm,
        "comm_name": "COMM_WORLD",
    }


def bootstrap_mpi_before_petsc() -> None:
    """Cache mpi4py communicator access before importing petsc4py."""

    global _BOOTSTRAPPED, _MPI_MODULE, _COMM_WORLD, _BOOTSTRAP_INFO

    if _BOOTSTRAPPED:
        return

    try:
        from mpi4py import MPI
    except Exception as exc:
        raise MPIBootstrapError(f"failed to import mpi4py: {exc}") from exc

    try:
        is_initialized = getattr(MPI, "Is_initialized", None)
        if callable(is_initialized) and not bool(is_initialized()):
            raise MPIBootstrapError(
                "mpi4py imported successfully but MPI is not initialized; unexpected runtime state"
            )

        comm = getattr(MPI, "COMM_WORLD", None)
        if comm is None:
            raise MPIBootstrapError("mpi4py.MPI does not expose COMM_WORLD")

        info = _build_bootstrap_info(MPI, comm)
    except Exception as exc:
        if isinstance(exc, MPIBootstrapError):
            raise
        raise MPIBootstrapError(
            f"failed to bootstrap MPI before PETSc import: {exc}"
        ) from exc

    _MPI_MODULE = MPI
    _COMM_WORLD = comm
    _BOOTSTRAP_INFO = info
    _BOOTSTRAPPED = True


def get_mpi_module():
    """Return the cached mpi4py.MPI module after bootstrap."""

    bootstrap_mpi_before_petsc()
    if _MPI_MODULE is None:
        raise MPIBootstrapError("MPI module cache is not available after bootstrap")
    return _MPI_MODULE


def get_comm_world():
    """Return the cached MPI.COMM_WORLD communicator."""

    bootstrap_mpi_before_petsc()
    if _COMM_WORLD is None:
        raise MPIBootstrapError("COMM_WORLD cache is not available after bootstrap")
    return _COMM_WORLD


def get_rank_size(comm: object | None = None) -> tuple[int, int]:
    """Return communicator rank and size for the default or supplied comm."""

    target = get_comm_world() if comm is None else comm
    return _read_rank_size_from_comm(target)


def is_parallel_active(comm: object | None = None) -> bool:
    """Return True when the communicator size is greater than one."""

    _, size = get_rank_size(comm)
    return size > 1


def get_bootstrap_info() -> dict[str, object]:
    """Return a defensive copy of cached bootstrap diagnostics."""

    bootstrap_mpi_before_petsc()
    if _BOOTSTRAP_INFO is None:
        raise MPIBootstrapError("bootstrap diagnostics are not available")
    return dict(_BOOTSTRAP_INFO)


def reset_bootstrap_state_for_tests() -> None:
    """Reset Python-side bootstrap caches. Production code must not call this."""

    global _BOOTSTRAPPED, _MPI_MODULE, _COMM_WORLD, _BOOTSTRAP_INFO

    _BOOTSTRAPPED = False
    _MPI_MODULE = None
    _COMM_WORLD = None
    _BOOTSTRAP_INFO = None


__all__ = [
    "MPIBootstrapError",
    "bootstrap_mpi_before_petsc",
    "get_mpi_module",
    "get_comm_world",
    "get_rank_size",
    "is_parallel_active",
    "get_bootstrap_info",
    "reset_bootstrap_state_for_tests",
]
