from __future__ import annotations

import numpy as np

from .types import (
    ConservativeContents,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    RecoveryConfig,
    SpeciesMaps,
    State,
    StateTransferRecord,
)
from .state_recovery import GasThermoProtocol, LiquidThermoProtocol


def _missing_recovery(*args, **kwargs):
    raise NotImplementedError("core.state_recovery.recover_state_from_contents is not available")


try:  # pragma: no cover - exercised indirectly once state_recovery exists
    from .state_recovery import recover_state_from_contents
except ImportError:  # pragma: no cover
    recover_state_from_contents = _missing_recovery


OVERLAP_RTOL = 1.0e-12


class RemapError(ValueError):
    """Raised when conservative transfer between two geometries cannot be completed safely."""


def _gas_local_slice(mesh: Mesh1D, region_slice: slice) -> slice:
    gas_start = mesh.region_slices.gas_all.start
    return slice(region_slice.start - gas_start, region_slice.stop - gas_start)


def _faces_for_region(mesh: Mesh1D, region_slice: slice) -> np.ndarray:
    return mesh.r_faces[region_slice.start : region_slice.stop + 1]


def _build_old_conservative_contents(old_state: State, old_mesh: Mesh1D) -> ConservativeContents:
    required = (
        ("rho_l", old_state.rho_l),
        ("rho_g", old_state.rho_g),
        ("hl", old_state.hl),
        ("hg", old_state.hg),
    )
    missing = [name for name, value in required if value is None]
    if missing:
        missing_names = ", ".join(missing)
        raise RemapError(f"old_state is missing required derived fields for remap: {missing_names}")

    liq_volumes = old_mesh.volumes[old_mesh.region_slices.liq]
    gas_volumes = old_mesh.volumes[old_mesh.region_slices.gas_all]

    mass_l = old_state.rho_l * liq_volumes
    species_mass_l = old_state.rho_l[:, None] * old_state.Yl_full * liq_volumes[:, None]
    enthalpy_l = old_state.rho_l * old_state.hl * liq_volumes

    mass_g = old_state.rho_g * gas_volumes
    species_mass_g = old_state.rho_g[:, None] * old_state.Yg_full * gas_volumes[:, None]
    enthalpy_g = old_state.rho_g * old_state.hg * gas_volumes

    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )


def _compute_overlap_1d(old_left: float, old_right: float, new_left: float, new_right: float) -> float:
    if old_right < old_left:
        raise RemapError("old interval must satisfy left <= right")
    if new_right < new_left:
        raise RemapError("new interval must satisfy left <= right")
    return max(0.0, min(old_right, new_right) - max(old_left, new_left))


def _spherical_shell_volume(r_left: float, r_right: float) -> float:
    if r_right < r_left:
        raise RemapError("spherical shell requires r_right >= r_left")
    return float((4.0 * np.pi / 3.0) * (r_right**3 - r_left**3))


def _compute_overlap_matrix_spherical(old_faces: np.ndarray, new_faces: np.ndarray) -> np.ndarray:
    old_faces = np.asarray(old_faces, dtype=np.float64)
    new_faces = np.asarray(new_faces, dtype=np.float64)
    if old_faces.ndim != 1 or new_faces.ndim != 1:
        raise RemapError("old_faces and new_faces must be one-dimensional")
    if old_faces.size < 2 or new_faces.size < 2:
        raise RemapError("old_faces and new_faces must contain at least two entries")
    if np.any(np.diff(old_faces) <= 0.0):
        raise RemapError("old_faces must be strictly increasing")
    if np.any(np.diff(new_faces) <= 0.0):
        raise RemapError("new_faces must be strictly increasing")

    n_old = old_faces.size - 1
    n_new = new_faces.size - 1
    overlap = np.zeros((n_old, n_new), dtype=np.float64)
    for i_old in range(n_old):
        for j_new in range(n_new):
            left = max(old_faces[i_old], new_faces[j_new])
            right = min(old_faces[i_old + 1], new_faces[j_new + 1])
            if right <= left:
                continue
            overlap[i_old, j_new] = _spherical_shell_volume(left, right)
    return overlap


def _remap_phase_contents_from_overlap(
    *,
    old_mass: np.ndarray,
    old_species_mass: np.ndarray,
    old_enthalpy: np.ndarray,
    old_cell_volumes: np.ndarray,
    overlap_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_mass = np.asarray(old_mass, dtype=np.float64)
    old_species_mass = np.asarray(old_species_mass, dtype=np.float64)
    old_enthalpy = np.asarray(old_enthalpy, dtype=np.float64)
    old_cell_volumes = np.asarray(old_cell_volumes, dtype=np.float64)
    overlap_matrix = np.asarray(overlap_matrix, dtype=np.float64)

    n_old, _n_new = overlap_matrix.shape
    if old_mass.shape != (n_old,):
        raise RemapError("old_mass shape must match overlap old-cell dimension")
    if old_enthalpy.shape != (n_old,):
        raise RemapError("old_enthalpy shape must match overlap old-cell dimension")
    if old_cell_volumes.shape != (n_old,):
        raise RemapError("old_cell_volumes shape must match overlap old-cell dimension")
    if old_species_mass.shape[0] != n_old:
        raise RemapError("old_species_mass row count must match overlap old-cell dimension")
    if np.any(old_cell_volumes <= 0.0):
        raise RemapError("old_cell_volumes must be strictly positive")
    if np.any(overlap_matrix < 0.0):
        raise RemapError("overlap_matrix must be non-negative")

    mass_density = old_mass / old_cell_volumes
    species_mass_density = old_species_mass / old_cell_volumes[:, None]
    enthalpy_density = old_enthalpy / old_cell_volumes

    new_mass = overlap_matrix.T @ mass_density
    new_species_mass = overlap_matrix.T @ species_mass_density
    new_enthalpy = overlap_matrix.T @ enthalpy_density
    return new_mass, new_species_mass, new_enthalpy


def _compute_uncovered_new_volume(new_faces: np.ndarray, overlap_matrix: np.ndarray) -> np.ndarray:
    new_faces = np.asarray(new_faces, dtype=np.float64)
    overlap_matrix = np.asarray(overlap_matrix, dtype=np.float64)
    new_volumes = np.array(
        [_spherical_shell_volume(new_faces[j], new_faces[j + 1]) for j in range(new_faces.size - 1)],
        dtype=np.float64,
    )
    covered = np.sum(overlap_matrix, axis=0)
    uncovered = new_volumes - covered

    scale = max(float(np.max(new_volumes)), 1.0e-30)
    tol = OVERLAP_RTOL * scale
    if np.any(uncovered < -tol):
        raise RemapError("computed uncovered volume is negative beyond tolerance")
    uncovered[uncovered < 0.0] = 0.0
    return uncovered


def _complete_newly_exposed_subvolume_liquid(
    *,
    new_mass: np.ndarray,
    new_species_mass: np.ndarray,
    new_enthalpy: np.ndarray,
    uncovered_volume: np.ndarray,
    reference_rho: float,
    reference_y_full: np.ndarray,
    reference_h: float,
    tol: float,
) -> None:
    if reference_rho <= 0.0 or not np.isfinite(reference_rho):
        raise RemapError("liquid completion reference_rho must be finite and > 0")
    if not np.isfinite(reference_h):
        raise RemapError("liquid completion reference_h must be finite")
    for j_new, vol in enumerate(uncovered_volume):
        if vol <= tol:
            continue
        new_mass[j_new] += reference_rho * vol
        new_species_mass[j_new, :] += reference_rho * reference_y_full * vol
        new_enthalpy[j_new] += reference_rho * reference_h * vol


def _complete_newly_exposed_subvolume_gas_near(
    *,
    new_mass: np.ndarray,
    new_species_mass: np.ndarray,
    new_enthalpy: np.ndarray,
    uncovered_volume: np.ndarray,
    reference_rho: float,
    reference_y_full: np.ndarray,
    reference_h: float,
    tol: float,
) -> None:
    if reference_rho <= 0.0 or not np.isfinite(reference_rho):
        raise RemapError("gas completion reference_rho must be finite and > 0")
    if not np.isfinite(reference_h):
        raise RemapError("gas completion reference_h must be finite")
    for j_new, vol in enumerate(uncovered_volume):
        if vol <= tol:
            continue
        new_mass[j_new] += reference_rho * vol
        new_species_mass[j_new, :] += reference_rho * reference_y_full * vol
        new_enthalpy[j_new] += reference_rho * reference_h * vol


def _identity_copy_region3_gas_contents(
    old_contents: ConservativeContents,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    old_far_faces = _faces_for_region(old_mesh, old_mesh.region_slices.gas_far)
    new_far_faces = _faces_for_region(new_mesh, new_mesh.region_slices.gas_far)
    if not np.array_equal(old_far_faces, new_far_faces):
        raise RemapError("gas region-3 identity remap requires identical old/new region-3 faces")

    old_far_local = _gas_local_slice(old_mesh, old_mesh.region_slices.gas_far)
    new_far_local = _gas_local_slice(new_mesh, new_mesh.region_slices.gas_far)
    if (old_far_local.stop - old_far_local.start) != (new_far_local.stop - new_far_local.start):
        raise RemapError("gas region-3 identity remap requires the same old/new region-3 cell count")

    return (
        old_contents.mass_g[old_far_local].copy(),
        old_contents.species_mass_g[old_far_local, :].copy(),
        old_contents.enthalpy_g[old_far_local].copy(),
    )


def _extract_liquid_completion_reference(old_state: State) -> tuple[float, np.ndarray, float]:
    if old_state.rho_l is None or old_state.hl is None:
        raise RemapError("liquid completion requires old_state.rho_l and old_state.hl")
    return float(old_state.rho_l[-1]), np.asarray(old_state.Yl_full[-1, :], dtype=np.float64), float(old_state.hl[-1])


def _extract_gas_near_completion_reference(old_state: State, old_mesh: Mesh1D) -> tuple[float, np.ndarray, float]:
    _ = old_mesh
    if old_state.rho_g is None or old_state.hg is None:
        raise RemapError("gas completion requires old_state.rho_g and old_state.hg")
    return float(old_state.rho_g[0]), np.asarray(old_state.Yg_full[0, :], dtype=np.float64), float(old_state.hg[0])


def _assemble_new_conservative_contents(
    *,
    liquid_part: tuple[np.ndarray, np.ndarray, np.ndarray],
    gas_near_part: tuple[np.ndarray, np.ndarray, np.ndarray],
    gas_far_part: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> ConservativeContents:
    mass_l, species_mass_l, enthalpy_l = liquid_part
    mass_g_near, species_mass_g_near, enthalpy_g_near = gas_near_part
    mass_g_far, species_mass_g_far, enthalpy_g_far = gas_far_part
    return ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=np.concatenate((mass_g_near, mass_g_far)),
        species_mass_g=np.concatenate((species_mass_g_near, species_mass_g_far), axis=0),
        enthalpy_g=np.concatenate((enthalpy_g_near, enthalpy_g_far)),
    )


def _build_transferred_contents(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
) -> ConservativeContents:
    old_contents = _build_old_conservative_contents(old_state, old_mesh)

    old_liq_faces = _faces_for_region(old_mesh, old_mesh.region_slices.liq)
    new_liq_faces = _faces_for_region(new_mesh, new_mesh.region_slices.liq)
    liq_overlap = _compute_overlap_matrix_spherical(old_liq_faces, new_liq_faces)
    liquid_part = _remap_phase_contents_from_overlap(
        old_mass=old_contents.mass_l,
        old_species_mass=old_contents.species_mass_l,
        old_enthalpy=old_contents.enthalpy_l,
        old_cell_volumes=old_mesh.volumes[old_mesh.region_slices.liq],
        overlap_matrix=liq_overlap,
    )
    liq_uncovered = _compute_uncovered_new_volume(new_liq_faces, liq_overlap)
    liq_scale = max(float(np.max(new_mesh.volumes[new_mesh.region_slices.liq])), 1.0e-30)
    liq_rho_ref, liq_y_ref, liq_h_ref = _extract_liquid_completion_reference(old_state)
    _complete_newly_exposed_subvolume_liquid(
        new_mass=liquid_part[0],
        new_species_mass=liquid_part[1],
        new_enthalpy=liquid_part[2],
        uncovered_volume=liq_uncovered,
        reference_rho=liq_rho_ref,
        reference_y_full=liq_y_ref,
        reference_h=liq_h_ref,
        tol=OVERLAP_RTOL * liq_scale,
    )

    old_gas_near_local = _gas_local_slice(old_mesh, old_mesh.region_slices.gas_near)
    old_gas_near_faces = _faces_for_region(old_mesh, old_mesh.region_slices.gas_near)
    new_gas_near_faces = _faces_for_region(new_mesh, new_mesh.region_slices.gas_near)
    gas_near_overlap = _compute_overlap_matrix_spherical(old_gas_near_faces, new_gas_near_faces)
    gas_near_part = _remap_phase_contents_from_overlap(
        old_mass=old_contents.mass_g[old_gas_near_local],
        old_species_mass=old_contents.species_mass_g[old_gas_near_local, :],
        old_enthalpy=old_contents.enthalpy_g[old_gas_near_local],
        old_cell_volumes=old_mesh.volumes[old_mesh.region_slices.gas_near],
        overlap_matrix=gas_near_overlap,
    )
    gas_near_uncovered = _compute_uncovered_new_volume(new_gas_near_faces, gas_near_overlap)
    gas_near_scale = max(float(np.max(new_mesh.volumes[new_mesh.region_slices.gas_near])), 1.0e-30)
    gas_rho_ref, gas_y_ref, gas_h_ref = _extract_gas_near_completion_reference(old_state, old_mesh)
    _complete_newly_exposed_subvolume_gas_near(
        new_mass=gas_near_part[0],
        new_species_mass=gas_near_part[1],
        new_enthalpy=gas_near_part[2],
        uncovered_volume=gas_near_uncovered,
        reference_rho=gas_rho_ref,
        reference_y_full=gas_y_ref,
        reference_h=gas_h_ref,
        tol=OVERLAP_RTOL * gas_near_scale,
    )

    gas_far_part = _identity_copy_region3_gas_contents(old_contents, old_mesh, new_mesh)
    return _assemble_new_conservative_contents(
        liquid_part=liquid_part,
        gas_near_part=gas_near_part,
        gas_far_part=gas_far_part,
    )


def build_old_contents_on_current_geometry(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
) -> ConservativeContents:
    return _build_transferred_contents(old_state=old_state, old_mesh=old_mesh, new_mesh=new_mesh)


def _recover_transferred_state(
    *,
    contents: ConservativeContents,
    new_mesh: Mesh1D,
    geometry_new: GeometryState,
    old_state: State,
    recovery_config: RecoveryConfig,
    species_maps: SpeciesMaps,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
) -> State:
    try:
        return recover_state_from_contents(
            contents=contents,
            mesh=new_mesh,
            species_maps=species_maps,
            recovery_config=recovery_config,
            liquid_thermo=liquid_thermo,
            gas_thermo=gas_thermo,
            interface_seed=old_state.interface,
            time=geometry_new.t,
            state_id=old_state.state_id,
        )
    except NotImplementedError as exc:  # pragma: no cover
        raise RemapError("state recovery is not available for remap finalization") from exc
    except Exception as exc:
        if isinstance(exc, RemapError):
            raise
        raise RemapError(str(exc)) from exc


def _build_transfer_record(
    *,
    contents: ConservativeContents,
    state: State,
    geometry_new: GeometryState,
    new_mesh: Mesh1D,
    source_outer_iter_index: int | None,
    identity_transfer: bool,
) -> StateTransferRecord:
    return StateTransferRecord(
        contents=contents,
        state=state,
        geometry=geometry_new,
        mesh=new_mesh,
        source_outer_iter_index=source_outer_iter_index,
        identity_transfer=identity_transfer,
    )


def _copy_interface_for_identity_transfer(interface: object) -> object:
    if isinstance(interface, InterfaceState):
        return InterfaceState(
            Ts=float(interface.Ts),
            mpp=float(interface.mpp),
            Ys_g_full=np.array(interface.Ys_g_full, dtype=np.float64, copy=True),
            Ys_l_full=np.array(interface.Ys_l_full, dtype=np.float64, copy=True),
        )
    return interface


def _copy_state_for_identity_transfer(old_state: State) -> State:
    return State(
        Tl=np.array(old_state.Tl, dtype=np.float64, copy=True),
        Yl_full=np.array(old_state.Yl_full, dtype=np.float64, copy=True),
        Tg=np.array(old_state.Tg, dtype=np.float64, copy=True),
        Yg_full=np.array(old_state.Yg_full, dtype=np.float64, copy=True),
        interface=_copy_interface_for_identity_transfer(old_state.interface),
        rho_l=None if old_state.rho_l is None else np.array(old_state.rho_l, dtype=np.float64, copy=True),
        rho_g=None if old_state.rho_g is None else np.array(old_state.rho_g, dtype=np.float64, copy=True),
        hl=None if old_state.hl is None else np.array(old_state.hl, dtype=np.float64, copy=True),
        hg=None if old_state.hg is None else np.array(old_state.hg, dtype=np.float64, copy=True),
        Xg_full=None if old_state.Xg_full is None else np.array(old_state.Xg_full, dtype=np.float64, copy=True),
        time=old_state.time,
        state_id=old_state.state_id,
    )


def _build_identity_transfer_record(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    geometry_new: GeometryState,
    source_outer_iter_index: int | None = None,
) -> StateTransferRecord:
    contents = _build_old_conservative_contents(old_state, old_mesh)
    state = _copy_state_for_identity_transfer(old_state)
    return _build_transfer_record(
        contents=contents,
        state=state,
        geometry_new=geometry_new,
        new_mesh=old_mesh,
        source_outer_iter_index=source_outer_iter_index,
        identity_transfer=True,
    )


def build_transfer_state_on_new_geometry(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
    geometry_new: GeometryState,
    recovery_config: RecoveryConfig,
    species_maps: SpeciesMaps,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
    source_outer_iter_index: int | None = None,
) -> StateTransferRecord:
    if old_mesh.same_geometry(new_mesh):
        return _build_identity_transfer_record(
            old_state=old_state,
            old_mesh=old_mesh,
            geometry_new=geometry_new,
            source_outer_iter_index=source_outer_iter_index,
        )

    contents = _build_transferred_contents(
        old_state=old_state,
        old_mesh=old_mesh,
        new_mesh=new_mesh,
    )
    state = _recover_transferred_state(
        contents=contents,
        new_mesh=new_mesh,
        geometry_new=geometry_new,
        old_state=old_state,
        recovery_config=recovery_config,
        species_maps=species_maps,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
    )
    return _build_transfer_record(
        contents=contents,
        state=state,
        geometry_new=geometry_new,
        new_mesh=new_mesh,
        source_outer_iter_index=source_outer_iter_index,
        identity_transfer=False,
    )


def build_old_state_on_current_geometry(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
    geometry: GeometryState,
    recovery_config: RecoveryConfig,
    species_maps: SpeciesMaps,
    liquid_thermo: LiquidThermoProtocol,
    gas_thermo: GasThermoProtocol,
) -> OldStateOnCurrentGeometry:
    """Compatibility wrapper for the legacy old-state/current-geometry name.

    New transfer mainline should use ``build_transfer_state_on_new_geometry``.
    """

    return build_transfer_state_on_new_geometry(
        old_state=old_state,
        old_mesh=old_mesh,
        new_mesh=new_mesh,
        geometry_new=geometry,
        recovery_config=recovery_config,
        species_maps=species_maps,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
    )


def summarize_remap_diagnostics(
    old_contents: ConservativeContents,
    new_contents: ConservativeContents,
) -> dict[str, float]:
    return {
        "mass_l_before": float(np.sum(old_contents.mass_l)),
        "mass_l_after": float(np.sum(new_contents.mass_l)),
        "mass_g_before": float(np.sum(old_contents.mass_g)),
        "mass_g_after": float(np.sum(new_contents.mass_g)),
        "species_mass_l_before": float(np.sum(old_contents.species_mass_l)),
        "species_mass_l_after": float(np.sum(new_contents.species_mass_l)),
        "species_mass_g_before": float(np.sum(old_contents.species_mass_g)),
        "species_mass_g_after": float(np.sum(new_contents.species_mass_g)),
        "enthalpy_l_before": float(np.sum(old_contents.enthalpy_l)),
        "enthalpy_l_after": float(np.sum(new_contents.enthalpy_l)),
        "enthalpy_g_before": float(np.sum(old_contents.enthalpy_g)),
        "enthalpy_g_after": float(np.sum(new_contents.enthalpy_g)),
    }


def summarize_transfer_diagnostics(
    *,
    old_contents: ConservativeContents,
    new_contents: ConservativeContents,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
    identity_transfer: bool,
) -> dict[str, float | bool]:
    return {
        "identity_transfer": bool(identity_transfer),
        "old_n_liq": int(old_mesh.n_liq),
        "new_n_liq": int(new_mesh.n_liq),
        "old_n_gas": int(old_mesh.n_gas),
        "new_n_gas": int(new_mesh.n_gas),
        **summarize_remap_diagnostics(old_contents, new_contents),
    }


__all__ = [
    "RemapError",
    "build_old_contents_on_current_geometry",
    "build_old_state_on_current_geometry",
    "build_transfer_state_on_new_geometry",
    "summarize_remap_diagnostics",
    "summarize_transfer_diagnostics",
]
