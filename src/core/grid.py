from __future__ import annotations

import numpy as np

from .types import ControlSurfaceMetrics, GeometryState, Mesh1D, MeshConfig, RegionSlices, RunConfig


REGION2_OUTER_RADIUS_FACTOR = 5.0
_REGION3_MAX_CELLS = 1_000_000


def _validate_mesh_inputs(mesh_cfg: MeshConfig, geom: GeometryState) -> None:
    if not np.isfinite(mesh_cfg.a0) or mesh_cfg.a0 <= 0.0:
        raise ValueError("mesh_cfg.a0 must be > 0")
    if not np.isfinite(geom.a) or geom.a <= 0.0:
        raise ValueError("geometry.a must be > 0")
    if mesh_cfg.n_liq < 1:
        raise ValueError("mesh_cfg.n_liq must be >= 1")
    if mesh_cfg.n_gas_near < 1:
        raise ValueError("mesh_cfg.n_gas_near must be >= 1")
    if not np.isfinite(mesh_cfg.far_stretch_ratio) or mesh_cfg.far_stretch_ratio < 1.0:
        raise ValueError("mesh_cfg.far_stretch_ratio must be finite and >= 1")

    r_I = _region2_outer_radius(mesh_cfg.a0)
    if not np.isfinite(mesh_cfg.r_end) or mesh_cfg.r_end <= r_I:
        raise ValueError("mesh_cfg.r_end must be greater than 5 * a0")
    if geom.a >= r_I:
        raise ValueError("geometry.a must be strictly less than 5 * a0")


def _region2_outer_radius(a0: float) -> float:
    return REGION2_OUTER_RADIUS_FACTOR * a0


def _build_region1_faces(a: float, n_liq: int) -> np.ndarray:
    return np.linspace(0.0, a, n_liq + 1, dtype=np.float64)


def _build_region2_faces(a: float, r_I: float, n_gas_near: int) -> np.ndarray:
    return np.linspace(a, r_I, n_gas_near + 1, dtype=np.float64)


def _build_region3_faces(
    *,
    a0: float,
    r_I: float,
    r_end: float,
    n_gas_near: int,
    far_stretch_ratio: float,
) -> np.ndarray:
    # paper_v1 formal rule:
    # the first region-3 cell size is set to r_I / N2 = 5*a0/N2,
    # which is the upper envelope of region-2 spacing as a -> 0.
    dr3_first = r_I / float(n_gas_near)
    if dr3_first <= 0.0 or not np.isfinite(dr3_first):
        raise ValueError("region-3 first cell size must be finite and > 0")

    alpha = float(far_stretch_ratio)
    faces = [float(r_I)]
    dr = dr3_first

    # Engineering termination rule:
    # region-3 cells follow the geometric growth law until the next face would exceed r_end.
    # The last face is then truncated to exactly r_end.
    for _ in range(_REGION3_MAX_CELLS):
        next_face = faces[-1] + dr
        if next_face >= r_end or np.isclose(next_face, r_end, rtol=0.0, atol=1e-15 * max(1.0, r_end)):
            faces.append(float(r_end))
            break
        faces.append(float(next_face))
        dr *= alpha
    else:
        raise RuntimeError("region-3 grid generation exceeded the safety iteration limit")

    region3_faces = np.asarray(faces, dtype=np.float64)
    if region3_faces.size < 2:
        raise ValueError("region-3 must contain at least one cell")
    return region3_faces


def _concatenate_region_faces(
    region1_faces: np.ndarray,
    region2_faces: np.ndarray,
    region3_faces: np.ndarray,
) -> np.ndarray:
    return np.concatenate((region1_faces, region2_faces[1:], region3_faces[1:])).astype(np.float64, copy=False)


def _compute_centers(r_faces: np.ndarray) -> np.ndarray:
    return 0.5 * (r_faces[:-1] + r_faces[1:])


def _compute_dr(r_faces: np.ndarray) -> np.ndarray:
    return r_faces[1:] - r_faces[:-1]


def _compute_cell_volumes_spherical(r_faces: np.ndarray) -> np.ndarray:
    return (4.0 * np.pi / 3.0) * (r_faces[1:] ** 3 - r_faces[:-1] ** 3)


def _compute_face_areas_spherical(r_faces: np.ndarray) -> np.ndarray:
    return 4.0 * np.pi * r_faces**2


def _build_region_slices(n_liq: int, n_gas_near: int, n_gas_total: int) -> RegionSlices:
    n_gas_far = n_gas_total - n_gas_near
    if n_gas_far < 1:
        raise ValueError("region-3 must contain at least one gas cell")
    gas_start = n_liq
    gas_near_stop = gas_start + n_gas_near
    gas_all_stop = gas_start + n_gas_total
    return RegionSlices(
        liq=slice(0, n_liq),
        gas_near=slice(gas_start, gas_near_stop),
        gas_far=slice(gas_near_stop, gas_all_stop),
        gas_all=slice(gas_start, gas_all_stop),
    )


def _build_control_surface_velocity_faces(
    *,
    r_faces: np.ndarray,
    n_liq: int,
    n_gas_near: int,
    a: float,
    a0: float,
    dot_a: float,
) -> np.ndarray:
    r_I = _region2_outer_radius(a0)
    interface_face_index = n_liq
    region2_outer_face_index = n_liq + n_gas_near
    v_c_faces = np.zeros_like(r_faces)

    # paper_v1 formal rule:
    # region-2 outer boundary is fixed at r_I = 5 * a0.
    # This is intentionally not exposed as a user config parameter in the current version.
    v_c_faces[: interface_face_index + 1] = dot_a * r_faces[: interface_face_index + 1] / a
    v_c_faces[interface_face_index : region2_outer_face_index + 1] = (
        dot_a * (r_I - r_faces[interface_face_index : region2_outer_face_index + 1]) / (r_I - a)
    )
    return v_c_faces


def _build_control_surface_metrics(
    *,
    r_faces: np.ndarray,
    v_c_faces: np.ndarray,
    n_liq: int,
    n_gas_near: int,
) -> ControlSurfaceMetrics:
    v_c_cells = 0.5 * (v_c_faces[:-1] + v_c_faces[1:])
    return ControlSurfaceMetrics(
        v_c_faces=v_c_faces,
        v_c_cells=v_c_cells,
        interface_face_index=n_liq,
        region2_outer_face_index=n_liq + n_gas_near,
    )


def _build_grid_from_geometry(run_cfg: RunConfig, geometry: GeometryState) -> tuple[Mesh1D, ControlSurfaceMetrics]:
    mesh_cfg = run_cfg.mesh
    _validate_mesh_inputs(mesh_cfg, geometry)

    r_I = _region2_outer_radius(mesh_cfg.a0)
    region1_faces = _build_region1_faces(geometry.a, mesh_cfg.n_liq)
    region2_faces = _build_region2_faces(geometry.a, r_I, mesh_cfg.n_gas_near)
    region3_faces = _build_region3_faces(
        a0=mesh_cfg.a0,
        r_I=r_I,
        r_end=mesh_cfg.r_end,
        n_gas_near=mesh_cfg.n_gas_near,
        far_stretch_ratio=mesh_cfg.far_stretch_ratio,
    )

    r_faces = _concatenate_region_faces(region1_faces, region2_faces, region3_faces)
    r_centers = _compute_centers(r_faces)
    dr = _compute_dr(r_faces)
    volumes = _compute_cell_volumes_spherical(r_faces)
    face_areas = _compute_face_areas_spherical(r_faces)

    n_gas_far = len(region3_faces) - 1
    n_gas_total = mesh_cfg.n_gas_near + n_gas_far
    region_slices = _build_region_slices(mesh_cfg.n_liq, mesh_cfg.n_gas_near, n_gas_total)

    interface_face_index = mesh_cfg.n_liq
    face_owner_phase = np.zeros_like(r_faces, dtype=np.int64)
    face_owner_phase[interface_face_index + 1 :] = 1

    mesh = Mesh1D(
        r_faces=r_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=region_slices,
        face_owner_phase=face_owner_phase,
        interface_face_index=interface_face_index,
        interface_cell_liq=mesh_cfg.n_liq - 1,
        interface_cell_gas=mesh_cfg.n_liq,
    )

    v_c_faces = _build_control_surface_velocity_faces(
        r_faces=r_faces,
        n_liq=mesh_cfg.n_liq,
        n_gas_near=mesh_cfg.n_gas_near,
        a=geometry.a,
        a0=mesh_cfg.a0,
        dot_a=geometry.dot_a,
    )
    metrics = _build_control_surface_metrics(
        r_faces=r_faces,
        v_c_faces=v_c_faces,
        n_liq=mesh_cfg.n_liq,
        n_gas_near=mesh_cfg.n_gas_near,
    )
    return mesh, metrics


def build_initial_grid(run_cfg: RunConfig, geometry: GeometryState) -> Mesh1D:
    mesh, _ = _build_grid_from_geometry(run_cfg, geometry)
    return mesh


def rebuild_grid(run_cfg: RunConfig, geometry: GeometryState) -> Mesh1D:
    mesh, _ = _build_grid_from_geometry(run_cfg, geometry)
    return mesh


def build_grid_and_metrics(run_cfg: RunConfig, geometry: GeometryState) -> tuple[Mesh1D, ControlSurfaceMetrics]:
    return _build_grid_from_geometry(run_cfg, geometry)


__all__ = [
    "REGION2_OUTER_RADIUS_FACTOR",
    "build_grid_and_metrics",
    "build_initial_grid",
    "rebuild_grid",
]
