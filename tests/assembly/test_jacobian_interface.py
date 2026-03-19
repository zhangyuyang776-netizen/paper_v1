from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from assembly.jacobian_interface import (
    InterfaceJacobianAssemblyError,
    InterfaceJacobianFDOptions,
    assemble_interface_jacobian,
)
from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from assembly.petsc_prealloc import build_petsc_prealloc
from assembly.residual_interface import InterfaceResidualResult, assemble_interface_residual
from core.layout import UnknownLayout
from core.state_pack import apply_trial_vector_to_state, pack_state_to_array
from core.types import (
    ConservativeContents,
    ControlSurfaceMetrics,
    GeometryState,
    InterfaceState,
    Mesh1D,
    OldStateOnCurrentGeometry,
    RegionSlices,
    SpeciesMaps,
    State,
)
from physics.interface_energy import build_interface_energy_residual_package
from physics.interface_face import build_interface_face_package
from physics.interface_mass import build_interface_mass_residual_package
from physics.velocity_recovery import build_velocity_recovery_package
from properties.aggregator import build_bulk_props


@dataclass
class DummyLiquidThermo:
    n_species: int

    def density_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        _ = T
        _ = Y
        return np.ones(T.shape[0], dtype=np.float64)

    def cp_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        _ = T
        _ = Y
        return np.ones(T.shape[0], dtype=np.float64)

    def enthalpy_mass_batch(self, T: np.ndarray, Y: np.ndarray) -> np.ndarray:
        _ = Y
        return np.asarray(T, dtype=np.float64)

    def conductivity(self, T: float, Y_full: np.ndarray) -> float:
        _ = T
        _ = Y_full
        return 1.0

    def viscosity(self, T: float, Y_full: np.ndarray) -> float:
        _ = T
        _ = Y_full
        return 1.0

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        return np.linspace(float(T), float(T) + self.n_species - 1, self.n_species, dtype=np.float64)

    def diffusivity(self, T: float, Y_full: np.ndarray) -> np.ndarray | None:
        _ = T
        _ = Y_full
        if self.n_species == 1:
            return None
        return np.linspace(1.0e-9, 2.0e-9, self.n_species, dtype=np.float64)

    def density_mass(self, T: float, Y: np.ndarray) -> float:
        _ = T
        _ = Y
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray) -> float:
        _ = Y
        return float(T)


@dataclass
class DummyGasThermo:
    molecular_weights: np.ndarray

    def __post_init__(self) -> None:
        self.molecular_weights = np.asarray(self.molecular_weights, dtype=np.float64)
        self.n_species = self.molecular_weights.shape[0]
        self.reference_pressure = 101325.0

    def density_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.ones(Y.shape[0], dtype=np.float64)

    def cp_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.ones(Y.shape[0], dtype=np.float64)

    def enthalpy_mass_batch(self, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> np.ndarray:
        _ = Y
        _ = p
        return np.asarray(T, dtype=np.float64)

    def conductivity(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = T
        _ = Y
        _ = p
        return 1.0

    def viscosity(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = T
        _ = Y
        _ = p
        return 1.0

    def diffusivity(self, T: float, Y: np.ndarray, p: float) -> np.ndarray:
        _ = T
        _ = Y
        _ = p
        return np.full(self.n_species, 1.0e-5, dtype=np.float64)

    def density_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = T
        _ = Y
        _ = p
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray, p: float) -> float:
        _ = Y
        _ = p
        return float(T)

    def species_enthalpies_mass(self, T: float) -> np.ndarray:
        return np.full(self.n_species, float(T), dtype=np.float64)


class _EqResult:
    def __init__(self, Yg_eq_full: np.ndarray, gamma_cond: np.ndarray) -> None:
        self.Yg_eq_full = np.asarray(Yg_eq_full, dtype=np.float64)
        self.gamma_cond = np.asarray(gamma_cond, dtype=np.float64)


class _EqModelSingle:
    liquid_cond_indices = np.asarray([0], dtype=np.int64)


class _EqModelMulti:
    liquid_cond_indices = np.asarray([0, 1], dtype=np.int64)


class _EqModelMixed:
    liquid_cond_indices = np.asarray([0, 1], dtype=np.int64)


class _FakeMat:
    class Option:
        NEW_NONZERO_ALLOCATION_ERR = "NEW_NONZERO_ALLOCATION_ERR"

    def __init__(self) -> None:
        self.size = None
        self.nnz = None
        self.options = {}
        self.allowed_slots: dict[int, set[int]] = {}
        self.values: dict[tuple[int, int], float] = {}
        self.ownership_range = (0, 0)
        self.reused = False

    def createAIJ(self, *, size, nnz, comm=None):
        _ = comm
        self.size = size
        self.nnz = nnz
        return self

    def setPreallocationNNZ(self, nnz):
        self.nnz = nnz
        self.reused = True

    def setOption(self, option, value):
        self.options[option] = bool(value)

    def setUp(self):
        return None

    def zeroEntries(self):
        self.values.clear()

    def getSize(self):
        return (int(self.size[0][1]), int(self.size[1][1]))

    def getLocalSize(self):
        return (int(self.size[0][0]), int(self.size[1][0]))

    def getOwnershipRange(self):
        return self.ownership_range

    def _set_allowed_slots(self, rows, row_cols):
        self.allowed_slots = {int(r): set(np.asarray(c, dtype=np.int64).tolist()) for r, c in zip(rows, row_cols)}
        rows_arr = np.asarray(rows, dtype=np.int64)
        if rows_arr.size == 0:
            self.ownership_range = (0, 0)
        else:
            self.ownership_range = (int(np.min(rows_arr)), int(np.max(rows_arr)) + 1)

    def setValue(self, row: int, col: int, value: float):
        allowed = self.allowed_slots.get(int(row), set())
        if self.options.get(self.Option.NEW_NONZERO_ALLOCATION_ERR, False) and int(col) not in allowed:
            raise RuntimeError("new nonzero allocation error")
        self.values[(int(row), int(col))] = float(value)


class _FakePETSc:
    Mat = _FakeMat


def _make_mesh() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 2),
            gas_near=slice(2, 3),
            gas_far=slice(3, 3),
            gas_all=slice(2, 3),
        ),
        interface_face_index=2,
        interface_cell_liq=1,
        interface_cell_gas=2,
    )


def _make_control_metrics(mesh: Mesh1D) -> ControlSurfaceMetrics:
    return ControlSurfaceMetrics(
        v_c_faces=np.zeros(mesh.n_faces, dtype=np.float64),
        v_c_cells=np.zeros(mesh.n_cells, dtype=np.float64),
        interface_face_index=mesh.interface_face_index,
        region2_outer_face_index=mesh.n_faces - 2,
    )


def _make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name="ethanol",
        gas_full_names=("C2H5OH", "N2"),
        gas_active_names=("C2H5OH",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([-1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0], dtype=np.int64),
    )


def _make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("C2H5OH", "H2O", "N2"),
        gas_active_names=("C2H5OH", "H2O"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([0], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0, 1], dtype=np.int64),
    )


def _make_layout_single(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_A",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=0,
        n_gas_red=1,
        liq_cell_width=1,
        gas_cell_width=2,
        n_if_unknowns=3,
        liq_block=slice(0, 2),
        if_block=slice(2, 5),
        gas_block=slice(5, 7),
        total_size=7,
        liq_temperature_slice=slice(0, 2, 1),
        liq_species_local_slice=slice(1, 1),
        if_temperature_slice=slice(2, 3),
        if_gas_species_slice=slice(3, 4),
        if_mpp_slice=slice(4, 5),
        if_liq_species_slice=slice(5, 5),
        gas_temperature_slice=slice(5, 7, 2),
        gas_species_local_slice=slice(1, 2),
    )


def _make_layout_multi(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=1,
        n_gas_red=2,
        liq_cell_width=2,
        gas_cell_width=3,
        n_if_unknowns=5,
        liq_block=slice(0, 4),
        if_block=slice(4, 9),
        gas_block=slice(9, 12),
        total_size=12,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 7),
        if_mpp_slice=slice(7, 8),
        if_liq_species_slice=slice(8, 9),
        gas_temperature_slice=slice(9, 12, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_state_single() -> State:
    return State(
        Tl=np.asarray([300.0, 301.0], dtype=np.float64),
        Yl_full=np.asarray([[1.0], [1.0]], dtype=np.float64),
        Tg=np.asarray([320.0], dtype=np.float64),
        Yg_full=np.asarray([[0.5, 0.5]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.5,
            mpp=0.01,
            Ys_g_full=np.asarray([0.4, 0.6], dtype=np.float64),
            Ys_l_full=np.asarray([1.0], dtype=np.float64),
        ),
        time=0.0,
        state_id="jac_if_single",
    )


def _make_state_multi() -> State:
    return State(
        Tl=np.asarray([300.0, 301.0], dtype=np.float64),
        Yl_full=np.asarray([[0.60, 0.40], [0.55, 0.45]], dtype=np.float64),
        Tg=np.asarray([320.0], dtype=np.float64),
        Yg_full=np.asarray([[0.20, 0.10, 0.70]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.5,
            mpp=0.01,
            Ys_g_full=np.asarray([0.20, 0.10, 0.70], dtype=np.float64),
            Ys_l_full=np.asarray([0.58, 0.42], dtype=np.float64),
        ),
        time=0.0,
        state_id="jac_if_multi",
    )


def _make_old_state_current_geom(mesh: Mesh1D, state: State) -> OldStateOnCurrentGeometry:
    rho_l = np.ones(state.n_liq_cells, dtype=np.float64)
    rho_g = np.ones(state.n_gas_cells, dtype=np.float64)
    contents = ConservativeContents(
        mass_l=rho_l * mesh.volumes[mesh.liq_slice],
        species_mass_l=rho_l[:, None] * state.Yl_full * mesh.volumes[mesh.liq_slice][:, None],
        enthalpy_l=rho_l * state.Tl * mesh.volumes[mesh.liq_slice],
        mass_g=rho_g * mesh.volumes[mesh.gas_slice],
        species_mass_g=rho_g[:, None] * state.Yg_full * mesh.volumes[mesh.gas_slice][:, None],
        enthalpy_g=rho_g * state.Tg * mesh.volumes[mesh.gas_slice],
    )
    geometry = GeometryState(
        t=0.0,
        dt=1.0,
        a=2.0,
        dot_a=0.0,
        r_end=3.0,
        step_index=0,
        outer_iter_index=0,
    )
    return OldStateOnCurrentGeometry(contents=contents, state=state, geometry=geometry, mesh=mesh)


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(pressure=101325.0, species_maps=species_maps)


def _row_cols(pattern, row: int) -> np.ndarray:
    return pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]]


def _make_interface_evaluator(
    *,
    base_state: State,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    old_star: OldStateOnCurrentGeometry,
    run_cfg,
    liquid_thermo,
    gas_thermo,
    equilibrium_model,
) -> callable:
    def _evaluate(u_layout: np.ndarray) -> InterfaceResidualResult:
        state = apply_trial_vector_to_state(base_state, u_layout, layout, species_maps)
        props = build_bulk_props(
            state=state,
            grid=mesh,
            liquid_thermo=liquid_thermo,
            gas_thermo=gas_thermo,
            gas_pressure=float(run_cfg.pressure),
        )
        state_with_props = state.copy_shallow()
        state_with_props.rho_l = props.rho_l.copy()
        state_with_props.rho_g = props.rho_g.copy()
        state_with_props.hl = props.h_l.copy()
        state_with_props.hg = props.h_g.copy()
        iface_pkg = build_interface_face_package(
            run_cfg=run_cfg,
            mesh=mesh,
            state=state_with_props,
            gas_props=gas_thermo,
            liquid_props=liquid_thermo,
            equilibrium_model=equilibrium_model,
            dot_a_frozen=float(old_star.geometry.dot_a),
        )
        control_metrics = _make_control_metrics(mesh)
        vel_pkg = build_velocity_recovery_package(
            mesh=mesh,
            state=state_with_props,
            old_mass_on_current_geometry=old_star,
            iface_pkg=iface_pkg,
            vc_face_liq=control_metrics.v_c_faces[: mesh.n_liq + 1],
            vc_face_gas=control_metrics.v_c_faces[mesh.interface_face_index :],
            dt=float(old_star.geometry.dt),
        )
        iface_mass_pkg = build_interface_mass_residual_package(
            run_cfg=run_cfg,
            iface_pkg=iface_pkg,
            u_l_if_abs=vel_pkg.u_l_if_abs,
            u_g_if_abs=vel_pkg.u_g_if_abs,
        )
        iface_energy_pkg = build_interface_energy_residual_package(iface_pkg)
        return assemble_interface_residual(
            layout=layout,
            species_maps=species_maps,
            iface_face_pkg=iface_pkg,
            iface_mass_pkg=iface_mass_pkg,
            iface_energy_pkg=iface_energy_pkg,
            owner_active=True,
        )

    return _evaluate


def _make_species_maps_mixed() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("C2H5OH", "H2O", "O2", "N2"),
        gas_active_names=("C2H5OH", "H2O", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.asarray([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.asarray([0], dtype=np.int64),
        gas_full_to_reduced=np.asarray([0, 1, 2, -1], dtype=np.int64),
        gas_reduced_to_full=np.asarray([0, 1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.asarray([0, 1], dtype=np.int64),
    )


def _make_layout_mixed(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=1,
        n_gas_red=3,
        liq_cell_width=2,
        gas_cell_width=4,
        n_if_unknowns=6,
        liq_block=slice(0, 4),
        if_block=slice(4, 10),
        gas_block=slice(10, 14),
        total_size=14,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 8),
        if_mpp_slice=slice(8, 9),
        if_liq_species_slice=slice(9, 10),
        gas_temperature_slice=slice(10, 14, 4),
        gas_species_local_slice=slice(1, 4),
    )


def _make_state_mixed() -> State:
    return State(
        Tl=np.asarray([300.0, 301.0], dtype=np.float64),
        Yl_full=np.asarray([[0.60, 0.40], [0.55, 0.45]], dtype=np.float64),
        Tg=np.asarray([320.0], dtype=np.float64),
        Yg_full=np.asarray([[0.20, 0.10, 0.30, 0.40]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.5,
            mpp=0.01,
            Ys_g_full=np.asarray([0.20, 0.10, 0.30, 0.40], dtype=np.float64),
            Ys_l_full=np.asarray([0.58, 0.42], dtype=np.float64),
        ),
        time=0.0,
        state_id="jac_if_mixed",
    )


def _make_mesh_two_gas() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5, 3.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 2.0, 4.0, 6.0, 8.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 2),
            gas_near=slice(2, 4),
            gas_far=slice(4, 4),
            gas_all=slice(2, 4),
        ),
        interface_face_index=2,
        interface_cell_liq=1,
        interface_cell_gas=2,
    )


def _make_layout_mixed_two_gas(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=1,
        n_gas_red=3,
        liq_cell_width=2,
        gas_cell_width=4,
        n_if_unknowns=6,
        liq_block=slice(0, 4),
        if_block=slice(4, 10),
        gas_block=slice(10, 18),
        total_size=18,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 8),
        if_mpp_slice=slice(8, 9),
        if_liq_species_slice=slice(9, 10),
        gas_temperature_slice=slice(10, 18, 4),
        gas_species_local_slice=slice(1, 4),
    )


def _make_state_mixed_two_gas() -> State:
    return State(
        Tl=np.asarray([300.0, 301.0], dtype=np.float64),
        Yl_full=np.asarray([[0.60, 0.40], [0.55, 0.45]], dtype=np.float64),
        Tg=np.asarray([320.0, 325.0], dtype=np.float64),
        Yg_full=np.asarray(
            [
                [0.20, 0.10, 0.30, 0.40],
                [0.18, 0.12, 0.30, 0.40],
            ],
            dtype=np.float64,
        ),
        interface=InterfaceState(
            Ts=300.5,
            mpp=0.01,
            Ys_g_full=np.asarray([0.20, 0.10, 0.30, 0.40], dtype=np.float64),
            Ys_l_full=np.asarray([0.58, 0.42], dtype=np.float64),
        ),
        time=0.0,
        state_id="jac_if_two_gas",
    )


def _pattern_with_extra_edge(pattern: JacobianPattern, *, row: int, extra_col: int) -> JacobianPattern:
    row_sets: list[set[int]] = []
    for i in range(pattern.shape[0]):
        row_cols = np.asarray(pattern.indices[pattern.indptr[i] : pattern.indptr[i + 1]], dtype=np.int64)
        cols = set(row_cols.tolist())
        if i == row:
            cols.add(int(extra_col))
        row_sets.append(cols)

    indptr = np.zeros(pattern.shape[0] + 1, dtype=np.int64)
    row_nnz = np.zeros(pattern.shape[0], dtype=np.int64)
    parts: list[np.ndarray] = []
    cursor = 0
    for i, cols in enumerate(row_sets):
        arr = np.asarray(sorted(cols), dtype=np.int64)
        parts.append(arr)
        row_nnz[i] = arr.size
        cursor += arr.size
        indptr[i + 1] = cursor
    indices = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return JacobianPattern(
        indptr=indptr,
        indices=indices,
        shape=pattern.shape,
        row_nnz=row_nnz,
        diagnostics=dict(pattern.diagnostics),
    )


def test_owner_inactive_returns_empty_without_callback() -> None:
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    vec = pack_state_to_array(_make_state_multi(), layout, species_maps)

    result = assemble_interface_jacobian(
        u_trial_layout=vec,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=lambda _: (_ for _ in ()).throw(AssertionError("callback must not be called")),
        owner_active=False,
    )

    assert result.rows_global.size == 0
    assert result.cols_global.size == 0
    assert result.values.size == 0
    assert result.owned_interface_rows_global.size == 0


def test_outputs_only_interface_rows_and_triplets_stay_in_pattern(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )

    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )

    owned_rows = set(jac.owned_interface_rows_global.tolist())
    assert owned_rows == {
        layout.if_liq_species_slice.start,
        layout.if_temperature_index,
        layout.if_gas_species_slice.start,
        layout.if_gas_species_slice.start + 1,
        layout.if_mpp_index,
    }
    assert set(jac.rows_global.tolist()) <= owned_rows
    for row, col in zip(jac.rows_global.tolist(), jac.cols_global.tolist()):
        assert int(col) in set(_row_cols(pattern, int(row)).tolist())


def test_single_component_interface_works_without_if_liq_species(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.80], dtype=np.float64), np.asarray([1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_single(mesh)
    species_maps = _make_species_maps_single()
    state = _make_state_single()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(1),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelSingle(),
    )

    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )

    assert layout.if_liq_species_slice.start == layout.if_liq_species_slice.stop
    assert np.all(jac.rows_global >= layout.if_block.start)
    assert np.all(jac.rows_global < layout.if_block.stop)


def test_multicomponent_interface_has_liq_prefix_and_gas0_entries(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )

    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )

    mpp_row = layout.if_mpp_index
    row_cols = {
        int(col)
        for row, col in zip(jac.rows_global.tolist(), jac.cols_global.tolist())
        if int(row) == mpp_row
    }
    liq_cols = {
        layout.liq_temperature_index(0),
        layout.liq_species_slice_for_cell(0).start,
        layout.liq_temperature_index(1),
        layout.liq_species_slice_for_cell(1).start,
    }
    assert liq_cols <= row_cols
    assert layout.gas_temperature_index(0) in set(jac.cols_global.tolist())
    assert layout.gas_species_slice_for_cell(0).start in set(jac.cols_global.tolist())


def test_mixed_eq16_eq19_interface_rows_generate_jacobian(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.30, 0.40], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_mixed(mesh)
    species_maps = _make_species_maps_mixed()
    state = _make_state_mixed()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 32.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMixed(),
    )

    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )
    gas_rows = set(range(layout.if_gas_species_slice.start, layout.if_gas_species_slice.stop))
    assert gas_rows <= set(jac.rows_global.tolist())


def test_manual_single_column_fd_matches_output(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )
    vec = pack_state_to_array(state, layout, species_maps)
    jac = assemble_interface_jacobian(
        u_trial_layout=vec,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
        fd_options=InterfaceJacobianFDOptions(),
    )

    target_col = layout.if_mpp_index
    h = float(jac.diagnostics["column_step_stats"][target_col]["step"])
    base = evaluator(vec)
    plus_vec = vec.copy()
    plus_vec[target_col] += h
    plus = evaluator(plus_vec)
    base_map = {int(r): float(v) for r, v in zip(base.rows_global.tolist(), base.values.tolist())}
    plus_map = {int(r): float(v) for r, v in zip(plus.rows_global.tolist(), plus.values.tolist())}
    supported_rows = {row for row in base_map if target_col in _row_cols(pattern, row)}
    expected = {row: (plus_map[row] - base_map[row]) / h for row in supported_rows}
    actual = {
        int(row): float(val)
        for row, col, val in zip(jac.rows_global.tolist(), jac.cols_global.tolist(), jac.values.tolist())
        if int(col) == target_col
    }
    assert actual.keys() == expected.keys()
    for row in expected:
        assert np.isclose(actual[row], expected[row], rtol=0.0, atol=1.0e-8)


def test_baseline_vector_and_residual_are_not_mutated(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )
    vec = pack_state_to_array(state, layout, species_maps)
    vec_before = vec.copy()
    base_before = evaluator(vec)

    _ = assemble_interface_jacobian(
        u_trial_layout=vec,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )

    base_after = evaluator(vec)
    assert np.array_equal(vec, vec_before)
    assert np.array_equal(base_before.rows_global, base_after.rows_global)
    assert np.allclose(base_before.values, base_after.values)


def test_triplets_write_into_preallocated_mat_without_new_slots(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )
    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
    )
    prealloc = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=(0, layout.total_size),
        ownership_ranges=np.asarray([[0, layout.total_size]], dtype=np.int64),
        PETSc=_FakePETSc,
        comm=None,
        mat=None,
        layout_to_petsc=None,
    )
    for row, col, value in zip(jac.rows_global.tolist(), jac.cols_global.tolist(), jac.values.tolist()):
        prealloc.mat.setValue(int(row), int(col), float(value))


def test_central_difference_uses_strict_directional_steps(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
    )

    jac = assemble_interface_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        pattern=pattern,
        evaluate_interface_residual_from_layout_vector=evaluator,
        owner_active=True,
        fd_options=InterfaceJacobianFDOptions(use_central=True),
    )

    stats = jac.diagnostics["column_step_stats"][layout.if_mpp_index]
    assert stats["plus_step"] > 0.0
    assert stats["minus_step"] < 0.0


def test_gas_other_column_in_pattern_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(np.asarray([0.20, 0.10, 0.30, 0.40], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64)),
    )
    mesh = _make_mesh_two_gas()
    layout = _make_layout_mixed_two_gas(mesh)
    species_maps = _make_species_maps_mixed()
    state = _make_state_mixed_two_gas()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    bad_pattern = _pattern_with_extra_edge(
        pattern,
        row=layout.if_mpp_index,
        extra_col=layout.gas_temperature_index(1),
    )
    evaluator = _make_interface_evaluator(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 32.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMixed(),
    )

    with pytest.raises(InterfaceJacobianAssemblyError, match="gas_other"):
        assemble_interface_jacobian(
            u_trial_layout=pack_state_to_array(state, layout, species_maps),
            layout=layout,
            mesh=mesh,
            species_maps=species_maps,
            pattern=bad_pattern,
            evaluate_interface_residual_from_layout_vector=evaluator,
            owner_active=True,
        )
