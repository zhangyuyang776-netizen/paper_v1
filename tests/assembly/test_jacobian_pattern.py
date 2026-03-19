from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from assembly.jacobian_pattern import JacobianPattern, build_jacobian_pattern
from assembly.residual_gas import GasFarFieldBC
from assembly.residual_global import ResidualOwnership, assemble_global_residual
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


class DummyLiquidThermo:
    def __init__(self, n_species: int) -> None:
        self.n_species = n_species

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

    def conductivity(self, T: float, Y: np.ndarray) -> float:
        _ = T
        _ = Y
        return 1.0

    def viscosity(self, T: float, Y: np.ndarray) -> float:
        _ = T
        _ = Y
        return 1.0

    def diffusivity(self, T: float, Y: np.ndarray) -> np.ndarray | None:
        _ = T
        _ = Y
        if self.n_species == 1:
            return None
        return np.full(self.n_species, 1.0e-9, dtype=np.float64)

    def density_mass(self, T: float, Y: np.ndarray) -> float:
        _ = T
        _ = Y
        return 1.0

    def enthalpy_mass(self, T: float, Y: np.ndarray) -> float:
        _ = Y
        return float(T)

    def pure_enthalpy_vector(self, T: float) -> np.ndarray:
        return np.full(self.n_species, float(T), dtype=np.float64)


class DummyGasThermo:
    def __init__(self, molecular_weights: np.ndarray) -> None:
        self.molecular_weights = np.asarray(molecular_weights, dtype=np.float64)
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


def _make_mesh_single() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 1),
            gas_near=slice(1, 3),
            gas_far=slice(3, 3),
            gas_all=slice(1, 3),
        ),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )


def _make_mesh_multi() -> Mesh1D:
    return Mesh1D(
        r_faces=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        r_centers=np.asarray([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64),
        volumes=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        face_areas=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        dr=np.asarray([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64),
        region_slices=RegionSlices(
            liq=slice(0, 2),
            gas_near=slice(2, 5),
            gas_far=slice(5, 5),
            gas_all=slice(2, 5),
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
        gas_block=slice(9, 18),
        total_size=18,
        liq_temperature_slice=slice(0, 4, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(4, 5),
        if_gas_species_slice=slice(5, 7),
        if_mpp_slice=slice(7, 8),
        if_liq_species_slice=slice(8, 9),
        gas_temperature_slice=slice(9, 18, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_state_single() -> State:
    return State(
        Tl=np.asarray([300.0], dtype=np.float64),
        Yl_full=np.asarray([[1.0]], dtype=np.float64),
        Tg=np.asarray([300.0, 300.0], dtype=np.float64),
        Yg_full=np.asarray([[0.2, 0.8], [0.2, 0.8]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.0,
            mpp=0.0,
            Ys_g_full=np.asarray([0.2, 0.8], dtype=np.float64),
            Ys_l_full=np.asarray([1.0], dtype=np.float64),
        ),
        time=0.0,
        state_id="pattern_single",
    )


def _make_state_multi() -> State:
    return State(
        Tl=np.asarray([300.0, 302.0], dtype=np.float64),
        Yl_full=np.asarray([[0.60, 0.40], [0.55, 0.45]], dtype=np.float64),
        Tg=np.asarray([303.0, 304.0, 305.0], dtype=np.float64),
        Yg_full=np.asarray(
            [
                [0.20, 0.10, 0.70],
                [0.21, 0.09, 0.70],
                [0.22, 0.08, 0.70],
            ],
            dtype=np.float64,
        ),
        interface=InterfaceState(
            Ts=301.0,
            mpp=0.02,
            Ys_g_full=np.asarray([0.20, 0.10, 0.70], dtype=np.float64),
            Ys_l_full=np.asarray([0.58, 0.42], dtype=np.float64),
        ),
        time=0.0,
        state_id="pattern_multi",
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
        a=float(mesh.r_faces[mesh.interface_face_index]),
        dot_a=0.0,
        r_end=float(mesh.r_faces[-1]),
        step_index=0,
        outer_iter_index=0,
    )
    return OldStateOnCurrentGeometry(contents=contents, state=state, geometry=geometry, mesh=mesh)


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(pressure=101325.0, species_maps=species_maps)


def _make_farfield_bc_single() -> GasFarFieldBC:
    return GasFarFieldBC(T_inf=300.0, Yg_inf_full=np.asarray([0.2, 0.8], dtype=np.float64), p_inf=101325.0)


def _make_farfield_bc_multi() -> GasFarFieldBC:
    return GasFarFieldBC(T_inf=305.0, Yg_inf_full=np.asarray([0.22, 0.08, 0.70], dtype=np.float64), p_inf=101325.0)


def _row_cols(pattern: JacobianPattern, row: int) -> np.ndarray:
    return pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]]


def _column_support_rows(pattern: JacobianPattern, col: int) -> np.ndarray:
    rows: list[int] = []
    for row in range(pattern.shape[0]):
        if np.any(_row_cols(pattern, row) == col):
            rows.append(row)
    return np.asarray(rows, dtype=np.int64)


def _dense_residual_values(result, n_dof: int) -> np.ndarray:
    dense = np.zeros(n_dof, dtype=np.float64)
    dense[result.rows_global] = result.values
    return dense


def test_csr_structure_is_valid_and_every_row_has_diagonal() -> None:
    mesh = _make_mesh_multi()
    species_maps = _make_species_maps_multi()
    layout = _make_layout_multi(mesh)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)

    assert pattern.indptr.shape == (layout.total_size + 1,)
    assert pattern.indptr[0] == 0
    assert pattern.indptr[-1] == pattern.indices.size
    assert np.all(pattern.indptr[1:] >= pattern.indptr[:-1])
    assert np.all(pattern.indices >= 0)
    assert np.all(pattern.indices < layout.total_size)

    for row in range(layout.total_size):
        cols = _row_cols(pattern, row)
        assert cols.size >= 1
        assert np.all(cols[1:] > cols[:-1])
        assert row in cols


def test_single_component_case_has_no_liquid_interface_species_and_no_rd() -> None:
    mesh = _make_mesh_single()
    species_maps = _make_species_maps_single()
    layout = _make_layout_single(mesh)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)

    assert layout.n_liq_red == 0
    assert layout.if_liq_species_slice.start == layout.if_liq_species_slice.stop
    assert pattern.shape == (layout.total_size, layout.total_size)
    assert pattern.diagnostics["has_iface_liq_species"] is False
    assert pattern.diagnostics["block_sizes"]["iface"] == 3


def test_multicomponent_case_expands_bulk_and_interface_blocks() -> None:
    mesh = _make_mesh_multi()
    single_pattern = build_jacobian_pattern(
        mesh=_make_mesh_single(),
        layout=_make_layout_single(_make_mesh_single()),
        species_maps=_make_species_maps_single(),
    )
    multi_pattern = build_jacobian_pattern(mesh=mesh, layout=_make_layout_multi(mesh), species_maps=_make_species_maps_multi())

    assert multi_pattern.shape[0] > single_pattern.shape[0]
    assert multi_pattern.diagnostics["has_liq_species"] is True
    assert multi_pattern.diagnostics["has_iface_liq_species"] is True
    assert multi_pattern.diagnostics["has_gas_species"] is True
    assert multi_pattern.diagnostics["block_sizes"]["liq"] == 4
    assert multi_pattern.diagnostics["block_sizes"]["iface"] == 5
    assert multi_pattern.diagnostics["block_sizes"]["gas"] == 9


def test_last_liquid_and_interface_rows_have_required_cross_block_couplings() -> None:
    mesh = _make_mesh_multi()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)

    iface_cols = np.arange(layout.if_block.start, layout.if_block.stop, dtype=np.int64)
    all_liq_cols = []
    for liq_cell in range(mesh.n_liq):
        all_liq_cols.append(np.asarray([layout.liq_temperature_index(liq_cell)], dtype=np.int64))
        all_liq_cols.append(
            np.arange(
                layout.liq_species_slice_for_cell(liq_cell).start,
                layout.liq_species_slice_for_cell(liq_cell).stop,
                dtype=np.int64,
            )
        )
    all_liq_cols = np.concatenate(all_liq_cols) if all_liq_cols else np.zeros(0, dtype=np.int64)
    first_gas_cols = np.concatenate(
        [
            np.asarray([layout.gas_temperature_index(0)], dtype=np.int64),
            np.arange(layout.gas_species_slice_for_cell(0).start, layout.gas_species_slice_for_cell(0).stop, dtype=np.int64),
        ]
    )

    last_liq_rows = np.concatenate(
        [
            np.asarray([layout.liq_temperature_index(mesh.n_liq - 1)], dtype=np.int64),
            np.arange(
                layout.liq_species_slice_for_cell(mesh.n_liq - 1).start,
                layout.liq_species_slice_for_cell(mesh.n_liq - 1).stop,
                dtype=np.int64,
            ),
        ]
    )
    for row in last_liq_rows.tolist():
        cols = _row_cols(pattern, int(row))
        assert set(iface_cols.tolist()).issubset(set(cols.tolist()))

    for row in np.arange(layout.if_block.start, layout.if_block.stop, dtype=np.int64).tolist():
        cols = _row_cols(pattern, int(row))
        assert set(all_liq_cols.tolist()).issubset(set(cols.tolist()))
        assert set(first_gas_cols.tolist()).issubset(set(cols.tolist()))


def test_gas_rows_include_interface_and_velocity_prefix_coupling() -> None:
    mesh = _make_mesh_multi()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)

    iface_cols = set(np.arange(layout.if_block.start, layout.if_block.stop, dtype=np.int64).tolist())
    liq0_cols = set(_row_cols(pattern, layout.liq_temperature_index(0)).tolist())  # conservative subset source
    gas0_cell_cols = set(
        np.concatenate(
            [
                np.asarray([layout.gas_temperature_index(0)], dtype=np.int64),
                np.arange(layout.gas_species_slice_for_cell(0).start, layout.gas_species_slice_for_cell(0).stop, dtype=np.int64),
            ]
        ).tolist()
    )
    gas1_cell_cols = set(
        np.concatenate(
            [
                np.asarray([layout.gas_temperature_index(1)], dtype=np.int64),
                np.arange(layout.gas_species_slice_for_cell(1).start, layout.gas_species_slice_for_cell(1).stop, dtype=np.int64),
            ]
        ).tolist()
    )
    gas2_cell_cols = set(
        np.concatenate(
            [
                np.asarray([layout.gas_temperature_index(2)], dtype=np.int64),
                np.arange(layout.gas_species_slice_for_cell(2).start, layout.gas_species_slice_for_cell(2).stop, dtype=np.int64),
            ]
        ).tolist()
    )

    gas_rows = []
    for n in range(mesh.n_gas):
        gas_rows.append(layout.gas_temperature_index(n))
        gas_rows.extend(range(layout.gas_species_slice_for_cell(n).start, layout.gas_species_slice_for_cell(n).stop))

    for row in gas_rows:
        cols = set(_row_cols(pattern, row).tolist())
        assert iface_cols.issubset(cols)

    first_gas_cols = set(_row_cols(pattern, layout.gas_temperature_index(0)).tolist())
    assert gas0_cell_cols.issubset(first_gas_cols)
    second_gas_cols = set(_row_cols(pattern, layout.gas_temperature_index(1)).tolist())
    assert gas0_cell_cols.issubset(second_gas_cols)
    assert gas1_cell_cols.issubset(second_gas_cols)
    third_gas_cols = set(_row_cols(pattern, layout.gas_temperature_index(2)).tolist())
    assert gas0_cell_cols.issubset(third_gas_cols)
    assert gas1_cell_cols.issubset(third_gas_cols)
    assert gas2_cell_cols.issubset(third_gas_cols)
    _ = liq0_cols


def test_fd_support_subset_matches_residual_global_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_eq(model, *, Ts, P, Yl_if_full):
        _ = model
        _ = Ts
        _ = P
        _ = Yl_if_full
        return _EqResult(np.asarray([0.20, 0.10, 0.70], dtype=np.float64), np.asarray([1.0, 1.0], dtype=np.float64))

    monkeypatch.setattr("physics.interface_face.compute_interface_equilibrium", _fake_eq)

    mesh = _make_mesh_multi()
    layout = _make_layout_multi(mesh)
    species_maps = _make_species_maps_multi()
    state = _make_state_multi()
    old_star = _make_old_state_current_geom(mesh, state)
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)

    kwargs = dict(
        old_state_current_geom=old_star,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        ownership=ResidualOwnership(
            owned_liq_cells=np.arange(mesh.n_liq, dtype=np.int64),
            owned_gas_cells=np.arange(mesh.n_gas, dtype=np.int64),
            interface_owner_active=True,
        ),
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc_multi(),
    )

    base_result = assemble_global_residual(state_trial=state, **kwargs)
    base_dense = _dense_residual_values(base_result, layout.total_size)
    vec = pack_state_to_array(state, layout, species_maps)

    representative_cols = [
        layout.liq_temperature_index(0),
        layout.liq_species_slice_for_cell(mesh.n_liq - 1).start,
        layout.if_temperature_index,
        layout.if_mpp_index,
        layout.gas_species_slice_for_cell(0).start,
        layout.gas_temperature_index(mesh.n_gas - 1),
    ]

    for col in representative_cols:
        perturbed = vec.copy()
        perturbed[col] += 1.0e-6
        state_pert = apply_trial_vector_to_state(state, perturbed, layout, species_maps)
        pert_result = assemble_global_residual(state_trial=state_pert, **kwargs)
        pert_dense = _dense_residual_values(pert_result, layout.total_size)
        changed_rows = np.flatnonzero(np.abs(pert_dense - base_dense) > 1.0e-10)
        support_rows = _column_support_rows(pattern, col)
        assert set(changed_rows.tolist()).issubset(set(support_rows.tolist()))
