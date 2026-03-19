from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from assembly.jacobian_global import (
    GlobalJacobianFDOptions,
    JacobianOwnership,
    assemble_and_insert_global_jacobian,
    assemble_global_jacobian,
)
from assembly.jacobian_pattern import build_jacobian_pattern
from assembly.petsc_prealloc import build_petsc_prealloc
from assembly.residual_gas import GasFarFieldBC
from assembly.residual_global import ResidualOwnership, assemble_global_residual_from_trial_view
from core.layout import UnknownLayout
from core.state_pack import pack_state_to_array
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


class _EqModelMulti:
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

    def createAIJ(self, *, size, nnz, comm=None):
        _ = comm
        self.size = size
        self.nnz = nnz
        return self

    def setPreallocationNNZ(self, nnz):
        self.nnz = nnz

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
            liq=slice(0, 1),
            gas_near=slice(1, 3),
            gas_far=slice(3, 3),
            gas_all=slice(1, 3),
        ),
        interface_face_index=1,
        interface_cell_liq=0,
        interface_cell_gas=1,
    )


def _make_control_metrics(mesh: Mesh1D) -> ControlSurfaceMetrics:
    return ControlSurfaceMetrics(
        v_c_faces=np.zeros(mesh.n_faces, dtype=np.float64),
        v_c_cells=np.zeros(mesh.n_cells, dtype=np.float64),
        interface_face_index=mesh.interface_face_index,
        region2_outer_face_index=mesh.n_faces - 2,
    )


def _make_species_maps() -> SpeciesMaps:
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


def _make_layout(mesh: Mesh1D) -> UnknownLayout:
    return UnknownLayout(
        unknowns_profile="U_B",
        n_liq_cells=mesh.n_liq,
        n_gas_cells=mesh.n_gas,
        n_liq_red=1,
        n_gas_red=2,
        liq_cell_width=2,
        gas_cell_width=3,
        n_if_unknowns=5,
        liq_block=slice(0, 2),
        if_block=slice(2, 7),
        gas_block=slice(7, 13),
        total_size=13,
        liq_temperature_slice=slice(0, 2, 2),
        liq_species_local_slice=slice(1, 2),
        if_temperature_slice=slice(2, 3),
        if_gas_species_slice=slice(3, 5),
        if_mpp_slice=slice(5, 6),
        if_liq_species_slice=slice(6, 7),
        gas_temperature_slice=slice(7, 13, 3),
        gas_species_local_slice=slice(1, 3),
    )


def _make_state() -> State:
    return State(
        Tl=np.asarray([300.0], dtype=np.float64),
        Yl_full=np.asarray([[0.6, 0.4]], dtype=np.float64),
        Tg=np.asarray([320.0, 325.0], dtype=np.float64),
        Yg_full=np.asarray([[0.2, 0.1, 0.7], [0.18, 0.12, 0.70]], dtype=np.float64),
        interface=InterfaceState(
            Ts=300.5,
            mpp=0.01,
            Ys_g_full=np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            Ys_l_full=np.asarray([0.6, 0.4], dtype=np.float64),
        ),
        time=0.0,
        state_id="jac_global",
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
        a=1.0,
        dot_a=0.0,
        r_end=3.0,
        step_index=0,
        outer_iter_index=0,
    )
    return OldStateOnCurrentGeometry(contents=contents, state=state, geometry=geometry, mesh=mesh)


def _make_run_cfg(species_maps: SpeciesMaps) -> SimpleNamespace:
    return SimpleNamespace(pressure=101325.0, species_maps=species_maps)


def _make_farfield_bc() -> GasFarFieldBC:
    return GasFarFieldBC(T_inf=300.0, Yg_inf_full=np.asarray([0.2, 0.1, 0.7], dtype=np.float64), p_inf=101325.0)


def _row_cols(pattern, row: int) -> np.ndarray:
    return pattern.indices[pattern.indptr[row] : pattern.indptr[row + 1]]


def _make_builder(
    *,
    base_state: State,
    mesh: Mesh1D,
    layout: UnknownLayout,
    species_maps: SpeciesMaps,
    old_star: OldStateOnCurrentGeometry,
    ownership: JacobianOwnership,
    run_cfg,
    liquid_thermo,
    gas_thermo,
    equilibrium_model,
    control_surface_metrics: ControlSurfaceMetrics,
    farfield_bc: GasFarFieldBC,
):
    residual_ownership = ResidualOwnership(
        owned_liq_cells=np.asarray(ownership.owned_liq_cells, dtype=np.int64),
        owned_gas_cells=np.asarray(ownership.owned_gas_cells, dtype=np.int64),
        interface_owner_active=bool(ownership.interface_owner_active),
    )

    def _build_all(u_layout: np.ndarray):
        result = assemble_global_residual_from_trial_view(
            vec_trial=np.asarray(u_layout, dtype=np.float64),
            base_state=base_state,
            old_state_current_geom=old_star,
            mesh=mesh,
            layout=layout,
            species_maps=species_maps,
            ownership=residual_ownership,
            run_cfg=run_cfg,
            liquid_thermo=liquid_thermo,
            gas_thermo=gas_thermo,
            equilibrium_model=equilibrium_model,
            control_surface_metrics=control_surface_metrics,
            farfield_bc=farfield_bc,
        )
        return result.liquid, result.interface, result.gas

    return _build_all


def _global_residual_map(build_all, u_layout: np.ndarray) -> dict[int, float]:
    liquid, interface, gas = build_all(np.asarray(u_layout, dtype=np.float64))
    rows = np.concatenate([liquid.rows_global, interface.rows_global, gas.rows_global])
    values = np.concatenate([liquid.values, interface.values, gas.values])
    return {int(row): float(val) for row, val in zip(rows.tolist(), values.tolist())}


def test_serial_full_ownership_covers_all_rows_and_has_no_duplicate_triplets(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
        interface_owner_active=True,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )

    result = assemble_global_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
    )

    assert result.rows_global.size == result.cols_global.size == result.values.size
    assert set(result.rows_global.tolist()) == set(range(layout.total_size))
    assert set(result.liquid.owned_liq_rows_global.tolist()).isdisjoint(result.interface.owned_interface_rows_global.tolist())
    assert set(result.liquid.owned_liq_rows_global.tolist()).isdisjoint(result.gas.owned_gas_rows_global.tolist())
    assert set(result.interface.owned_interface_rows_global.tolist()).isdisjoint(result.gas.owned_gas_rows_global.tolist())
    pairs = np.column_stack((result.rows_global, result.cols_global))
    assert np.unique(pairs, axis=0).shape[0] == pairs.shape[0]
    for row, col in zip(result.rows_global.tolist(), result.cols_global.tolist()):
        assert int(col) in set(_row_cols(pattern, int(row)).tolist())
    assert result.diagnostics["cache_hits"] > 0


def test_interface_owner_off_returns_only_liquid_and_gas_triplets(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
        interface_owner_active=False,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )

    result = assemble_global_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
    )

    assert result.interface.values.size == 0
    assert result.interface.owned_interface_rows_global.size == 0
    assert set(result.rows_global.tolist()).isdisjoint(set(range(layout.if_block.start, layout.if_block.stop)))


def test_partial_ownership_only_emits_owned_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([1], dtype=np.int64),
        interface_owner_active=False,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )

    result = assemble_global_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
    )

    expected_rows = set(result.liquid.owned_liq_rows_global.tolist()) | set(result.gas.owned_gas_rows_global.tolist())
    assert set(result.rows_global.tolist()) == expected_rows


def test_manual_global_fd_matches_for_mpp_column_and_baseline_is_clean(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
        interface_owner_active=True,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )
    u0 = pack_state_to_array(state, layout, species_maps)
    baseline_before = _global_residual_map(builder, u0)

    result = assemble_global_jacobian(
        u_trial_layout=u0,
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
        fd_options=GlobalJacobianFDOptions(cache_size=16),
    )

    col = layout.if_mpp_index
    step_if = result.interface.diagnostics["column_step_stats"][col]["step"]
    step_liq = result.liquid.diagnostics["column_step_stats"][col]["step"]
    step_gas = result.gas.diagnostics["column_step_stats"][col]["step"]
    assert step_if == pytest.approx(step_liq)
    assert step_if == pytest.approx(step_gas)

    u_plus = np.array(u0, copy=True)
    u_plus[col] += step_if
    baseline_map = _global_residual_map(builder, u0)
    plus_map = _global_residual_map(builder, u_plus)

    col_mask = result.cols_global == col
    actual = {int(row): float(val) for row, val in zip(result.rows_global[col_mask].tolist(), result.values[col_mask].tolist())}
    expected = {row: (plus_map[row] - baseline_map[row]) / step_if for row in actual}
    assert set(actual) == set(expected)
    for row in actual:
        assert actual[row] == pytest.approx(expected[row], rel=1e-8, abs=1e-8)

    assert np.array_equal(u0, pack_state_to_array(state, layout, species_maps))
    baseline_after = _global_residual_map(builder, u0)
    assert baseline_before == baseline_after


def test_prealloc_and_petsc_insert_work_with_non_identity_mapping(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([0, 1], dtype=np.int64),
        interface_owner_active=True,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )
    perm = np.asarray(list(reversed(range(layout.total_size))), dtype=np.int64)
    prealloc = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=(0, layout.total_size),
        ownership_ranges=np.asarray([[0, layout.total_size]], dtype=np.int64),
        PETSc=_FakePETSc,
        comm=None,
        layout_to_petsc=perm,
    )

    inserted = assemble_and_insert_global_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
        mat=prealloc.mat,
        PETSc=_FakePETSc,
        layout_to_petsc=perm,
    )

    assert inserted.triplets.values.size > 0
    assert len(inserted.mat.values) == inserted.triplets.values.size
    sample_row = int(inserted.triplets.rows_global[0])
    sample_col = int(inserted.triplets.cols_global[0])
    assert (int(perm[sample_row]), int(perm[sample_col])) in inserted.mat.values


def test_partial_ownership_non_identity_mapping_does_not_reject_global_permutation(monkeypatch) -> None:
    monkeypatch.setattr(
        "physics.interface_face.compute_interface_equilibrium",
        lambda model, *, Ts, P, Yl_if_full: _EqResult(
            np.asarray([0.2, 0.1, 0.7], dtype=np.float64),
            np.asarray([1.0, 1.0], dtype=np.float64),
        ),
    )
    mesh = _make_mesh()
    layout = _make_layout(mesh)
    species_maps = _make_species_maps()
    state = _make_state()
    old_star = _make_old_state_current_geom(mesh, state)
    ownership = JacobianOwnership(
        owned_liq_cells=np.asarray([0], dtype=np.int64),
        owned_gas_cells=np.asarray([1], dtype=np.int64),
        interface_owner_active=False,
    )
    pattern = build_jacobian_pattern(mesh=mesh, layout=layout, species_maps=species_maps)
    builder = _make_builder(
        base_state=state,
        mesh=mesh,
        layout=layout,
        species_maps=species_maps,
        old_star=old_star,
        ownership=ownership,
        run_cfg=_make_run_cfg(species_maps),
        liquid_thermo=DummyLiquidThermo(2),
        gas_thermo=DummyGasThermo(np.asarray([46.0e-3, 18.0e-3, 28.0e-3])),
        equilibrium_model=_EqModelMulti(),
        control_surface_metrics=_make_control_metrics(mesh),
        farfield_bc=_make_farfield_bc(),
    )
    perm = np.asarray(list(reversed(range(layout.total_size))), dtype=np.int64)
    prealloc = build_petsc_prealloc(
        pattern=pattern,
        ownership_range=(0, layout.total_size),
        ownership_ranges=np.asarray([[0, layout.total_size]], dtype=np.int64),
        PETSc=_FakePETSc,
        comm=None,
        layout_to_petsc=perm,
    )

    inserted = assemble_and_insert_global_jacobian(
        u_trial_layout=pack_state_to_array(state, layout, species_maps),
        layout=layout,
        mesh=mesh,
        species_maps=species_maps,
        ownership=ownership,
        pattern=pattern,
        build_all_residual_blocks_from_layout_vector=builder,
        mat=prealloc.mat,
        PETSc=_FakePETSc,
        layout_to_petsc=perm,
    )

    assert inserted.triplets.values.size > 0
    for row, col in zip(inserted.triplets.rows_global.tolist(), inserted.triplets.cols_global.tolist()):
        assert (int(perm[row]), int(perm[col])) in inserted.mat.values
