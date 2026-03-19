from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.layout import build_layout
from core.state_pack import (
    StatePackError,
    apply_trial_vector_to_state,
    extract_block_views,
    pack_state_to_array,
    reshape_bulk_block_views,
    unpack_array_to_state,
)
from core.types import (
    CasePaths,
    DiagnosticsConfig,
    InitializationConfig,
    InnerSolverConfig,
    InterfaceState,
    Mesh1D,
    MeshConfig,
    OuterStepperConfig,
    OutputConfig,
    RecoveryConfig,
    RegionSlices,
    RunConfig,
    SpeciesControlConfig,
    SpeciesMaps,
    State,
    TimeStepperConfig,
    ValidationConfig,
)


def make_dummy_mesh(n_liq: int, n_gas: int) -> Mesh1D:
    n_cells = n_liq + n_gas
    r_faces = np.linspace(0.0, float(n_cells), n_cells + 1, dtype=np.float64)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    volumes = np.ones(n_cells, dtype=np.float64)
    face_areas = np.linspace(0.0, float(n_cells), n_cells + 1, dtype=np.float64)
    dr = np.ones(n_cells, dtype=np.float64)
    return Mesh1D(
        r_faces=r_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=RegionSlices(
            liq=slice(0, n_liq),
            gas_near=slice(n_liq, n_cells),
            gas_far=slice(n_cells, n_cells),
            gas_all=slice(n_liq, n_cells),
        ),
        interface_face_index=n_liq,
        interface_cell_liq=n_liq - 1,
        interface_cell_gas=n_liq,
    )


def make_run_config(profile: str, species_maps: SpeciesMaps) -> RunConfig:
    liquid_names = species_maps.liq_full_names
    liquid_to_gas = {
        liq_name: species_maps.gas_full_names[species_maps.liq_full_to_gas_full[idx]]
        for idx, liq_name in enumerate(liquid_names)
    }
    liquid_closure = species_maps.liq_closure_name

    if len(liquid_names) == 1:
        liquid_y0 = np.array([1.0], dtype=np.float64)
        y_vap = np.zeros(species_maps.n_gas_full, dtype=np.float64)
        y_vap[species_maps.liq_full_to_gas_full[0]] = 1.0e-6
    else:
        liquid_y0 = np.zeros(species_maps.n_liq_full, dtype=np.float64)
        liquid_y0[: species_maps.n_liq_full - 1] = 0.2
        liquid_y0[-1] = 1.0 - np.sum(liquid_y0[:-1])
        y_vap = np.zeros(species_maps.n_gas_full, dtype=np.float64)
        y_vap[species_maps.liq_full_to_gas_full[0]] = 1.0e-6

    gas_y0 = np.zeros(species_maps.n_gas_full, dtype=np.float64)
    gas_y0[0] = 0.79
    gas_y0[-1] = 0.21

    return RunConfig(
        case_name="state_pack_case",
        case_description="state pack test",
        paths=CasePaths(
            config_path=Path("case.yaml"),
            case_root=Path("."),
            gas_mechanism_path=Path("gas.yaml"),
            liquid_database_path=Path("liq.yaml"),
            output_root=Path("out"),
        ),
        mesh=MeshConfig(
            a0=1.0e-4,
            r_end=1.0e-2,
            n_liq=4,
            n_gas_near=8,
            far_stretch_ratio=1.05,
        ),
        initialization=InitializationConfig(
            gas_temperature=1000.0,
            gas_pressure=101325.0,
            liquid_temperature=300.0,
            gas_y_full_0=gas_y0,
            liquid_y_full_0=liquid_y0,
            y_vap_if0_gas_full=y_vap,
            t_init_T=1.0e-5,
        ),
        species=SpeciesControlConfig(
            gas_closure_species=species_maps.gas_closure_name,
            liquid_closure_species=liquid_closure,
            liquid_to_gas_species_map=liquid_to_gas,
        ),
        species_maps=species_maps,
        time_stepper=TimeStepperConfig(
            t0=0.0,
            t_end=1.0e-3,
            dt_start=1.0e-6,
            dt_min=1.0e-9,
            dt_max=1.0e-4,
            retry_max_per_step=4,
            q_success_for_growth=3,
            growth_factor=1.2,
            shrink_factor=0.5,
        ),
        outer_stepper=OuterStepperConfig(
            outer_max_iter=8,
            eps_dot_a_tol=1.0e-6,
            corrector_relaxation=1.0,
        ),
        inner_solver=InnerSolverConfig(
            inner_max_iter=50,
            snes_rtol=1.0e-8,
            snes_atol=1.0e-10,
            snes_stol=1.0e-12,
            ksp_rtol=1.0e-8,
            pc_type="ilu",
            ksp_type="gmres",
            use_fieldsplit=False,
        ),
        recovery=RecoveryConfig(
            T_min_l=200.0,
            T_max_l=800.0,
            T_min_g=200.0,
            T_max_g=4000.0,
            liq_h_inv_tol=1.0e-10,
            liq_h_inv_max_iter=50,
            gas_h_inv_tol=1.0e-10,
            gas_h_inv_max_iter=50,
            use_cantera_hpy_first=True,
        ),
        diagnostics=DiagnosticsConfig(
            verbose_interface_panel=False,
            verbose_property_warnings=False,
            write_step_diag=True,
            write_interface_diag=True,
            write_failure_report=True,
            output_every_n_steps=1,
        ),
        output=OutputConfig(
            write_spatial_fields=True,
            write_spatial_species=True,
            write_time_series_scalars=True,
            write_time_series_species=True,
            snapshot_format="npz",
        ),
        validation=ValidationConfig(
            enable_mass_balance_check=True,
            enable_energy_balance_check=True,
            enable_state_bounds_check=True,
        ),
        unknowns_profile=profile,
    )


def make_species_maps_u_a() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("N2", "O2", "C2H5OH"),
        gas_active_names=("O2", "C2H5OH"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.array([2], dtype=np.int64),
    )


def make_species_maps_u_b() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("A", "B", "C"),
        liq_active_names=("A", "B"),
        liq_closure_name="C",
        gas_full_names=("N2", "A_g", "B_g", "C_g", "O2"),
        gas_active_names=("A_g", "B_g", "C_g", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0, 1], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1, 2, 3], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2, 3, 4], dtype=np.int64),
        liq_full_to_gas_full=np.array([1, 2, 3], dtype=np.int64),
    )


def make_state_pack_case_u_a() -> tuple[SpeciesMaps, Mesh1D, object, State]:
    species_maps = make_species_maps_u_a()
    mesh = make_dummy_mesh(n_liq=2, n_gas=3)
    run_cfg = make_run_config("U_A", species_maps)
    layout = build_layout(run_cfg, mesh)
    state = State(
        Tl=np.array([320.0, 330.0]),
        Yl_full=np.array([[1.0], [1.0]]),
        Tg=np.array([800.0, 810.0, 820.0]),
        Yg_full=np.array(
            [
                [0.70, 0.20, 0.10],
                [0.60, 0.25, 0.15],
                [0.65, 0.20, 0.15],
            ]
        ),
        interface=InterfaceState(
            Ts=345.0,
            mpp=0.12,
            Ys_g_full=np.array([0.75, 0.15, 0.10]),
            Ys_l_full=np.array([1.0]),
        ),
        rho_l=np.array([700.0, 690.0]),
        rho_g=np.array([1.0, 1.1, 1.2]),
        hl=np.array([100.0, 105.0]),
        hg=np.array([200.0, 210.0, 220.0]),
        Xg_full=np.array(
            [
                [0.68, 0.21, 0.11],
                [0.58, 0.27, 0.15],
                [0.62, 0.22, 0.16],
            ]
        ),
        time=1.5e-6,
        state_id="accepted_u_a",
    )
    return species_maps, mesh, layout, state


def make_state_pack_case_u_b() -> tuple[SpeciesMaps, Mesh1D, object, State]:
    species_maps = make_species_maps_u_b()
    mesh = make_dummy_mesh(n_liq=2, n_gas=2)
    run_cfg = make_run_config("U_B", species_maps)
    layout = build_layout(run_cfg, mesh)
    state = State(
        Tl=np.array([350.0, 355.0]),
        Yl_full=np.array(
            [
                [0.20, 0.30, 0.50],
                [0.10, 0.20, 0.70],
            ]
        ),
        Tg=np.array([900.0, 910.0]),
        Yg_full=np.array(
            [
                [0.55, 0.10, 0.05, 0.10, 0.20],
                [0.50, 0.15, 0.05, 0.10, 0.20],
            ]
        ),
        interface=InterfaceState(
            Ts=360.0,
            mpp=0.22,
            Ys_g_full=np.array([0.58, 0.10, 0.06, 0.06, 0.20]),
            Ys_l_full=np.array([0.20, 0.25, 0.55]),
        ),
        rho_l=np.array([750.0, 740.0]),
        rho_g=np.array([1.3, 1.4]),
        hl=np.array([120.0, 121.0]),
        hg=np.array([240.0, 241.0]),
        Xg_full=np.array(
            [
                [0.53, 0.11, 0.06, 0.10, 0.20],
                [0.48, 0.16, 0.06, 0.10, 0.20],
            ]
        ),
        time=2.5e-6,
        state_id="accepted_u_b",
    )
    return species_maps, mesh, layout, state


def test_pack_unpack_roundtrip_u_a() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()

    vec = pack_state_to_array(state, layout, species_maps)
    recovered = unpack_array_to_state(vec, layout, species_maps, time=state.time, state_id=state.state_id)

    assert np.allclose(recovered.Tl, state.Tl)
    assert np.allclose(recovered.Tg, state.Tg)
    assert recovered.interface.Ts == pytest.approx(state.interface.Ts)
    assert recovered.interface.mpp == pytest.approx(state.interface.mpp)
    assert np.allclose(recovered.Yg_full, state.Yg_full)
    assert np.allclose(recovered.interface.Ys_g_full, state.interface.Ys_g_full)
    assert np.allclose(recovered.Yl_full, np.ones_like(state.Yl_full))
    assert np.allclose(recovered.interface.Ys_l_full, np.array([1.0]))


def test_pack_unpack_roundtrip_u_b() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()

    vec = pack_state_to_array(state, layout, species_maps)
    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert np.allclose(recovered.Tl, state.Tl)
    assert np.allclose(recovered.Tg, state.Tg)
    assert np.allclose(recovered.Yl_full, state.Yl_full)
    assert np.allclose(recovered.Yg_full, state.Yg_full)
    assert np.allclose(recovered.interface.Ys_l_full, state.interface.Ys_l_full)
    assert np.allclose(recovered.interface.Ys_g_full, state.interface.Ys_g_full)


def test_unpack_reconstructs_gas_closure_by_complement() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()
    vec = pack_state_to_array(state, layout, species_maps)
    gas_cell0 = layout.gas_species_slice_for_cell(0)
    vec[gas_cell0] = np.array([0.2, 0.3])

    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert recovered.Yg_full[0, 0] == pytest.approx(0.5)
    assert np.allclose(recovered.Yg_full[0, 1:], [0.2, 0.3])


def test_unpack_reconstructs_liquid_closure_by_complement() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)
    liq_cell0 = layout.liq_species_slice_for_cell(0)
    vec[liq_cell0] = np.array([0.2, 0.1])

    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert np.allclose(recovered.Yl_full[0], [0.2, 0.1, 0.7])


def test_unpack_single_component_liquid_returns_unit_full_fraction() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()
    vec = pack_state_to_array(state, layout, species_maps)

    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert np.allclose(recovered.Yl_full, 1.0)
    assert np.allclose(recovered.interface.Ys_l_full, [1.0])


def test_unpack_rejects_vector_length_mismatch() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()
    vec = pack_state_to_array(state, layout, species_maps)

    with pytest.raises(StatePackError, match="length must match"):
        unpack_array_to_state(vec[:-1], layout, species_maps)


def test_pack_rejects_state_layout_cell_count_mismatch() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()
    bad_state = State(
        Tl=np.array([320.0]),
        Yl_full=np.array([[1.0]]),
        Tg=state.Tg.copy(),
        Yg_full=state.Yg_full.copy(),
        interface=state.interface,
    )

    with pytest.raises(StatePackError, match="n_liq_cells"):
        pack_state_to_array(bad_state, layout, species_maps)


def test_pack_rejects_state_species_count_mismatch() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    bad_state = State(
        Tl=state.Tl.copy(),
        Yl_full=np.array([[0.3, 0.7], [0.4, 0.6]]),
        Tg=state.Tg.copy(),
        Yg_full=state.Yg_full.copy(),
        interface=InterfaceState(
            Ts=state.interface.Ts,
            mpp=state.interface.mpp,
            Ys_g_full=state.interface.Ys_g_full.copy(),
            Ys_l_full=np.array([0.3, 0.7]),
        ),
    )

    with pytest.raises(StatePackError, match="liquid full-species count"):
        pack_state_to_array(bad_state, layout, species_maps)


def test_unpack_resets_derived_fields_to_none() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)

    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert recovered.rho_l is None
    assert recovered.rho_g is None
    assert recovered.hl is None
    assert recovered.hg is None
    assert recovered.Xg_full is None


def test_apply_trial_vector_returns_non_aliasing_state() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)
    vec_trial = vec.copy()

    trial = apply_trial_vector_to_state(state, vec_trial, layout, species_maps)
    trial.Tl[0] = 999.0
    trial.Yg_full[0, 0] = -10.0

    assert state.Tl[0] != 999.0
    assert state.Yg_full[0, 0] != -10.0


def test_apply_trial_vector_preserves_state_metadata() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_a()
    vec = pack_state_to_array(state, layout, species_maps)

    trial = apply_trial_vector_to_state(state, vec, layout, species_maps)

    assert trial.time == state.time
    assert trial.state_id == state.state_id


def test_extract_block_views_match_layout_slices() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)

    liq_block, if_block, gas_block = extract_block_views(vec, layout)

    assert liq_block.shape[0] == layout.liq_block.stop - layout.liq_block.start
    assert if_block.shape[0] == layout.if_block.stop - layout.if_block.start
    assert gas_block.shape[0] == layout.gas_block.stop - layout.gas_block.start


def test_reshape_bulk_block_views_shapes() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)

    liq_cells, gas_cells = reshape_bulk_block_views(vec, layout)

    assert liq_cells.shape == (layout.n_liq_cells, layout.liq_cell_width)
    assert gas_cells.shape == (layout.n_gas_cells, layout.gas_cell_width)


def test_unpack_does_not_renormalize_or_clip_closure() -> None:
    species_maps, _, layout, state = make_state_pack_case_u_b()
    vec = pack_state_to_array(state, layout, species_maps)
    liq_cell0 = layout.liq_species_slice_for_cell(0)
    vec[liq_cell0] = np.array([0.8, 0.5])

    recovered = unpack_array_to_state(vec, layout, species_maps)

    assert recovered.Yl_full[0, 2] == pytest.approx(-0.3)
