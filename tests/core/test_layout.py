from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.layout import UnknownLayout, build_layout
from core.types import (
    CasePaths,
    DiagnosticsConfig,
    InitializationConfig,
    InnerSolverConfig,
    Mesh1D,
    MeshConfig,
    OuterStepperConfig,
    OutputConfig,
    RecoveryConfig,
    RegionSlices,
    RunConfig,
    SpeciesControlConfig,
    SpeciesMaps,
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
        liq_full_names=("A", "B"),
        liq_active_names=("A",),
        liq_closure_name="B",
        gas_full_names=("N2", "A_g", "B_g", "O2"),
        gas_active_names=("A_g", "B_g", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1, 2], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2, 3], dtype=np.int64),
        liq_full_to_gas_full=np.array([1, 2], dtype=np.int64),
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
        liquid_y0 = np.array([0.7, 0.3], dtype=np.float64)
        y_vap = np.zeros(species_maps.n_gas_full, dtype=np.float64)
        y_vap[species_maps.liq_full_to_gas_full[0]] = 1.0e-6

    gas_y0 = np.zeros(species_maps.n_gas_full, dtype=np.float64)
    gas_y0[0] = 0.79
    gas_y0[-1] = 0.21

    return RunConfig(
        case_name="layout_case",
        case_description="layout test",
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
            rho_min=1.0e-12,
            m_min=1.0e-20,
            species_recovery_eps_abs=1.0e-14,
            Y_sum_tol=1.0e-10,
            Y_hard_tol=1.0e-6,
            h_abs_tol=1.0e-8,
            h_rel_tol=1.0e-10,
            h_check_tol=1.0e-8,
            T_step_tol=1.0e-8,
            T_min_l=200.0,
            T_max_l=800.0,
            T_min_g=200.0,
            T_max_g=4000.0,
            liquid_h_inv_max_iter=50,
            cp_min=1.0,
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


def test_build_layout_single_component_u_a() -> None:
    mesh = make_dummy_mesh(n_liq=3, n_gas=5)
    run_cfg = make_run_config("U_A", make_species_maps_u_a())

    layout = build_layout(run_cfg, mesh)

    assert layout.unknowns_profile == "U_A"
    assert layout.n_liq_red == 0
    assert layout.liq_cell_width == 1
    assert layout.if_liq_species_slice.start == layout.if_liq_species_slice.stop
    assert layout.liq_species_local_slice.start == layout.liq_species_local_slice.stop
    assert layout.n_if_unknowns == 1 + layout.n_gas_red + 1
    assert layout.total_size == mesh.n_liq * 1 + layout.n_if_unknowns + mesh.n_gas * (1 + layout.n_gas_red)


def test_build_layout_multicomponent_u_b() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_B", make_species_maps_u_b())

    layout = build_layout(run_cfg, mesh)

    assert layout.unknowns_profile == "U_B"
    assert layout.n_liq_red > 0
    assert layout.liq_cell_width == 1 + layout.n_liq_red
    assert layout.if_liq_species_slice.stop > layout.if_liq_species_slice.start
    assert layout.if_temperature_index == layout.if_block.start
    assert layout.if_mpp_index == layout.if_gas_species_slice.stop


def test_layout_rejects_u_a_with_nonzero_liq_red() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_A", make_species_maps_u_b())

    with pytest.raises(ValueError, match="U_A"):
        build_layout(run_cfg, mesh)


def test_layout_rejects_u_b_with_zero_liq_red() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_B", make_species_maps_u_a())

    with pytest.raises(ValueError, match="U_B"):
        build_layout(run_cfg, mesh)


def test_layout_interface_contains_single_mpp() -> None:
    mesh = make_dummy_mesh(n_liq=3, n_gas=5)
    run_cfg = make_run_config("U_B", make_species_maps_u_b())

    layout = build_layout(run_cfg, mesh)

    assert layout.if_mpp_slice.stop - layout.if_mpp_slice.start == 1


def test_layout_blocks_are_contiguous_and_cover_total_size() -> None:
    mesh = make_dummy_mesh(n_liq=3, n_gas=5)
    run_cfg = make_run_config("U_A", make_species_maps_u_a())

    layout = build_layout(run_cfg, mesh)

    assert layout.liq_block.start == 0
    assert layout.liq_block.stop == layout.if_block.start
    assert layout.if_block.stop == layout.gas_block.start
    assert layout.gas_block.stop == layout.total_size


def test_layout_liq_cell_index_helpers() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_B", make_species_maps_u_b())

    layout = build_layout(run_cfg, mesh)
    cell = layout.liq_cell_slice(1)

    assert cell.stop - cell.start == layout.liq_cell_width
    assert layout.liq_temperature_index(1) == cell.start
    species_slice = layout.liq_species_slice_for_cell(1)
    assert species_slice.start == cell.start + 1
    assert species_slice.stop == species_slice.start + layout.n_liq_red


def test_layout_gas_cell_index_helpers() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_A", make_species_maps_u_a())

    layout = build_layout(run_cfg, mesh)
    cell = layout.gas_cell_slice(2)

    assert cell.stop - cell.start == layout.gas_cell_width
    assert layout.gas_temperature_index(2) == cell.start
    species_slice = layout.gas_species_slice_for_cell(2)
    assert species_slice.start == cell.start + 1
    assert species_slice.stop == species_slice.start + layout.n_gas_red


def test_layout_uses_actual_mesh_cell_counts() -> None:
    mesh = make_dummy_mesh(n_liq=4, n_gas=13)
    run_cfg = make_run_config("U_A", make_species_maps_u_a())
    assert run_cfg.mesh.n_gas_near == 8

    layout = build_layout(run_cfg, mesh)

    assert layout.n_gas_cells == 13


def test_layout_respects_species_maps_order_without_resorting() -> None:
    species_maps = SpeciesMaps(
        liq_full_names=("A", "B"),
        liq_active_names=("A",),
        liq_closure_name="B",
        gas_full_names=("N2", "Z_g", "A_g", "O2"),
        gas_active_names=("Z_g", "A_g", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1, 2], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2, 3], dtype=np.int64),
        liq_full_to_gas_full=np.array([2, 1], dtype=np.int64),
    )
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_B", species_maps)

    layout = build_layout(run_cfg, mesh)

    assert layout.n_gas_red == 3
    assert run_cfg.species_maps.gas_active_names == ("Z_g", "A_g", "O2")


def test_layout_rejects_incorrect_liq_block_length() -> None:
    with pytest.raises(ValueError, match="liq_block length"):
        UnknownLayout(
            unknowns_profile="U_A",
            n_liq_cells=2,
            n_gas_cells=3,
            n_liq_red=0,
            n_gas_red=2,
            liq_cell_width=1,
            gas_cell_width=3,
            n_if_unknowns=4,
            liq_block=slice(0, 1),
            if_block=slice(1, 5),
            gas_block=slice(5, 14),
            total_size=14,
            liq_temperature_slice=slice(0, 1, 1),
            liq_species_local_slice=slice(1, 1),
            if_temperature_slice=slice(1, 2),
            if_gas_species_slice=slice(2, 4),
            if_mpp_slice=slice(4, 5),
            if_liq_species_slice=slice(5, 5),
            gas_temperature_slice=slice(5, 14, 3),
            gas_species_local_slice=slice(1, 3),
        )


def test_layout_rejects_incorrect_if_block_length() -> None:
    with pytest.raises(ValueError, match="if_block length"):
        UnknownLayout(
            unknowns_profile="U_A",
            n_liq_cells=2,
            n_gas_cells=3,
            n_liq_red=0,
            n_gas_red=2,
            liq_cell_width=1,
            gas_cell_width=3,
            n_if_unknowns=4,
            liq_block=slice(0, 2),
            if_block=slice(2, 5),
            gas_block=slice(5, 14),
            total_size=14,
            liq_temperature_slice=slice(0, 2, 1),
            liq_species_local_slice=slice(1, 1),
            if_temperature_slice=slice(2, 3),
            if_gas_species_slice=slice(3, 5),
            if_mpp_slice=slice(5, 6),
            if_liq_species_slice=slice(6, 6),
            gas_temperature_slice=slice(5, 14, 3),
            gas_species_local_slice=slice(1, 3),
        )


def test_layout_rejects_incorrect_gas_block_length() -> None:
    with pytest.raises(ValueError, match="gas_block length"):
        UnknownLayout(
            unknowns_profile="U_A",
            n_liq_cells=2,
            n_gas_cells=3,
            n_liq_red=0,
            n_gas_red=2,
            liq_cell_width=1,
            gas_cell_width=3,
            n_if_unknowns=4,
            liq_block=slice(0, 2),
            if_block=slice(2, 6),
            gas_block=slice(6, 10),
            total_size=10,
            liq_temperature_slice=slice(0, 2, 1),
            liq_species_local_slice=slice(1, 1),
            if_temperature_slice=slice(2, 3),
            if_gas_species_slice=slice(3, 5),
            if_mpp_slice=slice(5, 6),
            if_liq_species_slice=slice(6, 6),
            gas_temperature_slice=slice(6, 10, 3),
            gas_species_local_slice=slice(1, 3),
        )


def test_layout_rejects_interface_fields_not_filling_if_block() -> None:
    with pytest.raises(ValueError, match="fill if_block exactly"):
        UnknownLayout(
            unknowns_profile="U_B",
            n_liq_cells=2,
            n_gas_cells=3,
            n_liq_red=1,
            n_gas_red=2,
            liq_cell_width=2,
            gas_cell_width=3,
            n_if_unknowns=6,
            liq_block=slice(0, 4),
            if_block=slice(4, 10),
            gas_block=slice(10, 19),
            total_size=19,
            liq_temperature_slice=slice(0, 4, 2),
            liq_species_local_slice=slice(1, 2),
            if_temperature_slice=slice(4, 5),
            if_gas_species_slice=slice(5, 7),
            if_mpp_slice=slice(7, 8),
            if_liq_species_slice=slice(8, 9),
            gas_temperature_slice=slice(10, 19, 3),
            gas_species_local_slice=slice(1, 3),
        )


def test_layout_rejects_misaligned_temperature_strided_slices() -> None:
    with pytest.raises(ValueError, match="liq_temperature_slice step"):
        UnknownLayout(
            unknowns_profile="U_B",
            n_liq_cells=2,
            n_gas_cells=3,
            n_liq_red=1,
            n_gas_red=2,
            liq_cell_width=2,
            gas_cell_width=3,
            n_if_unknowns=5,
            liq_block=slice(0, 4),
            if_block=slice(4, 9),
            gas_block=slice(9, 18),
            total_size=18,
            liq_temperature_slice=slice(0, 4, 1),
            liq_species_local_slice=slice(1, 2),
            if_temperature_slice=slice(4, 5),
            if_gas_species_slice=slice(5, 7),
            if_mpp_slice=slice(7, 8),
            if_liq_species_slice=slice(8, 9),
            gas_temperature_slice=slice(9, 18, 3),
            gas_species_local_slice=slice(1, 3),
        )


def test_unknown_layout_dataclass_is_frozen() -> None:
    mesh = make_dummy_mesh(n_liq=2, n_gas=4)
    run_cfg = make_run_config("U_A", make_species_maps_u_a())
    layout = build_layout(run_cfg, mesh)

    with pytest.raises(Exception):
        layout.total_size = 0  # type: ignore[misc]
