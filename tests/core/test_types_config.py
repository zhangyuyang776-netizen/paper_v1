from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.types import (
    CasePaths,
    DiagnosticsConfig,
    InitializationConfig,
    InnerSolverConfig,
    MeshConfig,
    OuterStepperConfig,
    OutputConfig,
    RecoveryConfig,
    RunConfig,
    SpeciesControlConfig,
    SpeciesMaps,
    TimeStepperConfig,
    ValidationConfig,
)


def make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("NC12H26",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("NC12H26", "O2", "N2"),
        gas_active_names=("NC12H26", "O2"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.array([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.array([0], dtype=np.int64),
    )


def make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("A", "B", "C"),
        liq_active_names=("A", "B"),
        liq_closure_name="C",
        gas_full_names=("A", "B", "N2"),
        gas_active_names=("A", "B"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0, 1], dtype=np.int64),
        gas_full_to_reduced=np.array([0, 1, -1], dtype=np.int64),
        gas_reduced_to_full=np.array([0, 1], dtype=np.int64),
        liq_full_to_gas_full=np.array([0, 1, 0], dtype=np.int64),
    )


def make_run_config() -> RunConfig:
    return RunConfig(
        case_name="baseline",
        case_description="transport-only baseline",
        paths=CasePaths(
            config_path=Path("paper_v1/cases/case.yaml"),
            case_root=Path("paper_v1"),
            gas_mechanism_path=Path("paper_v1/mechanism/gas.yaml"),
            liquid_database_path=Path("paper_v1/data/liquid.yaml"),
            output_root=Path("paper_v1/out"),
        ),
        mesh=MeshConfig(
            a0=5.0e-5,
            r_end=5.0e-3,
            n_liq=16,
            n_gas_near=32,
            far_stretch_ratio=1.08,
        ),
        initialization=InitializationConfig(
            gas_temperature=300.0,
            gas_pressure=101325.0,
            liquid_temperature=290.0,
            gas_y_full_0=np.array([0.0, 0.21, 0.79], dtype=np.float64),
            liquid_y_full_0=np.array([1.0], dtype=np.float64),
            y_vap_if0_gas_full=np.array([1.0e-6, 0.0, 0.0], dtype=np.float64),
            t_init_T=1.0e-5,
        ),
        species=SpeciesControlConfig(
            gas_closure_species="N2",
            liquid_closure_species=None,
            liquid_to_gas_species_map={"NC12H26": "NC12H26"},
        ),
        time_stepper=TimeStepperConfig(
            t0=0.0,
            t_end=1.0e-3,
            dt_start=1.0e-7,
            dt_min=1.0e-10,
            dt_max=1.0e-5,
            retry_max_per_step=8,
            q_success_for_growth=10,
            growth_factor=1.1,
            shrink_factor=0.5,
        ),
        outer_stepper=OuterStepperConfig(
            outer_max_iter=8,
            eps_dot_a_tol=1.0e-5,
            corrector_relaxation=1.0,
        ),
        inner_solver=InnerSolverConfig(
            inner_max_iter=30,
            snes_rtol=1.0e-8,
            snes_atol=1.0e-8,
            snes_stol=1.0e-12,
            ksp_rtol=1.0e-8,
            pc_type="fieldsplit",
            ksp_type="fgmres",
            use_fieldsplit=True,
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
            T_min_l=250.0,
            T_max_l=900.0,
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
        species_maps=make_species_maps_single(),
        gas_phase_name="gas",
        liquid_model_name="ideal_liquid",
        equilibrium_model_name="single_component_cc",
        unknowns_profile="U_A",
    )


def test_species_maps_single_component_liquid_valid() -> None:
    species = make_species_maps_single()
    assert species.liq_full_names == ("NC12H26",)
    assert species.liq_active_names == ()
    assert species.liq_closure_name is None
    assert species.is_single_component_liquid is True


def test_species_maps_multicomponent_valid() -> None:
    species = make_species_maps_multi()
    assert species.n_liq_full == 3
    assert species.n_liq_red == 2
    assert species.n_gas_full >= 2


def test_time_stepper_config_invalid_dt_order_raises() -> None:
    with pytest.raises(ValueError):
        TimeStepperConfig(
            t0=0.0,
            t_end=1.0,
            dt_start=1.0e-6,
            dt_min=1.0e-6,
            dt_max=1.0e-5,
            retry_max_per_step=1,
            q_success_for_growth=10,
            growth_factor=1.1,
            shrink_factor=0.5,
        )

    with pytest.raises(ValueError):
        TimeStepperConfig(
            t0=0.0,
            t_end=1.0,
            dt_start=1.0e-4,
            dt_min=1.0e-6,
            dt_max=1.0e-5,
            retry_max_per_step=1,
            q_success_for_growth=10,
            growth_factor=1.1,
            shrink_factor=0.5,
        )


def test_run_config_valid() -> None:
    cfg = make_run_config()
    assert cfg.case_name == "baseline"
    assert cfg.case_description == "transport-only baseline"
    assert cfg.unknowns_profile == "U_A"
    assert cfg.pressure == 101325.0
    assert not hasattr(cfg, "accepted_state")


def test_time_stepper_config_rejects_growth_factor_below_one() -> None:
    with pytest.raises(ValueError, match="growth_factor must be >= 1"):
        TimeStepperConfig(
            t0=0.0,
            t_end=1.0,
            dt_start=1.0e-6,
            dt_min=1.0e-8,
            dt_max=1.0e-5,
            retry_max_per_step=1,
            q_success_for_growth=2,
            growth_factor=0.9,
            shrink_factor=0.5,
        )


def test_time_stepper_config_rejects_shrink_factor_above_one() -> None:
    with pytest.raises(ValueError, match="shrink_factor must satisfy 0 < value <= 1"):
        TimeStepperConfig(
            t0=0.0,
            t_end=1.0,
            dt_start=1.0e-6,
            dt_min=1.0e-8,
            dt_max=1.0e-5,
            retry_max_per_step=1,
            q_success_for_growth=2,
            growth_factor=1.1,
            shrink_factor=1.2,
        )


def test_output_config_rejects_unsupported_snapshot_format() -> None:
    with pytest.raises(ValueError, match="snapshot_format must be 'npz'"):
        OutputConfig(
            write_spatial_fields=True,
            write_spatial_species=True,
            write_time_series_scalars=True,
            write_time_series_species=True,
            snapshot_format="h5",
        )
