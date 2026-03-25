from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from core.types import (
    CasePaths,
    DiagnosticsConfig,
    InitializationConfig,
    InnerSolverConfig,
    MeshConfig,
    OutputConfig,
    OuterStepperConfig,
    RecoveryConfig,
    RunConfig,
    SpeciesControlConfig,
    SpeciesMaps,
    TimeStepperConfig,
    ValidationConfig,
)
from properties.liquid_db import (
    LiquidDatabaseFormatError,
    LiquidDatabaseRangeError,
    LiquidDatabaseValidationError,
    LiquidSpeciesNotFoundError,
    build_liquid_database,
    get_common_cp_temperature_range,
    get_common_cp_temperature_range_for_pressure,
    get_species_record,
    has_species,
    load_liquid_database,
    resolve_species_name,
    select_pressure_bank,
)


def _make_db_dict(*, species: list[dict], activity_model_default: str | None = None, unifac: dict | None = None) -> dict:
    return {
        "meta": {
            "file_type": "liquid_properties_db",
            "version": 1,
            "units": {
                "temperature": "K",
                "pressure": "Pa",
                "molecular_weight": "kg/mol",
                "molar_volume": "cm^3/mol",
                "density": "kg/m^3",
                "cp_mass": "J/kg/K",
                "enthalpy_mass": "J/kg",
                "thermal_conductivity": "W/m/K",
                "viscosity": "Pa*s",
                "hvap": "J/kg",
            },
            "reference": {
                "T_ref": 298.15,
            },
        },
        "global_models": {
            "liquid_cp_default_model": "shomate",
            "liquid_density_default_model": "merino_log_poly",
            "liquid_conductivity_default_model": "merino_log_poly",
            "liquid_viscosity_default_model": "merino_log_poly",
            "liquid_diffusion_model": "wilke_chang",
            "liquid_mixture_density_model": "merino_x_sqrt_rho",
            "liquid_mixture_conductivity_model": "filippov",
            "liquid_mixture_viscosity_model": "grunberg_nissan",
            "activity_model_default": activity_model_default,
        },
        "unifac": unifac,
        "species": species,
    }


def _make_species(
    name: str,
    *,
    aliases: list[str] | None = None,
    cp_range: tuple[float, float] = (250.0, 900.0),
    activity_model: str | None = None,
    unifac_groups: dict[str, float] | None = None,
) -> dict:
    return {
        "name": name,
        "aliases": aliases or [],
        "molecular_weight": 0.04607,
        "boiling_temperature": 351.5,
        "molar_volume": 58.5,
        "association_factor": 1.5,
        "T_ref": 298.15,
        "cp_model": "shomate",
        "cp_T_range": list(cp_range),
        "cp_coeffs": {
            "A": 1.0,
            "B": 2.0,
            "C": 3.0,
            "D": 4.0,
            "E": 5.0,
            "F": 6.0,
            "G": 7.0,
            "H": 8.0,
        },
        "hvap_ref": 8.5e5,
        "Tc": 514.0,
        "hvap_model": "watson",
        "hvap_coeffs": {"exponent": 0.38},
        "rho_model": "merino_log_poly",
        "rho_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
        "k_model": "merino_log_poly",
        "k_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
        "mu_model": "merino_log_poly",
        "mu_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
        "activity_model": activity_model,
        "unifac_groups": unifac_groups,
    }


def _make_species_with_pressure_banks(
    name: str,
    *,
    p_fits: tuple[float, ...],
    cp_ranges: tuple[tuple[float, float], ...],
) -> dict:
    if len(p_fits) != len(cp_ranges):
        raise AssertionError("p_fits and cp_ranges must have the same length")
    pressure_banks = []
    for idx, (p_fit, cp_range) in enumerate(zip(p_fits, cp_ranges, strict=True)):
        pressure_banks.append(
            {
                "p_fit": float(p_fit),
                "boiling_temperature": 351.5 + idx,
                "T_ref": 298.15,
                "cp_model": "shomate",
                "cp_T_range": list(cp_range),
                "cp_coeffs": {
                    "A": 1.0 + idx,
                    "B": 2.0,
                    "C": 3.0,
                    "D": 4.0,
                    "E": 5.0,
                    "F": 6.0,
                    "G": 7.0,
                    "H": 8.0,
                },
                "hvap_ref": 8.5e5,
                "Tc": 514.0,
                "hvap_model": "watson",
                "hvap_coeffs": {"exponent": 0.38},
                "rho_model": "merino_log_poly",
                "rho_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
                "k_model": "merino_log_poly",
                "k_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
                "mu_model": "merino_log_poly",
                "mu_coeffs": {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0, "E": 5.0, "F": 6.0},
            }
        )
    return {
        "name": name,
        "aliases": [],
        "molecular_weight": 0.04607,
        "boiling_temperature_atm": 351.5,
        "molar_volume": 58.5,
        "association_factor": 1.5,
        "Tc": 514.0,
        "activity_model": None,
        "unifac_groups": None,
        "pressure_banks": pressure_banks,
    }


def _write_db(tmp_path: Path, data: dict) -> Path:
    db_path = tmp_path / "liquid_db.yaml"
    db_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return db_path


def _make_run_config(db_path: Path, liq_names: tuple[str, ...]) -> RunConfig:
    gas_names = tuple(f"{name}_g" for name in liq_names) + ("N2",)
    liq_full_to_gas_full = np.array(list(range(len(liq_names))), dtype=np.int64)
    if len(liq_names) == 1:
        liq_active = ()
        liq_closure = None
        liq_full_to_reduced = np.array([-1], dtype=np.int64)
        liq_reduced_to_full = np.array([], dtype=np.int64)
        unknowns_profile = "U_A"
        liquid_y0 = np.array([1.0], dtype=np.float64)
    else:
        liq_active = liq_names[:-1]
        liq_closure = liq_names[-1]
        liq_full_to_reduced = np.arange(len(liq_names), dtype=np.int64)
        liq_full_to_reduced[-1] = -1
        liq_reduced_to_full = np.arange(len(liq_names) - 1, dtype=np.int64)
        unknowns_profile = "U_B"
        liquid_y0 = np.zeros(len(liq_names), dtype=np.float64)
        liquid_y0[0] = 1.0

    gas_active = gas_names[:-1]
    gas_closure = gas_names[-1]
    gas_full_to_reduced = np.arange(len(gas_names), dtype=np.int64)
    gas_full_to_reduced[-1] = -1
    gas_reduced_to_full = np.arange(len(gas_names) - 1, dtype=np.int64)
    gas_y0 = np.zeros(len(gas_names), dtype=np.float64)
    gas_y0[-1] = 1.0
    y_vap = np.zeros(len(gas_names), dtype=np.float64)
    y_vap[0] = 1.0e-6

    return RunConfig(
        case_name="liquid_db_case",
        case_description="liquid db test",
        paths=CasePaths(
            config_path=Path("case.yaml"),
            case_root=db_path.parent,
            gas_mechanism_path=db_path.parent / "gas.yaml",
            liquid_database_path=db_path,
            output_root=db_path.parent / "out",
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
            gas_closure_species=gas_closure,
            liquid_closure_species=liq_closure,
            liquid_to_gas_species_map={liq: gas_names[i] for i, liq in enumerate(liq_names)},
        ),
        species_maps=SpeciesMaps(
            liq_full_names=liq_names,
            liq_active_names=liq_active,
            liq_closure_name=liq_closure,
            gas_full_names=gas_names,
            gas_active_names=gas_active,
            gas_closure_name=gas_closure,
            liq_full_to_reduced=liq_full_to_reduced,
            liq_reduced_to_full=liq_reduced_to_full,
            gas_full_to_reduced=gas_full_to_reduced,
            gas_reduced_to_full=gas_reduced_to_full,
            liq_full_to_gas_full=liq_full_to_gas_full,
        ),
        time_stepper=TimeStepperConfig(
            t0=0.0,
            t_end=1.0e-3,
            dt_start=1.0e-6,
            dt_min=1.0e-9,
            dt_max=1.0e-4,
            retry_max_per_step=2,
            q_success_for_growth=2,
            growth_factor=1.1,
            shrink_factor=0.5,
        ),
        outer_stepper=OuterStepperConfig(
            outer_max_iter=4,
            eps_dot_a_tol=1.0e-6,
            corrector_relaxation=1.0,
        ),
        inner_solver=InnerSolverConfig(
            inner_max_iter=20,
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
            T_min_l=250.0,
            T_max_l=700.0,
            T_min_g=250.0,
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
            write_spatial_species=False,
            write_time_series_scalars=True,
            write_time_series_species=False,
            snapshot_format="npz",
        ),
        validation=ValidationConfig(
            enable_mass_balance_check=True,
            enable_energy_balance_check=True,
            enable_state_bounds_check=True,
        ),
        unknowns_profile=unknowns_profile,
    )


def test_load_liquid_database_single_species(tmp_path: Path) -> None:
    db_path = _write_db(tmp_path, _make_db_dict(species=[_make_species("ethanol", aliases=["EtOH"])]))
    db = load_liquid_database(db_path)
    rec = get_species_record(db, "ethanol")

    assert db.meta.file_type == "liquid_properties_db"
    assert len(db.species_by_name) == 1
    assert rec.name == "ethanol"
    assert resolve_species_name(db, "EtOH") == "ethanol"


def test_load_liquid_database_multispecies_and_common_range(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species("ethanol", cp_range=(250.0, 800.0)),
                _make_species("water", aliases=["H2O_l"], cp_range=(300.0, 700.0)),
            ]
        ),
    )
    db = load_liquid_database(db_path)

    assert len(db.species_by_name) == 2
    assert resolve_species_name(db, "H2O_l") == "water"
    assert get_common_cp_temperature_range(db, ("ethanol", "water")) == pytest.approx((300.0, 700.0))


def test_select_pressure_bank_uses_log_pressure_distance(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species_with_pressure_banks(
                    "ethanol",
                    p_fits=(1.0e5, 5.0e5, 2.0e7),
                    cp_ranges=((250.0, 400.0), (260.0, 450.0), (300.0, 700.0)),
                )
            ]
        ),
    )
    db = load_liquid_database(db_path)
    rec = get_species_record(db, "ethanol")

    assert select_pressure_bank(rec, 2.0e5).p_fit == pytest.approx(1.0e5)
    assert select_pressure_bank(rec, 7.0e5).p_fit == pytest.approx(5.0e5)
    assert select_pressure_bank(rec, 1.0e7).p_fit == pytest.approx(2.0e7)


def test_pressure_banks_drive_common_cp_temperature_range(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species_with_pressure_banks(
                    "ethanol",
                    p_fits=(1.0e5, 2.0e6),
                    cp_ranges=((250.0, 350.0), (300.0, 600.0)),
                ),
                _make_species_with_pressure_banks(
                    "water",
                    p_fits=(1.0e5, 2.0e6),
                    cp_ranges=((280.0, 340.0), (320.0, 550.0)),
                ),
            ]
        ),
    )
    db = load_liquid_database(db_path)

    assert get_common_cp_temperature_range_for_pressure(
        db, ("ethanol", "water"), p_env=1.0e5
    ) == pytest.approx((280.0, 340.0))
    assert get_common_cp_temperature_range_for_pressure(
        db, ("ethanol", "water"), p_env=2.0e6
    ) == pytest.approx((320.0, 550.0))


def test_missing_meta_raises(tmp_path: Path) -> None:
    db_path = _write_db(tmp_path, {"species": [_make_species("ethanol")], "global_models": {}})
    with pytest.raises(LiquidDatabaseFormatError, match="meta"):
        load_liquid_database(db_path)


def test_wrong_file_type_raises(tmp_path: Path) -> None:
    data = _make_db_dict(species=[_make_species("ethanol")])
    data["meta"]["file_type"] = "not_liquid_db"
    db_path = _write_db(tmp_path, data)
    with pytest.raises(LiquidDatabaseFormatError, match="file_type"):
        load_liquid_database(db_path)


def test_missing_species_raises(tmp_path: Path) -> None:
    data = _make_db_dict(species=[])
    db_path = _write_db(tmp_path, data)
    with pytest.raises(LiquidDatabaseFormatError, match="species must not be empty"):
        load_liquid_database(db_path)


def test_missing_molecular_weight_raises(tmp_path: Path) -> None:
    rec = _make_species("ethanol")
    del rec["molecular_weight"]
    db_path = _write_db(tmp_path, _make_db_dict(species=[rec]))
    with pytest.raises(LiquidDatabaseValidationError, match="molecular_weight"):
        load_liquid_database(db_path)


def test_shomate_missing_coeff_raises(tmp_path: Path) -> None:
    rec = _make_species("ethanol")
    del rec["cp_coeffs"]["H"]
    db_path = _write_db(tmp_path, _make_db_dict(species=[rec]))
    with pytest.raises(LiquidDatabaseValidationError, match="cp_coeffs"):
        load_liquid_database(db_path)


def test_merino_missing_coeff_raises(tmp_path: Path) -> None:
    rec = _make_species("ethanol")
    del rec["rho_coeffs"]["F"]
    db_path = _write_db(tmp_path, _make_db_dict(species=[rec]))
    with pytest.raises(LiquidDatabaseValidationError, match="rho_coeffs"):
        load_liquid_database(db_path)


def test_invalid_cp_range_raises(tmp_path: Path) -> None:
    rec = _make_species("ethanol")
    rec["cp_T_range"] = [500.0, 300.0]
    db_path = _write_db(tmp_path, _make_db_dict(species=[rec]))
    with pytest.raises(LiquidDatabaseValidationError, match="cp_T_range"):
        load_liquid_database(db_path)


def test_nonpositive_hvap_ref_raises(tmp_path: Path) -> None:
    rec = _make_species("ethanol")
    rec["hvap_ref"] = 0.0
    db_path = _write_db(tmp_path, _make_db_dict(species=[rec]))
    with pytest.raises(LiquidDatabaseValidationError, match="hvap_ref"):
        load_liquid_database(db_path)


def test_alias_conflict_raises(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species("ethanol", aliases=["fuel"]),
                _make_species("water", aliases=["fuel"]),
            ]
        ),
    )
    with pytest.raises(LiquidDatabaseValidationError, match="alias 'fuel'"):
        load_liquid_database(db_path)


def test_duplicate_species_name_raises(tmp_path: Path) -> None:
    db_path = _write_db(tmp_path, _make_db_dict(species=[_make_species("ethanol"), _make_species("ethanol")]))
    with pytest.raises(LiquidDatabaseValidationError, match="duplicate liquid species name"):
        load_liquid_database(db_path)


def test_unifac_required_by_global_default_raises_without_section(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol")],
            activity_model_default="UNIFAC",
            unifac=None,
        ),
    )
    with pytest.raises(LiquidDatabaseValidationError, match="requires top-level unifac"):
        load_liquid_database(db_path)


def test_unifac_required_by_species_raises_without_groups(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol", activity_model="UNIFAC", unifac_groups=None)],
            unifac={
                "enabled": True,
                "groups": {"CH3": {"R": 1.0, "Q": 1.0}},
                "interactions": {"CH3": {"CH3": 0.0}},
            },
        ),
    )
    with pytest.raises(LiquidDatabaseValidationError, match="unifac_groups"):
        load_liquid_database(db_path)


def test_unifac_missing_groups_raises(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol", activity_model="UNIFAC", unifac_groups={"CH3": 2})],
            unifac={"enabled": True, "interactions": {"CH3": {"CH3": 0.0}}},
        ),
    )
    with pytest.raises((LiquidDatabaseFormatError, LiquidDatabaseValidationError), match="unifac.groups"):
        load_liquid_database(db_path)


def test_unifac_missing_interactions_raises(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol", activity_model="UNIFAC", unifac_groups={"CH3": 2})],
            unifac={"enabled": True, "groups": {"CH3": {"R": 1.0, "Q": 1.0}}},
        ),
    )
    with pytest.raises((LiquidDatabaseFormatError, LiquidDatabaseValidationError), match="unifac.interactions"):
        load_liquid_database(db_path)


def test_unifac_group_counts_reject_negative_values(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol", activity_model="UNIFAC", unifac_groups={"CH3": -1})],
            unifac={
                "enabled": True,
                "groups": {"CH3": {"R": 1.0, "Q": 1.0}},
                "interactions": {"CH3": {"CH3": 0.0}},
            },
        ),
    )
    with pytest.raises(LiquidDatabaseValidationError, match="must be non-negative"):
        load_liquid_database(db_path)


def test_unifac_group_counts_reject_non_integer_values(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[_make_species("ethanol", activity_model="UNIFAC", unifac_groups={"CH3": 1.5})],
            unifac={
                "enabled": True,
                "groups": {"CH3": {"R": 1.0, "Q": 1.0}},
                "interactions": {"CH3": {"CH3": 0.0}},
            },
        ),
    )
    with pytest.raises(LiquidDatabaseValidationError, match="integer-like"):
        load_liquid_database(db_path)


def test_build_liquid_database_with_run_config_passes(tmp_path: Path) -> None:
    db_path = _write_db(tmp_path, _make_db_dict(species=[_make_species("ethanol")]))
    run_cfg = _make_run_config(db_path, ("ethanol",))

    db = build_liquid_database(run_cfg)

    assert has_species(db, "ethanol")
    assert get_species_record(db, "ethanol").name == "ethanol"


def test_build_liquid_database_missing_required_species_raises(tmp_path: Path) -> None:
    db_path = _write_db(tmp_path, _make_db_dict(species=[_make_species("water")]))
    run_cfg = _make_run_config(db_path, ("ethanol",))

    with pytest.raises(LiquidSpeciesNotFoundError, match="ethanol"):
        build_liquid_database(run_cfg)


def test_build_liquid_database_rejects_empty_common_cp_range_early(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species("ethanol", cp_range=(250.0, 300.0)),
                _make_species("water", cp_range=(350.0, 400.0)),
            ]
        ),
    )
    run_cfg = _make_run_config(db_path, ("ethanol", "water"))

    with pytest.raises(LiquidDatabaseRangeError, match="do not share"):
        build_liquid_database(run_cfg)


def test_common_cp_temperature_range_empty_intersection_raises(tmp_path: Path) -> None:
    db_path = _write_db(
        tmp_path,
        _make_db_dict(
            species=[
                _make_species("ethanol", cp_range=(250.0, 300.0)),
                _make_species("water", cp_range=(350.0, 400.0)),
            ]
        ),
    )
    db = load_liquid_database(db_path)

    with pytest.raises(LiquidDatabaseRangeError, match="do not share"):
        get_common_cp_temperature_range(db, ("ethanol", "water"))
