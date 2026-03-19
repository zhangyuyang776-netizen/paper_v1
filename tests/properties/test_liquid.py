from __future__ import annotations

import numpy as np
import pytest

from core.state_recovery import recover_state_from_contents
from core.types import ConservativeContents, InterfaceState, Mesh1D, RecoveryConfig, RegionSlices, SpeciesMaps
from properties.liquid import (
    DEFAULT_DIFFUSIVITY_MODEL,
    DEFAULT_MIXTURE_CONDUCTIVITY_MODEL,
    DEFAULT_MIXTURE_DENSITY_MODEL,
    DEFAULT_MIXTURE_VISCOSITY_MODEL,
    LiquidThermoModelError,
    LiquidThermoValidationError,
    build_liquid_thermo_model,
)
from properties.liquid_db import LiquidDBGlobalModels, LiquidDatabase, LiquidDBMeta, LiquidPureSpeciesRecord


class FakeGasThermo:
    def __init__(self, cp: float) -> None:
        self.cp = float(cp)

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float:
        return self.cp * float(T)


def make_species_record(
    name: str,
    *,
    molecular_weight: float,
    cp_A: float,
    cp_T_range: tuple[float, float],
    rho_value: float,
    k_value: float,
    mu_value: float,
    alias: str | None = None,
) -> LiquidPureSpeciesRecord:
    def merino_coeffs(value: float) -> dict[str, float]:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": float(np.log(value)), "E": 0.0, "F": 0.0}

    return LiquidPureSpeciesRecord(
        name=name,
        aliases=(() if alias is None else (alias,)),
        molecular_weight=molecular_weight,
        boiling_temperature=350.0,
        molar_volume=10.0,
        association_factor=1.0,
        T_ref=298.15,
        cp_model="shomate",
        cp_T_range=cp_T_range,
        cp_coeffs={"A": cp_A, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
        hvap_ref=1.0e5,
        Tc=500.0,
        hvap_model="watson",
        hvap_coeffs={},
        rho_model="merino_log_poly",
        rho_coeffs=merino_coeffs(rho_value),
        k_model="merino_log_poly",
        k_coeffs=merino_coeffs(k_value),
        mu_model="merino_log_poly",
        mu_coeffs=merino_coeffs(mu_value),
        activity_model=None,
        unifac_groups=None,
    )


def make_liquid_database(*, multi: bool) -> LiquidDatabase:
    species_a = make_species_record(
        "ethanol",
        molecular_weight=0.04607,
        cp_A=100.0,
        cp_T_range=(250.0, 600.0),
        rho_value=780.0,
        k_value=0.16,
        mu_value=1.2e-3,
        alias="EtOH",
    )
    species_by_name: dict[str, LiquidPureSpeciesRecord] = {"ethanol": species_a}
    alias_to_name = {"EtOH": "ethanol"}
    if multi:
        species_b = make_species_record(
            "water",
            molecular_weight=0.01801528,
            cp_A=75.0,
            cp_T_range=(280.0, 550.0),
            rho_value=995.0,
            k_value=0.60,
            mu_value=0.9e-3,
            alias="H2O_liq",
        )
        species_by_name["water"] = species_b
        alias_to_name["H2O_liq"] = "water"

    return LiquidDatabase(
        meta=LiquidDBMeta(
            file_type="liquid_properties_db",
            version=1,
            units={"molecular_weight": "kg/mol"},
            reference_T=298.15,
        ),
        global_models=LiquidDBGlobalModels(
            liquid_cp_default_model="shomate",
            liquid_density_default_model="merino_log_poly",
            liquid_conductivity_default_model="merino_log_poly",
            liquid_viscosity_default_model="merino_log_poly",
            liquid_diffusion_model="wilke_chang",
            liquid_mixture_density_model="merino_x_sqrt_rho",
            liquid_mixture_conductivity_model="filippov",
            liquid_mixture_viscosity_model="grunberg_nissan",
            activity_model_default=None,
        ),
        unifac=None,
        species_by_name=species_by_name,
        alias_to_name=alias_to_name,
    )


def make_recovery_config() -> RecoveryConfig:
    return RecoveryConfig(
        T_min_l=280.0,
        T_max_l=540.0,
        T_min_g=250.0,
        T_max_g=2000.0,
        liq_h_inv_tol=1.0e-10,
        liq_h_inv_max_iter=200,
        gas_h_inv_tol=1.0e-12,
        gas_h_inv_max_iter=100,
        use_cantera_hpy_first=True,
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


def make_species_maps_single() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol",),
        liq_active_names=(),
        liq_closure_name=None,
        gas_full_names=("N2", "C2H5OH"),
        gas_active_names=("C2H5OH",),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([-1], dtype=np.int64),
        liq_reduced_to_full=np.array([], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0], dtype=np.int64),
        gas_reduced_to_full=np.array([1], dtype=np.int64),
        liq_full_to_gas_full=np.array([1], dtype=np.int64),
    )


def make_species_maps_multi() -> SpeciesMaps:
    return SpeciesMaps(
        liq_full_names=("ethanol", "water"),
        liq_active_names=("ethanol",),
        liq_closure_name="water",
        gas_full_names=("N2", "C2H5OH", "H2O"),
        gas_active_names=("C2H5OH", "H2O"),
        gas_closure_name="N2",
        liq_full_to_reduced=np.array([0, -1], dtype=np.int64),
        liq_reduced_to_full=np.array([0], dtype=np.int64),
        gas_full_to_reduced=np.array([-1, 0, 1], dtype=np.int64),
        gas_reduced_to_full=np.array([1, 2], dtype=np.int64),
        liq_full_to_gas_full=np.array([1, 2], dtype=np.int64),
    )


def make_interface_seed(species_maps: SpeciesMaps) -> InterfaceState:
    return InterfaceState(
        Ts=320.0,
        mpp=0.0,
        Ys_g_full=np.full(species_maps.n_gas_full, 1.0 / species_maps.n_gas_full, dtype=np.float64),
        Ys_l_full=np.full(species_maps.n_liq_full, 1.0 / species_maps.n_liq_full, dtype=np.float64),
    )


def test_build_liquid_thermo_model_single_component() -> None:
    db = make_liquid_database(multi=False)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    assert model.species_names == ("ethanol",)
    assert model.valid_temperature_range() == pytest.approx((250.0, 600.0))


def test_build_liquid_thermo_model_multicomponent_uses_database_global_model_defaults() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("EtOH", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    assert model.species_names == ("ethanol", "water")
    assert model.mixture_density_model == db.global_models.liquid_mixture_density_model
    assert model.mixture_conductivity_model == db.global_models.liquid_mixture_conductivity_model
    assert model.mixture_viscosity_model == db.global_models.liquid_mixture_viscosity_model
    assert model.diffusivity_model == db.global_models.liquid_diffusion_model


def test_build_liquid_thermo_model_allows_explicit_model_overrides() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
        mixture_density_model=DEFAULT_MIXTURE_DENSITY_MODEL,
        mixture_conductivity_model=DEFAULT_MIXTURE_CONDUCTIVITY_MODEL,
        mixture_viscosity_model=DEFAULT_MIXTURE_VISCOSITY_MODEL,
        diffusivity_model=DEFAULT_DIFFUSIVITY_MODEL,
    )
    assert model.mixture_density_model == DEFAULT_MIXTURE_DENSITY_MODEL
    assert model.mixture_conductivity_model == DEFAULT_MIXTURE_CONDUCTIVITY_MODEL
    assert model.mixture_viscosity_model == DEFAULT_MIXTURE_VISCOSITY_MODEL
    assert model.diffusivity_model == DEFAULT_DIFFUSIVITY_MODEL


def test_build_liquid_thermo_model_rejects_unsupported_database_defaults() -> None:
    db = make_liquid_database(multi=True)
    db = LiquidDatabase(
        meta=db.meta,
        global_models=LiquidDBGlobalModels(
            liquid_cp_default_model=db.global_models.liquid_cp_default_model,
            liquid_density_default_model=db.global_models.liquid_density_default_model,
            liquid_conductivity_default_model=db.global_models.liquid_conductivity_default_model,
            liquid_viscosity_default_model=db.global_models.liquid_viscosity_default_model,
            liquid_diffusion_model="maxwell_stefan",
            liquid_mixture_density_model="merino_x_sqrt_rho",
            liquid_mixture_conductivity_model="filippov",
            liquid_mixture_viscosity_model="grunberg_nissan",
            activity_model_default=None,
        ),
        unifac=db.unifac,
        species_by_name=db.species_by_name,
        alias_to_name=db.alias_to_name,
    )
    with pytest.raises(LiquidThermoModelError, match="Unsupported"):
        build_liquid_thermo_model(
            liquid_db=db,
            liquid_species_full=("ethanol", "water"),
            molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
            p_env=101325.0,
        )


def test_valid_temperature_range_returns_common_intersection() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    assert model.valid_temperature_range() == pytest.approx((280.0, 550.0))
    assert model.valid_temperature_range(("ethanol",)) == pytest.approx((250.0, 600.0))
    with pytest.raises(LiquidThermoValidationError, match="subset"):
        model.valid_temperature_range(("methanol",))


def test_pure_cp_and_enthalpy_reference() -> None:
    db = make_liquid_database(multi=False)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    cp = model.pure_cp_vector(320.0)[0]
    h_ref = model.pure_enthalpy_vector(298.15)[0]
    h = model.pure_enthalpy_vector(320.0)[0]
    expected_cp = 100.0 / 0.04607
    expected_h = expected_cp * (320.0 - 298.15)
    assert cp == pytest.approx(expected_cp)
    assert h_ref == pytest.approx(0.0, abs=1.0e-12)
    assert h == pytest.approx(expected_h)


def test_pure_density_conductivity_viscosity_positive() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    assert np.all(model.pure_density_vector(300.0) > 0.0)
    assert np.all(model.pure_conductivity_vector(300.0) > 0.0)
    assert np.all(model.pure_viscosity_vector(300.0) > 0.0)


def test_mixture_properties_single_component_degenerate() -> None:
    db = make_liquid_database(multi=False)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    Y = np.array([1.0], dtype=np.float64)
    T = 310.0
    assert model.density_mass(T, Y) == pytest.approx(model.pure_density_vector(T)[0])
    assert model.cp_mass(T, Y) == pytest.approx(model.pure_cp_vector(T)[0])
    assert model.enthalpy_mass(T, Y) == pytest.approx(model.pure_enthalpy_vector(T)[0])
    assert model.conductivity(T, Y) == pytest.approx(model.pure_conductivity_vector(T)[0])
    assert model.viscosity(T, Y) == pytest.approx(model.pure_viscosity_vector(T)[0])
    assert model.diffusivity(T, Y) is None


def test_batch_methods_match_scalar_evaluation() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    T = np.array([300.0, 320.0], dtype=np.float64)
    Y = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
    rho_batch = model.density_mass_batch(T, Y)
    cp_batch = model.cp_mass_batch(T, Y)
    h_batch = model.enthalpy_mass_batch(T, Y)
    rho_scalar = np.array([model.density_mass(T[i], Y[i, :]) for i in range(2)], dtype=np.float64)
    cp_scalar = np.array([model.cp_mass(T[i], Y[i, :]) for i in range(2)], dtype=np.float64)
    h_scalar = np.array([model.enthalpy_mass(T[i], Y[i, :]) for i in range(2)], dtype=np.float64)
    assert np.allclose(rho_batch, rho_scalar)
    assert np.allclose(cp_batch, cp_scalar)
    assert np.allclose(h_batch, h_scalar)


def test_invalid_mass_fractions_raise() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    with pytest.raises(LiquidThermoValidationError, match="sum to 1"):
        model.enthalpy_mass(300.0, np.array([0.4, 0.5], dtype=np.float64))
    with pytest.raises(LiquidThermoValidationError, match="non-negative"):
        model.density_mass(300.0, np.array([1.1, -0.1], dtype=np.float64))


def test_diffusivity_wilke_chang_returns_positive_vector_for_multicomponent_case() -> None:
    db = make_liquid_database(multi=True)
    model = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    diff = model.diffusivity(300.0, np.array([0.5, 0.5], dtype=np.float64))
    assert isinstance(diff, np.ndarray)
    assert diff.shape == (2,)
    assert np.all(diff > 0.0)


def test_enthalpy_roundtrip_with_state_recovery_single_component() -> None:
    db = make_liquid_database(multi=False)
    liquid_thermo = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol",),
        molecular_weights=np.array([0.04607], dtype=np.float64),
        p_env=101325.0,
    )
    mesh = make_dummy_mesh(n_liq=2, n_gas=1)
    species_maps = make_species_maps_single()
    Yl = np.ones((2, 1), dtype=np.float64)
    Tl_true = np.array([310.0, 330.0], dtype=np.float64)
    rho_l = np.array([780.0, 785.0], dtype=np.float64)
    mass_l = rho_l * mesh.volumes[mesh.region_slices.liq]
    species_mass_l = mass_l[:, None] * Yl
    enthalpy_l = mass_l * liquid_thermo.enthalpy_mass_batch(Tl_true, Yl)
    mass_g = np.array([1.2], dtype=np.float64)
    species_mass_g = np.array([[1.0, 0.2]], dtype=np.float64)
    enthalpy_g = mass_g * np.array([900.0 * 4.0], dtype=np.float64)
    contents = ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )
    state = recover_state_from_contents(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=make_recovery_config(),
        liquid_thermo=liquid_thermo,
        gas_thermo=FakeGasThermo(cp=4.0),
        interface_seed=make_interface_seed(species_maps),
    )
    assert np.allclose(state.Tl, Tl_true, atol=1.0e-8)


def test_enthalpy_roundtrip_with_state_recovery_multicomponent() -> None:
    db = make_liquid_database(multi=True)
    liquid_thermo = build_liquid_thermo_model(
        liquid_db=db,
        liquid_species_full=("ethanol", "water"),
        molecular_weights=np.array([0.04607, 0.01801528], dtype=np.float64),
        p_env=101325.0,
    )
    mesh = make_dummy_mesh(n_liq=2, n_gas=1)
    species_maps = make_species_maps_multi()
    Yl = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float64)
    Tl_true = np.array([300.0, 340.0], dtype=np.float64)
    rho_l = np.array([820.0, 830.0], dtype=np.float64)
    mass_l = rho_l * mesh.volumes[mesh.region_slices.liq]
    species_mass_l = mass_l[:, None] * Yl
    enthalpy_l = mass_l * liquid_thermo.enthalpy_mass_batch(Tl_true, Yl)
    mass_g = np.array([1.5], dtype=np.float64)
    species_mass_g = np.array([[1.1, 0.2, 0.2]], dtype=np.float64)
    enthalpy_g = mass_g * np.array([850.0 * 4.0], dtype=np.float64)
    contents = ConservativeContents(
        mass_l=mass_l,
        species_mass_l=species_mass_l,
        enthalpy_l=enthalpy_l,
        mass_g=mass_g,
        species_mass_g=species_mass_g,
        enthalpy_g=enthalpy_g,
    )
    state = recover_state_from_contents(
        contents=contents,
        mesh=mesh,
        species_maps=species_maps,
        recovery_cfg=make_recovery_config(),
        liquid_thermo=liquid_thermo,
        gas_thermo=FakeGasThermo(cp=4.0),
        interface_seed=make_interface_seed(species_maps),
    )
    assert np.allclose(state.Tl, Tl_true, atol=1.0e-8)
