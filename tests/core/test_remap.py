from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.grid import build_initial_grid
from core.preprocess import normalize_config
from core.remap import (
    RemapError,
    _build_old_conservative_contents,
    _complete_newly_exposed_subvolume_gas_near,
    _complete_newly_exposed_subvolume_liquid,
    _compute_overlap_1d,
    _compute_overlap_matrix_spherical,
    _compute_uncovered_new_volume,
    _faces_for_region,
    _gas_local_slice,
    _identity_copy_region3_gas_contents,
    _remap_phase_contents_from_overlap,
    build_old_contents_on_current_geometry,
    build_old_state_on_current_geometry,
    summarize_remap_diagnostics,
)
from core.types import GeometryState, InterfaceState, Mesh1D, State


def make_minimal_valid_raw_config() -> dict:
    return {
        "case": {"name": "remap_case", "description": "remap test"},
        "paths": {
            "liquid_database_path": "data/liquid_db.yaml",
            "gas_mechanism_path": "mech/gas.yaml",
            "output_root": "out/",
        },
        "mesh": {
            "a0": 1.0e-4,
            "r_end": 1.0e-2,
            "n_liq": 6,
            "n_gas_near": 8,
            "far_stretch_ratio": 1.03,
        },
        "species": {
            "gas_closure_species": "N2",
            "liquid_closure_species": None,
            "liquid_to_gas_species_map": {"ethanol": "C2H5OH"},
        },
        "initialization": {
            "gas_temperature": 1000.0,
            "gas_pressure": 101325.0,
            "liquid_temperature": 300.0,
            "gas_composition": {"O2": 0.21, "N2": 0.79},
            "liquid_composition": {"ethanol": 1.0},
            "t_init_T": 1.0e-5,
            "Y_vap_if0": {"C2H5OH": 0.10},
        },
        "time_stepper": {
            "t0": 0.0,
            "t_end": 1.0e-3,
            "dt_start": 1.0e-6,
            "dt_min": 1.0e-9,
            "dt_max": 1.0e-4,
            "retry_max_per_step": 4,
            "q_success_for_growth": 3,
            "growth_factor": 1.2,
            "shrink_factor": 0.5,
        },
        "outer_stepper": {
            "outer_max_iter": 8,
            "eps_dot_a_tol": 1.0e-6,
            "corrector_relaxation": 1.0,
        },
        "solver_inner_petsc": {
            "inner_max_iter": 50,
            "snes_rtol": 1.0e-8,
            "snes_atol": 1.0e-10,
            "snes_stol": 1.0e-12,
            "ksp_rtol": 1.0e-8,
            "pc_type": "ilu",
            "ksp_type": "gmres",
            "use_fieldsplit": False,
        },
        "recovery": {
            "rho_min": 1.0e-12,
            "m_min": 1.0e-20,
            "species_recovery_eps_abs": 1.0e-14,
            "Y_sum_tol": 1.0e-10,
            "Y_hard_tol": 1.0e-6,
            "h_abs_tol": 1.0e-8,
            "h_rel_tol": 1.0e-10,
            "h_check_tol": 1.0e-8,
            "T_step_tol": 1.0e-8,
            "T_min_l": 200.0,
            "T_max_l": 800.0,
            "T_min_g": 200.0,
            "T_max_g": 4000.0,
            "liquid_h_inv_max_iter": 50,
            "cp_min": 1.0,
            "gas_h_inv_max_iter": 50,
            "use_cantera_hpy_first": True,
        },
        "diagnostics": {
            "verbose_interface_panel": False,
            "verbose_property_warnings": True,
            "write_step_diag": True,
            "write_interface_diag": True,
            "write_failure_report": True,
            "output_every_n_steps": 10,
        },
        "output": {
            "write_spatial_fields": True,
            "write_spatial_species": False,
            "write_time_series_scalars": True,
            "write_time_series_species": False,
            "snapshot_format": "npz",
        },
        "validation": {
            "enable_mass_balance_check": True,
            "enable_energy_balance_check": True,
            "enable_state_bounds_check": True,
        },
    }


def _prepare_case_files(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    data_dir = case_dir / "data"
    mech_dir = case_dir / "mech"
    data_dir.mkdir(parents=True)
    mech_dir.mkdir(parents=True)
    (data_dir / "liquid_db.yaml").write_text("species: {}\n", encoding="utf-8")
    (mech_dir / "gas.yaml").write_text("dummy: true\n", encoding="utf-8")
    config_path = case_dir / "case.yaml"
    config_path.write_text("case: {}\n", encoding="utf-8")
    return config_path


def _patch_gas_species(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.preprocess._load_gas_species_names",
        lambda mechanism_path, phase_name="gas": ("N2", "O2", "C2H5OH"),
    )


def _make_run_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = _prepare_case_files(tmp_path)
    _patch_gas_species(monkeypatch)
    return normalize_config(make_minimal_valid_raw_config(), source_path=config_path)


def _make_geometry(run_cfg, *, a: float, dot_a: float = -0.2) -> GeometryState:
    return GeometryState(
        t=0.0,
        dt=1.0e-6,
        a=a,
        dot_a=dot_a,
        r_end=run_cfg.mesh.r_end,
        step_index=0,
        outer_iter_index=0,
    )


def _make_old_state(mesh: Mesh1D) -> State:
    n_liq = mesh.n_liq
    n_gas = mesh.n_gas
    Tl = np.linspace(320.0, 330.0, n_liq)
    Tg = np.linspace(800.0, 830.0, n_gas)
    Yl_full = np.ones((n_liq, 1), dtype=np.float64)
    Yg_full = np.tile(np.array([[0.70, 0.20, 0.10]], dtype=np.float64), (n_gas, 1))
    rho_l = np.linspace(700.0, 710.0, n_liq)
    rho_g = np.linspace(1.0, 1.2, n_gas)
    hl = np.linspace(100.0, 110.0, n_liq)
    hg = np.linspace(200.0, 220.0, n_gas)
    return State(
        Tl=Tl,
        Yl_full=Yl_full,
        Tg=Tg,
        Yg_full=Yg_full,
        interface=InterfaceState(
            Ts=340.0,
            mpp=0.1,
            Ys_g_full=np.array([0.72, 0.18, 0.10], dtype=np.float64),
            Ys_l_full=np.array([1.0], dtype=np.float64),
        ),
        rho_l=rho_l,
        rho_g=rho_g,
        hl=hl,
        hg=hg,
        time=1.0e-6,
        state_id="old_state",
    )


def _rebuild_mesh_with_faces(mesh: Mesh1D, new_faces: np.ndarray) -> Mesh1D:
    new_faces = np.asarray(new_faces, dtype=np.float64)
    r_centers = 0.5 * (new_faces[:-1] + new_faces[1:])
    volumes = (4.0 * np.pi / 3.0) * (new_faces[1:] ** 3 - new_faces[:-1] ** 3)
    face_areas = 4.0 * np.pi * new_faces**2
    dr = new_faces[1:] - new_faces[:-1]
    return Mesh1D(
        r_faces=new_faces,
        r_centers=r_centers,
        volumes=volumes,
        face_areas=face_areas,
        dr=dr,
        region_slices=mesh.region_slices,
        face_owner_phase=mesh.face_owner_phase,
        interface_face_index=mesh.interface_face_index,
        interface_cell_liq=mesh.interface_cell_liq,
        interface_cell_gas=mesh.interface_cell_gas,
    )


def test_build_old_conservative_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)

    contents = _build_old_conservative_contents(old_state, mesh)
    liq_vol = mesh.volumes[mesh.region_slices.liq]
    gas_vol = mesh.volumes[mesh.region_slices.gas_all]

    assert np.allclose(contents.mass_l, old_state.rho_l * liq_vol)
    assert np.allclose(contents.species_mass_l, old_state.rho_l[:, None] * old_state.Yl_full * liq_vol[:, None])
    assert np.allclose(contents.enthalpy_l, old_state.rho_l * old_state.hl * liq_vol)
    assert np.allclose(contents.mass_g, old_state.rho_g * gas_vol)
    assert np.allclose(contents.species_mass_g, old_state.rho_g[:, None] * old_state.Yg_full * gas_vol[:, None])
    assert np.allclose(contents.enthalpy_g, old_state.rho_g * old_state.hg * gas_vol)


def test_compute_overlap_1d_basic_cases() -> None:
    assert _compute_overlap_1d(0.0, 1.0, 2.0, 3.0) == pytest.approx(0.0)
    assert _compute_overlap_1d(0.0, 2.0, 0.5, 1.5) == pytest.approx(1.0)
    assert _compute_overlap_1d(0.0, 1.0, 0.25, 1.5) == pytest.approx(0.75)


def test_compute_overlap_matrix_spherical() -> None:
    faces = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    overlap = _compute_overlap_matrix_spherical(faces, faces)
    cell_volumes = np.array(
        [(4.0 * np.pi / 3.0) * (faces[i + 1] ** 3 - faces[i] ** 3) for i in range(faces.size - 1)],
        dtype=np.float64,
    )

    assert np.allclose(np.diag(overlap), cell_volumes)
    assert np.allclose(overlap - np.diag(np.diag(overlap)), 0.0)


def test_identity_remap_on_same_mesh_preserves_contents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)

    old_contents = _build_old_conservative_contents(old_state, mesh)
    new_contents = build_old_contents_on_current_geometry(
        old_state=old_state,
        old_mesh=mesh,
        new_mesh=mesh,
    )

    assert np.allclose(new_contents.mass_l, old_contents.mass_l)
    assert np.allclose(new_contents.species_mass_l, old_contents.species_mass_l)
    assert np.allclose(new_contents.enthalpy_l, old_contents.enthalpy_l)
    assert np.allclose(new_contents.mass_g, old_contents.mass_g)
    assert np.allclose(new_contents.species_mass_g, old_contents.species_mass_g)
    assert np.allclose(new_contents.enthalpy_g, old_contents.enthalpy_g)


def test_gas_region3_identity_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)
    old_contents = _build_old_conservative_contents(old_state, mesh)

    mass_far, species_far, enthalpy_far = _identity_copy_region3_gas_contents(old_contents, mesh, mesh)
    gas_far_local = _gas_local_slice(mesh, mesh.region_slices.gas_far)

    assert np.allclose(mass_far, old_contents.mass_g[gas_far_local])
    assert np.allclose(species_far, old_contents.species_mass_g[gas_far_local, :])
    assert np.allclose(enthalpy_far, old_contents.enthalpy_g[gas_far_local])


def test_liquid_newly_exposed_volume_is_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    old_mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=0.8 * run_cfg.mesh.a0))
    new_mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(old_mesh)
    old_contents = _build_old_conservative_contents(old_state, old_mesh)

    old_liq_faces = _faces_for_region(old_mesh, old_mesh.region_slices.liq)
    new_liq_faces = _faces_for_region(new_mesh, new_mesh.region_slices.liq)
    overlap = _compute_overlap_matrix_spherical(old_liq_faces, new_liq_faces)
    new_mass, new_species_mass, new_enthalpy = _remap_phase_contents_from_overlap(
        old_mass=old_contents.mass_l,
        old_species_mass=old_contents.species_mass_l,
        old_enthalpy=old_contents.enthalpy_l,
        old_cell_volumes=old_mesh.volumes[old_mesh.region_slices.liq],
        overlap_matrix=overlap,
    )
    baseline_mass = new_mass.copy()
    uncovered = _compute_uncovered_new_volume(new_liq_faces, overlap)

    _complete_newly_exposed_subvolume_liquid(
        new_mass=new_mass,
        new_species_mass=new_species_mass,
        new_enthalpy=new_enthalpy,
        uncovered_volume=uncovered,
        reference_rho=float(old_state.rho_l[-1]),
        reference_y_full=old_state.Yl_full[-1, :],
        reference_h=float(old_state.hl[-1]),
        tol=0.0,
    )

    assert np.any(uncovered > 0.0)
    assert np.all(new_mass >= baseline_mass)
    assert np.any(new_mass > baseline_mass)


def test_gas_near_newly_exposed_volume_is_completed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    old_mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    new_mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=0.8 * run_cfg.mesh.a0))
    old_state = _make_old_state(old_mesh)
    old_contents = _build_old_conservative_contents(old_state, old_mesh)

    old_gas_faces = _faces_for_region(old_mesh, old_mesh.region_slices.gas_near)
    new_gas_faces = _faces_for_region(new_mesh, new_mesh.region_slices.gas_near)
    old_gas_near_local = _gas_local_slice(old_mesh, old_mesh.region_slices.gas_near)
    overlap = _compute_overlap_matrix_spherical(old_gas_faces, new_gas_faces)
    new_mass, new_species_mass, new_enthalpy = _remap_phase_contents_from_overlap(
        old_mass=old_contents.mass_g[old_gas_near_local],
        old_species_mass=old_contents.species_mass_g[old_gas_near_local, :],
        old_enthalpy=old_contents.enthalpy_g[old_gas_near_local],
        old_cell_volumes=old_mesh.volumes[old_mesh.region_slices.gas_near],
        overlap_matrix=overlap,
    )
    baseline_mass = new_mass.copy()
    uncovered = _compute_uncovered_new_volume(new_gas_faces, overlap)

    _complete_newly_exposed_subvolume_gas_near(
        new_mass=new_mass,
        new_species_mass=new_species_mass,
        new_enthalpy=new_enthalpy,
        uncovered_volume=uncovered,
        reference_rho=float(old_state.rho_g[0]),
        reference_y_full=old_state.Yg_full[0, :],
        reference_h=float(old_state.hg[0]),
        tol=0.0,
    )

    assert np.any(uncovered > 0.0)
    assert np.all(new_mass >= baseline_mass)
    assert np.any(new_mass > baseline_mass)


def test_total_mass_conserved_on_identity_remap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)
    old_contents = _build_old_conservative_contents(old_state, mesh)
    new_contents = build_old_contents_on_current_geometry(old_state=old_state, old_mesh=mesh, new_mesh=mesh)
    diag = summarize_remap_diagnostics(old_contents, new_contents)

    assert diag["mass_l_after"] == pytest.approx(diag["mass_l_before"])
    assert diag["mass_g_after"] == pytest.approx(diag["mass_g_before"])
    assert diag["species_mass_l_after"] == pytest.approx(diag["species_mass_l_before"])
    assert diag["species_mass_g_after"] == pytest.approx(diag["species_mass_g_before"])
    assert diag["enthalpy_l_after"] == pytest.approx(diag["enthalpy_l_before"])
    assert diag["enthalpy_g_after"] == pytest.approx(diag["enthalpy_g_before"])


def test_build_old_state_on_current_geometry_returns_formal_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    old_mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    new_geometry = _make_geometry(run_cfg, a=0.9 * run_cfg.mesh.a0)
    new_mesh = build_initial_grid(run_cfg, new_geometry)
    old_state = _make_old_state(old_mesh)

    liquid_thermo = object()
    gas_thermo = object()
    captured = {}

    def fake_recover_state_from_contents(*, contents, mesh, species_maps, recovery_config, **kwargs):
        captured["liquid_thermo"] = kwargs["liquid_thermo"]
        captured["gas_thermo"] = kwargs["gas_thermo"]
        Yl_full = contents.species_mass_l / contents.mass_l[:, None]
        Yg_full = contents.species_mass_g / contents.mass_g[:, None]
        return State(
            Tl=np.full(mesh.n_liq, 300.0),
            Yl_full=Yl_full,
            Tg=np.full(mesh.n_gas, 800.0),
            Yg_full=Yg_full,
            interface=InterfaceState(
                Ts=350.0,
                mpp=0.0,
                Ys_g_full=Yg_full[0, :],
                Ys_l_full=Yl_full[-1, :],
            ),
        )

    monkeypatch.setattr("core.remap.recover_state_from_contents", fake_recover_state_from_contents)

    old_on_current = build_old_state_on_current_geometry(
        old_state=old_state,
        old_mesh=old_mesh,
        new_mesh=new_mesh,
        geometry=new_geometry,
        recovery_config=run_cfg.recovery,
        species_maps=run_cfg.species_maps,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
    )

    assert old_on_current.geometry is new_geometry
    assert old_on_current.mesh is new_mesh
    assert old_on_current.contents.n_liq_cells == new_mesh.n_liq
    assert old_on_current.contents.n_gas_cells == new_mesh.n_gas
    assert old_on_current.state.n_liq_cells == new_mesh.n_liq
    assert old_on_current.state.n_gas_cells == new_mesh.n_gas
    assert captured["liquid_thermo"] is liquid_thermo
    assert captured["gas_thermo"] is gas_thermo


def test_region3_identity_requires_same_fixed_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)
    old_contents = _build_old_conservative_contents(old_state, mesh)

    altered_faces = mesh.r_faces.copy()
    altered_faces[mesh.region_slices.gas_far.start + 1] += 1.0e-8
    bad_mesh = _rebuild_mesh_with_faces(mesh, altered_faces)

    with pytest.raises(RemapError, match="region-3 faces"):
        _identity_copy_region3_gas_contents(old_contents, mesh, bad_mesh)


def test_build_old_conservative_contents_requires_density_and_enthalpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_cfg = _make_run_cfg(tmp_path, monkeypatch)
    mesh = build_initial_grid(run_cfg, _make_geometry(run_cfg, a=run_cfg.mesh.a0))
    old_state = _make_old_state(mesh)
    bad_state = replace(old_state, rho_l=None)

    with pytest.raises(RemapError, match="missing required derived fields"):
        _build_old_conservative_contents(bad_state, mesh)


def test_negative_overlap_volume_raises() -> None:
    with pytest.raises(RemapError, match="strictly increasing"):
        _compute_overlap_matrix_spherical(
            np.array([0.0, 1.0, 0.5], dtype=np.float64),
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
        )
