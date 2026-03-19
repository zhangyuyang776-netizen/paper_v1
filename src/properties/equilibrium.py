from __future__ import annotations

"""Interface equilibrium closure for Eq.(2.19) in Phase 2."""

from dataclasses import dataclass
from typing import Any, Mapping

import cantera as ct
import numpy as np

from properties.liquid import LiquidThermoModel, LiquidThermoValidationError
from properties.gas import GasThermoModel
from properties.liquid_db import LiquidPureSpeciesRecord
from properties.mix_rules import MixRulesValidationError, mass_to_mole_fractions


class InterfaceEquilibriumError(Exception):
    """Base error for interface equilibrium closure."""


class InterfaceEquilibriumValidationError(InterfaceEquilibriumError):
    """Raised when interface equilibrium inputs or mappings are invalid."""


class InterfaceEquilibriumModelError(InterfaceEquilibriumError):
    """Raised when the equilibrium model configuration is inconsistent or unsupported."""


class InterfaceActivityModelError(InterfaceEquilibriumError):
    """Raised when activity-coefficient evaluation cannot be completed safely."""


def _as_1d_float_array(name: str, value: Any, *, expected_size: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise InterfaceEquilibriumValidationError(f"{name} must be a one-dimensional float array")
    if arr.size == 0:
        raise InterfaceEquilibriumValidationError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise InterfaceEquilibriumValidationError(f"{name} must contain only finite values")
    if expected_size is not None and arr.size != expected_size:
        raise InterfaceEquilibriumValidationError(f"{name} must have length {expected_size}")
    return arr


def _validate_temperature(Ts: float) -> float:
    if isinstance(Ts, bool) or not np.isscalar(Ts):
        raise InterfaceEquilibriumValidationError("Ts must be a finite positive scalar")
    value = float(Ts)
    if not np.isfinite(value) or value <= 0.0:
        raise InterfaceEquilibriumValidationError("Ts must be a finite positive scalar")
    return value


def _validate_pressure(P: float) -> float:
    if isinstance(P, bool) or not np.isscalar(P):
        raise InterfaceEquilibriumValidationError("P must be a finite positive scalar")
    value = float(P)
    if not np.isfinite(value) or value <= 0.0:
        raise InterfaceEquilibriumValidationError("P must be a finite positive scalar")
    return value


def _validate_mass_fractions(Y: np.ndarray, n_spec: int, *, atol: float = 1.0e-12) -> np.ndarray:
    arr = _as_1d_float_array("Yl_if_full", Y, expected_size=n_spec)
    if np.any(arr < 0.0):
        raise InterfaceEquilibriumValidationError("Yl_if_full must be non-negative")
    if not np.isclose(float(np.sum(arr)), 1.0, rtol=0.0, atol=atol):
        raise InterfaceEquilibriumValidationError("Yl_if_full must sum to 1 within tolerance")
    return arr


def _require_positive_finite_array(name: str, arr: np.ndarray) -> np.ndarray:
    out = _as_1d_float_array(name, arr)
    if np.any(out <= 0.0):
        raise InterfaceEquilibriumModelError(f"{name} must be strictly positive")
    return out


def _validate_species_mapping(
    *,
    liquid_species_names: tuple[str, ...],
    gas_species_names: tuple[str, ...],
    liquid_to_gas_species_map: Mapping[str, str],
    gas_closure_species: str,
) -> None:
    if set(liquid_to_gas_species_map.keys()) != set(liquid_species_names):
        raise InterfaceEquilibriumValidationError(
            "liquid_to_gas_species_map keys must exactly match liquid_thermo.species_names"
        )
    if gas_closure_species not in gas_species_names:
        raise InterfaceEquilibriumValidationError("gas_closure_species must belong to gas_thermo.species_names")
    gas_targets = tuple(liquid_to_gas_species_map[name] for name in liquid_species_names)
    if len(set(gas_targets)) != len(gas_targets):
        raise InterfaceEquilibriumValidationError("liquid_to_gas_species_map values must be unique")
    for gas_name in gas_targets:
        if gas_name not in gas_species_names:
            raise InterfaceEquilibriumValidationError(
                f"Mapped gas species {gas_name!r} is not present in gas_thermo.species_names"
            )
        if gas_name == gas_closure_species:
            raise InterfaceEquilibriumValidationError("gas_closure_species cannot be used as a condensable mapping target")


def _build_condensable_index_maps(
    liquid_species_names: tuple[str, ...],
    gas_species_names: tuple[str, ...],
    liquid_to_gas_species_map: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    liquid_cond_indices = np.arange(len(liquid_species_names), dtype=np.int64)
    gas_cond_indices = np.array(
        [gas_species_names.index(liquid_to_gas_species_map[name]) for name in liquid_species_names],
        dtype=np.int64,
    )
    return liquid_cond_indices, gas_cond_indices


def _activity_model_name_for_species(records: tuple[LiquidPureSpeciesRecord, ...]) -> str:
    activity_models = {record.activity_model for record in records}
    if None in activity_models:
        raise InterfaceActivityModelError("All condensable liquid species must declare activity_model explicitly")
    if len(activity_models) != 1:
        raise InterfaceActivityModelError("All condensable liquid species must use the same activity_model")
    return str(next(iter(activity_models)))


def _compute_activity_coefficients_ideal(n_cond: int) -> np.ndarray:
    return np.ones(n_cond, dtype=np.float64)


def _compute_activity_coefficients_unifac(
    *,
    Ts: float,
    Xl_cond: np.ndarray,
    liquid_records: tuple[LiquidPureSpeciesRecord, ...],
    unifac_global: Mapping[str, Any],
) -> np.ndarray:
    if not bool(unifac_global.get("enabled", False)):
        raise InterfaceActivityModelError("UNIFAC data is present but disabled")

    groups = unifac_global.get("groups")
    interactions = unifac_global.get("interactions")
    if not isinstance(groups, Mapping) or not isinstance(interactions, Mapping):
        raise InterfaceActivityModelError("UNIFAC groups/interactions must be mappings")

    group_names = tuple(groups.keys())
    q_group = np.array([float(groups[name]["Q"]) for name in group_names], dtype=np.float64)
    r_group = np.array([float(groups[name]["R"]) for name in group_names], dtype=np.float64)
    if np.any(q_group <= 0.0) or np.any(r_group <= 0.0):
        raise InterfaceActivityModelError("UNIFAC group R/Q must be strictly positive")

    nu = np.zeros((len(liquid_records), len(group_names)), dtype=np.float64)
    for i, record in enumerate(liquid_records):
        if record.unifac_groups is None:
            raise InterfaceActivityModelError(f"Species {record.name!r} requires unifac_groups for UNIFAC")
        for group_name, count in record.unifac_groups.items():
            if group_name not in groups:
                raise InterfaceActivityModelError(
                    f"Species {record.name!r} references unknown UNIFAC group {group_name!r}"
                )
            nu[i, group_names.index(group_name)] = float(count)

    r_species = nu @ r_group
    q_species = nu @ q_group
    if np.any(r_species <= 0.0) or np.any(q_species <= 0.0):
        raise InterfaceActivityModelError("UNIFAC species group decomposition must yield positive r and q")

    denom_r = float(np.dot(Xl_cond, r_species))
    denom_q = float(np.dot(Xl_cond, q_species))
    if denom_r <= 0.0 or denom_q <= 0.0:
        raise InterfaceActivityModelError("UNIFAC combinatorial denominators must be positive")

    V = r_species / denom_r
    F = q_species / denom_q
    z = 10.0
    l_species = (z / 2.0) * (r_species - q_species) - (r_species - 1.0)
    ln_gamma_c = (
        np.log(V)
        + 1.0
        - V
        - 5.0 * q_species * (np.log(V / F) + 1.0 - V / F)
    )

    psi = np.zeros((len(group_names), len(group_names)), dtype=np.float64)
    for m, row_name in enumerate(group_names):
        row = interactions[row_name]
        for n, col_name in enumerate(group_names):
            psi[m, n] = np.exp(-float(row[col_name]) / Ts)

    total_group_counts = np.sum(Xl_cond[:, None] * nu, axis=0)
    if np.sum(total_group_counts) <= 0.0:
        raise InterfaceActivityModelError("UNIFAC mixture group counts must be positive")
    X_group = total_group_counts / np.sum(total_group_counts)
    theta = q_group * X_group / float(np.dot(q_group, X_group))

    def ln_capital_gamma(theta_mix: np.ndarray) -> np.ndarray:
        sum_theta_psi_col = theta_mix @ psi
        if np.any(sum_theta_psi_col <= 0.0):
            raise InterfaceActivityModelError("UNIFAC theta-psi column sums must be positive")
        out = np.empty(len(group_names), dtype=np.float64)
        for k in range(len(group_names)):
            denom = psi @ theta_mix
            if np.any(denom <= 0.0):
                raise InterfaceActivityModelError("UNIFAC theta-psi row sums must be positive")
            second_term = np.sum(theta_mix * psi[k, :] / denom)
            out[k] = q_group[k] * (1.0 - np.log(sum_theta_psi_col[k]) - second_term)
        return out

    ln_gamma_group_mix = ln_capital_gamma(theta)
    ln_gamma_r = np.zeros(len(liquid_records), dtype=np.float64)
    for i in range(len(liquid_records)):
        pure_counts = nu[i, :]
        pure_total = np.sum(pure_counts)
        if pure_total <= 0.0:
            raise InterfaceActivityModelError("UNIFAC pure-species group counts must be positive")
        x_group_pure = pure_counts / pure_total
        theta_pure = q_group * x_group_pure / float(np.dot(q_group, x_group_pure))
        ln_gamma_group_pure = ln_capital_gamma(theta_pure)
        ln_gamma_r[i] = np.sum(nu[i, :] * (ln_gamma_group_mix - ln_gamma_group_pure))

    gamma = np.exp(ln_gamma_c + ln_gamma_r)
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0.0):
        raise InterfaceActivityModelError("UNIFAC activity coefficients must be finite and strictly positive")
    return gamma


def _species_gas_constants(mw_cond: np.ndarray) -> np.ndarray:
    mw = _require_positive_finite_array("mw_cond", mw_cond)
    return (ct.gas_constant / 1000.0) / mw


def _latent_heat_condensables_at_temperature(
    *,
    T: float,
    liquid_thermo: LiquidThermoModel,
    gas_thermo: GasThermoModel,
    liquid_cond_indices: np.ndarray,
    gas_cond_indices: np.ndarray,
    hvap_ref_cond: np.ndarray,
) -> np.ndarray:
    """Latent heat used in Eq.(2.19): L_i(T) = h_g,i(T) - h_l,i(T) + L_i^ref."""
    T_value = _validate_temperature(T)
    h_l = liquid_thermo.pure_enthalpy_vector(T_value)[liquid_cond_indices]
    h_g_T = gas_thermo.species_enthalpies_mass(T_value)[gas_cond_indices]
    h_g_ref = gas_thermo.species_enthalpies_mass(liquid_thermo.reference_T)[gas_cond_indices]
    latent = hvap_ref_cond + (h_g_T - h_g_ref) - h_l
    if not np.all(np.isfinite(latent)) or np.any(latent <= 0.0):
        raise InterfaceEquilibriumModelError(
            f"latent_cond(T={T_value:.6g}) must be finite and strictly positive"
        )
    return latent


def _eq219_temperature_integral_terms(
    *,
    Ts: float,
    Tb_cond: np.ndarray,
    Rg_cond: np.ndarray,
    hvap_ref_cond: np.ndarray,
    liquid_thermo: LiquidThermoModel,
    gas_thermo: GasThermoModel,
    liquid_cond_indices: np.ndarray,
    gas_cond_indices: np.ndarray,
    n_quad: int = 24,
) -> np.ndarray:
    """Evaluate the Eq.(2.19) integral ∫[Tb_i..Ts] L_i(T)/(Rg_i T^2) dT."""
    if not isinstance(n_quad, int) or n_quad < 2:
        raise InterfaceEquilibriumModelError("n_quad must be an integer >= 2")

    Ts_value = _validate_temperature(Ts)
    Tb = _require_positive_finite_array("Tb_cond", Tb_cond)
    Rg = _require_positive_finite_array("Rg_cond", Rg_cond)
    hvap = _require_positive_finite_array("hvap_ref_cond", hvap_ref_cond)
    if Tb.shape != Rg.shape or Tb.shape != hvap.shape:
        raise InterfaceEquilibriumModelError("Tb_cond, Rg_cond and hvap_ref_cond must be aligned")

    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    n_cond = Tb.shape[0]
    integrals = np.empty(n_cond, dtype=np.float64)

    for j in range(n_cond):
        a = float(Tb[j])
        b = Ts_value
        if a == b:
            integrals[j] = 0.0
            continue

        mid = 0.5 * (a + b)
        half = 0.5 * (b - a)
        T_nodes = mid + half * xi
        integrand = np.empty(n_quad, dtype=np.float64)
        for k, T_node in enumerate(T_nodes):
            latent_node = _latent_heat_condensables_at_temperature(
                T=float(T_node),
                liquid_thermo=liquid_thermo,
                gas_thermo=gas_thermo,
                liquid_cond_indices=liquid_cond_indices,
                gas_cond_indices=gas_cond_indices,
                hvap_ref_cond=hvap,
            )[j]
            integrand[k] = latent_node / (Rg[j] * T_node * T_node)
        integrals[j] = half * float(np.sum(wi * integrand))

    if not np.all(np.isfinite(integrals)):
        raise InterfaceEquilibriumModelError("Eq.(2.19) temperature integrals must be finite")
    return integrals


def _eq219_condensable_equilibrium_mass_fractions(
    *,
    Ts: float,
    P: float,
    Yl_cond: np.ndarray,
    Wl_if: float,
    gamma_cond: np.ndarray,
    Tb_cond: np.ndarray,
    mw_cond: np.ndarray,
    Rg_cond: np.ndarray,
    closure_mw: float,
    hvap_ref_cond: np.ndarray,
    liquid_thermo: LiquidThermoModel,
    gas_thermo: GasThermoModel,
    liquid_cond_indices: np.ndarray,
    gas_cond_indices: np.ndarray,
    reference_pressure: float,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Exact Eq.(2.19) closure written in gas-side mass fractions."""
    Ts_value = _validate_temperature(Ts)
    P_value = _validate_pressure(P)

    Yl_cond_arr = _as_1d_float_array("Yl_cond", Yl_cond)
    gamma = _require_positive_finite_array("gamma_cond", gamma_cond)
    Tb = _require_positive_finite_array("Tb_cond", Tb_cond)
    mw = _require_positive_finite_array("mw_cond", mw_cond)
    Rg = _require_positive_finite_array("Rg_cond", Rg_cond)
    hvap = _require_positive_finite_array("hvap_ref_cond", hvap_ref_cond)

    if (
        Yl_cond_arr.shape != gamma.shape
        or Yl_cond_arr.shape != Tb.shape
        or Yl_cond_arr.shape != mw.shape
        or Yl_cond_arr.shape != Rg.shape
        or Yl_cond_arr.shape != hvap.shape
    ):
        raise InterfaceEquilibriumModelError(
            "Yl_cond, gamma_cond, Tb_cond, mw_cond, Rg_cond and hvap_ref_cond must be aligned"
        )
    if not np.isfinite(Wl_if) or Wl_if <= 0.0:
        raise InterfaceEquilibriumModelError("Wl_if must be finite and strictly positive")
    if not np.isfinite(closure_mw) or closure_mw <= 0.0:
        raise InterfaceEquilibriumModelError("closure_mw must be finite and strictly positive")

    integral_terms = _eq219_temperature_integral_terms(
        Ts=Ts_value,
        Tb_cond=Tb,
        Rg_cond=Rg,
        hvap_ref_cond=hvap,
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        liquid_cond_indices=liquid_cond_indices,
        gas_cond_indices=gas_cond_indices,
    )

    exp_terms = np.exp(integral_terms)
    if not np.all(np.isfinite(exp_terms)) or np.any(exp_terms <= 0.0):
        raise InterfaceEquilibriumModelError("Eq.(2.19) exponential terms must be finite and strictly positive")

    psat_cond = reference_pressure * exp_terms
    if not np.all(np.isfinite(psat_cond)) or np.any(psat_cond <= 0.0):
        raise InterfaceEquilibriumModelError("psat_cond must be finite and strictly positive")

    eq_factor = (reference_pressure / P_value) * gamma * exp_terms
    A = Yl_cond_arr * Wl_if * eq_factor
    if not np.all(np.isfinite(A)) or np.any(A < 0.0):
        raise InterfaceEquilibriumModelError("Eq.(2.19) prefactor A_i must be finite and non-negative")

    inv_closure_mw = 1.0 / closure_mw
    Wg_eq = closure_mw * (1.0 - float(np.sum(A * (1.0 / mw - inv_closure_mw))))
    if not np.isfinite(Wg_eq) or Wg_eq <= 0.0:
        raise InterfaceEquilibriumModelError("Eq.(2.19) produced a non-physical gas mixture molecular weight")

    Yg_cond = A / Wg_eq
    if not np.all(np.isfinite(Yg_cond)) or np.any(Yg_cond < -atol):
        raise InterfaceEquilibriumModelError("Eq.(2.19) returned invalid condensable gas mass fractions")
    Yg_cond = np.maximum(Yg_cond, 0.0)
    return Yg_cond, psat_cond, integral_terms, Wg_eq


def _assemble_full_gas_mass_fractions(
    *,
    n_gas: int,
    gas_cond_indices: np.ndarray,
    Yg_cond: np.ndarray,
    gas_closure_index: int,
    atol: float = 1.0e-12,
) -> np.ndarray:
    Yg_full = np.zeros(n_gas, dtype=np.float64)
    Yg_full[gas_cond_indices] = Yg_cond
    closure_value = 1.0 - float(np.sum(Yg_cond))
    if closure_value < -atol:
        raise InterfaceEquilibriumModelError("Condensable equilibrium mass fractions exceed unity")
    if abs(closure_value) <= atol:
        closure_value = 0.0
    Yg_full[gas_closure_index] = closure_value
    if not np.isclose(float(np.sum(Yg_full)), 1.0, rtol=0.0, atol=atol):
        raise InterfaceEquilibriumModelError("Assembled gas equilibrium mass fractions must sum to 1")
    if np.any(Yg_full < -atol):
        raise InterfaceEquilibriumModelError("Assembled gas equilibrium mass fractions must be non-negative")
    return np.maximum(Yg_full, 0.0)


@dataclass(frozen=True)
class InterfaceEquilibriumModel:
    liquid_thermo: LiquidThermoModel
    gas_thermo: GasThermoModel
    liquid_species_names: tuple[str, ...]
    gas_species_names: tuple[str, ...]
    liquid_to_gas_species_map: dict[str, str]
    gas_closure_species: str
    liquid_cond_indices: np.ndarray
    gas_cond_indices: np.ndarray
    gas_closure_index: int
    reference_pressure: float

    def __post_init__(self) -> None:
        if self.liquid_species_names != self.liquid_thermo.species_names:
            raise InterfaceEquilibriumModelError("liquid_species_names must match liquid_thermo.species_names")
        if self.gas_species_names != self.gas_thermo.species_names:
            raise InterfaceEquilibriumModelError("gas_species_names must match gas_thermo.species_names")
        liquid_cond = np.asarray(self.liquid_cond_indices, dtype=np.int64)
        gas_cond = np.asarray(self.gas_cond_indices, dtype=np.int64)
        if liquid_cond.ndim != 1 or gas_cond.ndim != 1 or liquid_cond.shape != gas_cond.shape:
            raise InterfaceEquilibriumModelError("condensable index arrays must be one-dimensional and aligned")
        liquid_cond.setflags(write=False)
        gas_cond.setflags(write=False)
        object.__setattr__(self, "liquid_cond_indices", liquid_cond)
        object.__setattr__(self, "gas_cond_indices", gas_cond)
        object.__setattr__(self, "reference_pressure", _validate_pressure(self.reference_pressure))
        if not (0 <= self.gas_closure_index < len(self.gas_species_names)):
            raise InterfaceEquilibriumModelError("gas_closure_index must be a valid gas species index")
        if self.gas_species_names[self.gas_closure_index] != self.gas_closure_species:
            raise InterfaceEquilibriumModelError("gas_closure_index must point to gas_closure_species")


@dataclass(frozen=True)
class InterfaceEquilibriumResult:
    Ts: float
    P: float
    Xl_if: np.ndarray
    Yl_if_full: np.ndarray
    Wl_if: float
    gamma_cond: np.ndarray
    psat_cond: np.ndarray
    latent_cond: np.ndarray
    Xg_eq_full: np.ndarray
    Yg_eq_full: np.ndarray
    Wg_eq: float
    condensable_gas_indices: np.ndarray
    diagnostics: dict[str, Any]


def build_interface_equilibrium_model(
    *,
    liquid_thermo: LiquidThermoModel,
    gas_thermo: GasThermoModel,
    liquid_to_gas_species_map: Mapping[str, str],
    gas_closure_species: str,
    reference_pressure: float,
) -> InterfaceEquilibriumModel:
    mapping = dict(liquid_to_gas_species_map)
    _validate_species_mapping(
        liquid_species_names=liquid_thermo.species_names,
        gas_species_names=gas_thermo.species_names,
        liquid_to_gas_species_map=mapping,
        gas_closure_species=gas_closure_species,
    )
    liquid_cond_indices, gas_cond_indices = _build_condensable_index_maps(
        liquid_species_names=liquid_thermo.species_names,
        gas_species_names=gas_thermo.species_names,
        liquid_to_gas_species_map=mapping,
    )
    return InterfaceEquilibriumModel(
        liquid_thermo=liquid_thermo,
        gas_thermo=gas_thermo,
        liquid_species_names=liquid_thermo.species_names,
        gas_species_names=gas_thermo.species_names,
        liquid_to_gas_species_map=mapping,
        gas_closure_species=gas_closure_species,
        liquid_cond_indices=liquid_cond_indices,
        gas_cond_indices=gas_cond_indices,
        gas_closure_index=gas_thermo.species_names.index(gas_closure_species),
        reference_pressure=reference_pressure,
    )


def compute_interface_equilibrium(
    model: InterfaceEquilibriumModel,
    *,
    Ts: float,
    P: float,
    Yl_if_full: np.ndarray,
) -> InterfaceEquilibriumResult:
    Ts_value = _validate_temperature(Ts)
    P_value = _validate_pressure(P)
    try:
        liquid_range = model.liquid_thermo.valid_temperature_range()
    except LiquidThermoValidationError as exc:
        raise InterfaceEquilibriumValidationError(str(exc)) from exc
    if Ts_value < liquid_range[0] or Ts_value > liquid_range[1]:
        raise InterfaceEquilibriumValidationError(
            f"Ts={Ts_value:.6g} lies outside the liquid valid temperature range {liquid_range}"
        )

    Yl_full = _validate_mass_fractions(Yl_if_full, model.liquid_thermo.n_species)
    Xl_if = model.liquid_thermo.mole_fractions(Yl_full)
    Wl_if = model.liquid_thermo.mixture_molecular_weight(Yl_full)

    liquid_records = tuple(model.liquid_thermo.species_records[i] for i in model.liquid_cond_indices)
    activity_model = _activity_model_name_for_species(liquid_records)
    if activity_model.lower() == "ideal":
        gamma_cond = _compute_activity_coefficients_ideal(len(liquid_records))
    elif activity_model.upper() == "UNIFAC":
        if model.liquid_thermo.db.unifac is None:
            raise InterfaceActivityModelError("UNIFAC activity model requested but liquid database has no unifac data")
        gamma_cond = _compute_activity_coefficients_unifac(
            Ts=Ts_value,
            Xl_cond=Xl_if[model.liquid_cond_indices],
            liquid_records=liquid_records,
            unifac_global=model.liquid_thermo.db.unifac,
        )
    else:
        raise InterfaceActivityModelError(f"Unsupported activity_model: {activity_model}")

    Tb_cond = _require_positive_finite_array(
        "Tb_cond",
        np.array([record.boiling_temperature_atm for record in liquid_records], dtype=np.float64),
    )
    mw_cond = _require_positive_finite_array(
        "mw_cond",
        np.array([record.molecular_weight for record in liquid_records], dtype=np.float64),
    )
    hvap_ref_cond = _require_positive_finite_array(
        "hvap_ref_cond",
        np.array([record.hvap_ref for record in liquid_records], dtype=np.float64),
    )
    Rg_cond = _species_gas_constants(mw_cond)
    latent_cond = _latent_heat_condensables_at_temperature(
        T=Ts_value,
        liquid_thermo=model.liquid_thermo,
        gas_thermo=model.gas_thermo,
        liquid_cond_indices=model.liquid_cond_indices,
        gas_cond_indices=model.gas_cond_indices,
        hvap_ref_cond=hvap_ref_cond,
    )
    closure_mw = float(model.gas_thermo.molecular_weights[model.gas_closure_index])

    Yg_cond, psat_cond, eq219_integral_cond, Wg_eq_from_eq219 = _eq219_condensable_equilibrium_mass_fractions(
        Ts=Ts_value,
        P=P_value,
        Yl_cond=Yl_full[model.liquid_cond_indices],
        Wl_if=Wl_if,
        gamma_cond=gamma_cond,
        Tb_cond=Tb_cond,
        mw_cond=mw_cond,
        Rg_cond=Rg_cond,
        closure_mw=closure_mw,
        hvap_ref_cond=hvap_ref_cond,
        liquid_thermo=model.liquid_thermo,
        gas_thermo=model.gas_thermo,
        liquid_cond_indices=model.liquid_cond_indices,
        gas_cond_indices=model.gas_cond_indices,
        reference_pressure=model.reference_pressure,
    )

    Yg_eq_full = _assemble_full_gas_mass_fractions(
        n_gas=model.gas_thermo.n_species,
        gas_cond_indices=model.gas_cond_indices,
        Yg_cond=Yg_cond,
        gas_closure_index=model.gas_closure_index,
    )
    try:
        Xg_eq_full = mass_to_mole_fractions(Yg_eq_full, model.gas_thermo.molecular_weights)
    except MixRulesValidationError as exc:
        raise InterfaceEquilibriumModelError(str(exc)) from exc
    Wg_eq = model.gas_thermo.mixture_molecular_weight(Yg_eq_full)
    if not np.isclose(Wg_eq, Wg_eq_from_eq219, rtol=5.0e-5, atol=1.0e-9):
        raise InterfaceEquilibriumModelError(
            "Eq.(2.19) mass-fraction closure is inconsistent with the assembled gas mixture molecular weight"
        )

    diagnostics = {
        "activity_model": activity_model,
        "n_condensables": int(len(model.gas_cond_indices)),
        "liquid_cond_species": tuple(record.name for record in liquid_records),
        "gas_cond_species": tuple(model.gas_species_names[idx] for idx in model.gas_cond_indices),
        "sum_Xg_full": float(np.sum(Xg_eq_full)),
        "sum_Yg_cond": float(np.sum(Yg_cond)),
        "sum_Yg_full": float(np.sum(Yg_eq_full)),
        "closure_species_name": model.gas_closure_species,
        "closure_species_X": float(Xg_eq_full[model.gas_closure_index]),
        "closure_species_Y": float(Yg_eq_full[model.gas_closure_index]),
        "Wl_if": float(Wl_if),
        "Wg_eq": float(Wg_eq),
        "eq219_integral_cond": eq219_integral_cond.copy(),
    }

    return InterfaceEquilibriumResult(
        Ts=Ts_value,
        P=P_value,
        Xl_if=Xl_if.copy(),
        Yl_if_full=Yl_full.copy(),
        Wl_if=float(Wl_if),
        gamma_cond=gamma_cond.copy(),
        psat_cond=psat_cond.copy(),
        latent_cond=latent_cond.copy(),
        Xg_eq_full=Xg_eq_full.copy(),
        Yg_eq_full=Yg_eq_full.copy(),
        Wg_eq=float(Wg_eq),
        condensable_gas_indices=model.gas_cond_indices.copy(),
        diagnostics=diagnostics,
    )


__all__ = [
    "InterfaceActivityModelError",
    "InterfaceEquilibriumError",
    "InterfaceEquilibriumModel",
    "InterfaceEquilibriumModelError",
    "InterfaceEquilibriumResult",
    "InterfaceEquilibriumValidationError",
    "build_interface_equilibrium_model",
    "compute_interface_equilibrium",
]
