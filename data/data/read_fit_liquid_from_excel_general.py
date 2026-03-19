from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

    #python read_fit_liquid_from_excel_general.py ethanol_1atm.xlsx --species ethanol

REQUIRED_SHEETS = [
    "species_meta",
    "cp_data",
    "rho_data",
    "k_data",
    "mu_data",
]

REQUIRED_META_COLS = [
    "name",
    "molecular_weight_kg_per_mol",
    "boiling_temperature_K",
    "T_ref_K",
    "hvap_ref_J_per_kg",
    "Tc_K",
    "hvap_model",
    "hvap_watson_exponent",
    "activity_model",
    "cp_model",
    "rho_model",
    "k_model",
    "mu_model",
    "liquid_mixture_density_model",
    "liquid_mixture_conductivity_model",
    "liquid_mixture_viscosity_model",
    "liquid_diffusion_model",
]

DATA_SPECS = {
    "cp_data": ("cp_mass_J_per_kgK", 8),
    "rho_data": ("rho_kg_per_m3", 8),
    "k_data": ("k_W_per_mK", 6),
    "mu_data": ("mu_Pa_s", 8),
}


def _norm_scalar(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "":
            return None
        if s.lower() in {"null", "none", "nan"}:
            return None
        return s
    return v


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() if c is not None else c for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(_norm_scalar)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


OPTIONAL_SHEETS = ["aliases", "unifac_groups"]


def load_book(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    missing = [s for s in REQUIRED_SHEETS if s not in xl.sheet_names]
    if missing:
        raise ValueError(f"Missing required sheets: {missing}")
    book = {sheet: _clean_df(xl.parse(sheet)) for sheet in xl.sheet_names}
    for sheet in OPTIONAL_SHEETS:
        if sheet not in book:
            book[sheet] = pd.DataFrame()
    return book


def _require_columns(df: pd.DataFrame, cols: list[str], sheet: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet {sheet!r} missing required columns: {missing}")


def _check_positive(name: str, value):
    if value is None or not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return float(value)


def require_species_row(meta_df: pd.DataFrame, species: str) -> pd.Series:
    rows = meta_df[meta_df["name"] == species]
    if len(rows) != 1:
        raise ValueError(
            f"species_meta must contain exactly one cleaned row for species {species!r}; got {len(rows)}"
        )
    return rows.iloc[0]


def aliases_for(df: pd.DataFrame, species: str) -> list[str]:
    if df.empty or "name" not in df.columns or "alias" not in df.columns:
        return []
    rows = df[df["name"] == species]
    if len(rows) == 0:
        return []
    vals = []
    for v in rows["alias"].tolist():
        if v is None:
            continue
        vals.append(str(v))
    return vals


def unifac_groups_for(df: pd.DataFrame, species: str) -> dict[str, int]:
    if df.empty or "name" not in df.columns or "group" not in df.columns or "count" not in df.columns:
        return {}
    rows = df[df["name"] == species]
    groups = {}
    for _, r in rows.iterrows():
        g = r.get("group")
        c = r.get("count")
        if g is None or c is None:
            continue
        groups[str(g)] = int(float(c))
    return groups


def require_data(df: pd.DataFrame, species: str, ycol: str, min_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = df[df["name"] == species].copy()
    rows = rows[["temperature_K", "pressure_Pa", ycol]].dropna()
    if len(rows) < min_points:
        raise ValueError(f"{species!r} needs at least {min_points} points in {ycol}, got {len(rows)}")
    rows["temperature_K"] = rows["temperature_K"].astype(float)
    rows["pressure_Pa"] = rows["pressure_Pa"].astype(float)
    rows[ycol] = rows[ycol].astype(float)
    rows = rows.sort_values("temperature_K")
    return (
        rows["temperature_K"].to_numpy(),
        rows["pressure_Pa"].to_numpy(),
        rows[ycol].to_numpy(),
    )


def infer_single_pressure(pressures: np.ndarray, rel_tol: float = 0.02) -> float:
    p = np.asarray(pressures, dtype=float)
    p_med = float(np.median(p))
    max_rel = float(np.max(np.abs(p - p_med) / p_med))
    if max_rel > rel_tol:
        raise ValueError(
            f"Pressure spread too large for a single-pressure bank: median={p_med:.6g} Pa, max relative spread={max_rel:.4%}"
        )
    return p_med


def fit_shomate_cp(T: np.ndarray, cp_mass: np.ndarray, molecular_weight_kg_per_mol: float) -> dict[str, float]:
    """
    拟合 Shomate A-E 系数。

    当前项目 liquid.py 实际只依赖 A-E 来计算 cp(T)，
    F/G/H 这里先置 0，仅用于 YAML 字段完整性。
    """
    T = np.asarray(T, dtype=float)
    cp_mass = np.asarray(cp_mass, dtype=float)
    mw = float(molecular_weight_kg_per_mol)

    # 先转为摩尔基比热，因为当前 liquid.py 使用的是摩尔 Shomate 形式
    cp_molar = cp_mass * mw
    t = T / 1000.0

    X = np.column_stack([
        np.ones_like(t),
        t,
        t**2,
        t**3,
        1.0 / (t**2),
    ])
    coeffs, *_ = np.linalg.lstsq(X, cp_molar, rcond=None)
    A, B, C, D, E = coeffs

    return {
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "E": float(E),
        "F": 0.0,
        "G": 0.0,
        "H": 0.0,
    }


def fit_merino_log_poly(T: np.ndarray, prop: np.ndarray) -> dict[str, float]:
    """
    拟合：
        ln(prop) = A ln(T) + B/T + C/T^2 + D + E T + F T^2
    """
    T = np.asarray(T, dtype=float)
    prop = np.asarray(prop, dtype=float)

    if np.any(prop <= 0.0):
        raise ValueError("merino_log_poly fit requires all property values > 0")

    y = np.log(prop)
    X = np.column_stack([
        np.log(T),
        1.0 / T,
        1.0 / (T**2),
        np.ones_like(T),
        T,
        T**2,
    ])
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B, C, D, E, F = coeffs

    return {
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "E": float(E),
        "F": float(F),
    }


def build_bank(book: dict[str, pd.DataFrame], species: str, pressure_rel_tol: float) -> dict:
    meta_df = book["species_meta"]
    _require_columns(meta_df, REQUIRED_META_COLS, "species_meta")

    meta = require_species_row(meta_df, species)

    mw = _check_positive("molecular_weight_kg_per_mol", meta["molecular_weight_kg_per_mol"])
    boiling_temperature = _check_positive("boiling_temperature_K", meta["boiling_temperature_K"])
    T_ref = _check_positive("T_ref_K", meta["T_ref_K"])
    hvap_ref = _check_positive("hvap_ref_J_per_kg", meta["hvap_ref_J_per_kg"])

    Tc = None if meta["Tc_K"] is None else _check_positive("Tc_K", meta["Tc_K"])

    activity_model = str(meta["activity_model"]) if meta["activity_model"] is not None else "ideal"
    cp_model = str(meta["cp_model"])
    rho_model = str(meta["rho_model"])
    k_model = str(meta["k_model"])
    mu_model = str(meta["mu_model"])

    hvap_model = None if meta["hvap_model"] is None else str(meta["hvap_model"])
    hvap_watson_exponent = None if meta["hvap_watson_exponent"] is None else float(meta["hvap_watson_exponent"])

    T_cp, P_cp, cp = require_data(book["cp_data"], species, "cp_mass_J_per_kgK", DATA_SPECS["cp_data"][1])
    T_rho, P_rho, rho = require_data(book["rho_data"], species, "rho_kg_per_m3", DATA_SPECS["rho_data"][1])
    T_k, P_k, k = require_data(book["k_data"], species, "k_W_per_mK", DATA_SPECS["k_data"][1])
    T_mu, P_mu, mu = require_data(book["mu_data"], species, "mu_Pa_s", DATA_SPECS["mu_data"][1])

    p_fit = infer_single_pressure(np.concatenate([P_cp, P_rho, P_k, P_mu]), rel_tol=pressure_rel_tol)

    cp_coeffs = fit_shomate_cp(T_cp, cp, mw)
    rho_coeffs = fit_merino_log_poly(T_rho, rho)
    k_coeffs = fit_merino_log_poly(T_k, k)
    mu_coeffs = fit_merino_log_poly(T_mu, mu)

    return {
        "species_name": species,
        "bank": {
            "p_fit": p_fit,
            "boiling_temperature": boiling_temperature,
            "T_ref": T_ref,
            "cp_model": cp_model,
            "cp_T_range": [float(np.min(T_cp)), float(np.max(T_cp))],
            "cp_coeffs": cp_coeffs,
            "hvap_ref": hvap_ref,
            "hvap_model": hvap_model,
            "hvap_coeffs": {} if hvap_model is None else {"exponent": hvap_watson_exponent},
            "rho_model": rho_model,
            "rho_coeffs": rho_coeffs,
            "k_model": k_model,
            "k_coeffs": k_coeffs,
            "mu_model": mu_model,
            "mu_coeffs": mu_coeffs,
        },
    }


def _yaml_number(value: float) -> str:
    value_f = float(value)
    if np.isclose(value_f, round(value_f), rtol=0.0, atol=1.0e-12):
        return str(int(round(value_f)))
    return np.format_float_positional(value_f, trim="-")


def dump_bank_yaml(obj: dict) -> str:
    b = obj["bank"]
    lines = []
    indent = "  "

    lines.append("pressure_banks:")
    lines.append(f"{indent}- p_fit: {_yaml_number(b['p_fit'])}")
    lines.append(f"{indent*2}boiling_temperature: {_yaml_number(b['boiling_temperature'])}")
    lines.append(f"{indent*2}T_ref: {_yaml_number(b['T_ref'])}")
    lines.append(f"{indent*2}cp_model: {b['cp_model']}")
    lines.append(
        f"{indent*2}cp_T_range: [{_yaml_number(b['cp_T_range'][0])}, {_yaml_number(b['cp_T_range'][1])}]"
    )
    lines.append(f"{indent*2}cp_coeffs:")
    for key in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        lines.append(f"{indent*3}{key}: {_yaml_number(b['cp_coeffs'][key])}")

    lines.append(f"{indent*2}hvap_ref: {_yaml_number(b['hvap_ref'])}")
    if b["hvap_model"] is None:
        lines.append(f"{indent*2}hvap_model: null")
        lines.append(f"{indent*2}hvap_coeffs: {{}}")
    else:
        lines.append(f"{indent*2}hvap_model: {b['hvap_model']}")
        lines.append(f"{indent*2}hvap_coeffs:")
        lines.append(f"{indent*3}exponent: {_yaml_number(b['hvap_coeffs']['exponent'])}")

    for model_key, coeff_key in [
        ("rho_model", "rho_coeffs"),
        ("k_model", "k_coeffs"),
        ("mu_model", "mu_coeffs"),
    ]:
        lines.append(f"{indent*2}{model_key}: {b[model_key]}")
        lines.append(f"{indent*2}{coeff_key}:")
        for key in ["A", "B", "C", "D", "E", "F"]:
            lines.append(f"{indent*3}{key}: {_yaml_number(b[coeff_key][key])}")

    return "\n".join(lines)


def dump_full_species_yaml(obj: dict) -> str:
    raise NotImplementedError("Only pressure_banks block output is supported in the current version")


def validate_workbook(book: dict[str, pd.DataFrame], species: str) -> list[str]:
    issues = []

    meta_df = book["species_meta"]
    _require_columns(meta_df, REQUIRED_META_COLS, "species_meta")

    rows = meta_df[meta_df["name"] == species]
    if len(rows) != 1:
        issues.append(f"species_meta should have exactly 1 cleaned row for {species!r}; got {len(rows)}")
        return issues

    row = rows.iloc[0]

    # 软性语义检查，不阻塞运行，但会提醒你别拿错误单位自欺欺人
    mw = row["molecular_weight_kg_per_mol"]
    if mw is not None:
        mwf = float(mw)
        if mwf > 1.0:
            issues.append(
                f"molecular_weight_kg_per_mol={mwf} looks like g/mol, not kg/mol. "
                f"For ethanol expected about 0.04607, not 46.07."
            )

    for sname, (ycol, min_points) in DATA_SPECS.items():
        df = book[sname]
        if species not in set(df["name"].dropna().tolist()):
            issues.append(f"{sname} has no rows for cleaned species name {species!r}")

    if row["activity_model"] is not None and str(row["activity_model"]).strip() != row["activity_model"]:
        issues.append("activity_model contains leading/trailing whitespace")

    return issues


def main():
    ap = argparse.ArgumentParser(
        description="Read a liquid-property Excel workbook, validate it, fit one single-pressure bank, and print YAML."
    )
    ap.add_argument("xlsx", type=Path)
    ap.add_argument("--species", required=True, help="Species name after trimming, e.g. ethanol")
    ap.add_argument(
        "--pressure-rel-tol",
        type=float,
        default=0.02,
        help="Maximum allowed relative pressure spread inside a single bank",
    )
    args = ap.parse_args()

    book = load_book(args.xlsx)
    issues = validate_workbook(book, args.species)
    if issues:
        print("FORMAT_ISSUES_FOUND:")
        for issue in issues:
            print(f"- {issue}")
        print("\nPlease fix the workbook before trusting fitted coefficients.", file=sys.stderr)

    result = build_bank(book, args.species, pressure_rel_tol=args.pressure_rel_tol)

    print(dump_bank_yaml(result))


if __name__ == "__main__":
    main()
