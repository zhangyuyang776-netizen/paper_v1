# `src/properties/` 代码审查报告

**审查日期**：2026-03-24
**审查范围**：`src/properties/` 目录下全部 6 个模块
**审查依据**：指导文件（物理与数值方法规范）、Fredenslund 1975 UNIFAC 标准公式

---

## 总体结论

`mix_rules.py` 和 `gas.py` 与指导文件完全一致，无偏差。
`equilibrium.py`、`liquid.py`、`liquid_db.py`、`aggregator.py` 存在以下问题，其中 1 处严重、4 处中等、4 处轻微。

---

## 模块逐一审查

### 1. `equilibrium.py`

#### EQ-1【严重】UNIFAC 残差项分母索引方向错误

**位置**：`ln_capital_gamma` 函数，L202

**问题代码**：
```python
denom = psi @ theta_mix   # denom[m] = Σ_n ψ_mn θ_n  ← 行求和（错误）
second_term = np.sum(theta_mix * psi[k, :] / denom)
```

**标准 UNIFAC 残差贡献（Fredenslund 1975）**：
```
ln Γ_k = Q_k * [1 - ln(Σ_m θ_m ψ_mk) - Σ_m (θ_m ψ_km / Σ_n θ_n ψ_nm)]
```

第二项分母对每个 m 需要的是 `Σ_n θ_n ψ_nm`（列求和）。

设 `psi[m, n] = ψ_mn`，则：
- **需要**：`(theta_mix @ psi)[m]` = `Σ_n θ_n ψ_nm` = 列求和 = 变量 `sum_theta_psi_col[m]`（L197 已计算）
- **代码用的**：`(psi @ theta_mix)[m]` = `Σ_n ψ_mn θ_n` = 行求和（错误）

UNIFAC 相互作用参数矩阵通常非对称（`a_mn ≠ a_nm`，`ψ_mn ≠ ψ_nm`），行求和 ≠ 列求和，导致活度系数计算在多组分混合物中全部错误。

**修复**：将 `denom = psi @ theta_mix` 替换为 `denom = sum_theta_psi_col`（L197 已计算的变量）。

---

#### EQ-2【中等】UNIFAC 组合项：`l_species` 是死代码，简化形式需与论文核对

**位置**：L176 及 `ln_gamma_c` 计算块

```python
l_species = (z / 2.0) * (r_species - q_species) - (r_species - 1.0)
# 此后 l_species 从未被引用
```

`ln_gamma_c` 实现：
```python
ln_gamma_c = (
    np.log(V) + 1.0 - V
    - 5.0 * q_species * (np.log(V / F) + 1.0 - V / F)
)
```

**标准 Stavermann-Guggenheim 组合项**：
```
ln γ_i^C = ln(Φ_i/x_i) + (z/2)*q_i*ln(θ_i/Φ_i) + l_i - (Φ_i/x_i)*Σ_j(x_j l_j)
```

代码缺少 `l_i` 和 `Σ x_j l_j` 修正项，使用了简化形式（属于 modified UNIFAC 或 Bondi 近似）。**需对照论文附录 2.A 确认是否允许此简化**，若不允许则补全 `l_i` 修正项。

---

#### EQ-3【轻微】汽化潜热正值断言在近临界温度下误报

**位置**：L247

```python
if not np.all(np.isfinite(latent)) or np.any(latent <= 0.0):
    raise InterfaceEquilibriumModelError(...)
```

物理上当 `T → Tc` 时 `L_i(T) → 0`，在近临界温度的合法区间内潜热趋近零，会被误判为错误抛出异常。建议改为警告或放宽至 `latent < -ε`。

---

### 2. `liquid.py`

#### LQ-1【中等】T_ref 未强制校验为 298.15 K

**位置**：`build_liquid_thermo_model`，L558-559

现有校验仅保证所有 species 选出的 bank 的 T_ref 相同，但**未校验 T_ref == 298.15 K**。

指导文件 §2.2 明确规定：
> "固定参考温度 T_ref = 298.15 K；液相焓、气相焓和汽化潜热的定义都必须使用同一参考温度。"

若数据库中 T_ref 误填（如 300 K），`_latent_heat_condensables_at_temperature` 中的 `h_g_T - h_g_ref` 将基于错误参考点，导致潜热系统性偏移。

**修复**：在一致性校验后增加 `assert np.isclose(T_ref, 298.15, atol=0.01)`。

---

#### LQ-2【中等】`valid_temperature_range` 仅检查 cp 区间，忽略 ρ/k/μ 拟合范围

**位置**：L318-329

```python
ranges = [self.species_records[...].cp_T_range for name in subset]
t_min = max(item[0] for item in ranges)
t_max = min(item[1] for item in ranges)
```

该方法控制所有属性评估的温度上下限（`_validated_temperature` 调用此值），但仅基于 Shomate cp 的有效区间。密度、热导率、黏度的 `merino_log_poly` 拟合各有独立的有效区间，当 T 在 cp 有效范围内但超出 ρ/k/μ 有效范围时，代码不报错，静默给出外推值。

**修复**：对所有属性类型（`cp`、`rho`、`conductivity`、`viscosity`）取各自 T 范围的交集作为 `valid_temperature_range`。

---

#### LQ-3【轻微】Shomate 焓双重减法在 T_ref 异常时引入歧义

**位置**：L349

```python
h_mass = (_h_shomate_molar(T, cp_coeffs) - _h_shomate_molar(rec.T_ref, cp_coeffs)) / rec.molecular_weight
```

`_h_shomate_molar` 直接实现 NIST 标准式 `H°(T) − H°(298.15)`，当 T_ref = 298.15 K 时 `_h_shomate_molar(T_ref) ≡ 0`，双重减法冗余。若 LQ-1 问题存在导致 T_ref ≠ 298.15 K，此处引入双重偏移，逻辑歧义。当前逻辑正确，但与 LQ-1 形成潜在连锁风险。

---

### 3. `liquid_db.py`

#### DB-1【中等】压力 bank 模式下 T_ref 未与 meta.reference_T 交叉校验

flat 模式有严格校验（L562-566）：
```python
if not np.isclose(T_ref, meta.reference_T, ...):
    raise LiquidDatabaseValidationError(...)
```

但在 `_parse_pressure_bank`（L242-341）中，各 bank 的 T_ref 仅读取存储，**不校验是否等于 meta.reference_T**。若 pressure bank 数据中 T_ref 填错，加载时静默通过，只有运行时才暴露，形成难以追踪的计算偏差。

**修复**：在 `_parse_pressure_bank` 内对每个 bank 的 T_ref 增加与 `meta.reference_T` 的一致性校验。

---

#### DB-2【轻微】`boiling_temperature` 与 `boiling_temperature_atm` 字段命名歧义

**位置**：`_parse_species_record` 输出，L643-644

```python
boiling_temperature=boiling_temperature_atm,      # 与下方完全相同
boiling_temperature_atm=boiling_temperature_atm,
```

两字段始终赋相同值，但 `boiling_temperature` 易被误解为"当前 pressure bank 下的压力相关沸点"。指导文件 §5.4 规定界面方程 Eq.(2.19) 必须用大气压沸点 `boiling_temperature_atm`。现有 `equilibrium.py` 访问的是正确字段，但命名歧义增加维护风险。

---

### 4. `mix_rules.py` — 完全一致 ✓

所有混合规则与指导文件精确吻合：

| 规则 | 指导文件公式 | 代码实现 | 状态 |
|------|------------|---------|------|
| 混合密度 | `(Σ X_i ρ_i^½)²` | `np.square(np.sum(X * np.sqrt(rho)))` | ✓ |
| 混合 cp | `Σ Y_i c_p,i` | `mass_weighted_sum(Y, cp)` | ✓ |
| 混合焓 | `Σ Y_i h_i` | `mass_weighted_sum(Y, h)` | ✓ |
| Filippov 热导率 | `Σ_i Y_i(k_i − Σ_{j>i} 0.72·Y_j·\|k_i−k_j\|)` | `_filippov_conductivity(Y, k, kij=0.72)` | ✓ |
| Grunberg-Nissan 黏度 | `exp(Σ X_i ln μ_i)` | `log_mole_weighted_average(X, mu)` | ✓ |
| Wilke-Chang 扩散 | `1.173e-16·√(Σ_{j≠i} X_j φ_j W_j)·T / (μ·(V_i/1000)^0.6)` | 完全吻合，含 `mw_arr * 1000.0` 显式单位转换 | ✓ |

---

### 5. `gas.py` — 完全一致 ✓

`species_enthalpies_mass(T)` 使用 Cantera `standard_enthalpies_RT`，返回含生成焓的绝对纯组分焓（J/kg）。进入潜热公式时由 `equilibrium.py` 的差分 `h_g_T − h_g_ref` 转换为显热增量，符合指导文件 §6.3 要求。混合气相焓 `enthalpy_mass(T, Y, P)` 使用 Cantera 直接计算，自洽正确。

---

### 6. `aggregator.py`

#### AG-1【轻微】气相压力默认值使用 reference_pressure 而非 P_inf

**位置**：L125

```python
gas_pressure_arr = _normalize_gas_pressure(gas_pressure, ..., gas_thermo.reference_pressure)
```

`gas_pressure=None` 时默认 101325 Pa（1 atm）。若模拟的环境压力 P_inf ≠ 1 atm（高压喷雾工况），调用方必须显式传入，否则气相密度、cp、扩散系数计算全部基于错误压力。指导文件规定全域恒压 P_inf，该值应从配置层传入。

---

## 偏差汇总

| 编号 | 文件 | 位置 | 严重性 | 描述 |
|------|------|------|--------|------|
| EQ-1 | `equilibrium.py` | L202 | **严重** | UNIFAC 残差项分母用行求和替代列求和，非对称相互作用矩阵下活度系数全部错误 |
| EQ-2 | `equilibrium.py` | L176 | 中等 | `l_species` 死代码，组合项简化形式需与论文附录 2.A 核对 |
| LQ-1 | `liquid.py` | L558 | 中等 | T_ref 未强制 = 298.15 K，是潜热基准正确性的前提 |
| LQ-2 | `liquid.py` | L318 | 中等 | 有效温度范围仅检查 cp，忽略 ρ/k/μ 拟合区间，静默外推 |
| DB-1 | `liquid_db.py` | `_parse_pressure_bank` | 中等 | 压力 bank 的 T_ref 不校验等于 meta.reference_T |
| EQ-3 | `equilibrium.py` | L247 | 轻微 | 潜热正值断言在近临界区间对合法值误报 |
| LQ-3 | `liquid.py` | L349 | 轻微 | Shomate 焓双重减法在 T_ref 异常时引入连锁歧义 |
| DB-2 | `liquid_db.py` | L643 | 轻微 | `boiling_temperature` 与 `boiling_temperature_atm` 命名歧义 |
| AG-1 | `aggregator.py` | L125 | 轻微 | 气相压力默认 reference_pressure，高压工况须显式传入 P_inf |

---

## 修复优先级

1. **EQ-1**（最高）— 直接影响 Eq.(2.19) 界面平衡组成，是液滴蒸发模型最核心的计算环节
2. **DB-1 → LQ-1**（连锁）— 保证 T_ref = 298.15 K 贯通数据加载→液相焓→潜热全链
3. **EQ-2** — 对照论文附录 2.A 确认组合项形式；若不符则补全 `l_i` 修正项
4. **LQ-2** — 增加 ρ/k/μ 区间检查，防止温度外推时给出无物理意义的属性值
5. 其余轻微项可在后续迭代中修复
