# `src/physics/` 代码审查报告

**审查日期**：2026-03-24
**审查范围**：`src/physics/` 目录下全部 10 个模块
**审查依据**：`governing_equations_discretization_and_bc_guideline_final_v2.md`、`paper_v1 Initialization and First-Step Guideline.md`、`velocity_recovery_guideline_final.md`

---

## 总体结论

`physics/` 层整体与指导文件高度一致，物理公式、符号约定、离散方案均无严重偏差。
发现 **1 处中等**（界面能量方程形式与 Eq.(2.17) 的等价性依赖液气焓参考态一致性，需物性层保证）、**1 处轻微**（液相扩散包验证不对称）。其余 8 个模块完全正确。

---

## 1. `interface_face.py` — 界面真相源

### 结论：完全一致 ✓

#### 验证项

| 项目 | 指导文件规定 | 代码实现 | 结论 |
|------|-------------|----------|------|
| 液相温度梯度 | `(T_s - T_{l,N_l}) / Δr_{l,s}` §9.4 | `(Ts - Tl_last) / dr_l_s` L242 | ✓ |
| 气相温度梯度 | `(T_{g,1} - T_s) / Δr_{g,s}` §9.4 | `(Tg_first - Ts) / dr_g_s` L243 | ✓ |
| 液相组分梯度 | `(Y_{s,l,i} - Y_{l,N_l,i}) / Δr_{l,s}` §9.4 | `(Ys_l_full - Yl_last_full) / dr_l_s` L244 | ✓ |
| 气相摩尔分数梯度 | `(X_{g,1,i} - X_{s,g,i}) / Δr_{g,s}` §9.4 | `(Xg_first_full - Xs_g_full) / dr_g_s` L245 | ✓ |
| 气相扩散速度 | `Vd0 = -(D/X)∇X`，混合平均 §3.3.3 | `Vd0 = -(D/x_safe)*dXdr` L276 | ✓ |
| 修正速度 | `Vcd = -Σ Y_i Vd0_i` | `Vcd_g = -sum(Ys_g * Vd0)` L277 | ✓ |
| 气相扩散通量 | `J_g = -ρ Y (Vd0 + Vcd)` | `J_g = -rho_s_g * Ys * (Vd0+Vcd)` L278 | ✓ |
| 液相扩散通量 | `J_l = -ρ_l D_l ∇Y_l`，Fick 形式 §3.2.4 | `J[i] = -rho_s_l * D * dYdr` L262 | ✓ |
| `sum_i J_l = 0` | 闭合条件 | 闭合项 `J_closure = -Σ J_nonclosure` L263 | ✓ |
| `sum_i J_g = 0` | 修正速度保证 | 后验校验 L436–439 | ✓ |
| 气相界面绝对质量流率 | `G_g = (ρ_g ȧ - ṁ'') A`，由 Eq.(2.18) 导出 | `(rho_s_g * dot_a_frozen - mpp) * area_s` L342 | ✓ |
| 总能量通量 | `E = -ṁ'' h_s + q` | `E_l_s = -mpp*h_s_l + q_l_s` L332–333 | ✓ |

---

## 2. `interface_mass.py` — 界面质量残差

### 结论：完全一致 ✓

- **Eq.(2.15)**（可凝组分跳跃）：`eq215_values = N_g[gas_idx] - N_l[liq_idx] = 0`，即 `N_{g,i} = N_{l,i}`。✓
- **Eq.(2.16)**（非凝组分无穿越）：`eq216_values = N_g[noncond] = 0`。✓
- **Eq.(2.18) 单 mpp 策略**：液相侧 `mpp + ρ_l(u_l - ȧ) = 0` 为主残差（L197），气相侧仅为诊断项（L207–213），符合指导文件 §3.4.4 规定的"液侧主残差 + 气侧强施加"。✓
- **Eq.(2.19)**（可凝蒸汽相平衡）：`eq219_values = Ys_g[cond] - Yeq_g[cond] = 0`。✓
- **`eps_mass_gas_kinematic = Σ N_g + mpp`**：正确的气相质量运动学自洽度诊断量，收敛时应为零。✓

---

## 3. `interface_energy.py` — 界面能量残差

### IE-1【中等】能量残差形式与 Eq.(2.17) 的等价性依赖液气焓参考态

**位置**：`_build_energy_residual`，L66–67

```python
energy_residual = float(iface_pkg.E_g_s - iface_pkg.E_l_s)
# = (-mpp * h_s_g + q_g_s) - (-mpp * h_s_l + q_l_s) = 0
```

**指导文件 Eq.(2.17)**（§3.4.3）：
```
-ṁ'' Σ_i Y_{s,l,i} L_i(T_s) = (k_g ∂T_g/∂r - k_l ∂T_l/∂r)|_if - Σ_i J_{l,i}|_if L_i(T_s)
```

**等价性分析**：代码使用总焓通量连续性形式 `E_g = E_l`，与 Eq.(2.17) 数学等价，条件是：

```
h_{s,g} - h_{s,l} = Σ_i Y_{s,i} L_i(T_s) = 蒸发焓
```

即气相物种焓 `gas_props.species_enthalpies_mass(Ts)` 与液相纯物种焓 `liquid_props.pure_enthalpy_vector(Ts)` 需使用同一热力学参考态（如元素在标准状态下焓为零）。

**物理层本身正确**：公式结构无误，等价性已成立。**风险在物性层**：若液相 `pure_enthalpy_vector` 使用了不同参考态（如液相标准焓 = 0），则界面能量残差将包含一个常数偏置，相当于抹去了蒸发潜热，导致界面温度求解错误。

**建议**：在 `properties/` 层的合约文档中明确规定气液物种焓必须使用一致的绝对参考态；若已在 `properties_code_review.md` 中处理，可在此引用交叉确认。

---

## 4. `energy_flux.py` — 热通量组合器

### 结论：完全一致 ✓

实现 `q_total = -k ∇T + Σ_i J_i h_i`（L148–151），与指导文件 §5.3.3–§5.3.4 完全一致。物种扩散焓项永不省略（模块级约束注释已明确禁止）。✓

---

## 5. `flux_liq.py` — 液相内部面通量

### 结论：主体正确，存在 1 处轻微偏差

#### FL-1 ✓ 液相 Fick 扩散形式正确

**位置**：`build_liquid_internal_diffusion_package`，L276

```python
J_diff_full = -rho_face[:, None] * D_face_full * grad_Y_full
```

即 `J_{l,i,f} = -ρ_{l,f} D_{l,i,f} ∂Y_{l,i}/∂r|_f`，与指导文件 §3.2.4 和 §17（"液相扩散：Fick"）一致。✓

闭合条件（L277–280）：
```python
J_diff_full[:, closure_index] = -np.sum(J_diff_full[:, nonclosure_mask], axis=1)
```
由构造保证 `Σ_i J_i = 0`。✓

#### FL-2【轻微】`_validate_diffusion_package` 未后验校验 `Σ_i J_i = 0`

**位置**：`_validate_diffusion_package`，L207–247

液相扩散包的验证函数只检查 `rho_face > 0`、`D_face_full >= 0`，**未验证 `Σ J_i = 0`**。

对比气相 `_validate_internal_diffusion_package`（`flux_gas.py:282–283`）：
```python
if not np.allclose(np.sum(J_diff_full, axis=1), 0.0, rtol=0.0, atol=1.0e-10):
    raise GasFluxError("gas diffusive flux package must satisfy sum_i J_i = 0 on every internal face")
```

液相闭合逻辑若有 bug（如 `closure_index` 被跳过或 `nonclosure_mask` 逻辑错误），`_validate_diffusion_package` 不会报错，而气相会立刻报错。

**建议**：在 `_validate_diffusion_package` 中补充：
```python
if n_face > 0 and n_spec > 1:
    if not np.allclose(np.sum(J_diff_full, axis=1), 0.0, rtol=0.0, atol=1.0e-10):
        raise LiquidFluxError("liquid diffusive flux must satisfy sum_i J_i = 0")
```

---

## 6. `flux_gas.py` — 气相内部面及远场通量

### 结论：完全一致 ✓

#### 验证项

| 项目 | 指导文件规定 | 代码实现 | 结论 |
|------|-------------|----------|------|
| 内部面扩散梯度 | `∇X`（摩尔分数），mixture-averaged §5.3.2 | `dXdr = (X_right-X_left)/(r_right-r_left)` L304 | ✓ |
| 内部面扩散通量守恒 | `Σ J_i = 0` | 后验校验 L282–283 | ✓ |
| 远场梯度 | `(φ_bc - φ_N)/(r_N+ - r_N)`，one-sided §5.x | `(X_inf - X_cell)/dr`, `(T_inf - T_cell)/dr` L401–402 | ✓ |
| 远场导热 | `q = -k_{N+}(T∞-T_N)/dr` §5.x | `q_cond = -k_face * grad_T` L430 | ✓ |
| 远场对流上风 | `u_rel > 0` 取内部，`< 0` 取 BC §11.3 | `upwind_from_cell = u_rel > 0.0` L396 | ✓ |

---

## 7. `flux_convective.py` — 对流通量

### 结论：完全一致 ✓

- **相对速度上风**：`u_rel = u_abs - vc`（L95），严格使用相对控制面速度，符合指导文件 §6 要求。✓
- **中心边界**：`area_face = 0, flux = 0`（L323–329），球对称中心的正确退化处理。✓
- **远场上风判别**：`upwind_from_cell = u_rel > 0.0`（L396），与指导文件 §11.3"(u-vc)>0 取内部"完全一致。✓

---

## 8. `velocity_recovery.py` — 速度恢复

### 结论：完全一致 ✓

#### VR-1 ALE 连续性递推公式验证

代码（`_recover_liquid_phase_velocity`，L226–228）：
```python
C_n = rho_face[f_right]*vc_face[f_right]*area[f_right] - rho_face[f_left]*vc_face[f_left]*area[f_left]
S_n = (current_mass_cell[n] - old_mass_cell[n]) / dt
G_face_abs[f_right] = G_face_abs[f_left] + C_n - S_n
```

对应 ALE 连续方程积分形式：
```
(M^{n+1} - M^n)/dt + G_abs_right - G_abs_left - (ρ vc A)_right + (ρ vc A)_left = 0
```
即 `G_right - G_left = C_n - S_n` → `G_right = G_left + C_n - S_n` ✓

气相侧以 `G_g_if_abs`（由 `iface_pkg` 提供）作为边界条件起点（L260），符合指导文件"气相连续方程从界面面绝对质量流率出发"的要求。✓

---

## 9. `radius_update.py` — 物理界面速度

### 结论：完全一致 ✓

```python
dot_a_phys = u_l_if_abs + mpp / rho_s_l
```

由液相侧跳跃条件 Eq.(2.18)：`ṁ'' = -ρ_l(u_l - ȧ)` → `ȧ = u_l + ṁ''/ρ_l`，与指导文件 §3.5 完全一致。✓

---

## 10. `initial.py` — 初始状态构建

### 结论：完全一致 ✓

#### 验证项

| 项目 | 指导文件规定 | 代码实现 | 结论 |
|------|-------------|----------|------|
| 气相初温 Eq.(2.20) | `T∞ + (a₀/r)(Td0-T∞)erfc((r-a₀)/(2√D_T t₀))` §6.1 | `_build_initial_gas_temperature_profile` L247–264 | ✓ |
| 气相初组分 Eq.(2.21) | 同 erfc 轮廓，使用 `D_{T,g,∞}` §7.1 | `_build_initial_gas_composition_profile`，`diffusivity=D_T_g_inf` L281 | ✓ |
| 两式共用 `D_T` | `t₀ ≪ a₀²/D_{T,g,∞}` §4.3，Eq.(2.20)(2.21) 均用 `D_{T,g,∞}` | `t0_smooth` 统一传入两函数 | ✓ |
| 液相初始均匀场 Eq.(2.22) | `T_l = Td0`，`Y_l = Y_{l,0}` §5.1 | `np.full(n_liq, Td0)`, `np.tile(Yl0)` L304–306 | ✓ |
| 初始 mpp₀ = 0 | 项目规则 §8 | `mpp0=0.0` L475 | ✓ |
| 界面初始 Ts₀ = Td0 | 液相初温等于 Td0 §8.1 | `Ts0=Td0` L472 | ✓ |
| 初始界面气相组成 | 多组分扩展：缩放 Y_seed 保证和为 1 §7.3 | `_build_initial_interface_gas_composition` L190–224 | ✓ |

---

## 偏差汇总

| 编号 | 文件 | 位置 | 严重性 | 描述 |
|------|------|------|--------|------|
| IE-1 | `interface_energy.py` | L66–67 | 中等 | 能量残差等价性依赖液气焓参考态一致性（物性层风险，物理层结构本身正确） |
| FL-2 | `flux_liq.py` | `_validate_diffusion_package` | 轻微 | 未后验校验液相内部面 `Σ_i J_i = 0`，与气相验证逻辑不对称 |

---

## 各模块一致性总表

| 模块 | 主要功能 | 审查结论 |
|------|----------|----------|
| `interface_face.py` | 界面唯一真相源，构建所有界面量 | ✓ 完全一致 |
| `interface_mass.py` | Eq.(2.15/16/18/19) 质量残差 | ✓ 完全一致 |
| `interface_energy.py` | Eq.(2.17) 能量残差 | 中等：IE-1 |
| `energy_flux.py` | `q = -k∇T + ΣJh` 热通量 | ✓ 完全一致 |
| `flux_liq.py` | 液相内部面 Fick 扩散 + 能量通量 | 轻微：FL-2 |
| `flux_gas.py` | 气相内部面混合平均扩散 + 远场 | ✓ 完全一致 |
| `flux_convective.py` | ALE 相对速度上风对流通量 | ✓ 完全一致 |
| `velocity_recovery.py` | ALE 连续性递推恢复面速度 | ✓ 完全一致 |
| `radius_update.py` | `ȧ = u_l + ṁ''/ρ_l` 物理界面速度 | ✓ 完全一致 |
| `initial.py` | Eq.(2.20/21/22) 初始场构建 | ✓ 完全一致 |
