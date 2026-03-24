# src/core 模块代码审查报告

**审查日期**：2026-03-24
**审查范围**：`src/core/` 全部 9 个模块
**参照文档**：`md/physics and numerical/` 下全部指导文件
**审查结论总览**：发现 9 处偏差，其中 3 处为算法级严重简化，4 处为中等功能缺失，2 处为轻微结构偏差。

---

## 一、已审查文件清单

| 文件 | 主要职责 | 审查结论 |
|------|---------|---------|
| `grid.py` | 三区网格构建、控制面速度 | **一致** |
| `remap.py` | 保守量重映射 | **2 处偏差** |
| `state_recovery.py` | 守恒量→原始态恢复 | **6 处偏差** |
| `layout.py` | 未知量向量布局 | **1 处偏差（Phase B）** |
| `state_pack.py` | 状态向量打包/解包 | **一致** |
| `types.py` | 数据结构定义 | **1 处缺项** |
| `preprocess.py` | 配置归一化 | **1 处轻微缺失** |
| `config_loader.py` | YAML 加载与 schema 校验 | **一致** |
| `config_schema.py` | 配置 schema 约束 | **一致** |

---

## 二、`grid.py` — 与指导文件一致 ✓

参照：`grid_partition_and_moving_mesh_guideline_final.md`

| 计算项 | 指导文件规定 | 代码实现 | 状态 |
|--------|------------|---------|------|
| Region 1 面坐标 | `r_{1,f}(j) = j·(a/N₁)` | `np.linspace(0, a, n_liq+1)` | ✓ |
| Region 2 面坐标 | `r_{2,f}(j) = a + j·((r_I−a)/N₂)` | `np.linspace(a, r_I, n_gas_near+1)` | ✓ |
| Region 3 首格大小 | `Δr₃,₁ = r_I / N₂ = 5a₀/N₂` | `dr3_first = r_I / n_gas_near`（L54）| ✓ |
| 球坐标格体积 | `V = (4π/3)(r₊³ − r₋³)` | `(4π/3)*(r[1:]³ − r[:-1]³)`（L98）| ✓ |
| Region 1 控制面速度 | `v_c(r) = ȧ·r/a` | `dot_a * r / a`（L137）| ✓ |
| Region 2 控制面速度 | `v_c(r) = ȧ·(r_I−r)/(r_I−a)` | `dot_a*(r_I−r)/(r_I−a)`（L138-140）| ✓ |
| Region 3 控制面速度 | `v_c = 0` | 数组初始化为零，Region 3 不更新（L132）| ✓ |

`v_c_cells`（L151）取相邻面值的算术平均。指导文件仅定义面速度，未规定胞中心插值方式；对线性 v_c 分布（Region 1、2 均为线性），算术平均等于精确值，无误差。

---

## 三、`remap.py` — 发现 2 处偏差

参照：`remap_and_conservative_projection_guideline_final_v2.md`

### 偏差 R-1【严重】新暴露子体积填充使用体相端格而非界面态

**指导文件规定（§新暴露子体积处理）：**
> 新暴露子体积的参考态必须取**旧时间层界面态**：
> `(ρ·ΔV)_exp = ρ_{s,l}^n · ΔV`，
> `(ρY_i·ΔV)_exp = ρ_{s,l}^n · Y_{s,l,i}^n · ΔV`，
> `(ρh·ΔV)_exp = ρ_{s,l}^n · h_{s,l}^n · ΔV`

**代码实现（L287-314）：**
```python
# 液相新暴露体积
reference_rho   = float(old_state.rho_l[-1])      # 液相最外层格 ← 应为 rho_s_l^n
reference_y_full = old_state.Yl_full[-1, :]        # 液相最外层格 ← 应为 Ys_l_full^n
reference_h     = float(old_state.hl[-1])          # 液相最外层格 ← 应为 hs_l^n

# 气相新暴露体积
reference_rho   = float(old_state.rho_g[0])        # 气相最内层格 ← 应为 rho_s_g^n
reference_y_full = old_state.Yg_full[0, :]         # 气相最内层格 ← 应为 Ys_g_full^n
reference_h     = float(old_state.hg[0])           # 气相最内层格 ← 应为 hs_g^n
```

**影响**：液相均匀、界面态与端格接近时误差小；当界面组分或温度梯度显著时，引入额外守恒误差。根本原因在于 `InterfaceState`（`types.py`）目前未携带 `hs_l`、`hs_g` 字段（见偏差 T-1）。

---

### 偏差 R-2【轻微】重映射诊断未计算相对误差

**指导文件要求输出（§守恒验证）：**
```
remap_mass_err_liquid   = |M^{old,*} − M^{old}| / max(M^{old}, ε)
remap_mass_err_gas
remap_species_err_max
remap_enthalpy_err_liquid
remap_enthalpy_err_gas
```

**代码实现（`summarize_remap_diagnostics`，L366-383）：**
```python
"mass_l_before": ..., "mass_l_after": ...,   # 仅原始值，未计算相对误差
"enthalpy_l_before": ..., "enthalpy_l_after": ...,
```

**影响**：无相对误差计算，无法对守恒性做定量监控和阈值报警。

---

## 四、`state_recovery.py` — 发现 6 处偏差

参照：`state_recovery_and_enthalpy_inversion_guideline_final.md`

### 偏差 S-1【严重】焓反演算法：纯二分法，缺少 Newton 保险步

**指导文件规定（§液相焓反演）：**
> 使用 **Safeguarded Newton with Bisection Fallback**：
> 1. Newton 步：`T_new = T − f(T)/f'(T)`，其中 `f'(T) = c_p(T,Y)`
> 2. 若 Newton 步越出区间或导数异常，降级为二分
> 3. 每步同时更新区间端点收紧范围

**代码实现（`_invert_temperature_monotone_bisection`，L81-117）：**
```python
for _ in range(max_iter):
    mid = 0.5 * (left + right)   # 始终二分，无 Newton 步
    h_mid = float(thermo.enthalpy_mass(mid, y_full))
    if abs(h_mid - target_h) <= tol:
        return mid
    if h_mid < target_h:
        left = mid
    else:
        right = mid
```

**影响**：纯二分法收敛率 O(log₂N)，Newton 法在良好条件下二阶收敛。50 次迭代、温区 [270, 350] 时精度约 7×10⁻¹⁴ K，可满足当前精度需求，但在宽温区或多组分场景下迭代代价更高。

---

### 偏差 S-2【严重】气相焓反演：未实现 Cantera HPY 主方法

**指导文件规定（§气相焓反演）：**
> 主方法：Cantera HPY setter，`gas.HPY = h_target, P_inf, Y_full; T = gas.T`
> 失败时降级为 Newton/二分备选

**代码实现（`_invert_gas_h_to_T`，L147-171）：**
```python
# 与液相完全相同的纯二分实现，未调用任何 Cantera HPY 接口
return np.array([
    _invert_temperature_monotone_bisection(
        T_low=recovery_cfg.T_min_g,
        T_high=recovery_cfg.T_max_g,
        ...
    ) for ...
])
```

**影响**：配置项 `use_cantera_hpy_first: true` 被读取存储但**从不使用**，形成"死配置"；Cantera HPY 对多组分气相的收敛性和数值精度均优于纯二分法。

---

### 偏差 S-3【中等】无历史温度初始猜值

**指导文件规定（§初始猜值优先级）：**
1. `T_0 = T_prev`（上一时间层温度）
2. 线性估算：`T_0 = T_ref + (h_target − h(T_ref)) / c_p(T_ref)`，钳位至区间内

**代码**：纯二分法无需也未使用任何初始猜值，每次从区间中点开始搜索。

**影响**：在时间步之间解变化缓慢时，未利用历史信息，无谓消耗迭代次数。

---

### 偏差 S-4【中等】缺少组分守恒轻微偏差的修正规则

**指导文件规定（§最小修正规则）：**
1. 若 `|ρY_i| < ε_abs`，钳位为零
2. 若 `|Σ(ρY_i) − ρ| < ε_tol`，等比例缩放所有非零分量

**代码实现（L54-61）：**
```python
if np.any(species_mass < 0.0):
    raise StateRecoveryError("species_mass must be non-negative")
# 若 sum ≠ mass 也直接 raise
```

**影响**：小数值舍入产生的轻微负质量分数（如 −1e-15）在指导文件中属于可修正情形，代码直接抛出异常，将可恢复故障升级为致命错误。

---

### 偏差 S-5【中等】缺少反演后前向一致性校验

**指导文件规定（§后验一致性检查）：**
```
h_recomputed = h(T_recovered, Y)
要求: |h_recomputed − h_target| ≤ h_check_tol
```

**代码**：`recover_state_from_contents`（L256-273）在得到 T 后仅调用 `validate_recovered_state_bounds` 检查温度区间，**未重算 h(T,Y) 与目标值对比**。

**影响**：无法发现因截断退出或数值异常导致的温度与焓不自洽情况。

---

### 偏差 S-6【中等】气相摩尔分数 `Xg_full` 恒为 None

**指导文件规定（§气相恢复输出，Hard 要求）：**
> 气相恢复必须同时输出 `Yg_full`（质量分数）和 `Xg_full`（摩尔分数），
> 失败视为致命错误。

**代码（L269）：**
```python
state = State(
    ...
    Xg_full=None,   # 硬编码 None，从不计算
)
```

**影响**：界面平衡计算（Raoult 定律）、诊断量及后续组分残差均需摩尔分数，下游模块须自行重算或跳过此校验。

---

### 补充【轻微】后验检查范围不完整

`validate_recovered_state_bounds`（L222-226）仅检查温度区间。指导文件还要求：

| 检查项 | 指导文件要求 | 代码 |
|--------|------------|------|
| 密度下界 | `ρ > ρ_min` | 未检查 |
| 质量分数范围 | `0 ≤ Y_i ≤ 1`, `\|ΣY_i − 1\| ≤ Y_sum_tol` | 未检查 |
| 气相完整组分 | `X_i` 合法且与 `Y_i` 一致 | 未检查 |
| 前向一致性 | `\|h_recomputed − h_target\|` | 未检查（同 S-5）|

`summarize_recovery_diagnostics`（L277-285）仅返回 6 个量（`min/max_Tl`, `min/max_Tg`, `min_rho_l`, `min_rho_g`），指导文件要求约 20 个诊断量（各相反演迭代次数、残差、前向误差、修正次数等）。

---

## 五、`layout.py` — 发现 1 处 Phase B 结构偏差

参照：`unknowns_strategy_guideline_final.md`

### 偏差 L-1【轻微，仅影响 Phase B】界面块未知量排序不一致

**指导文件 Phase B 界面块排序：**
```
[Y_{s,l,1} … Y_{s,l,Nle} | T_s | Y_{s,g,1} … Y_{s,g,Nge} | mpp]
```

**代码排序（`_build_field_slices`，L303-307）：**
```python
if_temperature_slice  = [if_start : if_start+1]          # T_s 在首位
if_gas_species_slice  = [T_s.stop : T_s.stop+n_gas_red]  # Y_{s,g} 紧随
if_mpp_slice          = [Y_sg.stop : Y_sg.stop+1]         # mpp
if_liq_species_slice  = [mpp.stop : mpp.stop+n_liq_red]  # Y_{s,l} 在末尾
```

即代码布局为 `[T_s | Y_{s,g} | mpp | Y_{s,l}]`，液相接口物种被移到末尾。

**影响**：Phase A（当前算例，`n_liq_red=0`）液相接口块为空，此偏差无实际影响；切换 Phase B 多组分液相时，Jacobian fieldsplit 分块结构与指导不符，影响 Schur 预条件器的块结构假设。

---

## 六、`state_pack.py` — 一致 ✓

参照：`unknowns_strategy_guideline_final.md`

| 要求 | 规定 | 代码 | 状态 |
|------|------|------|------|
| 闭包物种重建 | `Y_cl = 1 − Σ Y_i^{red}` | `full[idx] = 1.0 - np.sum(reduced)`（L104）| ✓ |
| 单组分液相零 reduced | 返回 `[1.0]` | `return np.array([1.0])`（L99）| ✓ |
| Rd, u_l, u_g, ȧ 不入向量 | 明确排除 | 仅打包 T、Y、mpp、Ts | ✓ |
| 解包后 derived fields 为 None | 待属性计算后填充 | `rho_l=None, hl=None, Xg_full=None`（L211-215）| ✓ |

---

## 七、`types.py` — 发现 1 处关联缺项

参照：`remap_and_conservative_projection_guideline_final_v2.md`

### 偏差 T-1【轻微，结构根因】`InterfaceState` 缺少界面焓与密度字段

**现状（L868-888）：**
```python
class InterfaceState:
    Ts: float
    mpp: float
    Ys_g_full: FloatArray
    Ys_l_full: FloatArray
    # 缺少: hs_l, hs_g, rho_s_l, rho_s_g
```

**指导文件**：remap 新暴露体积填充需要 `ρ_{s,l}^n, h_{s,l}^n, ρ_{s,g}^n, h_{s,g}^n`。`rho_s_l`/`rho_s_g` 标注为 "future only"，但 `hs_l`/`hs_g` 未做此标注。

**影响**：`InterfaceState` 不携带这两个字段，使 `remap.py` 结构上无法实现指导文件规定的新暴露体积填充方案，是**偏差 R-1 的结构根因**。

---

## 八、`preprocess.py` — 发现 1 处轻微缺失

参照：`unknowns_strategy_guideline_final.md`（初始状态物理合法性）

### 偏差 P-1【轻微】初始质量分数未验证归一化

**代码（`_build_full_mass_fraction_vector`，L265-278）：**
```python
vector = np.zeros(len(full_names), dtype=np.float64)
for name, value in provided_mass_fractions.items():
    vector[full_name_to_index[name]] = float(value)
return vector   # 未检查 Σ Y_i ≈ 1.0
```

**影响**：若配置文件中组分质量分数之和不为 1（如笔误），将被静默接受进入计算，产生不物理的初始状态。

---

## 九、`config_loader.py` / `config_schema.py` — 一致 ✓

两者均为纯基础设施代码，无物理/数值内容，与指导文件架构规范一致。

---

## 十、偏差汇总与优先级

| 编号 | 文件 | 严重性 | 描述 | 修复依赖 |
|------|------|--------|------|---------|
| R-1 | `remap.py:287-314` | **严重** | 新暴露体积用端格代替界面态（ρ_{s,l}^n, h_{s,l}^n）| 先补 T-1 |
| S-1 | `state_recovery.py:81-117` | **严重** | 焓反演仅纯二分，无 Newton 保险步 | 独立 |
| S-2 | `state_recovery.py:147-171` | **严重** | 气相反演未实现 Cantera HPY 主方法（死配置项）| 独立 |
| S-3 | `state_recovery.py:130-144` | 中等 | 无历史温度初始猜值 | 独立 |
| S-4 | `state_recovery.py:54-61` | 中等 | 无组分轻微偏差修正，直接报错 | 独立 |
| S-5 | `state_recovery.py:256-273` | 中等 | 缺前向一致性校验 h_recomputed vs h_target | 独立 |
| S-6 | `state_recovery.py:269` | 中等 | Xg_full 恒为 None，气相摩尔分数未恢复 | 独立 |
| R-2 | `remap.py:366-383` | 轻微 | 诊断仅报绝对值，未计算相对守恒误差 | 独立 |
| L-1 | `layout.py:303-307` | 轻微 | Phase B 界面块 Y_{s,l} 排末尾而非 T_s 之前 | 独立 |
| T-1 | `types.py:868-888` | 轻微（根因）| InterfaceState 缺 hs_l/hs_g 字段 | — |
| P-1 | `preprocess.py:265-278` | 轻微 | 初始 Y 向量未校验 Σ=1 | 独立 |

### 推荐修复顺序

1. **T-1**：在 `InterfaceState` 补充 `hs_l`、`hs_g`（为 R-1 解锁）
2. **R-1**：remap 新暴露体积改用界面态而非端格
3. **S-2**：实现 Cantera HPY 主方法，激活已有配置项
4. **S-1**：液相/气相焓反演改为 Safeguarded Newton + 二分 fallback
5. **S-6**：气相恢复后计算并填充 `Xg_full`
6. **S-5**：反演后添加前向一致性校验
7. **S-3**、**S-4**：初始猜值与组分修正规则（可同步实现）
8. **R-2**、**P-1**、**L-1**：轻微修复，随版本迭代补充
