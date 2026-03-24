# src/core 模块代码审查报告

**审查日期**：2026-03-24
**审查范围**：`src/core/` 全部 9 个模块
**参照文档**：`md/physics and numerical/` 下全部指导文件
**审查结论总览**：发现 11 处偏差，其中 2 处为算法级严重问题，7 处为中等功能缺失，2 处为轻微结构偏差。（本报告为修订版：初版 S-1 误判已撤销，S-2/S-3 已更正，新增 S-6/S-7/S-8 三项。）

---

## 一、已审查文件清单

| 文件 | 主要职责 | 审查结论 |
|------|---------|---------|
| `grid.py` | 三区网格构建、控制面速度 | **一致** |
| `remap.py` | 保守量重映射 | **2 处偏差** |
| `state_recovery.py` | 守恒量→原始态恢复 | **7 处偏差** |
| `layout.py` | 未知量向量布局 | **1 处偏差（Phase B）** |
| `state_pack.py` | 状态向量打包/解包 | **一致** |
| `types.py` | 数据结构定义 | **1 处缺项** |
| `preprocess.py` | 配置归一化 | **1 处轻微缺失** |
| `config_loader.py` | YAML 加载与 schema 校验 | **一致** |
| `config_schema.py` | 配置 schema 约束 | **1 处关联缺项** |

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

## 四、`state_recovery.py` — 发现 7 处偏差

参照：`state_recovery_and_enthalpy_inversion_guideline_final.md`

> **修订说明（2026-03-24）**：初版报告 S-1、S-2 两项结论经核查存在误判，本版本予以更正。初版将辅助函数 `_invert_temperature_monotone_bisection` 误认为主调用路径；实际主路径为液相 `_invert_liquid_h_to_T_safeguarded → _invert_temperature_safeguarded_newton`、气相 `_invert_gas_h_to_T_hpy_first`，两者均已实现 Newton 保险步与 HPY 优先分支。原初版 S-1（纯二分法）结论撤销；S-2 更正为接口设计层面的准确描述。此外，S-3 关于"每次从区间中点开始"的表述不准确，一并修正。新增偏差 S-6/S-7/S-8。

---

### ~~偏差 S-1【已撤销】焓反演算法：纯二分法，缺少 Newton 保险步~~

> **该项为误判，已撤销。**
>
> 初版引用 `_invert_temperature_monotone_bisection`（L217-253）作为主路径。经复核，此函数是存在但**未被主路径调用**的辅助函数。实际运行路径为：
> - **液相**：`_recover_liquid_phase_state_with_diagnostics` → `_invert_liquid_h_to_T_safeguarded` → `_invert_temperature_safeguarded_newton`，已实现 Newton 步 + 二分 fallback + 区间收紧。
> - **气相**：`_recover_gas_phase_state_with_diagnostics` → `_invert_gas_h_to_T_hpy_first` → 内部同样调用 `_invert_temperature_safeguarded_newton` 作为 fallback。
>
> 算法实现与指导文件一致，本项不计入偏差。

---

### 偏差 S-1【中等】气相 HPY 主线：接口缺显式 P_inf，依赖 thermo 对象隐式属性

（替换初版错误结论"HPY 主方法未实现 / 死配置"）

**实际实现情况**：`_invert_gas_h_to_T_hpy_first` 已正确实现 HPY-first 分支，`use_cantera_hpy_first` 被真正检查并触发 `_call_temperature_from_hpy`；Newton + 二分 fallback 也已实现。HPY 分支并非"死配置"。

**真正的问题（`_infer_gas_recovery_pressure`）：**
```python
def _infer_gas_recovery_pressure(gas_thermo: GasThermoProtocol) -> float | None:
    reference_pressure = getattr(gas_thermo, "reference_pressure", None)
    if reference_pressure is None:
        return None   # → HPY 被静默跳过，记 skipped_reason="missing_reference_pressure"
```

`recover_state_from_contents` 签名不含显式 `P_inf` 参数，系统压力靠 thermo 对象的隐式属性 `reference_pressure` 推断。只要调用方传入的 thermo 对象未挂载该属性，HPY 主线便静默降级，调用方无感知。

**影响**：HPY 主线的激活与否取决于 thermo 对象实现细节，而非恢复模块的显式接口合同，违反指导文件"显式稳定主路径"要求，增加接口脆弱性。

---

### 偏差 S-2【中等】焓反演初始猜值：仅 sweep 内相邻格传递，缺时间层历史温度

（修正初版"每次从区间中点开始搜索"的不准确表述）

**指导文件规定（§初始猜值优先级）：**
1. `T_0 = T_prev`（**上一时间步**的温度场）
2. 线性估算：`T_0 = T_ref + (h_target − h(T_ref)) / c_p(T_ref)`，钳位至区间内

**实际实现**：代码中 `T_hint` 确实存在并被传入反演函数，但其来源是**当前 sweep 内相邻格**（即空间邻格插值），而非上一时间步保存的历史温度 `T_prev`。时间层历史温度在 API 层面根本未作为参数传入（见偏差 S-8）。

**影响**：相邻格 T_hint 有助于空间平滑情形，但在时间步间温度变化显著（如初始瞬态）时，无法利用真正的历史信息加速收敛。

---

### 偏差 S-3【中等】缺少组分守恒轻微偏差的修正规则

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

### 偏差 S-4【中等】缺少反演后前向一致性校验

**指导文件规定（§后验一致性检查）：**
```
h_recomputed = h(T_recovered, Y)
要求: |h_recomputed − h_target| ≤ h_check_tol
```

**代码**：`recover_state_from_contents`（L256-273）在得到 T 后仅调用 `validate_recovered_state_bounds` 检查温度区间，**未重算 h(T,Y) 与目标值对比**。

**影响**：无法发现因截断退出或数值异常导致的温度与焓不自洽情况。

---

### 偏差 S-5【中等】气相摩尔分数 `Xg_full` 恒为 None

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

### 偏差 S-6【严重，结构问题】`RecoveryConfig` 缺少指导文件要求的多项阈值参数

**指导文件规定（§恢复配置合同）**中明确要求以下参数必须可配置：

| 参数 | 用途 | 当前 `RecoveryConfig` |
|------|------|----------------------|
| `h_check_tol` | 前向一致性校验容差（偏差 S-4 所需）| **缺失** |
| `Y_sum_tol` | 组分归一化修正阈值（偏差 S-3 所需）| **缺失** |
| `rho_min_l` / `rho_min_g` | 密度下界检查（补充节所需）| **缺失** |
| `Y_neg_clip_tol` | 负质量分数钳位阈值（偏差 S-3 所需）| **缺失** |
| `newton_max_iter` / `bisection_max_iter` | 反演迭代上限 | **缺失** |

**代码现状**：`RecoveryConfig` 仅包含温度区间 `T_min_l/g`、`T_max_l/g` 以及 `use_cantera_hpy_first` 布尔开关，阈值参数被硬编码在各反演函数内部。

**影响**：缺失参数使偏差 S-3、S-4 的修正规则在结构层面**无法实现**（即便写了修正逻辑，阈值也无处注入）；同时 `config_schema.py` 的 recovery 块 schema 也随之不完整（见第九节）。此项是 S-3/S-4 修复的结构前置条件，优先级最高。

---

### 偏差 S-7【中等】`recover_state_from_contents` public API 丢弃 recovery diagnostics

**指导文件规定（§恢复接口合同）：**
> 公共接口应同时返回 `State` 和 `RecoveryDiagnostics`，后者包含各相迭代次数、残差、前向误差、修正标志等，供上层做定量监控和日志。

**代码实现**：内部函数 `_recover_liquid_phase_state_with_diagnostics` 和 `_recover_gas_phase_state_with_diagnostics` 已生成详细的每格诊断对象，但 `recover_state_from_contents`（公共入口）仅将诊断传入 `summarize_recovery_diagnostics` 做聚合摘要后再丢弃，返回值只有 `State`。

**影响**：上层调用方无法获取细粒度诊断（如某格迭代超限、某格 HPY 被跳过），诊断能力被锁死在模块内部，与 `remap.py` 的诊断设计模式不一致。

---

### 偏差 S-8【中等】API 缺显式 `T_prev` 参数，历史温度优先策略接口合同缺失

**指导文件规定（§初始猜值优先级）：**
> 恢复接口须接受 `T_prev`（上一时间步温度场数组）作为可选参数，作为最优先的初始猜值来源。

**代码实现**：`recover_state_from_contents` 签名为：
```python
def recover_state_from_contents(
    contents: ConservedContents,
    thermo: ...,
    recovery_cfg: RecoveryConfig,
) -> State:
```
无 `T_prev` 参数。偏差 S-2 中提到的"相邻格 T_hint"是 sweep 内的空间传递，并非时间层历史温度。

**影响**：即使在 `_invert_temperature_safeguarded_newton` 内部增加对时间历史猜值的支持，在当前 API 合同下也无任何通道将 `T_prev` 注入，S-2 的修复在结构层面被此缺陷阻塞。

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

## 九、`config_loader.py` / `config_schema.py`

`config_loader.py` 为纯基础设施代码，与架构规范一致 ✓。

`config_schema.py` 存在一处关联缺项：

### 偏差 CS-1【轻微，关联缺项】recovery 配置块 schema 缺少阈值参数约束

**现状**：`config_schema.py` 中 `recovery` 块的 schema 仅约束 `T_min_l`、`T_max_l`、`T_min_g`、`T_max_g`、`use_cantera_hpy_first`，与 `RecoveryConfig` 的现有字段保持一致。

**影响**：偏差 S-6 要求在 `RecoveryConfig` 中新增的 `h_check_tol`、`Y_sum_tol`、`rho_min_l/g`、`Y_neg_clip_tol` 等参数，需同步在此 schema 中添加约束（类型、范围、默认值），否则配置文件中的对应字段将被 schema 校验拒绝。此项是 **S-6 修复的联动后置步骤**，S-6 完成后须同步处理。

---

## 十、偏差汇总与优先级

> **说明**：初版 S-1（纯二分）为误判，已撤销，不计入下表。S-2/S-3 为更正后的描述。S-6/S-7/S-8/CS-1 为新增项。

| 编号 | 文件 | 严重性 | 描述 | 修复依赖 |
|------|------|--------|------|---------|
| R-1 | `remap.py:287-314` | **严重** | 新暴露体积用端格代替界面态（ρ_{s,l}^n, h_{s,l}^n）| 先补 T-1 |
| S-6 | `state_recovery.py` | **严重（结构）** | `RecoveryConfig` 缺 `h_check_tol`/`Y_sum_tol`/`rho_min` 等阈值参数 | 独立 |
| S-1 | `state_recovery.py` | 中等 | 气相 HPY 主线依赖 thermo 隐式属性 `reference_pressure`，缺显式 `P_inf` 接口 | 独立 |
| S-2 | `state_recovery.py` | 中等 | 焓反演初始猜值仅为 sweep 内相邻格，缺时间层 `T_prev` | 先补 S-8 |
| S-3 | `state_recovery.py` | 中等 | 无组分轻微偏差修正规则，直接抛异常 | 先补 S-6 |
| S-4 | `state_recovery.py:256-273` | 中等 | 缺前向一致性校验 h_recomputed vs h_target | 先补 S-6 |
| S-5 | `state_recovery.py:269` | 中等 | `Xg_full` 恒为 None，气相摩尔分数未恢复 | 独立 |
| S-7 | `state_recovery.py` | 中等 | public API `recover_state_from_contents` 丢弃 recovery diagnostics | 独立 |
| S-8 | `state_recovery.py` | 中等 | API 签名缺显式 `T_prev` 参数，历史温度优先策略无法注入 | 独立 |
| R-2 | `remap.py:366-383` | 轻微 | 诊断仅报绝对值，未计算相对守恒误差 | 独立 |
| L-1 | `layout.py:303-307` | 轻微 | Phase B 界面块 Y_{s,l} 排末尾而非 T_s 之前 | 独立 |
| T-1 | `types.py:868-888` | 轻微（根因）| `InterfaceState` 缺 `hs_l`/`hs_g` 字段 | — |
| P-1 | `preprocess.py:265-278` | 轻微 | 初始 Y 向量未校验 Σ=1 | 独立 |
| CS-1 | `config_schema.py` | 轻微（联动）| recovery schema 缺新增阈值参数约束 | 先补 S-6 |

### 推荐修复顺序

1. **S-6**：补全 `RecoveryConfig` 阈值参数组（解锁 S-3/S-4 修复）
2. **CS-1**：同步在 `config_schema.py` recovery 块中添加对应 schema 约束
3. **T-1**：在 `InterfaceState` 补充 `hs_l`、`hs_g`（解锁 R-1）
4. **R-1**：remap 新暴露体积改用界面态而非端格
5. **S-8**：`recover_state_from_contents` 签名添加 `T_prev` 可选参数（解锁 S-2 修复）
6. **S-1**：将 P_inf 作为显式参数纳入 recovery API，消除对 thermo 隐式属性的依赖
7. **S-7**：public API 返回值改为 `(State, RecoveryDiagnostics)`
8. **S-5**：气相恢复后计算并填充 `Xg_full`
9. **S-4**：反演后添加前向一致性校验
10. **S-3**、**S-2**：组分修正规则与历史温度初始猜值（可同步实现）
11. **R-2**、**P-1**、**L-1**：轻微修复，随版本迭代补充
