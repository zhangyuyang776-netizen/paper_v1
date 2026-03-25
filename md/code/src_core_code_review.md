# src/core 模块代码审查报告

**审查日期**：2026-03-25
**审查范围**：`src/core/` 全部 9 个模块
**参照文档**：`md/physics and numerical/` 下相关指导文件
**审查结论总览**：发现 16 处偏差，其中 2 处为严重实现偏差，7 处为中等功能/接口偏差，7 处为轻微结构或配置合同偏差。

> **修订说明（2026-03-25）**：
> - 保留并确认：`R-1`、`R-2`、`S-1`、`S-2`、`S-3`、`S-4`、`S-5`、`L-1`、`P-1`
> - 重写：`S-6`、`S-7`、`T-1`、`CS-1`
> - 新增：`R-3`、`CS-2`、`CS-3`
> - 删除：原 `S-8`

---

## 一、已审查文件清单

| 文件 | 主要职责 | 审查结论 |
|------|---------|---------|
| `grid.py` | 三区网格构建、控制面速度 | **一致** |
| `remap.py` | 保守量重映射 | **3 处偏差** |
| `state_recovery.py` | 守恒量→原始态恢复 | **7 处偏差** |
| `layout.py` | 未知量向量布局 | **1 处偏差（Phase B）** |
| `state_pack.py` | 状态向量打包/解包 | **一致** |
| `types.py` | 数据结构定义 | **1 处轻微结构缺口** |
| `preprocess.py` | 配置归一化 | **1 处轻微缺失** |
| `config_loader.py` | YAML 加载 | **一致** |
| `config_schema.py` | 配置 schema 约束 | **3 处偏差** |

---

## 二、`grid.py` — 与指导文件一致 ✓

参照：`grid_partition_and_moving_mesh_guideline_final.md`

| 计算项 | 指导文件规定 | 代码实现 | 状态 |
|--------|------------|---------|------|
| Region 1 面坐标 | `r_{1,f}(j) = j·(a/N₁)` | `np.linspace(0, a, n_liq+1)` | ✓ |
| Region 2 面坐标 | `r_{2,f}(j) = a + j·((r_I−a)/N₂)` | `np.linspace(a, r_I, n_gas_near+1)` | ✓ |
| Region 3 首格大小 | `Δr₃,₁ = r_I / N₂ = 5a₀/N₂` | `dr3_first = r_I / n_gas_near` | ✓ |
| 球坐标格体积 | `V = (4π/3)(r₊³ − r₋³)` | `(4π/3)*(r[1:]³ − r[:-1]³)` | ✓ |
| Region 1 控制面速度 | `v_c(r) = ȧ·r/a` | `dot_a * r / a` | ✓ |
| Region 2 控制面速度 | `v_c(r) = ȧ·(r_I−r)/(r_I−a)` | `dot_a*(r_I−r)/(r_I−a)` | ✓ |
| Region 3 控制面速度 | `v_c = 0` | Region 3 保持零值 | ✓ |

`v_c_cells` 取相邻面速度算术平均。指导文件未单独规定胞中心插值方式；对 Region 1/2 的线性分布，这一实现与精确值一致。

---

## 三、`remap.py` — 发现 3 处偏差

参照：`remap_and_conservative_projection_guideline_final_v2.md`

### 偏差 R-1【严重】新暴露子体积填充仍使用体相端格态，而非旧时间层界面态

**指导文件规定（§7.4.2）**：

对 newly exposed subvolume，正式采用的是：

- 旧已接受时间层的界面 `T_s^n`
- 旧已接受时间层的界面 `Y_{s,l}^n / Y_{s,g}^n`
- 再通过当前物性主线得到 `rho_{s,l/g}^n`、`h_{s,l/g}^n`

**代码实现**：

```python
def _extract_liquid_completion_reference(old_state: State) -> tuple[float, np.ndarray, float]:
    return float(old_state.rho_l[-1]), np.asarray(old_state.Yl_full[-1, :]), float(old_state.hl[-1])

def _extract_gas_near_completion_reference(old_state: State, old_mesh: Mesh1D) -> tuple[float, np.ndarray, float]:
    return float(old_state.rho_g[0]), np.asarray(old_state.Yg_full[0, :]), float(old_state.hg[0])
```

新暴露体积补全直接取液相最外层格、气相最内层格的 bulk cell-average，而不是旧界面态。

**影响**：当界面态与端格态存在明显温度或组分梯度时，remap 会在界面附近引入额外守恒误差与不一致初值。该问题成立，但其结构原因不止 `InterfaceState` 缺字段，见 `R-3` 与 `T-1`。

---

### 偏差 R-2【轻微】重映射诊断只报 before/after，总结里没有相对守恒误差

**指导文件规定（§8）**要求至少计算：

```text
epsilon_M_l = |M_l^{old,*} - M_l^{old}| / max(M_l^{old}, eps)
epsilon_M_g = |M_g^{old,*} - M_g^{old}| / max(M_g^{old}, eps)
epsilon_M_k = |M_k^{old,*} - M_k^{old}| / max(M_k^{old}, eps)
```

**代码实现（`summarize_remap_diagnostics`）**：

```python
"mass_l_before": ...,
"mass_l_after": ...,
"mass_g_before": ...,
"mass_g_after": ...,
"enthalpy_l_before": ...,
"enthalpy_l_after": ...,
```

只输出前后总量，没有计算相对误差、最大组分误差或阈值友好的守恒指标。

**影响**：无法直接做守恒监控、阈值报警和回归比较。

---

### 偏差 R-3【中等，结构级】remap 主路径没有 thermo / `P_inf` 或等价界面热力学参考态通道

**指导文件口径**：新暴露子体积可由旧界面 `Ts^n`、`Ys^n` 加当前物性主线派生 `rho_s`、`h_s`；也可以由上游先提供等价的界面热力学参考态。

**代码实现**：

```python
def _build_transferred_contents(
    *,
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
) -> ConservativeContents:
    ...
```

`_build_transferred_contents()` 只拿到 `old_state / old_mesh / new_mesh`，拿不到：

- `liquid_thermo`
- `gas_thermo`
- `P_inf`
- 或预计算的 `rho_s/h_s` 参考态对象

因此这条 remap 主链无法按指导文件从 `Ts/Ys` 严格派生 `rho_s/h_s`，只能退回去读 bulk 端格。

**影响**：`R-1` 不是简单补几个字段就能自动修好；真正缺的是 remap 补全路径的正式依赖通道。

---

## 四、`state_recovery.py` — 发现 7 处偏差

参照：`state_recovery_and_enthalpy_inversion_guideline_final.md`

> **修订说明**：初版关于“纯二分法 / HPY 未实现”的判断已撤销。复核后确认：液相主线为 safeguarded Newton，气相存在 HPY-first 分支，原始误判不再计入偏差。

### 偏差 S-1【中等】气相 HPY 分支存在，但激活依赖隐式 pressure 来源

**已确认正确的部分**：`_invert_gas_h_to_T_hpy_first()` 已实现 HPY-first 和 fallback 标量反演，`use_cantera_hpy_first` 不是死配置。

**真正的不一致点**：

```python
def _infer_gas_recovery_pressure(gas_thermo: GasThermoProtocol) -> float | None:
    reference_pressure = getattr(gas_thermo, "reference_pressure", None)
    ...
```

气相恢复的压力来源不是 recovery 接口显式输入，而是从 `gas_thermo.reference_pressure` 隐式推断。若 thermo 对象未挂载该属性，HPY 主线会被跳过并退回 scalar inversion。

**影响**：HPY 主线的可用性依赖 thermo 实现细节，而非 recovery 模块的正式接口合同。

---

### 偏差 S-2【中等】焓反演未接入时间层历史温度，只在 sweep 内传递相邻格 `T_hint`

**指导文件规定（§7.7）**：

第一优先初值为历史温度：

```text
T_0 = T_prev
```

第二优先才是基于 `cp_min` 的线性估计。

**代码实现**：

```python
T_hint: float | None = None
for i, h_i in enumerate(hl):
    T_i, mode_i, bounds_i = _invert_liquid_h_to_T_safeguarded(..., T_hint=T_hint)
    ...
    T_hint = T_i
```

液相和气相都只把上一个 sweep cell 的温度作为下一个 cell 的 `T_hint`，没有上一时间层或 remap 前同一控制体的 `T_prev` 输入通道。

**影响**：空间上平滑时可受益，但无法落实指导文件规定的“历史温度优先”策略。

---

### 偏差 S-3【中等】组分轻微偏差没有 minor-fix，仍是直接 hard fail

**指导文件规定（§5.6 / §11）**：

- 对绝对值小于 `species_recovery_eps_abs` 的负 partial density 置零
- 若 `|Σ(ρY_i) - ρ|` 小于阈值，则按比例缩放非零分量

**代码实现**：

```python
if np.any(species_mass < 0.0):
    raise StateRecoveryError("species_mass must be non-negative")
...
if np.any(diff > tol):
    raise StateRecoveryError("species_mass sums must match mass within tolerance")
```

只要出现轻微负值或轻微闭合误差，当前实现就直接失败。

**影响**：把指导文件允许的 rounding-level minor fix 全部升级成致命错误。

---

### 偏差 S-4【中等】缺少焓反演后的前向一致性校验

**指导文件规定（§7.9 / §12）**：

```text
h_recomputed = h(T_recovered, Y)
要求 |h_recomputed - h_target| <= h_check_tol
```

**代码实现**：恢复完温度后只调用 `validate_recovered_state_bounds()` 检查温区，没有正向回代 `h(T,Y)` 去对比 target enthalpy。

**影响**：无法发现温度收敛形式成立但焓一致性已偏离的情况。

---

### 偏差 S-5【中等】气相摩尔分数 `Xg_full` 仍恒为 `None`

**指导文件规定（§8.5）**：

```text
气相恢复后必须得到：
- Yg_full
- Xg_full
```

**代码实现**：

```python
state = State(
    ...
    Xg_full=None,
)
```

公共恢复路径从未计算气相 full mole fraction。

**影响**：后续 transport、界面平衡和 diagnostics 所需的 full gas composition 仍不完整。

---

### 偏差 S-6【严重，结构问题】`RecoveryConfig` 现有字段集合与指导文件要求相比仍有大面积缺项

**先纠正事实描述**：当前 `RecoveryConfig` 并非“只有温区和一个布尔开关”，代码里已经有：

- `liq_h_inv_tol`
- `liq_h_inv_max_iter`
- `gas_h_inv_tol`
- `gas_h_inv_max_iter`
- `use_cantera_hpy_first`

**真正的问题**：与指导文件 §13 / config schema 指导文件 §7.11 对照，`RecoveryConfig` 仍缺少大量正式参数，例如：

| 指导文件参数 | 用途 | 当前 `RecoveryConfig` |
|------|------|------|
| `rho_min` | 密度/状态合法性下界 | 缺失 |
| `m_min` | 质量下界 | 缺失 |
| `species_recovery_eps_abs` | minor-fix 负 partial density 阈值 | 缺失 |
| `Y_sum_tol` | 组分和偏差容差 | 缺失 |
| `Y_hard_tol` | 组分 hard-fail 容差 | 缺失 |
| `h_abs_tol` / `h_rel_tol` | 焓反演收敛容差体系 | 缺失 |
| `h_check_tol` | 前向一致性校验容差 | 缺失 |
| `T_step_tol` | 温度步长/收敛控制 | 缺失 |
| `cp_min` | 线性估计与稳定保护 | 缺失 |
| `liquid_h_inv_max_iter` | 正式液相最大迭代数命名 | 语义未对齐，见 `CS-2` |

**影响**：`S-3`、`S-4` 以及合法性校验相关逻辑缺少正式配置注入口，当前实现只能依赖硬编码或根本不实现。

---

### 偏差 S-7【中等】public API 没有向上层暴露 recovery diagnostics

**代码执行链**：

```python
def _recover_state_from_contents_internal(...) -> tuple[State, dict[str, object]]:
    ...
    return state, diagnostics

def recover_state_from_contents(...) -> State:
    state, _ = _recover_state_from_contents_internal(...)
    return state
```

内部实现确实生成并返回 `(state, diagnostics)`；公共入口则直接丢弃 diagnostics，并非“先 summarize 再丢”。

**指导文件约束的准确表述**：指导文件要求 recovery 至少输出 diagnostics，供 remap / diagnostics / logging 链路使用；但并未硬性规定 public API 必须采用 `(State, RecoveryDiagnostics)` 这一唯一返回型式。

**影响**：当前真实不一致是“上层没有正式通道拿到 recovery diagnostics”，而不是“返回值型式必须等于某个固定二元组”。

---

### 补充【轻微】恢复后的校验与诊断范围仍明显不完整

`validate_recovered_state_bounds()` 仅检查温度区间；结合指导文件，还缺少：

| 检查项 | 指导文件要求 | 当前代码 |
|--------|------------|------|
| 密度下界 | `rho > rho_min` | 未检查 |
| 质量分数范围 | `0 <= Y_i <= 1`，且 `|sum(Y)-1| <= Y_sum_tol` | 未检查 |
| 前向焓一致性 | `|h_recomputed - h_target| <= h_check_tol` | 未检查（同 `S-4`） |
| 气相完整组成 | `Xg_full` 合法且与 `Yg_full` 一致 | 未检查（同 `S-5`） |

同时，`summarize_recovery_diagnostics()` 只返回：

- `min/max_Tl`
- `min/max_Tg`
- `min_rho_l`
- `min_rho_g`

而指导文件 §12 要求的 diagnostics 还包括：

- enthalpy inversion iteration counts
- residual / recomputed enthalpy error
- minor-fix 次数
- HPY/fallback 标志
- 全局 fail 计数与极值类诊断

---

## 五、`layout.py` — 发现 1 处 Phase B 结构偏差

参照：`interface_block_unknowns_and_residuals_table_final.md`

### 偏差 L-1【轻微，仅影响 Phase B】界面块未知量顺序不符合指导文件

**指导文件 Phase B 顺序**：

```text
[Y_{s,l,red} | T_s | Y_{s,g,red} | mpp]
```

**代码顺序（`_build_field_slices`）**：

```python
if_temperature_slice = ...
if_gas_species_slice = ...
if_mpp_slice = ...
if_liq_species_slice = ...
```

实际布局为：

```text
[T_s | Y_{s,g,red} | mpp | Y_{s,l,red}]
```

**影响**：Phase A 下 `n_liq_red = 0`，当前算例几乎不受影响；切到 Phase B 多组分液滴后，界面块顺序与指导文件不一致。

---

## 六、`state_pack.py` — 与指导文件一致 ✓

参照：`unknowns_strategy_guideline_final.md`

| 要求 | 指导文件规定 | 代码实现 | 状态 |
|------|------|------|------|
| 闭包物种重建 | `Y_cl = 1 - ΣY_i^{red}` | 已实现 | ✓ |
| 单组分液相零 reduced | 返回 `[1.0]` | 已实现 | ✓ |
| `R_d`、`u_l`、`u_g`、`ȧ` 不入向量 | 明确排除 | 已实现 | ✓ |
| 解包后 derived fields 先置空 | 待属性恢复后填充 | 已实现 | ✓ |

---

## 七、`types.py` — 发现 1 处轻微结构缺口

参照：`remap_and_conservative_projection_guideline_final_v2.md`

### 偏差 T-1【轻微】`InterfaceState` 未携带预计算界面热力学参考态

**现状**：

```python
class InterfaceState:
    Ts: float
    mpp: float
    Ys_g_full: FloatArray
    Ys_l_full: FloatArray
```

`InterfaceState` 只保存 `Ts` 和 `Ys`，不保存预计算的：

- `hs_l` / `hs_g`
- `rho_s_l` / `rho_s_g`

**准确解读**：这不是 `R-1` 的唯一根因。因为指导文件允许两条实现路径：

1. 在界面态对象里直接携带 `rho_s/h_s`
2. 在 remap 阶段通过 `Ts/Ys + thermo/P_inf` 现场派生

当前 `InterfaceState` 缺的是第一条路径所需的数据槽位；而 `R-1` 更深层的结构阻塞是 `R-3` 中 remap 主链拿不到派生所需依赖。

**影响**：若希望把界面热力学参考态显式缓存进 state/types 层，当前类型定义仍不够用；但不能把它写成 `R-1` 的唯一结构根因。

---

## 八、`preprocess.py` — 发现 1 处轻微缺失

参照：`paper_v1 Initialization and First-Step Guideline.md`

### 偏差 P-1【轻微】初始质量分数向量未检查归一化

**代码（`_build_full_mass_fraction_vector`）**：

```python
vector = np.zeros(len(full_names), dtype=np.float64)
for name, value in provided_mass_fractions.items():
    vector[full_name_to_index[name]] = float(value)
return vector
```

这里只检查了未知 species 名称，没有检查：

- `sum(Y) ≈ 1`
- 各分量是否落在合法范围

**影响**：配置文件里若出现组分和不为 1 的笔误，会被静默接受进入初始化。

---

## 九、`config_loader.py` / `config_schema.py`

`config_loader.py` 为基础设施代码，本轮未发现偏差。

`config_schema.py` 存在三处 recovery 合同相关问题：

### 偏差 CS-1【轻微】recovery schema 仍缺少指导文件要求的大量显式字段

**先纠正事实描述**：当前 schema 并不是“只约束温区和布尔开关”，而是已经约束了：

- `T_min_l`
- `T_max_l`
- `T_min_g`
- `T_max_g`
- `liq_h_inv_tol`
- `liq_h_inv_max_iter`
- `gas_h_inv_tol`
- `gas_h_inv_max_iter`
- `use_cantera_hpy_first`

**真正问题**：相对于指导文件，schema 仍未覆盖以下应显式配置的 recovery 字段：

- `rho_min`
- `m_min`
- `species_recovery_eps_abs`
- `Y_sum_tol`
- `Y_hard_tol`
- `h_abs_tol`
- `h_rel_tol`
- `h_check_tol`
- `T_step_tol`
- `cp_min`
- `liquid_h_inv_max_iter`

**影响**：即使后续补齐 `RecoveryConfig`，schema 若不同步扩展，配置文件也无法合法表达指导文件要求的 recovery 合同。

---

### 偏差 CS-2【轻微，合同偏移】recovery 现有字段名与语义已经偏离指导文件正式命名

**当前代码字段**：

```text
liq_h_inv_tol
liq_h_inv_max_iter
gas_h_inv_tol
gas_h_inv_max_iter
```

**指导文件正式口径**强调的是：

```text
h_abs_tol
h_rel_tol
h_check_tol
liquid_h_inv_max_iter
gas_h_inv_max_iter
cp_min
```

也就是说，当前问题不只是“少字段”，还包括：

- 已有字段名与指导文件命名不一致
- 液相/气相容差被实现成另一套语义
- 指导文件要求的通用焓容差体系未被保留

**影响**：后续若只做“补字段”而不清理命名与语义合同，`RecoveryConfig`、schema、配置样例和指导文件仍会长期错位。

---

### 偏差 CS-3【轻微，内部合同不一致】schema 与 `RecoveryConfig` 对焓反演容差下界的约束不一致

**当前 schema 约束**：

```python
SchemaField("liq_h_inv_tol", float, required=True, min_value=0.0)
SchemaField("gas_h_inv_tol", float, required=True, min_value=0.0)
```

这意味着 `config_schema.py` 会放行：

- `liq_h_inv_tol = 0.0`
- `gas_h_inv_tol = 0.0`

**但 `RecoveryConfig` dataclass 约束**：

```python
_check_positive("liq_h_inv_tol", self.liq_h_inv_tol)
_check_positive("gas_h_inv_tol", self.gas_h_inv_tol)
```

`_check_positive()` 要求的是严格正值，因此 `0.0` 会在 schema 通过后、构造 `RecoveryConfig` 时再次被拒绝。

**影响**：这是 `core` 内部的配置合同自相矛盾。表现上会变成“schema 验证成功，但初始化阶段失败”，增加排障成本，也使 schema 失去作为前置合同检查的完整性。

---

## 十、偏差汇总与优先级

| 编号 | 文件 | 严重性 | 描述 | 修复依赖 |
|------|------|--------|------|---------|
| R-1 | `remap.py` | **严重** | 新暴露体积补全仍取 bulk 端格态，不是旧界面态 | 先补 `R-3` 或等价通道 |
| S-6 | `types.py` / `state_recovery.py` | **严重（结构）** | `RecoveryConfig` 相比指导文件仍缺大量恢复/校验参数 | 联动 `CS-1`、`CS-2` |
| R-3 | `remap.py` | 中等 | remap 主链缺 thermo / `P_inf` / 等价界面热力学参考态通道 | 独立 |
| S-1 | `state_recovery.py` | 中等 | 气相 HPY 分支依赖 thermo 隐式 pressure 来源 | 独立 |
| S-2 | `state_recovery.py` | 中等 | 未接入时间层历史温度 `T_prev`，仅 sweep 内传 `T_hint` | 独立 |
| S-3 | `state_recovery.py` | 中等 | minor-fix 未实现，轻微组分偏差直接 hard fail | 先补 `S-6` |
| S-4 | `state_recovery.py` | 中等 | 缺前向焓一致性校验 | 先补 `S-6` |
| S-5 | `state_recovery.py` | 中等 | `Xg_full` 恒为 `None` | 独立 |
| S-7 | `state_recovery.py` | 中等 | public API 未向上层暴露 recovery diagnostics | 独立 |
| R-2 | `remap.py` | 轻微 | 诊断未计算相对守恒误差 | 独立 |
| L-1 | `layout.py` | 轻微 | Phase B 界面块顺序不符 | 独立 |
| T-1 | `types.py` | 轻微 | `InterfaceState` 未携带预计算界面热力学参考态 | 与 `R-3` 相关 |
| P-1 | `preprocess.py` | 轻微 | 初始 `Y` 向量未校验归一化 | 独立 |
| CS-1 | `config_schema.py` | 轻微 | recovery schema 缺大量指导文件字段 | 先补 `S-6` |
| CS-2 | `config_schema.py` / `types.py` | 轻微 | recovery 字段命名与语义偏离指导文件合同 | 先补 `S-6` |
| CS-3 | `config_schema.py` / `types.py` | 轻微 | schema 允许 `liq_h_inv_tol/gas_h_inv_tol = 0.0`，但 `RecoveryConfig` 要求严格正值 | 独立 |

### 推荐修复顺序

1. **S-6 + CS-1 + CS-2**：先统一 recovery 配置合同、字段名与 schema
2. **CS-3**：消除 schema 与 dataclass 的容差下界冲突
3. **R-3**：给 remap 主路径补入 thermo / `P_inf` 或等价界面热力学参考态通道
4. **R-1**：把 newly exposed subvolume 补全从 bulk 端格切换到旧界面态
5. **S-7**：补 recovery diagnostics 的上行暴露通道
6. **S-5**：恢复并填充 `Xg_full`
7. **S-4 + S-3**：加前向焓一致性校验与 minor-fix
8. **S-2 + S-1**：补历史温度初值通道，并显式化 HPY pressure 来源
9. **R-2 + P-1 + L-1 + T-1**：完成轻微结构与诊断收尾
