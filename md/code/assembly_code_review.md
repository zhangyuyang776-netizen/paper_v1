# `src/assembly/` 代码审查报告

**审查日期**：2026-03-24
**审查范围**：`src/assembly/` 下全部 10 个模块
**对照文件**：
- `governing_equations_discretization_and_bc_guideline_final_v2.md`
- `paper_v1 Initialization and First-Step Guideline.md`
- `velocity_recovery_guideline_final.md`

---

## 一、模块清单

| 模块 | 职责 |
|------|------|
| `residual_liquid.py` | 液相守恒方程残差组装 |
| `residual_gas.py` | 气相守恒方程残差组装 |
| `residual_interface.py` | 界面条件 Eq.(2.15)–(2.19) 残差映射 |
| `residual_global.py` | 全局残差编排（单真相源流水线）|
| `jacobian_liquid.py` | 液相有限差分 Jacobian |
| `jacobian_gas.py` | 气相有限差分 Jacobian |
| `jacobian_interface.py` | 界面有限差分 Jacobian |
| `jacobian_global.py` | 全局 Jacobian 编排（共享缓存 + 合并校验）|
| `jacobian_pattern.py` | 全局 Jacobian 稀疏模式（CSR 构建）|
| `petsc_prealloc.py` | PETSc AIJ/MPIAIJ 预分配 |

---

## 二、残差组装审查

### 2.1 `residual_liquid.py`

#### ALE 时间项

时间项采用守恒形式：

```
time_term = (ρh_new · V - enthalpy_old_content) / dt
```

与后向欧拉 ALE FVM 规定一致。✅

#### 界面面能量通量分拆

```python
conv_rhoh[-1] = (E_l_s - q_l_s) * area_if   # → -mpp·h_s_l·A
q_face[-1]    = q_l_s                         # 扩散部分
```

总界面能量通量 `E_l_s = -mpp·h_s_l·A + q_l_s·A` 被正确分拆为对流部分（进 `conv_rhoh`）和扩散部分（进 `q_face`），符合界面面分裂规则（§5.1）。✅

#### 界面面组分通量分拆

```python
conv_rhoY_red[:, -1] = (N_l_red_if - J_l_red_if) * area_if   # → -mpp·Y_s_l·A
J_face_full[:, -1]   = J_l_full                               # 扩散部分
```

与液相 Fickian 扩散 + 守恒 ALE 形式一致。✅

**结论**：无偏差。

---

### 2.2 `residual_gas.py`

#### 界面面能量通量分拆

```python
conv_rhoh[0] = (E_g_s - q_g_s) * area_if   # → -mpp·h_s_g·A
```

与气相一侧对称，正确。✅

#### 速度一致性硬校验

```python
assert G_face_abs[0] == iface_face_pkg.G_g_if_abs
```

`build_velocity_recovery_package` 回传的 ALE 质量通量必须严格等于界面面包的 `G_g_if_abs`，防止速度场不一致传播至残差。✅

#### 远场边界

单侧梯度 + `ρh_inf` 上风插值，与 §11.3 规定一致。✅

**结论**：无偏差。

---

### 2.3 `residual_interface.py`

映射 Eq.(2.15)/(2.16)/(2.18)/(2.19) 至全局布局行：

| 行 | 对应方程 |
|----|---------|
| `Ts_row` | `E_g_s - E_l_s = 0`（能量连续性，Eq.(2.17)）|
| `mpp_row` | `mpp + ρ_l(u_l - ȧ)A = 0`（液侧 Eq.(2.18)，单 mpp 策略）|
| `Ys_l_rows` | 液相界面平衡，Eq.(2.15) |
| `Ys_g_rows` | 气相界面平衡，Eq.(2.16) |

气侧 Eq.(2.18) 仅作诊断输出，不进入残差向量，符合 single-mpp 规定。✅

模块对 `iface_face_pkg` 进行交叉验证（从包中重新推导期望值并与质量包对比），可在组装阶段提前捕获界面包不一致。✅

**结论**：无偏差。

---

### 2.4 `residual_global.py`

#### 调用顺序

```
build_bulk_props
  → build_interface_face_package          # 界面面真相源
    → build_velocity_recovery_package     # ALE 速度场
      → assemble_liquid_residual
      → assemble_interface_residual
      → assemble_gas_residual
```

#### 单真相源原则

全流水线共享唯一的 `iface_face_pkg`；所有块组装器均从该包取界面量，不重新计算界面梯度或通量。符合规定。✅

最终校验：无重复行，无越界索引。✅

**结论**：无偏差。

---

## 三、Jacobian 组装审查

### 3.1 `jacobian_pattern.py`

#### 液相行模式

`_build_liquid_row_pattern` 采用累积前缀策略：格子 `n` 的行包含列集合 `{0..n, n+1, iface（仅最后格子）, gas0（仅最后格子）}`。

这与 ALE 速度恢复的物理耦合一致：
- 液相面 ALE 速度 `G_{n+1}` 经连续性递推 `G_{n+1} = G_n + C_n - S_n` 从中心向外积分
- 因此格子 `n` 的残差通过 ALE 速度间接依赖格子 `0..n`

**模式物理正确，非保守过宽**。✅

#### 气相行模式

类似累积前缀：气相格子 `n` 依赖 `{iface, 0..n, n+1}`，气相格子 0 额外包含最后一个液相格子。✅

---

#### [JP-1] 轻微：界面行模式耦合所有液相格子

**位置**：`jacobian_pattern.py:142–143`

```python
def _build_interface_row_pattern(...):
    ...
    for liq_cell in range(mesh.n_liq):   # ← 耦合所有液相格子
        cols.update(_cell_liq_cols(layout, liq_cell))
```

**物理分析**：界面残差 Eq.(2.15)–(2.19) 仅依赖：
- 最后一个液相格子（`Tl_last`，`Yl_last`，用于界面侧单侧梯度）
- 所有界面未知量（`Ts`，`Ys_l`，`Ys_g`，`mpp`）
- 第一个气相格子（`Tg_first`，`Yg_first`）

界面方程不通过任何路径直接耦合液相格子 `0..n_liq-2`（ALE 速度恢复中 `G_g_if_abs` 仅依赖界面未知量）。

**后果**：
1. 界面 Jacobian 的候选列集包含 `(n_liq - 1) × (1 + n_liq_red)` 个无效列，每列需额外一次残差求值。
2. Jacobian 矩阵中产生多余零值条目。
3. PETSc 矩阵预分配量略大于必要值，随 `n_liq` 线性增长。

**无正确性风险**；纯粹为性能问题。

---

### 3.2 `jacobian_liquid.py`

步长选择按变量类型（温度/组分/界面变量/mpp）分别施加绝对步长下界，逻辑与气相、界面 Jacobian 一致。✅

---

#### [JL-1] 中等：中心差分模式下缺少符号回退保护

**位置**：`jacobian_liquid.py:222`

```python
# jacobian_liquid.py — _safe_perturb_and_evaluate 内部
for sign in (preferred_sign, -preferred_sign):   # 始终允许回退
    ...
```

对比 `jacobian_gas.py` 和 `jacobian_interface.py`：

```python
# jacobian_gas.py / jacobian_interface.py
signs = (preferred_sign, -preferred_sign) if allow_sign_fallback else (preferred_sign,)
```

后两者在 `use_central=True` 时均传入 `allow_sign_fallback=False`，强制加扰和减扰各自只尝试指定方向。`jacobian_liquid.py` 缺少此参数，在 `use_central=True` 时两次调用均允许符号回退。

**错误路径**：若加扰调用因 `+h` 失败而回退至 `-h'`（`h'` 为收缩后步长），而减扰调用以原步长 `-h` 成功，则：

```
denom = plus_h - minus_h = (-h') - (-h) = h - h' ≠ 2h
```

实际使用的公式为 `(R(u - h') - R(u - h)) / (h - h')`，既非前向差分也非标准中心差分，截断误差阶次不确定。`denom == 0` 防护仅捕获 `h' = h` 的退化情形，无法检测 `h' ≠ h` 的不对称情形。

**缓解因素**：
- 默认 `use_central=False`（前向差分），中心差分路径不在常规操作中触发。
- 步长收缩本身为低概率路径。

**建议**：在 `jacobian_liquid.py` 的 `_safe_perturb_and_evaluate` 中增加 `allow_sign_fallback: bool` 参数，并在 `use_central=True` 时对两侧均传入 `allow_sign_fallback=False`，与其他两个块 Jacobian 保持一致。

---

### 3.3 `jacobian_gas.py`

- 中心差分：`allow_sign_fallback=False`，正确禁止符号回退。✅
- 前向差分：`allow_sign_fallback=True`，合理。✅
- 禁止液相其他列（除最后液相格子外）出现在气相 Jacobian 中。✅

### 3.4 `jacobian_interface.py`

- 中心差分：加扰和减扰均传入 `allow_sign_fallback=False`。✅
- 列分类覆盖液相格子、所有界面变量、气相格子 0；禁止 `gas_other` 列。✅

### 3.5 `jacobian_global.py`

#### 共享残差缓存

```python
_make_shared_block_residual_evaluator(..., cache_size=32)
```

三块 Jacobian（液/界面/气）共享一个 LRU 缓存，避免同一扰动向量的重复残差求值，对每列只调用一次 `build_all_residual_blocks_from_layout_vector`。✅

#### 不重复三元组

```python
_coalesce_or_raise_duplicate_triplets(...)   # 遇重复即抛出，不静默合并
```

块边界处若出现重复 `(row, col)` 条目，立即以异常暴露，防止 Jacobian 条目被静默覆盖。✅

#### 最终模式校验

合并后所有三元组均针对 `JacobianPattern` 校验，且输出行集合必须与预期拥有行集合完全相等。✅

### 3.6 `petsc_prealloc.py`

- `d_nz` / `o_nz` 直接从冻结的 `JacobianPattern` 派生，无运行时稀疏性推断。✅
- `new_nonzero_allocation_err=True` 硬编码，防止模式外动态分配。✅
- `_extract_local_pattern_rows` 强制确保每行包含对角条目，防止 PETSc 因缺少对角元出错。✅

---

## 四、综合评估

### 物理与数值方法符合性

| 模块 | 物理方法 | 数值方案 | 符合状态 |
|------|---------|---------|---------|
| `residual_liquid` | ALE FVM + Fickian 液相扩散 + 守恒时间项 | 后向欧拉，上风对流 | ✅ |
| `residual_gas` | ALE FVM + 混合平均气相扩散 + 校正速度 | 后向欧拉，上风对流 | ✅ |
| `residual_interface` | Eq.(2.15)–(2.19)，单 mpp 策略，液侧有效行 | 精确映射 | ✅ |
| `residual_global` | 单真相源，速度恢复一致性，无重复行 | 正确调用链 | ✅ |
| `jacobian_liquid` | 模式限定 FD，按变量类型步长 | 前向差分（默认）| ✅（见 JL-1）|
| `jacobian_gas` | 模式限定 FD，符号回退保护 | 前向/中心差分 | ✅ |
| `jacobian_interface` | 模式限定 FD，符号回退保护 | 前向/中心差分 | ✅ |
| `jacobian_global` | 三块组装 + 共享缓存 + 合并校验 | — | ✅ |
| `jacobian_pattern` | ALE 累积耦合结构，物理正确 | CSR 构建 | ✅（见 JP-1）|
| `petsc_prealloc` | 冻结模式派生，无运行时推断 | d_nz/o_nz | ✅ |

### 发现汇总

| 编号 | 严重性 | 模块 | 行号 | 描述 |
|------|--------|------|------|------|
| JL-1 | **中等** | `jacobian_liquid.py` | 222 | 中心差分模式下缺少 `allow_sign_fallback=False` 保护；加扰/减扰步长不对称时差分公式误差阶次不确定；默认前向差分模式不受影响 |
| JP-1 | **轻微** | `jacobian_pattern.py` | 142–143 | 界面行模式包含所有液相格子，而物理上仅需最后一个液相格子；导致界面 Jacobian 多余列求值与 PETSc 多余预分配，随 `n_liq` 线性增长 |

**未发现物理方程偏差、数值方案简化或正确性 bug。**
