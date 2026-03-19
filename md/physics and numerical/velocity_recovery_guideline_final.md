# velocity_recovery_guideline_final

## 1. 文件目的

本文档用于为新代码项目 `paper_v1` 规定一个明确、可执行、可审查的速度恢复方法，并将其与外层 `predictor-corrector moving mesh` 组织方式统一起来，形成**最终定稿版**的方法约束。

本文档解决的核心问题是：

1. 速度是否为主未知量  
2. 速度如何由离散连续方程恢复  
3. 速度恢复与 `predictor-corrector` 是否属于同一套逻辑  
4. moving mesh、界面质量通量、连续方程、对流项是否在同一时间层和同一几何层上自洽  
5. 若采用“外层几何更新 + remap + 内层固定网格 transport solve”的工程路线，存储项中的 old mass 应如何定义

---

## 2. 总结论

新项目中应采用以下原则：

- `u_l`、`u_g` **不是主未知量**
- 速度场 **不是** 由单条 Stefan 关系直接构造
- 速度必须由**离散连续方程**在每个 trial state 下恢复
- 对流项使用的不是 `u`，而是相对控制面速度 `u - v_c`
- 速度恢复不是 `predictor-corrector` 外部的附加模块，而是其**内层 transport residual 的组成部分**
- 若采用“两层结构”：
  - 外层：几何更新、moving mesh、remap、accept/reject
  - 内层：固定当前网格上的 transport nonlinear solve
- 那么速度恢复必须发生在**内层 residual 评估阶段**
- 所有 continuity 存储项中的 old mass，统一定义为  
  **`old_mass_on_current_geometry`**

---

## 3. 方法适用范围

本文档适用于以下新项目路线：

- 1D 球对称液滴问题
- moving mesh / moving control volume
- 外层 `predictor-corrector` 更新界面位置和网格
- 内层在固定当前几何上求解 transport 非线性系统
- `Rd` 不进入内层主未知量
- 界面显式未知量首版至少包含 `Ts`、`Y_s,g,red`、`mpp`
- 速度由连续方程恢复，而非由动量方程直接积分

本文档不适用于以下旧思路：

- 直接假定全域满足 `ρ u r^2 = const`
- 用 Stefan 关系经验外推全域 bulk 速度场
- 将液相速度简单置零后继续做一般性 transport 对流
- 在 residual 外部后处理一个速度场，再反过来喂给 residual

---

## 4. 与整体算法的关系

### 4.1 核心结论

速度恢复逻辑与 `predictor-corrector` 是**同一套组织结构中的内外层分工**，不是两套彼此平行的方法。

- 外层负责：
  - 界面位置更新
  - 网格更新
  - remap / projection
  - 接受/拒绝当前时间步或当前外层迭代
- 内层负责：
  - 在**当前固定几何**上求解 transport 非线性系统
  - 在 residual 评估时恢复速度
  - 用恢复出的速度组装对流项和界面相关量

因此：

> 速度恢复不是时间步结束后的后处理，而是内层 transport solve 的内核组成部分。

---

## 5. 变量与职责边界

### 5.1 不进入内层主未知量的量

以下变量**不进入**内层 transport 主未知量：

- `Rd = a(t)`
- `u_l`
- `u_g`

### 5.2 内层 transport 主未知量

首版推荐：

- 液相 bulk：`T_l`，必要时 `Y_l`
- 界面：`T_s`, `mpp`
- 气相 bulk：`T_g`, `Y_g`

### 5.3 外层几何状态

外层持有并更新：

- `a = Rd`
- 当前网格节点位置
- 面位置 `r_f`
- 面面积 `A_f`
- 单元体积 `V_n`
- 控制面速度 `v_c`

### 5.4 派生量

以下量在 residual 评估或后续步骤中由当前 trial state 派生得到：

- `ρ_l`, `ρ_g`
- `u_l`, `u_g`
- `h_l`, `h_g`
- `J_i`
- `q`
- closure species
- 界面侧局部密度与速度

---

## 6. moving control volume 下的对流形式

在 moving control volume 框架中，守恒方程离散为：

\[
\frac{d}{dt}(\Phi_{\beta,n}V_n)
+
[\phi_\beta (u_\beta-v_c)A]_{n-}^{n+}
=
-[F(\phi_\beta)A]_{n-}^{n+}
+
P(\Phi_{\beta,n})V_n
\]

其中：

- `β` 表示相别（液相或气相）
- `n` 表示控制体编号
- `V_n` 是控制体体积
- `A` 是面面积
- `u_β` 是流体绝对速度
- `v_c` 是控制面速度

因此，对流项真正使用的是：

\[
u^{rel}=u-v_c
\]

而不是单独的 `u`。

---

## 7. 速度恢复的数学定义

### 7.1 连续方程离散后的恢复主式

连续方程离散后，可整理为：

\[
(\rho_\beta u_\beta A)_{n+}^{n+1}
=
(\rho_\beta u_\beta A)_{n-}^{n+1}
+
[(\rho_\beta v_c A)^{n+1}]_{n-}^{n+}
-
\frac{(\rho_{\beta,n}V_n)^{n+1}-(\rho_{\beta,n}V_n)^n}{\Delta t}
\]

定义绝对质量流率：

\[
G_{\beta,f} \equiv (\rho_\beta u_\beta A)_f
\]

则可写成递推形式：

\[
G_{n+}=G_{n-}+C_n-S_n
\]

其中

\[
C_n=(\rho v_cA)_{n+}-(\rho v_cA)_{n-}
\]

\[
S_n=\frac{(\rho_nV_n)^{n+1}-(\rho_nV_n)^n}{\Delta t}
\]

### 7.2 物理解释

- `S_n` 是控制体质量储存项变化率
- `C_n` 是控制面运动导致的 swept-mass 项
- `G_{n+}` 是右侧面的绝对质量流率
- 递推完成后，再由 `G_f` 反算 `u_f`

### 7.3 速度恢复的两步结构

#### 第一步：恢复面质量流率

\[
G_f=(\rho u A)_f
\]

#### 第二步：由面质量流率恢复面速度

\[
u_f=\frac{G_f}{\rho_fA_f}
\]

然后得到相对控制面的速度：

\[
u_f^{rel}=u_f-v_{c,f}
\]

最终：

- `u_f` 用于界面与诊断
- `u_f^{rel}` 用于对流通量和上风判断

---

## 8. 液相恢复逻辑

### 8.1 起点边界

液滴中心满足球对称条件：

\[
u_l(r=0)=0
\]

因此液相最左边界面有：

\[
G_{l,1-}=0
\]

### 8.2 扫描方向

液相按**从中心到界面**的方向递推：

\[
G_{l,n+}=G_{l,n-}+C_{l,n}-S_{l,n},
\quad n=1,2,\dots,N_l
\]

### 8.3 界面液相速度

最终得到液相界面内侧面的速度：

\[
u_{l,if}=\frac{G_{l,if}}{\rho_{l,if}A_{if}}
\]

该量参与界面位置更新。

---

## 9. 气相恢复逻辑

### 9.1 起点边界

气相恢复从**界面面**开始。

界面质量通量关系为：

\[
\dot m'' = -\rho_l(u_l-\dot a)_{r=a}
          = -\rho_g(u_g-\dot a)_{r=a}
\]

在 `paper_v1` 中，gas-side Eq.(2.18) 不作为第二条 `mpp` 主残差行，而是通过界面边界质量流率被强施加。
因此气相界面速度满足：

\[
u_{g,if}=\dot a-\frac{\dot m''}{\rho_{g,if}}
\]

于是界面面绝对质量流率定义为：

\[
G_{g,if}^{abs}
=
\rho_{g,if}u_{g,if}A_{if}
=
(\rho_{g,if}\dot a-\dot m'')A_{if}
\]

实现上应理解为：

- liquid-side Eq.(2.18) 产生 `mpp` 的主残差
- gas-side Eq.(2.18) 通过 `G_{g,if}^{abs}` 的边界构造被强施加
- 气相 continuity sweep 以该界面质量流率为起点向远场递推

### 9.2 扫描方向

气相按**从界面到远场**的方向递推：

\[
G_{g,n+}=G_{g,n-}+C_{g,n}-S_{g,n},
\quad n=1,2,\dots,N_g
\]

其中第一个气相单元的左面就是界面面。

---

## 10. 面密度的定义规则

连续方程恢复中需要面密度 `ρ_f`。为避免不同模块定义不一致，必须统一。

### 10.1 内部面

推荐首版采用中心平均：

\[
\rho_f = \frac{\rho_L+\rho_R}{2}
\]

后续若升级，也只能在**全系统统一替换**。

### 10.2 界面相邻面

- 液相界面内侧面：取液相最外单元密度
- 气相界面外侧面：取气相最内单元密度

### 10.3 一致性要求

面密度规则必须在以下位置保持完全一致：

- residual 评估
- Jacobian 评估（含 FD Jacobian）
- remap 后重建
- 结果输出与诊断

---

## 11. 存储项中的旧质量定义

这是最终版本必须明确写死的地方。

连续方程递推中含有：

\[
(\rho_nV_n)^n
\]

若项目采用的是：

- 外层几何更新
- remap / projection
- 内层在当前固定网格上做 transport solve

那么统一采用：

## `old_mass_on_current_geometry`

即：

- 将上一时刻的保守量先通过 remap / projection 映射到**当前几何对应的网格**
- 在当前几何上形成“old-equivalent conservative mass”
- 再与当前 trial state 的 `(ρV)^{n+1}` 做差

### 11.1 采用该定义的原因

1. 内层 transport residual 的几何是**当前固定网格**
2. 若 old mass 仍保留在旧几何上，则存储项会混入：
   - 几何变化
   - remap 误差
   - 时间变化
3. 使用 `old_mass_on_current_geometry` 后，存储项表示的是：
   - 在当前几何上
   - 从 remapped old state 到 current trial state
   - 的质量变化

### 11.2 方法约束

新项目中应明确规定：

- 若内层 residual 以“当前几何上的固定网格问题”定义，
- 则所有时间离散存储项中的 old mass，
- 一律指 **remap 到当前几何后的 old-equivalent conservative mass**，
- 而不是原始旧几何上的 cell mass。

---

## 12. 速度恢复与 predictor-corrector 的自洽组织方式

这一节是最终定稿新增的核心部分。

### 12.1 自洽性的定义

速度恢复逻辑要与 `predictor-corrector` 自洽，必须满足：

- 使用同一个**时间层**
- 使用同一个**几何层**
- 使用同一个**网格速度**
- 使用同一个**界面状态**
- 使用同一个**通量定义**

也就是说，速度恢复、界面更新、对流项、moving mesh 不能各用一套变量来源。

### 12.2 外层第 \(k\) 次几何迭代

给定当前几何试探：

\[
a^{n+1,(k)},\quad \text{grid}^{(k)},\quad v_c^{(k)}
\]

其中包括：

- 当前界面位置
- 当前网格节点位置
- 当前面面积
- 当前单元体积
- 当前控制面速度

### 12.3 第一步：将旧状态映射到当前几何

构造：

- `old_state_on_current_geometry`
- `old_mass_on_current_geometry`

这是为了让内层 BE 存储项与当前几何完全一致。

### 12.4 第二步：在固定当前几何上做内层 nonlinear transport solve

内层 unknown 为 `U_transport`，不含 `Rd`。

在任意 residual 评估时：

1. 从当前 trial state 读取 bulk/interface unknowns
2. 计算物性：
   - `ρ`
   - `h`
   - `k`
   - `D`
3. 使用当前几何的：
   - `V_n`
   - `A_f`
   - `v_c`
4. 使用 `old_mass_on_current_geometry`
5. 对液相做 continuity sweep，恢复 `G_l, u_l`
6. 对气相做 continuity sweep，恢复 `G_g, u_g`
7. 构造相对速度：
   - `u_l_rel = u_l - v_c`
   - `u_g_rel = u_g - v_c`
8. 用相对速度组装：
   - 对流质量通量
   - 物种上风通量
   - 能量对流通量
9. 用同一 trial state 的 `Ts, mpp, ρ_if, u_if` 组装界面残差

### 12.5 第三步：由当前内层解结果进行几何校正

当内层 transport solve 收敛后，取该轮内层解得到的：

- `mpp^(k)`
- `u_l,if^(k)`
- `rho_l,if^(k)`

更新界面速度：

\[
\dot a^{(k)}=\frac{\dot m''^{(k)}}{\rho_{l,if}^{(k)}}+u_{l,if}^{(k)}
\]

然后更新几何：

\[
a^{n+1,(k+1)}
\]

### 12.6 第四步：重建网格并进入下一轮外层迭代

由新的 `a_dot` 与新的 `a`：

- 重建网格
- 重建 `V_n`
- 重建 `A_f`
- 重建 `v_c`
- 将当前状态 remap 到新几何
- 进入下一轮外层迭代

### 12.7 收敛判据

外层 predictor-corrector 以界面速度收敛为准，例如：

- 相邻两次界面速度差的归一化量小于给定阈值

只有当外层几何收敛后，本时间步才算真正完成。

---

## 13. 自洽性的 5 条硬约束

这是新项目必须遵守的强约束。

### 13.1 几何层一致

在任意一个 outer iteration \(k\) 的 inner solve 中：

- `grid^(k)`
- `V^(k)`
- `A^(k)`
- `v_c^(k)`

必须全部冻结为同一套当前几何。

**禁止**在同一 inner solve 中混用不同 outer iteration 的几何量。

### 13.2 时间层一致

continuity 存储项中的比较必须是：

- 当前几何上的 `old_mass_on_current_geometry`
- 当前 trial state 下的 `current_mass_on_current_geometry`

**禁止**将旧几何上的原始 cell mass 直接混入当前几何 residual。

### 13.3 网格速度一致

同一 outer iteration 内：

- continuity recovery 使用的 `v_c`
- 对流项使用的 `u-v_c`
- moving mesh 使用的 `v_c`

必须来自**同一套 `a_dot^(k)`**

**禁止** continuity 用 `vc(pred)`，对流项又改用 `vc(corr)`。

### 13.4 界面状态一致

在用

\[
\dot a=\frac{\dot m''}{\rho_l}+u_{l,if}
\]

更新界面时：

- `mpp`
- `rho_l,if`
- `u_l,if`

必须来自**同一次内层解**和**同一 outer iteration**

**禁止**混用上一次 sweep、旧时间层、或其它外层轮次的界面量。

### 13.5 速度恢复来源唯一

全域 bulk 速度只能由**离散连续方程**恢复。

**禁止**在主框架中重新引入：

- `ρur^2 = const` 型经验速度场
- Stefan 流经验外推 bulk velocity
- residual 外部拼装速度场后回灌 residual

---

## 14. 推荐实现顺序

每次 residual 评估时，应执行以下流程：

1. 从当前 trial state 读取 bulk/interface unknowns
2. 计算物性：
   - `ρ`
   - `h`
   - `k`
   - `D`
3. 读取当前外层几何：
   - 面位置 `r_f`
   - 面面积 `A_f`
   - 单元体积 `V_n`
   - 控制面速度 `v_c`
4. 构造 `old_mass_on_current_geometry`
5. 对液相做 continuity sweep，恢复 `G_l, u_l`
6. 用同一个 `mpp` 与冻结 `dot_a^(k)` 构造 `G_{g,if}^{abs}`
7. 对气相做 continuity sweep，恢复 `G_g, u_g`
8. 构造相对速度：
    - `u_l_rel = u_l - v_c`
    - `u_g_rel = u_g - v_c`
9. 用相对速度做：
    - 对流质量通量
    - 物种上风方向判断
    - 能量对流通量
10. 用同一 trial state 的界面量组装界面残差
11. inner solve 收敛后，用界面量更新 `a_dot`
12. 外层更新网格并 remap
13. 检查外层几何收敛

---

## 15. 伪代码模板

```python
def reconstruct_velocity_from_continuity(
    phase,
    state_trial,
    state_old_current_geom,
    props,
    geom,
    dt,
):
    # geometry on current outer iteration
    r_f = geom.r_f
    A_f = 4.0 * pi * r_f**2
    V_n = geom.V_n
    vc_f = geom.vc_f

    # cell densities from current trial state
    rho_n = props.rho_cell[phase]

    # face densities from fixed interpolation rule
    rho_f = interp_face_density(rho_n, phase)

    # initialize face mass flux
    G = zeros_like(A_f)

    if phase == "liquid":
        # center boundary: u=0
        G[left_center_face] = 0.0
        sweep_cells = liquid_cells_center_to_interface

    elif phase == "gas":
        rho_if = rho_f[interface_face]
        G[interface_face] = (rho_if * geom.a_dot - state_trial.mpp) * A_f[interface_face]
        sweep_cells = gas_cells_interface_to_farfield

    for n in sweep_cells:
        fL = left_face(n)
        fR = right_face(n)

        Cn = rho_f[fR] * vc_f[fR] * A_f[fR] - rho_f[fL] * vc_f[fL] * A_f[fL]
        Sn = (
            rho_n[n] * V_n[n]
            - state_old_current_geom.mass_cell[phase][n]
        ) / dt

        G[fR] = G[fL] + Cn - Sn

    u_face = G / (rho_f * A_f)
    u_rel = u_face - vc_f

    return G, u_face, u_rel
	

## 16. 最终方法合同

### Velocity Reconstruction Contract

1. `u_l`、`u_g` are not primary unknowns.
2. Face velocities must be reconstructed from the discretized continuity equation.
3. Convective fluxes must use relative velocity `u - v_c`.
4. Liquid-phase sweep starts from center boundary `u=0`.
5. Gas-phase sweep starts from the interface mass-flux condition strongly imposed from the same `mpp` and frozen `dot_a^(k)`.
6. Storage term old mass must be defined as `old_mass_on_current_geometry`.
7. The same face-density rule must be used in residual, Jacobian and diagnostics.
8. Velocity reconstruction is part of the inner transport solve, not an external postprocess.
9. Geometry, storage term, interface state and control-surface velocity must remain layer-consistent within each outer iteration.
10. Bulk velocity field must not be replaced by Stefan-type empirical field reconstruction.

---

## 17. 最终结论

速度恢复的核心链条应固定为：

[
(\Phi,\rho,V,v_c)
\longrightarrow
G_f=(\rho uA)*f
\longrightarrow
u_f
\longrightarrow
u_f-v*{c,f}
]

并嵌入以下统一组织结构中：

[
\text{outer geometry predictor-corrector}
;\Longrightarrow;
\text{inner fixed-geometry transport solve}
;\Longrightarrow;
\text{continuity-based velocity recovery}
;\Longrightarrow;
\text{interface correction and mesh update}
]

而不是：

[
\dot m'' \longrightarrow \text{Stefan guess} \longrightarrow u(r)
]

前者是新代码项目 `paper_v1` 中应采用的正式方法路线；后者只能视为经验近似，不得进入主残差框架。
