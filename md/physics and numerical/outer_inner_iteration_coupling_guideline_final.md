# outer_inner_iteration_coupling_guideline_final

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定**两层迭代循环**的实现方法，明确：

1. outer loop 负责什么
2. inner loop 负责什么
3. predictor 和 corrector 各在什么层执行
4. inner 和 outer 的数据交换关系是什么
5. 一个时间步从开始到接受的完整计算流程是什么
6. 哪些量属于时间层变量，哪些量属于 outer iterate，哪些量属于 inner trial state
7. 什么情况下回退、拒绝时间步或终止计算

本文档是 `paper_v1` 的**时间推进 + outer/inner 耦合**正式定稿版本。

---

## 2. 总体原则

### 2.1 两层结构的正式定义

`paper_v1` 采用两层结构：

### 外层 outer loop
负责：
- 界面位置 `a/Rd`
- 界面速度 `dot_a`
- 网格构建
- 控制面速度 `v_c`
- remap / projection
- outer predictor-corrector
- 时间步接受/拒绝

### 内层 inner loop
负责：
- 在固定当前几何上求解液相 bulk、界面块、气相 bulk 的耦合非线性系统
- 组装所有 residual
- 速度恢复
- 物性计算
- 界面通量与界面方程求解

### 2.2 最重要的边界

**内层不更新几何，外层不直接解物理状态。**

也就是说：

- inner loop 只在当前冻结的几何与网格上求 `U`
- outer loop 只根据 inner 收敛后的物理状态去更新 `a` 和网格

这条边界不能被任何“为了方便”的实现打破。

---

## 3. 三类变量的层级归属

为了避免 outer/inner 混乱，必须先分清三类变量。

## 3.1 已接受时间层变量

这些量属于上一个已接受时间层 `t^n`：

- `U^n`
- `a^n`
- `dot_a^n`
- `G^n`（已接受网格）
- `v_c^n`（已接受控制面速度）

它们是整个时间步推进的**起点**。

---

## 3.2 outer iterate 变量

这些量属于当前时间步 `t^n -> t^{n+1}` 内，第 `k` 轮 outer iteration 的几何猜测：

- `a^(k)`
- `dot_a^(k)`
- `G^(k)`（当前猜测几何下的网格）
- `v_c^(k)`（当前猜测几何下的控制面速度）
- `old_state_on_current_geometry^(k)`

这些量在 inner solve 开始前固定，在 inner solve 结束前**不得改变**。

---

## 3.3 inner trial state 变量

这些量属于当前 outer iteration 下，inner nonlinear solver 中的 trial / iterate state：

- `U_trial`
- `rho_trial`
- `Y_trial`
- `h_trial`
- `T_trial`
- `u_trial`
- `mpp_trial`
- `Ts_trial`
- `Ys_trial`

这些量在 Newton / SNES 迭代中不断更新，但**不允许**反过来修改：
- `a^(k)`
- `dot_a^(k)`
- `G^(k)`
- `v_c^(k)`

---

## 4. 时间推进的数学对象

## 4.1 目标不是推进 outer 迭代，而是推进时间层

在时间步 `t^n -> t^{n+1}` 中，真正要求的是：

\[
U^{n+1}, \qquad a^{n+1}, \qquad \dot a^{n+1}
\]

outer iteration 的作用不是“继续推进时间”，而是在同一个时间步里，反复逼近这一组新时间层量。

---

## 4.2 `a` 的时间离散主线

界面位置采用 trapezoidal corrector 主线：

\[
a^{n+1}
=
a^n + \frac{\Delta t}{2}\left(\dot a^n + \dot a^{n+1}\right)
\]

这里：
- `a^n`、`dot_a^n` 是旧时间层已知量
- `dot_a^{n+1}` 由当前新时间层的物理状态决定

由于 `dot_a^{n+1}` 不能直接先验得到，因此 outer iteration 的本质是：

> 在同一个时间步内，通过反复 inner solve，寻找与该隐式更新公式自洽的 `a^{n+1}` 和 `dot_a^{n+1}`。

---

## 5. Predictor 与 Corrector 的正式定义

## 5.1 Predictor

predictor 只在每个时间步开始时做**一次**。

推荐使用显式 Euler predictor：

\[
\dot a^{(0)} = \dot a^n
\]

\[
a^{(0)} = a^n + \Delta t \,\dot a^n
\]

### 作用
- 给 new time level 的几何一个初始猜测
- 为第 0 轮 outer iteration 提供起点

### 正式规定
- predictor 只做一次
- 不在每轮 outer iteration 里重复做 predictor

---

## 5.2 Corrector

corrector 在**每一轮 outer iteration**后执行一次。

当第 `k` 轮 inner solve 收敛后，得到当前几何猜测下的物理解：

- `mpp^(k)`
- `u_l_if^(k)`
- `rho_s,l^(k)`
- 完整状态 `U^(k)`

然后先计算基于物理解的界面速度：

\[
\dot a_{phys}^{(k)}
=
u_{l,if}^{(k)} + \frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
\]

再执行 corrector：

\[
a_{target}^{(k+1)}
=
a^n + \frac{\Delta t}{2}\left(\dot a^n + \dot a_{phys}^{(k)}\right)
\]

### 说明
注意这里用的是：
- `a^n`
- `dot_a^n`

而不是：
- `a^(k)`
- `dot_a^(k)`

### 原因
因为 outer iteration 是在求同一个隐式时间推进方程的 fixed point，不是每轮重新推进一次时间。

---

## 5.3 欠松弛 corrector（推荐）

为了改善 outer 收敛，推荐采用欠松弛：

\[
a^{(k+1)}
=
(1-\omega_a)a^{(k)} + \omega_a a_{target}^{(k+1)}
\]

\[
\dot a^{(k+1)}
=
(1-\omega_v)\dot a^{(k)} + \omega_v \dot a_{phys}^{(k)}
\]

其中：

- `0 < omega_a <= 1`
- `0 < omega_v <= 1`

### 首版推荐
可先取：

\[
\omega_a = 1,\qquad \omega_v = 1
\]

若 outer 振荡，再降低。

---

## 6. 外层循环的职责与公式

## 6.1 outer loop 的输入

在时间步开始时，outer loop 的输入为：

- `U^n`
- `a^n`
- `dot_a^n`
- `G^n`
- `dt`

---

## 6.2 outer loop 的输出

outer loop 收敛并通过接受检查后，输出：

- `U^{n+1}`
- `a^{n+1}`
- `dot_a^{n+1}`
- `G^{n+1}`
- `v_c^{n+1}`

---

## 6.3 outer loop 每轮做什么

对第 `k` 轮 outer iteration：

### Step O1. 构建当前几何
已知：
- `a^(k)`
- `dot_a^(k)`

构建当前三段网格 `G^(k)`：

- 第一段液相网格
- 第二段近界面气相网格
- 第三段固定远场网格

并计算当前控制面速度 `v_c^(k)`。

### Step O2. remap old state 到当前几何
将已接受旧状态 `U^n` 映射到当前几何，得到：

\[
old\_state\_on\_current\_geometry^{(k)}
\]

### Step O3. 调用 inner solve
在冻结的：
- `a^(k)`
- `dot_a^(k)`
- `G^(k)`
- `v_c^(k)`
- `old_state_on_current_geometry^(k)`

上，求解当前 fixed-geometry transport system。

### Step O4. 根据 inner 收敛结果计算 `dot_a_phys^(k)`
用：

\[
\dot a_{phys}^{(k)}
=
u_{l,if}^{(k)} + \frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
\]

### Step O5. corrector 更新几何猜测
按第 5 节公式计算：

- `a^(k+1)`
- `dot_a^(k+1)`

### Step O6. 判断 outer 收敛
若满足收敛条件，则跳出 outer loop。
否则进入下一轮。

---

## 7. 内层循环的职责与公式

## 7.1 inner loop 的输入

inner loop 的输入必须是**冻结后的当前几何问题**：

- `a^(k)`
- `dot_a^(k)`
- `G^(k)`
- `v_c^(k)`
- `old_state_on_current_geometry^(k)`

以及当前轮的初始猜测 `U_init^(k)`。

### `U_init^(k)` 推荐来源
- 第 0 轮 outer：由 remapped old state 初始化
- 第 `k>0` 轮 outer：由上一轮 inner 收敛状态 `U^(k-1)` 初始化

---

## 7.2 inner loop 的未知量

inner loop 求解的是：

\[
U = [\Phi_l,\phi_s,\Phi_g]
\]

具体包括：

### 液相 bulk
- `T_l`
- `Y_l,red`（多组分阶段）

### 界面块
- `T_s`
- `Y_s,g,red`
- `mpp`
- 多组分阶段还包括 `Y_s,l,red`

### 气相 bulk
- `T_g`
- `Y_g,red`

---

## 7.3 inner loop 中冻结不变的量

inner Newton / SNES 迭代中，下列量绝对不允许修改：

- `a^(k)`
- `dot_a^(k)`
- `G^(k)`
- `v_c^(k)`
- `old_state_on_current_geometry^(k)`

这些量定义了“当前 fixed-geometry nonlinear problem”。

---

## 7.4 inner residual 的构成

inner residual 由三块组成：

### 液相 bulk residual
- 能量方程 residual
- 组分方程 residual（多组分阶段）
- 连续方程仅用于速度恢复，不作为独立主 unknown residual

### 界面 residual
- Eq.(2.15)
- Eq.(2.16)
- Eq.(2.17)
- Eq.(2.18)
- Eq.(2.19)

其中 `mpp` 的实现规则必须固定为：

- liquid-side Eq.(2.18) 作为主残差
- gas-side Eq.(2.18) 通过界面边界质量流率被强施加
- 不允许为同一个 `mpp` 在 inner system 中再增加第二条独立 residual row

### 气相 bulk residual
- 能量方程 residual
- 组分方程 residual
- 连续方程仅用于速度恢复，不作为独立主 unknown residual

---

## 7.5 inner loop 中 `dot_a` 的使用方式

这是必须写死的地方。

若界面 `mpp` 残差使用 Eq.(2.18)，则其中的 `dot_a` 必须取：

\[
\dot a = \dot a^{(k)}
\]

也就是当前 outer iteration 的冻结值。

同时，gas bulk 的界面起点必须由同一个 `mpp` 构造：

\[
G_{g,if}^{abs}
=
(\rho_{s,g}\dot a^{(k)}-mpp)A_{if}
\]

### 明确禁止
在 inner Newton 过程中，禁止：

1. 用当前 trial `mpp`
2. 现算新的 `dot_a_trial`
3. 再把 `dot_a_trial` 塞回当前 residual

否则 inner residual 就不再是 fixed-geometry residual。

---

## 7.6 inner 收敛输出

inner loop 收敛后，输出：

- `U^(k)`
- `mpp^(k)`
- `u_l_if^(k)`
- `rho_s,l^(k)`
- `Ts^(k)`
- `Ys^(k)`
- inner diagnostics

供 outer corrector 使用。

---

## 8. inner 与 outer 的关系

## 8.1 outer 调 inner
outer 负责构造“当前几何问题”，inner 负责解这个问题。

### outer 提供给 inner
- 几何
- 网格
- 控制面速度
- remapped old state

### inner 提供给 outer
- 收敛后的物理状态
- 新的物理界面速度 `dot_a_phys`
- 诊断信息

---

## 8.2 inner 不能直接改 outer 变量
inner loop 的职责止于“在当前几何上求收敛状态”，不允许：

- 直接改 `a^(k)`
- 直接改 `dot_a^(k)`
- 直接改网格
- 直接做 remap

---

## 8.3 outer 不能跳过 inner 自己更新几何
outer 的几何更新必须基于 inner 收敛输出，不能在没有 inner 收敛的情况下自行推进 `a`。

---

## 9. outer 收敛判据

推荐使用界面速度收敛作为主判据：

\[
\varepsilon_{\dot a}^{(k)}
=
\frac{
\left|\dot a_{phys}^{(k)}-\dot a^{(k)}\right|
}{
\max\left(|\dot a_{phys}^{(k)}|,\epsilon_{\dot a}\right)
}
\]

其中推荐：

\[
\epsilon_{\dot a}=10^{-12}
\]

收敛条件：

\[
\varepsilon_{\dot a}^{(k)} < 10^{-5}
\]

正式规定：

- `paper_v1` 的 outer 收敛判据只使用界面速度一致性误差 `\varepsilon_{\dot a}^{(k)}`
- 不再额外引入位置收敛判据 `\varepsilon_a`
- `a^{(k+1)}` 仍然参与 corrector 更新与 diagnostics 输出，但不作为 outer accept/reject 的独立判据

---

## 10. 时间步接受 / 拒绝

## 10.1 接受条件

当且仅当以下条件同时满足时，时间步接受：

1. inner solve 收敛
2. outer predictor-corrector 收敛
3. 没有出现非物理状态：
   - `a <= 0`
   - `T` 越界
   - `Y < 0`
   - `sum(Y)` 严重偏离 1
   - 物性计算失败
4. remap 守恒误差在容忍范围内

---

## 10.2 拒绝与回退

若失败，则：

1. 丢弃当前时间步的所有 outer/inner 结果
2. 回退到：
   - `U^n`
   - `a^n`
   - `dot_a^n`
3. 缩小 `dt`
4. 重新开始该时间步

---

## 11. 正式推荐的单步算法

下面给出完整流程。

### Step T0. 时间步开始
已知：

- `U^n`
- `a^n`
- `dot_a^n`
- `G^n`
- `dt`

### Step T1. predictor
\[
\dot a^{(0)} = \dot a^n
\]
\[
a^{(0)} = a^n + \Delta t \dot a^n
\]

### Step T2. outer loop
对 `k = 0,1,2,...,k_max`：

#### O2.1 构建当前几何和网格
- 用 `a^(k)` 构三段网格 `G^(k)`
- 用 `dot_a^(k)` 构控制面速度 `v_c^(k)`

#### O2.2 remap
- 由 `U^n, G^n -> G^(k)` 形成 `old_state_on_current_geometry^(k)`

#### O2.3 设置 inner 初值
- `k=0` 时用 remapped old state
- `k>0` 时优先用上一轮收敛状态 `U^(k-1)`

#### O2.4 inner solve
冻结：
- `a^(k)`
- `dot_a^(k)`
- `G^(k)`
- `v_c^(k)`
- `old_state_on_current_geometry^(k)`

解出 `U^(k)`。

#### O2.5 提取物理解
\[
\dot a_{phys}^{(k)}
=
u_{l,if}^{(k)}+\frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
\]

#### O2.6 corrector
\[
a_{target}^{(k+1)}
=
a^n+\frac{\Delta t}{2}\left(\dot a^n+\dot a_{phys}^{(k)}\right)
\]

若无欠松弛：
\[
a^{(k+1)} = a_{target}^{(k+1)}
\]
\[
\dot a^{(k+1)} = \dot a_{phys}^{(k)}
\]

若有欠松弛：
\[
a^{(k+1)}
=
(1-\omega_a)a^{(k)}+\omega_a a_{target}^{(k+1)}
\]
\[
\dot a^{(k+1)}
=
(1-\omega_v)\dot a^{(k)}+\omega_v \dot a_{phys}^{(k)}
\]

#### O2.7 检查 outer 收敛
若收敛，则跳出 outer loop。

### Step T3. 时间步接受检查
若：
- inner 收敛
- outer 收敛
- 状态物理可接受
- remap 误差可接受

则接受：

\[
U^{n+1}=U^{(k)}
\]
\[
a^{n+1}=a^{(k+1)}
\]
\[
\dot a^{n+1}=\dot a_{phys}^{(k)}
\]

并更新：

- `G^{n+1}`
- `v_c^{n+1}`

否则拒绝该时间步并缩小 `dt`。

---

## 12. 为什么 corrector 必须 anchored 到 `a^n, dot_a^n`

这是最容易被写错的地方，必须单独写清楚。

corrector 的本质是在求解隐式时间离散方程：

\[
a^{n+1}
=
a^n+\frac{\Delta t}{2}\left(\dot a^n+\dot a^{n+1}\right)
\]

outer iteration 只是求这个方程的 fixed point。  
因此每一轮 corrector 都必须写成：

\[
a^{(k+1)}
=
a^n+\frac{\Delta t}{2}\left(\dot a^n+\dot a_{phys}^{(k)}\right)
\]

而不能写成：

\[
a^{(k+1)}
=
a^{(k)}+\frac{\Delta t}{2}\left(\dot a^{(k)}+\dot a_{phys}^{(k)}\right)
\]

后者相当于在 outer 迭代中“再次推进时间”，会把 outer iteration 错写成伪时间步进。

---

## 13. 正式禁止事项

以下做法在本文件下明确禁止：

1. 在每一轮 outer iteration 里重新做 predictor
2. inner solve 中更新 `a^(k)` 或 `dot_a^(k)`
3. outer corrector 直接从 `a^(k)` 再推进一个 `dt`
4. 用未收敛的 inner 结果更新 outer 几何
5. inner 不收敛时仍然尝试接受时间步
6. outer 不收敛时强行接受时间步
7. 将 remap 放到 inner Newton 过程中执行
8. 用当前 trial `mpp` 实时更新 `dot_a` 并回灌 residual

---

## 14. 最终方法合同

### Outer/Inner Iteration Coupling Contract

1. `paper_v1` uses a two-layer iteration structure.
2. The outer loop owns geometry, grid, remap, and timestep acceptance/rejection.
3. The inner loop owns the fixed-geometry transport nonlinear solve.
4. Predictor is executed once per timestep.
5. Corrector is executed once per outer iteration.
6. The corrector is anchored to the accepted old-time values `a^n` and `dot_a^n`.
7. Inner residuals are evaluated on frozen geometry defined by the current outer iterate.
8. Eq.(2.18) inside the inner residual uses the frozen `dot_a^(k)`.
9. `dot_a_phys^(k)` is computed only after inner convergence.
10. A timestep is accepted only if both inner and outer loops converge and the resulting state is physically admissible.

---

## 15. 最终结论

`paper_v1` 的两层迭代循环正式定为：

- 每个时间步开始时只做一次 predictor
- 每一轮 outer iteration 后做一次 corrector
- outer loop 负责几何与网格
- inner loop 负责 fixed-geometry 物理解
- corrector 必须 anchored 到 `a^n, dot_a^n`
- inner 与 outer 通过：
  - `old_state_on_current_geometry`
  - `dot_a_phys`
  - `U^(k)`
  三类信息耦合

这就是 `paper_v1` 的 outer/inner iteration coupling 指导文件最终定稿版本。
