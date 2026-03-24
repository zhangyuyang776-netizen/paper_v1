# interface_block_unknowns_and_residuals_table_final

> Sync note:
> ???? outer/inner ????
> `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`
> ???????

## 1. 文件目的

本文档用于为 `paper_v1` 中的界面块建立一套**物理一致、数值自洽、可直接编码实现**的方法规范，重点解决以下问题：

1. 界面未知量、液相域未知量、气相域未知量分别对应哪些控制方程
2. `mpp` 应该使用哪条方程构建残差
3. Eq.(2.15)-Eq.(2.19) 在数值实现中各自承担什么角色
4. 各条界面残差需要哪些参数，这些参数从哪里来
5. 内层 fixed-geometry nonlinear solve 与外层 predictor-corrector / moving mesh 如何分层组织

本文档是 `paper_v1` 的**物理控制 + 数值离散**联合指导文件，不是单纯的接口说明。

---

## 2. 适用范围与总原则

### 2.1 适用范围

本文档适用于 `paper_v1` 的以下路线：

* 1D 球对称液滴问题
* moving control volume / moving mesh
* 外层 predictor-corrector 更新界面位置和网格
* 内层在固定当前几何上求解液相 bulk + 界面块 + 气相 bulk 的耦合非线性系统
* transport 主系统不含化学源项
* 辐射在 `paper_v1` 中关闭

这些边界与主论文第二章的总体框架一致，也与 `PAPER_V1_ROADMAP.md` 中已经定下的“两层结构”一致。  

### 2.2 总原则

1. **界面块是总未知量向量的一部分，不是后处理附件。** 主论文把总未知量写成
   [
   \Omega=(\dots,\Phi_{l,n},\dots,\phi_s,\dots,\Phi_{g,n},\dots)
   ]
   并把界面变量块 `φ_s` 与界面残差块 `f_s` 一起并入整体非线性问题。
2. **`mpp` 是独立界面未知量，其实现采用“liquid-side Eq.(2.18) 主残差 + gas-side Eq.(2.18) 边界强施加”。**
3. **`Rd/a(t)` 不是内层主未知量**，由外层 predictor-corrector 在内层状态收敛后更新。主论文明确写了：新变量值已知后，再用
   [
   \dot a=\dot m''/\rho_l + (u_l)_{r=a}
   ]
   更新界面位置与网格。
4. **内层 residual 始终是固定当前几何的 residual**。因此若 Eq.(2.18) 进入 `mpp` 残差，则其中出现的 `\dot a` 必须取当前 outer iterate 的冻结值，而不能在 inner Newton 过程中临时更新。这个“冻结外层几何参数”的实现解释虽然不是论文逐字展开的程序细节，但它与论文的 predictor-corrector 结构和你路线图中的两层结构是自洽的。 

---

## 3. 总体 unknown — residual — control equation 对照表

## 3.1 液相域 bulk

| 位置             | 显式未知量                      | 残差               | 控制/闭合方程       |
| -------------- | -------------------------- | ---------------- | ------------- |
| 液相 cell `n`    | `T_l,n`                    | `R_l^E(n)=0`     | 液相能量方程        |
| 液相 cell `n`    | `Y_l,n,i`（reduced species） | `R_l^{Y_i}(n)=0` | 液相组分守恒方程      |
| 液相 face        | `u_l,n±`（非显式未知）            | 无单独主残差           | 连续方程离散恢复      |
| 液相 cell / face | `ρ_l`（非显式未知）               | 无单独主残差           | 液相 EOS / 物性模型 |

液相 bulk 守恒量在主论文中统一按
[
\phi_\beta={\rho_\beta,\rho_\beta Y_{\beta,i},\rho_\beta h_\beta}
]
写出，并在 moving control volume 上离散；有限体积统一形式见 Eq.(2.26)，时间离散后的单元残差形式见 Eq.(2.31)。 

## 3.2 气相域 bulk

| 位置             | 显式未知量                      | 残差               | 控制/闭合方程       |
| -------------- | -------------------------- | ---------------- | ------------- |
| 气相 cell `n`    | `T_g,n`                    | `R_g^E(n)=0`     | 气相能量方程        |
| 气相 cell `n`    | `Y_g,n,i`（reduced species） | `R_g^{Y_i}(n)=0` | 气相组分守恒方程      |
| 气相 face        | `u_g,n±`（非显式未知）            | 无单独主残差           | 连续方程离散恢复      |
| 气相 cell / face | `ρ_g`（非显式未知）               | 无单独主残差           | 气相 EOS / 物性模型 |

与液相同理，气相 bulk residual 仍来自主论文第二章的守恒方程离散；`paper_v1` 只是把化学源项从 transport residual 中移除，不改第二章的输运骨架。 

## 3.3 界面块 interface

### 3.3.1 论文式界面变量块

主论文将界面变量写为
[
\phi_s={\rho_{s,l},Y_{s,l,i},T_s,\rho_{s,g},Y_{s,g,i},\dot m''}
]
并明确把界面方程 Eq.(2.15)-Eq.(2.19) 作为数值问题的一部分。

### 3.3.2 `paper_v1` 推荐界面块

为了保持物理输运与主论文一致，同时避免把已由状态唯一决定的密度重复当作代数 unknown，`paper_v1` 推荐：

* **单组分液滴首版**
  [
  \phi_{s,A}={T_s,;Y_{s,g,red},;mpp}
  ]
* **多组分液滴升级版**
  [
  \phi_{s,B}={Y_{s,l,red},;T_s,;Y_{s,g,red},;mpp}
  ]

其中 `ρs,l`、`ρs,g` 由 EOS / 物性派生，不作为首版显式未知量。这是对主论文界面块的**物理一致、代数适度约简**版本；不是物理裁剪，而是代数去冗余。这个约束也和你前面已确认的未知量策略一致。 

### 3.3.3 界面块的一一对应关系

| 界面未知量                                | 残差            | 控制/闭合方程         |
| ------------------------------------ | ------------- | --------------- |
| `Y_s,l,i`（可凝/共有液相侧界面组分，reduced）      | `R_if,l,i=0`  | Eq.(2.15)       |
| `Y_s,g,j(cond)`（可凝蒸汽界面组分）            | `R_if,eq,j=0` | Eq.(2.19)       |
| `Y_s,g,k(noncond)`（非凝气体界面组分，reduced） | `R_if,g,k=0`  | Eq.(2.16)       |
| `T_s`                                | `R_if,E=0`    | Eq.(2.17)       |
| `mpp=\dot m''`                       | `R_if,m=0`    | Eq.(2.18)       |
| `ρ_s,l`（若显式保留）                       | `R_if,ρl=0`   | 液相界面 EOS / 物性闭合 |
| `ρ_s,g`（若显式保留）                       | `R_if,ρg=0`   | 气相界面 EOS / 状态方程 |

这张表就是本文件最核心的 bookkeeping。它解决了“`mpp` 放哪”的问题：
**如果 `mpp` 作为独立界面 unknown，那么它的主残差位置就是 Eq.(2.18)。**
若不用 Eq.(2.18)，那只能把 `mpp` 从界面 unknown block 里删掉，改成由某个 species 方程反解，这就不再是论文式界面变量块主线了。

---

## 4. 界面方程分块说明

## 4.1 Eq.(2.15): 共有/可凝组分的界面物种守恒

### 功能

Eq.(2.15) 约束气液两相共有的可凝/可溶组分在界面的物种守恒。它属于**液相侧界面组分 unknown 的主方程组**。

### 对应未知量

* `Y_s,l,i`（reduced）
* 与之耦合的 `Y_s,g,i(cond)`、`mpp`、界面扩散通量

### 需要的参数

* `mpp`
* `Y_s,l,i`
* `Y_s,g,i`
* `J_l,i|if`
* `J_g,i|if`

### 参数如何得到

* `mpp`：当前界面 unknown
* `Y_s,l,i`：当前界面 unknown（多组分阶段）
* `Y_s,g,i(cond)`：当前界面 unknown，且同时受 Eq.(2.19) 约束
* `J_l,i|if`：液相界面侧扩散通量，由液相组分、液相扩散系数、液相界面梯度离散得到
* `J_g,i|if`：气相界面侧扩散通量，由气相组分、气相扩散系数、气相界面梯度离散得到

### 离散实现要点

* `J` 不是这条方程“直接算出来”的，而是由传输模型闭合后代入
* 界面梯度采用与 bulk 一致的一侧重构 / 中心差分邻近式
* 单组分液滴首版通常没有这组方程的自由度，因为液相只有一个组分

---

## 4.2 Eq.(2.16): 非凝组分的界面物种守恒

### 功能

Eq.(2.16) 约束只存在于气相的非凝组分在界面的物种守恒。它属于**气相侧非凝界面组分 unknown 的主方程组**。

### 对应未知量

* `Y_s,g,k(noncond)`（reduced）
* 与之耦合的 `mpp`、`J_g,k|if`

### 需要的参数

* `mpp`
* `Y_s,g,k(noncond)`
* `J_g,k|if`

### 参数如何得到

* `mpp`：当前界面 unknown
* `Y_s,g,k(noncond)`：当前界面 unknown
* `J_g,k|if`：由气相 bulk 侧温度/组分、扩散系数和界面梯度计算得到

### 离散实现要点

* 这条方程在“只用两未知 `Ts,mpp`”的工程裁剪版里，可以被拿来反解 `mpp`
* 但在**论文式界面变量块主线**中，Eq.(2.16) 的主要职责是约束**非凝界面组分 unknown**
* 因此本文件中**不再把 Eq.(2.16) 作为 `mpp` 的主残差位置**

---

## 4.3 Eq.(2.17): 界面能量守恒

### 功能

Eq.(2.17) 是界面能量平衡，属于 `Ts` 的主残差方程。主论文中该式包含导热、相变潜热、液相扩散潜热项，以及辐射项。

### 对应未知量

* `T_s`

### 需要的参数

* `mpp`
* `Y_s,l,i`
* `L_i(T_s)` 或等价潜热模型
* `k_l|if`, `k_g|if`
* `(\partial T_l/\partial r)|if`, `(\partial T_g/\partial r)|if`
* 液相界面扩散焓/潜热项
* 辐射项（`paper_v1` 中关闭）

### 参数如何得到

* `mpp`：当前界面 unknown
* `Y_s,l,i`：界面 unknown 或单组分下固定 `[1.0]`
* `L_i(T_s)`：由液相相变热模型或纯组分数据库按 `Ts` 计算，例如 Watson/Rackett 等闭合模型已经在你当前代码库中存在雏形。
* `k_l, k_g`：由液相 / 气相物性模块按当前 trial state 计算
* 温度梯度：由液相和气相界面相邻 cell 的一侧离散得到
* 辐射项：`paper_v1` 置零，但方程结构保留，不能替换成另一套方程。`PAPER_V1_ROADMAP.md` 已明确首版不考虑辐射。

### 离散实现要点

* `Ts` 是显式界面 unknown
* 残差必须保持主论文界面能量守恒结构
* 关闭辐射是删项，不是换方程

---

## 4.4 Eq.(2.18): 总质量通量定义 / 运动学相容关系

### 功能

Eq.(2.18) 定义了界面总质量通量 `mpp=\dot m''` 与两相界面相对速度、界面移动速度之间的关系，是**`mpp` 的主残差方程**。主论文同时也用它重排出界面推进公式。

### 对应未知量

* `mpp`

### 对应残差

`paper_v1` 中对 Eq.(2.18) 的正式实现不是“一个 unknown 对两条独立 residual rows”，而是：

- liquid-side Eq.(2.18) 作为 `mpp` 的主残差
- gas-side Eq.(2.18) 通过界面面质量流率边界构造被强施加

推荐主残差写成液相侧形式：
[
R_{if,m}
========

\dot m'' + \rho_{s,l}\big(u_{l,if}-\dot a^{(k)}\big)
=0
]

对应的 gas-side 强施加边界写法为：
[
G_{g,if}^{abs}
===========

(\rho_{s,g}\dot a^{(k)}-\dot m'')A_{if}
]

首版推荐统一采用**液相侧主残差 + 气相侧边界强施加**，与论文后续用液相侧公式推进半径保持一致，同时保持两相共同耦合。

### 需要的参数

* `u_l|if`
* `u_g|if`
* `ρ_s,l`
* `ρ_s,g`
* `\dot a`

### 参数如何得到

* `u_l|if`：由液相连续方程离散恢复，从中心边界向界面 sweep 得到
* `u_g|if`：由 gas-side 边界强施加关系
  [
  u_{g,if}=\dot a^{(k)}-\dot m''/\rho_{s,g}
  ]
  给出，并作为气相 continuity sweep 的界面起点
* `ρ_s,l`, `ρ_s,g`：由当前界面状态通过 EOS / 物性闭合得到
* `\dot a`：**不是 inner unknown**，而是当前 outer iteration 的冻结几何参数
  [
  \dot a^{(k)} = v_{c,if}^{(k)}
  ]

### 离散实现要点

这是全文件最关键的约束：

1. 若 Eq.(2.18) 进入 `mpp` 的 inner residual，则其中的 `\dot a` 必须取当前 outer iterate 的冻结值
   [
   \dot a=\dot a^{(k)}
   ]
2. gas-side Eq.(2.18) 必须通过界面边界质量流率被强施加，而不是降级成“完全不参与求解的注释”
3. 在 inner Newton / SNES 迭代过程中，不允许用 trial `mpp` 再即时更新 `\dot a`
4. 否则 residual 就不再是 fixed-geometry residual，而变成 moving-target residual，Newton 会被你自己拆台

这条是从论文“先求新变量、再更新界面位置和网格”的 predictor-corrector 顺序，与 `PAPER_V1_ROADMAP.md` 的“外层几何、内层固定网格 transport solve”一起推出来的自洽实现约束。论文没有逐字写“inner 中冻结 \dot a”的程序说明，但按其算法结构这是最合理也最干净的实现。 

---

## 4.5 Eq.(2.19): 可凝蒸汽界面相平衡

### 功能

Eq.(2.19) 约束可凝蒸汽在界面的平衡组成，属于**可凝蒸汽界面组分 unknown 的主方程组**。

### 对应未知量

* `Y_s,g,j(cond)`（或等价 mole fraction unknown）

### 需要的参数

* `T_s`
* `Y_s,l,i`
* `P_inf`
* 活度系数 `γ_i`
* 分子量 `W_l, W_g`
* 汽化潜热 / Clausius 型积分所需参数

### 参数如何得到

* `T_s`：当前界面 unknown
* `Y_s,l,i`：液相侧界面 unknown（多组分）或单组分固定值
* `P_inf`：全域恒压，已由 unknowns 策略定死
* `γ_i`：首版若单组分可取 `γ=1`；多组分阶段由 UNIFAC / 活度模型给出
* 潜热与饱和性质：由纯组分数据库 / 饱和模型给出

### 离散实现要点

* 这是一组**平衡 residuals**
* 它只负责**可凝蒸汽**界面组分
* 它不负责背景气 / 非凝气体，这一点必须和 Eq.(2.16) 分清

---

## 5. 参数来源总表

## 5.1 bulk → interface 传递的参数

| 参数                     | 来源                     |               |                          |
| ---------------------- | ---------------------- | ------------- | ------------------------ |
| `T_l,if^-`, `T_g,if^+` | 液相 / 气相界面相邻 cell 的一侧重构 |               |                          |
| `Y_l,if^-`, `Y_g,if^+` | 液相 / 气相界面相邻 cell 的一侧重构 |               |                          |
| `∂T/∂r                 | if`                    | 界面相邻单元的一侧梯度离散 |                          |
| `∂Y/∂r                 | if`                    | 界面相邻单元的一侧梯度离散 |                          |
| `J_l                   | if`, `J_g              | if`           | 由相内扩散模型 + 界面梯度计算         |
| `u_l                   | if`, `u_g              | if`           | 由 continuity recovery 得到 |
| `ρ_l                   | if`, `ρ_g              | if`           | 由 EOS / 物性模块按当前界面状态计算    |

## 5.2 interface 自身未知提供的参数

| 参数      | 来源                |
| ------- | ----------------- |
| `T_s`   | 界面 unknown        |
| `Y_s,l` | 界面 unknown（多组分阶段） |
| `Y_s,g` | 界面 unknown        |
| `mpp`   | 界面 unknown        |

## 5.3 outer geometry 提供的参数

| 参数              | 来源                               |
| --------------- | -------------------------------- |
| `a^(k)`         | 当前 outer iteration 的几何试探值        |
| `\dot a^(k)`    | 当前 outer iteration 的界面速度试探值      |
| _f, A_f, V_n` | 当前 outer geometry 生成的 fixed grid |
| `v_c`           | 当前 outer geometry 给定的控制面速度       |

---

## 6. `mpp` 使用 Eq.(2.18) 时的正式方法约束

### 6.1 必须冻结 `\dot a`

在任意 outer iteration `k` 的 inner solve 中：

[
R_{if,m}(U ; \dot a^{(k)}) = 0
]

其中 `\dot a^{(k)}` 是当前 fixed-geometry 问题的外层冻结参数，不是 inner unknown。

### 6.2 不允许 inner 中实时更新 `\dot a`

禁止以下做法：

1. inner 试一个 `mpp_trial`
2. 立刻用它更新 `\dot a_trial`
3. 再把新的 `\dot a_trial` 塞回当前 residual
4. 下一次 Newton 再继续改

这种写法会破坏 fixed-geometry residual 的定义，让 Jacobian 对应的函数都不再固定。典型的人类工程事故。

### 6.3 为什么仍然选择 Eq.(2.18)

因为在论文式界面变量块主线下：

* `mpp` 是独立界面 unknown
* 它需要一个独立界面 residual 位置
* 2.15 / 2.16 / 2.19 都属于 species-level residuals
* 2.17 属于 energy residual
* 只有 2.18 是**总质量通量定义式**

因此，**若保留 `mpp` 为独立界面 unknown，就应当由 Eq.(2.18) 给出主残差。**
但由于一个 `mpp` unknown 不应在同一个 square system 中对应两条独立 residual rows，所以 `paper_v1` 的正式实现是：

- liquid-side Eq.(2.18) 给主残差
- gas-side Eq.(2.18) 通过界面边界质量流率强施加

否则你只能把 `mpp` 从界面 unknown block 中删掉，改成由某条 species 方程派生，这就不再是论文式界面变量块路线。

---

## 7. 内外层整体计算框架

## 7.1 外层职责

外层 predictor-corrector / geometry module 负责：

* 保存 `a^n`
* 预测 `a^{n+1,(k)}` 或 `\dot a^{(k)}`
* 在当前 outer iterate 上构建 fixed grid
* 对旧场做 remap / projection
* 管理 time-step accept / reject
* 在 inner 收敛后更新界面位置与网格

这与 `PAPER_V1_ROADMAP.md` 中的职责边界完全一致。

## 7.2 内层职责

inner transport nonlinear solve 负责：

* 在固定当前几何和固定当前网格上求解
  [
  U = [\Phi_l,\phi_s,\Phi_g]
  ]
* 组装 liquid bulk residual
* 组装 interface residual
* 组装 gas bulk residual
* 在 residual 评估时恢复速度、计算通量、计算物性

SNES / Newton 只负责 fixed-geometry transport system，不负责更新 `Rd`、重建网格、remap 或 timestep decision。

## 7.3 单步计算流程

### 外层第 `k` 次几何迭代

1. 给定 `a^(k)`, `\dot a^(k)`, `grid^(k)`, `v_c^(k)`
2. 设置当前轮 inner 入口状态；`k=0` 时直接取时间层开始状态，`k>0` 时使用上一轮 outer 状态转移结果
3. 进入 inner nonlinear solve

### inner residual 评估

1. 从当前 trial state 读取 liquid bulk / interface / gas bulk unknowns
2. 计算 EOS / 物性：`ρ, h, k, D, L, gamma...`
3. 用当前 fixed geometry 的 `V_n, A_f, v_c`
4. 用 continuity recovery 计算 `u_l|if, u_g|if` 以及各面速度。主论文 Eq.(2.30) 正是这一恢复关系。
5. 组装 liquid bulk residuals
6. 组装 interface residuals：2.15 / 2.16 / 2.17 / 2.18 / 2.19
7. 组装 gas bulk residuals
8. 由 SNES / Newton 迭代直至 inner 收敛

### inner 收敛后做 outer corrector

1. 取收敛后的 `mpp`, `u_l|if`, `ρ_l|if`
2. 用主论文给出的公式更新界面速度
   [
   \dot a^{new}=u_l|*{if}+\frac{mpp}{\rho_l|*{if}}
   ]
   这是论文 2.3.3 明确写出来的界面推进公式。
3. 更新界面位置 `a`
4. 重建网格与 `v_c`
5. 检查外层界面速度收敛；若未收敛，进入下一轮 outer iteration

---

## 8. 单组分首版与多组分升级版的界面块差异

## 8.1 单组分液滴首版

推荐界面块：

[
\phi_{s,A}={T_s,;Y_{s,g,red},;mpp}
]

此时：

* `Y_s,l,full=[1.0]` 固定存在于 state，但不是自由 unknown
* Eq.(2.15) 对单组分液滴没有可解自由度
* Eq.(2.19) 给出燃料蒸汽界面平衡组成
* Eq.(2.16) 给出非凝组分界面守恒
* Eq.(2.17) 给 `Ts`
* Eq.(2.18) 给 `mpp`

## 8.2 多组分液滴升级版

推荐界面块：

[
\phi_{s,B}={Y_{s,l,red},;T_s,;Y_{s,g,red},;mpp}
]

此时：

* Eq.(2.15) 恢复其完整作用，约束共有/可凝组分界面守恒
* Eq.(2.19) 继续约束可凝蒸汽界面平衡
* Eq.(2.16) 继续约束非凝组分
* Eq.(2.17) 给 `Ts`
* Eq.(2.18) 给 `mpp`

---

## 9. 正式禁止事项

以下做法在本文件下明确禁止：

1. 把 `mpp` 保留为独立界面 unknown，却不为它提供 Eq.(2.18) 残差位置
2. 在 inner residual 里一边用 Eq.(2.18)，一边在 Newton 过程中实时更新 `\dot a`
3. 把 Eq.(2.16) 强行当成 `mpp` 的主残差，同时又保留完整论文式界面 species block
4. 让 `equilibrium_model` 同时包办可凝蒸汽和背景气组分
5. 让界面 residual 使用与当前 fixed geometry 不一致的 `v_c` / `\dot a`
6. 将 `Rd` 再塞回 inner unknown vector

---

## 10. 最终方法合同

### Interface MPP / Coupling Contract

1. `mpp` is an independent interface unknown in the paper-like interface block.
2. The primary residual for `mpp` is the liquid-side form of Eq.(2.18).
3. Eq.(2.15), Eq.(2.16), Eq.(2.19) are species-level interface residuals.
4. Eq.(2.17) is the interface energy residual for `Ts`.
5. Interface densities are closed by EOS / property relations, not by Eq.(2.15)-Eq.(2.19).
6. Gas-side Eq.(2.18) is strongly imposed through the interface boundary mass-flux construction, not added as a second independent `mpp` residual row.
7. If Eq.(2.18) is used inside the inner residual, `\dot a` must be the frozen outer-iteration value.
8. `Rd/a(t)` remains an outer predictor-corrector variable, not an inner transport unknown.
9. After inner convergence, `\dot a` is updated from
   [
   \dot a = u_l|*{if} + mpp/\rho*{l,if}
   ]
   and then the mesh is rebuilt.
10. The inner solver works on a fixed-geometry coupled system
   [
   U=[\Phi_l,\phi_s,\Phi_g]
   ]
   only.

---

## 11. 最终结论

`paper_v1` 中关于界面 `mpp` 的正式结论是：

* **`mpp` 是独立界面 unknown**
* **其主残差对应 liquid-side Eq.(2.18)**
* **gas-side Eq.(2.18) 通过界面边界质量流率被强施加**
* **Eq.(2.18) 中的 `\dot a` 在 inner solve 中取当前 outer iterate 的冻结值**
* **界面位置 `a/Rd` 仍由 outer predictor-corrector 在 inner 收敛后更新**
* **因此，物理控制、数值离散与内外层框架三者是自洽的**

这份文件就是 `paper_v1` 中“界面 mpp 方程与耦合框架”的正式定稿版本。
