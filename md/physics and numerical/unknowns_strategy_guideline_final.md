# unknowns_strategy_guideline_final

## 1. 文件目的

本文档用于为新代码项目 `paper_v1` 规定**未知量策略**，并将其与以下内容统一起来：

- 主参考文献第二章的物理与数值框架
- 当前项目已经确定的“两层结构”
- moving mesh / predictor-corrector 的职责边界
- 连续方程恢复速度的逻辑
- 首版 `transport-only baseline` 的工程目标
- 论文式界面变量块主线

本文档的目标不是罗列“可能有哪些未知量”，而是把**新项目必须采用的未知量组织方式定死**，避免后续实现时又把界面、几何、速度、平衡辅助量随意塞回主系统，或在不同模块中采用彼此冲突的未知量定义。

---

## 2. 总结论

新项目 `paper_v1` 应采用以下未知量总策略：

### 2.1 第一版与推荐最终版的总策略

- `Rd` **不进入**内层 transport 主未知量
- `u_l`、`u_g` **不进入**内层 transport 主未知量
- bulk 主未知量采用：
  - 液相：`T_l`，以及需要时的 `Y_l`
  - 气相：`T_g`，以及 `Y_g`
- 每相都去掉一个闭合组分，用 `sum(Y)=1` 回补
- 能量方程在**物理意义上**按焓守恒处理，但在第一版**代数未知量形式**上采用温度 `T`
- 首版不做显式 unknown scaling
- 首版与推荐最终版都采用**论文式界面变量块主线**
- 但对论文式界面块做“物理一致、代数适度约简”：
  - `ρ_s,l`、`ρ_s,g` 由 EOS / 物性派生
  - 不作为首版显式界面 unknown

### 2.2 两层结构下的职责边界

- 外层负责：
  - `Rd = a(t)` 更新
  - moving mesh
  - remap / projection
  - 时间步接受/拒绝
  - 外层 predictor-corrector 几何迭代
- 内层负责：
  - 在固定当前几何上求解 `U_transport`
  - 恢复速度
  - 组装 liquid bulk residual、interface residual、gas bulk residual

### 2.3 关键升级

本版正式取消旧口径中“界面显式未知量仅为 `T_s, mpp`”的主线表述。  
新的正式主线是：

- **单组分液滴首版**：界面块至少包含 `T_s`、`Y_s,g,red`、`mpp`
- **多组分液滴升级版**：界面块包含 `Y_s,l,red`、`T_s`、`Y_s,g,red`、`mpp`

---

## 3. 未知量设计的总原则

新项目的未知量设计必须遵守以下总原则。

### 3.1 原则一：几何与 transport 分离

首版与推荐最终版都不把 `Rd` 放进内层 transport 主系统。

理由：

1. 内层 `SNES` 面对的是固定几何问题，更容易诊断收敛性
2. moving mesh、remap、geometry iteration 的职责边界更清晰
3. 与外层 predictor-corrector 组织方式一致
4. 更适合后续扩展到 MPI 和更复杂的网格/装配结构

### 3.2 原则二：速度不是主未知量

`u_l`、`u_g` 不是主未知量，而是由离散连续方程恢复的派生量。

理由：

1. 与主参考文献第二章一致
2. 避免重新引入“Stefan 速度场经验外推”的旧问题
3. 对流项、界面质量通量、moving control volume 可保持统一

### 3.3 原则三：能量方程按焓一致形式处理，但代数未知量用温度

能量方程的物理控制方程应保持焓守恒含义，但主未知量采用温度 `T`，而不是直接采用 `h` 或 `ρh` 作为代数未知量。

### 3.4 原则四：界面采用论文式变量块，而非过度裁剪的两未知量块

主参考文献把界面变量块写为：

\[
\phi_s=\{\rho_{s,l},Y_{s,l,i},T_s,\rho_{s,g},Y_{s,g,i},\dot m''\}
\]

并把界面方程 Eq.(2.15)-Eq.(2.19) 连同 EoS 一起纳入数值问题。  
`paper_v1` 应沿用其**物理结构**，而不再以“仅保留 `T_s, mpp` 两未知量”为正式主线。

### 3.5 原则五：界面密度不显式作为首版未知量

虽然主论文的 `φ_s` 字面包含 `ρ_s,l` 与 `ρ_s,g`，但在 `paper_v1` 中：

- `ρ_s,l = ρ_l(T_s, Y_s,l)`
- `ρ_s,g = ρ_g(T_s, Y_s,g, P_inf)`

由 EOS / 物性模型唯一给出，因此首版与推荐最终版都不把它们作为显式代数 unknown。

---

## 4. 论文中的未知量应如何正确理解

### 4.1 `φ_s` 不是整个系统的全部未知量

主参考文献中，类似

\[
\phi_s = \{\rho_{s,l}, Y_{s,l,i}, T_s, \rho_{s,g}, Y_{s,g,i}, \dot m''\}
\]

的表达，表示的是**界面变量块**，不是整个系统全部未知量。

### 4.2 完整系统未知量由三部分组成

新项目应沿用这个基本理解：

\[
\Omega = (\Phi_l,\phi_s,\Phi_g)
\]

也就是：

- 液相 bulk unknown block
- 界面 unknown block
- 气相 bulk unknown block

### 4.3 界面块与 bulk 块的关系

界面块不是后处理附件，也不是“从 bulk 插值一下就完事”的辅助数据结构。  
它是总离散系统中的一个独立块，用来承载：

- 界面物种守恒
- 界面能量守恒
- 界面总质量通量定义
- 界面相平衡
- 必要的界面状态闭合

### 4.4 为什么 `T_l`、`T_g` 不在 `φ_s` 中

因为它们属于 liquid/gas bulk 状态，不属于界面块。  
同理，bulk 的 `Y_l(r)`、`Y_g(r)` 是域内未知量；界面的 `Y_s,l`、`Y_s,g` 是边界/界面块中的未知量。

---

## 5. 第一版与推荐最终版的正式未知量定义

## 5.1 液相域 bulk

### 单组分液滴首版
```text
liquid bulk = [T_l]
````

### 多组分液滴升级版

```text
liquid bulk = [T_l, Y_l,1 ... Y_l,Nl_eff]
```

说明：

* `T_l` 必须是液相 bulk 的基本未知量
* 单组分阶段液相组分不作为显式主未知量
* 多组分升级时，液相 reduced species 进入主系统

---

## 5.2 气相域 bulk

第一版起就按多组分设计：

```text
gas bulk = [T_g, Y_g,1 ... Y_g,Ng_eff]
```

说明：

* 气相从第一版起就是多组分
* transport 步内不含化学源项，但气相组分守恒仍然存在
* 一个气相组分作为 closure species，不显式进入未知量

---

## 5.3 界面块 interface

### 5.3.1 单组分液滴首版

推荐界面块为：

```text
interface_A =
[
  T_s,
  Y_s,g,1 ... Y_s,g,Ng_eff,
  mpp
]
```

说明：

* 单组分液相中 `Y_s,l,full = [1.0]`
* 它固定存在于 state，但不是独立自由 unknown
* 界面气相组分块从第一版起就应显式保留
* `mpp` 作为独立界面 unknown 保留

### 5.3.2 多组分液滴升级版

推荐界面块为：

```text
interface_B =
[
  Y_s,l,1 ... Y_s,l,Nl_eff,
  T_s,
  Y_s,g,1 ... Y_s,g,Ng_eff,
  mpp
]
```

说明：

* 多组分液滴阶段，液相侧界面组分成为显式界面 unknown
* 气相侧界面组分继续作为显式界面 unknown
* `mpp` 继续作为独立界面 unknown
* `T_s` 继续作为独立界面 unknown

### 5.3.3 首版不显式进入界面块的量

以下量在首版与推荐最终版中**不作为显式界面 unknown**：

* `ρ_s,l`
* `ρ_s,g`
* `a_dot = da/dt`
* `Rd`
* 任意未定义清楚的 `y_if_aux...`

其中：

* `ρ_s,l`、`ρ_s,g` 由 EOS / 物性派生
* `a_dot`、`Rd` 属于外层几何模块
* `y_if_aux...` 占位写法正式废止

---

## 6. unknown — residual — control/closure equation 对照

这是 `paper_v1` 中必须遵守的 bookkeeping 总表。

## 6.1 液相域

| 显式未知量              | 对应残差             | 控制/闭合方程       |
| ------------------ | ---------------- | ------------- |
| `T_l,n`            | `R_l^E(n)=0`     | 液相能量方程        |
| `Y_l,n,i`（reduced） | `R_l^{Y_i}(n)=0` | 液相组分守恒方程      |
| `u_l`（非显式 unknown） | 无单独主残差           | 连续方程离散恢复      |
| `ρ_l`（非显式 unknown） | 无单独主残差           | 液相 EOS / 物性闭合 |

## 6.2 气相域

| 显式未知量              | 对应残差             | 控制/闭合方程       |
| ------------------ | ---------------- | ------------- |
| `T_g,n`            | `R_g^E(n)=0`     | 气相能量方程        |
| `Y_g,n,i`（reduced） | `R_g^{Y_i}(n)=0` | 气相组分守恒方程      |
| `u_g`（非显式 unknown） | 无单独主残差           | 连续方程离散恢复      |
| `ρ_g`（非显式 unknown） | 无单独主残差           | 气相 EOS / 物性闭合 |

## 6.3 界面块

| 显式未知量                                | 对应残差          | 控制/闭合方程         |
| ------------------------------------ | ------------- | --------------- |
| `Y_s,l,i`（可凝/共有液相侧界面组分，reduced）      | `R_if,l,i=0`  | Eq.(2.15)       |
| `Y_s,g,j(cond)`（可凝蒸汽界面组分）            | `R_if,eq,j=0` | Eq.(2.19)       |
| `Y_s,g,k(noncond)`（非凝气体界面组分，reduced） | `R_if,g,k=0`  | Eq.(2.16)       |
| `T_s`                                | `R_if,E=0`    | Eq.(2.17)       |
| `mpp = \dot m''`                     | `R_if,m=0`    | Eq.(2.18)       |
| `ρ_s,l`（若未来显式保留）                     | `R_if,ρl=0`   | 液相界面 EOS / 物性闭合 |
| `ρ_s,g`（若未来显式保留）                     | `R_if,ρg=0`   | 气相界面 EOS / 状态方程 |

### 6.3.1 对 `mpp` 的正式结论

在 `paper_v1` 的论文式界面变量块主线中：

* `mpp` 是独立界面 unknown
* 它的主残差位置是 **liquid-side Eq.(2.18)**
* gas-side Eq.(2.18) 通过界面边界质量流率被强施加，而不是再增加第二条 `mpp` residual row
* 若不用 Eq.(2.18)，则只能把 `mpp` 从界面 unknown block 中删掉，改成由某条 species 方程派生
* 那将不再是论文式界面变量块主线

### 6.3.2 对 `a/Rd` 的正式结论

* `a/Rd` 不属于内层界面块
* `a_dot` 也不属于内层界面块
* 主论文中是“新变量值已知后”，再由
  [
  \dot a = \dot m''/\rho_l + (u_l)_{r=a}
  ]
  更新界面位置和网格
* 因此 `a/Rd` 仍由外层 predictor-corrector 管理

---

## 7. 压力与 EOS 策略

### 7.1 正式结论

`paper_v1` 首版采用：

* **全域恒压** `P_inf`
* 不引入额外压力未知量
* 界面平衡统一使用 `Pg_if = P_inf`

### 7.2 气相密度

气相密度由气相状态派生：

[
\rho_g = \rho_g(T_g, Y_g, P_\infty)
]

### 7.3 液相密度

液相密度由液相物性模型派生：

[
\rho_l = \rho_l(T_l, Y_l)
]

界面密度相应为：

[
\rho_{s,l} = \rho_l(T_s, Y_{s,l})
]
[
\rho_{s,g} = \rho_g(T_s, Y_{s,g}, P_\infty)
]

### 7.4 方法约束

```text
Pressure/EOS Contract
- paper_v1 首版气相压力采用全域给定常压 P_inf
- 界面平衡统一使用 Pg_if = P_inf
- 不引入额外界面压力未知量或气相压力未知量
- rho_g 由 (Tg, Yg, P_inf) 派生
- rho_l 由选定液相物性模型按 (Tl, Yl) 派生
- rho_s,l 与 rho_s,g 首版不作为显式 unknown
```

---

## 8. 闭合组分策略

### 8.1 总原则

每相都采用“去掉一个闭合组分、由 `sum(Y)=1` 回补”的 reduced-unknown 策略。

### 8.2 配置约束

闭合物种必须由配置文件**显式给出**：

* 气相闭合物种：配置文件显式设置
* 液相闭合物种：配置文件显式设置

### 8.3 禁止默认值

代码逻辑中**不允许存在默认闭合物种**。

若配置文件未设置：

* 立即报错
* 拒绝启动
* 不允许 silent fallback

### 8.4 单组分液滴的液相闭合物种

即便 `Nl = 1`，液相闭合物种也必须显式配置。

规则：

* 当 `Nl = 1` 时，液相闭合物种必须等于液相唯一物种
* 若未配置或配置错误，直接报错

### 8.5 固定索引策略

闭合组分采用**固定索引策略**，不允许按状态动态切换。

### 8.6 一致性要求

同一相内的闭合组分映射必须在以下模块中保持一致：

* unknown layout
* interface layout
* equilibrium 输入输出
* residual 装配
* Jacobian 装配
* I/O 输出
* diagnostics

---

## 9. 单组分阶段的状态结构策略

### 9.1 单组分液相

即使在单组分液滴阶段，state 中仍保留液相组分数组：

```text
Yl_full = [1.0]
```

说明：

* 它不是主未知量
* 但它必须作为统一接口字段存在
* 不允许单组分阶段把液相组分数组从 state 结构中删掉

### 9.2 气相

气相始终保留：

```text
Yg_full
```

说明：

* state 中始终保存 full-order 气相组分数组
* 显式未知量只放 reduced gas species
* 闭合组分由 `1 - sum(other species)` 回补

### 9.3 单组分首版的界面液相状态

单组分首版中：

```text
Ys_l_full = [1.0]
```

说明：

* 它固定存在
* 它不是自由 unknown
* 但界面块和 EOS / 界面能量方程都可以统一读取这个字段

---

## 9.4 `State` / `state_pack` 的正式状态字段合同

正式规定：

- `core/types.py` 中的 `State` 不能只保存 reduced unknown
- `core/state_pack.py` 不能只围绕 reduced unknown 做打包/解包

必须显式保留的 full-order 状态字段包括：

- `Yl_full`
- `Yg_full`
- `Ys_l_full`
- `Ys_g_full`

说明：

- 对单组分液相，`Yl_full = [1.0]`
- 对单组分首版界面液相，`Ys_l_full = [1.0]`
- 即使某一侧当前没有自由度，这些字段也不能从 `State` 中删除

正式目标是：

- interface residual
- equilibrium
- diagnostics
- I/O

都读取统一的 full-order state，而不是在各自模块里反复从 reduced unknown 临时重构

---

## 10. 界面组成与相平衡策略

### 10.1 界面组分必须按论文式界面块理解

在新的正式主线中：

* `Y_s,l` 和 `Y_s,g` 不是简单“从 bulk 临时插值出来就不管了”的辅助量
* 它们是界面块中的状态量
* 其中不同类别的界面组分由不同方程约束

### 10.2 可凝蒸汽组分

* 属于界面气相组分块
* 对应 Eq.(2.19)
* 多组分阶段与 `Y_s,l` 和 `Ts` 强耦合

### 10.3 非凝气体组分

* 属于界面气相组分块
* 对应 Eq.(2.16)
* 不由相平衡公式决定
* 不允许使用比例补偿式背景气算法进入主框架

### 10.4 共有/可溶液相侧界面组分

* 属于液相侧界面组分块
* 对应 Eq.(2.15)
* 单组分液滴首版中没有这组自由度
* 多组分液滴阶段必须正式打开

### 10.5 `equilibrium_model` 的角色

在本版 unknown strategy 下，`equilibrium_model` 不再被定义为“首版外算一切界面组成的唯一来源”。
它的职责应缩窄为：

* 提供可凝蒸汽平衡关系所需热力学信息
* 提供 Eq.(2.19) 所需的热力学闭合
* 提供必要的活度系数 / 平衡元信息

不允许让它包办所有界面组分。

---

## 11. 能量未知量策略

### 11.1 正式结论

代数未知量采用温度：

```text
energy unknowns = [T_l, T_s, T_g]
```

而不是：

```text
[h_l, h_s, h_g]
```

或：

```text
[rho*h]_l, [rho*h]_g
```

### 11.2 但 residual 必须保持焓一致

尽管代数未知量采用温度，能量方程的残差组装必须满足：

* 时间项按焓守恒意义处理
* 对流项按焓通量处理
* 热流项包含：

  * 导热
  * 物种扩散焓通量
* 界面潜热与界面能量跳跃保持一致

### 11.3 界面 `Ts` 的残差位置

* `Ts` 是显式界面 unknown
* 对应残差为 `R_if,E`
* 对应控制方程为 Eq.(2.17)

---

## 12. 速度与连续方程策略

### 12.1 正式结论

* `u_l`
* `u_g`

都**不是主未知量**

它们由离散连续方程恢复，并作为：

* 对流相对速度 `u-v_c`
* 界面速度
* 界面方程参数
* 半径更新输入
* diagnostics

使用。

### 12.2 与界面块的一致性要求

当 Eq.(2.18) 用作 `mpp` 残差时：

* `u_l|if`
* `u_g|if`

都必须来自**当前 fixed-geometry inner trial state** 的一致界面构造：

- `u_l|if` 由 liquid continuity recovery 得到
- `u_g|if` 由 gas-side Eq.(2.18) 的边界强施加关系构造，并作为 gas continuity sweep 的起点

不允许：

* 用 Stefan 经验场代替
* 用旧时间层或旧 outer iteration 的速度
* 用和当前 `v_c` 不一致的恢复逻辑

---

## 13. `mpp` 与 Eq.(2.18) 的正式约束

### 13.1 正式结论

`mpp` 的主残差方程为 **liquid-side Eq.(2.18)**。

### 13.2 `Eq.(2.18)` 在内层 residual 中的使用方式

若 Eq.(2.18) 进入 inner residual，则其中出现的 `a_dot` / `\dot a` 必须取：

[
\dot a = \dot a^{(k)} = v_{c,if}^{(k)}
]

即当前 outer iteration 所定义的固定几何 / 固定网格问题中的冻结界面控制面速度。

同时，gas-side Eq.(2.18) 必须通过界面边界质量流率被强施加：

[
G_{g,if}^{abs}
=
(\rho_{s,g}\dot a^{(k)}-mpp)A_{if}
]

### 13.3 禁止事项

在 inner Newton / SNES 迭代过程中，禁止：

1. 用 trial `mpp` 现算 `\dot a_trial`
2. 再把新的 `\dot a_trial` 塞回当前 residual
3. 以此边解物理场边改几何参数

这会破坏 fixed-geometry residual 的定义。

### 13.4 `mpp` 与 `Rd` 的层级区分

* `mpp`：内层界面物理 unknown
* `a_dot`：外层几何更新量
* `Rd`：外层几何状态量

三者相关，但不属于同一求解层级。

---

## 14. Regime 策略：是否更换界面方程结构

### 14.1 正式结论

首版只允许**一套**界面方程结构：

* Eq.(2.15)
* Eq.(2.16)
* Eq.(2.17)
* Eq.(2.18)
* Eq.(2.19)

在运行过程中不允许切换成另一套“sat/boil 替代系统”。

### 14.2 允许的保护方式

允许的保护方式包括：

* line search
* trial-step reject
* timestep reject
* bounds / feasibility check
* 外层 accept / reject

但这些保护只能：

* 控制步长
* 控制可行域
* 控制接受与回退

**不能替换方程系统本身。**

---

## 15. Unknown scaling 策略

### 15.1 正式结论

`paper_v1` 首版**不做显式 unknown scaling**。

### 15.2 具体规定

* 解向量直接使用物理量原始单位
* 不在 layout 中维护额外 scaling vector
* 不在 state 中维护额外 scaling vector
* 不在 residual 层显式进行 unknown scaling / unscaling

### 15.3 后续升级限制

若未来需要重新引入 scaling，必须：

* 作为单独版本升级
* 单独编写方法文档
* 不允许局部模块偷偷引入

---

## 16. 分阶段未知量版本

为了避免以后改来改去把自己绕晕，正式定义三个阶段。

### 16.1 阶段 A：首个单组分乙醇基线

```text
U_A =
[
  liquid bulk:
      T_l
  interface:
      T_s,
      Y_s,g,1 ... Y_s,g,Ng_eff,
      mpp
  gas bulk:
      T_g, Y_g,1 ... Y_g,Ng_eff
]
```

适用：

* 单组分乙醇液滴
* inert / no chemistry in transport residual
* no radiation
* moving mesh + predictor-corrector

### 16.2 阶段 B：多组分液滴 transport

```text
U_B =
[
  liquid bulk:
      T_l, Y_l,1 ... Y_l,Nl_eff
  interface:
      Y_s,l,1 ... Y_s,l,Nl_eff,
      T_s,
      Y_s,g,1 ... Y_s,g,Ng_eff,
      mpp
  gas bulk:
      T_g, Y_g,1 ... Y_g,Ng_eff
]
```

适用：

* 多组分液滴
* 多组分液相扩散
* 更完整的界面组分守恒与相平衡

### 16.3 阶段 C：字面更贴近论文的显式密度版（可选）

```text
U_C =
[
  liquid bulk:
      T_l, Y_l,1 ... Y_l,Nl_eff
  interface:
      rho_s,l,
      Y_s,l,1 ... Y_s,l,Nl_eff,
      T_s,
      rho_s,g,
      Y_s,g,1 ... Y_s,g,Ng_eff,
      mpp
  gas bulk:
      T_g, Y_g,1 ... Y_g,Ng_eff
]
```

说明：

* 这条路线仅用于未来研究“字面完全贴近论文 block”的版本
* 不属于 `paper_v1` 首版与推荐最终版
* 只有在证明显式保留界面密度确有必要时才允许采用

---

## 17. 正式禁止事项

以下做法在新项目中明确禁止：

1. 把界面正式主线重新写回“仅 `Ts, mpp` 两未知量”
2. 把 `mpp` 保留为独立界面 unknown，却不为它提供 Eq.(2.18) 残差位置
3. 在首版里同时把 `Rd`、`u_l`、`u_g` 塞进内层主系统
4. 一边说“外层几何更新”，一边又在内层 layout 里偷偷保留 `Rd`
5. 一边说“速度由连续方程恢复”，一边又把 `u` 作为主未知量
6. 一边说“能量按焓一致形式处理”，一边又把残差退化为简化温度扩散方程
7. 在 closure species 上使用任何默认选项
8. 在单组分阶段删除 `Yl_full` 或 `Ys_l_full`
9. 让 `equilibrium_model` 同时包办可凝蒸汽与背景气界面组成
10. 在主框架中使用比例补偿式背景气算法
11. 通过 sat/boil regime 动态替换界面方程系统
12. 在某个局部模块偷偷恢复 unknown scaling 逻辑
13. 在 inner Newton 过程中实时更新 `a_dot`

---

## 18. 正式方法合同

### Unknowns Strategy Contract

1. `Rd` is not an inner transport unknown in `paper_v1`.
2. `u_l` and `u_g` are not primary unknowns.
3. `paper_v1` adopts a paper-like interface block as the mainline strategy.
4. First-release single-component interface block includes `T_s`, `Y_s,g,red`, and `mpp`.
5. Multi-component interface block includes `Y_s,l,red`, `T_s`, `Y_s,g,red`, and `mpp`.
6. Interface densities are derived from EOS / property closure and are not first-release explicit unknowns.
7. Bulk algebraic energy unknowns use temperature `T`.
8. Energy residuals must remain enthalpy-consistent.
9. Each phase removes one closure species and reconstructs it from `sum(Y)=1`.
10. Closure species must be explicitly configured; no default closure species is allowed.
11. `paper_v1` keeps `Yl_full`, `Yg_full`, and where applicable `Ys_l_full`, `Ys_g_full` as explicit state fields.
12. `mpp` is an independent interface unknown and its primary residual is the liquid-side form of Eq.(2.18).
13. Gas-side Eq.(2.18) is strongly imposed through the interface boundary mass-flux construction, not added as a second independent `mpp` residual row.
14. `a_dot` is not an inner unknown; if Eq.(2.18) is used in the inner residual, `a_dot` must be the frozen outer-iteration value.
15. `paper_v1` uses one fixed interface equation structure; no regime-based equation replacement is allowed.
16. `paper_v1` Phase A does not use explicit unknown scaling.
17. Unknown layout must remain consistent with the outer-geometry / inner-transport split.
18. Any future extension to explicit interface densities, monolithic `Rd`, or explicit scaling requires a new formal guideline, not an ad-hoc patch.

---

## 19. 最终定稿版结论

`paper_v1` 的未知量策略正式定为：

### 首版基线

```text
U_A =
[
  liquid bulk:
      T_l
  interface:
      T_s,
      Y_s,g,1 ... Y_s,g,Ng_eff,
      mpp
  gas bulk:
      T_g, Y_g,1 ... Y_g,Ng_eff
]
```

### 多组分液滴升级版

```text
U_B =
[
  liquid bulk:
      T_l, Y_l,1 ... Y_l,Nl_eff
  interface:
      Y_s,l,1 ... Y_s,l,Nl_eff,
      T_s,
      Y_s,g,1 ... Y_s,g,Ng_eff,
      mpp
  gas bulk:
      T_g, Y_g,1 ... Y_g,Ng_eff
]
```

并且以下量始终不属于 `paper_v1` 首版与推荐最终版的内层主未知量：

```text
{ Rd, u_l, u_g, a_dot, undefined y_if_aux... }
```

同时，以下附加策略已被正式写死：

```text
- global constant pressure P_inf
- closure species must be explicitly configured
- single-component liquid still keeps Yl_full = [1.0]
- gas state always keeps Yg_full
- paper-like interface block is the official mainline
- interface densities are EOS-derived, not first-release explicit unknowns
- mpp uses liquid-side Eq.(2.18) as its primary residual
- gas-side Eq.(2.18) is strongly imposed through the interface boundary mass-flux construction
- no regime-based equation replacement
- no explicit unknown scaling in Phase A
- Rd stays in the outer predictor-corrector layer
```

这就是新代码项目中“未知量策略”的修订完善后最终定稿版本。
