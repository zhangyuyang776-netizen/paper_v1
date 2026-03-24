# remap_and_conservative_projection_guideline_final

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定网格更新后的变量转移方法，即：

1. 什么是 remap / projection
2. 为什么 `paper_v1` 需要 conservative remap
3. remap 应该对哪些量执行
4. 1D 球对称下的 conservative remap 具体计算公式
5. 液相、气相、界面附近薄层分别如何处理
6. remap 后如何恢复 state variables
7. remap 在 outer predictor-corrector / inner fixed-geometry solve 中的职责边界

本文档是 `paper_v1` 中**网格更新后守恒变量转移**的正式定稿版本。

---

## 2. 文献与 `paper_v1` 的立场

### 2.1 文献明确给出的内容

主论文明确给出以下流程：

1. 已知新变量值
2. 根据
   \[
   \dot a = \frac{\dot m''}{\rho_l} + (u_l)_{r=a}
   \]
   更新界面位置
3. 用节点速度更新网格
4. 变量在新网格上再次计算
5. 重复 predictor-corrector，直到界面速度收敛

### 2.2 文献未明确展开的内容

在当前已给出的章节中，主论文**没有明确写出**：

- 变量在新网格上“再次计算”时究竟采用哪一种 remap / projection 方法
- 是否是 conservative remap
- 是否是简单插值
- 是否带高阶限制器
- 是否带守恒修正
- 对完全无旧同相 overlap 贡献的界面新暴露子体积，如何给出初始化闭合

### 2.3 `paper_v1` 的正式选择

虽然文献没有把 remap 算法逐字展开，但 `paper_v1` 正式规定：

> **第一版与推荐最终版都采用一阶、分相、cell-average conservative remap。**

理由：

1. `paper_v1` 的相内方程和界面方程都以守恒形式组织
2. inner residual 在固定当前几何上比较的是：
   - 当前几何上的 new state
   - 当前几何上的 old-equivalent state
3. 若 remap 不守恒，会将 remap 误差混入：
   - 时间项
   - 质量闭合
   - 组分闭合
   - 焓量闭合
4. `PAPER_V1_ROADMAP.md` 已经将 conservative remap 作为首版推荐方案

### 2.4 `paper_v1` 对界面新暴露子体积的工程化闭合

对于“新几何中属于某相、但旧几何中无同相 overlap 贡献”的界面相邻新增体积，主论文未给出可直接照抄的初始化公式。  
因此，`paper_v1` 在保持文献 moving-mesh 主线不变的前提下，增加如下工程化闭合：

> **界面新暴露子体积不作为独立网格层长期存储，而作为界面附近控制体的新增子体积；其守恒量由旧已接受时间层的相别一致界面状态构造后补入新控制体。**

这一规则属于 `paper_v1` 的工程化数值实现细则，不宣称为主论文逐字给出的原始算法。

---

## 3. remap 的基本思想

## 3.1 remap 不是“插值变量值”

在 `paper_v1` 中，remap 的主对象不是：

- `T`
- `Y`
- `u`

而是**守恒量的 cell content**。

也就是说，真正 remap 的是：

- 总质量：
  \[
  M = \rho V
  \]
- 组分质量：
  \[
  M_i = \rho Y_i V
  \]
- 总焓量：
  \[
  H = \rho h V
  \]

然后再从这些守恒量恢复：

- ho`
- `Y`
- `h`
- `T`

---

## 3.2 为什么要 remap 守恒量

若直接插值 primitive variables，例如 `T`、`Y`，则通常不能保证：

- 总质量守恒
- 每个组分的总质量守恒
- 总焓量守恒

而 conservative remap 的核心目标正是：

> 在网格更新后，尽量保持旧状态在新几何上的等价守恒量总和不变。

---

## 4. remap 的对象与范围

## 4.1 对每一相 separately 进行 remap

正式规定：

- 液相只在液相子域内 remap
- 气相只在气相子域内 remap
- **不允许跨相 remap**

也就是说：

- 旧液相内容不会直接 remap 到新气相 cell
- 旧气相内容不会直接 remap 到新液相 cell

跨相变化应由**界面质量通量与界面几何更新**负责，而不是由 remap 偷偷替代。

---

## 4.2 remap 的守恒量集合

### 液相
对液相 remap 以下量：

\[
Q_l = \left\{ \rho_l,\ \rho_l Y_{l,i},\ \rho_l h_l \right\}
\]

### 气相
对气相 remap 以下量：

\[
Q_g = \left\{ \rho_g,\ \rho_g Y_{g,i},\ \rho_g h_g \right\}
\]

### 说明
若某阶段某相没有组分自由度，例如单组分液滴首版液相，则该相的组分 remap 可以退化，但数据结构仍应保留。

---

## 4.3 与三段网格策略的配合

结合 `grid_partition_and_moving_mesh_guideline_final.md` 中的三段网格主线：

- 第一段：液相动网格
- 第二段：近界面气相动网格
- 第三段：远场气相固定网格

则 remap 范围正式规定为：

### 液相
- 对整个液相子域做 conservative remap

### 气相第二段
- 对第二段动网格区做 conservative remap

### 气相第三段
- 第三段网格固定，因此：
  - 原则上不需要几何 remap
  - 可以直接复用旧状态
  - 若实现层统一走 remap 接口，也必须保证该段等价于 identity mapping

---

## 5. 1D 球对称 conservative remap 的具体公式

## 5.1 几何定义

对旧网格第 `i` 个 cell：

\[
[r_{i-}^{old},\ r_{i+}^{old}]
\]

其体积为：

\[
V_i^{old}
=
\frac{4\pi}{3}
\left[
(r_{i+}^{old})^3-(r_{i-}^{old})^3
\right]
\]

对新网格第 `j` 个 cell：

\[
[r_{j-}^{new},\ r_{j+}^{new}]
\]

其体积为：

\[
V_j^{new}
=
\frac{4\pi}{3}
\left[
(r_{j+}^{new})^3-(r_{j-}^{new})^3
\right]
\]

---

## 5.2 重叠区间

旧 cell `i` 与新 cell `j` 的重叠区间定义为：

\[
r_-^{ov} = \max(r_{i-}^{old},\ r_{j-}^{new})
\]

\[
r_+^{ov} = \min(r_{i+}^{old},\ r_{j+}^{new})
\]

若：

\[
r_+^{ov} > r_-^{ov}
\]

则有重叠；否则无重叠。

---

## 5.3 重叠体积

重叠体积定义为：

\[
V_{ij}^{ov}
=
\frac{4\pi}{3}
\left[
(r_+^{ov})^3-(r_-^{ov})^3
\right]
\]

若无重叠，则：

\[
V_{ij}^{ov}=0
\]

---

## 5.4 一阶 cell-average conservative remap

设旧网格 cell `i` 上守恒变量 `Q` 的 cell-average 为：

\[
\bar Q_i^{old}
\]

则新网格 cell `j` 上的 old-equivalent cell content 定义为：

\[
(QV)_j^{old,*}
=
\sum_i \bar Q_i^{old} V_{ij}^{ov}
\]

再除以新 cell 体积，得到新几何上的旧 cell-average：

\[
\bar Q_j^{old,*}
=
\frac{(QV)_j^{old,*}}{V_j^{new}}
\]

---

## 5.5 对各类守恒量分别写出公式

### 总质量

\[
(\rho V)_j^{old,*}
=
\sum_i \bar \rho_i^{old} V_{ij}^{ov}
\]

\[
\bar \rho_j^{old,*}
=
\frac{(\rho V)_j^{old,*}}{V_j^{new}}
\]

### 组分质量

对每个 reduced species `k`：

\[
((\rho Y_k)V)_j^{old,*}
=
\sum_i \overline{(\rho Y_k)}_i^{old} V_{ij}^{ov}
\]

\[
\overline{(\rho Y_k)}_j^{old,*}
=
\frac{((\rho Y_k)V)_j^{old,*}}{V_j^{new}}
\]

### 总焓量

\[
((\rho h)V)_j^{old,*}
=
\sum_i \overline{(\rho h)}_i^{old} V_{ij}^{ov}
\]

\[
\overline{(\rho h)}_j^{old,*}
=
\frac{((\rho h)V)_j^{old,*}}{V_j^{new}}
\]

---

## 6. remap 后的状态恢复

remap 完成后，得到的是新几何上的守恒量：

- ho`
- ho Y`
- ho h`

而 inner residual 组装需要的往往是：

- ho`
- `Y`
- `h`
- `T`

因此必须做状态恢复。

---

## 6.1 组分恢复

对每个 reduced species：

\[
Y_{k,j}^{old,*}
=
\frac{(\rho Y_k)_j^{old,*}}{\rho_j^{old,*}}
\]

然后由闭合组分策略恢复 closure species：

\[
Y_{cl,j}^{old,*}
=
1-\sum_{k\neq cl} Y_{k,j}^{old,*}
\]

### 规定
- 若由于数值误差出现轻微负值或略大于 1 的值，允许在 post-correction 中做轻微修正
- 但不允许在 remap 主体中直接“无条件裁剪”，以免破坏守恒性

---

## 6.2 焓恢复

\[
h_j^{old,*}
=
\frac{(\rho h)_j^{old,*}}{\rho_j^{old,*}}
\]

---

## 6.3 温度恢复

由相应物性模型求反函数：

### 液相
\[
T_{l,j}^{old,*}
=
T_l\big(h_{l,j}^{old,*},\ Y_{l,j}^{old,*}\big)
\]

### 气相
\[
T_{g,j}^{old,*}
=
T_g\big(h_{g,j}^{old,*},\ Y_{g,j}^{old,*},\ P_\infty\big)
\]

### 规定
- 液相温度反解由液相焓模型负责
- 气相温度反解由 Cantera 气相焓状态负责
- 温度恢复必须与当前物性主线保持一致，不能另走一套经验关系

---

## 7. 界面附近薄层的特殊处理

这是 conservative remap 中最容易写错的部分。

## 7.1 问题来源

当界面从 `a^n` 更新到 `a^(k)` 时，相域本身会改变。

若液滴缩小，则：

\[
a^{(k)} < a^n
\]

于是旧液相中一部分靠近界面的区域：

\[
[a^{(k)},\ a^n]
\]

在新几何中已经属于气相。

反之，若某些特殊情况下界面外移，也可能出现相反问题。

---

## 7.2 正式原则

### 不允许跨相 remap
即：

- 旧液相内容不能直接 remap 到新气相
- 旧气相内容不能直接 remap 到新液相

这是正式硬约束。

### 原因
跨相变化应由以下物理过程负责：

- 界面质量通量 `mpp`
- 界面组分守恒
- 界面能量守恒
- 几何更新 `a(t)`

若用 remap 跨相“偷运”质量或焓，会使：
- 相变通量与 remap 混淆
- 界面残差失去明确物理意义
- 质量/能量闭合无法审查

---

## 7.3 新暴露 / 新覆盖薄层如何处理

对由于界面移动而出现的新气相薄层或新液相薄层，不作为普通 overlap remap 的对象。

### 正式做法
- 先通过界面一侧状态做初始化
- 再由 inner nonlinear solve 将该区域校正到与 bulk 方程和界面方程一致

### 规定澄清
这里所说的“薄层”不应理解为一个需要长期存储、单独增加自由度的独立网格层。  
在 `paper_v1` 的 1D cell-centered finite-volume 框架下，它更准确地应理解为：

> **界面相邻控制体中的新暴露子体积（newly exposed subcell volume）**

也就是说：

- 它通常只是界面附近第一个气相控制体或最后一个液相控制体中的一部分体积
- 它不作为独立 cell 长期存储
- 它的作用是帮助构造新控制体的守恒量初值
- 最终仍由 inner solve 在完整控制体上统一校正

---

## 7.4 界面相邻新暴露子体积的守恒补全规则

本节是 `paper_v1` 在主论文未明确给出时采用的正式工程化闭合规则。

### 7.4.1 概念

设某新几何控制体 `j` 的体积可分为两部分：

1. 与旧同相控制体有 overlap 的体积：
   \[
   V_{j,\mathrm{ov}}
   \]
2. 在旧几何中无同相 overlap 的界面相邻新暴露子体积：
   \[
   \Delta V_{j,\mathrm{exp}}
   \]

则新控制体的 old-equivalent 守恒 content 不仅来自普通 overlap remap，还必须加上这部分新暴露子体积的守恒补全。

### 7.4.2 状态来源原则

对完全无旧同相 overlap 贡献的新暴露子体积，不采用：

- 远场状态
- 邻近 bulk cell-average
- 跨相 remap
- 当前 inner/outer 迭代中的新状态

而正式采用：

> **旧已接受时间层的、相别一致的界面状态**

也就是说：

#### 新暴露到气相侧
使用旧已接受时间层的界面气相状态：

\[
T_s^n,\qquad Y_{s,g}^n
\]

再通过当前气相物性主线得到：

\[
\rho_{s,g}^n,\qquad h_{s,g}^n
\]

#### 新暴露到液相侧
若 outer 修正导致局部液相新暴露，则使用旧已接受时间层的界面液相状态：

\[
T_s^n,\qquad Y_{s,l}^n
\]

再通过当前液相物性主线得到：

\[
\rho_{s,l}^n,\qquad h_{s,l}^n
\]

### 7.4.3 为什么采用界面状态而不是相邻 cell-average

正式理由如下：

1. 新暴露子体积在几何上紧邻界面，厚度趋于 0 时，其极限状态应更接近界面状态，而不是距界面仍有一个有限控制体尺度距离的 cell-average。
2. `paper_v1` 当前已显式引入了界面变量块，因此界面状态本身就是系统中的单独物理对象，而不是隐含边界值。
3. 采用界面状态补全，有助于使界面相邻控制体的初值与 interface residual 的主未知量保持一致。

### 7.4.4 补全对象

对子体积进行补全时，正式补的不是 primitive variables，而是守恒量内容：

- 总质量：
  \[
  (\rho \Delta V)_{\mathrm{exp}}
  \]
- 组分质量：
  \[
  (\rho Y_i \Delta V)_{\mathrm{exp}}
  \]
- 总焓量：
  \[
  (\rho h \Delta V)_{\mathrm{exp}}
  \]

即：

#### 新暴露到气相侧
\[
(\rho \Delta V)_{\mathrm{exp},g}
=
\rho_{s,g}^n \,\Delta V_{j,\mathrm{exp}}
\]

\[
(\rho Y_i \Delta V)_{\mathrm{exp},g}
=
\rho_{s,g}^n Y_{s,g,i}^n \,\Delta V_{j,\mathrm{exp}}
\]

\[
(\rho h \Delta V)_{\mathrm{exp},g}
=
\rho_{s,g}^n h_{s,g}^n \,\Delta V_{j,\mathrm{exp}}
\]

#### 新暴露到液相侧
\[
(\rho \Delta V)_{\mathrm{exp},l}
=
\rho_{s,l}^n \,\Delta V_{j,\mathrm{exp}}
\]

\[
(\rho Y_i \Delta V)_{\mathrm{exp},l}
=
\rho_{s,l}^n Y_{s,l,i}^n \,\Delta V_{j,\mathrm{exp}}
\]

\[
(\rho h \Delta V)_{\mathrm{exp},l}
=
\rho_{s,l}^n h_{s,l}^n \,\Delta V_{j,\mathrm{exp}}
\]

### 7.4.5 与普通 overlap remap 的组合方式

新控制体 `j` 的 old-equivalent 守恒量，应按下面方式构造：

#### 总质量
\[
(\rho V)_j^{old,*}
=
(\rho V)_{j,\mathrm{ov}}^{old,*}
+
(\rho \Delta V)_{j,\mathrm{exp}}
\]

#### 组分质量
\[
((\rho Y_i)V)_j^{old,*}
=
((\rho Y_i)V)_{j,\mathrm{ov}}^{old,*}
+
(\rho Y_i \Delta V)_{j,\mathrm{exp}}
\]

#### 总焓量
\[
((\rho h)V)_j^{old,*}
=
((\rho h)V)_{j,\mathrm{ov}}^{old,*}
+
(\rho h \Delta V)_{j,\mathrm{exp}}
\]

然后再除以新控制体体积 `V_j^{new}`，得到新的 old-equivalent cell-average，并继续执行状态恢复。

### 7.4.6 新覆盖体积的处理

若某一相在新几何中失去一部分原体积，则这部分属于**新覆盖体积**。  
正式规定：

- 新覆盖体积**不做初始化**
- 而是直接从该相控制体中删去
- 因此在几何更新意义下：
  - **新暴露体积需要“补”**
  - **新覆盖体积需要“删”**

### 7.4.7 与 inner solve 的职责分工

这一守恒补全只负责：

- 给出新几何下界面相邻控制体的 old-equivalent 初值
- 保证时间项、守恒量和状态恢复可继续进行

它**不是**最终物理解。  
最终仍由 inner nonlinear solve 在当前几何下把该控制体校正到与：

- bulk transport residual
- interface residual
- face flux package

一致的状态。

---

## 7.5 post-correction 的角色

若 remap 后在界面附近出现：

- 明显非物理的 `Y`
- 明显不一致的 `T`
- 由 thin-layer / newly exposed subvolume 初始化引入的小幅不连续

允许执行局部 post-correction。

### post-correction 的作用
- 轻微修正界面附近状态
- 改善 inner solve 初值
- 不改变主守恒 remap 的基本结构

### 规定
- post-correction 只能作为局部修正
- 不能替代 conservative remap 主算法
- 不能偷偷承担跨相转移职责

---

## 8. remap 的守恒性检查

每轮 remap 后，推荐至少检查以下守恒量。

## 8.1 相内总质量

### 液相
\[
M_l^{old}
=
\sum_i \rho_{l,i}^{old}V_i^{old}
\]

\[
M_l^{old,*}
=
\sum_j \rho_{l,j}^{old,*}V_j^{new}
\]

检查误差：

\[
\epsilon_{M_l}
=
\frac{|M_l^{old,*}-M_l^{old}|}{\max(M_l^{old},\epsilon)}
\]

### 气相
同理定义 `epsilon_M_g`。

---

## 8.2 相内组分质量

对每个 species `k`：

\[
M_{k}^{old}
=
\sum_i (\rho Y_k)_i^{old}V_i^{old}
\]

\[
M_{k}^{old,*}
=
\sum_j (\rho Y_k)_j^{old,*}V_j^{new}
\]

检查：

\[
\epsilon_{M_k}
=
\frac{|M_k^{old,*}-M_k^{old}|}{\max(M_k^{old},\epsilon)}
\]

---

## 8.3 相内总焓量

### 液相
\[
H_l^{old}
=
\sum_i (\rho h)_i^{old}V_i^{old}
\]

\[
H_l^{old,*}
=
\sum_j (\rho h)_j^{old,*}V_j^{new}
\]

同理定义相对误差。

### 气相
同理。

---

## 8.4 正式要求

首版至少应输出以下 diagnostics：

- emap_mass_err_liquid`
- emap_mass_err_gas`
- emap_species_err_max`
- emap_enthalpy_err_liquid`
- emap_enthalpy_err_gas`

这样才能在 outer/inner 循环出问题时，第一时间判断是不是 remap 在作恶。

---

## 9. remap 与 outer/inner 两层结构的关系

## 9.1 外层负责什么

outer module 负责：

1. 根据 `a^(k)` 构建当前网格
2. 将旧状态 remap 到当前几何
3. 形成：
   \[
   U^{old,*\,(k)}
   \]
4. 将其作为 inner solve 的时间层旧状态输入

---

## 9.2 内层负责什么

inner `SNES` / Newton solve 只负责：

- 在当前 fixed geometry 上解 transport nonlinear system
- 使用当前轮已经准备好的 current-geometry old-state data；`k=0` 时可严格退化为 identity，`k>0` 时来自上一轮 outer 状态转移
- 不负责重建网格
- 不负责 remap
- 不负责几何 accept / reject

---

## 9.3 remap 输出的正式对象

remap / projection 模块输出：

### 液相
- ho_l_old_star`
- hoY_l_old_star`
- hoh_l_old_star`
- `T_l_old_star`
- `Y_l_old_star`
- `h_l_old_star`

### 气相
- ho_g_old_star`
- hoY_g_old_star`
- hoh_g_old_star`
- `T_g_old_star`
- `Y_g_old_star`
- `h_g_old_star`

这些量共同构成：

\[
old\_state\_on\_current\_geometry
\]

供 inner solve 的时间项和速度恢复使用。

---

## 10. 推荐算法流程

下面给出正式推荐的 remap 算法流程。

### Step 1
由 outer iterate 给定当前几何和当前网格：

- `a^(k)`
- `G^(k)`

### Step 2
分别识别：

- 旧液相子域与新液相子域
- 旧气相第二段与新气相第二段
- 固定第三段

### Step 3
对液相执行一阶 conservative overlap remap：
- ho`
- hoY`
- hoh`

### Step 4
对气相第二段执行一阶 conservative overlap remap：
- ho`
- hoY`
- hoh`

### Step 5
对气相第三段直接拷贝旧状态

### Step 6
对界面相邻新暴露子体积做守恒补全，并对新覆盖体积执行几何删去

### Step 7
从 remapped conservative variables 恢复：
- `Y`
- `h`
- `T`

### Step 8
做 closure species 回补

### Step 9
检查守恒误差

### Step 10
必要时做局部 post-correction

### Step 11
形成下一轮 inner 入口所需的 current-geometry transfer state；它在 outer 未收敛并更新网格后使用，而不是每轮 inner 前都从 `U^n` 重新现做一次

---

## 11. 可选升级路线

首版主线是一阶 conservative remap。  
后续若需要升级，可考虑：

### 11.1 二阶 conservative remap
在旧 cell 中引入 slope，做 piecewise linear reconstruction，再与 overlap volume 结合。

### 11.2 bounded conservative remap
在保证守恒的同时加入 limiter，降低 `Y<0` 风险。

### 11.3 界面附近专用 subcell remap
专门提高新暴露/新覆盖子体积初始化质量。

### 规定
这些都属于后续升级路线，不属于 `paper_v1` 首版主线。

---

## 12. 正式禁止事项

以下做法在本文件下明确禁止：

1. 用普通节点/中心值插值替代守恒 remap 作为主线实现
2. 对 `T`、`Y` 直接插值后不检查质量与焓守恒
3. 在液相和气相之间做跨相 conservative remap
4. 把界面新暴露薄层也按普通 overlap remap 处理
5. 在第三段固定网格区仍然做无意义的几何 remap
6. 省略 remap 后的守恒误差检查
7. 让 post-correction 替代主 remap 算法
8. 在 inner solve 中修改状态转移产物
9. 把界面新暴露子体积当作独立长期网格层来存储
10. 用当前 inner/outer 迭代中的新界面状态去补 old-state remap

---

## 13. 最终方法合同

### Remap and Conservative Projection Contract

1. `paper_v1` uses first-order, phase-wise, cell-average conservative remap as the mainline strategy.
2. Remap is performed on conservative cell contents, not directly on primitive variables.
3. The remapped quantities are:
   - ho`
   - hoY`
   - hoh`
4. Liquid and gas are remapped separately; no cross-phase remap is allowed.
5. Region-3 fixed gas grid is not geometrically remapped.
6. Newly uncovered / newly covered interface-adjacent thin layers are handled by dedicated conservative completion / geometric deletion, not by ordinary overlap remap.
7. Interface-adjacent newly exposed subcell volume is not stored as an independent persistent mesh layer; it is treated as a temporary subvolume used to complete the conservative content of the interface-nearest control volume.
8. The newly exposed subcell volume must be completed using the phase-consistent interface state from the old accepted time level, not from neighboring cell-average values and not from current iterative new states.
9. The only valid old-state input for the inner fixed-geometry residual is the current-geometry transfer state already prepared for that outer iterate.
10. Remap must preserve phase-wise total mass, species mass, and enthalpy up to numerical tolerance.
11. Post-correction is optional and local; it cannot replace the conservative remap core.
12. Higher-order remap is a future upgrade path, not part of the first-release mainline.

---

## 14. 最终结论

`paper_v1` 的网格更新后变量转移正式定为：

- **一阶**
- **分相**
- **cell-average conservative remap**
- **第三段固定网格直接复用旧状态**
- **界面新暴露子体积按旧已接受时间层的相别一致界面状态进行守恒补全**
- **新覆盖体积执行几何删去**
- **形成供下一轮 inner 使用的 current-geometry transfer state**

这就是 `paper_v1` 的 remap 与 conservative projection 指导文件最终定稿版本。
