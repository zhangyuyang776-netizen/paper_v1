# governing_equations_discretization_and_bc_guideline_final_v2

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定以下内容：

1. 采用哪些物理控制方程  
2. bulk 与 interface 的未知量、残差方程、控制方程对应关系  
3. 空间离散格式  
4. 时间离散格式  
5. 特殊控制面（中心、界面、远场）的离散闭合  
6. 界面方程的离散与 bulk 方程的严格对接方式  
7. 内层 fixed-geometry nonlinear solve 与外层 predictor-corrector 的整体计算框架  

本文档是 `paper_v1` 的**物理控制 + 数值离散 + 边界处理 + 特殊控制面闭合**联合指导文件。

---

## 2. 适用范围与总原则

### 2.1 适用范围

本文档适用于 `paper_v1` 的以下路线：

- 1D 球对称液滴问题
- moving control volume / moving mesh
- 外层 predictor-corrector 更新界面位置和网格
- 内层在固定当前几何上求解液相 bulk + 界面块 + 气相 bulk 的耦合非线性系统
- transport 主系统不含化学源项
- 辐射在 `paper_v1` 中关闭
- 全域恒压 `P_inf`

### 2.2 总原则

1. **相内方程按守恒形式写，按有限体积离散。**
2. **界面块不是普通体积单元，而是零厚度界面控制体极限下的耦合块。**
3. **bulk + interface 在内层一起耦合求解。**
4. **`Rd/a(t)` 不进入内层主未知量，由外层 predictor-corrector 更新。**
5. **`mpp` 是独立界面 unknown；Eq.(2.18) 在内层实现中采用“液相侧主残差 + 气相侧界面边界强施加”的双侧耦合方式。**
6. **对流项统一使用相对控制面速度 `u - v_c`。**
7. **时间离散统一采用 backward Euler。**
8. **首版不做显式 unknown scaling。**
9. **首版 transport residual 中不含化学源项；辐射项置零，但不更换界面能量方程结构。**
10. **界面物理只能有一个真相源（single source of truth），不允许 bulk 与 interface 各自独立发明界面通量。**
11. **特殊控制面不是“给点值”，而是“给面通量闭合规则”。**

---

## 3. 物理控制方程

## 3.1 相内守恒变量

对每一相 `β ∈ {l, g}`，采用如下守恒变量表示：

\[
\phi_\beta
=
\left\{
\rho_\beta,\;
\rho_\beta Y_{\beta,i},\;
\rho_\beta h_\beta
\right\}
\]

其中：

- `ρβ`：相密度
- `Yβ,i`：质量分数
- `hβ`：混合物比焓

### 3.1.1 moving control volume 守恒形式

对任意控制体 `n`，采用统一守恒形式：

\[
\frac{d}{dt}\left(\Phi_{\beta,n}V_n\right)
+
\left[\phi_\beta (u_\beta-v_c)A\right]_{n-}^{n+}
=
-
\left[F(\phi_\beta)A\right]_{n-}^{n+}
+
P(\Phi_{\beta,n})V_n
\]

其中：

- `V_n`：控制体体积
- `A`：控制面面积
- `uβ`：流体绝对速度
- `v_c`：控制面速度
- `F(φβ)`：扩散/导热通量算子
- `P(Φβ,n)`：源项

在 `paper_v1` Phase A 中：

- 气相/液相 transport residual 内的化学源项取零
- 纯蒸发 / inert 背景情形中，源项项 `P=0`
- 若未来升级到 chemistry splitting，则 chemistry source 不进入本 transport residual

---

## 3.2 液相控制方程

### 3.2.1 液相连续方程

液相连续方程用于恢复速度，不作为独立主未知量方程进入 layout：

\[
\frac{\partial \rho_l}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_l u_l\right)
= 0
\]

### 3.2.2 液相组分方程

对液相 reduced species：

\[
\frac{\partial (\rho_l Y_{l,i})}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_l Y_{l,i} u_l\right)
=
-
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 J_{l,i}\right)
\]

### 3.2.3 液相能量方程

\[
\frac{\partial (\rho_l h_l)}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_l h_l u_l\right)
=
-
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 q_l\right)
\]

液相热流采用 generalized Fourier’s law：

\[
q_l = -k_l \frac{\partial T_l}{\partial r} + \sum_i J_{l,i} h_{l,i}
\]

### 3.2.4 液相扩散通量模型

液相物种扩散通量保持主参考文献 Fick 形式：

\[
J_{l,i}=-\rho_l D_{l,i}\frac{\partial Y_{l,i}}{\partial r}
\]

---

## 3.3 气相控制方程

### 3.3.1 气相连续方程

气相连续方程用于恢复速度，不作为独立主未知量方程进入 layout：

\[
\frac{\partial \rho_g}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_g u_g\right)
= 0
\]

### 3.3.2 气相组分方程

对气相 reduced species：

\[
\frac{\partial (\rho_g Y_{g,i})}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_g Y_{g,i} u_g\right)
=
-
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 J_{g,i}\right)
\]

在 `paper_v1` 中，transport residual 内不包含化学源项。

### 3.3.3 气相能量方程

\[
\frac{\partial (\rho_g h_g)}{\partial t}
+
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 \rho_g h_g u_g\right)
=
-
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2 q_g\right)
\]

气相热流同样采用 generalized Fourier’s law：

\[
q_g = -k_g \frac{\partial T_g}{\partial r} + \sum_i J_{g,i} h_{g,i}
\]

### 3.3.4 气相扩散通量模型

气相物种扩散通量采用主参考文献 Eq.(2.4) 的 mixture-averaged model with conservative flux correction：

\[
J_{g,i}=-\rho_gY_{g,i}\left(V^0_{d,i}+V_{cd}\right)
\]

其中

\[
V^0_{d,i}= -\frac{D_{g,i}}{X_{g,i}}\frac{\partial X_{g,i}}{\partial r}
\]

\[
V_{cd}= -\sum_{m=1}^{N_g} Y_{g,m}V^0_{d,m}
\]

### 3.3.5 说明

1. 气相扩散通量**不是**简单的
   \[
   -\rho_g D_i \frac{\partial Y_i}{\partial r}
   \]
2. conservative correction 的计算必须基于**全物种 full composition**
3. reduced species 只影响 unknown layout，不改变 face transport 的 full-species 计算需求

---

## 3.4 界面控制方程

界面位于 `r = a(t)`，界面块采用论文式界面变量块主线。

### 3.4.1 共有/可凝组分守恒：Eq.(2.15)

对气液两相共有的可凝/可溶组分：

\[
-\dot m''
\left(
Y_{s,g,i} - Y_{s,l,i}
\right)
=
-
\left(
J_{g,i} - J_{l,i}
\right)_{if}
\]

该方程对应液相侧界面组分 unknown `Y_s,l,i`。

### 3.4.2 非凝组分守恒：Eq.(2.16)

对只存在于气相的非凝组分：

\[
-\dot m'' Y_{s,g,i}
=
-
J_{g,i}|_{if}
\]

该方程对应气相侧非凝组分 unknown `Y_s,g,i(noncond)`。

### 3.4.3 界面能量守恒：Eq.(2.17)

\[
-\dot m'' \sum_i Y_{s,l,i} L_i(T_s)
=
\left(
k_g \frac{\partial T_g}{\partial r}
-
k_l \frac{\partial T_l}{\partial r}
\right)_{if}
-\sum_i J_{l,i}|_{if} L_i(T_s)
\]

在 `paper_v1` 中：

- 辐射项显式关闭
- 但仍保留 Eq.(2.17) 的结构

该方程对应 `T_s`。

### 3.4.4 总质量通量定义：Eq.(2.18)

\[
\dot m''
=
-\rho_{s,l}(u_{l,if}-\dot a)
=
-\rho_{s,g}(u_{g,if}-\dot a)
\]

该方程对应 `mpp = \dot m''`。

在 `paper_v1` 的 inner fixed-geometry 实现中，Eq.(2.18) 的离散采用以下规则：

- `mpp` 只有一个标量 unknown，因此不在同一个 square inner system 中同时放入两条独立的 `mpp` 残差行
- liquid-side Eq.(2.18) 作为 `mpp` 的**主残差**
- gas-side Eq.(2.18) 不降为“仅供参考的物理概念”，而是通过界面面绝对质量流率的边界构造被**强施加**
- gas-side 仍输出一致性诊断，用于检查实现是否保持了同一个 interface face package

### 3.4.5 可凝蒸汽相平衡：Eq.(2.19)

可凝蒸汽界面组成由相平衡关系给出。保留通用记号：

\[
R_{eq,j}(Y_{s,g,j}^{cond}, Y_{s,l}, T_s, P_\infty, \gamma_j, \ldots)=0
\]

单组分首版中，这条方程退化为单组分饱和蒸汽平衡关系。

---

## 3.5 几何更新方程

界面位置 `a(t)` 不进入内层主未知量。  
内层状态收敛后，界面速度由：

\[
\dot a
=
u_{l,if}
+
\frac{\dot m''}{\rho_{s,l}}
\]

更新，然后外层重建网格并继续 predictor-corrector。

---

## 4. unknown — residual — controlling equation 对照

## 4.1 液相域

| 显式未知量 | 残差 | 控制/闭合方程 |
|---|---|---|
| `T_l,n` | `R_l^E(n)=0` | 液相能量方程 |
| `Y_l,n,i`（reduced） | `R_l^{Y_i}(n)=0` | 液相组分守恒方程 |
| `u_l`（非显式） | 无主残差 | 液相连续方程恢复 |
| `ρ_l`（非显式） | 无主残差 | 液相 EOS / 物性闭合 |

## 4.2 气相域

| 显式未知量 | 残差 | 控制/闭合方程 |
|---|---|---|
| `T_g,n` | `R_g^E(n)=0` | 气相能量方程 |
| `Y_g,n,i`（reduced） | `R_g^{Y_i}(n)=0` | 气相组分守恒方程 |
| `u_g`（非显式） | 无主残差 | 气相连续方程恢复 |
| `ρ_g`（非显式） | 无主残差 | 气相 EOS / 物性闭合 |

## 4.3 界面块

### 单组分液滴首版
\[
\phi_{s,A} = \{T_s,\;Y_{s,g,red},\;mpp\}
\]

### 多组分液滴升级版
\[
\phi_{s,B} = \{Y_{s,l,red},\;T_s,\;Y_{s,g,red},\;mpp\}
\]

| 显式未知量 | 残差 | 控制/闭合方程 |
|---|---|---|
| `Y_s,l,i`（reduced） | `R_if,l,i=0` | Eq.(2.15) |
| `Y_s,g,j(cond)` | `R_if,eq,j=0` | Eq.(2.19) |
| `Y_s,g,k(noncond)` | `R_if,g,k=0` | Eq.(2.16) |
| `T_s` | `R_if,E=0` | Eq.(2.17) / 等价总能量通量连续 |
| `mpp = \dot m''` | `R_if,m=0` | Eq.(2.18) |
| `ρ_s,l`, `ρ_s,g`（非显式） | 无主残差 | EOS / 物性闭合 |

---

## 5. 空间离散格式

## 5.1 网格与控制体

采用 1D 球对称 cell-centered finite volume。

- cell center：`r_n`
- left/right face：`r_{n-}, r_{n+}`
- face area：
  \[
  A_{n\pm} = 4\pi r_{n\pm}^2
  \]
- cell volume：
  \[
  V_n = \frac{4\pi}{3}\left(r_{n+}^3-r_{n-}^3\right)
  \]

界面 face 满足：

\[
r_{if} = a(t)
\]

---

## 5.2 对流项离散：一阶上风

对任意 face `f`，相对控制面速度为：

\[
u_f^{rel} = u_f - v_{c,f}
\]

对流通量中的 face state 采用一阶上风：

\[
\phi_f^{up}=
\begin{cases}
\Phi_L, & u_f^{rel} > 0 \\
\Phi_R, & u_f^{rel} < 0
\end{cases}
\]

于是对流项写为：

\[
\mathcal{C}_{n}^{(\phi)}
=
\left[\phi^{up}(u-v_c)A\right]_{n-}^{n+}
\]

注意：

- 所有对流通量都用 `u-v_c`
- 不允许把 moving mesh 下的对流项退化成单独 `u`

---

## 5.3 扩散项离散：内部 face 二阶中心

对普通内部 face `f=n+`，梯度采用中心差分：

\[
\left.\frac{\partial \psi}{\partial r}\right|_{n+}
=
\frac{\psi_{n+1}-\psi_n}{r_{n+1}-r_n}
\]

### 5.3.1 液相组分扩散通量

\[
J_{l,i,f}
=
-\rho_{l,f} D_{l,i,f}
\left.\frac{\partial Y_{l,i}}{\partial r}\right|_f
\]

### 5.3.2 气相组分扩散通量

\[
V^0_{d,i,f}
=
-\frac{D_{g,i,f}}{X_{g,i,f}}
\left.\frac{\partial X_{g,i}}{\partial r}\right|_f
\]

\[
V_{cd,f}
=
-\sum_{m=1}^{N_g}Y_{g,m,f}V^0_{d,m,f}
\]

\[
J_{g,i,f}
=
-\rho_{g,f}Y_{g,i,f}\left(V^0_{d,i,f}+V_{cd,f}\right)
\]

### 5.3.3 导热项

\[
q_{\beta,f}^{cond} = -k_{\beta,f} \left.\frac{\partial T_\beta}{\partial r}\right|_f
\]

### 5.3.4 总热流

\[
q_{\beta,f} = q_{\beta,f}^{cond} + \sum_i J_{\beta,i,f} h_{\beta,i,f}
\]

### 5.3.5 face 物性值

对普通内部 face，首版推荐简单平均：

\[
\chi_f = \frac{\chi_L+\chi_R}{2}
\]

其中 `χ` 可为：

- `ρ`
- `k`
- `D`
- `h_i`
- `Y_i`
- `X_i`

但对气相扩散通量修正项，必须基于 **full gas composition** 计算。

---

## 5.4 普通内部 face 与特殊 face 的区别

### 5.4.1 普通内部 face
普通内部 face 使用：

- 中心差分梯度
- 简单平均物性
- 上风对流取值

### 5.4.2 特殊 face
以下 face **不按普通内部 face 统一处理**：

1. 球心 face  
2. 界面 face  
3. far-field 外边界 face  

它们必须有单独的离散闭合规则。

---

## 6. 时间离散与 bulk 残差模板

## 6.1 backward Euler

时间离散统一采用 backward Euler。

对任意 bulk 控制变量 `φ`，在当前 outer iterate 的 fixed geometry 上，采用：

\[
R_{\beta,n}^{(\phi)}
=
\frac{\Phi_{\beta,n}^{n+1}V_n^{(k)} - \Phi_{\beta,n}^{old,*}V_n^{(k)}}{\Delta t}
+
\mathcal{C}_{\beta,n}^{(\phi)}
+
\mathcal{D}_{\beta,n}^{(\phi)}
-
\mathcal{S}_{\beta,n}^{(\phi)}
\]

其中：

- `old,*` 表示 **经过 conservative remap 后、定义在当前 fixed geometry `G^(k)` 上的 old-equivalent cell-average state**
- `\mathcal{C}`：对流项
- `\mathcal{D}`：扩散/导热项
- `\mathcal{S}`：源项（Phase A 通常为 0）

### 说明
这里不使用旧几何上的原始 cell mass，而统一使用：

\[
old\_state\_on\_current\_geometry
\]

因此，时间项中的旧量使用**当前几何控制体体积**，而不是旧几何体积。
换言之：

\[
\frac{\Phi^{n+1}V - \Phi^{old,*}V}{\Delta t}
\]

才是 `paper_v1` 在 fixed-geometry inner residual 中唯一允许的主写法。

---

## 6.2 液相 bulk 残差

### 液相能量残差
\[
R_l^E(n)
=
\frac{(\rho_l h_l)_n^{n+1}V_n^{(k)} - (\rho_l h_l)_n^{old,*}V_n^{(k)}}{\Delta t}
+
\left[(\rho_l h_l)^{up}(u_l-v_c)A\right]_{n-}^{n+}
+
\left[q_l A\right]_{n-}^{n+}
\]

### 液相组分残差
\[
R_l^{Y_i}(n)
=
\frac{(\rho_l Y_{l,i})_n^{n+1}V_n^{(k)} - (\rho_l Y_{l,i})_n^{old,*}V_n^{(k)}}{\Delta t}
+
\left[(\rho_l Y_{l,i})^{up}(u_l-v_c)A\right]_{n-}^{n+}
+
\left[J_{l,i}A\right]_{n-}^{n+}
\]

单组分液滴首版中，液相组分残差不存在。

---

## 6.3 气相 bulk 残差

### 气相能量残差
\[
R_g^E(n)
=
\frac{(\rho_g h_g)_n^{n+1}V_n^{(k)} - (\rho_g h_g)_n^{old,*}V_n^{(k)}}{\Delta t}
+
\left[(\rho_g h_g)^{up}(u_g-v_c)A\right]_{n-}^{n+}
+
\left[q_g A\right]_{n-}^{n+}
\]

### 气相组分残差
\[
R_g^{Y_i}(n)
=
\frac{(\rho_g Y_{g,i})_n^{n+1}V_n^{(k)} - (\rho_g Y_{g,i})_n^{old,*}V_n^{(k)}}{\Delta t}
+
\left[(\rho_g Y_{g,i})^{up}(u_g-v_c)A\right]_{n-}^{n+}
+
\left[J_{g,i}A\right]_{n-}^{n+}
\]

在 `paper_v1` 中：

- 不加化学源项
- 未来 chemistry splitting 也不改这里的残差结构

---

## 7. 连续方程离散与速度恢复

## 7.1 连续方程恢复主式

定义 face 绝对质量流率：

\[
G_f = (\rho u A)_f
\]

在当前 fixed geometry 上，连续方程离散写成递推形式：

\[
G_{n+}
=
G_{n-}
+
\left[(\rho v_c A)\right]_{n-}^{n+}
-
\frac{(\rho_n V_n)^{n+1} - (\rho_n V_n)^{old,*}}{\Delta t}
\]

然后恢复 face 速度：

\[
u_f = \frac{G_f}{\rho_f A_f}
\]

再得到真正进入对流项的相对速度：

\[
u_f^{rel} = u_f - v_{c,f}
\]

---

## 7.2 液相恢复

液滴中心边界满足：

\[
u_l(r=0)=0
\]

因此从中心开始 sweep：

\[
G_{l,1-}=0
\]

\[
G_{l,n+}=G_{l,n-}+C_{l,n}-S_{l,n}
\]

直到界面面，得到 `u_l|if`。

---

## 7.3 气相恢复

气相从界面面开始。  
气相恢复从界面面开始，且 gas-side Eq.(2.18) 在这里被**强施加**。
界面面质量流率不由远场条件反推，而是由同一个 `mpp`、同一个冻结 `\dot a^(k)`、同一个界面状态统一给定：

\[
G_{g,if}^{abs} = (\rho_{s,g}\dot a^{(k)} - mpp)A_{if}
\]

于是界面气相速度定义为：

\[
u_{g,if}
=
\frac{G_{g,if}^{abs}}{\rho_{s,g}A_{if}}
=
\dot a^{(k)}-\frac{mpp}{\rho_{s,g}}
\]

然后从界面向远场 sweep：

\[
G_{g,n+}=G_{g,n-}+C_{g,n}-S_{g,n}
\]

最终得到各个气相 face 速度与 `u_g|if`。

---

## 8. 特殊控制面离散闭合：中心与远场

## 8.1 中心边界 `r = 0`

### 连续边界条件
液相中心满足：

\[
u_l(0)=0
\]

\[
\left.\frac{\partial T_l}{\partial r}\right|_{r=0}=0
\]

\[
\left.\frac{\partial Y_{l,i}}{\partial r}\right|_{r=0}=0
\]

### 几何事实
球心 face 半径为 0，因此：

\[
A_{1-}=4\pi r_{1-}^2=0
\]

### 正式离散闭合
对第一个液相控制体：

- 中心面对流质量通量取 0
- 中心面组分扩散通量取 0
- 中心面热通量取 0

即：

\[
(\rho_l(u_l-v_c)A)_{1-}=0
\]

\[
J_{l,i,1-}A_{1-}=0
\]

\[
q_{l,1-}A_{1-}=0
\]

### 速度恢复
连续方程恢复速度时，从球心面零质量流率开始：

\[
G_{l,1-}=0
\]

### 实现说明
球心不作为普通 ghost-cell 边界统一处理；首版推荐直接把球心面封成**零净通量面**。

---

## 8.2 far-field 边界 `r = R_{end}`

### 连续边界条件
`paper_v1` 首版采用 far-field Dirichlet：

\[
T_g(R_{end}) = T_\infty
\]

\[
Y_{g,i}(R_{end}) = Y_{g,i,\infty}
\]

### 速度条件
不对 `u_g` 直接施加独立 Dirichlet。  
气相速度由连续方程恢复；远场只对 `T_g` 与 `Y_g` 给边界值。

### 扩散项离散闭合
对最后一个气相控制体外侧面 `N+`，采用 face value + one-sided gradient：

\[
\left.\frac{\partial \phi}{\partial r}\right|_{N+}
\approx
\frac{\phi_{bc}-\phi_N}{r_{N+}-r_N}
\]

其中：

- \(\phi\) 可为 \(T_g\)
- 也可为 \(Y_{g,i}\) 或 \(X_{g,i}\)
- \(\phi_{bc}\) 为对应远场边界值

于是：

#### 导热项
\[
q^{cond}_{g,N+}
=
-k_{g,N+}
\frac{T_\infty-T_{g,N}}{r_{N+}-r_N}
\]

#### 气相扩散速度
\[
V^0_{d,i,N+}
=
-\frac{D_{g,i,N+}}{X_{g,i,N+}}
\frac{X_{g,i,\infty}-X_{g,i,N}}{r_{N+}-r_N}
\]

\[
V_{cd,N+}
=
-\sum_{m=1}^{N_g}Y_{g,m,N+}V^0_{d,m,N+}
\]

\[
J_{g,i,N+}
=
-\rho_{g,N+}Y_{g,i,N+}(V^0_{d,i,N+}+V_{cd,N+})
\]

#### 总热流
\[
q_{g,N+}
=
q^{cond}_{g,N+}
+
\sum_i J_{g,i,N+}h_{g,i,N+}
\]

### 对流项离散闭合
对 far-field 外侧 face，仍按一阶上风处理：

- 若 \((u_g-v_c)_{N+}>0\)，取内部值
- 若 \((u_g-v_c)_{N+}<0\)，取远场边界值

而第三段固定网格区有：

\[
v_c=0
\]

因此 far-field face 上对流闭合就是标准物理速度上风。

### 说明
远场离散闭合的作用是：  
把连续层面的 `T∞, Y∞` 边界条件真正写入最后一个气相控制体的 residual，而不是只在概念上说“远场给 Dirichlet”。

---

## 9. 界面严格对接：single interface face package

## 9.1 基本原则

界面不是普通边界，而是唯一一个物理耦合面。  
因此，界面面上的状态与通量必须只有一个真相源。

正式规定：

> **最后一个液相控制体、界面块、第一个气相控制体，必须共享同一套 interface face package。**

并且进一步规定：

- `properties/aggregator.py` 只允许构造 bulk cell-centered `Props`
- 界面位置上的状态、物性、派生通量只能由 `physics/interface_face.py` 统一构造
- 不允许 bulk residual 与 interface residual 各自再造一套界面值

不允许出现：

- bulk 自己算一套界面通量
- interface residual 自己又算另一套界面通量

---

## 9.2 几何与距离

在当前 outer iterate `k` 的冻结几何下：

\[
r_s = a^{(k)}
\]

\[
A_s = 4\pi (a^{(k)})^2
\]

界面到两侧 cell center 的距离定义为：

\[
\Delta r_{l,s}=a^{(k)}-r_{l,N_l}
\]

\[
\Delta r_{g,s}=r_{g,1}-a^{(k)}
\]

---

## 9.3 interface face package 的内容

在每一次 inner residual 评估中，统一构造界面面数据包，至少包含：

### 界面主状态
- \(T_s\)
- \(Y_{s,l,i}\)
- \(Y_{s,g,i}\)
- \(mpp = \dot m''\)

### 物性与派生量
- \(\rho_{s,l}\), \(\rho_{s,g}\)
- \(h_{s,l}\), \(h_{s,g}\)
- \(h_{l,i,s}\), \(h_{g,i,s}\)
- \(k_{l,s}\), \(k_{g,s}\)
- \(D_{l,i,s}\), \(D_{g,i,s}\)

### 界面两侧梯度
- \(\partial T_l/\partial r|_s\)
- \(\partial T_g/\partial r|_s\)
- \(\partial Y_{l,i}/\partial r|_s\)
- \(\partial X_{g,i}/\partial r|_s\)

### 界面两侧扩散通量
- \(J_{l,i,s}\)
- \(J_{g,i,s}\)

### 界面两侧总物种通量
- \(N_{l,i,s}\)
- \(N_{g,i,s}\)

### 界面两侧总热通量
- \(q_{l,s}\)
- \(q_{g,s}\)

### 界面两侧总能量通量
- \(E_{l,s}\)
- \(E_{g,s}\)

---

## 9.4 界面一侧梯度：one-sided face gradient

### 温度梯度

\[
\left.\frac{\partial T_l}{\partial r}\right|_s
\approx
\frac{T_s-T_{l,N_l}}{\Delta r_{l,s}}
\]

\[
\left.\frac{\partial T_g}{\partial r}\right|_s
\approx
\frac{T_{g,1}-T_s}{\Delta r_{g,s}}
\]

### 液相组分梯度

\[
\left.\frac{\partial Y_{l,i}}{\partial r}\right|_s
\approx
\frac{Y_{s,l,i}-Y_{l,N_l,i}}{\Delta r_{l,s}}
\]

### 气相摩尔分数梯度

\[
\left.\frac{\partial X_{g,i}}{\partial r}\right|_s
\approx
\frac{X_{g,1,i}-X_{s,g,i}}{\Delta r_{g,s}}
\]

说明：

- 气相修正扩散通量必须使用 **mole fraction gradient**
- 因此界面 face package 在气相侧必须先恢复 full composition，再构造 full mole fractions

---

## 9.5 界面一侧扩散通量

### 液相侧（Fick）

\[
J_{l,i,s}
=
-\rho_{s,l}D_{l,i,s}
\frac{Y_{s,l,i}-Y_{l,N_l,i}}{\Delta r_{l,s}}
\]

### 气相侧（mixture-averaged + conservative correction）

\[
V^0_{d,i,s}
=
-\frac{D_{g,i,s}}{X_{s,g,i}}
\frac{X_{g,1,i}-X_{s,g,i}}{\Delta r_{g,s}}
\]

\[
V_{cd,s}
=
-\sum_{m=1}^{N_g}Y_{s,g,m}V^0_{d,m,s}
\]

\[
J_{g,i,s}
=
-\rho_{s,g}Y_{s,g,i}\left(V^0_{d,i,s}+V_{cd,s}\right)
\]

---

## 9.6 界面一侧总物种通量

定义径向向外为正方向。  
于是界面总物种通量统一定义为：

### 液相侧
\[
N_{l,i,s}
=
-mpp\,Y_{s,l,i}+J_{l,i,s}
\]

### 气相侧
\[
N_{g,i,s}
=
-mpp\,Y_{s,g,i}+J_{g,i,s}
\]

---

## 9.7 界面一侧总热通量与总能量通量

### 液相总热通量
\[
q_{l,s}
=
-k_{l,s}\frac{T_s-T_{l,N_l}}{\Delta r_{l,s}}
+\sum_{i=1}^{N_l}J_{l,i,s}h_{l,i,s}
\]

### 气相总热通量
\[
q_{g,s}
=
-k_{g,s}\frac{T_{g,1}-T_s}{\Delta r_{g,s}}
+\sum_{i=1}^{N_g}J_{g,i,s}h_{g,i,s}
\]

### 液相总能量通量
\[
E_{l,s}
=
-mpp\,h_{s,l}+q_{l,s}
\]

### 气相总能量通量
\[
E_{g,s}
=
-mpp\,h_{s,g}+q_{g,s}
\]

说明：

- 内部实现推荐使用总能量通量连续形式
- 它与文献的潜热平衡形式在连续层面等价
- 这样最适合 bulk residual 与 interface residual 共用一套通量 package

---

## 9.8 bulk 两侧控制体如何使用 interface face package

### 最后一个液相控制体
其外侧面就是界面面，因此：

- 组分方程外侧面通量，直接取 \(N_{l,i,s}\)
- 能量方程外侧面通量，直接取 \(E_{l,s}\)

### 第一个气相控制体
其内侧面也是同一个界面面，因此：

- 组分方程内侧面通量，直接取 \(N_{g,i,s}\)
- 能量方程内侧面通量，直接取 \(E_{g,s}\)

正式禁止：

- 液相最后一格自己再重算一套界面通量
- 气相第一格自己再重算另一套界面通量

---

## 10. 界面残差的严格对接离散

## 10.1 `mpp` 残差：Eq.(2.18)

正式写法统一采用液相侧主残差：

\[
R_{if,m}
=
mpp + \rho_{s,l}\left(u_{l,if}-\dot a^{(k)}\right)
=0
\]

### 关键约束
在 inner nonlinear solve 中：

\[
\dot a = \dot a^{(k)} = v_{c,if}^{(k)}
\]

即当前 outer iteration 的冻结值。
禁止在 inner Newton 过程中用当前 trial `mpp` 现算新的 `\dot a` 并回灌 residual。

### 气相侧如何共同耦合
气相侧 Eq.(2.18) 在 `paper_v1` 中不是第二条主残差，而是通过 gas-side interface boundary mass flux 被强施加：

\[
G_{g,if}^{abs} = (\rho_{s,g}\dot a^{(k)} - mpp)A_{if}
\]

这意味着：

- `mpp` 同时进入 liquid-side 主残差与 gas-side 界面边界构造
- gas bulk continuity sweep 的起点直接由同一个 `mpp` 给定
- 两相都与同一个 `mpp` 共同耦合

### 气相侧诊断
在强施加 gas-side Eq.(2.18) 的同时，仍应输出实现一致性诊断：

\[
\epsilon_{m,g}
=
\left|
mpp + \rho_{s,g}(u_{g,if}-\dot a^{(k)})
\right|
\]

---

## 10.2 可溶组分残差：Eq.(2.15) 的推荐离散实现

对共有/可凝组分，正式采用**总物种通量连续**写法：

\[
R_{if,l,i}
=
N_{g,i,s}-N_{l,i,s}
=0
\]

这与文献 Eq.(2.15) 等价，但更适合 strict coupling 的统一实现。

---

## 10.3 非凝组分残差：Eq.(2.16) 的推荐离散实现

对非凝组分：

\[
R_{if,g,k}
=
N_{g,k,s}
=0
\]

这与文献 Eq.(2.16) 等价，表示非凝组分不能穿入液相。

---

## 10.4 `Ts` 残差：Eq.(2.17) 的推荐离散实现

正式推荐采用**总能量通量连续**写法：

\[
R_{if,E}
=
E_{g,s}-E_{l,s}
=0
\]

说明：

- 该写法与文献的潜热平衡形式等价
- 它自动包含：
  - 导热
  - 扩散焓通量
  - 相变焓输运
- 它更适合 bulk/interface 共用同一界面通量包

### 与单个 `T_s` unknown 的关系
由于 `paper_v1` 只使用单个界面温度 unknown `T_s`，因此：

- **温度连续性已由 unknown 结构内建**
- 不再额外写一条 `T_{l,s}=T_{g,s}` 残差

### 固定方程结构约束

`paper_v1` 的界面能量行只允许采用这一套固定结构：

- Eq.(2.17) 直接形式
或
- 等价的总能量通量连续形式

正式禁止：

- sat / boil regime switching
- hard pin `T_s`
- 在运行中替换成另一套界面能量方程结构

---

## 10.5 `Y_s,g(cond)` 残差：Eq.(2.19)

对可凝蒸汽组分：

\[
R_{if,eq,j}
=
Y_{s,g,j}^{cond}
-
Y_{eq,j}(T_s, Y_{s,l}, P_\infty, \gamma_j, \ldots)
=0
\]

其中 `Y_eq,j(...)` 由相平衡模型给出。

### 单组分首版
\[
R_{if,eq}
=
Y_{s,g,fuel}
-
Y_{eq}(T_s,P_\infty)
=0
\]

---

## 10.6 strict coupling 的必需诊断量

至少输出：

- `if_mass_balance_err_gas_side`
- `if_energy_flux_jump`
- `if_species_flux_jump_max`
- `if_noncond_flux_max`

这样才能判断 strict coupling 是否真的做到，而不是只靠“看起来能收敛”。

---

## 11. 边界条件与 full-order state

## 11.1 闭合组分与 full-order state

### 规定
- 每相 closure species 由配置文件显式给出
- 不允许默认 closure species
- 若缺失配置，立即报错

### 状态结构
- 单组分液滴阶段，保留 `Yl_full=[1.0]`
- 气相始终保留 `Yg_full`
- 界面也保留 `Ys_l_full` / `Ys_g_full`
- reduced unknown 只是不全部进未知量，不代表 full-order state 不存在

## 11.2 气相修正扩散与 full state 的关系

正式规定：

- 界面气相修正扩散通量
- 普通内部气相 face 修正扩散通量
- far-field 气相 face 修正扩散通量

都必须使用 full gas composition 计算 `X_i`、`V^0_{d,i}`、`V_{cd}`。

---

## 12. 物性与参数计算

## 12.1 压力
\[
P = P_\infty = \text{constant}
\]

不引入额外压力未知量。

## 12.2 密度
\[
\rho_l = \rho_l(T_l, Y_l)
\]

\[
\rho_g = \rho_g(T_g, Y_g, P_\infty)
\]

\[
\rho_{s,l} = \rho_l(T_s, Y_{s,l})
\]

\[
\rho_{s,g} = \rho_g(T_s, Y_{s,g}, P_\infty)
\]

## 12.3 扩散系数、导热系数、焓
由当前 trial state 调用物性模块计算：

- `D_l, D_g`
- `k_l, k_g`
- `h_l, h_g`
- `h_i`
- `L_i(T_s)`

## 12.4 相平衡
由 `equilibrium_model(...)` 计算：

\[
Y_{eq,j}=Y_{eq,j}(T_s,Y_{s,l},P_\infty,\gamma_j,\ldots)
\]

但它只服务于可凝蒸汽平衡，不包办背景气。

---

## 13. 内外层整体计算框架

## 13.1 外层职责

外层 predictor-corrector 负责：

- 给定 `a^(k)` 与 `\dot a^(k)`
- 构建当前 fixed geometry 的网格与 `v_c`
- 将 old state remap 到当前几何
- inner 收敛后更新 `a` 与网格
- 处理 accept / reject / timestep control

## 13.2 内层职责

内层 fixed-geometry nonlinear solve 负责：

\[
U = [\Phi_l,\phi_s,\Phi_g]
\]

其中：

- `Φ_l`：液相 bulk unknown block
- `φ_s`：界面块
- `Φ_g`：气相 bulk unknown block

在 residual 评估时：

1. 从 trial unknown 恢复 `ρ, h, D, k, L, gamma...`
2. 使用当前 fixed geometry 的 `V_n, A_f, v_c`
3. 用 continuity recovery 得到 `u_l, u_g`
4. 组装 liquid bulk residual
5. 用同一个 `mpp` 与冻结 `\dot a^(k)` 构造 gas-side interface boundary mass flux
6. 组装统一的 interface face package
7. 组装 interface residual
8. 组装 gas bulk residual

## 13.3 单步流程

### 外层第 `k` 轮
1. 给定 `a^(k), \dot a^(k), grid^(k), v_c^(k)`
2. 形成 `old_state_on_current_geometry`
3. 进入 inner solve

### inner solve
1. Newton / SNES 迭代
2. 计算 bulk + interface 残差
3. 收敛后得到：
   - `mpp^(k)`
   - `T_s^(k)`
   - `Y_s^(k)`
   - `u_l|if^(k)`
   - `ρ_s,l^(k)`

### outer corrector
4. 用
   \[
   \dot a^{new}
   =
   u_{l,if}^{(k)}
   +
   \frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
   \]
   更新界面速度
5. 更新 `a`
6. 重建网格与 `v_c`
7. 检查几何收敛；未收敛则继续下一轮

---

## 14. 单组分首版与多组分升级版差异

## 14.1 单组分首版

- 液相 bulk：`T_l`
- 界面块：`T_s, Y_s,g,red, mpp`
- 气相 bulk：`T_g, Y_g,red`

特点：

- `Y_s,l,full=[1.0]`
- Eq.(2.15) 无自由度
- Eq.(2.16) 约束非凝界面组分
- Eq.(2.17) / 等价总能量通量连续 约束 `T_s`
- Eq.(2.18) 约束 `mpp`
- Eq.(2.19) 约束燃料蒸汽界面平衡

## 14.2 多组分升级版

- 液相 bulk：`T_l, Y_l,red`
- 界面块：`Y_s,l,red, T_s, Y_s,g,red, mpp`
- 气相 bulk：`T_g, Y_g,red`

特点：

- Eq.(2.15) 恢复完整作用
- Eq.(2.19) 继续约束可凝蒸汽平衡
- 界面块完全成为论文式物理主线

---

## 15. 正式禁止事项

以下做法明确禁止：

1. 把界面正式主线再退回“只有 `Ts,mpp` 两 unknown”
2. 把 `mpp` 保留为独立界面 unknown，却不给它 Eq.(2.18) 残差位置
3. 在 inner Newton 中实时更新 `\dot a`
4. 用 Stefan 经验速度场替代 continuity recovery
5. 在主框架中使用比例补偿式背景气算法
6. 让 `equilibrium_model` 同时包办可凝蒸汽与背景气界面组成
7. 把 `Rd` 再塞回 inner unknown vector
8. 在某个局部模块偷偷引入 explicit scaling
9. 把气相扩散通量退化成统一 Fick 形式
10. 让最后一个液相控制体、界面块、第一个气相控制体分别重算各自的界面通量
11. 在 far-field 边界对 `u_g` 另加独立 Dirichlet
12. 在球心面引入非零净通量

---

## 16. 最终方法合同

### Governing Equations / Discretization / BC Contract

1. `paper_v1` uses the conservative moving-control-volume formulation for both phases.
2. Spatial discretization uses cell-centered finite volume in spherical symmetry.
3. Convective terms use first-order upwind with relative velocity `u-v_c`.
4. Internal diffusive gradients use second-order central differences.
5. Time discretization uses backward Euler.
6. Velocities are recovered from the discretized continuity equation, not solved as primary unknowns.
7. `mpp` is an independent interface unknown and its primary residual is Eq.(2.18).
8. `Ts` is an independent interface unknown and its primary residual is Eq.(2.17) or its equivalent total-energy-flux form.
9. Species-level interface unknowns are governed by Eq.(2.15), Eq.(2.16), and Eq.(2.19).
10. Interface densities are EOS-derived in the first-release and recommended-final versions.
11. `Rd/a(t)` remains in the outer predictor-corrector layer.
12. If Eq.(2.18) is used in the inner residual, `\dot a` must be the frozen outer-iteration value.
13. Far-field boundary conditions are Dirichlet for `T_g` and `Y_g`, but not for `u_g`.
14. Center boundary conditions are symmetry conditions, implemented as zero net flux at the center face.
15. Radiation is disabled in Phase A by removing the radiation term, not by replacing the interface energy equation.
16. Gas diffusive flux must use the mixture-averaged form with conservative correction.
17. Liquid diffusive flux keeps the Fick form.
18. Gas-side Eq.(2.18) is strongly imposed through the interface boundary mass-flux construction, not added as a second independent `mpp` residual row.
19. Interface coupling must be implemented through a single interface face package shared by bulk and interface residuals.
20. The last liquid cell and the first gas cell must use the same interface package as the interface residual.
21. Special control faces (center, interface, far-field) require dedicated discrete closure rules and must not be treated as ordinary internal faces.

---

## 17. 最终结论

`paper_v1` 的控制方程、离散格式、边界条件和特殊控制面闭合正式定为：

- 相内：守恒形式 + FVM + backward Euler
- 对流：一阶上风，使用 `u-v_c`
- 液相扩散：Fick
- 气相扩散：mixture-averaged + conservative correction
- 热流：两相都用 generalized Fourier’s law
- 速度：由连续方程恢复
- 中心：零净通量面
- 远场：`T∞, Y∞` Dirichlet + one-sided face closure，不单独给 `u_g`
- 界面：Eq.(2.15)-Eq.(2.19) 成组闭合，并通过 single interface face package 与 bulk 严格对接
- `mpp`：用 liquid-side Eq.(2.18) 主残差，并通过 gas-side boundary mass-flux 强施加双侧耦合
- `Ts`：用 Eq.(2.17) 或等价总能量通量连续
- `Rd`：outer predictor-corrector 更新

这就是 `paper_v1` 的物理控制 + 数值离散 + 边界处理 + 特殊控制面闭合的正式定稿 v2 版本。
