# paper_v1 总纲（第二版）

## 1. 文件定位

本文档用于作为新代码项目 `paper_v1` 的**第二版总纲**。  
它不是旧项目的修补说明，而是新项目的正式方法总览。

本版总纲吸收了后续专题讨论的修正结果，重点解决了初版中以下问题：

1. 界面未知量与界面方程的对应关系不清晰  
2. `mpp` 的控制方程位置没有定死  
3. outer / inner 两层循环职责边界不够明确  
4. 三段网格、动网格与控制面速度 `v_c` 的定义不够严谨  
5. remap 的守恒策略没有正式固定  
6. 液相和气相物性主线没有分开  
7. 焓、潜热、参考温度与基准没有统一  
8. 初版中 `y_if_aux...` 这类未定义占位写法不应再保留

因此，**本版总纲是初版的替代版**，不是补丁版。

---

## 2. 项目目标

`paper_v1` 的目标不是继续修补旧骨架，而是建立一套：

- 更容易跑通
- 更容易验证
- 更容易诊断
- 更容易扩展

的蒸发/传输基线方法。

项目遵循两条明确的工程约束：

1. **化学源项不进入 transport 主系统**  
   采用外置 splitting 路线：
   \[
   chem(dt/2)\;\rightarrow\;transport(dt)\;\rightarrow\;chem(dt/2)
   \]

2. **不考虑辐射**
   界面能量方程保留原结构，但首版删去辐射项，不另换方程系统。

因此，`paper_v1` 的定位是：

> 以主论文第二章的  
> **transport / interface / moving-mesh / moving-control-volume**  
> 主线为基础，  
> 做一个 **chemistry 外置、radiation 关闭** 的工程化版本。

---

## 3. 参考文献分工

## 3.1 主参考文献

主参考文献负责提供以下主线：

- 1D 球对称自由液滴问题
- moving control volume / moving mesh 框架
- 界面质量、能量、相平衡条件
- backward Euler
- 上风对流、中心扩散
- predictor-corrector 的总体思想
- outer geometry + inner field solve 的大框架

## 3.2 [30] 的作用

[30] 只能作为方法论补充，不作为自由球滴问题的 1:1 模板。

它支持的判断包括：

- moving interface 可以和场变量求解分层处理
- 当前步几何固定后再解 PDE 是合理的
- conforming mesh / ALE / remeshing 这类思想在方法上成立

但它不能直接决定：

- 1D spherical FVM 的离散细节
- 自由球滴问题中的界面块组织方式
- 论文第二章那套三段网格和界面方程的实现细节

## 3.3 PETSc 的作用

PETSc 负责的是**求解器底座能力**，不是问题专属逻辑。

PETSc 可以承担：

- Newton / line search
- FD Jacobian / matrix-free
- LU / KSP / PC
- Jacobian lagging / reuse

但以下内容仍然属于应用层：

- outer predictor-corrector
- 变量物理裁剪
- 界面块定义
- 网格重建
- remap / projection
- 时间步接受/拒绝
- chemistry splitting

---

## 4. 第一版与推荐最终版的正式定位

## 4.1 第一版定位

第一版是 `transport-only baseline`，目标是先得到一套：

- 可收敛
- 可验证蒸发曲线
- 可诊断界面方程
- 可扩展到后续 chemistry

的基线系统。

### 第一版范围
- 几何：1D 球对称
- 液滴：首个算例先做单组分乙醇
- 气相：从第一版起就是多组分
- 液相：接口和数据结构按多组分设计
- 化学：外置，transport 步内无反应源项
- 辐射：关闭
- 求解器：PETSc SNES + KSP + LU
- 多 rank 生产运行：首版不要求
- fieldsplit 生产配置：首个可跑通配置可以暂不启用

### 第一版对 MPI / fieldsplit 的正式解释

这里的“首版不要求 MPI / 不要求 fieldsplit 生产配置”，必须统一理解为：

- **首版不要求多 rank 生产运行**
- **首个可跑通 case 可以先使用 debug linear profile，例如 `preonly + lu`**
- **但代码骨架必须从第一天起保持 MPI-ready / fieldsplit-ready**

因此，以下结构不能缺：

- `DMDA(liq) + DMREDUNDANT(interface) + DMDA(gas) + DMComposite(global)`
- block layout
- fieldsplit index-set 导出能力
- owned-row / global-local / ghosted local view 语义

换句话说：

- 可以暂时不用多 rank 跑首个基线
- 可以暂时不用 fieldsplit 作为首个线性求解配置
- 但不能因此把 `layout / DM / IS / block structure` 写死成单块串行脚本式结构

## 4.2 推荐最终版（在 `paper_v1` 范围内）

推荐最终版仍然保持：

- 两层结构
- `Rd` 不进内层主系统
- 论文式界面变量块主线
- phase-wise conservative remap
- 三段网格
- 气相 Cantera，液相文献方法

它不是 monolithic geometry+field 全耦合版本。

---

## 5. 控制方程主线

transport 步求解下列方程：

### 相内
- 液相连续方程（用于速度恢复）
- 液相能量方程
- 液相组分方程（首个单组分算例可退化）
- 气相连续方程（用于速度恢复）
- 气相能量方程
- 气相多组分守恒方程

### 界面
- Eq.(2.15)：共有/可凝组分界面守恒
- Eq.(2.16)：非凝组分界面守恒
- Eq.(2.17)：界面能量守恒
- Eq.(2.18)：总质量通量定义
- Eq.(2.19)：可凝蒸汽相平衡

### 外层几何
- 界面推进公式
  \[
  \dot a = u_{l,if} + \frac{mpp}{\rho_{s,l}}
  \]

---

## 6. unknown strategy（正式修订版）

这是初版变化最大的部分之一。

## 6.1 总原则

- `Rd/a(t)` **不进入**内层 transport 主未知量
- `u_l, u_g` **不作为**主未知量
- 能量方程物理上按焓守恒处理，但代数 unknown 用温度
- 每相去掉一个闭合组分，由 `sum(Y)=1` 回补
- 界面采用**论文式界面变量块主线**
- 但首版不把 `rho_s,l, rho_s,g` 显式放进界面 unknown

## 6.2 bulk unknowns

### 液相
- 单组分首版：
  \[
  [T_l]
  \]
- 多组分升级版：
  \[
  [T_l,\;Y_{l,1}\ldots Y_{l,N_l-1}]
  \]

### 气相
从第一版起即为：
\[
[T_g,\;Y_{g,1}\ldots Y_{g,N_g-1}]
\]

## 6.3 interface block（正式修订）

### 单组分液滴首版
\[
\phi_{s,A}=\{T_s,\;Y_{s,g,red},\;mpp\}
\]

### 多组分液滴升级版
\[
\phi_{s,B}=\{Y_{s,l,red},\;T_s,\;Y_{s,g,red},\;mpp\}
\]

### 不再允许的旧写法
以下旧写法正式废止：

```text
interface: T_s, mpp, y_if_aux...
