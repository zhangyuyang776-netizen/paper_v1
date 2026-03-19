# petsc_solver_and_parallel_architecture_guideline_final

## 1. 文件定位

本文档统一规定 `paper_v1` 中与 PETSc 求解相关的三部分内容：

1. **nonlinear guard and solver policy**
   规定 inner 非线性问题如何用 PETSc 求解，选择哪些对象、算法与 guard 机制。
2. **serial → parallel 改造路线**
   规定从单进程实现到多进程 MPI 实现的正式演进路径。
3. **parallel-ready serial architecture**
   规定初版“串行代码”必须如何书写，才能从第一天就具备未来并行扩展的骨架。

本文档生效后，`paper_v1` 在 PETSc 相关架构上不再采用“先简单版、再升级版、最后重构版”的路线，而直接按最终目标组织代码。

---

# 2. 总体边界：哪些交给 PETSc，哪些不交

`paper_v1` 采用两层结构：

* **outer**：负责 `a / dot_a / geometry / v_c / remap / accept-reject`
* **inner**：负责 fixed-geometry 下的 liquid bulk + interface block + gas bulk 耦合非线性求解

正式规定：

* **PETSc 只负责 inner**
* **outer 不交给 PETSc 的 TS 或第二个 SNES**
* **时间步接受/拒绝与步长调整仍遵循主参考文献的项目级控制逻辑**

这条边界来自两个事实：

第一，PETSc 的 `SNES` 是管理非线性求解的对象，最常用类型是 `SNESNEWTONLS`，对 Newton 类方法会调用 `KSP` 去解线性化系统。([petsc.org][1])

第二，主参考文献第二章的时间推进主线是：**backward Euler + moving-mesh predictor-corrector + 外层界面速度一致性收敛 + 基于收敛难度的时间步增减**，而不是把整步问题完全交给一个标准时间推进器。

---

# 3. 文献基线在 `paper_v1` 中的保留内容

以下内容保留为 `paper_v1` 的正式项目级主线，不由 PETSc 自动替代：

## 3.1 时间推进

* backward Euler
* 每步的 predictor-corrector 外层循环
* 界面位置与网格更新
* outer 收敛判据：界面速度归一化差小于 `1e-5`

## 3.2 时间步控制

* 不是局部截断误差驱动
* 而是以“非线性求解是否容易收敛”作为刚性指标
* 若失败：回退到最近 accepted state，并执行 `dt -> dt/2`
* 若最近 `q = 10` 个 accepted steps 全部稳定，且期间 `dt` 未改变，则执行
  [
  dt_{\text{new}} = \min(1.1,dt,;dt_{\max})
  ]

这些都是主参考文献第二章已经明确给出的主线，因此 `paper_v1` 保留，不重新发明。

---

# 4. PETSc 中 inner 非线性求解的正式理解

在当前外层给定的冻结几何下，inner 要解的是一个非线性方程组：

[
F(U)=0
]

其中：

* `U`：当前 fixed-geometry 下的全局未知向量
* `F(U)`：液相 bulk residual + interface residual + 气相 bulk residual

一次 inner 迭代的基本过程如下：

1. 给定当前猜测 `U_k`
2. 计算残差 `F(U_k)`
3. 若残差满足收敛准则，则 inner 成功
4. 否则构造 Jacobian
   [
   J_k \approx \partial F/\partial U
   ]
5. 解线性系统
   [
   J_k ,\delta U_k = -F(U_k)
   ]
6. 通过 line search 选取步长 `λ_k`
7. 更新
   [
   U_{k+1} = U_k + \lambda_k \delta U_k
   ]
8. 若 trial state 非法，则缩步或报告 domain error
9. 直到收敛或发散

PETSc 官方文档中，`SNES` 管非线性求解，Newton 类 `SNES` 通过 `KSP` 近似求解线性系统。([petsc.org][1])

---

# 5. PETSc 关键对象的职责

## 5.1 DM

`DM` 负责网格/数据布局/通信语义。
对 `paper_v1`，正式采用：

* liquid: `DMDA`
* gas: `DMDA`
* interface: `DMREDUNDANT`
* global coupled layout: `DMComposite`

PETSc 文档中，`DMDA` 用于结构网格，`DMComposite` 用于组合多个 `DM`，而 `DMREDUNDANT` 用于少量、全局耦合的变量块。([petsc.org][2])

## 5.2 SNES

`SNES` 是非线性求解器对象。
最常用类型是 `SNESNEWTONLS`，即带 line search 的 Newton。([petsc.org][1])

## 5.3 KSP

`KSP` 是线性求解器对象，负责解 Newton 线性化系统。
正式采用 `KSPFGMRES`，因为它允许预条件器在迭代中变化，更适合 `fieldsplit + schur + asm` 这种分层结构。([petsc.org][3])

## 5.4 PC

`PC` 是预条件器对象。
正式采用 `PCFIELDSPLIT`，并使用 Schur complement 模式；bulk 子块用 `PCASM`，interface 子块用 `PCLU`。PETSc 对这三个对象的语义都有明确文档。([petsc.org][4])

---

# 6. `paper_v1` 的最终 PETSc 配置

## 6.1 数据布局

正式规定：

* `DMDA(liquid)`
* `DMDA(gas)`
* `DMREDUNDANT(interface)`
* `DMComposite(global)`

理由：

* liquid / gas 都是 1D 结构网格
* interface 是少量全局耦合变量
* `DMComposite` 天然适合把三块组合成一个全局问题

这一布局与物理结构一致，也天然支持并行扩展。([petsc.org][2])

---

## 6.2 向量与数据同步

正式规定：

* 主状态向量使用 **global Vec**
* stencil / flux / residual kernel 使用 **local ghosted Vec**
* 同步通过
  `DMGlobalToLocalBegin()`
  `DMGlobalToLocalEnd()`
  显式完成

PETSc 文档明确说明：

* `DMCreateGlobalVector()` 创建的是 global vector
* global vector 没有 ghost
* `DMGlobalToLocalBegin/End()` 用于把 global 数据散到 local ghosted 表示，而且 Begin/End 之间可以重叠本地计算与通信。([petsc.org][5])

---

## 6.3 非线性求解器

正式规定：

* `snes_type = newtonls`

理由：

* inner 是标准非线性残差方程组
* `newtonls` 是 PETSc 默认和最常用的主方法
* 它比 `qn / anderson / ngmres` 更适合你这种明确的 PDE residual 框架

PETSc 文档对 `SNES` 和 `SNESNEWTONLS` 的说明足够直接。([petsc.org][1])

---

## 6.4 Line search

正式规定：

* `snes_linesearch_type = bt`

即 backtracking line search。

理由：

* 这是 `SNESNEWTONLS` 的自然搭配
* 它适合当前这种界面/物性/焓反解都可能带来非线性陡变的问题
* 它应被视为主线保命器，而不是可有可无的附件

同时允许使用：

* `SNESLineSearchSetPreCheck()`
* `SNESLineSearchSetPostCheck()`

但其作用仅限于**合法性检查与有限修正**，不允许把大量静默投影/裁剪埋进去。PETSc 官方提供了这两个接口。([petsc.org][6])

---

## 6.5 Jacobian 模式

正式规定：

* Jacobian 使用 **user-provided explicit sparse Jacobian**
* 不采用纯 matrix-free 作为主线
* 矩阵类型采用 **显式并行稀疏 AIJ**
* Jacobian 装配基于 **固定 sparsity pattern + 稀疏局部差分/显式条目计算**

理由：

1. 纯 matrix-free 不利于 `fieldsplit + schur + asm/lu/ilu` 这类显式块预条件结构。
2. 全解析 Jacobian 在当前项目中工作量过大，且容易在复杂物性/界面耦合处出错。
3. 显式稀疏 Jacobian + 固定 pattern 更符合你当前项目的工程可靠性目标。

PETSc 文档说明 `MatCreateAIJ()` 是默认并行稀疏矩阵格式，且必须重视预分配。([petsc.org][7])

---

## 6.6 线性求解器与预条件器

正式规定：

### 顶层

* `ksp_type = fgmres`
* `pc_type = fieldsplit`
* `pc_fieldsplit_type = schur`
* `pc_fieldsplit_schur_fact_type = full`
* `pc_fieldsplit_schur_precondition = a11`

### 块划分

* split 0: bulk
* split 1: interface

### bulk 子块

* `fieldsplit_0_ksp_type = fgmres`
* `fieldsplit_0_pc_type = asm`
* `fieldsplit_0_sub_ksp_type = preonly`
* `fieldsplit_0_sub_pc_type = ilu`

### interface 子块

* `fieldsplit_1_ksp_type = preonly`
* `fieldsplit_1_pc_type = lu`

理由如下：

* `PCFIELDSPLIT` 适合 bulk/interface 这类天然块结构系统；其 Schur 模式正是为块耦合问题准备的。([petsc.org][4])
* `PCASM` 是并行加性 Schwarz，默认一个 MPI 进程一个 block，很适合当前 bulk 1D 分区。([petsc.org][8])
* `PCLU` 适合小而强耦合的 interface 子块。([petsc.org][9])
* `KSPFGMRES` 允许预条件器在迭代中变化，更适合这类分层预条件结构。([petsc.org][3])

---

## 6.7 非线性收敛判据

正式规定：

* 以**自定义 scaled residual infinity norm** 作为 inner 主收敛判据
* 默认阈值：
  [
  |R_{\text{scaled}}|_\infty < 10^{-8}
  ]

实现方式：

* 通过 `SNESSetConvergenceTest()` 提供自定义收敛测试
* PETSc 默认 `atol / rtol / stol / max_it` 仅作为辅助安全网，不作为主判据

PETSc 官方提供自定义收敛测试接口。该阈值则与主参考文献第二章中 SSM 的 normalized residual 基线保持一致。([petsc.org][10]) 

---

## 6.8 非线性 guard

正式规定：

* 主线 guard 采用 **domain error + line search backtracking**
* 不采用“静默裁剪 + 惩罚残差”作为默认主线

具体规则：

1. 在 residual callback 中检查物理合法性
2. 若 trial state 非法，则调用 `SNESSetFunctionDomainError()`
3. 若 Jacobian 评估点非法，则调用 Jacobian domain error 机制
4. 让 `newtonls + bt` 尝试通过缩步恢复
5. 若不能恢复，则作为 inner fail 交给 step controller

PETSc 官方明确提供 `SNESSetFunctionDomainError()` 以及相关 domain error 处理机制。([petsc.org][11])

---

## 6.9 KSP 容差控制

正式规定：

* 启用 Eisenstat-Walker
* 使用 `SNESKSPSetUseEW()`

理由：

* 它允许线性求解精度随非线性迭代阶段动态变化
* 比固定的 KSP tolerance 更符合 inner Newton 的成本控制逻辑

PETSc 官方提供该机制。([petsc.org][12])

---

## 6.10 Jacobian / preconditioner lagging

正式规定：

* 首版使用保守策略：

  * `lag_jacobian = 1`
  * `lag_preconditioner = 1`

理由：

* 首版优先保证语义清晰和求解稳定
* 不在 v1 就引入激进 Jacobian reuse
* 后续如果 profile 明确证明是瓶颈，再专题讨论 lagging 优化

PETSc 官方提供 `SNESSetLagJacobian()` 等接口。([petsc.org][13])

---

# 7. 最终 PETSc options 骨架

下面这组可以作为默认 options 骨架：

```text
-snes_type newtonls
-snes_linesearch_type bt
-snes_max_it 30
-snes_max_funcs 200
-snes_ksp_ew
-snes_lag_jacobian 1
-snes_lag_preconditioner 1

-ksp_type fgmres
-ksp_gmres_restart 50
-ksp_gmres_modifiedgramschmidt
-ksp_gmres_preallocate
-ksp_max_it 200

-pc_type fieldsplit
-pc_fieldsplit_type schur
-pc_fieldsplit_schur_fact_type full
-pc_fieldsplit_schur_precondition a11

-fieldsplit_0_ksp_type fgmres
-fieldsplit_0_pc_type asm
-fieldsplit_0_pc_asm_overlap 1
-fieldsplit_0_sub_ksp_type preonly
-fieldsplit_0_sub_pc_type ilu

-fieldsplit_1_ksp_type preonly
-fieldsplit_1_pc_type lu
```

这组配置对应上面已经定下的正式主线。

---

# 8. 串行 → 并行的正式改造路线

虽然最终目标是并行，但实施上仍应分阶段推进。
只是这里的“阶段”不是换方法，而是**在同一最终架构下逐步打开并行维度**。

## 8.1 阶段 A：并行原生数据布局先行

首先只做一件事：

* 建好 `DMDA(liquid)`
* 建好 `DMDA(gas)`
* 建好 `DMREDUNDANT(interface)`
* 建好 `DMComposite(global)`

即使只跑 `MPI size = 1`，也必须按这一布局构造主状态。

## 8.2 阶段 B：只验证 parallel-aware residual

在 `MPI size = 1` 下：

* global → local ghost 更新流程必须已经写好
* residual 只按 owned rows 装配
* local kernel 只读取 local + ghost 数据

然后再验证：

* `mpiexec -n 1` 与串行语义一致
* `mpiexec -n 2`、`-n 4` 时 residual 与 1-rank 结果一致

## 8.3 阶段 C：并行 Jacobian 装配

目标：

* AIJ 矩阵预分配正确
* diagonal / off-diagonal 非零模式正确
* bulk/interface 耦合列完整
* 1-rank 与多-rank Jacobian action 一致

## 8.4 阶段 D：接入 SNES/KSP/PC

只有在 residual/Jacobian 并行语义都正确后，才正式接入：

* `SNESNEWTONLS`
* `KSPFGMRES`
* `PCFIELDSPLIT`
* `PCASM`
* `PCLU`

## 8.5 阶段 E：最后再包 outer / timestep driver

原因很简单：

* outer 是项目级逻辑
* inner 是 PETSc 负责的 fixed-geometry nonlinear solve
* 先确保 inner 并行正确，再让 outer 驱动它

---

# 9. 串行 → 并行改造中最难的地方

## 9.1 ghost / halo 语义

最容易出错的不是公式，而是：

* 什么时候更新 ghost
* residual 用的是不是当前轮 ghost
* Jacobian 与 residual 是否读的是同一轮 local state

## 9.2 interface block 的 ownership

`DMREDUNDANT` 很适合小界面块，但必须明确：

* local 上所有 rank 都可读
* global ownership 是特定 rank
* 不允许多进程重复装配 interface global 行

## 9.3 稀疏模式预分配

如果 pattern 没先独立出来，后期并行 AIJ 装配会非常难看：

* 动态扩容
* 装配慢
* 远端列模式混乱
* performance profile 失真

## 9.4 负载均衡不天然理想

虽然是 1D 问题，但 liquid / gas / interface 的计算代价并不均匀。
尤其 near-interface gas 与 Cantera 调用成本，可能让“点数均匀分配”并不等于“耗时均匀分配”。

## 9.5 串行正确 ≠ 并行正确

很多 bug 只在多 rank 下出现：

* interface 重复装配
* 分区边界 stencil 错
* local/global 索引混淆
* off-diagonal Jacobian 列缺失

因此并行改造必须一直保留：

* 1-rank baseline
* 2-rank equivalence
* 4-rank equivalence

---

# 10. parallel-ready serial architecture：初版串行代码必须如何写

这是本文最重要的部分。
因为你的“串行初版”不能是传统意义上的串行代码，而必须是：

## **单 MPI 进程运行的并行架构**

---

## 10.1 主状态必须以 PETSc Vec 为真相源

正式规定：

* 主状态不允许以 Numpy 全局数组作为唯一真相源
* 主状态必须存于 PETSc global `Vec`
* local 工作视图通过 DM 获取

原因：

* `DMCreateGlobalVector()` 给的是 global vector
* 它没有 ghost
* ghost 应通过 local vector 和 `DMGlobalToLocalBegin/End()` 获得

PETSc 文档对此有明确说明。([petsc.org][5])

---

## 10.2 串行时也必须保留 global/local 两层视图

即使只跑 `-n 1`，也必须遵守：

1. global `Vec` 保存真相状态
2. 进入 stencil kernel 前，先 scatter 到 local ghosted `Vec`
3. 所有 flux / residual / Jacobian kernel 都在 local 视图上读取

禁止写法：

* 在模块内部长期缓存一份全场 Numpy 数组，并绕开 DM 同步流程

---

## 10.3 residual / Jacobian 从第一版就按 owned rows only 写

正式规定：

* 只对 owned rows 装配
* 读取可使用 ghost
* 绝不在 kernel 中按“全局数组 0..N-1”直接遍历

`DMDAGetCorners()` 与 `DMDAGetGhostCorners()` 的区别必须从第一版就用清楚。PETSc 文档明确指出，前者给 owned 范围，后者给含 ghost 的可读范围。([petsc.org][14])

---

## 10.4 从第一版就显式保留三块结构

禁止把 liquid / interface / gas 粗暴平铺成一个手写大数组再说。

正式要求：

* liquid、gas、interface 分块存在
* block layout 必须显式描述
* future fieldsplit index sets 必须能从 layout 自动导出

---

## 10.5 从第一版就独立出 UnknownLayout

必须有明确布局层，至少描述：

* liquid 每 cell 的 unknown 顺序
* gas 每 cell 的 unknown 顺序
* interface block 的 unknown 顺序
* reduced species 到 full species 的映射
* closure species 的排除规则

禁止在 residual/Jacobian 中到处写 magic index。

---

## 10.6 从第一版就使用 AIJ 稀疏矩阵 + 预分配

正式规定：

* Jacobian 不准先用 dense 或随手插条目的试验写法
* 直接采用 `MatCreateAIJ()`
* 直接做 pattern builder
* 直接做 diagonal/off-diagonal 非零预分配

PETSc 对 AIJ 预分配的重要性说得非常直接。([petsc.org][7])

---

## 10.7 物理 kernel、数据同步、装配必须三层分离

函数组织必须明确分开：

1. **数据同步层**
   global/local scatter、ghost 更新
2. **物理 kernel 层**
   flux、residual、局部 Jacobian 条目、物性计算
3. **装配层**
   写入全局 `Vec/Mat`

禁止一个函数从 global state 一路读到局部物性再顺手装配并更新缓存，这种写法以后并行时会像霉一样蔓延。

---

## 10.8 remap / recovery 从第一版就按“分相 + 局部守恒量”实现

因为你已经定下了：

* remap 的对象是 `rho V / rhoY V / rhoh V`
* 分相 remap
* 不跨相 remap
* 第三段固定区可直接复用

所以初版实现必须也保持这种局部化特征，不能偷懒写成 gather-all → global interpolation → scatter-back 的脚本式做法。

---

## 10.9 界面块从第一版起必须保持单独所有权语义

即使单进程，也要当它是 `DMREDUNDANT` 意义下的独立块：

* 不与第一个气相 cell 或最后一个液相 cell 混为一体
* 不跟 bulk 共用同一个隐式数组布局
* 不让界面 unknown 的物理意义依赖“刚好在数组末尾”

---

## 10.10 全局归约必须显式化

第一版就要区分：

* local diagnostics
* global diagnostics
* solver-critical reductions
* post-processing reductions

禁止在物理 kernel 内部顺手做全局 sum/min/max。
因为并行后这些都会变成 MPI reduction，同步代价和语义边界都必须清楚。

---

# 11. 初版代码的强制编码规则

以下规则建议直接写进项目开发规范：

## 必须遵守

1. 状态真相源是 PETSc `Vec`
2. 所有 stencil kernel 输入都是 local ghosted view
3. 所有装配函数只装 owned rows
4. 所有同步都通过 DM 接口显式完成
5. 所有字段布局都通过 layout 层描述
6. 所有 Jacobian 从第一版起使用稀疏 AIJ + pattern builder
7. 串行测试必须使用 `mpiexec -n 1` 语义运行

## 禁止写法

1. Numpy 全域数组作为长期真相源
2. residual 直接按全局数组遍历
3. interface 混进 bulk 平铺数组
4. Jacobian 无 pattern、无预分配
5. 物理 kernel 内部偷偷做全局统计
6. 为了图快，跳过 `DMGlobalToLocalBegin/End()`

---

# 12. 与正式代码架构文件的对应关系

`paper_v1` 的正式项目目录与模块边界，以：

- `paper_v1_code_architecture_and_coding_guideline_final.md`

为准。

因此，PETSc 相关实现不再单独维护一套平行目录草图，而是映射到正式代码分类：

* `src/core/layout.py`：UnknownLayout、field offsets、species maps
* `src/core/state_pack.py`：global/local state 与 PETSc Vec 映射
* `src/core/grid.py`：几何与 control-surface 相关几何量
* `src/parallel/local_state.py`：global -> local ghost scatter 与 local ghosted views
* `src/assembly/jacobian_pattern.py`：pattern builder
* `src/assembly/petsc_prealloc.py`：AIJ 预分配
* `src/assembly/jacobian_global.py`：global Mat assembly
* `src/solvers/petsc_linear.py`：KSP/PC 线性层
* `src/solvers/petsc_snes.py`：SNES inner nonlinear solve
* `src/parallel/dm_manager.py`：DM / scatter / ownership helpers
* `src/parallel/fieldsplit_is.py`：fieldsplit IS 构造
* `src/parallel/mpi_bootstrap.py`：MPI / PETSc bootstrap

正式要求：

* PETSc 语义相关代码只能落在上述模块及其直接协作模块中
* 不允许在 `physics/` 或 `properties/` 内部偷偷嵌入 MPI ownership / DM scatter 逻辑
* outer predictor-corrector、outer 收敛与 accept/reject 仍只允许放在 `src/solvers/outer_*.py` 与 `src/solvers/timestepper.py`

---

# 13. 生产配置与调试配置必须分离

## 13.1 生产配置

正式生产配置采用本文第 6 节确定的主线：

* `newtonls`
* `bt`
* explicit sparse Jacobian
* `fgmres`
* `fieldsplit schur`
* bulk: `asm + sub(ilu)`
* interface: `lu`

## 13.2 调试配置

保留一个排错 profile：

```text
-ksp_type preonly
-pc_type lu
```

作用：

* 当 `SNES` 发散时，用它快速判断问题主要来自 Jacobian/残差实现，还是来自预条件器与 Krylov 配置

调试配置不是生产配置，不得混用。

---

# 14. 最终执行结论

本文档生效后，`paper_v1` 的 PETSc 求解相关内容正式按以下主线执行：

1. **outer / timestep policy**
   继续遵循主参考文献第二章的项目级主线，不交给 PETSc 自动管理。

2. **inner nonlinear solve**
   由 PETSc `SNESNEWTONLS` 负责。([petsc.org][1])

3. **linear solver stack**
   `KSPFGMRES + PCFIELDSPLIT(schur) + bulk(ASM+ILU) + interface(LU)`。([petsc.org][3])

4. **data layout**
   `DMDA(liq) + DMREDUNDANT(interface) + DMDA(gas) + DMComposite(global)`。([petsc.org][2])

5. **nonlinear guard**
   采用 domain error + backtracking，不以静默裁剪为主线。([petsc.org][11])

6. **serial code standard**
   初版代码必须写成“单 MPI 进程下运行的并行架构”，而不是传统的全局数组串行脚本。

---

# 15. 附：建议写入配置文件的骨架

```yaml
petsc_solver:
  dm:
    liquid: dmda_1d
    gas: dmda_1d
    interface: dmredundant
    global: dmcomposite

  jacobian:
    mode: user_provided_explicit_sparse
    matrix_type: mpiaij
    assembly: fixed_pattern_sparse_local_assembly
    use_same_matrix_for_J_and_P: true

  snes:
    type: newtonls
    linesearch: bt
    convergence: custom_scaled_residual_inf
    scaled_residual_inf_tol: 1.0e-8
    max_it: 30
    max_funcs: 200
    use_eisenstat_walker: true
    lag_jacobian: 1
    lag_preconditioner: 1

  ksp:
    type: fgmres
    gmres_restart: 50
    max_it: 200

  pc:
    type: fieldsplit
    fieldsplit_type: schur
    schur_fact_type: full
    schur_precondition: a11

    split_bulk:
      ksp_type: fgmres
      pc_type: asm
      asm_overlap: 1
      sub_ksp_type: preonly
      sub_pc_type: ilu

    split_interface:
      ksp_type: preonly
      pc_type: lu
```

[1]: https://petsc.org/release/manualpages/SNES/SNES/ "SNES — PETSc 3.24.5 documentation"
[2]: https://petsc.org/release/manualpages/DMDA/DMDA/ "DMDA — PETSc 3.24.5 documentation"
[3]: https://petsc.org/release/manualpages/KSP/KSPFGMRES/ "KSPFGMRES — PETSc 3.24.5 documentation"
[4]: https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/ "PCFIELDSPLIT — PETSc 3.24.5 documentation"
[5]: https://petsc.org/release/manualpages/DM/DMCreateGlobalVector/ "DMCreateGlobalVector — PETSc 3.24.5 documentation"
[6]: https://petsc.org/release/manualpages/SNES/SNESLineSearchSetPreCheck/ "SNESLineSearchSetPreCheck — PETSc 3.24.5 documentation"
[7]: https://petsc.org/release/manualpages/Mat/MatCreateAIJ/ "MatCreateAIJ — PETSc 3.24.5 documentation"
[8]: https://petsc.org/release/manualpages/PC/PCASM/ "PCASM — PETSc 3.24.5 documentation"
[9]: https://petsc.org/release/manualpages/PC/PCLU/ "PCLU — PETSc 3.24.5 documentation"
[10]: https://petsc.org/release/manualpages/SNES/SNESSetConvergenceTest/ "SNESSetConvergenceTest — PETSc 3.24.5 documentation"
[11]: https://petsc.org/release/manualpages/SNES/SNESSetFunctionDomainError/ "SNESSetFunctionDomainError — PETSc 3.24.5 documentation"
[12]: https://petsc.org/release/manualpages/SNES/SNESKSPSetUseEW/ "SNESKSPSetUseEW — PETSc 3.24.5 documentation"
[13]: https://petsc.org/release/manualpages/SNES/SNESSetLagJacobian/ "SNESSetLagJacobian — PETSc 3.24.5 documentation"
[14]: https://petsc.org/release/manualpages/DMDA/DMDAGetCorners/ "DMDAGetCorners — PETSc 3.24.5 documentation"
