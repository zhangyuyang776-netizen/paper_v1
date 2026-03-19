下面给你一份可直接复制到本地的 **Markdown 方案文件**。
我按“**先定总体路线，再逐模块实现**”来写，重点是 **开发顺序、阶段目标、关键约束、验收标准**，不去发散重划架构，免得又把已经定稿的东西重新搅成泥。

---

# `paper_v1` 新代码开发总体路线方案

## 1. 文档目的

本文档用于统一 `paper_v1` 新代码项目的开发路线，明确：

1. 代码编写的总体阶段划分与实施顺序
2. 各阶段的目标、关键约束与最小验收标准
3. 各模块在开发时必须遵守的物理与数值边界
4. 逐模块实现前的统一执行口径

本文档是“**先定编写方案，再开始逐模块实现**”的总路线文件，不直接展开具体代码实现细节。

---

## 2. 正式约束来源与执行优先级

### 2.1 唯一正式约束来源

`paper_v1` 的物理与数值主线，**只以本次上传的指导文件为正式约束来源**。旧项目 `code/` 目录只能用于：

* 借鉴工程组织方式
* 借鉴已有实现经验
* 参考某些工具函数风格

但**不得反向定义** `paper_v1` 的物理、数值、未知量、迭代结构、并行语义和模块边界。

### 2.2 文件优先级

执行时按以下优先级理解：

1. **专题定稿文件优先于总纲类文件**
2. **`*_final_v2.md` 优先于同名旧版 `*_final.md`**
3. **代码结构与模块依赖以架构/依赖图文件为准**
4. 若总纲与专题文件措辞略有差异，以**专题定稿文件**为准

### 2.3 本方案主要依赖的指导文件

以下文件是本方案的主要依据：

* `PAPER_V1_ROADMAP.md`
* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `unknowns_strategy_guideline_final.md`
* `interface_block_unknowns_and_residuals_guideline_final.md`
* `velocity_recovery_guideline_final.md`
* `outer_inner_iteration_coupling_guideline_final.md`
* `remap_and_conservative_projection_guideline_final_v2.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `liquid_properties_guideline_final.md`
* `timestep_and_failure_policy_guideline_final_v2.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`
* `paper_v1_config_schema_guideline_final.md`
* `paper_v1_code_architecture_and_coding_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md`

---

## 3. 项目总目标与当前范围

## 3.1 项目目标

`paper_v1` 是旧液滴蒸发/燃烧项目的重构版本，目标是建立一套：

* 物理主线清晰
* 数值主线稳定
* 自第一天起遵守 PETSc/MPI 并行语义
* 支持后续扩展至多组分液滴与 chemistry
* 能与主参考文献数值路线一致对齐

的新代码体系。

### 3.2 当前阶段范围

当前阶段只实现 **transport-only** 主线：

* 暂不做 chemistry splitting
* 暂不做辐射
* 正式采用 PETSc 路线
* 串行开发基线也必须是 **MPI size = 1** 的并行语义
* 代码结构必须从第一版起支持**多组分液相**

---

## 4. 全项目必须遵守的总约束

## 4.1 outer-inner 两层职责必须严格分离

依据：`outer_inner_iteration_coupling_guideline_final.md`

### outer 负责

* `a / Rd`
* `dot_a`
* 几何更新
* moving mesh 几何构造
* predictor / corrector
* remap
* accept / reject
* 时间步失败回退

### inner 负责

* fixed geometry 下的 fully implicit transport nonlinear solve
* bulk + interface 残差
* fixed geometry 下的 Jacobian
* velocity recovery
* fixed geometry 下的物性与界面通量闭合

### 强约束

* **PETSc 只负责 inner**
* **outer 逻辑不能塞进 `petsc_snes.py`**
* **inner 不能更新几何**
* **outer 不能绕过 inner 去直接替代传输求解**

---

## 4.2 `Rd / a` 不进入 inner 主未知量

依据：`outer_inner_iteration_coupling_guideline_final.md`

* 半径 `a`（或 `Rd`）不属于 inner unknown block
* `dot_a` 是 outer 冻结给 inner 的量
* inner 收敛后，只负责给出 `dot_a_phys`
* outer corrector 再根据 `dot_a_phys` 更新几何

---

## 4.3 `mpp` 只有一个界面未知量

依据：

* `unknowns_strategy_guideline_final.md`
* `interface_block_unknowns_and_residuals_guideline_final.md`
* `governing_equations_discretization_and_bc_guideline_final_v2.md`

执行口径：

* 只保留 **1 个 `mpp` 界面 unknown**
* `mpp` 的主残差取 **liquid-side Eq.(2.18)**
* gas-side Eq.(2.18) 不再作为第二条独立 residual row
* gas-side Eq.(2.18) 通过**界面边界质量流率强施加**
* gas-side 仍保留一致性 diagnostic

### 强约束

* 不允许把一个 `mpp` 同时对应两条独立标量残差
* 不允许重新引入“双 `mpp`”或“liquid/gas 两个蒸发通量未知量”

---

## 4.4 时间项统一采用 current geometry backward Euler

依据：`remap_and_conservative_projection_guideline_final_v2.md`

执行口径：

* 时间项写在 **current geometry** 上
* `old_state_on_current_geometry` 通过 conservative remap 得到
* inner 时间项唯一合法旧状态输入就是 `old_state_on_current_geometry`

### 强约束

禁止再使用旧写法：

* `Phi(old, *) * V(old, *)`
* 旧几何体积直接进当前时间项
* 未经 remap 的旧状态直接代入 current geometry residual

---

## 4.5 remap 与 recovery 必须分开

依据：

* `remap_and_conservative_projection_guideline_final_v2.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`

执行口径：

* `remap` 处理守恒量
* `state_recovery` 负责从守恒量恢复状态量
* recovery 后形成 inner 可用的旧状态

### 强约束

* 不允许把 remap 直接做成“状态量插值”
* 不允许跳过 recovery 直接拿守恒量当状态量参与物性计算

---

## 4.6 并行逻辑必须集中管理

依据：`petsc_solver_and_parallel_architecture_guideline_final.md`

执行口径：

* 正式采用 DM / Vec / Mat / KSP / SNES 语义
* 物理 kernel 不感知 MPI ownership
* 并行相关逻辑集中在 `parallel/` 目录

### 强约束

* 不允许把 ownership、ghost scatter、rank 分支散落进 `physics/` 或 `properties/`
* 不允许写成传统全局 numpy 串行脚本，再试图“事后并行化”

---

## 4.7 液相从第一版起必须支持多组分结构

依据：

* `unknowns_strategy_guideline_final.md`
* `liquid_properties_guideline_final.md`

执行口径：

* 首个运行 case 可以是单组分液滴
* 但代码结构必须支持多组分液相
* 不能把液相 unknown、liquid property、liquid flux、interface liquid composition 写死为单组分

### 强约束

* 不允许把液相数组 shape、布局、recovery、property eval 写成单组分硬编码
* 不允许以“先做单组分，后面再改”为由破坏 layout 设计

---

## 5. 总体开发阶段与顺序

本项目总体按以下顺序推进：

**Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7**

其中推荐的大顺序为：

**`core -> properties -> physics -> assembly -> solvers -> parallel -> driver/io`**

说明：

* 这是**代码落地顺序**
* 并不表示前期可以忽略并行语义
* 从 `core` 阶段起，接口设计就必须保持 **MPI-ready / PETSc-ready**

---

# Phase 0：冻结执行口径与公共合同

## 5.0.1 目标

在正式写代码前，冻结以下公共合同：

1. 文件优先级
2. unknown baseline
3. outer-inner 数据交换格式
4. interface block 执行口径
5. 配置 schema 执行口径
6. PETSc 真相源与 local view 规则

## 5.0.2 关键约束

必须先统一以下事项：

* 只认新指导文件，不回退旧方法
* 只认新 config schema，不混用旧项目 YAML 语义
* `mpp` 单 unknown 路线写死
* `Rd` 不进 inner
* `outer` 不进 `petsc_snes.py`
* `parallel` 是运行时机制，不是 physics 逻辑的一部分

## 5.0.3 最小验收标准

达到以下条件，才进入 Phase 1：

1. 已明确所有 `*_final_v2.md` 的优先级高于旧版
2. 已明确 config 只按新 schema 解析
3. 已明确 `U_A / U_B` 的 unknown baseline
4. 已明确 outer/inner 输入输出边界
5. 已明确 interface 和 velocity recovery 的耦合关系
6. 已明确 global state 真相源为 PETSc Vec，而不是全局 numpy 数组

---

# Phase 1：`core/` 基础骨架阶段

## 5.1.1 目标

建立全项目统一的数据骨架、配置系统、网格结构、状态打包机制、remap/recovery 主线。

这是全项目最先必须完成的阶段。
如果这一阶段没钉死，后面的物理、装配和求解器都会不断返工，仿佛返工本身是研究目标。

## 5.1.2 建议开发顺序

1. `core/types.py`
2. `core/config_schema.py`
3. `core/config_loader.py`
4. `core/preprocess.py`
5. `core/layout.py`
6. `core/grid.py`
7. `core/state_pack.py`
8. `core/remap.py`
9. `core/state_recovery.py`

可并行补充：

* `core/logging_utils.py`

## 5.1.3 本阶段核心职责

### `core/types.py`

职责：

* 定义核心 dataclass / typed container
* 冻结 state / geometry / context / config 的程序内表示

要点约束：

* 类型命名在此阶段一次性冻结
* state 中必须保留 full composition 视图
* 不允许只保留 reduced unknown 而丢掉 full state

### `core/config_schema.py`

职责：

* 定义新项目唯一合法配置结构

要点约束：

* 只支持新 schema
* 不兼容旧项目键名体系
* 所有关键块必须能在 schema 层检查

### `core/config_loader.py`

职责：

* 读取 YAML
* 做 schema 校验
* 报告缺失项与非法项

要点约束：

* 配置错误必须早失败
* 禁止默默吞掉未知字段

### `core/preprocess.py`

职责：

* 配置归一化
* 生成派生设置
* 建立 species map、closure map、输出路径等

要点约束：

* 不能把“默认值推断”写成隐式魔法
* 影响物理主线的派生规则必须显式记录

### `core/layout.py`

职责：

* 定义 `U_A / U_B`
* 建立 bulk/interface/global 的 slice、index、mapping

要点约束：

* `Rd` 不在 layout 中
* 液相必须支持多组分扩展
* interface block 必须兼容 `Ts + Ys_g_red + mpp (+ 可选 Ys_l_red)` 的策略

### `core/grid.py`

职责：

* 建立三段网格与当前几何量
* 支持 outer 阶段重建网格
* 提供几何体积、面积、`v_c` 所需支持

要点约束：

* 网格对象必须明确区分液相、界面、气相区域
* 几何量必须绑定 current geometry
* 不能混入旧几何量

### `core/state_pack.py`

职责：

* `State <-> PETSc Vec`
* global/local block 视图
* full/reduced 转换入口

要点约束：

* 真相源是 Vec
* 不允许另搞一套 shadow 全局 numpy 真相源

### `core/remap.py`

职责：

* 做 conservative remap
* 构建 `old_state_on_current_geometry`

要点约束：

* 按 phase 做守恒 remap
* 不跨相 remap
* 新暴露子体积要有明确补全策略

### `core/state_recovery.py`

职责：

* 从守恒量恢复 `rho / Y / h / T`
* 完成液相/气相焓反解

要点约束：

* recovery 与 remap 逻辑必须分离
* enthalpy inversion 失败必须可诊断

## 5.1.4 本阶段关键约束

* `layout` 设计必须自第一天支持多组分液相
* `state_recovery` 必须能处理 full state
* `old_state_on_current_geometry` 必须成为 inner solve 的显式输入对象
* 所有数据结构都要兼容 PETSc/MPI size=1 语义

## 5.1.5 最小验收标准

达到以下标准，Phase 1 才算完成：

1. 能读取并通过一份新 schema case 文件
2. `UnknownLayout` 能正确生成单组分 case 与多组分 case 的布局
3. `Grid` 能构建初始几何并支持 outer 几何更新后的重建
4. `State` 可以稳定 pack/unpack 到 global Vec
5. remap 在“几何不变”时退化为恒等
6. remap + recovery 后可生成合法的 `old_state_on_current_geometry`
7. 类型、布局、配置系统不再反复改接口

---

# Phase 2：`properties/` 物性与平衡闭合阶段

## 5.2.1 目标

建立气相、液相、混合规则、界面平衡的统一物性层，为后续 physics 和 assembly 提供稳定输入。

## 5.2.2 建议开发顺序

1. `properties/liquid_db.py`
2. `properties/mix_rules.py`
3. `properties/liquid.py`
4. `properties/gas.py`
5. `properties/equilibrium.py`
6. `properties/aggregator.py`

## 5.2.3 本阶段核心职责

### `properties/liquid_db.py`

职责：

* 管理液相纯组分数据库
* 提供参数查询

要点约束：

* 数据库层与物性计算层分离
* 不能把相关式直接散写到 physics 中

### `properties/mix_rules.py`

职责：

* 液相混合规则
* 质量分数/摩尔分数转换
* mixture property 聚合

要点约束：

* 混合规则必须能处理多组分液相
* 不允许只按单组分接口写

### `properties/liquid.py`

职责：

* 液相体相物性
* 界面附近液相物性支持
* 液相焓模型

要点约束：

* 必须支持多组分液相
* 与 `state_recovery` 中 enthalpy inversion 接口保持一致

### `properties/gas.py`

职责：

* 气相热力学、传输性质的统一封装
* 面向 bulk property 计算

要点约束：

* closure species 处理规则必须清晰
* full composition 与 reduced unknown 必须明确区分

### `properties/equilibrium.py`

职责：

* 执行界面平衡闭合
* 给出 `Ys_g_eq_full` 等平衡结果

要点约束：

* 只负责 Eq.(2.19) 的热力学闭合
* 不应顺带接管界面整个 residual 逻辑

### `properties/aggregator.py`

职责：

* 基于 `State + Grid + Models` 统一构造 bulk `Props`

要点约束：

* 推荐只聚合 **bulk props**
* interface-specific 的界面态与界面通量派生值，应由 `physics/interface_face.py` 统一构造
* 不要形成“两套界面物性来源”

## 5.2.4 本阶段关键约束

* 液相模型必须多组分友好
* 界面平衡与 bulk property 逻辑分开
* 不允许在 physics 模块里私自重算一套不受控的物性
* property 层不能感知并行 ownership 细节

## 5.2.5 最小验收标准

1. 给定合法 `State` 和 `Grid`，能稳定求出 bulk `Props`
2. 气相/液相关键物性为有限正值
3. `equilibrium.py` 能给出界面平衡输出
4. 单组分 case 与多组分 case 均通过 shape 和字段检查
5. property 接口稳定，后续 physics 无需改动 property API

---

# Phase 3：`physics/` 物理 kernel 阶段

## 5.3.1 目标

建立所有 fixed geometry 下的物理 kernel，包括：

* 初始场
* bulk flux
* interface face package
* interface mass/energy term
* velocity recovery
* radius update 所需物理量

这一阶段**只写物理 kernel**，不直接装全局 residual。

## 5.3.2 建议开发顺序

1. `physics/initial.py`
2. `physics/flux_liq.py`
3. `physics/flux_gas.py`
4. `physics/flux_convective.py`
5. `physics/energy_flux.py`
6. `physics/interface_face.py`
7. `physics/interface_mass.py`
8. `physics/interface_energy.py`
9. `physics/velocity_recovery.py`
10. `physics/radius_update.py`

## 5.3.3 本阶段核心职责

### `physics/initial.py`

职责：

* 构造初始状态
* 支持首步 `mpp^0 = 0`、`dot_a^0 = 0`

要点约束：

* 按文献初始化主线执行
* 不允许把界面气相初值强行改成平衡态，除非指导文件明确要求

### `physics/flux_liq.py`

职责：

* 液相扩散通量 kernel

要点约束：

* 必须支持多组分液相
* 不能写死成单组分导热+无组分扩散

### `physics/flux_gas.py`

职责：

* 气相扩散通量 kernel

要点约束：

* 与气相组分 layout 对齐
* closure species 处理与 reduced/full state 保持一致

### `physics/flux_convective.py`

职责：

* 基于 `u - v_c` 计算对流通量

要点约束：

* moving control volume 语义必须正确
* `v_c` 来源于 outer 冻结几何

### `physics/energy_flux.py`

职责：

* 导热项
* 物种扩散焓通量
* 总能量通量支持

要点约束：

* 不能把能量方程简化成只剩导热
* 要兼容多组分扩散焓项

### `physics/interface_face.py`

职责：

* 统一构造 **唯一的 `InterfaceFacePackage`**

要点约束：

* 这是界面真相源
* liquid bulk、interface residual、gas bulk 必须共享这一套界面包
* 不允许不同 residual 各算一套界面态

### `physics/interface_mass.py`

职责：

* 构造界面质量与组分守恒相关物理项

要点约束：

* `mpp` 主残差来自 liquid-side Eq.(2.18)
* gas-side Eq.(2.18) 通过边界质量流率强施加反映
* gas-side 仍保留 diagnostic

### `physics/interface_energy.py`

职责：

* 构造界面能量守恒物理项

要点约束：

* 与 interface face package 共用界面状态
* 不允许另起一套界面温度/物性来源

### `physics/velocity_recovery.py`

职责：

* 基于连续方程恢复 `u_l / u_g`
* 提供 face mass flux/velocity

要点约束：

* 必须吃 current trial state
* 必须使用 `old_mass_on_current_geometry`
* gas-side 起点必须使用强施加的界面边界质量流率
* 必须体现 interface 与 velocity recovery 的耦合

### `physics/radius_update.py`

职责：

* 根据 inner 收敛界面状态计算 `dot_a_phys`

要点约束：

* 只负责计算物理解
* 不负责 corrector 更新几何

## 5.3.4 本阶段关键约束

* `interface_face.py` 必须成为唯一界面真相源
* `velocity_recovery.py` 是 residual 装配前的必要支持，不是事后 diagnostic
* `radius_update.py` 不能取代 outer corrector
* 物理 kernel 不感知 rank ownership

## 5.3.5 最小验收标准

1. 能生成首步物理解初值
2. `InterfaceFacePackage` 能服务液相、界面、气相三方
3. `velocity_recovery.py` 能恢复满足边界条件的速度场
4. `dot_a_phys = u_l|if + mpp / rho_s,l` 能独立算出
5. 所有 physics kernel 在单组分 case 与多组分 case 下接口一致

---

# Phase 4：`assembly/` 残差与 Jacobian 装配阶段

## 5.4.1 目标

将 physics kernel 组装成 fixed geometry 下的全局 nonlinear system，包括：

* liquid residual
* interface residual
* gas residual
* global residual callback
* Jacobian sparsity pattern
* PETSc 预分配
* 后续 Jacobian 各块装配

## 5.4.2 建议开发顺序

### 先写 residual 主线

1. `assembly/residual_liquid.py`
2. `assembly/residual_interface.py`
3. `assembly/residual_gas.py`
4. `assembly/residual_global.py`

### 再写 pattern / prealloc

5. `assembly/jacobian_pattern.py`
6. `assembly/petsc_prealloc.py`

### 最后补 Jacobian 分块装配

7. `assembly/jacobian_liquid.py`
8. `assembly/jacobian_interface.py`
9. `assembly/jacobian_gas.py`
10. `assembly/jacobian_global.py`

## 5.4.3 本阶段核心职责

### `assembly/residual_liquid.py`

职责：

* 组装液相 bulk residual rows

要点约束：

* 时间项使用 `old_state_on_current_geometry`
* 当前几何体积是唯一合法体积
* 必须支持多组分液相 bulk 方程

### `assembly/residual_interface.py`

职责：

* 组装 interface block residual rows

要点约束：

* `mpp` 主残差是 liquid-side Eq.(2.18)
* 不允许将 gas-side Eq.(2.18) 再加成第二条独立标量残差
* `Ts / Ys_g_red / mpp` 及必要时 `Ys_l_red` 的布局必须与 `layout.py` 一致

### `assembly/residual_gas.py`

职责：

* 组装气相 bulk residual rows

要点约束：

* 界面边界项来自统一 interface package
* gas-side 质量流率强施加必须与 velocity recovery 起点一致

### `assembly/residual_global.py`

职责：

* 将 liquid/interface/gas 三部分统一组装成全局 residual

要点约束：

* 统一调用 local view
* 只装 owned rows
* 不依赖全局 numpy 真相源

### `assembly/jacobian_pattern.py`

职责：

* 定义全局 Jacobian 的稀疏图结构

要点约束：

* 必须显式体现 liquid/interface/gas block 耦合
* 不能临时靠运行时插值猜 pattern

### `assembly/petsc_prealloc.py`

职责：

* 依据 pattern 做 PETSc Mat 预分配

要点约束：

* 必须服务正式 PETSc 路线
* 不允许跳过预分配直接胡乱插值

### `assembly/jacobian_*`

职责：

* 分块装配 Jacobian

要点约束：

* 初期可分阶段推进
* 但接口必须先留好
* Jacobian 的变量顺序、块边界必须与 `layout.py` 和 `jacobian_pattern.py` 完全一致

## 5.4.4 本阶段关键约束

* `Rd` 不得重新进入 layout 或 residual
* interface residual 与 velocity recovery 的耦合必须一致
* bulk 与 interface 必须共享同一 interface package
* 不允许在 residual 中偷偷重算一套与 physics 不一致的界面状态

## 5.4.5 最小验收标准

1. 全局 residual 长度严格等于 global unknown 长度
2. 单组分 case 下 interface block 行数正确
3. 多组分液滴 case 下液相与界面布局能自动扩展
4. Jacobian sparsity pattern 可显式导出
5. residual callback 完全遵守 current geometry + remapped old state 主线

---

# Phase 5：`solvers/` 求解器与时间推进主干阶段

## 5.5.1 目标

建立：

* inner fixed-geometry nonlinear solve
* outer predictor-corrector
* outer convergence 判断
* accept/reject 机制
* 单步推进调度

这是把前面所有阶段真正串成完整数值流程的阶段。

## 5.5.2 建议开发顺序

1. `solvers/nonlinear_types.py`
2. `solvers/nonlinear_context.py`
3. `solvers/linesearch_guards.py`
4. `solvers/petsc_linear.py`
5. `solvers/petsc_snes.py`
6. `solvers/outer_predictor.py`
7. `solvers/outer_corrector.py`
8. `solvers/outer_convergence.py`
9. `solvers/step_acceptance.py`
10. `solvers/timestepper.py`

## 5.5.3 本阶段核心职责

### `solvers/nonlinear_types.py`

职责：

* 定义 inner/outer/step 的结果类型与状态对象

要点约束：

* 所有 solver 结果都必须结构化
* 不能只返回散装 tuple

### `solvers/nonlinear_context.py`

职责：

* 构造 inner solve 所需 fixed geometry 上下文

要点约束：

* 必须包含：

  * current geometry
  * frozen `dot_a`
  * `old_state_on_current_geometry`
  * layout / grid / props / models
* inner context 是 residual/Jacobian 的唯一环境入口

### `solvers/linesearch_guards.py`

职责：

* 检查 trial state 是否越界或进入非法物理域

要点约束：

* 这是 domain guard，不是替代物理模型
* 失败信息必须结构化输出

### `solvers/petsc_linear.py`

职责：

* 配置 KSP / PC / fieldsplit 等线性求解部分

要点约束：

* 与 PETSc 架构文件一致
* 不能在 physics 模块里私配线性求解器

### `solvers/petsc_snes.py`

职责：

* 执行 **fixed geometry** 下的 inner SNES solve

要点约束：

* 只负责 inner
* 不得混入 outer predictor-corrector
* 所有 outer 变量必须视为冻结输入

### `solvers/outer_predictor.py`

职责：

* 每个时间步只执行一次 predictor

要点约束：

* predictor 只给出当前步初始几何猜测
* 不能在 outer 迭代中反复重做 predictor

### `solvers/outer_corrector.py`

职责：

* 根据 `dot_a_phys` 修正 `a^(k+1), dot_a^(k+1)`

要点约束：

* corrector 必须 anchored 到上一 accepted 时间步
* 不允许漂移成“以上一 outer iterate 自己为基准”的错误形式

### `solvers/outer_convergence.py`

职责：

* 判断 outer 是否收敛

要点约束：

* 按指导文件，优先基于 `eps_dot_a`
* 不擅自引入额外收敛判据替代主判据

### `solvers/step_acceptance.py`

职责：

* 处理 accept / reject / rollback / dt 调整

要点约束：

* inner 收敛不等于 step 接受
* outer 不收敛必须继续或 reject
* reject 后必须严格 rollback

### `solvers/timestepper.py`

职责：

* 组织单步推进全过程

要点约束：

* 必须完整体现 outer-inner 分工
* 不得把 outer 逻辑塞入 `petsc_snes.py`
* 必须保留失败策略与重试逻辑

## 5.5.4 本阶段关键约束

* inner solve 与 outer 几何更新职责不可混
* predictor 每步一次
* corrector anchored
* accept/reject 与 inner convergence 不可混为一谈
* timestep failure policy 必须可追踪、可回退

## 5.5.5 最小验收标准

1. 单个时间步可以完整执行 outer-inner 结构
2. `petsc_snes.py` 不包含 outer 更新逻辑
3. predictor 每步只执行一次
4. inner 收敛但 outer 未收敛时，流程不误判 accepted
5. reject 后能正确 rollback 并调整 `dt`
6. 首步能够满足 `mpp^0 = 0`、`dot_a^0 = 0`

---

# Phase 6：`parallel/` 并行管理阶段

## 5.6.1 目标

补齐正式的 PETSc/MPI 并行管理层，包括：

* MPI 启动
* DM 管理
* global/local scatter
* ghost view
* fieldsplit IS 导出

虽然在总顺序中放在 `solvers` 后，但接口语义必须从 Phase 1 就预留。

## 5.6.2 建议开发顺序

1. `parallel/mpi_bootstrap.py`
2. `parallel/dm_manager.py`
3. `parallel/local_state.py`
4. `parallel/fieldsplit_is.py`

## 5.6.3 本阶段核心职责

### `parallel/mpi_bootstrap.py`

职责：

* 启动 MPI / PETSc 环境
* 构建 communicator 基础

### `parallel/dm_manager.py`

职责：

* 构建 liquid / interface / gas 的 DM
* 形成 DMComposite

要点约束：

* interface block 应有清晰的 ownership 策略
* 必须为多 rank 做准备

### `parallel/local_state.py`

职责：

* global Vec 与 local ghosted view 的转换
* 统一 local 视图入口

要点约束：

* physics/assembly 读取 local view
* 不能让每个模块自己乱 scatter

### `parallel/fieldsplit_is.py`

职责：

* 根据 layout 自动导出 block IS

要点约束：

* 不手写魔法索引
* 与 `layout.py` 保持统一

## 5.6.4 本阶段关键约束

* 并行逻辑集中在 `parallel/`
* 物理 kernel 不感知 rank ownership
* residual/Jacobian 装配只负责 owned rows
* interface block ownership 必须明确

## 5.6.5 最小验收标准

1. `mpiexec -n 1` 行为正确
2. local ghosted view 有唯一统一入口
3. owned rows 装配逻辑清晰
4. fieldsplit IS 能由 layout 自动生成
5. 多 rank 扩展时不需要重写 physics/properties

---

# Phase 7：`driver/` 与 `io/` 运行组织与输出阶段

## 5.7.1 目标

建立完整 case 运行入口、输出结构与诊断落盘机制。

## 5.7.2 建议开发顺序

1. `driver/run_case.py`
2. `io/output_layout.py`
3. `io/writers.py`

## 5.7.3 本阶段核心职责

### `driver/run_case.py`

职责：

* 初始化 case
* 调用 preprocess / models / solver / writer
* 控制全流程运行

### `io/output_layout.py`

职责：

* 建立输出目录
* 写 normalized config 回显
* 维护 output metadata

### `io/writers.py`

职责：

* 写 step 诊断
* 写 interface 诊断
* 写 snapshot
* 写 failure report
* 写 mapping 信息

## 5.7.4 本阶段关键约束

* 输出必须服务诊断，而不是只服务“看起来跑了”
* 失败路径必须可取证
* 不能把日志与诊断散落在 solver 里临时拼接

## 5.7.5 最小验收标准

1. 运行时自动建立标准输出目录
2. 写出 normalized config 回显
3. 写出 `step_diag.csv`
4. 写出 `interface_diag.csv`
5. 失败时能写出 failure 取证文件
6. 若启用空间输出，必须写出 `mapping.json`

---

## 6. 各阶段之间的阻塞关系

## 6.1 绝对前置依赖

以下关系必须严格遵守：

* `core/layout.py` 先于所有 assembly / solver
* `core/state_pack.py` 先于 parallel / assembly
* `core/remap.py + core/state_recovery.py` 先于 timestepper
* `properties/*` 先于 physics/*
* `physics/interface_face.py` 先于 residual_interface / residual_gas / residual_liquid
* `physics/velocity_recovery.py` 先于 bulk residual 装配
* `assembly/residual_global.py` 先于 `petsc_snes.py`
* `jacobian_pattern.py + petsc_prealloc.py` 先于正式 Jacobian 装配
* `outer_*` 与 `step_acceptance.py` 先于 `timestepper.py`

## 6.2 可延后但必须留接口的部分

可以稍后补充，但不能不留接口：

* `assembly/jacobian_liquid.py`
* `assembly/jacobian_interface.py`
* `assembly/jacobian_gas.py`
* `assembly/jacobian_global.py`
* `parallel/fieldsplit_is.py` 的性能优化
* 某些扩展型输出字段

---

## 7. 每阶段的里程碑验收

## 7.1 里程碑 M1：基础骨架完成

对应阶段：

* Phase 0
* Phase 1

验收标志：

* 新 schema 可读
* layout 可用
* grid 可建
* remap/recovery 主线闭合
* state 与 Vec 的转换接口稳定

## 7.2 里程碑 M2：物性与物理 kernel 完成

对应阶段：

* Phase 2
* Phase 3

验收标志：

* bulk props 可算
* interface package 可算
* velocity recovery 可算
* 首步物理解可建立

## 7.3 里程碑 M3：inner residual 主线跑通

对应阶段：

* Phase 4（至少 residual + pattern + prealloc）

验收标志：

* 全局 residual callback 跑通
* fixed geometry 下 nonlinear system 结构完整
* residual 各块与 layout 一致

## 7.4 里程碑 M4：outer-inner 单步闭环跑通

对应阶段：

* Phase 5

验收标志：

* 单步推进可执行
* accept/reject 有效
* predictor/corrector 合法
* inner 与 outer 边界清晰

## 7.5 里程碑 M5：PETSc/MPI-ready 基线完成

对应阶段：

* Phase 6
* Phase 7

验收标志：

* `mpiexec -n 1` 全流程可运行
* local/global state 管理清楚
* 输出与失败取证完整

---

## 8. 开发时禁止事项

以下事项在整个开发过程中禁止出现。

### 8.1 禁止回退到 SciPy 路线

* 不再单独维护 serial-only 代码树
* 不再写传统 numpy 全局求解器替代 PETSc 主线

### 8.2 禁止将 outer 逻辑塞回 `petsc_snes.py`

* `petsc_snes.py` 只负责 inner
* 不得加入半径更新、outer predictor、outer corrector

### 8.3 禁止把液相默认成单组分结构

* 首个 case 可以单组分
* 代码结构不能单组分化

### 8.4 禁止把并行逻辑散落进 physics / properties

* ownership / ghost / scatter / rank 判断只出现在 `parallel/`

### 8.5 禁止把 remap 做成状态量插值

* remap 必须是 conservative remap
* recovery 必须显式存在

### 8.6 禁止在 interface 位置维护多套真相源

* `InterfaceFacePackage` 必须唯一
* bulk residual、interface residual、velocity recovery 必须共享界面状态定义

### 8.7 禁止将 `Rd` 重新塞回 inner unknown

* `Rd` 只属于 outer
* inner 只接收 frozen geometry 与 frozen `dot_a`

### 8.8 禁止引入 regime-based equation replacement

* 不允许运行中切换另一套界面方程体系替代当前方程结构
* 不允许沸腾/saturation 分支改写 unknown structure

---

## 9. 推荐的实际执行顺序

建议实际执行时按以下批次推进。

## 第一批：基础骨架

* `core/types.py`
* `core/config_schema.py`
* `core/config_loader.py`
* `core/preprocess.py`
* `core/layout.py`
* `core/grid.py`
* `core/state_pack.py`
* `core/remap.py`
* `core/state_recovery.py`

## 第二批：物性层

* `properties/liquid_db.py`
* `properties/mix_rules.py`
* `properties/liquid.py`
* `properties/gas.py`
* `properties/equilibrium.py`
* `properties/aggregator.py`

## 第三批：物理 kernel

* `physics/initial.py`
* `physics/flux_liq.py`
* `physics/flux_gas.py`
* `physics/flux_convective.py`
* `physics/energy_flux.py`
* `physics/interface_face.py`
* `physics/interface_mass.py`
* `physics/interface_energy.py`
* `physics/velocity_recovery.py`
* `physics/radius_update.py`

## 第四批：残差与装配

* `assembly/residual_liquid.py`
* `assembly/residual_interface.py`
* `assembly/residual_gas.py`
* `assembly/residual_global.py`
* `assembly/jacobian_pattern.py`
* `assembly/petsc_prealloc.py`

## 第五批：求解器主干

* `solvers/nonlinear_types.py`
* `solvers/nonlinear_context.py`
* `solvers/linesearch_guards.py`
* `solvers/petsc_linear.py`
* `solvers/petsc_snes.py`
* `solvers/outer_predictor.py`
* `solvers/outer_corrector.py`
* `solvers/outer_convergence.py`
* `solvers/step_acceptance.py`
* `solvers/timestepper.py`

## 第六批：并行与输出

* `parallel/mpi_bootstrap.py`
* `parallel/dm_manager.py`
* `parallel/local_state.py`
* `parallel/fieldsplit_is.py`
* `driver/run_case.py`
* `io/output_layout.py`
* `io/writers.py`

## 第七批：补齐 Jacobian 与调优

* `assembly/jacobian_liquid.py`
* `assembly/jacobian_interface.py`
* `assembly/jacobian_gas.py`
* `assembly/jacobian_global.py`
* fieldsplit / preconditioner 调优
* 输出增强与诊断扩展

---

## 10. 当前建议的开发起点

如果按本方案执行，下一步应从以下模块开始逐个实现：

1. `core/types.py`
2. `core/config_schema.py`
3. `core/config_loader.py`
4. `core/preprocess.py`
5. `core/layout.py`

理由：

* 这些模块决定全项目的数据骨架
* 一旦它们稳定，后续模块的接口才不会不断返工
* 它们直接决定多组分液相支持、PETSc layout、outer-inner 边界是否能落地

---

## 11. 结论

`paper_v1` 的开发必须坚持以下路线：

1. **先冻结约束，再写代码**
2. **先搭 `core` 骨架，再写物性**
3. **先写 physics kernel，再写 assembly**
4. **先形成 inner 闭环，再接 outer**
5. **全程坚持 PETSc/MPI-ready 语义**
6. **从第一版起支持多组分液相**
7. **绝不回退到旧项目的串行思路**
8. **绝不打乱 outer-inner 分工**
9. **绝不忽略 interface 与 velocity recovery 的耦合**

本路线文件的作用不是追求“最快写出能跑的东西”，而是确保新代码从第一天起沿着**正确、可扩展、可诊断、可并行**的主线推进，避免后期因为基础边界不清而大规模返工。

---

如果你接下来要继续推进，我建议下一步就直接进入：
**“Phase 1：`core/` 逐模块实施方案”**，先从 `core/types.py` 和 `core/config_schema.py` 开始定接口。
