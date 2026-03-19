# paper_v1_code_architecture_and_coding_guideline_final

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定以下内容：

1. 项目目录 -> 分类子目录 -> 模块代码 的三层结构
2. 最终采用的代码主路线
3. 各分类子目录的职责边界
4. 每个模块代码文件应承担的功能
5. outer / inner / PETSc / MPI 在代码中的明确落点

配套的“模块代码 -> 指导文件”快速索引见：

- `paper_v1_module_guideline_dependency_map_final.md`

文档优先级与 superseded 关系见：

- `../DOCUMENT_PRIORITY_AND_SUPERSEDED_NOTICE.md`

本文档的目标是给出一套**可直接开工**的工程结构，而不是继续停留在方法级抽象。

---

## 2. 最终定稿的总原则

### 2.1 顶层分类

`paper_v1` 的正式源码根目录为 `src/`。  
`src/` 下直接放置以下分类子目录：

- `core`
- `properties`
- `physics`
- `assembly`
- `solvers`
- `driver`
- `io`
- `parallel`

不再建立 `src/paper_v1/` 这一层。

### 2.2 PETSc / 串并行路线

正式采用：

- **单一代码路线**
- **从第一版起就采用 PETSc 语义**
- **串行开发基线 = MPI size 1 的并行架构**

也就是说：

- 不再维护一套 `SciPy serial` 主路线
- 不再为并行单独再开一个平行代码目录
- 单进程运行时，也必须走与未来 MPI 相同的 `DM / global-local / owned-rows / fieldsplit IS` 语义

补充解释：

- 首个可跑通 case 可以先采用 debug linear profile，例如 `preonly + lu`
- 但这**不等于**允许删除 block layout、`DMComposite`、fieldsplit IS 导出能力
- 正式要求是：**生产配置可以后收紧，代码骨架不能后补**

这条原则是为了避免旧项目那种“串并行代码混在一起、又互相绕过”的混乱。

### 2.3 outer / inner 的代码边界

正式采用：

- `outer` 是项目级时间推进结构
- `inner` 是 PETSc `SNES` 管理的 fixed-geometry nonlinear solve

因此：

- `outer` 由 `solvers/` 内的显式模块实现
- `inner` 由 `solvers/petsc_snes.py` 实现
- PETSc 不负责 outer predictor-corrector

### 2.4 多组分液相要求

正式要求：

- 初版基线 case 可以是单组分液滴
- 但代码结构**必须从第一版起支持多组分液相**
- 任何核心模块都不得写死成 `Nl = 1`

必须满足：

- `layout.py` 支持 `Y_l,red`
- `state_pack.py` 保留 `Yl_full`、`Yg_full`、`Ys_l_full`、`Ys_g_full`
- `flux_liq.py` 支持多组分液相扩散
- `interface_mass.py` 支持 `Eq.(2.15)` 的液相界面组分方程
- `assembly/residual_liquid.py` 与 `assembly/residual_interface.py` 必须支持液相多组分行

### 2.5 配置 schema 路线

正式采用：

- `paper_v1` **只接受新 schema**
- 不提供旧项目 YAML 的兼容层

因此，`core/config_loader.py` 必须：

- 只接受 `case / geometry / grid / unknowns / liquid_db / gas_mech / interface_equilibrium / time_stepper / outer_stepper / solver_inner_petsc / recovery / diagnostics / output / validation`
- 显式拒绝旧项目风格的顶层块，例如 `physics / species / petsc / io / paths`
- 未知顶层键直接报错，而不是尝试自动翻译

### 2.6 冻结的公共 dataclass 命名

`core/types.py` 中以下公共 dataclass 名称正式冻结，不再允许在实现过程中随意改名：

- `RunConfig`
- `Mesh1D`
- `GeometryState`
- `State`
- `Props`
- `StepContext`

正式禁止把它们在后续实现中又改回：

- `CaseConfig`
- `Grid1D`
- 或其他临时命名

---

## 3. 项目目录结构

正式目录结构如下：

```text
paper_v1/
  md/
    physics and numerical/
  mechanism/
  data/
  cases/
  tests/
  src/
    core/
    properties/
    physics/
    assembly/
    solvers/
    driver/
    io/
    parallel/
```

说明：

- `md/physics and numerical/`：方法合同
- `mechanism/`：气相机理
- `data/`：液相数据库、UNIFAC 数据、参考参数
- `cases/`：案例输入
- `tests/`：测试
- `src/`：正式源码

---

## 4. 分类子目录与模块代码

### 4.1 `core/`

`core/` 是中性基础设施层，负责配置、类型、layout、网格、状态转移与基础日志。

正式模块：

- `types.py`
  - 核心 dataclass
  - `RunConfig`
  - `Mesh1D`
  - `GeometryState`
  - `State`
  - `Props`
  - `StepContext`
  - 其中 `Props` 只表示 bulk cell-centered properties
  - 界面真相源单独使用 `InterfaceFacePackage`，不与 `Props` 混用
  - `State` 必须显式保留 full-order 状态字段：
    - `Yl_full`
    - `Yg_full`
    - `Ys_l_full`
    - `Ys_g_full`
- `config_schema.py`
  - schema 定义
- `config_loader.py`
  - 配置读取与校验
  - 只接受新 schema
  - 显式拒绝 legacy YAML 顶层块
- `preprocess.py`
  - normalization
  - derived config
  - closure species / active species / guardrails
  - 保持 MPI-ready / fieldsplit-ready 所需的 block/layout 派生信息
- `layout.py`
  - unknown layout
  - field offsets
  - species maps
  - interface block offsets
- `grid.py`
  - 初始建网格
  - rebuild 网格
  - 几何量
  - control-surface geometry helpers
- `state_pack.py`
  - `State <-> PETSc Vec`
  - phase/interface views
  - reduced unknown views 与 full-order state fields 的双向映射
  - 不允许把 `State` 缩减成只保存 reduced unknown
- `remap.py`
  - conservative remap
  - newly exposed subvolume completion
  - `old_state_on_current_geometry`
- `state_recovery.py`
  - `rho, rhoY, rhoh -> rho, Y, h, T`
  - 液相焓反解
  - 气相 HPY/fallback 恢复
  - recovery post-check
- `logging_utils.py`
  - 统一日志/诊断入口
  - `run.log`
  - `step_diag.csv`
  - `interface_diag.csv`
  - failure forensic records

边界：

- `core/` 不定义物理通量
- `core/` 不定义物性闭合
- `core/` 不写 residual/Jacobian
- `remap.py + state_recovery.py` 共同组成 state-transfer 子系统

### 4.2 `properties/`

`properties/` 负责所有物性、混合规则、数据库访问和相平衡闭合。

正式模块：

- `gas.py`
  - Cantera 气相 thermo / transport
- `liquid_db.py`
  - 液相数据库读取
- `liquid.py`
  - 纯组分与混合液相物性
- `mix_rules.py`
  - 液相混合规则
- `equilibrium.py`
  - 界面相平衡
  - 单组分/多组分平衡关系
- `aggregator.py`
  - 从 state + models 构造统一 `Props`
  - 只产出 bulk cell-centered `Props`

边界：

- `properties/` 不知道 residual 行号
- `properties/` 不装配 residual/Jacobian
- `equilibrium.py` 只给热力学闭合和界面平衡信息
- `aggregator.py` 不允许顺手计算或缓存一套独立的界面物性/界面通量

### 4.3 `physics/`

`physics/` 负责局部物理项。  
它是“物理公式层”，不是“全局代数系统装配层”。

正式模块：

- `initial.py`
  - 初值剖面
  - 第一步初始化辅助
- `flux_gas.py`
  - 气相扩散通量
- `flux_liq.py`
  - 液相扩散通量
  - 必须支持多组分液相
- `flux_convective.py`
  - 对流项
  - 相对速度 `u - v_c`
- `energy_flux.py`
  - 导热与扩散焓项
- `interface_face.py`
  - 唯一 `InterfaceFacePackage`
  - 界面 full composition
  - 界面状态 / 通量 / 几何
  - 基于 `State + property models` 在界面位置计算界面态与界面派生量
- `interface_mass.py`
  - `Eq.(2.15)`
  - `Eq.(2.16)`
  - `Eq.(2.18)`
  - `Eq.(2.19)`
  - 产出界面质量相关行物理项
- `interface_energy.py`
  - `Eq.(2.17)`
  - 产出界面能量行物理项
  - 只允许固定的 Eq.(2.17) / 总能量通量连续主线
  - 不允许 sat/boil regime 切换或 hard pin `T_s`
- `velocity_recovery.py`
  - 连续方程恢复速度
  - gas-side `Eq.(2.18)` 强施加边界起点
- `radius_update.py`
  - `dot_a_phys` 计算
  - 为 outer corrector 提供半径推进物理量

边界：

- `physics/` 不持有全局 `F(u)` / `J(u)`
- `interface_face.py` 是界面物理唯一真相源
- `interface_mass.py` 和 `interface_energy.py` 只返回行物理项，不直接写入全局 residual
- `interface_face.py` 是唯一允许计算界面物性与界面派生通量的模块

### 4.4 `assembly/`

`assembly/` 负责把 `core + properties + physics` 的局部项写成代数系统。

正式模块：

- `residual_liquid.py`
  - 液相 bulk 行
  - 能量与液相组分
- `residual_interface.py`
  - 界面块行
  - 调用 `interface_face.py`、`interface_mass.py`、`interface_energy.py`
  - 组装唯一固定的界面方程结构
  - 不允许 regime-based equation replacement
- `residual_gas.py`
  - 气相 bulk 行
  - 能量与气相组分
- `residual_global.py`
  - 组装全局 residual
- `jacobian_pattern.py`
  - 稀疏 pattern builder
- `jacobian_liquid.py`
  - 液相块 Jacobian
- `jacobian_interface.py`
  - 界面块 Jacobian
- `jacobian_gas.py`
  - 气相块 Jacobian
- `jacobian_global.py`
  - 全局 Jacobian 装配
- `petsc_prealloc.py`
  - PETSc AIJ 预分配

边界：

- `assembly/` 负责“写行写列”
- `assembly/` 不重新定义物理公式
- `assembly/` 不得更新 `dot_a`
- `assembly/` 必须复用 `interface_face.py` 的同一个界面 package
- `assembly/residual_interface.py` 不允许根据 sat/boil 等状态切换另一套行结构

### 4.5 `solvers/`

`solvers/` 负责 inner nonlinear solve、outer predictor-corrector 和 step policy。

正式模块：

- `nonlinear_types.py`
  - 求解结果与状态类型
- `nonlinear_context.py`
  - inner residual / Jacobian 回调上下文
- `linesearch_guards.py`
  - line search 前后 guard
- `petsc_linear.py`
  - KSP / PC 配置
- `petsc_snes.py`
  - fixed-geometry inner `SNES`
- `outer_predictor.py`
  - `a^(0), dot_a^(0)` predictor
- `outer_corrector.py`
  - `a^(k+1), dot_a^(k+1)` corrector
- `outer_convergence.py`
  - `eps_dot_a` 计算与 outer 收敛判断
- `step_acceptance.py`
  - accept/reject
  - `dt` 调整
- `timestepper.py`
  - 单时间步总编排
  - outer-inner 嵌套流程

边界：

- `petsc_snes.py` 只负责 inner
- `outer_predictor.py`、`outer_corrector.py`、`outer_convergence.py`、`step_acceptance.py` 共同覆盖 outer
- `timestepper.py` 只做流程编排，不自己定义物理通量

### 4.6 `driver/`

`driver/` 是案例入口层。

正式模块：

- `run_case.py`
  - CLI / case runner
  - 初始化全流程
  - 循环调用 `solvers/timestepper.py`

边界：

- `driver/` 不携带物理公式
- `driver/` 不持有 residual kernel

### 4.7 `io/`

`io/` 负责输出。

正式模块：

- `output_layout.py`
  - 输出目录组织
  - metadata echo
- `writers.py`
  - CSV / NPZ / mapping / failure report

边界：

- `io/writers.py` 负责落盘，不负责生成 solver diagnostics 逻辑
- 结构化 step/interface/failure 诊断的统一入口仍在 `core/logging_utils.py`

### 4.8 `parallel/`

`parallel/` 负责 PETSc/MPI 语义层。  
这不是第二套物理代码，而是统一主路线中的分布式支持层。

正式模块：

- `mpi_bootstrap.py`
  - MPI / PETSc bootstrap
- `dm_manager.py`
  - `DMDA(liq)`
  - `DMDA(gas)`
  - `DMREDUNDANT(interface)`
  - `DMComposite(global)`
- `local_state.py`
  - global -> local ghost scatter
  - local ghosted views
  - owned / ghost ranges
- `fieldsplit_is.py`
  - fieldsplit IS 构造

边界：

- `parallel/` 只负责并行语义
- `parallel/` 不定义物理公式
- 不允许在 `physics/` 或 `properties/` 中偷偷写 DM scatter / ownership 逻辑

---

## 5. 依赖方向

正式依赖方向为：

```text
core -> properties -> physics -> assembly -> solvers -> driver
                          ^                      |
                          |                      v
                       parallel --------------> io
```

更准确地说：

- `properties/` 依赖 `core`
- `physics/` 依赖 `core + properties`
- `assembly/` 依赖 `core + properties + physics + parallel`
- `solvers/` 依赖 `assembly + core + parallel`
- `driver/` 依赖全部运行模块
- `io/` 只消费状态与诊断量

禁止：

- `physics/` 反向依赖 `assembly/`
- `properties/` 反向依赖 `physics/`
- `core/` 反向依赖 `solvers/`

---

## 6. outer / inner / 并行 覆盖性检查

### 6.1 outer 是否被完整覆盖

是，正式由以下模块共同覆盖：

- `solvers/outer_predictor.py`
- `solvers/outer_corrector.py`
- `solvers/outer_convergence.py`
- `solvers/step_acceptance.py`
- `solvers/timestepper.py`

因此 outer 不再只是 `timestepper.py` 里的隐含逻辑。

### 6.2 inner 是否被完整覆盖

是，正式由以下模块共同覆盖：

- `solvers/nonlinear_context.py`
- `solvers/petsc_snes.py`
- `assembly/residual_*.py`
- `assembly/jacobian_*.py`
- `parallel/local_state.py`

### 6.3 多组分液相是否被完整覆盖

必须覆盖在以下模块中：

- `core/layout.py`
- `core/state_pack.py`
- `properties/liquid.py`
- `properties/mix_rules.py`
- `physics/flux_liq.py`
- `physics/interface_mass.py`
- `assembly/residual_liquid.py`
- `assembly/residual_interface.py`
- `assembly/jacobian_liquid.py`
- `assembly/jacobian_interface.py`

因此，多组分液相不是“以后再补”，而是当前结构的硬约束。

### 6.4 并行是否被完整规划

正式结论：

- **是，且采用统一主路线**
- **不再单开第二套 parallel 代码目录**

覆盖模块为：

- `parallel/mpi_bootstrap.py`
- `parallel/dm_manager.py`
- `parallel/local_state.py`
- `parallel/fieldsplit_is.py`
- `assembly/petsc_prealloc.py`
- `solvers/petsc_linear.py`
- `solvers/petsc_snes.py`

单进程开发时，也必须走同一路线。

---

## 7. Position 1-8 的实际映射

`paper_v1_config_schema_guideline_final.md` 中 Position 1-8，正式映射为：

1. Position 1 `Raw Config File`
   - `cases/<case_id>.yaml`
2. Position 2 `Schema Definition`
   - `src/core/config_schema.py`
3. Position 3 `Schema Validation`
   - `src/core/config_loader.py`
4. Position 4 `Config Normalization`
   - `src/core/preprocess.py`
5. Position 5 `Sub-config Projection`
   - `src/core/types.py`
   - `src/core/preprocess.py`
6. Position 6 `Module Initialization`
   - `src/driver/run_case.py`
   - `src/core/layout.py`
   - `src/core/grid.py`
   - `src/parallel/dm_manager.py`
   - `src/properties/gas.py`
   - `src/properties/liquid_db.py`
7. Position 7 `Runtime Context`
   - `src/solvers/nonlinear_context.py`
   - `src/solvers/timestepper.py`
8. Position 8 `Metadata Echo`
   - `src/io/output_layout.py`
   - `src/io/writers.py`

---

## 8. 单步调用主线

正式主线：

```text
driver/run_case.py
-> core/config_loader.py
-> core/preprocess.py
-> core/layout.py
-> core/grid.py
-> parallel/mpi_bootstrap.py
-> parallel/dm_manager.py
-> parallel/fieldsplit_is.py
-> physics/initial.py
-> properties/aggregator.py
-> solvers/timestepper.py
   -> solvers/outer_predictor.py
   -> loop:
      -> core/remap.py
      -> core/state_recovery.py
      -> parallel/local_state.py
      -> assembly/residual_global.py
      -> assembly/jacobian_global.py
      -> solvers/petsc_snes.py
      -> physics/velocity_recovery.py
      -> physics/radius_update.py
      -> solvers/outer_corrector.py
      -> solvers/outer_convergence.py
   -> solvers/step_acceptance.py
-> io/writers.py
```

关键边界：

- `remap -> recovery` 在 `core/` 内部完成
- `interface_face -> interface_mass/interface_energy -> assembly` 是单向链条
- `parallel/local_state.py` 是 local ghosted view 的唯一入口

---

## 9. 首版最小可运行模块清单

### 9.1 `core/`

- `types.py`
- `config_schema.py`
- `config_loader.py`
- `preprocess.py`
- `layout.py`
- `grid.py`
- `state_pack.py`
- `remap.py`
- `state_recovery.py`
- `logging_utils.py`

### 9.2 `properties/`

- `gas.py`
- `liquid_db.py`
- `liquid.py`
- `mix_rules.py`
- `equilibrium.py`
- `aggregator.py`

### 9.3 `physics/`

- `initial.py`
- `flux_gas.py`
- `flux_liq.py`
- `flux_convective.py`
- `energy_flux.py`
- `interface_face.py`
- `interface_mass.py`
- `interface_energy.py`
- `velocity_recovery.py`
- `radius_update.py`

### 9.4 `assembly/`

- `residual_liquid.py`
- `residual_interface.py`
- `residual_gas.py`
- `residual_global.py`
- `jacobian_pattern.py`
- `jacobian_liquid.py`
- `jacobian_interface.py`
- `jacobian_gas.py`
- `jacobian_global.py`
- `petsc_prealloc.py`

### 9.5 `solvers/`

- `nonlinear_types.py`
- `nonlinear_context.py`
- `linesearch_guards.py`
- `petsc_linear.py`
- `petsc_snes.py`
- `outer_predictor.py`
- `outer_corrector.py`
- `outer_convergence.py`
- `step_acceptance.py`
- `timestepper.py`

### 9.6 `driver/io/parallel`

- `driver/run_case.py`
- `io/output_layout.py`
- `io/writers.py`
- `parallel/mpi_bootstrap.py`
- `parallel/dm_manager.py`
- `parallel/local_state.py`
- `parallel/fieldsplit_is.py`

---

## 10. 正式禁止事项

1. 再开一套与主路线并行的 `serial-only` 代码树。
2. 把并行逻辑散落进 `physics/` 或 `properties/`。
3. 在 `assembly/` 中重新推导界面物理公式。
4. 在 `solvers/petsc_snes.py` 中直接实现 outer predictor-corrector。
5. 在单组分基线阶段删除 `Yl_full` / `Ys_l_full` 的统一接口字段。
6. 让任何模块假定 `Nl = 1` 才能工作。
7. 让 `timestepper.py` 越权替代 `radius_update.py`、`outer_corrector.py`、`step_acceptance.py`。
8. 在 `core/config_loader.py` 中偷偷兼容旧项目 YAML 顶层命名体系。
9. 让 `properties/aggregator.py` 计算第二套界面物性/界面通量。
10. 在 `physics/interface_energy.py` 或 `assembly/residual_interface.py` 中引入 regime-based equation replacement。

---

## 11. 最终结论

`paper_v1` 的最终代码结构正式定稿为：

- 项目目录层：`md / mechanism / data / cases / tests / src`
- 分类子目录层：`core / properties / physics / assembly / solvers / driver / io / parallel`
- 模块代码层：按本文件第 4 节所列模块文件实施

核心结论：

- 多组分液相从结构上第一天就支持
- outer 功能由独立 `solvers` 子模块完整覆盖
- PETSc 与 MPI 从第一版起统一规划为单一路线
- 不再维护第二套串并行分裂代码结构

这就是 `paper_v1` 的最终代码结构与模块边界定稿版本。
