# paper_v1 修改日志

## 2026-03-11

- 新建 `paper_v1` 目录，作为新代码项目总目录。
- 新建 `paper_v1/md` 文档目录。
- 新建指导文件 `PAPER_V1_ROADMAP.md`，给出：
  - 第一版总纲
  - 推荐最终版总纲
  - `Rd` 是否进入主系统的方案比较
  - 从第一版到最终版的阶段划分与难度判断
- 调整路线图中的非线性求解建议：
  - 第一版优先使用 `PETSc SNES`
  - 不把自编 Newton 作为首版主路线
- 调整路线图中的 `Rd` 处理建议：
  - 第一版与推荐最终版都采用外层 moving-mesh / predictor-corrector
  - 不把 `Rd` 直接塞进内层 transport 主未知量
  - 旧项目中简化的 `Rd-mpp` 方程不作为新项目长期主框架
- 根据新增参考文献 `[30]` 更新路线图：
  - 明确 `[30]` 可以支撑 ALE / move-remesh / 固定网格单步求解的思路
  - 同时明确 `[30]` 不是自由球滴问题的 1:1 模板
- 在路线图中新增“界面追踪、moving mesh 与 SNES 的职责边界”章节。
- 在路线图中新增“当前仍未完全明确、尚未规定的地方”章节，并按优先级区分：
  - 开工前必须先定的事项
  - 可以边做边定的事项
  - 当前仍缺的关键参考材料

## 2026-03-14

- 统一 `paper_v1` 中 `mpp / Eq.(2.18)` 的正式实现口径：
  - `mpp` 继续作为单一界面 unknown
  - liquid-side Eq.(2.18) 作为 `mpp` 的主残差
  - gas-side Eq.(2.18) 不再表述为“仅 diagnostic”，而是通过界面边界质量流率被强施加
  - 同时保留 gas-side 一致性诊断
- 在 `governing_equations_discretization_and_bc_guideline_final_v2.md` 中修正 backward Euler 时间项写法：
  - 删除 `\Phi^{old,*} V^{old,*}` 的旧口径
  - 明确 `old_state_on_current_geometry` 使用当前固定几何控制体体积
  - bulk 时间项统一为“当前几何体积上的 new-old 差”
- 在 `velocity_recovery_guideline_final.md` 中同步修正速度恢复边界：
  - 气相 continuity sweep 从由 `mpp + dot_a^(k)` 强施加的界面质量流率出发
  - 速度恢复与 `mpp` 的双侧耦合关系写入正式合同
- 清理指导文件中的生成残留：
  - 删除 `interface_block_unknowns_and_residuals_table_final.md` 头部的生成说明
  - 修正 `interface_block_unknowns_and_residuals_table_final.md` 最终合同中的重复编号
  - 删除 `velocity_recovery_guideline_final.md` 末尾无关的生成对话
- 进一步清理指导文件卫生问题并补足工程骨架：
  - `outer_inner_iteration_coupling_guideline_final.md` 正式改为仅使用 `eps_dot_a` 作为 outer 收敛判据
  - 清理 `paper_v1_config_schema_guideline_final.md`、`unknowns_strategy_guideline_final.md`、`liquid_properties_guideline_final.md`、`diagnostics_and_conservation_monitoring_guideline_final.md`、`petsc_solver_and_parallel_architecture_guideline_final.md` 中的生成前言、尾部建议段和多余围栏
  - 移除 `liquid_properties_guideline_final.md` 中的 `oaicite/contentReference` 占位残留
  - 将 `state_recovery_and_enthalpy_inversion_guideline_final` 统一更名为 `state_recovery_and_enthalpy_inversion_guideline_final.md`
  - 新增 `paper_v1_code_architecture_and_coding_guideline_final.md`，正式给出代码结构、模块边界与 Position 1-8 映射
- 参考旧项目 `code/` 的一级目录骨架，重写 `paper_v1` 代码架构文件：
  - 顶层分类正式采用 `core / properties / physics / assembly / solvers / driver / io / parallel`
  - `remap + recovery` 回收到 `core/` 的 state-transfer 子系统
  - `interface` 从一级分类改为 `physics/` 中的界面物理子模块
  - `assembly/` 明确只负责 residual/Jacobian 装配，不再与 `interface` 职责重叠

## 2026-03-15

- 调整 `paper_v1` 正式源码根目录结构：
  - `src/` 下不再嵌套 `paper_v1/`
  - `src/` 下直接放置 `core / properties / physics / assembly / solvers / driver / io / parallel`
- 新增 `paper_v1_module_guideline_dependency_map_final.md`：
  - 按模块代码级别给出“模块 -> 必须先读的指导文件”快速索引
  - 覆盖 `core / properties / physics / assembly / solvers / driver / io / parallel` 全部首版模块
- 同步修正 `paper_v1_code_architecture_and_coding_guideline_final.md`：
  - 目录树改为 `src/core` 等直列结构
  - 增加对模块指导文件索引的引用
- 同步修正 `petsc_solver_and_parallel_architecture_guideline_final.md`：
  - 删除旧的并行目录草图
  - 改为映射到正式代码结构文件中的 `src/core`、`src/assembly`、`src/solvers`、`src/parallel` 模块
- 进一步将代码结构收紧为最终定稿版本：
  - 明确初版代码必须从结构上支持多组分液相，禁止核心模块假定 `Nl = 1`
  - 将 `physics/` 细化为 `interface_face.py + interface_mass.py + interface_energy.py`
  - 将 `assembly/` 细化为 `residual_liquid/interface/gas` 与 `jacobian_liquid/interface/gas/global`
  - 将 `solvers/` 细化为 `outer_predictor.py`、`outer_corrector.py`、`outer_convergence.py`、`step_acceptance.py`、`timestepper.py`
  - 明确 PETSc/MPI 采用单一路线：单进程开发基线即 `MPI size = 1` 的并行架构，不再维护独立 serial-only 代码树
- 补充文件优先级与 superseded 规则：
  - 新增 `DOCUMENT_PRIORITY_AND_SUPERSEDED_NOTICE.md`
  - 明确根目录旧版 `governing/remap/timestep` 文件只作为历史快照保留
  - 明确同主题下 `md/physics and numerical/*_final_v2.md` 为唯一正式执行版本
- 收紧配置与架构边界：
  - `paper_v1` 配置系统正式改为只接受新 schema，不提供旧项目 YAML 兼容层
  - 在代码架构文件中冻结 `RunConfig / Mesh1D / GeometryState / State / Props / StepContext` 公共 dataclass 命名
  - 明确 `State` / `state_pack` 必须显式保留 `Yl_full / Yg_full / Ys_l_full / Ys_g_full`
- 收紧 properties / interface 边界：
  - `properties/aggregator.py` 只允许构造 bulk cell-centered `Props`
  - `physics/interface_face.py` 作为唯一界面状态、界面物性和界面派生通量真相源
- 收紧界面方程结构约束：
  - `physics/interface_energy.py` 与 `assembly/residual_interface.py` 不允许引入 regime-based equation replacement
  - 明确 sat/boil 切换和 hard pin `T_s` 不得迁回 `paper_v1`
- 补齐模块清单一致性：
  - 在最小可运行模块清单中补回 `core/logging_utils.py`
  - 同步补齐 `solvers/nonlinear_types.py` 与 `solvers/linesearch_guards.py`
- 统一解释 ROADMAP 中“首版不做 MPI / fieldsplit”的旧口径：
  - 改为“首版不要求多 rank 生产运行，也不要求首个 case 立即使用 fieldsplit 生产配置”
  - 但代码骨架必须从第一天起保持 MPI-ready / fieldsplit-ready

## 日志约定

- 每次架构决策变化，都应追加一条记录。
- 每次物理模型变化，都应说明影响范围。
- 每次数值方法变化，都应说明是否影响已有基线结果。
- 每次与参考文献不一致的实现选择，都应在日志中说明理由。
## 2026-03-16

- Added `md/code/CONFIG_PARAMETER_EXTENSION_GUIDE.md` for the standard workflow of adding new configuration parameters.
- Recorded the current audit conclusion for `core` 1-5: `schema -> loader -> preprocess -> layout` is now the stable primary chain.
- Clarified that `paper_v1/cases/config.yaml` is loader/schema-valid, but still needs real external resource files under the case root before preprocess can run successfully.
