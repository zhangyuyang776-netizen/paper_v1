进入 **Phase 1：`core/` 基础骨架阶段**。

这一阶段不追求“能跑一个 case”，而是先把 **数据骨架、配置骨架、layout 骨架、几何骨架、remap/recovery 骨架** 钉死。否则后面每写一个 physics 模块，都得回头改接口，像拿焊枪修积木。

---

# Phase 1 实施方案

## 1. 本阶段目标

Phase 1 的目标只有一个：

> 建立 `paper_v1` 在程序层面的统一“真相结构”，让后续 `properties / physics / assembly / solvers` 都围绕同一套对象和接口工作。

本阶段完成后，必须具备以下能力：

1. 读取并严格校验新 schema 配置
2. 生成标准化运行配置 `RunConfig`
3. 建立 `UnknownLayout`
4. 建立初始几何与网格对象
5. 建立 `State <-> Vec` 的唯一映射
6. 完成 conservative remap
7. 完成 `old_state_on_current_geometry` 的恢复与构造

---

## 2. 本阶段依赖的正式指导文件

按 `paper_v1_module_guideline_dependency_map_final.md`，Phase 1 需要优先服从这些文件：

* `paper_v1_config_schema_guideline_final.md`
* `unknowns_strategy_guideline_final.md`
* `outer_inner_iteration_coupling_guideline_final.md`
* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `remap_and_conservative_projection_guideline_final_v2.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `velocity_recovery_guideline_final.md`
* `paper_v1 Initialization and First-Step Guideline.md`
* `timestep_and_failure_policy_guideline_final_v2.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`
* `diagnostics_and_conservation_monitoring_guideline_final.md`
* `paper_v1_code_architecture_and_coding_guideline_final.md`

---

## 3. Phase 1 必须先冻结的程序口径

在写任何 `core/*.py` 之前，先把下面这些东西定死。

## 3.1 配置只认新 schema

执行口径：

* `paper_v1` **只接受新 schema**
* 不兼容旧项目 YAML 键名体系
* 不做“看起来像旧格式就自动猜”的兼容层

原因很简单：旧项目遗留命名一旦混进来，后面 layout、species map、solver profile 都会被污染。

---

## 3.2 `State` 必须保留 full composition 视图

执行口径：

* `State` 内必须能拿到 `Yl_full`
* 必须能拿到 `Yg_full`
* 若界面块需要，也要支持 `Ys_l_full`、`Ys_g_full`

不能只保存 reduced unknown。
reduced unknown 只是 **打包/求解布局**，不是整个程序的真实物理状态。

---

## 3.3 `Rd / a` 不进入 layout

执行口径：

* `a / Rd` 是 outer 变量
* `dot_a` 是 outer iterate 冻结输入
* `layout.py` 只描述 inner unknown block

因此 `UnknownLayout` 中**绝不出现半径变量**。

---

## 3.4 真相源必须围绕 PETSc Vec 设计

执行口径：

* 程序逻辑上的主 unknown 真相源是 global Vec
* `State` 是结构化视图，不是第二套“偷偷更权威”的全局数组
* `state_pack.py` 必须成为唯一的 pack/unpack 入口

这条不先定，后面很快就会长出一堆野生 `numpy` 全局状态副本。人类就爱这样。

---

## 3.5 remap 和 recovery 绝不能混成一个函数

执行口径：

* `remap.py` 只搬运守恒量
* `state_recovery.py` 只做恢复
* `old_state_on_current_geometry` 由二者串联构造

不能写成一个“看上去很方便”的混合函数，否则 diagnostics 和 failure tracing 会立刻变得含糊。

---

## 3.6 `core/` 阶段先不碰具体 PETSc 对象创建

执行口径：

* Phase 1 只建立 **PETSc-ready 的数据结构**
* 不在 `core/` 阶段创建 DM / SNES / Mat / KSP
* 但接口必须从第一天兼容 future PETSc usage

也就是说：

* 可以定义 pack/unpack 的逻辑语义
* 可以用 numpy 做局部测试
* 但函数签名要按未来 Vec/DM 语义设计

---

# 4. Phase 1 的开发顺序

正式顺序如下：

1. `core/types.py`
2. `core/config_schema.py`
3. `core/config_loader.py`
4. `core/preprocess.py`
5. `core/layout.py`
6. `core/grid.py`
7. `core/state_pack.py`
8. `core/remap.py`
9. `core/state_recovery.py`

可并行后补：

10. `core/logging_utils.py`

这个顺序不要反过来。特别是：

* `layout.py` 不能早于 `preprocess.py`
* `state_pack.py` 不能早于 `layout.py`
* `remap.py` 不能早于 `grid.py`
* `state_recovery.py` 不能早于 `liquid properties` 的接口约定
  但可以先写外壳和恢复流程，具体物性反解入口先留 stub

---

# 5. 各模块逐个实施方案

---

## 5.1 `core/types.py`

## 5.1.1 模块目标

定义全项目最核心的数据对象，统一程序内部的数据表达。

## 5.1.2 直接依赖文件

* `paper_v1_config_schema_guideline_final.md`
* `unknowns_strategy_guideline_final.md`
* `outer_inner_iteration_coupling_guideline_final.md`
* `timestep_and_failure_policy_guideline_final_v2.md`

## 5.1.3 建议定义的核心类型

### 配置与运行层

* `RunConfig`
* `CasePaths`
* `SpeciesConfig`
* `TimeStepperConfig`
* `OuterStepperConfig`
* `InnerSolverConfig`
* `DiagnosticsConfig`

### 网格与几何层

* `Mesh1D`
* `GeometryState`
* `RegionSlices`
* `ControlSurfaceMetrics`

### 状态层

* `State`
* `InterfaceState`
* `BulkStateLiquid`
* `BulkStateGas`
* `ConservativeContents`
* `OldStateOnCurrentGeometry`

### 布局与上下文层

* `UnknownLayout`
* `SpeciesMaps`
* `StepContext`
* `OuterIterState`

## 5.1.4 建议字段口径

### `GeometryState`

至少包含：

* `t`
* `a`
* `dot_a`
* `r_end`
* `outer_iter`
* `dt`

### `Mesh1D`

至少包含：

* `r_nodes`
* `r_faces`
* `r_centers`
* `volumes`
* `areas`
* `dr`
* `region_slices`
* `n_liq`
* `n_gas_near`
* `n_gas_far`

### `State`

至少包含：

* `Tl`
* `Tg`
* `Ts`
* `Yl_full`
* `Yg_full`
* `Ys_l_full`（可选但结构保留）
* `Ys_g_full`
* `mpp`
* `rho_l`
* `rho_g`
* `hl`
* `hg`
* `Xg_full`（气相建议显式保留）

### `ConservativeContents`

至少包含：

* `mass_l`
* `species_mass_l`
* `enthalpy_l`
* `mass_g`
* `species_mass_g`
* `enthalpy_g`

### `OldStateOnCurrentGeometry`

建议包含两部分：

* `contents`：守恒量内容
* `state`：recovery 后状态

这样后面 diagnostics 才能同时做守恒检查和状态检查。

## 5.1.5 硬约束

* 类型命名在此模块完成后尽量不再改
* `State` 中保留 full state
* `UnknownLayout` 是正式对象，不是散装 dict
* 不要把 PETSc 对象本体塞进 `State`

## 5.1.6 最小验收标准

1. 能实例化一套最小 `RunConfig`
2. 能实例化 `GeometryState + Mesh1D + State`
3. `State` 能同时承载单组分液滴和多组分液滴 shape
4. 类型层不再依赖任何 physics 计算

---

## 5.2 `core/config_schema.py`

## 5.2.1 模块目标

定义新项目唯一合法配置结构，负责“允许什么，不允许什么”。

## 5.2.2 直接依赖文件

* `paper_v1_config_schema_guideline_final.md`
* `liquid_properties_guideline_final.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`
* `timestep_and_failure_policy_guideline_final_v2.md`
* `diagnostics_and_conservation_monitoring_guideline_final.md`

## 5.2.3 建议功能

* 定义 required sections
* 定义 section 内 required keys
* 定义可选键默认规则
* 定义 enum 取值范围
* 定义 numeric bound rules
* 定义 cross-field validation hooks

## 5.2.4 建议核心函数

* `build_schema()`
* `validate_required_sections(raw_cfg)`
* `validate_unknown_keys(raw_cfg)`
* `validate_value_ranges(raw_cfg)`
* `validate_cross_field_rules(raw_cfg)`
* `schema_defaults()`

## 5.2.5 建议 schema 顶层块

按指导文件建议，至少包括：

* `case`
* `geometry`
* `grid`
* `unknowns`
* `liquid_db`
* `gas_mech`
* `interface_equilibrium`
* `time_stepper`
* `outer_stepper`
* `solver_inner_petsc`
* `recovery`
* `diagnostics`
* `output`
* `validation`

## 5.2.6 硬约束

* 不允许旧项目 schema 混入
* 不允许未知字段静默通过
* 影响物理主线的参数必须显式要求
* `t0` 这类初始化关键量不能偷偷默认

## 5.2.7 最小验收标准

1. 非法顶层块会报错
2. 缺关键字段会报错
3. 非法枚举值会报错
4. schema 层就能拦下明显不合法配置

---

## 5.3 `core/config_loader.py`

## 5.3.1 模块目标

负责读取 YAML 并调用 schema 完成严格校验。

## 5.3.2 直接依赖文件

* `paper_v1_config_schema_guideline_final.md`

## 5.3.3 建议核心函数

* `load_yaml(path)`
* `load_raw_config(path)`
* `validate_config(raw_cfg, schema)`
* `load_and_validate(path)`

## 5.3.4 输出

建议返回：

* `raw_cfg`
* `validation_report`

或者直接返回：

* `validated_raw_cfg`

## 5.3.5 硬约束

* 不做 normalize
* 不做 derived config
* 不做物理推断
* 只做读取和校验

## 5.3.6 最小验收标准

1. 能读 YAML
2. 能给出明确错误信息
3. 错误定位到 section/key 级别
4. 成功输出与输入结构一致的已验证配置

---

## 5.4 `core/preprocess.py`

## 5.4.1 模块目标

把“已验证原始配置”变成“程序真正使用的标准化配置”。

## 5.4.2 直接依赖文件

* `paper_v1_config_schema_guideline_final.md`
* `unknowns_strategy_guideline_final.md`
* `paper_v1 Initialization and First-Step Guideline.md`
* `liquid_properties_guideline_final.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`
* `timestep_and_failure_policy_guideline_final_v2.md`

## 5.4.3 本模块必须完成的事

1. species 列表标准化
2. closure species 明确化
3. active/reduced/full species map 生成
4. 单组分液滴 vs 多组分液滴模式识别
5. baseline unknown profile 选择
6. solver profile 归一化
7. dt / fail policy / guard policy 标准化
8. 初始化参数标准化
9. 路径与输出目录派生

## 5.4.4 建议核心函数

* `normalize_config(raw_cfg) -> RunConfig`
* `derive_species_maps(raw_cfg) -> SpeciesMaps`
* `derive_unknown_strategy(raw_cfg)`
* `derive_solver_profile(raw_cfg)`
* `derive_paths(raw_cfg)`
* `check_preprocess_guardrails(run_cfg)`

## 5.4.5 特别关键的输出

### `SpeciesMaps`

至少要给出：

* `liq_full_names`
* `liq_active_names`
* `liq_closure_name`
* `gas_full_names`
* `gas_active_names`
* `gas_closure_name`
* `full_to_reduced`
* `reduced_to_full`

### unknown 策略

至少要明确：

* 是否采用 `U_A`
* 是否采用 `U_B`
* interface block 是否含 `Ys_l_red`

## 5.4.6 硬约束

* 不能在别的模块重复推 species map
* 不能由 `layout.py` 再私自猜 closure species
* 不能把 solver profile 硬编码在 `petsc_snes.py`

## 5.4.7 最小验收标准

1. 单组分液滴 case 能得到合法 `RunConfig`
2. 多组分液滴 case 能得到合法 `RunConfig`
3. `SpeciesMaps` 完整且自洽
4. unknown strategy 已明确，不再依赖后续模块再猜

---

## 5.5 `core/layout.py`

## 5.5.1 模块目标

定义 inner nonlinear system 的唯一 unknown 布局。

## 5.5.2 直接依赖文件

* `unknowns_strategy_guideline_final.md`
* `interface_block_unknowns_and_residuals_table_final.md`
* `paper_v1_config_schema_guideline_final.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`

## 5.5.3 本模块必须解决的问题

1. 液相 bulk unknown 顺序
2. 界面 block unknown 顺序
3. 气相 bulk unknown 顺序
4. 单组分 vs 多组分液滴扩展规则
5. full/reduced composition 索引映射
6. 全局 offsets 与 block slices

## 5.5.4 建议核心对象

`UnknownLayout` 至少包含：

* `n_liq_cells`
* `n_gas_cells`
* `n_if_unknowns`
* `liq_block_slice`
* `if_block_slice`
* `gas_block_slice`
* `field_slices`
* `field_names`
* `reduced/full species index maps`

## 5.5.5 建议核心函数

* `build_layout(run_cfg, mesh, species_maps) -> UnknownLayout`
* `field_slice(layout, field_name)`
* `interface_field_slice(layout, field_name)`
* `reduced_to_full_massfractions(...)`
* `full_to_reduced_massfractions(...)`

## 5.5.6 推荐 unknown 顺序口径

### 液相 bulk

建议顺序：

* `Tl`
* `Yl_red...`（若单组分首版则为空或退化）

### 界面 block

建议顺序：

* `Ts`
* `Ys_g_red...`
* `mpp`
* `Ys_l_red...`（若当前策略要求）

### 气相 bulk

建议顺序：

* `Tg`
* `Yg_red...`

真正顺序可按后续 assembly 便利性微调，但必须在此模块一次性冻结。

## 5.5.7 硬约束

* `a / Rd / dot_a` 不进入 layout
* `mpp` 只有一个
* layout 必须支持多组分液相
* 不能让 assembly 自己计算 magic index

## 5.5.8 最小验收标准

1. 单组分液滴 layout 正确生成
2. 多组分液滴 layout 正确扩展
3. 全局未知量总长度可明确计算
4. 所有 field slice 都可直接查询

---

## 5.6 `core/grid.py`

## 5.6.1 模块目标

建立和更新当前几何下的 1D 球对称三段网格，并提供几何量。

## 5.6.2 直接依赖文件

* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `outer_inner_iteration_coupling_guideline_final.md`
* `paper_v1 Initialization and First-Step Guideline.md`
* `velocity_recovery_guideline_final.md`

## 5.6.3 本模块必须完成的事

1. 初始三段网格构造
2. outer corrector 后网格重建
3. faces / centers / volumes / areas 计算
4. 液相、近界面气相、远场气相区域切分
5. 为 `v_c` 计算提供几何支持
6. 为 remap overlap 计算提供几何边界数据

## 5.6.4 建议核心函数

* `build_initial_grid(run_cfg, geometry_state) -> Mesh1D`
* `rebuild_grid(run_cfg, geometry_state) -> Mesh1D`
* `compute_cell_volumes(r_faces)`
* `compute_face_areas(r_faces)`
* `build_region_slices(mesh)`
* `compute_control_surface_metrics(mesh, geometry_state)`

## 5.6.5 关键程序口径

* 球对称体积和面积公式要集中在这里
* 三段网格的接口层面要清楚，但不要把 solver 逻辑塞进来
* `grid.py` 给几何量，不负责 predictor-corrector

## 5.6.6 硬约束

* 当前几何量与旧几何量不可混用
* 初始网格直接由 `a0` 构造
* rebuild 后 mesh 必须能直接交给 remap 和 layout 使用

## 5.6.7 最小验收标准

1. 初始几何能生成三段网格
2. 改变 `a` 后能重建新网格
3. `r_faces / r_centers / volumes / areas` 自洽
4. 可明确标识液相段与气相段

---

## 5.7 `core/state_pack.py`

## 5.7.1 模块目标

建立 `State` 与 global unknown vector 之间的唯一映射。

## 5.7.2 直接依赖文件

* `unknowns_strategy_guideline_final.md`
* `interface_block_unknowns_and_residuals_table_final.md`
* `petsc_solver_and_parallel_architecture_guideline_final.md`

## 5.7.3 本模块必须完成的事

1. `State -> vector`
2. `vector -> State`
3. phase/interface 视图切片
4. full state 与 reduced unknown 互转
5. 单组分/多组分统一 pack 逻辑

## 5.7.4 建议核心函数

* `pack_state_to_array(state, layout) -> np.ndarray`
* `unpack_array_to_state(vec, layout, species_maps) -> State`
* `extract_liq_view(vec, layout)`
* `extract_if_view(vec, layout)`
* `extract_gas_view(vec, layout)`
* `apply_trial_vector(state_base, vec_trial, layout) -> State`

后面接 PETSc 时，再在此模块补：

* `pack_state_to_vec(...)`
* `unpack_vec_to_state(...)`

## 5.7.5 硬约束

* `state_pack.py` 是唯一 pack/unpack 入口
* `physics/assembly/solvers` 不自己手搓索引
* full/reduced 转换规则必须统一依赖 `SpeciesMaps`

## 5.7.6 最小验收标准

1. `pack -> unpack` 往返不丢字段
2. 单组分和多组分 case 都能工作
3. interface block 的 `mpp / Ts / Ys_*` 能正确映射
4. 未来切换 PETSc Vec 时不需要改 physics 接口

---

## 5.8 `core/remap.py`

## 5.8.1 模块目标

实现 conservative remap，并构建 `old_state_on_current_geometry`。

## 5.8.2 直接依赖文件

* `remap_and_conservative_projection_guideline_final_v2.md`
* `outer_inner_iteration_coupling_guideline_final.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `velocity_recovery_guideline_final.md`
* `diagnostics_and_conservation_monitoring_guideline_final.md`

## 5.8.3 本模块必须完成的事

1. 液相分相 remap
2. 气相分相 remap
3. overlap-based conservative contents transfer
4. newly exposed subvolume completion
5. 构造 `ConservativeContents`
6. 组织调用 recovery，得到 `OldStateOnCurrentGeometry`

## 5.8.4 建议核心函数

* `compute_overlap_matrix_1d_spherical(old_faces, new_faces, phase_region)`
* `remap_phase_contents(old_contents, overlap) -> new_contents`
* `complete_newly_exposed_subvolume(...)`
* `build_old_contents_on_current_geometry(old_state, old_mesh, new_mesh, geometry_k)`
* `build_old_state_on_current_geometry(old_state, old_mesh, new_mesh, ...)`

## 5.8.5 硬约束

* remap 对象是守恒量 content，不是 `T/Y` 值
* 液相、气相分别 remap
* 不允许跨相 remap
* newly exposed volume 处理必须显式，不可偷偷归零

## 5.8.6 特别需要预留的 diagnostics

本模块输出应带：

* remap 前后总质量差
* 各相各组分质量差
* 焓量差
* newly exposed volume 统计

否则后面一旦守恒炸掉，就只能靠猜。猜这种活动通常只在玄学有效。

## 5.8.7 最小验收标准

1. 几何不变时 remap 退化为恒等
2. 简单几何变化时质量/组分/焓守恒误差可控
3. `old_state_on_current_geometry` 可完整构造
4. remap 输出能直接进入 state recovery

---

## 5.9 `core/state_recovery.py`

## 5.9.1 模块目标

从守恒量内容恢复 `rho / Y / h / T`，生成可供 inner 时间项使用的旧状态。

## 5.9.2 直接依赖文件

* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `liquid_properties_guideline_final.md`
* `unknowns_strategy_guideline_final.md`
* `diagnostics_and_conservation_monitoring_guideline_final.md`

## 5.9.3 本模块必须完成的事

1. 由质量 content 恢复 `rho`
2. 由 species mass content 恢复 `Y_full`
3. 由 enthalpy content 恢复 `h`
4. 液相焓反解 `h -> T`
5. 气相焓反解 `h -> T`
6. 失败判定与局部修正接口

## 5.9.4 建议核心函数

* `recover_density(contents, volumes)`
* `recover_full_massfractions(contents, rho, species_maps)`
* `recover_specific_enthalpy(contents, rho)`
* `invert_liquid_h_to_T(h, Yl_full, liquid_model)`
* `invert_gas_h_to_T(h, Yg_full, gas_model)`
* `recover_phase_state_from_contents(...)`
* `recover_state_from_contents(...)`

## 5.9.5 实施建议

这一模块可以分两步写：

### 第一步

先把恢复流程和 shape 逻辑写好，液相/气相焓反解先留接口。

### 第二步

等 `properties/liquid.py`、`properties/gas.py` 稳定后，把真实反解接上。

这样做比把一堆焓反解细节硬塞进 Phase 1 更稳。

## 5.9.6 硬约束

* 不允许直接用 primitive interpolation 替代 recovery
* full species 恢复规则必须唯一
* 失败信息必须结构化，便于 diagnostics

## 5.9.7 最小验收标准

1. 能从 contents 恢复 `rho`、`Y_full`、`h`
2. 单组分液滴下流程可通
3. 多组分气相下 full composition 恢复可通
4. 焓反解接口已固定，不再影响上层模块接口

---

## 5.10 `core/logging_utils.py`（可后补）

## 5.10.1 模块目标

为后续 step/outer/inner/remap diagnostics 提供统一日志接口。

## 5.10.2 说明

它不是 Phase 1 主阻塞项，但建议尽早补上，不然后面错误信息会散落在各模块里，像屋里到处都是电线头。

---

# 6. Phase 1 内部的先后依赖关系

推荐按下面的微顺序实施：

## 第一组：先钉数据骨架

1. `types.py`
2. `config_schema.py`
3. `config_loader.py`

## 第二组：再钉运行标准化

4. `preprocess.py`

## 第三组：再钉求解布局与几何

5. `layout.py`
6. `grid.py`

## 第四组：再钉状态映射

7. `state_pack.py`

## 第五组：最后钉 remap / recovery

8. `remap.py`
9. `state_recovery.py`

---

# 7. Phase 1 结束时必须通过的最小测试

## 7.1 配置测试

* 新 schema case 可读
* 缺字段报错
* 多余字段报错
* 非法 enum 报错

## 7.2 布局测试

* 单组分液滴 layout 正确
* 多组分液滴 layout 正确
* interface block 顺序正确
* `mpp` 只出现一次

## 7.3 网格测试

* 初始三段网格生成正确
* 修改 `a` 后可 rebuild
* 体积与面积为正
* 区域 slices 正确

## 7.4 state pack 测试

* `pack -> unpack -> pack` 一致
* 单组分和多组分都通过
* interface 状态不丢失

## 7.5 remap / recovery 测试

* 几何不变时 remap 近似恒等
* 简单收缩/膨胀时 contents 合理
* `old_state_on_current_geometry` 生成成功
* recovery 后 state shape 合法

---

# 8. Phase 1 的阶段出口条件

只有满足下面这些条件，才允许进入 Phase 2：

1. `RunConfig` 接口冻结
2. `SpeciesMaps` 接口冻结
3. `UnknownLayout` 接口冻结
4. `Mesh1D / GeometryState / State` 接口冻结
5. `pack/unpack` 接口冻结
6. `old_state_on_current_geometry` 构造主线冻结
7. 后续 `properties / physics / assembly` 不再需要反向修改 `core` 接口

---

# 9. 进入具体实现时的建议动作

按实施效率，建议你接下来这样推进：

## 第一步

先写：

* `core/types.py`
* `core/config_schema.py`

因为这两个文件会决定其余 `core` 模块的签名。

## 第二步

再写：

* `core/config_loader.py`
* `core/preprocess.py`

因为 `preprocess.py` 会把 unknown strategy、species map、solver profile 一次性钉死。

## 第三步

然后写：

* `core/layout.py`
* `core/grid.py`
* `core/state_pack.py`

因为这三者决定后续 assembly 和 solver 的主接口。

## 第四步

最后写：

* `core/remap.py`
* `core/state_recovery.py`

因为它们依赖前面的类型、网格、layout 和 state 骨架都先稳定。

---

# 10. 当前我建议先冻结的 3 个具体接口决定

这三个问题现在就该拍板，不要拖。

## 10.1 类型命名

我建议正式采用：

* `RunConfig`
* `Mesh1D`
* `GeometryState`
* `State`
* `ConservativeContents`
* `OldStateOnCurrentGeometry`
* `UnknownLayout`
* `SpeciesMaps`
* `StepContext`
* `OuterIterState`

不要再在 `CaseConfig / Grid1D / Grid / Mesh` 之间摇摆。

## 10.2 `State` 中是否显式保存 full composition

结论：**必须保存**。

## 10.3 `state_recovery.py` 是否先写接口再接物性

结论：**是**。
Phase 1 先固定恢复流程与签名，真实反解在 Phase 2 接入。

---

# 11. 这一阶段不做什么

Phase 1 明确**不做**：

* 不写具体 liquid/gas property model
* 不写 flux kernel
* 不写 interface residual
* 不写 Jacobian
* 不写 SNES
* 不写 DM / Vec / Mat 创建
* 不写 timestepper

别在这一步偷跑去写 residual。那样通常不是“效率高”，而是“准备把基础接口写废”。

---

下一步最合理的动作是直接开始 **`core/types.py` 的接口定稿**。
