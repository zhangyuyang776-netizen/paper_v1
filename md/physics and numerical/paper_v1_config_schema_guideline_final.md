# paper_v1_config_schema_guideline_final

## 1. 文件定位

本文档用于为 `paper_v1` 正式规定以下内容：

1. 配置文件（YAML/JSON）在项目中的角色与边界；
2. 配置参数的分类方法；
3. 配置参数从“用户输入”到“运行代码模块”的正式传递链条；
4. 后续新增参数时必须修改的固定位置与检查清单；
5. 当前阶段尚未最终定稿的内容，以及这些未定稿内容对配置系统的影响；
6. `paper_v1` 首版最小可运行配置草案。

本文档的目标不是一次性穷举未来所有参数，而是：

> **建立一套参数治理方法，使 `paper_v1` 的配置系统能够随着代码增长而不失控。**

---

## 2. 总原则

## 2.1 配置文件不是运行时状态仓库

正式规定：

配置文件只用于描述**输入决策**，不用于存储运行时派生状态。

### 配置中应存在的内容
- 案例定义
- 物理模型选择
- 网格策略
- 时间步控制参数
- outer / inner 求解策略
- recovery 参数
- diagnostics / output 开关
- validation 开关

### 配置中不应存在的内容
- 当前时间步的 `a^(k)`、`dot_a^(k)`
- 当前 outer 轮的 `eps_dot_a`
- 当前 accepted/rejected 状态
- `old_state_on_current_geometry` / `U_init^(k)` / `U_transfer^(k->k+1)` ???????????
- Jacobian 当前尺寸
- 当前 rank 的局部范围
- 当前 residual 的最大分量

这些属于 runtime context，不属于 config。

---

## 2.2 配置系统不追求一次性穷举所有参数

正式规定：

`paper_v1_config_schema_guideline_final.md` 不以“一次性固定未来所有参数名”为目标。  
它的正式目标是：

1. 固定参数组织方法；
2. 固定参数传递链条；
3. 固定新增参数的修改规则；
4. 提供首版最小可运行 schema。

原因很简单：

- 代码在编写过程中，参数一定会继续增长；
- 若试图在项目初期就穷举未来所有配置项，文件极易迅速失效；
- 真正需要定下来的，是“如何增长而不混乱”。

---

## 2.3 关键参数禁止模块内隐式默认值

正式规定：

以下类型的参数，**不允许**在模块内部藏默认值：

- 影响物理含义的参数
- 影响数值方法的参数
- 影响 timestep / solver / recovery / diagnostics 行为的关键参数

例如：

- `dt_start`
- `dt_min`
- `dt_max`
- `inner_max_iter`
- `outer_max_iter`
- `closure_species`
- `output_every_n_steps`
- `T_min_l`
- `T_max_l`

这些参数必须由配置文件显式给出，或在**单一集中位置**统一归一化，不允许散落在模块里偷偷兜底。

### 当前阶段正式立场
`paper_v1` 首版优先采用更严格路线：

> **关键参数尽量全部显式配置，不依赖代码内部默认值。**

---

## 2.4 只接受新 schema，不兼容旧项目 YAML

正式规定：

`paper_v1` 配置系统**只接受新 schema**，不提供旧项目 YAML 的兼容层，也不做一次性自动翻译。

因此，`core/config_loader.py` 与 schema validation 必须：

- 只接受本文件规定的顶层块
- 对未知顶层块直接报错
- 对旧项目风格顶层块直接报错

明确拒绝的旧风格示例包括：

- `physics`
- `species`
- `petsc`
- `io`
- `paths`

原因是：

- `paper_v1` 的指导文件已经把新 schema 作为唯一正式约束来源
- 若再保留旧 YAML 兼容层，会让 `paper_v1` 长期处于半新半旧状态
- 这会直接破坏 `config_loader.py -> normalize_config -> build_subconfigs` 的单一语义

---

## 3. 参数分类方法

正式将参数分为三类。

## 3.1 Required Explicit Parameters（必须显式配置）

这类参数必须由案例配置显式给出，缺失即报错。

### 特征
- 影响物理主线
- 影响求解策略
- 影响收敛/失败行为
- 影响输出频率或布局含义

### 典型例子
- `a0`
- _end`
- `dt_start`
- `dt_min`
- `dt_max`
- etry_max_per_step`
- `inner_max_iter`
- `outer_max_iter`
- `liquid_closure_species`
- `gas_closure_species`
- `mechanism_path`
- `species_database_path`

---

## 3.2 Optional Centralized Defaults（允许集中默认）

这类参数允许不在 YAML 中显式给出，但其默认值必须只存在于**一个集中位置**，且归一化后必须被完整记录。

### 特征
- 不改变主物理含义
- 主要影响日志详细程度、导出细节、调试粒度
- 即便有默认，也必须进入 normalized config

### 典型例子
- `diag.verbose_interface_panel`
- `diag.verbose_property_warnings`
- 某些输出文件命名模板
- 某些 debug 级别

### 规则
- 不允许在多个模块里各自写一套默认值
- 不允许“模块发现没传值就自己补一个”

---

## 3.3 Derived Parameters（派生参数）

这类参数不是用户输入，而是由 normalized config 或 layout/build 阶段派生得到。

### 特征
- 与输入配置强相关
- 用于减少运行时重复推导
- 不能反向写回原始 config

### 典型例子
- `n_species_liquid_full`
- `n_species_gas_full`
- educed_to_full_map`
- `active_interface_unknowns`
- egion3_is_fixed`
- `field_offsets`
- `layout_version`

---

## 4. 配置参数的正式传递链条

这是本文件最关键的内容之一。

本文件给出参数治理与传递链条的**抽象模板**。  
其模块级落地、目录边界和实际文件映射，统一由 `paper_v1_code_architecture_and_coding_guideline_final.md` 规定。

---

## 4.1 正式传递链条（抽象模板）

### Position 1：Raw Config File
用户编写的原始 YAML / JSON 文件。

职责：
- 提供案例输入
- 不包含运行时状态
- 不包含隐藏派生量

---

### Position 2：Schema Definition
配置结构定义层。

职责：
- 声明允许的顶层块
- 声明每个块的字段类型
- 声明哪些字段 required / optional / derived
- 声明枚举和数值范围

说明：
- 这一层是“项目允许哪些键”的真相源
- 后续新增参数，首先要改这里

---

### Position 3：Schema Validation
配置校验层。

职责：
- 检查必填键是否存在
- 检查字段类型是否正确
- 检查未知键是否出现
- 检查枚举值与范围是否合法
- 检查参数间约束是否满足

示例：
- `dt_min < dt_start < dt_max`
- `outer_max_iter >= 1`
- `closure_species ∈ species_full`

说明：
- 这一层是“配置是否合法”的真相源
- 后续新增参数，必须同步加校验规则

---

### Position 4：Config Normalization
配置归一化层。

职责：
- 解析路径
- 填充允许的集中默认值
- 标准化字段结构
- 生成派生参数
- 输出 normalized config

说明：
- 若首版不想引入太重的配置框架，这一层也必须显式存在
- 哪怕只是一个 `normalize_config(raw_cfg)` 函数，也必须单独存在
- 这是“配置输入”变成“模块可消费配置对象”的唯一正途

---

### Position 5：Sub-config Projection
子配置投影层。

职责：
把统一的 normalized config 拆成各模块只读视图，例如：

- `grid_cfg`
- `unknowns_cfg`
- `solver_cfg`
- ecovery_cfg`
- `diag_cfg`
- `output_cfg`

说明：
- 模块不允许直接拿整份 raw YAML 到处查键
- 模块只允许读取自己的 sub-config

---

### Position 6：Module Constructor / Initializer
运行模块初始化层。

职责：
- 接收对应 sub-config
- 构造模块对象
- 建立模块内只读参数视图
- 禁止在此层偷偷新增 schema 之外的隐藏参数

示例模块：
- grid builder
- state/layout builder
- outer stepper
- inner PETSc solver
- recovery module
- diagnostics/output module

---

### Position 7：Runtime Context
运行时上下文层。

职责：
- 存放运行时状态
- 存放动态统计量
- 存放 step / outer / inner 的最新结果
- 不允许被视作配置的一部分

典型内容：
- current state
- geometry
- `a^(k)`, `dot_a^(k)`
- solver stats
- diagnostics buffers

---

### Position 8：Metadata / Echo Output
配置回显与案例元信息层。

职责：
- 将最终生效的 normalized config 输出到文件
- 让后处理与复现实验知道“真正生效的配置是什么”

正式要求：
每次运行都必须输出至少一个：
- `case_config.normalized.yaml`
或
- `case_metadata.json`

---

## 4.2 与代码结构文件的映射关系

抽象链条在本文件中固定；实际落地映射由：

# `paper_v1_code_architecture_and_coding_guideline_final.md`

统一规定。

因此，Position 5 / 6 / 7 / 8 在实现中必须进一步映射到：

- 具体 Python 文件
- 具体 dataclass / config object
- 具体模块构造器
- 具体 runtime manager / step driver

正式要求：

> **任何配置参数的实现都必须同时满足：一方面符合本文件的治理链条，另一方面符合代码结构文件给出的实际模块落点。**

---

## 5. 新增参数时必须修改的位置

后续增加任意新参数时，必须按固定链条修改。  
正式规定如下：

---

## 5.1 必改位置清单

每新增一个参数，至少必须检查并修改以下位置：

### Position 1：案例配置文件示例
- 是否需要写入示例 YAML
- 是否属于 required explicit

### Position 2：Schema Definition
- 加入字段定义
- 指定所属块
- 指定类型
- 指定 required / optional / derived 分类

### Position 3：Schema Validation
- 增加类型检查
- 增加范围检查
- 增加关联约束检查（若需要）

### Position 4：Config Normalization
- 若该参数允许集中默认，则在这里填充默认
- 若该参数会生成派生量，则在这里生成
- 若该参数需标准化命名或结构，也在这里完成

### Position 5：Sub-config Projection
- 决定该参数进入哪个模块的 sub-config
- 不允许所有模块都从全局 config 自己去找

### Position 6：Module Constructor / Initializer
- 把该参数传给真正使用它的运行模块
- 禁止中途丢失
- 禁止构造器私自补默认覆盖配置

### Position 7：Runtime Metadata / Diagnostics（若适用）
- 若该参数会影响运行行为或失败排查，需要写入 metadata / diagnostics
- 例如 solver 参数、recovery 参数、timestep 参数通常应进入 case metadata 与 failure report

### Position 8：Tests
- 增加“缺失该参数时报错”的测试（若 required）
- 增加“非法值时报错”的测试
- 增加“归一化后值正确传递”的测试

### Position 9：Documentation
- 更新本 config guideline
- 若影响求解器/恢复/输出，还需更新对应专题指导文件

---

## 5.2 新增参数前必须回答的 5 个问题

每新增一个参数，必须先回答：

1. 它属于哪个模块拥有？
2. 它属于 required / optional / derived 哪一类？
3. 它的合法范围是什么？
4. 它会影响哪些日志、元信息、failure report、tests？
5. 它是否需要进入 normalized config 回显？

只有回答完这五个问题，参数才能进入 schema。

---

## 5.3 明确禁止的做法

正式禁止：

1. 先在某个模块构造器里加个 `foo=None`，以后再考虑是否放进 config
2. 让 raw YAML 在模块间到处传递
3. 模块读取整份 config 自己到处找键
4. 模块发现没传值就偷偷设默认
5. schema 不改，只在代码里硬加参数
6. 参数只改了 YAML 示例，没改 validation / normalization / tests

这些做法会迅速让配置系统失控，后面谁都不知道哪个值真正生效。

---

## 6. 顶层 schema 结构建议

`paper_v1` 首版正式采用如下顶层块结构：

```yaml
case:
geometry:
grid:
unknowns:
liquid_db:
gas_mech:
interface_equilibrium:
time_stepper:
outer_stepper:
solver_inner_petsc:
recovery:
diagnostics:
output:
validation:
````

这不是未来永远不能改，但它是当前阶段**唯一允许的正式顶层组织**。

---

## 7. 各顶层块的职责

## 7.1 `case`

职责：

* 案例身份
* 输出路径
* 人类可读描述

建议字段：

* `case_id`
* `description`
* `output_root`

---

## 7.2 `geometry`

职责：

* 液滴初始几何
* 计算域范围
* 压力与时间区间

建议字段：

* `a0`
* _end`
* `pressure`
* `t_start`
* `t_end`

---

## 7.3 `grid`

职责：

* 三段网格参数
* 段数与分辨率
* 第三段固定区参数

建议字段：

* `n_liquid`
* `n_gas_inner`
* `n_gas_outer`
* `inner_gas_extent_factor`
* `outer_grid_stretch_ratio`
* egion3_fixed`

---

## 7.4 `unknowns`

职责：

* full / reduced species
* closure species
* 布局基础信息

建议字段：

* `liquid_species_full`
* `gas_species_full`
* `liquid_closure_species`
* `gas_closure_species`
* `liquid_reduced_species`
* `gas_reduced_species`

---

## 7.5 `liquid_db`

职责：

* 液相物性数据库来源
* 参考温度
* 活度系数模型
* 潜热模型策略

建议字段：

* `species_database_path`
* eference_temperature`
* `activity_model`
* `latent_heat_model_policy`

---

## 7.6 `gas_mech`

职责：

* Cantera 机理与 phase

建议字段：

* `mechanism_path`
* `phase_name`

---

## 7.7 `interface_equilibrium`

职责：

* 界面平衡模型
* 可凝/非凝物种分类
* UNIFAC 是否启用

建议字段：

* `equilibrium_model`
* `use_unifac`
* `condensable_species`
* `noncondensable_species`

---

## 7.8 `time_stepper`

职责：

* 时间步控制
* rollback / growth 参数

建议字段（必须显式配置）：

* `dt_start`
* `dt_min`
* `dt_max`
* etry_max_per_step`
* `q_success_for_growth`

---

## 7.9 `outer_stepper`

职责：

* outer predictor-corrector
* 外层收敛阈值
* 欠松弛

建议字段（必须显式配置）：

* `outer_max_iter`
* `outer_relaxation_initial`
* `outer_convergence_eps`
* `predictor_mode`
* `corrector_mode`

---

## 7.10 `solver_inner_petsc`

职责：

* PETSc inner nonlinear solve 配置

建议字段（首版建议显式配置）：

* `inner_max_iter`
* `snes_type`
* `line_search_type`
* `ksp_type`
* `pc_type`
* `fieldsplit_type`
* `schur_fact_type`
* `schur_precondition`
* `asm_overlap`
* `use_eisenstat_walker`

---

## 7.11 ecovery`

职责：

* 状态恢复与焓反解参数

建议字段（必须显式配置）：

* ho_min`
* `m_min`
* `species_recovery_eps_abs`
* `Y_sum_tol`
* `Y_hard_tol`
* `h_abs_tol`
* `h_rel_tol`
* `h_check_tol`
* `T_step_tol`
* `T_min_l`
* `T_max_l`
* `T_min_g`
* `T_max_g`
* `liquid_h_inv_max_iter`
* `gas_h_inv_max_iter`
* `use_cantera_hpy_first`

---

## 7.12 `diagnostics`

职责：

* 运行日志
* 结构化诊断
* 守恒监控
* 失败报告

建议字段：

* `enable_run_log`
* `write_step_diag`
* `write_interface_diag`
* `enable_conservation_monitoring`
* `write_failure_report`
* `write_rank_failure_meta`

---

## 7.13 `output`

职责：

* 标量输出
* 空间快照输出
* 失败快照输出

建议字段：

* `write_scalars`
* `scalars_every_n_steps`
* `scalar_field_list`
* `write_spatial_snapshots`
* `output_every_n_steps`
* `spatial_field_groups`
* `write_failure_snapshot`

---

## 7.14 `validation`

职责：

* 配置严格检查
* 运行时断言
* post-step 自检

建议字段：

* `strict_unknown_key_check`
* `strict_required_key_check`
* `enable_runtime_assertions`
* `enable_post_step_budget_check`

---

## 8. 首版最小可运行 schema 草案

下面给出 `paper_v1` 首版的**最小可运行配置草案**。
这不是未来所有参数的终版列表，而是当前阶段足以支撑首版代码开工的一版建议。

```yaml id="paper_v1_initial_yaml_v1"
case:
  case_id: mono_droplet_baseline
  description: single-component transport-only baseline
  output_root: out/paper_v1

geometry:
  a0: 5.0e-5
  r_end: 5.0e-3
  pressure: 101325.0
  t_start: 0.0
  t_end: 5.0e-3

grid:
  n_liquid: 20
  n_gas_inner: 60
  n_gas_outer: 60
  inner_gas_extent_factor: 5.0
  outer_grid_stretch_ratio: 1.08
  region3_fixed: true

unknowns:
  liquid_species_full: [NC12H26]
  gas_species_full: [NC12H26, O2, N2]
  liquid_closure_species: NC12H26
  gas_closure_species: N2
  liquid_reduced_species: []
  gas_reduced_species: [NC12H26, O2]

liquid_db:
  species_database_path: data/liquid_species.yaml
  reference_temperature: 298.15
  activity_model: ideal
  latent_heat_model_policy: sensible_difference

gas_mech:
  mechanism_path: mechanisms/n_dodecane_reduce.yaml
  phase_name: gas

interface_equilibrium:
  equilibrium_model: clausius_clapeyron_single_component
  use_unifac: false
  condensable_species: [NC12H26]
  noncondensable_species: [O2, N2]

time_stepper:
  dt_start: 1.0e-7
  dt_min: 1.0e-12
  dt_max: 1.0e-5
  retry_max_per_step: 8
  q_success_for_growth: 10

outer_stepper:
  outer_max_iter: 8
  outer_relaxation_initial: 1.0
  outer_convergence_eps: 1.0e-5
  predictor_mode: explicit_from_previous_dot_a
  corrector_mode: trapezoidal_anchored

solver_inner_petsc:
  inner_max_iter: 30
  snes_type: newtonls
  line_search_type: bt
  ksp_type: fgmres
  pc_type: fieldsplit
  fieldsplit_type: schur
  schur_fact_type: full
  schur_precondition: a11
  asm_overlap: 1
  use_eisenstat_walker: true

recovery:
  rho_min: 1.0e-12
  m_min: 1.0e-20
  species_recovery_eps_abs: 1.0e-14
  Y_sum_tol: 1.0e-10
  Y_hard_tol: 1.0e-6
  h_abs_tol: 1.0e-8
  h_rel_tol: 1.0e-10
  h_check_tol: 1.0e-8
  T_step_tol: 1.0e-8
  T_min_l: 250.0
  T_max_l: 900.0
  T_min_g: 200.0
  T_max_g: 4000.0
  liquid_h_inv_max_iter: 50
  gas_h_inv_max_iter: 50
  use_cantera_hpy_first: true

diagnostics:
  enable_run_log: true
  write_step_diag: true
  write_interface_diag: true
  enable_conservation_monitoring: true
  write_failure_report: true
  write_rank_failure_meta: true

output:
  write_scalars: true
  scalars_every_n_steps: 1
  scalar_field_list: [step, t, dt, Ts, a, mpp, dot_a_phys, Tg_min, Tg_max, Tl_min, Tl_max]
  write_spatial_snapshots: true
  output_every_n_steps: 10
  spatial_field_groups: [liquid, gas, interface]
  write_failure_snapshot: true

validation:
  strict_unknown_key_check: true
  strict_required_key_check: true
  enable_runtime_assertions: true
  enable_post_step_budget_check: true
```

---

## 9. 配置系统的实施要求

## 9.1 必须有显式 normalize 阶段

正式规定：

即使首版不引入复杂配置框架，也必须有显式的：

* `load_raw_config(...)`
* `validate_config(...)`
* `normalize_config(...)`
* `build_subconfigs(...)`

这样的处理阶段。

不允许 raw YAML 读进来后直接在各模块里到处查字段。

---

## 9.2 必须有 strict unknown key check

正式建议首版就启用：

* 未知键报错
* 拼错键名不允许静默忽略

原因很简单：
配置系统里最讨厌的不是缺参数，而是用户明明写了一个键，结果拼错了，代码却装作没看见。

## 9.4 必须拒绝 legacy 顶层块

正式规定：

- `core/config_loader.py` 不做 legacy schema auto-translation
- 不允许把旧项目 `physics/species/petsc/io/paths` 风格顶层块偷偷映射到新 schema
- 首次加载时即报错，要求用户改为新 schema

---

## 9.3 必须输出 normalized config

每次运行都必须输出至少一个：

* `case_config.normalized.yaml`
  或
* `case_metadata.json`

用来记录：

* 最终生效参数
* 派生参数
* 版本号
* schema version

这样实验才可复现，失败才可追溯。

---

## 10. 与代码结构文件的关系

`paper_v1_code_architecture_and_coding_guideline_final.md` 已给出代码级目录与模块边界。  
因此，本文件中的 Position 1–8 必须具体映射为：

* 实际文件路径
* 实际 dataclass / schema 对象
* 实际模块构造器
* 实际 runtime context 结构

届时，本文件不重写原则，只补充“模板到代码的具体映射”。

---

## 11. 正式禁止事项

以下做法在本文件下明确禁止：

1. 试图在项目初期一次性穷举未来所有参数并写死；
2. 模块内部私自添加 schema 之外的参数；
3. raw config 在模块间到处传递；
4. 模块自行读取整份 config 到处找键；
5. 关键物理/数值参数在模块内藏默认值；
6. 新增参数时只改 YAML，不改 schema / validation / normalization / tests；
7. 派生参数直接回写 raw config；
8. 不输出 normalized config；
9. 参数引入后不补文档、不补测试；
10. 因为代码结构尚未定稿，就完全不定义参数治理方法。
11. 在 `config_loader.py` 中偷偷兼容旧项目 YAML 顶层块。

---

## 12. 最终方法合同

### Config Schema Governance Contract

1. `paper_v1` configuration is governed by method and transmission rules, not by attempting to enumerate all future parameters up front.
2. Configuration contains input decisions, not runtime state.
3. Parameters are classified as:

   * required explicit
   * optional centralized default
   * derived
4. A formal transmission chain must exist:

   * raw config
   * schema definition
   * schema validation
   * config normalization
   * sub-config projection
   * module initialization
   * runtime context
   * metadata echo
5. Every new parameter must be added through the full chain, not by bypassing schema.
6. Key physical and numerical parameters must not hide as module-local defaults.
7. Normalized configuration must be written out for every run.
8. The exact module-level mapping of this chain is defined in `paper_v1_code_architecture_and_coding_guideline_final.md`; the governance method remains fixed here.
9. `paper_v1` accepts only the new schema; no legacy YAML compatibility layer is allowed in `core/config_loader.py`.

---

## 13. 最终结论

`paper_v1` 的配置系统正式采用如下主线：

* 不以“一次性固定所有参数”为目标；
* 以“参数治理方法 + 参数传递链条 + 新增参数修改清单”为核心；
* 当前阶段先固定 schema 方法，再给出首版最小可运行配置草案；
* 关键参数由配置文件显式给出，不允许模块内部藏默认值；
* 参数必须经过：

  * schema definition
  * validation
  * normalization
  * sub-config projection
  * module initialization
    之后才能进入运行模块；
* 参数治理主线由本文件固定；
* Position 1–8 的模块级映射由 `paper_v1_code_architecture_and_coding_guideline_final.md` 固定。

这就是 `paper_v1_config_schema_guideline_final.md` 的正式定稿版本。
