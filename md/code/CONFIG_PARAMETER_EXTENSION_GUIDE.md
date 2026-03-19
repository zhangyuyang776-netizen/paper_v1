# paper_v1 配置参数新增标准修改流程

## 适用范围

本文件基于当前 `paper_v1` 已完成的 `core` 前 5 个模块整理：

- `core/types.py`
- `core/config_schema.py`
- `core/config_loader.py`
- `core/preprocess.py`
- `core/layout.py`

并以模板配置文件 [config.yaml](/d:/a_yuyangz/2020XJTU/STCK/research/02droplet%20simulation/codex/paper_v1/cases/config.yaml) 为当前正式模板。

本文件的目的不是重复解释各模块细节，而是固定一套后续新增配置参数时必须遵守的修改顺序、影响范围和检查单。

## 当前主链结论

当前 `core` 1-5 模块已经形成一条清晰的正式主链：

```text
YAML
-> config_loader.load_and_validate_config()
-> raw_cfg
-> preprocess.normalize_config(...)
-> RunConfig
-> layout.build_layout(run_cfg, mesh)
-> UnknownLayout
```

这条链当前是成立的，且职责边界清楚：

- `config_schema.py` 只做 raw config 合法性检查。
- `config_loader.py` 只做 YAML 读取、重复键检查、空文档/非 mapping 拦截和 schema 调用。
- `preprocess.py` 负责标准化、派生 `SpeciesMaps`、冻结 `unknowns_profile`、构造 `RunConfig`。
- `layout.py` 负责基于 `RunConfig + Mesh1D` 冻结 inner unknown 布局。

## 模板配置文件检查结果

当前模板 [config.yaml](/d:/a_yuyangz/2020XJTU/STCK/research/02droplet%20simulation/codex/paper_v1/cases/config.yaml) 与现有 `schema/loader` 是一致的：

- `load_and_validate_config("paper_v1/cases/config.yaml")` 可以通过。
- 顶层 section、字段命名、类型和数值范围与当前 `core` 1-5 主链匹配。

但要注意一件事：

- 该模板目前只是 `loader/schema-valid`。
- 它还不是 `preprocess-ready`，因为相对路径会按配置文件所在目录 `paper_v1/cases/` 解析。
- 当前模板引用的
  - `data/liquid_db.yaml`
  - `mech/gas.yaml`
  在 `paper_v1/cases/` 下并不存在。

因此，如果要让模板直接进入 `preprocess.normalize_config(...)`，必须先在 case 目录下放置对应外部文件，或改成正确的相对/绝对路径。

## 配置参数的正式传递链

每个配置参数都应沿着下面的正式链条传递：

```text
YAML 字段
-> config_schema.py 声明与校验
-> config_loader.py 读入 raw_cfg
-> preprocess.py 标准化并写入 RunConfig
-> 下游模块仅读取 RunConfig / SpeciesMaps / UnknownLayout
```

硬规则：

- 下游模块不得绕过 `RunConfig` 直接读 raw YAML 或 raw dict。
- `preprocess.py` 不得做 silent correction。
- `layout.py` 不得重新派生 species 规则、closure species 或 unknown profile。

## 新增配置参数时的标准原则

### 原则 1

先判断参数属于哪个 section，再决定修改哪些文件。

不要先改代码，再回头找它应该放在哪个块里。

### 原则 2

区分两层责任：

- `config_schema.py` 负责“这个字段在 raw config 中是否合法出现”。
- `preprocess.py` 负责“程序内部如何解释并标准化这个字段”。

### 原则 3

不要绕过 `RunConfig`。

如果一个参数会被运行期正式使用，它最终必须进入某个 dataclass，并由 `RunConfig` 向下游传播。

### 原则 4

如果一个参数的约束已经在 schema 层固定，且它最终进入某个 dataclass，那么类型层约束也应同步镜像。

典型例子：

- `TimeStepperConfig.growth_factor >= 1`
- `0 < shrink_factor <= 1`
- `OutputConfig.snapshot_format == "npz"`

### 原则 5

`config_loader.py` 通常不需要因为“新增一个普通配置参数”而改动。

只有当 raw 文件读取语义变化时，才改 `config_loader.py`，例如：

- 新增 include 机制
- 新增环境变量替换
- 新增另一种输入文件格式
- 修改重复键策略

普通业务字段新增，一般只改：

- `config_schema.py`
- YAML 模板
- `types.py`
- `preprocess.py`
- 受影响的下游模块

## 标准修改流程

### 第一步：确定 section

先判断新参数属于哪个顶层 section。

常见分类：

- `case`
- `paths`
- `mesh`
- `species`
- `initialization`
- `time_stepper`
- `outer_stepper`
- `solver_inner_petsc`
- `recovery`
- `diagnostics`
- `output`
- `validation`

不要把一个参数塞进“差不多能放”的 section。

### 第二步：修改 `config_schema.py`

新增参数的第一个正式落点是 schema。

你需要做：

1. 在对应 `SchemaSection` 中新增 `SchemaField`
2. 视需要补充 `validate_cross_field_rules()`
3. 若参数是复杂 dict/list 结构，补对应 helper 校验

目标：

- raw config 中该字段可被正式声明
- 缺失、类型错误、取值错误在 schema 层就能被拦住

### 第三步：同步更新 YAML 模板与测试工厂

必须同步修改：

- 正式模板 [config.yaml](/d:/a_yuyangz/2020XJTU/STCK/research/02droplet%20simulation/codex/paper_v1/cases/config.yaml)
- `tests` 中的 `make_minimal_valid_raw_config()`

否则代码已经进入新世界，模板和测试还停在旧世界，后续会越来越乱。

### 第四步：修改 `types.py`

判断这个参数是否会进入程序内部标准化对象。

若会进入，则必须：

1. 修改对应 dataclass
2. 在 `__post_init__()` 中补必要约束
3. 必要时补 alias / property

不要让 `preprocess.py` 里出现散装字段到处飞。

### 第五步：修改 `preprocess.py`

这是参数的正式“程序语义落点”。

需要决定：

- 是否需要默认值
- 是否需要路径解析
- 是否需要 cross-source consistency check
- 是否需要从 raw dict 派生成标准化格式
- 是否需要写入 `SpeciesMaps`
- 是否需要影响 `unknowns_profile`

所有“程序内部如何解释这个参数”的逻辑，都应集中在 `preprocess.py`，不要散落到下游模块。

### 第六步：判断是否影响下游模块

按参数类型决定是否还要改：

- `grid.py`
- `layout.py`
- `state_pack.py`
- `remap.py`
- `state_recovery.py`
- `solvers/*`
- `io/*`

典型判断：

- 影响 mesh 生成：改 `grid.py`
- 影响 unknown 结构：改 `layout.py` 和后续 `state_pack.py`
- 影响 solver 行为：改 `solvers/*`
- 影响输出/诊断：改 `io/*` 和 `diagnostics` 使用点

### 第七步：补测试

至少补三类测试：

1. schema 测试
2. preprocess 测试
3. 下游模块测试

不要只补 schema，不补标准化和实际使用点。

## 不同类型参数的修改范围速查

### A. 纯元信息参数

例如：

- `case.author`
- `case.tag`

通常修改：

- `config_schema.py`
- YAML 模板
- `types.py`
- `preprocess.py`

通常不影响：

- `layout.py`
- `grid.py`
- `state_pack.py`

### B. 路径类参数

例如：

- `paths.reference_data_path`

通常修改：

- `config_schema.py`
- YAML 模板
- `CasePaths`
- `preprocess._resolve_external_paths()`
- 对应 preprocess 测试

### C. mesh 生成参数

例如：

- `mesh.gas_near_width_factor`

通常修改：

- `config_schema.py`
- YAML 模板
- `MeshConfig`
- `preprocess.py`
- `grid.py`

备注：

- `layout.py` 一般不直接读这类配置参数。
- `layout.py` 只依赖真实 `Mesh1D` 的 cell 数和 `RunConfig.species_maps/unknowns_profile`。

### D. species / initialization 参数

例如：

- `initialization.gas_composition_basis`
- `species.allow_missing_vapor_init`

通常修改：

- `config_schema.py`
- YAML 模板
- `SpeciesControlConfig` 或 `InitializationConfig`
- `preprocess.py`

并可能影响：

- `SpeciesMaps`
- `layout.py`
- `state_pack.py`
- `physics/initial.py`

### E. solver / timestep / recovery 参数

例如：

- `solver_inner_petsc.linesearch_type`
- `time_stepper.max_step_count`

通常修改：

- `config_schema.py`
- YAML 模板
- 对应 dataclass
- `preprocess.py`
- 对应 solver 模块

### F. 只影响输出或诊断的参数

例如：

- `diagnostics.write_residual_norm`
- `output.write_restart_state`

通常修改：

- `config_schema.py`
- YAML 模板
- `DiagnosticsConfig` 或 `OutputConfig`
- `preprocess.py`
- `io/*`

## 当前配置参数的正式传递示例

### `mesh.*`

当前链条：

```text
YAML.mesh
-> schema 校验
-> preprocess 构造 MeshConfig
-> RunConfig.mesh
-> grid.py 生成真实 Mesh1D
-> layout.py 只消费真实 Mesh1D 的 n_liq / n_gas
```

关键口径：

- 配置只描述 mesh 生成参数。
- 真正的 cell 数以后以 `Mesh1D` 为准。

### `species.*`

当前链条：

```text
YAML.species
-> schema 校验
-> preprocess 结合 gas mechanism 和 liquid initialization 校验
-> SpeciesControlConfig
-> SpeciesMaps
-> RunConfig.species / RunConfig.species_maps / RunConfig.unknowns_profile
-> layout.py / state_pack.py / interface / assembly
```

### `initialization.*`

当前链条：

```text
YAML.initialization
-> schema 校验
-> preprocess 展开为 full-order 向量
-> InitializationConfig
-> RunConfig.initialization
-> physics/initial.py / first-step init / state construction
```

## 新增参数时的 10 条检查单

以后每加一个参数，都至少过一遍这张清单：

```text
1. 它属于哪个 section？
2. schema 是否已声明？
3. YAML 模板是否已更新？
4. make_minimal_valid_raw_config() 是否已更新？
5. types.py 对应 dataclass 是否已更新？
6. preprocess.py 是否已把它转入 RunConfig？
7. 是否需要路径解析？
8. 是否需要 cross-field / cross-source consistency check？
9. 是否影响 grid / layout / state_pack / remap / solver / io？
10. 是否补了对应 pytest？
```

## 当前方法的审查结论

当前这套“新增配置参数的标准修改方法”主线是准确的，可以作为 `paper_v1` 后续开发的正式维护规程。

但需要补充两条口径说明：

1. `config_loader.py` 通常不是新增普通配置参数时的修改点；只有 raw 文件读取语义变化时才修改它。
2. 模板 YAML 通过 `loader/schema` 不等于可以直接 `preprocess`；凡是相对路径引用的外部资源，都必须以配置文件所在目录为基准检查真实存在性。

## 推荐的后续开发顺序

当前 `core` 1-5 已经形成：

```text
raw YAML -> validated raw_cfg -> normalized RunConfig -> UnknownLayout
```

在这条链成立的前提下，建议后续顺序为：

1. `core/grid.py`
2. `core/state_pack.py`
3. `core/remap.py`
4. `core/state_recovery.py`

理由：

- `layout.py` 已经依赖 `Mesh1D`
- `remap.py` 和 `state_recovery.py` 都会更深地依赖网格和状态组织
- 先把 `grid.py` 和 `state_pack.py` 钉死，后续链条更稳
