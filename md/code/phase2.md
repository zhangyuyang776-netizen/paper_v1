下面给你一份 **Phase 2 详细工作路线**。我按“**先冻结共享合同，再按官方顺序逐模块实现，再做阶段性接通与验收**”来写，不重划架构，不回退旧路线。官方顺序上，Phase 2 应按 `liquid_db -> mix_rules -> liquid -> gas -> equilibrium -> aggregator` 推进；Phase 2 的目标是建立统一物性层，并且 property 层不能感知并行 ownership、液相必须从第一版起支持多组分、界面平衡与 bulk property 必须分开。 

另外，你现在有一个比“模块顺序”更硬的现实约束：Phase 1 审查后唯一明确 carry-forward 的链路，就是把 `remap -> state_recovery -> thermo backends` 正式接通。所以这份路线里我把它放在 **Phase 2 的 0 号子任务**，作为所有模块开工前的共享接口冻结。否则写着写着又会返工，软件工程最擅长干这种事。

---

# Phase 2 总目标与执行原则

## 本阶段总目标

建立统一的：

* 液相纯组分数据库层
* 液相混合规则层
* 液相 / 气相 thermo-property backend
* 界面平衡闭合
* bulk props 聚合出口

为后续 Phase 3 的 physics kernel 和 Phase 4 的 assembly 提供稳定 API。

## 本阶段必须始终遵守的边界

* 不把 `properties` 写成 YAML 读取层
* 不把 `properties` 写成 outer/inner 调度层
* 不在 `properties` 里散落 MPI ownership / rank / ghost 逻辑
* 不把液相写成单组分硬编码
* 不让 `aggregator` 再制造一套 interface 真相源
* 不在 physics 里偷偷再算一套不受控物性  

---

# Phase 2 详细工作路线

---

## 子任务 0：先冻结 Phase 2 共享接口合同

### 0.1 目的

在正式写 `properties/*` 之前，先把所有模块共用的最小接口固定下来，尤其是 recovery 依赖的 thermo backend 协议。否则后面 `liquid.py`、`gas.py`、`aggregator.py` 会各自发明一套口径。

### 0.2 需要先读的指导文件

* `paper_v1_code_architecture_and_coding_guideline_final.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `liquid_properties_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md`
* `paper_v1_代码开发总体路线方案.md` 

### 0.3 要冻结的内容

先明确以下最小协议：

#### A. Thermo backend 最小协议

```python
class GasThermoBackend(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def density_mass(self, T: float, Y_full: np.ndarray, P: float) -> float: ...
    def cp_mass(self, T: float, Y_full: np.ndarray, P: float) -> float: ...

class LiquidThermoBackend(Protocol):
    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def density_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def cp_mass(self, T: float, Y_full: np.ndarray) -> float: ...
```

#### B. `remap -> state_recovery` 透传协议

`remap.build_old_state_on_current_geometry(...)` 必须显式接收并透传：

* `liquid_thermo`
* `gas_thermo`

不允许 recovery 自己到处“找 backend”。

#### C. full composition 统一原则

* `properties` 层一律吃 **full-order composition**
* reduced/full 的互转只由 `state_pack`、`layout`、`SpeciesMaps` 负责
* `gas.py` 和 `equilibrium.py` 内部不接受 reduced mass fraction 当正式输入

#### D. Props 最小字段协议

先冻结 `Props` 第一版最小字段：

* `rho_l`, `rho_g`
* `cp_l`, `cp_g`
* `h_l`, `h_g`
* `k_l`, `k_g`
* `D_l`, `D_g`

后面扩展可以加，但前缀、shape、cell-centered 语义先定死。

### 0.4 本子任务输出

* 一页接口说明文档，或者直接在代码里形成：

  * `properties/protocols.py` 或放在 `properties/__init__.py` 旁边
  * 对 `remap.py` 的函数签名微调计划
  * `Props` 最小字段草案

### 0.5 验收标准

* 所有后续模块都能引用这份接口合同
* 不再出现“liquid backend 和 gas backend 返回字段风格不一致”的情况
* `state_recovery` 所需最小接口被正式定义

---

## 子任务 1：编写 `properties/liquid_db.py`

官方顺序里它是第一个模块，因为这是数据库层，不先定参数来源，后面的液相模型就会到处散公式。很符合人类编程传统，先乱写，后面再花三倍时间清理。这里别这么干。

### 1.1 依赖指导文件

* `liquid_properties_guideline_final.md`
* `paper_v1_config_schema_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 1.2 代码目标

实现液相纯组分数据库层，负责：

* 从 `RunConfig` 中得到 `liquid_database_path`
* 读取液相纯组分数据文件
* 校验字段完整性和单位一致性
* 按规范化后的 liquid species 名称提供参数查询

### 1.3 具体子任务

#### 1.3.1 设计数据结构

建议：

```python
@dataclass(frozen=True)
class LiquidPureSpeciesRecord:
    name: str
    molecular_weight: float
    rho_model: dict
    cp_model: dict
    k_model: dict
    mu_model: dict
    diffusivity_model: dict | None
    vapor_pressure_model: dict | None
    latent_heat_model: dict | None
    activity_model: dict | None
```

```python
@dataclass(frozen=True)
class LiquidDatabase:
    species_names: tuple[str, ...]
    records: dict[str, LiquidPureSpeciesRecord]
```

#### 1.3.2 实现加载函数

建议函数：

```python
load_liquid_database(path: PathLike) -> LiquidDatabase
validate_liquid_database(db: LiquidDatabase) -> None
get_liquid_record(db: LiquidDatabase, species_name: str) -> LiquidPureSpeciesRecord
```

#### 1.3.3 实现与 `RunConfig` 的桥接

建议函数：

```python
build_liquid_database_from_run_config(run_cfg: RunConfig) -> LiquidDatabase
```

### 1.4 关键约束

* 只管数据，不做 mixture property
* 不做 state recovery
* 不做界面平衡
* 不做 YAML 原始解析，`config_loader/preprocess` 已经做过前置规范化

### 1.5 单元测试

至少写 4 类测试：

1. 正常加载单组分数据库
2. 正常加载多组分数据库
3. 缺字段时报错
4. species 名称不匹配时报错

### 1.6 本模块完成标志

* `liquid.py` 能稳定查询纯组分参数
* 数据库层与计算层完全分离
* 后续不再需要在 `physics` 里手写液相纯组分参数

---

## 子任务 2：编写 `properties/mix_rules.py`

### 2.1 依赖指导文件

* `liquid_properties_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 2.2 代码目标

实现液相 mixture 的公共混合规则工具层，供 `liquid.py` 调用。

### 2.3 具体子任务

#### 2.3.1 组分换算工具

实现：

```python
mass_to_mole_fractions(Y: np.ndarray, mw: np.ndarray) -> np.ndarray
mole_to_mass_fractions(X: np.ndarray, mw: np.ndarray) -> np.ndarray
```

#### 2.3.2 mixture 标量聚合规则

实现：

```python
mix_mass_weighted(values: np.ndarray, Y: np.ndarray) -> float
mix_mole_weighted(values: np.ndarray, X: np.ndarray) -> float
```

#### 2.3.3 mixture 液相密度 / cp / h / k / mu / D 的规则

建议拆成显式函数：

```python
mixture_liquid_density(...)
mixture_liquid_cp(...)
mixture_liquid_enthalpy(...)
mixture_liquid_conductivity(...)
mixture_liquid_viscosity(...)
mixture_liquid_diffusivity(...)
```

### 2.4 关键约束

* 第一版就按多组分写
* 不允许“单组分先糊一个标量接口，后面再改 shape”
* 这里只放规则，不放数据库读取、不放状态对象、不放界面平衡

### 2.5 单元测试

1. 单组分退化时，mixture 值应回到纯组分值
2. 二元混合时，`Y -> X -> Y` 往返误差可控
3. mixture 函数输入 shape 非法时报错
4. 总和不为 1 时应显式报错或要求上层先保证合法，不做 silent correction

### 2.6 本模块完成标志

* `liquid.py` 不再需要自己重复写混合数学
* 单组分与多组分接口完全一致

---

## 子任务 3：编写 `properties/liquid.py`

这是 Phase 2 第一关键模块之一，因为它直接服务 `state_recovery` 的液相焓反解。官方职责里也明确要求它与 enthalpy inversion 接口保持一致。

### 3.1 依赖指导文件

* `liquid_properties_guideline_final.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `unknowns_strategy_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 3.2 代码目标

实现液相 thermo-property backend，支持：

* bulk liquid
* 界面液相状态
* recovery 需要的前向焓模型

### 3.3 具体子任务

#### 3.3.1 设计 backend 类

建议：

```python
@dataclass
class LiquidThermo:
    db: LiquidDatabase
    species_names: tuple[str, ...]
    molecular_weights: np.ndarray

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def density_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def cp_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def conductivity(self, T: float, Y_full: np.ndarray) -> float: ...
    def viscosity(self, T: float, Y_full: np.ndarray) -> float: ...
    def diffusivity_matrix(self, T: float, Y_full: np.ndarray) -> np.ndarray: ...
```

#### 3.3.2 纯组分前向模型

从 `liquid_db.py` 读出每个 pure species record，在给定 `T` 下先求：

* `rho_i(T)`
* `cp_i(T)`
* `h_i(T)`
* `k_i(T)`
* `mu_i(T)`

#### 3.3.3 mixture 聚合

调用 `mix_rules.py` 形成 mixture 值：

* `rho_l(T,Y)`
* `cp_l(T,Y)`
* `h_l(T,Y)`
* `k_l(T,Y)`
* `mu_l(T,Y)`

#### 3.3.4 与 recovery 的一致性校核

做一组 round-trip：

* 先给 `T_true, Y_full`
* 用 `liquid.py` 算 `h`
* 再交给 `state_recovery` 的 `h -> T` 反解
* 看能否回到 `T_true`

这一步必须做，不然你后面 debug 的不是方程，是你自己接口打架。

### 3.4 关键约束

* 必须支持单组分与多组分
* 不做 silent renormalize / silent clip
* 不吃 reduced unknown
* 不直接构造 `Props` 总表，那是 `aggregator.py` 的职责

### 3.5 单元测试

1. 单组分液体 `Y=[1]` 的所有函数能正常返回
2. 二元液体 `Y_full` 能得到有限正值
3. `enthalpy_mass(T, Y)` 随 T 单调，满足 recovery 反解需要
4. `round-trip h->T` 误差在容限内

### 3.6 模块完成标志

* `state_recovery` 的液相恢复可正式接通
* 后续 physics/liquid flux 不需要自己重建液相热容、焓、密度接口

---

## 子任务 4：编写 `properties/gas.py`

官方顺序里它排在 `liquid.py` 之后，但从 carry-forward 看，它也是 Phase 2 第一批必须尽快打通的模块，因为 recovery 也需要气相焓反解接口。

### 4.1 依赖指导文件

* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `unknowns_strategy_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 4.2 代码目标

实现气相 thermo-property backend，面向 bulk props 与 recovery。

### 4.3 具体子任务

#### 4.3.1 设计 backend 类

建议：

```python
@dataclass
class GasThermo:
    mechanism_path: str
    species_names: tuple[str, ...]
    closure_species: str

    def enthalpy_mass(self, T: float, Y_full: np.ndarray) -> float: ...
    def density_mass(self, T: float, Y_full: np.ndarray, P: float) -> float: ...
    def cp_mass(self, T: float, Y_full: np.ndarray, P: float) -> float: ...
    def conductivity(self, T: float, Y_full: np.ndarray, P: float) -> float: ...
    def mixture_diffusivity(self, T: float, Y_full: np.ndarray, P: float) -> np.ndarray: ...
    def species_enthalpies_mass(self, T: float) -> np.ndarray: ...
```

#### 4.3.2 明确 closure species 规则

* `gas.py` 不负责从 reduced 恢复 full
* 它只接受 full `Y_full`
* closure species 的补全由上游 state reconstruction 保证

#### 4.3.3 统一 full composition 校验

实现内部 helper：

```python
validate_full_mass_fractions(Y_full: np.ndarray, n_spec: int) -> None
```

#### 4.3.4 与 recovery 的一致性校核

同样做气相的 `T -> h -> T` 回归测试。

### 4.4 关键约束

* reduced/full 区分必须非常明确
* 不做 interface equilibrium
* 不私自改 composition
* property 层不感知 MPI ownership 

### 4.5 单元测试

1. full `Y_full` 输入能得到有限正值
2. closure species 存在但不参与 reduced 输入逻辑
3. `enthalpy_mass` 与 `state_recovery` 的气相反解一致
4. 非法 composition 报错，不 silent fix

### 4.6 模块完成标志

* `state_recovery` 的气相恢复可正式接通
* Phase 2 最核心的 thermo backend 链路完成一半以上

---

## 子任务 5：接通 `remap -> state_recovery -> thermo backends`

这一步虽然不是 `properties/` 单独某个模块，但它是 Phase 2 的第一个里程碑。没有它，前面写的 `liquid.py` 和 `gas.py` 只是漂亮的零件，没有闭成主线。你已经明确把这条链当作 carry-forward 第一优先项了，这里必须单列。

### 5.1 涉及模块

* `core/remap.py`
* `core/state_recovery.py`
* `properties/liquid.py`
* `properties/gas.py`

### 5.2 具体子任务

#### 5.2.1 调整 `remap.py` 签名

例如：

```python
build_old_state_on_current_geometry(
    old_state: State,
    old_mesh: Mesh1D,
    new_mesh: Mesh1D,
    run_cfg: RunConfig,
    liquid_thermo: LiquidThermoBackend,
    gas_thermo: GasThermoBackend,
    interface_seed: InterfaceSeed,
) -> OldStateOnCurrentGeometry
```

#### 5.2.2 调整 `state_recovery.py` 使用方式

确保 recovery 明确拿到：

* `liquid_thermo`
* `gas_thermo`

#### 5.2.3 做最小集成测试

测试路径：

* 构造旧几何合法 `State`
* 做几何变化后的 conservative remap
* 调用 recovery
* 生成 `old_state_on_current_geometry`
* 检查 `rho/Y/h/T` 都有限、shape 正确、interface seed 被正确透传

### 5.3 验收标准

* `old_state_on_current_geometry` 能合法生成
* recovery 不再依赖“未来才会出现的临时物性函数”
* Phase 1 到 Phase 2 的唯一关键接口真正打通

---

## 子任务 6：编写 `properties/equilibrium.py`

官方职责是只负责 Eq.(2.19) 的热力学闭合，不能顺手接管整个 interface residual。这个边界必须守住，不然 Phase 3/4 会马上糊成浆。

### 6.1 依赖指导文件

* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `interface_block_unknowns_and_residuals_table_final.md`
* `liquid_properties_guideline_final.md`
* `paper_v1 Initialization and First-Step Guideline.md`
* `unknowns_strategy_guideline_final.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 6.2 代码目标

给定界面状态，计算界面平衡闭合结果，例如：

* `Yg_eq_full`
* 可凝组分平衡分压 / 摩尔分数
* 必要的活动系数
* 诊断信息

### 6.3 具体子任务

#### 6.3.1 设计结果对象

建议：

```python
@dataclass
class InterfaceEquilibriumResult:
    Yg_eq_full: np.ndarray
    Xg_eq_condensable: np.ndarray
    psat_condensable: np.ndarray
    activity_coefficients: np.ndarray | None
    diagnostics: dict[str, Any]
```

#### 6.3.2 设计求解接口

```python
compute_interface_equilibrium(
    Ts: float,
    Yl_if_full: np.ndarray,
    pressure: float,
    liquid_thermo: LiquidThermo,
    species_maps: SpeciesMaps,
) -> InterfaceEquilibriumResult
```

#### 6.3.3 计算流程

1. `Yl_if_full -> Xl_if`
2. 从液相数据库/模型求纯组分 `psat_i(Ts)`
3. 必要时引入活动系数 `gamma_i`
4. 形成可凝组分平衡分压
5. 组装气侧 full-order 平衡组成 `Yg_eq_full`
6. 对背景气体补全并校核和为 1

### 6.4 关键约束

* 只做热力学闭合
* 不装配界面残差
* 不定义 `mpp` 主残差
* 不代替 `physics/interface_face.py`

### 6.5 单元测试

1. 单组分液滴能给出合理的 `Yg_eq_full`
2. 多组分液滴 shape 正确
3. species map 映射错误时报错
4. 结果总和合法，且背景气补全逻辑明确

### 6.6 模块完成标志

* Phase 3 的 `interface_face.py` 有稳定 equilibrium 输入源
* 不再需要在 residual 中临时算一套界面平衡

---

## 子任务 7：编写 `properties/aggregator.py`

官方职责是“基于 `State + Grid + Models` 统一构造 bulk `Props`”，并且推荐只聚合 bulk props，interface-specific 内容交给 `physics/interface_face.py`。这条边界特别重要，因为 Phase 3 明确要求 `InterfaceFacePackage` 成为唯一界面真相源。 

### 7.1 依赖指导文件

* `liquid_properties_guideline_final.md`
* `state_recovery_and_enthalpy_inversion_guideline_final.md`
* `governing_equations_discretization_and_bc_guideline_final_v2.md`
* `paper_v1_module_guideline_dependency_map_final.md` 

### 7.2 代码目标

根据：

* `State`
* `Mesh1D`
* `LiquidThermo`
* `GasThermo`

统一生成 cell-centered bulk `Props`。

### 7.3 具体子任务

#### 7.3.1 设计 `Props` 数据结构

建议：

```python
@dataclass
class BulkProps:
    rho_l: np.ndarray
    cp_l: np.ndarray
    h_l: np.ndarray
    k_l: np.ndarray
    D_l: np.ndarray | None

    rho_g: np.ndarray
    cp_g: np.ndarray
    h_g: np.ndarray
    k_g: np.ndarray
    D_g: np.ndarray | None

    diagnostics: dict[str, Any]
```

#### 7.3.2 实现液相 bulk 聚合

对每个 liquid cell：

* 读 `Tl[j]`, `Yl_full[j]`
* 调 `liquid_thermo` 得到 cell props

#### 7.3.3 实现气相 bulk 聚合

对每个 gas cell：

* 读 `Tg[j]`, `Yg_full[j]`, `P`
* 调 `gas_thermo` 得到 cell props

#### 7.3.4 保持 interface 空间边界清晰

* 不在 `aggregator` 里构造 interface face package
* 不在 `aggregator` 里产出界面通量
* 最多允许提供界面两侧相邻 cell 的 bulk props 读取支持

### 7.4 关键约束

* 只做 bulk props
* 不维护第二套界面真相源
* 不涉及 MPI ownership
* 输出 shape 必须与 mesh/liquid-gas 分区一致

### 7.5 单元测试

1. 合法 `State + Mesh1D` 可稳定生成 `BulkProps`
2. 单组分/多组分都能通过 shape 检查
3. 关键物性有限且应为正值
4. 不生成 interface-specific 字段，防止职责漂移

### 7.6 模块完成标志

* 达到官方的 Phase 2 最小验收条件之一：给定合法 `State` 和 `Grid`，能稳定求出 bulk `Props`。

---

## 子任务 8：Phase 2 集成验收

Phase 2 不能只看单元测试。还必须做一次完整的“物性层串联检查”，否则每个模块都自称没问题，合起来像灾后现场。

### 8.1 集成链路

按下面顺序做：

1. `build_liquid_database_from_run_config(run_cfg)`
2. `LiquidThermo(db, ...)`
3. `GasThermo(mechanism, ...)`
4. `remap.build_old_state_on_current_geometry(..., liquid_thermo, gas_thermo, ...)`
5. `state_recovery.recover_state_from_contents(...)`
6. `aggregator.build_bulk_props(state, mesh, liquid_thermo, gas_thermo)`
7. `equilibrium.compute_interface_equilibrium(Ts, Yl_if_full, P, ...)`

### 8.2 必做测试场景

#### 场景 A：单组分液滴

* 单组分 liquid
* 简单背景气
* 检查 shape、正值、round-trip 恢复

#### 场景 B：多组分 liquid 结构测试

* 即使首个运行 case 不一定是多组分，也必须做结构测试
* 检查 `Yl_full` 多组分路径全通
* 检查 `equilibrium` 的多组分映射与输出 shape

#### 场景 C：几何变化后的 remap + recovery

* 几何改变后仍能形成合法 `old_state_on_current_geometry`
* 这一步是 current-geometry backward Euler 的前提，不做就是自欺欺人。

### 8.3 Phase 2 官方验收目标对应检查

根据总路线文件，本阶段至少要满足：

1. 给定合法 `State` 和 `Grid`，能稳定求出 bulk `Props`
2. 气相/液相关键物性为有限正值
3. `equilibrium.py` 能给出界面平衡输出
4. 单组分 case 与多组分 case 均通过 shape 和字段检查
5. property 接口稳定，后续 physics 无需改动 property API 

---

# 推荐的实际编码批次

为了减少返工，我建议你按下面 4 个批次写，而不是模块写完一个就以为世界和平了。

## 批次 A：接口冻结与数据库层

* 子任务 0
* 子任务 1

**产出**：接口合同 + `liquid_db.py`

## 批次 B：热力学最小闭环

* 子任务 2
* 子任务 3
* 子任务 4
* 子任务 5

**产出**：`mix_rules.py`、`liquid.py`、`gas.py`，以及 `remap -> recovery -> thermo` 接通

这是 **Phase 2 最关键的第一里程碑**。

## 批次 C：界面平衡与 bulk 聚合

* 子任务 6
* 子任务 7

**产出**：`equilibrium.py`、`aggregator.py`

## 批次 D：Phase 2 总体验收

* 子任务 8

**产出**：Phase 2 验收报告 / 测试清单

---

# 我建议你在编写时采用的交付顺序

真正落笔时，我建议按下面顺序逐个让我给方案或代码，不要一口气并行铺开六个模块：

1. `properties/liquid_db.py`
2. `properties/mix_rules.py`
3. `properties/liquid.py`
4. `properties/gas.py`
5. `core/remap.py` 与 `core/state_recovery.py` 的接口接通补丁
6. `properties/equilibrium.py`
7. `properties/aggregator.py`
8. Phase 2 集成测试

这既符合官方 Phase 2 顺序，也照顾了你当前最优先的 carry-forward 链路。

---

# 一句话总结

Phase 2 不是“写几个物性函数”，而是要先把 **数据库层、混合规则层、前向 thermo 层、界面平衡层、bulk 聚合层** 分清，再把 **`remap -> recovery -> thermo`** 这条主线真正接通。只要这一步走对，Phase 3 的 `physics` 才不会被迫替 `properties` 擦屁股。

下一步最合理的推进方式，是我直接按这条路线，先给出 **`properties/liquid_db.py` 的详细代码编写方案与验收点**。
