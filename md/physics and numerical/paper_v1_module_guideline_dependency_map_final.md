# paper_v1_module_guideline_dependency_map_final

## 1. 文件目的

本文档用于提供 `paper_v1` 的“模块代码 -> 指导文件”快速索引。  
后续制定编写方案时，先查本文件，再去打开对应指导文件。

使用规则：

1. 所有模块默认都先服从：
   - `paper_v1_code_architecture_and_coding_guideline_final.md`
2. 本文件列的是每个模块的**额外强依赖指导文件**
3. 涉及物理原式时，仍需回到主参考文献核对
4. 若同主题同时存在旧版与 `*_final_v2.md`，必须先看：
   - `../DOCUMENT_PRIORITY_AND_SUPERSEDED_NOTICE.md`
   然后只执行新版文件

---

## 2. `core/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `core/types.py` | `paper_v1_config_schema_guideline_final.md`, `unknowns_strategy_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `core/config_schema.py` | `paper_v1_config_schema_guideline_final.md`, `liquid_properties_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md`, `diagnostics_and_conservation_monitoring_guideline_final.md` |
| `core/config_loader.py` | `paper_v1_config_schema_guideline_final.md` |
| `core/preprocess.py` | `paper_v1_config_schema_guideline_final.md`, `unknowns_strategy_guideline_final.md`, `paper_v1 Initialization and First-Step Guideline.md`, `liquid_properties_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `core/layout.py` | `unknowns_strategy_guideline_final.md`, `interface_block_unknowns_and_residuals_table_final.md`, `paper_v1_config_schema_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `core/grid.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `paper_v1 Initialization and First-Step Guideline.md`, `velocity_recovery_guideline_final.md` |
| `core/state_pack.py` | `unknowns_strategy_guideline_final.md`, `interface_block_unknowns_and_residuals_table_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `core/remap.py` | emap_and_conservative_projection_guideline_final_v2.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `velocity_recovery_guideline_final.md`, `diagnostics_and_conservation_monitoring_guideline_final.md` |
| `core/state_recovery.py` | `state_recovery_and_enthalpy_inversion_guideline_final.md`, `liquid_properties_guideline_final.md`, `unknowns_strategy_guideline_final.md`, `diagnostics_and_conservation_monitoring_guideline_final.md` |
| `core/logging_utils.py` | `diagnostics_and_conservation_monitoring_guideline_final.md` |

---

## 3. `properties/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `properties/gas.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `unknowns_strategy_guideline_final.md` |
| `properties/liquid_db.py` | `liquid_properties_guideline_final.md`, `paper_v1_config_schema_guideline_final.md` |
| `properties/liquid.py` | `liquid_properties_guideline_final.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md` |
| `properties/mix_rules.py` | `liquid_properties_guideline_final.md` |
| `properties/equilibrium.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `liquid_properties_guideline_final.md`, `paper_v1 Initialization and First-Step Guideline.md`, `unknowns_strategy_guideline_final.md` |
| `properties/aggregator.py` | `liquid_properties_guideline_final.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `governing_equations_discretization_and_bc_guideline_final_v2.md` |

---

## 4. `physics/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `physics/initial.py` | `paper_v1 Initialization and First-Step Guideline.md`, `unknowns_strategy_guideline_final.md`, `governing_equations_discretization_and_bc_guideline_final_v2.md` |
| `physics/flux_gas.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `velocity_recovery_guideline_final.md`, `unknowns_strategy_guideline_final.md` |
| `physics/flux_liq.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `liquid_properties_guideline_final.md`, `velocity_recovery_guideline_final.md` |
| `physics/flux_convective.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `velocity_recovery_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `physics/energy_flux.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `liquid_properties_guideline_final.md`, `interface_block_unknowns_and_residuals_table_final.md` |
| `physics/interface_face.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `liquid_properties_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `physics/interface_mass.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `velocity_recovery_guideline_final.md`, `liquid_properties_guideline_final.md` |
| `physics/interface_energy.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `liquid_properties_guideline_final.md`, `unknowns_strategy_guideline_final.md` |
| `physics/velocity_recovery.py` | `velocity_recovery_guideline_final.md`, `governing_equations_discretization_and_bc_guideline_final_v2.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `physics/radius_update.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `timestep_and_failure_policy_guideline_final_v2.md` |

---

## 5. `assembly/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `assembly/residual_liquid.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `velocity_recovery_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `assembly/residual_interface.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `velocity_recovery_guideline_final.md` |
| `assembly/residual_gas.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `velocity_recovery_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `assembly/residual_global.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `velocity_recovery_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/jacobian_pattern.py` | `unknowns_strategy_guideline_final.md`, `interface_block_unknowns_and_residuals_table_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/jacobian_liquid.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/jacobian_interface.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/jacobian_gas.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/jacobian_global.py` | `governing_equations_discretization_and_bc_guideline_final_v2.md`, `interface_block_unknowns_and_residuals_table_final.md`, `unknowns_strategy_guideline_final.md`, `velocity_recovery_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `assembly/petsc_prealloc.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `unknowns_strategy_guideline_final.md` |

---

## 6. `solvers/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `solvers/nonlinear_types.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `solvers/nonlinear_context.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `timestep_and_failure_policy_guideline_final_v2.md`, `unknowns_strategy_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `solvers/linesearch_guards.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `diagnostics_and_conservation_monitoring_guideline_final.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `solvers/petsc_linear.py` | `petsc_solver_and_parallel_architecture_guideline_final.md` |
| `solvers/petsc_snes.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md`, `diagnostics_and_conservation_monitoring_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `solvers/outer_predictor.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `paper_v1 Initialization and First-Step Guideline.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `solvers/outer_corrector.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `solvers/outer_convergence.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `diagnostics_and_conservation_monitoring_guideline_final.md`, `timestep_and_failure_policy_guideline_final_v2.md` |
| `solvers/step_acceptance.py` | `timestep_and_failure_policy_guideline_final_v2.md`, `diagnostics_and_conservation_monitoring_guideline_final.md`, `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md` |
| `solvers/timestepper.py` | `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`, `timestep_and_failure_policy_guideline_final_v2.md`, `paper_v1 Initialization and First-Step Guideline.md`, emap_and_conservative_projection_guideline_final_v2.md`, `state_recovery_and_enthalpy_inversion_guideline_final.md`, `velocity_recovery_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md`, `diagnostics_and_conservation_monitoring_guideline_final.md` |

---

## 7. `driver/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `driver/run_case.py` | `paper_v1_config_schema_guideline_final.md`, `paper_v1 Initialization and First-Step Guideline.md`, `timestep_and_failure_policy_guideline_final_v2.md`, `diagnostics_and_conservation_monitoring_guideline_final.md`, `petsc_solver_and_parallel_architecture_guideline_final.md` |

---

## 8. `io/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `io/output_layout.py` | `diagnostics_and_conservation_monitoring_guideline_final.md`, `paper_v1_config_schema_guideline_final.md` |
| `io/writers.py` | `diagnostics_and_conservation_monitoring_guideline_final.md`, `paper_v1_config_schema_guideline_final.md` |

---

## 9. `parallel/`

| 模块 | 必须先读的指导文件 |
|---|---|
| `parallel/mpi_bootstrap.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `paper_v1_config_schema_guideline_final.md` |
| `parallel/dm_manager.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `unknowns_strategy_guideline_final.md`, `paper_v1_code_architecture_and_coding_guideline_final.md` |
| `parallel/local_state.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `velocity_recovery_guideline_final.md`, `paper_v1_code_architecture_and_coding_guideline_final.md` |
| `parallel/fieldsplit_is.py` | `petsc_solver_and_parallel_architecture_guideline_final.md`, `unknowns_strategy_guideline_final.md`, `paper_v1_code_architecture_and_coding_guideline_final.md` |

---

## 10. 快速使用建议

制定编写方案时，推荐顺序：

1. 先看本文件，确定模块必须依赖哪些指导文件
2. 再看 `paper_v1_code_architecture_and_coding_guideline_final.md`，确认模块职责边界
3. 最后进入对应指导文件，按合同实现

这就是 `paper_v1` 的最终模块代码指导文件依赖索引。
