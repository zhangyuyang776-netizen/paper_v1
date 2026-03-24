# diagnostics_and_conservation_monitoring_guideline_final

## 1. 文件定位

本文档用于为 `paper_v1` 正式规定以下内容：

1. 运行日志（console / run.log）的输出层级与字段；
2. step-level、outer-level、inner-summary 级别的结构化诊断输出；
3. 界面专属诊断、remap/recovery/物性异常诊断；
4. 守恒监控与 budget 监控；
5. 标量时间序列输出；
6. 间隔步空间场快照输出；
7. 失败步取证输出；
8. 配置文件中与 diagnostics/output 相关的强制键；
9. diagnostics 相关的禁止事项。

本文档是 `paper_v1` 中**运行诊断、守恒监控与数据输出**的正式定稿文件。

---

## 2. 总原则

### 2.1 诊断不是临时 debug 打印

正式原则：

> `paper_v1` 的 diagnostics 是数值方法的一部分，不是临时调试附件。

旧项目已经表明，只有当日志、CSV、JSON、状态快照、专项脚本和测试约束形成体系时，失败步才真正可定位，而不是只剩一句“SNES diverged”。旧项目已经形成了“日志 + 结构化诊断文件 + 状态快照 + 验收脚本 + 测试契约”的多层体系，这一点应作为 `paper_v1` 的直接经验基础。 

### 2.2 诊断必须分层

正式规定，`paper_v1` 的 diagnostics 分为四层：

1. **运行日志层**：控制台 + un.log`
2. **结构化诊断层**：`step_diag.csv`、`interface_diag.csv`
3. **空间场快照层**：`3D_out/mapping.json`、`3D_out/steps/*.npz`
4. **失败步取证层**：`failed_step_xxxxxx_retry_yy/`

### 2.3 accepted step 与 rejected attempt 都必须留痕

正式规定：

* accepted step 必须写日志、标量、必要时写空间场；
* rejected attempt 也必须写摘要诊断；
* fatal stop 必须生成结构化失败报告。

### 2.4 inner / outer / step 三层必须分开记录

正式规定：

* inner 逐 Newton 迭代不做日志洪水
* outer 逐轮必须有明确编号与收敛指标
* step 必须有 accepted / rejected / retry 的明确记录

---

## 3. 诊断体系总览

参考旧项目的经验，`paper_v1` 正式采用如下层级化载体。旧项目中已经证明高价值的载体包括 un.log`、`scalars.csv`、`interface_diag.csv`、`3D_out/mapping.json + steps/*.npz`、`failed_step_xxxxxx/` 等。  

| 层级            | 载体                                         | 作用                                             |
| ------------- | ------------------------------------------ | ---------------------------------------------- |
| Driver 层      | un.log`、控制台日志                            | case 配置、每步摘要、失败原因、早停原因                         |
| Step 层        | `step_diag.csv`、`scalars.csv`              | 时间步、outer、inner 摘要，标量趋势                        |
| Interface 层   | `interface_diag.csv`                       | 界面平衡、能量通量、质量平衡、Jacobian/line-search/guard 相关诊断 |
| Spatial 层     | `3D_out/mapping.json`、`3D_out/steps/*.npz` | 全场状态快照、布局映射、后处理基础                              |
| Failure 层     | `failed_step_xxxxxx_retry_yy/`             | 失败报告、最近状态、尝试状态、最终状态、并行 rank 元信息                |
| Test/Script 层 | 脚本与测试                                      | 诊断字段是否存在、是否自洽、是否可后处理                           |

---

## 4. 运行日志规范（console / run.log）

## 4.1 总体要求

un.log` 是最直接的诊断入口。旧项目的 un.log` 已经证明，最有价值的不是海量逐迭代打印，而是：
**先兆警告 → 求解器异常 → 步失败** 这条时序链。旧项目日志中已经能记录 `[D_FLOOR]`、`SNES not converged`、`GLOBAL_DOMAIN_EVENT`、`Step N failed` 这类高价值事件。 

正式规定：

* un.log` 必须默认开启；
* 控制台日志与 un.log` 内容应保持一致；
* 日志必须按 **step → outer → inner-summary** 三层组织；
* 不允许只输出成功步，不输出失败步。

---

## 4.2 Step 级日志字段

每个 step attempt 都必须有唯一标识：

* `step_id`
* etry_id`
* `accepted_state_id`
* `t_old`
* `t_new_target`
* `dt_try`

每个 step attempt 的摘要日志至少包含：

* `step_id`
* etry_id`
* `t_old -> t_new_target`
* `dt_try`
* `status = accepted/rejected`
* `failure_class`（若 rejected）
* eject_reason`（若 rejected）
* `outer_iter_count`
* `inner_iter_count_last`
* `Ts`
* `mpp`
* `a`
* `dot_a_phys`
* `Tg[min,max]`
* `Tl[min,max]`

### 推荐日志示意

```text
[STEP 0041 RETRY 00] t=4.100e-5 -> 4.200e-5 dt=1.0e-6 status=attempt
[STEP 0041 RETRY 00 OUTER 00] a_guess=... dot_a_guess=... Ts=... mpp=...
[STEP 0041 RETRY 00 OUTER 00 INNER] converged=true it=9 Rinf=... snes_reason=...
[STEP 0041 RETRY 00 OUTER 00] dot_a_phys=... eps_dot_a=...
[STEP 0041 RETRY 00] status=accepted outer=2 Ts=... mpp=... Tg[min,max]=... Tl[min,max]=...
```

---

## 4.3 Outer 级日志字段

每个 outer iteration 都必须输出：

* `outer_iter_id`
* `a_guess`
* `dot_a_guess`
* `dot_a_phys`
* `eps_dot_a`
* `outer_relaxation_used`
* `inner_converged`
* `inner_iter_count`
* `R_total_inf`
* `R_if_inf`

理由很简单：
outer 是当前项目的显式第二层，不单独记录 outer，就无法知道“inner 已经收敛但 outer 还没收敛”的真实情况。

---

## 4.4 Inner 摘要日志字段

inner 不输出逐 Newton 序号流水，但每次 outer 轮结束后必须输出摘要：

* `inner_converged`
* `inner_iter_count`
* `snes_reason`
* `ksp_reason`
* `R_total_inf`
* `R_total_2`
* `R_liq_inf`
* `R_if_inf`
* `R_gas_inf`
* `line_search_backtracks`
* `jacobian_rebuild_count`
* `local_recovery_used`

旧项目已经证明，`snes_reason / ksp_reason / residual hotspot / Jmpp 行` 这类摘要比逐迭代刷屏更有价值。 

---

## 4.5 日志中的先兆警告

正式规定，以下类型的事件必须单独输出 warning，而不能埋进普通摘要中：

* 扩散系数触底，如旧项目中的 `[D_FLOOR]` / `[D_FLOOR_SEVERE]` 类事件；
* 恢复模块 minor fix 次数明显上升；
* remap 守恒误差超过预警阈值；
* 物性 fallback 触发；
* line search 连续 shrink；
* 气相/液相温度接近配置边界；
* 界面质量/能量/物种通量 jump 明显增大。
  旧项目里 `[D_FLOOR]` 的先兆信息非常有用，建议在 `paper_v1` 中保留“先兆警告”这一层，而不是等到 SNES 才报错。

---

## 5. 结构化 step diagnostics

## 5.1 文件职责

正式规定 `paper_v1` 至少输出两类结构化 CSV：

1. `step_diag.csv`
2. `interface_diag.csv`

### 分工

* `step_diag.csv`：时间步、outer、inner 摘要与全局标量
* `interface_diag.csv`：界面平衡、界面通量、一致性、界面 guard/line-search/Jacobian 相关诊断

---

## 5.2 `step_diag.csv` 的正式字段

每个 step attempt 写一行；accepted 与 rejected 都要写，但必须带 `accepted` 布尔字段。

### 必需字段

#### 标识

* `step_id`
* etry_id`
* `accepted_state_id`
* `accepted`

#### 时间推进

* `t_old`
* `t_new_target`
* `dt_try`
* `dt_next`
* `dt_change_reason`

#### outer 摘要

* `outer_iter_count`
* `eps_dot_a`
* `dot_a_guess_last`
* `dot_a_phys_last`
* `a_guess_last`
* `a_final_or_last`
* `outer_relaxation_used_last`
* `outer_nonmonotonic_flag`

#### inner 摘要

* `inner_converged`
* `inner_iter_count_last`
* `snes_reason`
* `ksp_reason`
* `R_total_inf`
* `R_total_2`
* `R_liq_inf`
* `R_if_inf`
* `R_gas_inf`
* `R_argmax_block`
* `R_argmax_desc`
* `R_argmax_value`

#### 关键物理量

* `Ts`
* `mpp`
* `a`
* `dot_a_phys`
* `Tg_min`
* `Tg_max`
* `Tl_min`
* `Tl_max`

#### 失败分类

* `failure_class`
* eject_reason`

#### remap / recovery / property

* emap_success`
* ecovery_success`
* `property_success`
* `nonphysical_flag`

---

## 5.3 `scalars.csv` 的定位

旧项目中的 `scalars.csv` 是可配置标量输出，适合长期趋势监控，而不是失败点定位。`paper_v1` 保留这个思路，但不允许它替代 `step_diag.csv`。

正式规定：

* `scalars.csv` 用于用户关心的长期标量趋势；
* `step_diag.csv` 用于求解器与方法诊断；
* 两者可以有字段重叠，但职责不同。

---

## 5.4 `scalars.csv` 的首版推荐字段

虽然 `scalars.csv` 允许配置选择字段，但首版建议至少默认包括：

* `step`
* `t`
* `dt`
* `Ts`
* `a`
* `mpp`
* `dot_a_phys`
* `Tg_min`
* `Tg_max`
* `Tl_min`
* `Tl_max`
* `if_energy_flux_jump`
* `if_species_flux_jump_max`
* emap_mass_err_liquid`
* emap_mass_err_gas`

旧项目配置里已经使用过 `step/t/Ts/Rd/mpp/mass_balance...` 这一类标量输出字段，可以作为 `paper_v1` 标量输出风格的参考。

---

## 6. Interface diagnostics

## 6.1 文件定位

正式规定保留独立的：

* `interface_diag.csv`

原因：

* 界面块是单独 unknown block；
* 界面平衡、界面通量、一致性 jump、line-search/guard/Jmpp 行结构，本来就不属于普通 step scalar；
* 旧项目已经证明，`interface_diag.csv` 是最核心的结构化诊断面板之一。 

---

## 6.2 `interface_diag.csv` 的正式字段组

### A. 基本信息

* `step_id`
* etry_id`
* `accepted`
* `t`
* `dt_try`
* `Ts`
* `a`
* `mpp`

### B. 界面组成与闭合

* `sum_y_cond`
* `sum_y_noncond`
* `Ys_g_closure_err`
* `eq_model_status`
* `eq_exc_type`
* `eq_exc_msg`
* `eq_source`
* `fallback_reason`

### C. 界面质量一致性

* `if_mass_balance_err_liq_side`
* `if_mass_balance_err_gas_side`
* `mpp_eval_minus_state`

### D. 界面物种一致性

* `if_species_flux_jump_max`
* `if_noncond_flux_max`

### E. 界面能量一致性

* `q_g`
* `q_l`
* `q_diff`
* `latent`
* `if_energy_flux_jump`

### F. 近沸腾/退化诊断

* `Tbub`
* `Ts_minus_Tbub`
* `DeltaY_raw`
* `DeltaY_eff`
* `deltaY_min_applied`

### G. guard / line-search / penalty

* `guard_active`
* `penalty_used`
* `penalty_reason`
* `n_penalty_residual`
* `ls_lambda`
* `ls_clip_ts`
* `ls_clip_mpp`
* `ls_shrink_count`

### H. Jacobian / preconditioner / mpp 行诊断

* `Jmpp_ok`
* `Jmpp_diag`
* `Jmpp_row_l2`
* `Jmpp_diag_over_row_l2`
* `A_mpp_diag_over_l2`
* `P_mpp_diag_over_l2`
* `mpp_row_signature`

旧项目里这组字段已经被证明很有价值，因为它们能直接告诉你问题是从“界面热平衡、组分平衡、线搜索、VI 投影、Jacobian 行结构”中的哪一层开始失真。 

---

## 6.3 accepted 与 rejected 的写入规则

正式规定：

* accepted step：必须写 `interface_diag.csv`
* rejected attempt：若已经构造到界面诊断层，则也必须写一行，且 `accepted=false`
* 若失败发生得太早，无法形成完整界面字段，则允许写 NaN，但必须写出：

  * `step_id`
  * etry_id`
  * `accepted=false`
  * `failure_class`
  * eject_reason`

---

## 7. residual / remap / recovery / property diagnostics

## 7.1 残差热点定位

正式规定，每次 inner summary 都必须保留：

* `f_idx_global`
* `f_max_abs`
* `f_max_val`
* `f_desc`

或等价字段：

* `R_argmax_block`
* `R_argmax_desc`
* `R_argmax_value`

旧项目已经证明，“最大残差项在哪、属于哪个物理块”是失败排查第一手信息。 

---

## 7.2 remap diagnostics

正式纳入统一 diagnostics 体系的 remap 字段包括：

* emap_mass_err_liquid`
* emap_mass_err_gas`
* emap_species_err_max`
* emap_enthalpy_err_liquid`
* emap_enthalpy_err_gas`

它们应同时出现在：

* `step_diag.csv`
* 失败步的 `failure_report.json`
* 必要时写入 un.log` warning

---

## 7.3 recovery diagnostics

正式纳入统一 diagnostics 体系的 recovery 字段包括：

### 液相

* `liq_recovery_success`
* `liq_enthalpy_inversion_iters_max`
* `liq_enthalpy_residual_max`
* `liq_recomputed_enthalpy_err_max`

### 气相

* `gas_recovery_success`
* `gas_recovery_used_HPY`
* `gas_recovery_used_fallback`
* `gas_enthalpy_inversion_iters_max`
* `gas_enthalpy_residual_max`
* `gas_recomputed_enthalpy_err_max`

### 组分

* `max_Y_sum_err`
* `max_negative_partial_density`
* `n_species_minor_fix`

---

## 7.4 property diagnostics

建议正式保留以下字段：

* `n_property_fallback_hits`
* `n_diffusivity_floor_hits`
* `n_T_clamped`
* `max_property_eval_retry`
* `property_last_error_class`

旧项目中的 `[D_FLOOR]`、property floor 先兆、sanity 检查信息都表明，物性异常必须在真正求解崩掉之前被看到。 

---

## 8. 守恒监控

## 8.1 总原则

守恒监控分三层：

1. **remap conservation**
2. **accepted-step budgets**
3. **interface consistency jumps**

不追求首版就做成一套冗长的全局解析守恒报告，但必须从第一版开始保留足够的数值闭合证据。

---

## 8.2 accepted-step 全局/分相预算字段

对每个 accepted step，建议至少输出以下 budget 字段：

### 液相总量

* `mass_liquid_old`
* `mass_liquid_new`
* `enthalpy_liquid_old`
* `enthalpy_liquid_new`

### 气相总量

* `mass_gas_old`
* `mass_gas_new`
* `enthalpy_gas_old`
* `enthalpy_gas_new`

### 组分总量

* `species_mass_i_old`
* `species_mass_i_new`
  （实现上可拆成单独文件或多列展开）

### 界面交换积分

* `interface_mass_transfer_integral`
* `interface_energy_transfer_integral`

### 派生误差

* `mass_budget_err_total`
* `enthalpy_budget_err_total`
* `species_budget_err_max`

---

## 8.3 accepted-step budget 的解释

首版中，这些 budget 主要作为：

## **step diagnostic budget**

也就是：

* 比较两个 accepted state 之间的总量变化；
* 与该步界面传递量、边界通量量级做比对；
* 用于监控“是不是某一步开始明显跑偏”。

它不是一开始就要求写成“理论上逐项精确匹配的闭合报告”，但必须足够用来发现数值偏移。

---

## 8.4 界面一致性 jump 也计入守恒监控

以下界面 jump 正式纳入 conservation monitoring：

* `if_mass_balance_err_liq_side`
* `if_mass_balance_err_gas_side`
* `if_species_flux_jump_max`
* `if_energy_flux_jump`

因为它们本质上就是局部守恒/一致性诊断。

---

## 9. 标量时间序列输出

## 9.1 文件定位

正式建议：

* `scalars.csv` 作为用户与后处理最常用的标量时间序列；
* 每个 accepted step 写一行。

## 9.2 首版最小必需字段

* `step`
* `t`
* `dt`
* `Ts`
* `a`
* `mpp`
* `dot_a_phys`
* `Tg_min`
* `Tg_max`
* `Tl_min`
* `Tl_max`

## 9.3 首版建议扩展字段

* `R_total_inf`
* `R_if_inf`
* `outer_iter_count`
* `inner_iter_count_last`
* `if_energy_flux_jump`
* `if_species_flux_jump_max`
* emap_mass_err_liquid`
* emap_mass_err_gas`

---

## 10. 空间场快照输出

## 10.1 文件定位

空间场输出正式采用：

* `3D_out/mapping.json`
* `3D_out/steps/step_XXXXXX_time_*.npz`

旧项目这套“mapping + npz snapshot”的设计已经被证明很有价值，因为它能把抽象向量重新翻译回物理字段。`mapping.json` 描述块布局、offset、shape、species 顺序，`steps/*.npz` 保存 `u`、网格与界面信息。 

---

## 10.2 输出触发规则

正式规定：

* 每 `output_every_n_steps` 个 accepted steps 输出一次；
* 失败前允许强制输出一次 failure snapshot；
* 不要求每一步都输出空间场，避免无谓 I/O 膨胀。

---

## 10.3 空间场快照的正式内容

### liquid fields

* _l`
* f_l`
* `Tl`
* `Yl_full`
* ho_l`
* `h_l`
* `u_l_faces` 或 `u_l_centers`（必须统一一种）

### gas fields

* _g`
* f_g`
* `Tg`
* `Yg_full`
* `Xg_full`
* ho_g`
* `h_g`
* `u_g_faces` 或 `u_g_centers`

### interface fields

* `a`
* `Ts`
* `mpp`
* `Ys_l_full`
* `Ys_g_full`

---

## 10.4 mapping 文件的正式内容

`mapping.json` 至少必须包含：

* `case_id`
* `layout_version`
* `field_group`
* `field_names`
* `units`
* `species_order`
* `offsets`
* `shapes`
* `grid_convention`
* adial_normal_convention`

这样后续后处理脚本才能稳定工作。
旧项目的 `mapping.json + postprocess_u_to_csv.py` 路线证明了这套做法的必要性。

---

## 11. 失败步取证

## 11.1 目录命名

正式规定：

每个 rejected attempt 或 fatal stop，应创建：

* `failed_step_000041_retry_02/`

不允许只用 `failed_step_000041/` 把多次重试混在一起。

---

## 11.2 失败目录最小内容

### 必需文件 1：`failure_report.json`

至少包含：

* `step_id`
* etry_id`
* `t_old`
* `dt_try`
* `failure_class`
* eject_reason`
* `inner_converged`
* `outer_converged`
* `snes_reason`
* `ksp_reason`
* `R_total_inf`
* `R_if_inf`
* `eps_dot_a`
* `Ts`
* `mpp`
* `a_guess`
* `dot_a_guess`
* `dot_a_phys`
* emap_fail_flag`
* ecovery_fail_flag`
* `property_fail_flag`
* `nonphysical_flag`

旧项目里的 `failure_report.json` 已经证明这类聚合报告对失败排查最有价值。 

### 必需文件 2：`state_last_good.npz`

最近 accepted state

### 必需文件 3：`state_attempt_input.npz`

本次 attempt 的输入状态

### 必需文件 4：`state_attempt_final.npz`

失败前最后一次可获得状态

### 必需文件 5：`mapping.json`

布局映射

---

## 11.3 并行取证文件

在 MPI > 1 时，额外要求输出：

* ank_000_meta.json`
* ank_001_meta.json`
* ...
* ank_000_snes_last_x.csv`
* ank_001_snes_last_x.csv`
* ...

旧项目里 ank_*_meta.json` 与 ank_*_snes_last_x.csv` 已经证明，对定位“哪个 rank 的哪段自由度先坏掉”非常有效。

### 推荐 rank meta 字段

* ank_id`
* `local_dof_range`
* `f_idx_global`
* `f_desc`
* `line_search_clip_count`
* `jacobian_rows`
* `Jmpp_diag`
* `vi_bounds`
* `x_min/x_max`
* `f_min/f_max`

---

## 11.4 失败步的最短排查路径

正式建议的排查顺序可直接写进开发文档：

1. 看 un.log` 最后 50~200 行；
2. 看 `interface_diag.csv` 最后一条相关记录；
3. 看 `failure_report.json`；
4. 若需定位具体变量，再看 `state_attempt_final.npz` + `mapping.json`；
5. 若是并行失败，再看 ank_*_meta.json` 与 ank_*_snes_last_x.csv`。

旧项目报告已经明确给出过类似的高效排查路径，值得直接吸收。

---

## 12. 诊断配置键

以下配置键必须在配置文件中显式存在：

### 日志与结构化输出

* `log.enable_run_log`
* `log.console_level`
* `diag.write_step_diag`
* `diag.write_interface_diag`
* `diag.write_failure_report`

### 标量输出

* `output.write_scalars`
* `output.scalars_every_n_steps`
* `output.scalar_field_list`

### 空间场输出

* `output.write_spatial_snapshots`
* `output.output_every_n_steps`
* `output.spatial_field_groups`
* `output.write_failure_snapshot`

### 守恒监控

* `diag.enable_conservation_monitoring`
* `diag.write_budget_fields`
* `diag.write_remap_errors`

### 并行失败取证

* `diag.write_rank_failure_meta`
* `diag.write_rank_last_x`
* `diag.failure_dump_stride`

### 诊断详细程度

* `diag.verbose_nonlinear_summary`
* `diag.verbose_interface_panel`
* `diag.verbose_property_warnings`

正式原则：

> 不允许把关键 diagnostics 开关和输出频率偷偷写成代码内默认值覆盖配置值。

---

## 13. 正式禁止事项

以下做法在本文件下明确禁止：

1. 只写 accepted step，不写 rejected attempt；
2. 只打印报错字符串，不写 `failure_report.json`；
3. 只输出总残差，不输出分块残差与热点定位；
4. 只输出 `scalars.csv`，不保留 `step_diag.csv`；
5. 不给 `interface_diag.csv` 单独面板；
6. 把空间场快照只保存成裸数组，不写 `mapping.json`；
7. 并行失败时不落 rank 级局部取证；
8. 让 run.log 变成逐 Newton 迭代流水，淹没真正高价值事件；
9. 在物性/恢复/remap 出现失败时不写结构化失败标记；
10. 让 diagnostics 配置键隐式默认并可能覆盖用户配置；
11. 把 diagnostics 当成“后面再补”的次要功能；
12. 失败后只留下 console 输出，不留可复查文件。

---

## 14. 最终方法合同

### Diagnostics and Conservation Monitoring Contract

1. `paper_v1` uses a layered diagnostics system: run log, structured CSV diagnostics, spatial snapshots, and failure forensics.
2. Every step attempt must have a unique `(step_id, retry_id)` identity.
3. Accepted steps and rejected attempts must both leave structured traces.
4. Outer iterations must be explicitly numbered and logged.
5. Inner nonlinear iterations are summarized, not spammed in full iteration-by-iteration logs.
6. `step_diag.csv` is mandatory for step-level and solver-level summaries.
7. `interface_diag.csv` is mandatory as a dedicated interface diagnostics panel.
8. Conservation monitoring must include remap errors, accepted-step budgets, and interface consistency jumps.
9. Spatial snapshots must be accompanied by a machine-readable `mapping.json`.
10. Failure steps must generate structured forensic artifacts, not just text logs.
11. Diagnostics keys and output controls must be explicitly provided by configuration.
12. Diagnostics is a formal part of the numerical method contract, not an optional debug convenience.

---

## 15. 最终结论

`paper_v1` 的 diagnostics、守恒监控与数据输出正式采用如下主线：

* 运行日志分 step / outer / inner-summary 三层；
* 每个时间步、每次重试、每轮 outer 都必须可追踪；
* `step_diag.csv` 负责时间步与求解器摘要；
* `interface_diag.csv` 负责界面平衡、通量、一致性与 line-search/Jacobian 面板；
* accepted step 与 rejected attempt 都必须留结构化痕迹；
* 守恒监控至少包含 remap 误差、accepted-step budgets、界面 jump；
* 标量时间序列与空间场快照都纳入本文件统一规定；
* 失败步必须生成完整取证目录；
* diagnostics 是 `paper_v1` 数值方法的一部分，不是附属功能。

这就是 `paper_v1` 的 `diagnostics_and_conservation_monitoring_guideline_final.md` 正式定稿版本。
