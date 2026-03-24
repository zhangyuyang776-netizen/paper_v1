# outer_inner_iteration_coupling_guideline_final

## Status

本文件已废止，不再作为 `paper_v1` 的 outer/inner 主时序合同。

## Superseded By

请改用：

- `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`

## Binding Rule

关于以下内容，均以新文件为唯一主合同：

1. outer / inner 的正式职责边界
2. 第 0 轮 outer 的 inner 入口状态定义
3. predictor / corrector 的执行时序
4. remap / state recovery 在 outer 循环中的时间位置
5. 时间步接受、拒绝与回退前的 outer/inner 耦合流程

## Migration Note

旧版文件中的下列表述不再有效：

- 将 `old_state_on_current_geometry^(k)` 作为 outer iterate 主变量
- 在每轮 inner 之前固定执行“由 `U^n` remap 到当前几何”
- 将 remap / state recovery 视为每轮 inner 的固定前置动作
- 用 `a^(k)`、`dot_a^(k)` 再推进一次 `dt` 来构造 corrector

若其他文档仍引用本文件，其 outer/inner 主线语义也必须按
`outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`
解释。
