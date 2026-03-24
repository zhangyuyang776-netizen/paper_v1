# timestep_and_failure_policy_guideline_final

> Sync note:
> outer 主时序统一以
> `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`
> 为唯一主合同。本文件不再单独定义与其冲突的 outer/inner 顺序。

## 1. 文件定位

本文档用于正式规定 `paper_v1` 项目的以下内容：

1. 单时间步推进的总体结构；
2. `outer` / `inner` 两层迭代各自的职责边界；
3. 主参考文献中的时间步控制、失败回退与外层收敛策略如何映射到 `paper_v1`；
4. PETSc 在新项目中的替代边界；
5. 时间步接受/拒绝、失败分类、局部补救与回退逻辑；
6. step-level diagnostics 与 fatal stop 规则。

本文档是 `paper_v1` 中 **time step / step acceptance / failure handling** 的正式约束文件。  
其基线来源于主参考文献第二章，尤其是：

- 2.3.3 droplet surface tracking and meshing strategy
- 2.3.4 algorithms for non-linear equations
- 2.3.5 time stepping strategy

---

## 2. 总原则

### 2.1 文献基线优先

在时间推进、外层界面/网格校正、时间步调整方面，`paper_v1` 以主参考文献为基准。

文献已经明确给出：

1. 时间离散采用 backward Euler；
2. 界面/网格更新采用 predictor-corrector；
3. 外层收敛以“连续两次迭代的界面速度差”作为判据；
4. 时间步控制不是基于局部截断误差，而是基于非线性收敛难度；
5. 不收敛时采用 rollback + half-step；
6. 连续若干步稳定后，时间步以 1.1 倍缓慢增长。

### 2.2 PETSc 替代边界

`paper_v1` 中，PETSc 主要用于替代文献中 **固定几何下非线性代数系统求解器** 的部分。  
也就是说：

- PETSc 替代文献中的 SSM（Stepped Shamanskii Method）这一“inner nonlinear solver”
- PETSc **不替代** 文献的 outer predictor-corrector 主线
- PETSc **不直接替代** 文献的 timestep acceptance / rollback / growth policy

### 2.3 两层迭代职责分离

正式规定：

- `inner`：在 **fixed geometry / fixed outer iterate** 下，求解 liquid bulk + interface block + gas bulk 的耦合非线性方程组；
- `outer`：更新 `a` / `dot_a`、网格、`v_c`、remap，并检查界面速度一致性；
- `step controller`：负责时间步接受/拒绝、失败回退、步长缩小/增长。

其中，outer 的正式时序只认
`outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`。

必须严格区分以下三个状态：

1. `inner_converged`
2. `outer_converged`
3. `step_accepted`

禁止将 `inner` 收敛误判为整步成功。

---

## 3. 文献算法与 `paper_v1` 的映射关系

## 3.1 文献中的整体结构

文献在每个时间步中同时处理两件事：

1. 在 \( t+\Delta t \) 上求解液相、界面、气相耦合后的非线性代数系统；
2. 更新液滴界面位置与动网格，并继续校正，直到界面速度收敛。

因此，文献虽然没有使用 `outer/inner` 这两个名词，但其数值结构本质上就是：

- 一层“场变量非线性求解”
- 一层“界面/网格外循环校正”

## 3.2 `paper_v1` 的正式映射

文献结构映射到 `paper_v1` 后，正式定义为：

### outer
负责：

- predictor / corrector
- 更新 `a^(k)`、`dot_a^(k)`
- 构建当前几何 `G^(k)`
- 构建控制面速度 `v_c^(k)`
- 设置当前轮 inner 入口状态：`k=0` 时 `U_init^(0)=U^n`，`k>0` 时入口状态来自上一轮 outer 的状态转移结果
- 当 outer 未收敛并更新网格后，执行 remap / recovery，形成下一轮入口状态
- 调用 inner
- 检查界面速度一致性
- 决定本时间步是否接受

### inner
负责：

- 在当前 outer iterate 冻结的几何与 `dot_a^(k)` 下
- 计算液相 bulk residual
- 计算界面 residual
- 计算气相 bulk residual
- 由 PETSc `SNES` 求解该固定几何下的非线性方程组

### timestep controller
负责：

- 记录本步尝试的 `dt`
- 接收 outer / inner 的成功与失败信息
- 决定 accept / reject
- 决定 rollback
- 决定 `dt -> dt/2` 还是 `dt -> min(1.1 dt, dt_max)`

---

## 4. backward Euler 与 step-level unknown evolution

### 4.1 时间离散

正式规定：  
`paper_v1` 的相内控制方程继续采用 backward Euler 离散，与主参考文献一致。

### 4.2 每个时间步的求解目标

在从 `t^n` 到 `t^{n+1}=t^n+dt` 的单步推进中，目标不是直接“得到一个新状态”，而是要同时满足：

1. 固定几何下的 `inner residual = 0`
2. 界面运动与 inner 解所给出的物理界面速度一致
3. 网格、状态恢复、物性计算均无失败
4. 该步满足 outer convergence criterion

因此，单步成功必须同时满足：

- `inner_converged == True`
- `outer_converged == True`
- `state_physical == True`

---

## 5. 文献 outer 主线在 `paper_v1` 中的正式实现

## 5.1 文献基线

文献明确说明：

1. 先根据新变量值计算界面位置；
2. 再按节点速度更新网格；
3. 在新网格上重新计算变量；
4. 重复该过程，直到连续两次界面速度的归一化差小于 \(10^{-5}\)。

## 5.2 `paper_v1` 的 outer 结构

每个时间步内，outer 采用显式循环，不使用第二个大型非线性求解器。  
正式流程为：

1. 用 predictor 给出第 0 轮 `a^(0)`、`dot_a^(0)`；
2. 对于第 `k` 轮 outer iteration：
   - 构建 `G^(k)`；
   - 计算 `v_c^(k)`；
   - 设置当前轮入口状态：`k=0` 时直接取 `U^n`，`k>0` 时取上一轮状态转移结果；
   - 调用 inner 求解固定几何下的新状态；
   - 若 inner fail，则当前 step attempt 进入局部补救或直接 reject；
   - 若 inner converged，则计算
     \[
     \dot a_{\text{phys}}^{(k)} = u_{l,\mathrm{if}}^{(k)} + \frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
     \]
   - 检查 outer convergence；
   - 若未收敛，则 corrector 更新 `a` 和 `dot_a`，进入下一轮 outer

## 5.3 outer 收敛判据

正式采用界面速度一致性判据：

\[
\varepsilon_{\dot a}^{(k)}
=
\frac{\left|\dot a_{\text{phys}}^{(k)}-\dot a^{(k)}\right|}
{\max\left(\left|\dot a_{\text{phys}}^{(k)}\right|,\varepsilon\right)}
\]

当

\[
\varepsilon_{\dot a}^{(k)} < 10^{-5}
\]

时，判定 outer 收敛。

## 5.4 outer 是否自动单调收敛

必须明确：

- outer 收敛判据只是一个“停止条件”
- 它不是一个会自动单调减小的量
- outer 可能单调收敛，也可能振荡、停滞或发散

因此，outer 必须有：

- 最大外层迭代数 `outer_max_iter`
- corrector 更新公式
- 欠松弛策略
- outer fail 后的 reject / retry 规则

## 5.5 outer 振荡/停滞的处理

首版正式规定：

- **不引入额外的振荡/停滞硬终止规则**
- outer 是否失败，只按：
  - `outer_max_iter`
  - `eps_dot_a < 1e-5`
  两项判定
- 若 outer 误差出现非单调变化或振荡，只作为 diagnostics 输出，不作为额外硬 fail 条件

---

## 6. 文献 inner 主线与 PETSc 替代边界

## 6.1 文献中的 inner 求解器

文献的 fixed-geometry 非线性系统求解器为 SSM（Stepped Shamanskii Method），并具有以下特征：

1. backward Euler 离散后形成总 residual vector；
2. 使用缩放后的 normalized residual 判断收敛；
3. 常用收敛容限为
   \[
   \|f_0\|_\infty < 10^{-8}
   \]
4. Jacobian 每 `p = 20` 次迭代更新一次；
5. 使用 Armijo damping；
6. 初始若 profile 较差，则前 `p` 次采用小阻尼 \( \lambda = 10^{-2} \)；
7. 通过截断避免非物理解。

## 6.2 `paper_v1` 中 PETSc 的职责

正式规定：

- PETSc `SNES` 替代文献中的 SSM；
- PETSc `KSP/PC` 替代文献中 banded LU 对 Jacobian 线性系统的求解；
- PETSc 的 line search / damping / convergence reason 用于 `inner`；
- PETSc 不直接管理 outer predictor-corrector；
- PETSc 不直接决定 step accept/reject。

## 6.3 inner 收敛判据

`paper_v1` 正式采用：

\[
\|R_{\text{inner}}\|_\infty < 10^{-8}
\]

其中 \(R_{\text{inner}}\) 为适当缩放后的 inner residual。

---

## 7. 文献时间步控制主线在 `paper_v1` 中的正式实现

## 7.1 总原则

文献不采用基于局部截断误差的实时双方法步长控制，而是通过“收敛难度”间接衡量刚性。  
因此 `paper_v1` 正式采用：

- **convergence-driven timestep control**
- 而不是标准 ODE-style LTE-driven adaptivity

## 7.2 基本思想

当系统变化缓慢、收敛容易时：

- 时间步可以适当增长

当系统变得刚性、收敛困难或失败时：

- 时间步必须减小

---

## 8. 时间步接受 / 拒绝规则

## 8.1 时间步被接受的必要条件

一个时间步从 `t^n` 到 `t^{n+1}` 被判定为 accepted，当且仅当同时满足：

1. inner 在某一轮 outer iteration 中收敛；
2. outer 在 `outer_max_iter` 内达到
   \[
   \varepsilon_{\dot a} < 10^{-5}
   \]
3. 若发生网格更新，则状态转移与状态恢复成功；
4. 物性、EOS、焓反解、界面平衡等计算无失败；
5. 新状态未出现不可接受的非物理值；
6. 形成了完整的新已接受状态 `state^{n+1}`。

## 8.2 时间步被拒绝的情况

当出现以下任一情况时，该步判定为 rejected：

1. `inner` 在允许的局部补救后仍失败；
2. `outer` 在 `outer_max_iter` 内不收敛；
3. remap 失败；
4. 状态恢复失败；
5. 物性调用失败；
6. 非物理状态不可恢复；
7. 新几何非法，例如 `a <= 0`。

---

## 9. failure taxonomy（项目级失败分类）

文献只笼统说“convergence criteria are not met”，但 `paper_v1` 必须更细化。  
正式将 step failure 拆分为以下几类：

## 9.1 inner fail
包括但不限于：

1. PETSc `SNES` 未在 `inner_max_iter` 内收敛；
2. line search / damping 失败；
3. Jacobian / linear solve 失败；
4. residual 变为 NaN / Inf；
5. 物性、界面方程、焓反解在 inner 过程中失败。

## 9.2 outer fail
指：

- inner 每轮都成功，但 outer 在 `outer_max_iter` 内始终不能达到界面速度收敛条件；
- 或 outer 达到最大允许轮数后仍未收敛。

## 9.3 remap / recovery fail
包括：

- conservative remap 后无法稳定恢复 `T / Y / h`
- 新暴露薄层初始化失败
- 恢复后出现不可纠正的不一致状态

## 9.4 nonphysical state fail
包括但不限于：

- 温度越界
- 质量分数越界且归一化后仍不合法
- 密度非正
- 半径非正
- EOS / UNIFAC / saturation / Cantera 调用所需状态非法

所有以上类别统一归入 **step-level failure**，由 timestep controller 负责处理。

---

## 10. PETSc failure reason 到 `paper_v1` failure class 的映射

## 10.1 正式原则

凡属于 `SNES` / `KSP` 非收敛、线搜索失败、函数/雅可比域错误、线性求解失败、最大迭代数耗尽等情况，统一映射为：

- `failure_class = inner_fail`

不允许把某些 PETSc divergence 仅当作 warning 而继续推进。

## 10.2 推荐映射范围

以下 PETSc reason 或同类情形，应统一归入 `inner_fail`：

- `SNES_DIVERGED_FUNCTION_DOMAIN`
- `SNES_DIVERGED_JACOBIAN_DOMAIN`
- `SNES_DIVERGED_LINE_SEARCH`
- `SNES_DIVERGED_LINEAR_SOLVE`
- `SNES_DIVERGED_MAX_IT`
- `SNES_DIVERGED_LOCAL_MIN`
- `SNES_DIVERGED_FNORM_NAN`
- `KSP_DIVERGED_*`
- 任意 Jacobian assembly fail
- 任意 preconditioner setup fail

### 说明
具体 reason 枚举以 PETSc 实际运行返回值为准，但**映射策略必须统一**：  
只要 inner nonlinear solve 没有成功形成可信的新状态，就归入 `inner_fail`。

---

## 11. 文献式时间步缩小策略

## 11.1 基线规则

文献明确采用如下回退主线：

- 若目标时间层不能收敛，
- 则返回到最近一个已收敛状态，
- 然后以更小时间步重新逼近，
- 默认缩小为 half-step。

因此 `paper_v1` 正式规定：

\[
dt_{\text{new}} = \frac{1}{2}dt
\]

作为默认、标准、首选的 reject 后缩步规则。

## 11.2 rollback 规则

时间步被 reject 后，必须：

1. 丢弃该步内产生的所有新状态；
2. 恢复到最近一个 accepted state：
   - `state^n`
   - `a^n`
   - `dot_a^n`
   - `grid^n`
3. 设置新的尝试步长
   \[
   dt \leftarrow \frac{1}{2}dt
   \]
4. 重新尝试从 `t^n` 推进。

## 11.3 不采用激进骤降作为主线

首版正式不采用：

- `dt *= 0.1`
- `dt *= 0.01`
- 其他更激进的默认骤降

过于激进的缩步虽然可能有助于寻找最终稳态，但会损害非定常解质量。  
`paper_v1` 首版遵循文献的保守 half-step 路线。

---

## 12. 文献式时间步增长策略

## 12.1 基线规则

文献规定：  
当之前 `q` 个时间步都成功收敛，且这期间时间步没有变化时，更新为

\[
dt_{\text{new}} = \min(1.1\,dt,\ dt_{\max})
\]

通常取 `q = 10`。

## 12.2 `paper_v1` 的正式采用

正式规定：

- `q_success_for_growth = 10`
- 仅当最近 10 个 accepted steps 中：
  1. 都成功；
  2. 且 `dt` 未变化；
- 才允许增长：

\[
dt \leftarrow \min(1.1\,dt,\ dt_{\max})
\]

## 12.3 解释

该策略是保守、缓慢增长策略，优先保证鲁棒性，而不是尽快把步长拉大。

---

## 13. step-level 失败后的局部补救与恢复顺序

为了避免一遇到失败就直接盲目缩步，`paper_v1` 允许有限次局部补救，但必须严格受限。

## 13.1 允许的局部补救

对于一次 step attempt 中出现的 inner fail，允许在当前 `dt` 下进行以下局部补救：

1. 强制重建 Jacobian；
2. 切换到更保守的 line search / damping；
3. 减小 outer corrector 的欠松弛强度；
4. 重做当前 outer 轮的 fixed-geometry inner solve。

## 13.2 局部补救次数上限

正式规定：

> **每个 step attempt 最多只允许 1 次局部补救。**

即：

- 第一次 inner fail：允许做 1 次局部补救
- 若补救后仍失败：直接 `step reject`
- 不允许在同一 `dt_try` 上进行第 2 次、第 3 次无限重试

## 13.3 主线原则

正式原则：

- **优先保证算法语义清晰**
- 不允许在一个注定失败的时间层上进行无休止的“再试一次”
- 局部补救是有限缓冲，不是第二套时间推进器

---

## 14. fatal stop policy（必须显式存在）

文献没有给出完整的软件级终止条件，因此 `paper_v1` 必须工程化补足。

## 14.1 正式要求

以下任一条件满足时，应触发 fatal stop：

1. 该步 reject 后若继续 half-step 会使
   \[
   dt < dt_{\min}
   \]
2. 同一已接受状态下的重试次数超过 etry_max_per_step`
3. 连续出现不可恢复的 property / remap / recovery failure
4. 连续出现非物理状态，且缩步后仍不能恢复

## 14.2 参数管理原则

以下参数必须在配置文件中**显式给出**，不允许使用隐式默认值，也不允许在代码内部私自兜底：

- `dt_start`
- `dt_max`
- `dt_min`
- etry_max_per_step`
- `inner_max_iter`
- `outer_max_iter`

### 特别规定
`paper_v1` 正式采用：

> **关键 timestep / iteration 参数必须由配置文件强制提供。**

原因：

- 避免“默认值覆盖配置值”这类极其烦人的实现错误
- 避免不同模块各自藏一套默认常量
- 保证数值实验可追溯、可复现

---

## 15. outer 与 inner 的关系：必须明确的若干事实

## 15.1 inner 收敛但 outer 不收敛是允许且常见的

正式说明：

- inner 收敛只意味着在当前冻结几何和当前 outer iterate 下，非线性场方程组被解出；
- 这不等于界面速度一致性已经满足；
- 因此完全可能出现
  - `inner_converged == True`
  - `outer_converged == False`

这不是异常，而是两层结构的正常情形。

## 15.2 outer 首版不使用第二个 PETSc 非线性求解器

首版正式不对 outer 再嵌套一个 `SNES`。  
outer 采用：

- 显式 predictor-corrector
- 自定义收敛判据
- 自定义欠松弛
- 自定义最大迭代数

原因是：

1. outer 的 residual 评估代价很高，每次都要调用 inner；
2. outer 同时涉及几何重建、remap、恢复，不是单纯的光滑代数函数；
3. 首版更重视可控性和可诊断性。

---

## 16. 单时间步正式流程

以下流程为 `paper_v1` 的正式单步推进骨架：

### 输入
- accepted state at time `t^n`
- `a^n`
- `dot_a^n`
- `dt`

### Step attempt
1. predictor:
   - 构造 `a^(0)` 与 `dot_a^(0)`
2. outer loop:
   - for `k = 0 ... outer_max_iter-1`
     1. build geometry `G^(k)`
     2. compute `v_c^(k)`
     3. set current inner entry state (`U^n` for `k=0`, transferred state for `k>0`)
     4. call PETSc SNES for inner solve
     5. if inner fails:
        - 若当前 step attempt 尚未使用局部补救，则执行 1 次局部补救
        - 否则 step reject
     6. compute `dot_a_phys^(k)`
     7. evaluate outer convergence
     8. if converged:
        - finalize state at `n+1`
        - accept step
        - exit outer
     9. else:
        - apply corrector / relaxation
        - continue next outer iteration
3. if outer exhausts max iterations:
   - step reject

### On reject
1. rollback to accepted state `n`
2. `dt <- dt/2`
3. retry same target interval

### On accept
1. store new accepted state
2. update `dt` using growth policy
3. proceed to next step

---

## 17. 必需诊断输出

每个 step attempt 必须至少记录以下诊断量：

### 17.1 step-level
- `step_id`
- `t_n`
- `dt_try`
- `accepted / rejected`
- eject_reason`
- etry_count_for_current_state`

### 17.2 outer-level
- `outer_iter_count`
- `eps_dot_a`
- `dot_a_guess`
- `dot_a_phys`
- `a_guess`
- `a_final`
- `outer_relaxation_used`

### 17.3 inner-level
- `inner_converged`
- `snes_reason`
- `ksp_reason`
- `inner_iter_count`
- `linear_iter_count`
- `jacobian_rebuild_count`
- `line_search_backtracks`
- `local_recovery_used`

### 17.4 failure-level
- `failure_class`
- `property_fail_flag`
- emap_fail_flag`
- ecovery_fail_flag`
- `nonphysical_flag`

### 17.5 dt-control
- `dt_old`
- `dt_new`
- `dt_change_reason`
  - `growth_after_q_success`
  - eject_half_step`
  - `hold_constant`

### 17.6 额外建议输出
为了后续诊断 outer 振荡，但不将其设为硬规则，建议额外输出：

- `eps_dot_a_prev`
- `eps_dot_a_change_sign_flag`
- `outer_nonmonotonic_flag`

---

## 18. 参数配置约束

以下参数本文件不提供默认值。  
它们必须在配置文件中显式给出：

1. `dt_start`
2. `dt_max`
3. `dt_min`
4. etry_max_per_step`
5. `inner_max_iter`
6. `outer_max_iter`
7. `outer_relaxation_initial`
8. `inner_local_recovery_max_per_attempt`
9. `q_success_for_growth`

### 正式规定
- `inner_local_recovery_max_per_attempt` 在首版必须配置为 `1`
- `q_success_for_growth` 在首版必须配置为 `10`

即便这两个值已经在文件中作为正式主线给出，代码仍不得偷偷提供内部默认值。  
实现上应检查配置是否存在；若缺失，则启动阶段直接报错。

---

## 19. 最终执行结论

本文件生效后，`paper_v1` 的时间步与失败处理正式遵循以下主线：

1. **outer predictor-corrector + outer convergence**  
   按文献主线执行；
2. **inner nonlinear solve**  
   由 PETSc `SNES` 执行，替代文献中的 SSM；
3. **timestep accept/reject**  
   由项目级 controller 执行；
4. **reject 后默认 half-step rollback**  
   按文献基线执行；
5. **连续 q=10 个稳定步后按 1.1 缓慢增步**  
   按文献基线执行；
6. **每个 step attempt 最多只允许 1 次局部补救**；
7. **inner、outer、step 必须分层判断**，不得混淆；
8. **关键 timestep / iteration 参数必须由配置文件显式给出**，不允许默认值藏在代码里。

---
