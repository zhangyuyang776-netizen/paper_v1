# state_recovery_and_enthalpy_inversion_guideline_final

> Sync note:
> state recovery ?? outer ????????????????
> outer ????
> `outer_inner_iteration_coupling_guideline_literature_aligned_v2.md`
> ???????

## 1. 文件目的

本文档用于为 `paper_v1` 正式规定以下内容：

1. 在 emap` / `conservative projection` / `newly exposed subvolume completion` 之后，如何从守恒量恢复物理状态；
2. 如何从
   - \(\rho V\)
   - \(\rho Y_i V\)
   - \(\rho h V\)
   恢复出
   - \(\rho\)
   - \(Y_i\)
   - \(h\)
   - \(T\)
3. 液相焓反解（enthalpy inversion）的正式算法；
4. 气相焓反解的正式算法；
5. 恢复失败的判据、允许的局部修正、以及必须终止恢复的情况；
6. recovery 模块与 emap`、`inner residual`、`diagnostics` 的职责边界。

本文档是 `paper_v1` 中**状态恢复与焓反解**的正式定稿文件。

---

## 2. 适用范围与角色边界

## 2.1 本文件解决的问题

本文件只解决以下问题：

> 已知某相某控制体上的守恒量内容或守恒量平均值，如何稳定恢复出该控制体的热力学状态。

典型输入来自：

- emap_and_conservative_projection_guideline_final.md`
- 界面相邻新暴露子体积守恒补全
- current-geometry transfer state 的构造过程

## 2.2 本文件不解决的问题

本文件不负责：

1. 守恒量如何从旧网格搬运到新网格  
   → 由 emap_and_conservative_projection_guideline_final.md` 负责
2. `T, Y` 已知时如何正向算物性  
   → 由液相物性模块 / Cantera 正向接口负责
3. bulk / interface residual 如何组装  
   → 由 `governing_equations_discretization_and_bc_guideline_final_v2.md` 负责
4. 时间步接受/拒绝  
   → 由 timestep/failure policy 负责

## 2.3 与 inner solve 的关系

在 `paper_v1` 中，焓反解的主要使用场景不是 inner 主未知量本身，而是：

- `remap` 后的 current-geometry transfer state
- newly exposed subvolume 守恒补全后的状态恢复
- 某些局部 post-correction 后的一致性恢复

因为 inner 主未知量已经显式包含：

- `T_l`, `T_g`
- `Y_l`, `Y_g`
- `T_s`, `Y_s,l`, `Y_s,g`

因此 inner residual 中通常是**正向物性计算**，不是焓反解。

---

## 3. 输入与输出

## 3.1 每相每控制体的输入守恒量

对相 `β ∈ {l, g}` 的某控制体，恢复模块的正式输入为：

- 控制体体积：
  \[
  V_\beta
  \]
- 总质量 content：
  \[
  (\rho_\beta V)
  \]
- 各 reduced species 的组分质量 content：
  \[
  ((\rho_\beta Y_{\beta,i})V)
  \]
- 总焓 content：
  \[
  ((\rho_\beta h_\beta)V)
  \]

其中：

- 液相首版单组分情形，液相组分恢复可退化；
- 气相必须始终保留 full composition 的恢复通道。

## 3.2 输出状态

恢复模块的正式输出为：

### 液相
- \(\rho_l\)
- \(Y_{l,\text{full}}\)
- \(h_l\)
- \(T_l\)

### 气相
- \(\rho_g\)
- \(Y_{g,\text{full}}\)
- \(X_{g,\text{full}}\)
- \(h_g\)
- \(T_g\)

并应进一步支持后续物性调用得到：

- \(c_p\)
- \(k\)
- \(D_i\)
- species enthalpies
- 其他需要的热物性量

---

## 4. 恢复总顺序

对任意相任意控制体，正式采用如下恢复顺序：

### Step 1
由总质量 content 恢复密度：

\[
\rho = \frac{\rho V}{V}
\]

### Step 2
由组分质量 content 恢复各组分 partial density：

\[
\rho Y_i = \frac{(\rho Y_i)V}{V}
\]

### Step 3
由 closure species 规则恢复 full partial density 集合

### Step 4
由 full partial density 恢复 full mass fractions：

\[
Y_i = \frac{\rho Y_i}{\rho}
\]

### Step 5
由总焓 content 恢复比焓：

\[
h = \frac{(\rho h)V}{\rho V}
\]

### Step 6
由 \(h, Y\) 反解温度 \(T\)

### Step 7
由恢复出的 \(T, Y\) 再正向计算一次：
- \(h(T,Y)\)
- \(c_p(T,Y)\)
- \(\rho(T,Y)\)（若该相正向模型会用到）
并执行一致性检查

### Step 8
若通过合法性检查，则形成 recovered state；  
若不通过，则根据失败规则处理。

---

## 5. 组分恢复

## 5.1 基本原则

正式原则：

> **先恢复 full partial density，再恢复 full mass fractions。**

不推荐直接先恢复 reduced \(Y_i\)，再由
\[
Y_{cl}=1-\sum Y_i
\]
“临时补齐”，因为那会掩盖 partial density 层面的守恒与越界问题。

---

## 5.2 reduced species 的 partial density 恢复

对每个 reduced species：

\[
(\rho Y_i) = \frac{((\rho Y_i)V)}{V}
\]

---

## 5.3 closure species 的 partial density 恢复

设 closure species 为 `cl`，则：

\[
(\rho Y_{cl}) = \rho - \sum_{i\in red}(\rho Y_i)
\]

该式是正式恢复公式，不允许以“默认 closure species”替代。

### 规定
- closure species 必须由配置文件显式给出
- 配置缺失应直接报错
- 不允许在 recovery 模块内自己猜 closure species

---

## 5.4 full mass fraction 恢复

对所有 full species：

\[
Y_i = \frac{\rho Y_i}{\rho}
\]

因此：

\[
Y_{cl} = \frac{\rho Y_{cl}}{\rho}
\]

---

## 5.5 单组分液相的退化情形

对单组分液滴首版液相：

\[
Y_{l,\text{full}} = [1.0]
\]

此时液相不需要一般性的组分恢复算法，但 recovery 模块仍应保留统一接口。

---

## 5.6 允许的小误差修正

若由于浮点误差导致：

- 某些 \(\rho Y_i\) 略微负值
- \(\sum_i \rho Y_i\) 与 \(\rho\) 的差仅为舍入级误差

允许执行**最小修正**。

### 正式修正步骤

1. 对绝对值小于 `species_recovery_eps_abs` 的负 partial density 置零；
2. 计算修正后的 partial density 总和：
   \[
   \Sigma_{\rho Y} = \sum_i \rho Y_i
   \]
3. 若
   \[
   |\Sigma_{\rho Y}-\rho|
   \]
   小于允许阈值，则对所有非零 partial density 做统一比例缩放，使：
   \[
   \sum_i \rho Y_i = \rho
   \]

### 说明
- 修正发生在 partial density 层面，不发生在 `Y_i` 层面
- 目标是保持总质量闭合、最小扰动、并保留 full composition

---

## 5.7 必须失败的情况

出现以下任一情况时，组分恢复直接失败：

1. \(\rho \le \rho_{min}\)
2. 多个 partial density 明显为负，且超过修正阈值
3. closure partial density 明显为负
4. 修正后仍无法满足
   \[
   \sum_i \rho Y_i \approx \rho
   \]
5. 恢复后的 full mass fractions 中存在明显超界：
   \[
   Y_i < -Y_{hard\_tol}
   \quad\text{或}\quad
   Y_i > 1+Y_{hard\_tol}
   \]

一旦发生上述情况，应返回：

- ecovery_fail = True`
- `failure_class = composition_recovery_fail`

---

## 6. 比焓恢复

## 6.1 正式公式

\[
h = \frac{(\rho h)V}{\rho V}
\]

即：

\[
h = \frac{\rho h}{\rho}
\]

### 前提
必须满足：

\[
\rho V > m_{min}
\]

且

\[
\rho > \rho_{min}
\]

否则直接恢复失败，不进入温度反解。

---

## 6.2 说明

本文件中 \(h\) 指的是相内混合物比焓：

- 液相：液相混合物比焓
- 气相：气相混合物比焓

不是某单一组分焓，也不是总能量。

---

## 7. 液相焓反解

## 7.1 正向液相焓模型的唯一来源

正式规定：

> 液相焓反解所使用的正向模型，必须完全引用 `liquid_properties_guideline_final.md` 中已经定稿的液相焓模型。

即：

### 纯组分液相显热焓
\[
\tilde h_{l,i}(T)=\int_{T_{ref}}^T c_{p,l,i}(\tau)\,d\tau
\]

### 混合液相焓
\[
h_l(T,Y)=\sum_i Y_i \tilde h_{l,i}(T)
\]

### 混合液相比热
\[
c_{p,l}(T,Y)=\sum_i Y_i c_{p,l,i}(T)
\]

恢复模块**不得**另写一套“简化液相焓公式”或“临时 cp 近似”。

---

## 7.2 液相焓反解目标方程

对已恢复的 liquid full composition `Y_l` 和目标比焓 `h_target`，定义：

\[
f_l(T)=h_l(T,Y_l)-h_{target}
\]

温度反解即求：

\[
f_l(T)=0
\]

---

## 7.3 导数

使用液相混合比热作为导数：

\[
f_l'(T)=c_{p,l}(T,Y_l)
\]

这要求：

- `cp_l(T,Y)` 与 `h_l(T,Y)` 完全一致
- 两者都来自 liquid properties 正向模型

---

## 7.4 正式算法

液相焓反解正式采用：

# **safeguarded Newton with bisection fallback**

### 算法步骤

1. 构造温度括区：
   \[
   [T_{min,l},\ T_{max,l}]
   \]
2. 取初值 \(T_0\)
3. 计算 \(f_l(T)\)、\(f_l'(T)\)
4. 优先尝试 Newton 步：
   \[
   T_{new}=T-\frac{f_l(T)}{f_l'(T)}
   \]
5. 若出现以下任一情况，则放弃 Newton，改为二分：
   - \(f_l'(T)\) 过小
   - \(T_{new}\) 超出括区
   - Newton 步导致数值异常
6. 用新的 \(T\) 更新括区
7. 直到满足收敛准则
8. 若超过最大迭代步数仍不收敛，则恢复失败

---

## 7.5 为什么不用纯 Newton

正式理由：

1. 液相物性未来允许多组分与温区限制
2. 纯 Newton 虽快，但对初值和局部斜率更敏感
3. 当前项目追求的是稳健首版，而不是最短实现
4. 有括区保护的 Newton 足够快，也足够稳

---

## 7.6 温度括区

### 正式规定
液相焓反解必须使用显式温度括区：

\[
T_{min,l} \le T \le T_{max,l}
\]

括区来源优先级如下：

1. 配置文件显式给出的 liquid recovery 温区
2. 液相物性数据库的共同有效温区
3. 若多组分，则取所有组分有效温区交集：

\[
T_{min,l}=\max_i(T_{min,l,i})
\]

\[
T_{max,l}=\min_i(T_{max,l,i})
\]

若括区为空，则直接报配置或物性数据库错误。

---

## 7.7 初值选择

正式采用如下优先级：

### 第一优先
使用上一时间层、上一 outer 轮、或 remap 前同一控制体的历史温度：

\[
T_0 = T_{prev}
\]

### 第二优先
使用线性估计：

\[
T_0 = T_{ref} + \frac{h_{target}-h_l(T_{ref},Y)}{\max(c_{p,l}(T_{ref},Y),c_{p,min})}
\]

再裁剪进括区：

\[
T_0 \leftarrow \min(\max(T_0,T_{min,l}),T_{max,l})
\]

---

## 7.8 收敛准则

液相焓反解同时检查：

### 焓残差
\[
|f_l(T)| \le \max(h_{abs\_tol},\ h_{rel\_tol}|h_{target}|)
\]

### 温度步长
\[
|T_{new}-T| \le T_{step\_tol}
\]

两者至少满足其一，且最终必须通过正反一致性检查。

---

## 7.9 失败条件

液相焓反解在以下情况判定失败：

1. 初始括区内没有根迹象
2. 超过 `liquid_h_inv_max_iter`
3. 导数持续异常且二分也无法收敛
4. 收敛后温度超出合法温区
5. 正向回代后：
   \[
   |h_l(T,Y)-h_{target}|
   \]
   仍超过容差

---

## 8. 气相焓反解

## 8.1 气相正向物性的唯一来源

正式规定：

> 气相的焓、温度、密度、物种摩尔分数及相关热物性，均以 Cantera 为唯一正向主引擎。

恢复模块不得另写第二套气相热力学模型。

---

## 8.2 正式主线：Cantera `HPY`

气相焓反解首选直接调用 Cantera 的等价状态设置：

\[
(T,\ P,\ Y)\ \longleftrightarrow\ (h,\ P,\ Y)
\]

即主线使用相当于：

- `gas.HPY = h_target, P_inf, Y_full`

然后读取：

- `T`
- ho`
- `X`
- `cp`
- `k`
- species enthalpies

---

## 8.3 fallback 主线

若 Cantera `HPY` 恢复失败，则正式 fallback 为：

### 定义目标函数
\[
f_g(T)=h_g(T,P_\infty,Y_g)-h_{target}
\]

其中 \(h_g(T,P_\infty,Y_g)\) 通过 Cantera 的正向 `TPY` 状态计算得到。

### 算法
同液相一样，采用：

# **safeguarded Newton with bisection fallback**

---

## 8.4 气相温度括区

正式规定：

\[
T_{min,g} \le T \le T_{max,g}
\]

括区由配置文件显式提供。  
恢复模块不得把温度上下界写死在代码里。

---

## 8.5 full gas composition 的正式要求

气相恢复后必须得到：

- `Yg_full`
- `Xg_full`

这是正式硬要求。  
因为后续需要：

- mixture-averaged corrected diffusion
- conservative flux correction
- generalized Fourier’s law
- interface face package

都依赖 full gas composition。

只恢复 reduced gas state 不算恢复成功。

---

## 8.6 气相失败条件

以下情况之一，判定为 gas recovery fail：

1. `HPY` 失败且 fallback 也失败
2. 得到的温度超出合法范围
3. full composition 非法
4. 正向回代后
   \[
   |h_g(T,P,Y)-h_{target}|
   \]
   超过容差
5. 得到的密度或其他必需状态量非法

---

## 9. 恢复后的统一合法性检查

对液相和气相，恢复结束后都必须执行统一 post-check。

---

## 9.1 密度检查

\[
\rho > \rho_{min}
\]

若不满足，直接失败。

---

## 9.2 质量分数检查

对所有组分：

\[
0 \le Y_i \le 1
\]

同时要求：

\[
\left|\sum_i Y_i - 1\right| \le Y_{sum\_tol}
\]

若只存在舍入级偏差，允许按第 5 节规则做轻微修正。  
若偏差明显，则失败。

---

## 9.3 温度检查

\[
T_{min,\beta} \le T_\beta \le T_{max,\beta}
\]

对 `β=l,g` 都必须检查。

---

## 9.4 正反一致性检查

恢复出 \(T, Y\) 之后，必须再次正向计算：

### 液相
\[
h_{l,recomputed}=h_l(T_l,Y_l)
\]

### 气相
\[
h_{g,recomputed}=h_g(T_g,P_\infty,Y_g)
\]

并检查：

\[
|h_{recomputed}-h_{target}| \le h_{check\_tol}
\]

---

## 9.5 full gas consistency 检查

对气相，还必须额外检查：

- \(X_i\) 已成功恢复
- \(X_i\) 与 \(Y_i\) 相互一致
- 后续 transport 需要的 species enthalpies 可正常获取

若其中任一失败，则恢复失败。

---

## 10. 与 remap / conservative projection 的接口关系

## 10.1 输入来源

恢复模块接收 emap` 模块或守恒补全模块输出的：

- \((\rho V)\)
- \(((\rho Y_i)V)\)
- \((\rho hV)\)

这些量已经包含：

- 普通 overlap remap 贡献
- newly exposed subvolume completion 贡献（若有）

## 10.2 正式顺序

正式顺序固定为：

1. overlap remap
2. newly exposed subvolume conservative completion
3. 合并成新控制体守恒量
4. 再进入状态恢复

恢复模块不应区分：
- 哪部分来自 overlap
- 哪部分来自界面新暴露子体积补全

它只对最终守恒量负责。

---

## 10.3 输出去向

恢复后的状态共同构成：

\[
old\_state\_on\_current\_geometry
\]

并供以下模块使用：

- inner residual 的时间项
- continuity-based velocity recovery
- 物性 consistency diagnostics
- 守恒检查与日志输出

---

## 11. minor fix 与 hard fail 的边界

## 11.1 允许 minor fix 的对象

正式只允许对以下对象进行微小修正：

- partial density / mass fraction 的舍入级误差
- sum(Y) 的舍入级偏差

## 11.2 不允许 minor fix 的对象

以下情况不得通过“裁剪”或“强行修补”伪装成成功：

1. 温度反解失败
2. 温度明显越界
3. 密度非正
4. 焓正反不一致
5. Cantera 状态设置失败后仍无可用 fallback

这些都必须直接判为 ecovery_fail`。

---

## 12. diagnostics 与 failure flags

每次 recovery 至少输出以下诊断量：

### 液相
- `liq_recovery_success`
- `liq_recovery_num_minor_fixes`
- `liq_enthalpy_inversion_iters`
- `liq_enthalpy_residual`
- `liq_recomputed_enthalpy_err`
- `liq_recovered_T_min`
- `liq_recovered_T_max`

### 气相
- `gas_recovery_success`
- `gas_recovery_used_HPY`
- `gas_recovery_used_fallback`
- `gas_enthalpy_inversion_iters`
- `gas_enthalpy_residual`
- `gas_recomputed_enthalpy_err`
- `gas_recovered_T_min`
- `gas_recovered_T_max`

### 全局
- ecovery_fail_count`
- ecovery_failure_class_max`
- `max_Y_sum_err`
- `max_negative_partial_density`

---

## 13. 配置参数

以下参数必须在配置文件中显式存在，不允许隐藏默认值：

### 通用
- ho_min`
- `m_min`
- `species_recovery_eps_abs`
- `Y_sum_tol`
- `Y_hard_tol`
- `h_abs_tol`
- `h_rel_tol`
- `h_check_tol`
- `T_step_tol`

### 液相
- `T_min_l`
- `T_max_l`
- `liquid_h_inv_max_iter`
- `cp_min`

### 气相
- `T_min_g`
- `T_max_g`
- `gas_h_inv_max_iter`
- `use_cantera_hpy_first` = true

---

## 14. 正式禁止事项

以下做法在本文件下明确禁止：

1. recovery 模块另写一套液相焓正向模型
2. 用 reduced species 直接替代 full composition 做 gas recovery
3. 对明显错误的温度/焓状态做强行裁剪后继续求解
4. 把气相 `HPY` 失败静默吞掉，不记录 fallback
5. 在 recovery 中默认猜 closure species
6. 跳过正反一致性检查
7. 用邻近单元温度直接代替焓反解结果
8. 用“温度插值”替代 ho, rhoY, rhoh -> T` 的正式恢复
9. 让 recovery 模块偷偷修改 remap 的守恒量输入
10. 对恢复失败只打印 warning 而不向 step-level failure 传递错误状态

---

## 15. 最终方法合同

### State Recovery and Enthalpy Inversion Contract

1. The recovery module reconstructs physical state from conservative cell contents, not from primitive-variable interpolation.
2. Recovery proceeds in the order:
   - \(\rho\)
   - full partial densities
   - full mass fractions
   - \(h\)
   - \(T\)
3. Closure species is recovered from density closure, not guessed.
4. Liquid enthalpy inversion must use the forward liquid enthalpy model defined in `liquid_properties_guideline_final.md`.
5. Liquid temperature recovery uses safeguarded Newton with bisection fallback.
6. Gas temperature recovery uses Cantera `HPY` first, then a scalar root-solve fallback if needed.
7. Only small composition errors may be minimally corrected.
8. Temperature, density, and enthalpy inconsistency are hard-fail conditions.
9. Recovery outputs the full state required by transport, flux, and diagnostics modules.
10. Recovery failure is a formal step-level failure source and must not be silently ignored.

---

## 16. 最终结论

`paper_v1` 的状态恢复与焓反解正式采用如下主线：

- 输入是守恒量内容：
  \[
  \rho V,\ \rho Y_i V,\ \rho h V
  \]
- 先恢复 full partial density，再恢复 full composition
- 由
  \[
  h=\frac{\rho hV}{\rho V}
  \]
  恢复比焓
- 液相用与 `liquid_properties_guideline_final.md` 完全一致的正向焓模型做反解
- 液相温度反解使用 safeguarded Newton + bisection
- 气相温度恢复首选 Cantera `HPY`，失败则使用标量根求解 fallback
- 恢复后必须做合法性检查与正反一致性检查
- 只有组分的舍入级误差允许做最小修正
- 焓/温度/密度问题一律视为 hard fail
- 恢复结果构成 current-geometry transfer state，供下一轮 inner residual 与 diagnostics 使用；若 `k=0`，则该链路必须退化为 identity

这就是 `paper_v1` 的状态恢复与焓反解正式定稿版本。
