# liquid_properties_guideline_final

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定**液相物性计算方法**，并明确：

1. 单组分液相物性计算公式  
2. 多组分液体混合物性计算公式  
3. 界面相平衡式和界面能量方程所需的液相参数  
4. 文献中未直接展开的热力学公式应如何补全  
5. 液相物性参数文件中必须提供哪些参数  
6. 物性参数文件的推荐模板

本文档是 `paper_v1` 中液相物性计算的正式定稿版本。

---

## 2. 总体原则

### 2.1 主线选择

`paper_v1` 的液相物性主线正式定为：

- **纯组分液相物性**：由文献相关式、标准热力学参数化或实验数据拟合得到
- **混合液相物性**：按文献第二章给出的混合规则计算
- **液相焓**：由液相比热容积分得到
- **液相潜热**：由参考温度下的汽化潜热加上气液显热差得到
- **界面活度系数**：采用 UNIFAC 主线
- **不采用 Cantera 作为液相主物性引擎**

### 2.2 统一参考温度

固定参考温度为：

\[
T_{\mathrm{ref}} = 298.15\ \mathrm{K}
\]

该参考温度与主论文的符号表一致。液相焓、气相焓和汽化潜热的定义都必须使用同一参考温度。

### 2.3 纯组分与混合物的分层

每种液相物性都必须分为两层：

1. **纯组分物性模型**
2. **混合物混合规则**

不允许跳过纯组分层，直接为多组分混合物拼经验公式。

---

## 3. 纯组分液相物性

## 3.1 分子量

每个液相物种必须提供分子量：

\[
W_{l,i}
\]

单位建议统一为：

- `kg/mol` 作为程序内部单位
- 参数文件主存储单位也统一为 `kg/mol`
- 若某个经验式内部需要 `g/mol`，必须在公式实现处显式转换，不允许把数据库主单位切成 `g/mol`

### 用途
- Wilke-Chang 扩散系数
- 液相/气相混合分子量
- 界面相平衡式 2.19
- 组分比气体常数

---

## 3.2 摩尔体积

每个液相物种必须提供液相摩尔体积：

\[
V_{l,i}
\]

文献中的 Wilke-Chang 公式直接需要该量。

### 单位建议
- `cm^3/mol` 作为参数文件单位
- 程序内部保持 `cm^3/mol` 也可以，但若某个经验式需要额外尺度变换，必须在该经验式实现处显式写出

### 用途
- 液相扩散系数

---

## 3.3 association factor

Wilke-Chang 公式还需要缔合因子：

\[
\phi_i
\]

论文中对典型物种给出的值为：

- 水：\(\phi = 2.6\)
- 乙醇：\(\phi = 1.5\) 

### 规定
- 对所有液相物种，参数文件中必须显式给出 `association_factor`
- 不允许代码默认填值
- 若文献没有值，应在数据整理阶段给出来源或拟定取值依据

### 用途
- 液相扩散系数

---

## 3.4 纯组分液相密度

### 3.4.1 文献主线

论文给出的纯组分液相密度拟合式为：

\[
\log \rho_{l,i}
=
A_{\rho,i}\log T
+
\frac{B_{\rho,i}}{T}
+
\frac{C_{\rho,i}}{T^2}
+
D_{\rho,i}
+
E_{\rho,i}T
+
F_{\rho,i}T^2
\]

其中：

- \(\rho_{l,i}\) 单位为 `kg/m^3`
- \(T\) 单位为 `K`  

### 3.4.2 所需参数
- `A_rho`
- `B_rho`
- `C_rho`
- `D_rho`
- `E_rho`
- `F_rho`

### 3.4.3 用途
- 单组分液体密度
- 混合液相密度
- 界面液相密度 \(\rho_{s,l}\)

---

## 3.5 纯组分液相比热容

### 3.5.1 正式主线：Shomate

由于论文没有在正文中给出单独的液相比热容显式拟合式，但明确说纯组分 \(c_p\) 和 \(h\) 使用 NASA polynomials，缺失时再由实验数据拟合，因此 `paper_v1` 对液相纯组分热容采用更适合数据库维护的**Shomate** 主线。

定义：

\[
t = \frac{T}{1000}
\]

\[
c_{p,l,i}^{\mathrm{molar}}(T)
=
A_i + B_i t + C_i t^2 + D_i t^3 + \frac{E_i}{t^2}
\]

其中：

- \(c_p\) 单位：`J/mol/K`
- \(T\) 单位：`K`  
这是 NIST 使用的标准 Shomate 形式。

### 3.5.2 质量基比热容

程序内部若采用质量基，应转换为：

\[
c_{p,l,i}(T)
=
\frac{c_{p,l,i}^{\mathrm{molar}}(T)}{W_{l,i}}
\]

其中 `W_l,i` 用 `kg/mol`。

### 3.5.3 所需参数
- `cp_model = shomate`
- `A_cp`
- `B_cp`
- `C_cp`
- `D_cp`
- `E_cp`
- `F_cp`
- `G_cp`
- `H_cp`
- `T_min`
- `T_max`

### 说明
虽然焓只直接用到 `A-E,F,H`，但若你以后还要扩展熵或与标准数据库比对，保留完整 Shomate 参数最稳。

---

## 3.6 纯组分液相焓

### 3.6.1 采用比热容积分定义显热焓

对每个液相物种，定义**相对于参考温度的显热焓**：

\[
\tilde h_{l,i}(T)
=
\int_{T_{\mathrm{ref}}}^{T} c_{p,l,i}(\tau)\,d\tau
\]

这意味着：

\[
\tilde h_{l,i}(T_{\mathrm{ref}})=0
\]

### 3.6.2 若使用 Shomate，则解析式为

NIST 给出的 Shomate 焓函数是：

\[
H^\circ(T)-H^\circ(298.15)
=
A_i t
+
\frac{B_i}{2} t^2
+
\frac{C_i}{3} t^3
+
\frac{D_i}{4} t^4
-
\frac{E_i}{t}
+
F_i
-
H_i
\]

其单位为 `kJ/mol`。  
因此可定义液相纯组分显热焓：

\[
\tilde h_{l,i}^{\mathrm{molar}}(T)
=
1000
\left(
A_i t
+
\frac{B_i}{2} t^2
+
\frac{C_i}{3} t^3
+
\frac{D_i}{4} t^4
-
\frac{E_i}{t}
+
F_i
-
H_i
\right)
\]

单位 `J/mol`，再除以分子量转成质量基：

\[
\tilde h_{l,i}(T)
=
\frac{\tilde h_{l,i}^{\mathrm{molar}}(T)}{W_{l,i}}
\]

### 3.6.3 允许的替代方案

若某液相物种没有可靠 Shomate 参数，但有高质量 \(c_p(T)\) 数据点，则允许采用**分段低阶多项式拟合**：

\[
c_{p,l,i}(T)=a_0+a_1\theta+a_2\theta^2+a_3\theta^3
\]

其中：

\[
\theta = \frac{T-T_{\mathrm{ref}}}{100}
\]

然后解析积分得到 \(\tilde h_{l,i}(T)\)。

### 3.6.4 正式限制
- 不允许默认使用一个全温区高阶多项式主线
- 不允许液相焓与气相焓使用不同参考温度
- 不允许把“液相焓是否有参考项”散落在各模块里自由发挥

---

## 3.7 纯组分液相热导率

### 3.7.1 文献主线

论文给出的纯组分液相热导率拟合式为：

\[
\log k_{l,i}
=
A_{k,i}\log T
+
\frac{B_{k,i}}{T}
+
\frac{C_{k,i}}{T^2}
+
D_{k,i}
+
E_{k,i}T
+
F_{k,i}T^2
\]

其中：
- \(k_{l,i}\) 单位通常为 `W/m/K`
- \(T\) 单位为 `K` 

### 3.7.2 所需参数
- `A_k`
- `B_k`
- `C_k`
- `D_k`
- `E_k`
- `F_k`

### 3.7.3 用途
- 单组分液体热导率
- 混合液相热导率
- 界面液相导热系数

---

## 3.8 纯组分液相黏度

### 3.8.1 文献主线

论文说明：纯组分液相黏度使用**与 Eq.(2.11) 类似的表达式**。

因此，正式规定为：

\[
\log \mu_{l,i}
=
A_{\mu,i}\log T
+
\frac{B_{\mu,i}}{T}
+
\frac{C_{\mu,i}}{T^2}
+
D_{\mu,i}
+
E_{\mu,i}T
+
F_{\mu,i}T^2
\]

### 3.8.2 所需参数
- `A_mu`
- `B_mu`
- `C_mu`
- `D_mu`
- `E_mu`
- `F_mu`

### 3.8.3 用途
- 单组分液体黏度
- 混合液相黏度
- Wilke-Chang 扩散系数
- 界面液相黏度

---

## 3.9 纯组分液相沸点

界面平衡式 2.19 需要每个可凝液相物种的常压沸点：

\[
T_{b,i}
\]

文献明确写出 `Tb,i is the boiling temperature at atmospheric pressure`。

### 所需参数
- `boiling_temperature_atm`

### 用途
- 界面相平衡式 2.19
- 一些近沸点检查与物理诊断
- 可选的潜热模型校验

---

## 3.10 纯组分液相的组分比气体常数

界面平衡式 2.19 还需要组分比气体常数：

\[
R_{g,i}
\]

它不是需要单独查表的独立经验参数，而应**由通用气体常数和分子量计算**：

\[
R_{g,i}=\frac{R_u}{W_i}
\]

其中：
- \(R_u = 8.314462618\ \mathrm{J/mol/K}\)
- \(W_i\) 使用 `kg/mol`
- 则 \(R_{g,i}\) 单位为 `J/kg/K`

### 规定
- `Rg_i` 不作为机理文件独立输入参数
- 由 `molecular_weight` 自动派生
- 但文档中必须明确其存在和单位

---

## 3.11 纯组分液相活度系数相关数据

界面平衡式 2.19 中的 \(\gamma_i\) 不是单个“每物种一个数字”的参数，而是 **UNIFAC 模型输出**。论文明确说明 \(\gamma_i\) 用 UNIFAC 获得，并在附录 2.A 给出组参数与相互作用参数。 

### 这意味着机理/参数文件里必须提供两层信息

#### A. 每个液相物种的 UNIFAC 基团分解
例如每个物种需要给出：
- `CH3`: 个数
- `CH2`: 个数
- `OH`: 个数
- `H2O`: 个数
- ...

#### B. 全局 UNIFAC 基团参数
至少包括：
- 每个 group 的 `R_k`
- 每个 group 的 `Q_k`
- group interaction matrix `a_mn`

### 规定
- `activity_coefficient` **不能**在参数文件中只留一个占位标量
- 必须明确 `activity_model = UNIFAC`
- 并提供 UNIFAC 所需的 group 数据和 interaction 数据
- 如果首版单组分液滴可取 \(\gamma=1\)，也必须在文件中显式写出 `activity_model: ideal`，不能靠代码默认

---

## 4. 混合液相物性

## 4.1 混合液相密度

文献公式：

\[
\rho_l
=
\left(
\sum_{i=1}^{N_l} X_{l,i}\rho_{l,i}^{1/2}
\right)^2
\]

其中：
- `X_l,i` 为液相摩尔分数
- `rho_l,i` 为纯组分液相密度 

### 所需输入
- `X_l,i`
- `rho_l,i(T)`

---

## 4.2 混合液相比热容

文献对两相都采用理想混合：

\[
c_{p,l} = \sum_{i=1}^{N_l} Y_{l,i} c_{p,l,i}
\]



### 所需输入
- `Y_l,i`
- `cp_l,i(T)`

---

## 4.3 混合液相焓

混合液相焓采用质量分数加权：

\[
h_l = \sum_{i=1}^{N_l} Y_{l,i}\tilde h_{l,i}(T)
\]

这里 \( \tilde h_{l,i}(T) \) 是相对于 \(T_{ref}\) 的显热焓。

### 所需输入
- `Y_l,i`
- `h_l,i(T)` 或 `cp_l,i(T)` 积分结果

---

## 4.4 混合液相热导率

文献采用 Filippov 广义式：

\[
k_l
=
\sum_{i=1}^{N_l}
Y_{l,i}
\left(
k_{l,i}
-
\sum_{j=i+1}^{N_l}
K_{i,j}Y_{l,j}|k_{l,i}-k_{l,j}|
\right)
\]

其中：

\[
K_{i,j}=0.72
\]



### 所需输入
- `Y_l,i`
- `k_l,i(T)`

---

## 4.5 混合液相黏度

文献采用 Grunberg-Nissan 方程：

\[
\mu_l
=
\exp\left(
\sum_{i=1}^{N_l} X_{l,i}\ln\mu_{l,i}
\right)
\]



### 所需输入
- `X_l,i`
- `mu_l,i(T)`

---

## 4.6 混合液相扩散系数

文献直接给的是液相组分 \(i\) 在混合液体中的 Wilke-Chang 型扩散系数：

\[
D_{l,i}
=
1.173\times10^{-16}
\frac{
\sqrt{
\sum_{j\neq i} X_{l,j}\phi_j W_j
}\,T
}{
\mu_l\,(V_{l,i}/1000)^{0.6}
}
\]

其中：
- `X_l,j`：液相摩尔分数
- `phi_j`：缔合因子
- `W_j`：分子量；数据库主单位为 `kg/mol`，但在该经验式内部必须显式转换为 `g/mol`
- `mu_l`：混合液相黏度
- `V_l,i`：组分摩尔体积；参数文件主单位为 `cm^3/mol`，当前实现按 `(V_l,i/1000)^{0.6}` 进入分母

### 所需输入
- `X_l,j`
- `phi_j`
- `W_j`
- `T`
- `mu_l`
- `V_l,i`

### 说明
- 这条式子已经是“混合液体中的组分扩散系数”，不是先算纯组分再混合
- 温度 `T` 在平方根外；不允许写成 `\sqrt{\sum X_{l,j}\phi_j W_j T}`
- 单组分液滴阶段可不启用液相组分方程，但参数结构应保留

---

## 5. 界面相关液相参数

## 5.1 界面液相密度

\[
\rho_{s,l} = \rho_l(T_s, Y_{s,l})
\]

由液相密度主线派生。

## 5.2 界面液相比热与焓

\[
c_{p,s,l} = \sum_i Y_{s,l,i} c_{p,l,i}(T_s)
\]

\[
h_{s,l} = \sum_i Y_{s,l,i} \tilde h_{l,i}(T_s)
\]

## 5.3 界面液相热导率与黏度

\[
k_{s,l} = k_l(T_s, Y_{s,l})
\]

\[
\mu_{s,l} = \mu_l(T_s, Y_{s,l})
\]

## 5.4 界面相平衡式 2.19 所需的液相参数

界面平衡式需要以下液相侧参数：

- `Y_s,l,i`
- `W_l`
- `Tb_i`
- `R_g,i = R_u/W_i`
- `gamma_i`

其中：
- `W_l` 为液相混合分子量
- `Tb_i` 对应物种级 `boiling_temperature_atm`，不是 pressure bank 中随 `p_fit` 变化的 `boiling_temperature`
- `gamma_i` 来自 UNIFAC，不是单一常数 

---

## 6. 汽化潜热

这是本文件最关键的一条。

## 6.1 正式主线公式

论文给出：

\[
L_i(T)=h_{g,i}(T)-h_{l,i}(T)+L_i^{ref}
\]

其中：
- \(L_i^{ref}\)：参考温度下的汽化潜热
- \(T_{ref}=298.15\ \mathrm{K}\)  

为避免焓基准混乱，`paper_v1` 正式采用**显热焓写法**：

\[
L_i(T)
=
L_i^{ref}
+
\tilde h_{g,i}(T)
-
\tilde h_{l,i}(T)
\]

其中：
- \(\tilde h_{g,i}(T)\)：相对于 \(T_{ref}\) 的气相显热焓
- \(\tilde h_{l,i}(T)\)：相对于 \(T_{ref}\) 的液相显热焓

## 6.2 这条公式的意义

它等价于说：

> 当前温度的汽化潜热  
> = 参考温度下的汽化潜热  
> + 从参考温度升到当前温度时，气液两相显热增量的差

## 6.3 对气相焓的要求

为了让这条公式成立，气相组分焓必须也转换到**同一参考温度基准**：

\[
\tilde h_{g,i}(T)
=
h_{g,i}(T)-h_{g,i}(T_{ref})
\]

### 规定
- 即使气相主物性由 Cantera 提供，进入潜热公式时也必须转换到相同参考焓基准
- 不允许直接把 Cantera 的某个“绝对标准焓”不经处理地和液相显热焓混用

## 6.4 参考潜热 `L_ref`

每个液相可凝物种都必须在参数文件中显式提供：

\[
L_i^{ref}
\]

单位建议为 `J/kg`。

### 用途
- 界面能量方程 2.17
- 界面平衡式 2.19 中的积分项
- 单组分基线验证

---

## 7. 还遗漏了什么参数？

你列的 8 项已经很接近完整了，但还缺下面这些必须正式加入：

### 7.1 纯组分液相比热容的温度区间
每个 `cp_model` 都必须提供：
- `T_min`
- `T_max`

否则后面不会知道拟合有效范围。

### 7.2 混合分子量
界面式 2.19 里不仅要纯组分 `W_i`，还要：
- `W_l`
- `W_g`

因此程序中必须显式定义液相混合分子量：

\[
W_l = \left(\sum_i \frac{Y_{l,i}}{W_i}\right)^{-1}
\]

### 7.3 活度模型类型
不能只写“有 activity coefficient”，还必须写：
- `activity_model = ideal` 或 `UNIFAC`

### 7.4 UNIFAC group 数据
若采用 UNIFAC，则还必须有：
- 每物种 group counts
- 全局 group R/Q
- 全局 interaction matrix

这部分不能藏在代码里当魔法常量。

### 7.5 临界温度（可选但强烈建议）
如果未来你要用 Watson 型潜热模型或做潜热校验，则应提供：
- `Tc`

你当前代码里也已经有 Watson 型潜热模型雏形，要求 `Tc, Tref, Hvap_Tref, exponent`。

---

## 8. 液相物性参数文件必须提供的字段

对每个液相物种，至少需要：

### 基本信息
- `name`
- `molecular_weight`
- `boiling_temperature_atm`
- `molar_volume`
- `association_factor`
- `Tc`（若后续采用 Watson 等潜热模型）

### 压力 bank
- `pressure_banks`
- 每个 bank 至少包含：
  - `p_fit`
  - `boiling_temperature`
  - `T_ref`
  - `cp_model`
  - `cp_coeffs`
  - `cp_T_range`
  - `hvap_ref`
  - `rho_model`
  - `rho_coeffs`
  - `k_model`
  - `k_coeffs`
  - `mu_model`
  - `mu_coeffs`

### 压力 bank 可选增强字段
- `hvap_model`
- `hvap_coeffs`

### 相平衡
- `activity_model`
- `unifac_groups`（若非 ideal）

---

## 9. 参数文件模板

下面给出推荐模板。  
注意，这不是“反应机理文件”，更准确地说是**液相物性数据库文件**。

```yaml
meta:
  file_type: liquid_properties_db
  version: 1
  units:
    temperature: K
    pressure: Pa
    molecular_weight: kg/mol
    molar_volume: cm^3/mol
    density: kg/m^3
    cp_mass: J/kg/K
    cp_molar: J/mol/K
    enthalpy_mass: J/kg
    thermal_conductivity: W/m/K
    viscosity: Pa*s
    hvap: J/kg
  reference:
    T_ref: 298.15

global_models:
  liquid_cp_default_model: shomate
  liquid_density_default_model: merino_log_poly
  liquid_conductivity_default_model: merino_log_poly
  liquid_viscosity_default_model: merino_log_poly
  liquid_diffusion_model: wilke_chang
  liquid_mixture_density_model: merino_x_sqrt_rho
  liquid_mixture_conductivity_model: filippov
  liquid_mixture_viscosity_model: grunberg_nissan
  activity_model_default: UNIFAC

unifac:
  enabled: true
  groups:
    # 占位：每个 group 的 R/Q
    CH3:
      R: <R_CH3>
      Q: <Q_CH3>
    CH2:
      R: <R_CH2>
      Q: <Q_CH2>
    OH:
      R: <R_OH>
      Q: <Q_OH>
    H2O:
      R: <R_H2O>
      Q: <Q_H2O>
  interactions:
    # 占位：a_mn 矩阵，单位按你最终实现统一
    CH3:
      CH3: 0.0
      CH2: <a_CH3_CH2>
      OH: <a_CH3_OH>
      H2O: <a_CH3_H2O>
    CH2:
      CH3: <a_CH2_CH3>
      CH2: 0.0
      OH: <a_CH2_OH>
      H2O: <a_CH2_H2O>
    OH:
      CH3: <a_OH_CH3>
      CH2: <a_OH_CH2>
      OH: 0.0
      H2O: <a_OH_H2O>
    H2O:
      CH3: <a_H2O_CH3>
      CH2: <a_H2O_CH2>
      OH: <a_H2O_OH>
      H2O: 0.0

species:
  - name: <LIQUID_SPECIES_NAME>
    aliases: [<optional_alias_1>, <optional_alias_2>]

    # --------------------------
    # 基本参数
    # --------------------------
    molecular_weight: <W_i_kg_per_mol>
    boiling_temperature_atm: <Tb_i_at_101325Pa_K>
    molar_volume: <V_i_cm3_per_mol>
    association_factor: <phi_i>
    Tc: <critical_temperature_K>

    # --------------------------
    # 活度模型
    # --------------------------
    activity_model: UNIFAC
    unifac_groups:
      CH3: <count_CH3>
      CH2: <count_CH2>
      OH: <count_OH>
      H2O: <count_H2O>

    # --------------------------
    # 压力 bank
    # --------------------------
    pressure_banks:
      - p_fit: <fit_pressure_Pa>
        boiling_temperature: <Tb_i_at_p_fit_K>
        T_ref: 298.15
        cp_model: shomate
        cp_T_range: [<Tmin_K>, <Tmax_K>]
        cp_coeffs:
          A: <A_cp>
          B: <B_cp>
          C: <C_cp>
          D: <D_cp>
          E: <E_cp>
          F: <F_cp>
          G: <G_cp>
          H: <H_cp>

        hvap_ref: <L_ref_J_per_kg>
        hvap_model: <none_or_watson>
        hvap_coeffs:
          exponent: <n_if_needed>

        rho_model: merino_log_poly
        rho_coeffs:
          A: <A_rho>
          B: <B_rho>
          C: <C_rho>
          D: <D_rho>
          E: <E_rho>
          F: <F_rho>

        k_model: merino_log_poly
        k_coeffs:
          A: <A_k>
          B: <B_k>
          C: <C_k>
          D: <D_k>
          E: <E_k>
          F: <F_k>

        mu_model: merino_log_poly
        mu_coeffs:
          A: <A_mu>
          B: <B_mu>
          C: <C_mu>
          D: <D_mu>
          E: <E_mu>
          F: <F_mu>
````

### 模板说明

1. `global_models` 只定义默认模型，不负责给出物种参数本身  
2. `species` 中的参数是液相物性数据库主体  
3. pressure-sensitive 的拟合参数统一下沉到 `pressure_banks`  
4. 运行时按环境压力 `p_env` 从每个物种的 `pressure_banks` 中选择最近 bank，推荐准则为 `|\ln(p_env/p_fit)|` 最小  
5. 界面相平衡式 2.19 使用物种级 `boiling_temperature_atm`，而不是 selected bank 的 `boiling_temperature`  
6. `cp/h/rho/k/mu` 采用“纯组分模型 + 混合规则”的二层结构  

---

## 10. 每个物性需要哪些参数，占位总结

### 10.1 扩散系数 (D_{l,i})

需要：

* `association_factor`
* `molar_volume`
* `molecular_weight`（对所有参与混合的液相物种）
* 混合液相黏度所需参数（因为公式里要 `mu_l`）

### 10.2 密度 (\rho_{l,i})

需要：

* `rho_coeffs: A-F`

### 10.3 比热容 (c_{p,l,i})

需要：

* `cp_model`
* `cp_coeffs`
* `cp_T_range`

### 10.4 焓 (\tilde h_{l,i})

需要：

* `cp_model`
* `cp_coeffs`
* `T_ref`

### 10.5 热导率 (k_{l,i})

需要：

* `k_coeffs: A-F`

### 10.6 黏度 (\mu_{l,i})

需要：

* `mu_coeffs: A-F`

### 10.7 相平衡式 2.19

需要：

* `boiling_temperature_atm`
* `molecular_weight`
* `activity_model`
* `unifac_groups`
* 全局 `unifac.groups`
* 全局 `unifac.interactions`

### 10.8 汽化潜热 (L_i(T))

需要：

* `hvap_ref`
* `T_ref`
* 液相 `cp_model/cp_coeffs`
* 气相 species 焓（来自气相侧热力学主线）
* 可选增强：`Tc`, `hvap_model`, `hvap_coeffs`

---

## 11. 最终规定

`paper_v1` 的液相物性主线正式定为：

1. 纯组分液相密度：文献 Eq.(2.8) 型拟合
2. 纯组分液相比热容：Shomate 主线，缺失时允许分段低阶多项式
3. 纯组分液相焓：由液相比热容积分得到
4. 纯组分液相热导率：文献 Eq.(2.11) 型拟合
5. 纯组分液相黏度：与 Eq.(2.11) 同型拟合
6. 混合液相密度：((\sum X_i \rho_i^{1/2})^2)
7. 混合液相比热容：(\sum Y_i c_{p,i})
8. 混合液相焓：(\sum Y_i h_i)
9. 混合液相热导率：Filippov
10. 混合液相黏度：Grunberg-Nissan
11. 液相扩散系数：Wilke-Chang
12. 界面活度系数：UNIFAC
13. 汽化潜热：
    [
    L_i(T)=L_i^{ref}+\tilde h_{g,i}(T)-\tilde h_{l,i}(T)
    ]

这就是 `paper_v1` 的液相物性计算最终指导文件。

[1]: https://webbook.nist.gov/chemistry/guide/?utm_source=chatgpt.com "A Guide to the NIST Chemistry WebBook"

