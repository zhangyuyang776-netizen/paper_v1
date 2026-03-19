# grid_partition_and_moving_mesh_guideline_final

## 1. 文件目的

本文档用于为 `paper_v1` 正式固定：

1. 三段网格的划分方式  
2. 每一段网格的节点/控制面位置定义  
3. 动网格中控制面速度 `v_c` 的分段定义  
4. 网格速度、流体速度、界面速度之间的关系  
5. outer predictor-corrector 与 inner fixed-geometry solve 的具体循环流程  
6. remap / projection 的作用范围  
7. 时间步接受/拒绝与外层收敛判据

本文档是 `paper_v1` 中“网格与动网格实现”的正式定稿版本。

---

## 2. 总体原则

### 2.1 三段网格主线

`paper_v1` 与主论文一样，将计算域分为三段：

1. **第一段：液相区**
   \[
   0 \le r \le a(t)
   \]

2. **第二段：近界面气相区**
   \[
   a(t) \le r \le r_I
   \]
   其中
   \[
   r_I = 5 a_0
   \]

3. **第三段：远场气相区**
   \[
   r_I \le r \le r_{end}
   \]

其中：
- `a(t)` 为当前液滴半径
- `a0` 为初始液滴半径
- `r_end` 为固定远场边界

### 2.2 与文献的一致性与改造

本文件保留文献的三段网格思想，但对动网格实现做如下工程化改造：

- **第一段**：采用与文献一致的同尺度缩放网格速度
- **第二段**：采用满足 `v(a)=\dot a` 且 `v(r_I)=0` 的仿射网格速度
- **第三段**：采用固定网格，不随时间步移动

这样做的目的，是在保留“三段网格 + 近界面精细、远场放粗”的数值思想下，避免把文献中的

\[
v_n = \dot a \frac{r_n}{a}
\]

生搬到第二段和第三段，造成：
- 在 `r_I=5a0` 处网格速度不为 0
- 在远场 `r_end` 处网格速度不为 0
- 第三段远场网格速度被不合理放大
- 与固定远场边界和固定第三段网格的实现目标冲突

### 2.3 三种速度必须区分

本文件中必须严格区分：

1. **流体速度**
   \[
   u_l,\;u_g
   \]
   由连续方程恢复得到

2. **界面速度**
   \[
   \dot a = \frac{da}{dt}
   \]
   由 inner 收敛后通过界面关系更新

3. **网格/控制面速度**
   \[
   v_c
   \]
   由当前 outer iterate 的网格运动规则给出

它们不是同一个量。

---

## 3. 三段网格划分

## 3.1 第一段：液相区

区间：

\[
0 \le r \le a(t)
\]

采用均匀网格，设液相 cell 数为 `N1`。

### 控制面位置
\[
r_{1,f}(j) = j\,\Delta r_1,\qquad j=0,1,\dots,N_1
\]

其中

\[
\Delta r_1 = \frac{a}{N_1}
\]

### cell center 位置
\[
r_{1,c}(j) = \left(j+\frac12\right)\Delta r_1,\qquad j=0,1,\dots,N_1-1
\]

---

## 3.2 第二段：近界面气相区

区间：

\[
a(t) \le r \le r_I,\qquad r_I=5a_0
\]

采用均匀网格，设 cell 数为 `N2`。

### 控制面位置
\[
r_{2,f}(j) = a + j\,\Delta r_2,\qquad j=0,1,\dots,N_2
\]

其中

\[
\Delta r_2 = \frac{r_I-a}{N_2}
\]

### cell center 位置
\[
r_{2,c}(j) = a + \left(j+\frac12\right)\Delta r_2,\qquad j=0,1,\dots,N_2-1
\]

### 第二段步长的时间行为
若液滴总体随时间缩小，则 `a(t)` 递减，因此：

\[
\Delta r_2(t)=\frac{5a_0-a(t)}{N_2}
\]

会随时间逐渐增大。

---

## 3.3 第三段：远场气相区

区间：

\[
r_I \le r \le r_{end}
\]

第三段采用**固定网格**，不随时间步移动。  
推荐使用**固定几何扩张比的几何网格**，并在初始化时一次性生成。

与当前 `mesh schema` 保持一致，第三段的正式输入参数不是 `N3`，而是：

- `far_stretch_ratio`

因此第三段的正式参数化逻辑为：

1. 第三段首格尺寸由第二段步长上包络给出
2. 第三段扩张比直接取配置中的 `far_stretch_ratio`
3. 从 `r_I` 开始按固定扩张比逐格向外生成
4. 直到最后一个控制面到达 `r_end`
5. 第三段实际 cell 数 `N3` 是生成结果，不是配置主输入

### 第三段首格尺寸

本文件正式规定第三段的第一个网格尺寸为：

\[
\Delta r_{3,1} = \frac{r_I}{N_2} = \frac{5a_0}{N_2}
\]

这是第二段网格尺寸在 `a \to 0` 时理论上能达到的最大值。

### 这样定义的意义

因为：

\[
\Delta r_2(t)=\frac{5a_0-a(t)}{N_2}\le \frac{5a_0}{N_2}=\Delta r_{3,1}
\]

因此：
- 在 `t=0` 时，第二段步长为
  \[
  \Delta r_2(0)=\frac{4a_0}{N_2}
  \]
  第三段首格为
  \[
  \Delta r_{3,1}=\frac{5a_0}{N_2}
  \]
  两者比值为 `5/4`
- 随着液滴缩小，第二段步长逐步逼近第三段首格尺寸
- 第三段首格始终大于等于第二段步长，不会出现“第二段更粗、第三段第一格反而更细”的步长折返

这保留了文献中“在第二段和第三段交界处保持网格尺度相对连续”的主要数值意图。

### 第三段几何扩张网格

正式采用固定扩张比：

\[
\alpha = \texttt{far\_stretch\_ratio}, \qquad \alpha \ge 1
\]

并按

\[
\Delta r_{3,m} = \Delta r_{3,1}\,\alpha^{m-1},\qquad m=1,2,\dots
\]

逐格向外生成，直到到达 `r_end`。

因此第三段的构造逻辑是：

- `\Delta r_{3,1}` 由 `5a_0/N_2` 给定
- `\alpha` 由 `far_stretch_ratio` 给定
- `N3` 由“铺满到 `r_end`”的过程决定

这与当前 `MeshConfig` 的正式字段完全一致：

- `n_liq`
- `n_gas_near`
- `far_stretch_ratio`

### 控制面位置
\[
r_{3,f}(0)=r_I
\]

\[
r_{3,f}(j)=r_I+\sum_{m=1}^{j}\Delta r_{3,m},\qquad j=1,2,\dots,N_3
\]

### cell center 位置
\[
r_{3,c}(j)=\frac12\left(r_{3,f}(j)+r_{3,f}(j+1)\right),\qquad j=0,1,\dots,N_3-1
\]

### 说明
本文件不采用“第三段每个时间步重新根据第二段步长重构”的主线，而是采用**固定第三段网格**。  
这是对文献三段网格的工程化修改，优点是：

- 远场边界条件更干净
- 第三段控制面速度可直接取 0
- remap 范围局部化
- 首版实现更稳定
- 与当前 `schema -> preprocess -> MeshConfig` 的参数化链条一致

---

## 4. 控制面速度 `v_c` 的分段定义

## 4.1 总原则

`v_c` 是**网格/控制面速度**，不是流体速度。  
在 moving control volume 方程中，对流通量统一使用：

\[
u-v_c
\]

### 为什么对流项是 `u-v_c`
因为对流通量必须表示**流体相对于移动控制面**的净输运。  
若控制面与流体一起运动，则相对对流输运应为 0。

### 为什么扩散项不减 `v_c`
因为扩散通量 `J_i`、导热通量 `q` 是相对于流体本身定义的构成通量，不属于控制面运动修正项。

---

## 4.2 第一段控制面速度

在液相区，采用同尺度缩放律：

\[
v_c(r)=\dot a\,\frac{r}{a},\qquad 0\le r\le a
\]

### 这个公式的原理
它保证液相网格点满足：

\[
\frac{r}{a}=\text{const}
\]

即液相节点相对于界面的归一化位置保持不变。

### 边界验证
- 在中心 `r=0`：
  \[
  v_c(0)=0
  \]
- 在界面 `r=a`：
  \[
  v_c(a)=\dot a
  \]

---

## 4.3 第二段控制面速度

在近界面气相区，采用仿射映射速度：

\[
v_c(r)=\dot a\,\frac{r_I-r}{r_I-a},\qquad a\le r\le r_I
\]

其中：

\[
r_I=5a_0
\]

### 这个公式的原理
第二段节点可写成：

\[
r=a+\xi(r_I-a),\qquad 0\le\xi\le 1
\]

若令 `xi` 固定，则对时间求导：

\[
\dot r=\dot a(1-\xi)
\]

代入

\[
\xi=\frac{r-a}{r_I-a}
\]

即可得到：

\[
\dot r=\dot a\,\frac{r_I-r}{r_I-a}
\]

### 边界验证
- 在界面 `r=a`：
  \[
  v_c(a)=\dot a
  \]
- 在第二段上边界 `r=r_I`：
  \[
  v_c(r_I)=0
  \]

因此它与“第二段上边界固定在 `5a0`”这一实现目标自洽。

---

## 4.4 第三段控制面速度

第三段为固定网格，因此：

\[
v_c(r)=0,\qquad r_I\le r\le r_{end}
\]

### 说明
这意味着：

- 第三段控制体体积不因 moving mesh 改变
- 第三段不需要网格位置更新
- 第三段不需要网格重映射

这是首版与推荐版的正式主线。

---

## 4.5 分段速度的连续性

按本文件的定义：

- 第一段在界面 `r=a` 处给出 `v_c=\dot a`
- 第二段在界面 `r=a` 处也给出 `v_c=\dot a`
- 第二段在 `r=r_I` 处给出 `v_c=0`
- 第三段在 `r=r_I` 处也给出 `v_c=0`

因此控制面速度在两个分段点都是连续的。

---

## 5. 三种速度的关系

## 5.1 流体速度 `u`

- `u_l`, `u_g` 为流体速度
- 由连续方程离散恢复
- 在 inner solve 中作为派生量得到
- 不作为主 unknown

## 5.2 界面速度 `\dot a`

- `\dot a = da/dt`
- 是几何更新量
- 不属于 inner unknown
- 在 inner 收敛后由

\[
\dot a = u_{l,if} + \frac{mpp}{\rho_{s,l}}
\]

更新

## 5.3 网格速度 `v_c`

- `v_c` 是控制面/网格速度
- 由当前 outer iterate 的几何试探值 `a^(k), \dot a^(k)` 和本文件的分段规则给出
- 进入对流项中的形式为 `u-v_c`

---

## 6. outer predictor-corrector 与 inner solve 的分层

## 6.1 已接受时间层

在时间层 `t^n`，已知：

- 已接受状态 `U^n`
- 已接受半径 `a^n`
- 已接受界面速度 `\dot a^n`
- 对应网格 `G^n`

目标推进到：

\[
t^{n+1}=t^n+\Delta t
\]

---

## 6.2 Outer predictor 初始化

第 0 轮 outer iteration 取：

\[
\dot a^{(0)} = \dot a^n
\]

并采用 Euler predictor：

\[
a^{(0)} = a^n + \Delta t\,\dot a^{(0)}
\]

这定义了当前候选几何。

---

## 6.3 当前 outer iterate 的网格构建

对第 `k` 轮 outer iteration：

1. 给定 `a^(k)`
2. 用本文件第 3 节的三段规则构建当前网格 `G^(k)`
3. 用本文件第 4 节的分段规则构建控制面速度 `v_c^(k)`

其中：

- 第一段和第二段由 `a^(k)` 改变
- 第三段固定不变

### 特别说明
在 inner solve 中冻结：

- `a^(k)`
- `\dot a^(k)`
- `G^(k)`
- `v_c^(k)`

不允许在 inner Newton 过程中再更新这些量。

---

## 6.4 remap / projection

将已接受旧状态 `U^n` 投影到当前 outer iterate 网格 `G^(k)` 上，形成：

\[
U^{old,*\,(k)}
\]

### 正式规定
- 第一段、第二段：执行 conservative remap
- 第三段：由于网格固定，可直接拷贝对应状态

### 这样做的好处
- 只在动网格区域重映射
- 降低 remap 误差传播范围
- 与第三段固定网格设计一致

---

## 6.5 Inner fixed-geometry solve

在当前固定几何与固定网格上，求解：

\[
U^{(k)} = [\Phi_l,\phi_s,\Phi_g]
\]

inner solve 中：

- `u_l, u_g` 由连续方程恢复
- `mpp` 用 Eq.(2.18) 做主残差
- Eq.(2.18)`中的 `\dot a` 必须取当前 outer iterate 的冻结值：
  \[
  \dot a=\dot a^{(k)}
  \]

不允许：
1. inner 试一个 `mpp_trial`
2. 再用它实时更新 `\dot a_trial`
3. 再回灌当前 residual

否则 residual 不再是 fixed-geometry residual。

---

## 6.6 Outer corrector

inner 收敛后，得到：

- `mpp^(k)`
- `u_{l,if}^{(k)}`
- `rho_{s,l}^{(k)}`
- 以及完整状态 `U^(k)`

更新新的界面速度：

\[
\dot a^{new}
=
u_{l,if}^{(k)}
+
\frac{mpp^{(k)}}{\rho_{s,l}^{(k)}}
\]

再采用 trapezoidal corrector 更新界面位置：

\[
a^{new}
=
a^n + \frac{\Delta t}{2}\left(\dot a^n+\dot a^{new}\right)
\]

---

## 6.7 Outer 收敛判据

定义外层界面速度收敛误差：

\[
\varepsilon_{\dot a}^{(k)}
=
\frac{
\left|\dot a^{new}-\dot a^{(k)}\right|
}{
\max\left(|\dot a^{new}|,\epsilon_{\dot a}\right)
}
\]

推荐：

\[
\epsilon_{\dot a} = 10^{-12}
\]

并采用收敛条件：

\[
\varepsilon_{\dot a}^{(k)} < 10^{-5}
\]

若满足，则 outer 收敛，时间步可进入接受检查。

若不满足，则：

\[
\dot a^{(k+1)} = \dot a^{new}
\]
\[
a^{(k+1)} = a^{new}
\]

然后重建第一、第二段网格，进入下一轮 outer iteration。

---

## 6.8 时间步接受 / 拒绝

### 接受条件
时间步仅在以下条件同时满足时接受：

1. inner nonlinear solve 收敛
2. outer predictor-corrector 收敛
3. 没有出现非物理状态：
   - `a <= 0`
   - `T` 越界
   - `Y < 0`
   - `sum(Y)` 明显偏离 1
   - 物性计算失败

### 失败处理
若失败：

- 缩小 `Δt`
- 回到已接受状态 `U^n, a^n, \dot a^n`
- 重新尝试该时间步

---

## 7. 本方案相对文献的数值意义

## 7.1 保留的文献思想

本方案保留了文献的以下数值思想：

1. 三段网格结构  
2. 第一段液相随界面缩放  
3. 第二段近界面气相保持均匀细网格  
4. 第三段远场逐渐放粗  
5. 分段点附近避免网格尺度突变  
6. moving control volume 中对流使用 `u-v_c`

## 7.2 本方案的工程化改造

本方案相对文献的主要改造是：

1. 第二段使用仿射速度，而不是继续套用 `\dot a r/a`
2. 第三段固定，不随时间更新
3. 第三段首格尺寸固定为 `5a0/N2`，作为第二段步长的上包络
4. 第三段扩张比由 `far_stretch_ratio` 直接给定，而不是由 `N3` 反解
5. remap 仅在第一、第二段执行

## 7.3 这样做的优点

1. `r_I=5a0` 和 `r_end` 可以严格固定  
2. 第二段与第三段交界处的网格速度连续为 0  
3. 避免远场网格速度被不合理放大  
4. 第三段边界条件和离散更稳定  
5. 与当前 mesh schema 完全一致  
6. 首版实现复杂度明显降低  
7. 仍保留了文献网格连续性的主要数值意图

---

## 8. 正式禁止事项

以下做法在本文件下明确禁止：

1. 将文献中的
   \[
   v=\dot a\,\frac{r}{a}
   \]
   无差别用于第二段和第三段
2. 在第二段上边界 `r=5a0` 处让控制面速度非零
3. 在第三段固定网格的前提下，又在 inner residual 中使用非零第三段 `v_c`
4. 在 inner Newton 过程中更新 `a^(k)` 或 `\dot a^(k)`
5. 对第三段固定网格仍执行不必要的 remap
6. 省略 outer predictor-corrector 收敛判据

---

## 9. 最终方法合同

### Grid Partition and Moving-Mesh Contract

1. `paper_v1` uses a three-region radial grid:
   - liquid region `[0,a]`
   - near-interface gas region `[a,5a0]`
   - far-field gas region `[5a0,r_end]`
2. Region 1 uses uniform moving mesh with
   \[
   v_c=\dot a\,r/a
   \]
3. Region 2 uses uniform moving mesh with
   \[
   v_c=\dot a\,(r_I-r)/(r_I-a)
   \]
4. Region 3 is a fixed grid with
   \[
   v_c=0
   \]
5. The first cell size in region 3 is fixed to
   \[
   \Delta r_{3,1}=5a_0/N_2
   \]
6. Region 3 uses fixed geometric growth with `far_stretch_ratio` from the mesh schema; the actual far-field cell count is derived, not prescribed.
7. Only regions 1 and 2 are remapped during outer iterations.
8. Inner solve is always performed on a fixed geometry with frozen `a^(k)` and `\dot a^(k)`.
9. Outer convergence uses the interface-velocity consistency error `\varepsilon_{\dot a}`.
10. Outer predictor-corrector updates `a` only after inner convergence.
11. Convective fluxes use `u-v_c`; diffusive fluxes do not include `v_c`.

---

## 10. 最终结论

`paper_v1` 的网格划分与动网格正式定为：

- 三段网格
- 第一段液相缩放
- 第二段仿射动网格
- 第三段固定几何扩张网格
- outer predictor-corrector 更新 `a`
- inner fixed-geometry solve 只解物理状态
- `v_c` 按分段规则给定
- 对流通量统一使用 `u-v_c`

这就是 `paper_v1` 的网格划分和动网格指导文件最终定稿版本。
