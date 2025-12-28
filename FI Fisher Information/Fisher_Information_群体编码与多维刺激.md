# Fisher Information：群体编码与多维刺激详解

## 目录
- [一、概念与定义](#一概念与定义)
- [二、物理意义与直觉理解](#二物理意义与直觉理解)
- [三、数学推导](#三数学推导)
- [四、群体编码中的噪声相关](#四群体编码中的噪声相关)
- [五、多维刺激的Fisher Information Matrix](#五多维刺激的fisher-information-matrix)
- [六、应用流程](#六应用流程)
- [七、MATLAB代码示例](#七matlab代码示例)
- [八、总结与要点](#八总结与要点)

---

## 一、概念与定义

### 1.1 什么是Fisher Information？
**Fisher Information (费雪信息)** 是衡量**神经系统编码能力**的核心指标。它量化了从神经元发放（Spikes）中能提取多少关于外界刺激（Stimulus）的信息。

**核心问题**：观察到神经活动后，我们能多精确地推断出引起这些活动的刺激是什么？

### 1.2 单神经元的Fisher Information
对于单个神经元，刺激 $s$ 的Fisher Information定义为：

$$I(s) = -E\left[\frac{\partial^2}{\partial s^2} \ln P(r|s)\right]$$

等价形式（更常用）：

$$I(s) = E\left[\left(\frac{\partial}{\partial s} \ln P(r|s)\right)^2\right]$$

其中：
- $r$：神经元的响应（发放率）
- $s$：刺激参数
- $P(r|s)$：给定刺激 $s$ 下观察到响应 $r$ 的概率

### 1.3 多维刺激的Fisher Information Matrix (FIM)
当刺激是 $k$ 维向量 $\mathbf{s} = [s_1, s_2, \dots, s_k]^T$ 时，Fisher Information变成一个 $k \times k$ 的矩阵：

$$\mathcal{I}_{ij}(\mathbf{s}) = -E\left[\frac{\partial^2}{\partial s_i \partial s_j} \ln P(\mathbf{r}|\mathbf{s})\right]$$

或：

$$\mathcal{I}_{ij}(\mathbf{s}) = E\left[\left(\frac{\partial \ln P}{\partial s_i}\right)\left(\frac{\partial \ln P}{\partial s_j}\right)\right]$$

---

## 二、物理意义与直觉理解

### 2.1 Cramér-Rao下界
Fisher Information的最重要性质是它提供了**估计误差的理论下界**：

$$\text{Var}(\hat{s}) \geq \frac{1}{I(s)}$$

对于多维情况：

$$\text{Cov}(\hat{\mathbf{s}}) \succeq \mathcal{I}(\mathbf{s})^{-1}$$

**含义**：无论使用多么聪明的解码算法，从神经活动中估计刺激的误差都不可能低于 $1/I(s)$（或 $\mathcal{I}^{-1}$）。

### 2.2 直观理解
Fisher Information可以理解为"**信号的敏感度平方除以噪声**"：

$$I(s) = \frac{[\text{Signal Sensitivity}]^2}{\text{Noise}}$$

- **高FI**：神经元对刺激变化敏感（斜率大）且噪声小 → 编码精确
- **低FI**：神经元反应迟钝或噪声大 → 编码模糊

### 2.3 群体编码的几何意义
在群体编码中，FI描述了：
- **信号方向 $\mathbf{f}'(s)$**：刺激改变时，群体活动在高维空间中移动的方向
- **噪声椭球 $Q$**：群体活动的随机波动形状

**关键**：如果噪声方向与信号方向平行 → FI降低；如果垂直 → FI受影响小。

---

## 三、数学推导

### 3.1 单神经元泊松模型
假设神经元发放服从泊松分布，平均发放率为 $f(s)$：

$$P(r|s) = \frac{f(s)^r e^{-f(s)}}{r!}$$

**步骤1：取对数**
$$\ln P(r|s) = r \ln f(s) - f(s) - \ln(r!)$$

**步骤2：对 $s$ 求一阶导数**
$$\frac{\partial}{\partial s}\ln P(r|s) = \frac{r \cdot f'(s)}{f(s)} - f'(s) = \left(\frac{r}{f(s)} - 1\right)f'(s)$$

**步骤3：计算期望的平方**
$$I(s) = E\left[\left(\left(\frac{r}{f(s)} - 1\right)f'(s)\right)^2\right]$$

由于 $E[r] = f(s)$，$\text{Var}(r) = f(s)$（泊松性质）：

$$I(s) = \frac{[f'(s)]^2}{f(s)^2} \cdot \text{Var}(r) = \frac{[f'(s)]^2}{f(s)^2} \cdot f(s)$$

**最终结果**：
$$\boxed{I(s) = \frac{[f'(s)]^2}{f(s)}}$$

### 3.2 群体编码（无相关）
如果有 $N$ 个神经元，且它们的噪声**相互独立**，则群体FI为简单求和：

$$I_{\text{pop}}(s) = \sum_{n=1}^{N} I_n(s) = \sum_{n=1}^{N} \frac{[f_n'(s)]^2}{f_n(s)}$$

---

## 四、群体编码中的噪声相关

### 4.1 问题背景
真实神经系统中，神经元之间存在**噪声相关（Noise Correlations）**：当神经元A因随机因素发放增加时，神经元B往往也跟着增加。

### 4.2 数学模型
假设群体响应 $\mathbf{r} = [r_1, r_2, \dots, r_N]^T$ 服从多元高斯分布：

$$P(\mathbf{r}|\mathbf{s}) = \mathcal{N}(\mathbf{f}(\mathbf{s}), Q(\mathbf{s}))$$

其中：
- $\mathbf{f}(\mathbf{s})$：平均调谐曲线向量（$N \times 1$）
- $Q(\mathbf{s})$：协方差矩阵（$N \times N$），描述噪声相关性

### 4.3 群体Fisher Information公式
对于多元高斯分布，群体FI为：

$$I_{\text{pop}}(s) = \mathbf{f}'(s)^T Q^{-1}(s) \mathbf{f}'(s) + \frac{1}{2}\text{Tr}\left[Q^{-1}\frac{\partial Q}{\partial s}Q^{-1}\frac{\partial Q}{\partial s}\right]$$

**通常简化**（假设 $Q$ 不随 $s$ 变化）：

$$\boxed{I_{\text{pop}}(s) = \mathbf{f}'(s)^T Q^{-1}(s) \mathbf{f}'(s)}$$

维度：$(1 \times N)(N \times N)(N \times 1) = 1 \times 1$（标量）

### 4.4 相关性的影响
- **有害相关（Differential Correlations）**：噪声方向与 $\mathbf{f}'(s)$ 平行 → 严重降低FI
- **无害相关**：噪声方向与 $\mathbf{f}'(s)$ 垂直 → 影响小
- **独立情况**：$Q = \text{diag}(\sigma_1^2, \dots, \sigma_N^2)$ → 回到简单求和

### 4.5 深入理解：信号方向与噪声方向的几何关系

#### 4.5.1 什么是信号方向？
**信号方向 $\mathbf{f}'(s)$** 是一个N维向量，描述当刺激 $s$ 改变时，神经元群体活动在高维空间中改变的方向。

**具体含义**：
- 第 $n$ 个分量 $f_n'(s)$ 表示第 $n$ 个神经元对刺激改变的敏感度（调谐曲线的斜率）
- 如果神经元 $n$ 对刺激变化不敏感，则 $f_n'(s) \approx 0$
- 如果神经元 $n$ 对刺激变化很敏感，则 $|f_n'(s)|$ 很大

**例子**：考虑3个神经元的群体，当刺激从 $s$ 变到 $s + \Delta s$ 时：
- 神经元1的发放率变化：$\Delta r_1 \approx f_1'(s) \cdot \Delta s$
- 神经元2的发放率变化：$\Delta r_2 \approx f_2'(s) \cdot \Delta s$
- 神经元3的发放率变化：$\Delta r_3 \approx f_3'(s) \cdot \Delta s$

那么群体活动的改变向量就是 $(f_1'(s), f_2'(s), f_3'(s))^T \cdot \Delta s$，方向就是 $\mathbf{f}'(s)$。

#### 4.5.2 什么是噪声方向？
**噪声方向** 由协方差矩阵 $Q$ 的特征向量决定。协方差矩阵 $Q$ 描述了神经元群体活动的随机波动的结构。

**特征向量分解**：
$$Q = U \Lambda U^T$$

其中：
- $U$ 的列向量 $\mathbf{u}_i$ 是噪声的主要方向（Principal Noise Directions）
- $\Lambda$ 的对角元 $\lambda_i$ 是沿着这些方向的噪声方差

**几何直觉**：
- 神经元群体活动的噪声波动不是各向同性的（均匀分布），而是有特定的方向偏好
- 沿着 $Q$ 的大特征向量方向，噪声很大
- 沿着 $Q$ 的小特征向量方向，噪声很小

#### 4.5.3 有害与无害相关的几何解释

假设我们有3个神经元，在2维刺激空间中工作。考虑以下两种情况：

**情况1：有害相关（信号与噪声平行）**

```
高维空间示意图（简化为2D）：

    神经元响应空间
         |
    f_2'| 
        |     • 噪声椭球长轴
        |    /
        |   / (信号方向 f')
        |  /
    ----•----------- f_1'
```

- **信号方向 $\mathbf{f}'$** 和 **噪声椭球长轴** 几乎平行
- 当刺激改变时，群体活动沿着 $\mathbf{f}'$ 方向移动
- **但恰好这个方向噪声最大**（噪声椭球的长轴方向）
- 结果：解码器很难分辨"这是刺激改变导致的活动变化"还是"这只是随机噪声"
- **Fisher Information 严重损失**

**情况2：无害相关（信号与噪声垂直）**

```
高维空间示意图（简化为2D）：

    神经元响应空间
         |
    f_2'| 
        |    • 噪声椭球长轴
        |    |
        |    | (噪声大)
        |    |
    ----•---•-------- f_1' (信号方向 f')
        (噪声小)
```

- **信号方向 $\mathbf{f}'$** 和 **噪声椭球长轴** 垂直
- 当刺激改变时，群体活动沿着 $\mathbf{f}'$ 方向移动
- **这个方向的噪声很小**（噪声椭球的短轴方向）
- 结果：刺激诱导的活动变化能清晰地从随机噪声中凸显出来
- **Fisher Information 基本不受影响**

#### 4.5.4 数学解释：为什么是 $\mathbf{f}'^T Q^{-1} \mathbf{f}'$？

Fisher Information 的公式 $I = \mathbf{f}'^T Q^{-1} \mathbf{f}'$ 包含了这个几何关系：

**分解推导**：
设 $Q = U \Lambda U^T$（特征值分解），其中 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_N)$

则 $Q^{-1} = U \Lambda^{-1} U^T = U \text{diag}(1/\lambda_1, 1/\lambda_2, \dots, 1/\lambda_N) U^T$

代入Fisher Information公式：
$$I = \mathbf{f}'^T U \Lambda^{-1} U^T \mathbf{f}'$$

令 $\mathbf{c} = U^T \mathbf{f}'$（将信号向量投影到噪声主成分坐标系）：
$$I = \sum_{i=1}^{N} \frac{c_i^2}{\lambda_i}$$

**解释**：
- $c_i$ 是信号方向在第 $i$ 个噪声主轴上的投影
- $\lambda_i$ 是沿着第 $i$ 个主轴的噪声方差
- 若 $c_i$ 大、$\lambda_i$ 小（信号强、噪声小）→ $c_i^2/\lambda_i$ 大 → 贡献大
- 若 $c_i$ 大、$\lambda_i$ 大（信号强、但噪声也大）→ $c_i^2/\lambda_i$ 小 → 贡献被削弱
- 若 $c_i$ 小（信号方向与该噪声主轴垂直）→ 贡献为0（不被噪声影响）

**关键结论**：
- **有害相关**：信号方向 $\mathbf{f}'$ 与大特征值（大噪声）的特征向量平行 → Fisher Information 大幅下降
- **无害相关**：信号方向 $\mathbf{f}'$ 与小特征值（小噪声）的特征向量平行 → Fisher Information 基本不变

#### 4.5.5 具体数值例子

考虑2个神经元的简单例子：

**情况A：完全正相关（有害）**
- 两个神经元发放完全同步（噪声完全相关）
- $Q = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$，$Q^{-1}$ 不存在（退化）
- 信号向量 $\mathbf{f}' = (1, 1)^T$（两个神经元对刺激敏感度相同）
- 问题：两个神经元提供的信息完全冗余，就像只有一个神经元

**情况B：正交不相关（无害）**
- 两个神经元噪声独立
- $Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$，$Q^{-1} = Q$
- 信号向量 $\mathbf{f}' = (1, 1)^T$（两个神经元对刺激敏感度相同）
- $I = \mathbf{f}'^T Q^{-1} \mathbf{f}' = (1, 1) \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} (1, 1)^T = 2$
- 两个独立神经元的信息相加：$I = 1 + 1 = 2$

**情况C：垂直相关（无害）**
- 两个神经元的噪声相关，但信号方向垂直于相关方向
- $Q = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$（中等相关）
- 信号向量 $\mathbf{f}' = (1, -1)^T$（一个神经元增加，另一个减少）
- $Q^{-1} = \frac{1}{3} \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$
- $I = (1, -1) \frac{1}{3} \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} (1, -1)^T = \frac{1}{3}(3 - 1 - 1 + 3) = \frac{4}{3}$
- 尽管有相关，Fisher Information 仍然相当可观，因为信号方向避开了最大噪声方向

---



## 五、多维刺激的Fisher Information Matrix

### 5.1 定义
当刺激是 $k$ 维向量 $\mathbf{s} = [s_1, \dots, s_k]^T$ 时，FIM 是 $k \times k$ 矩阵：

$$\mathcal{I}(\mathbf{s}) = \mathbf{F}'(\mathbf{s})^T Q^{-1} \mathbf{F}'(\mathbf{s})$$

其中 $\mathbf{F}'(\mathbf{s})$ 是**雅可比矩阵**（$N \times k$）：

$$\mathbf{F}'(\mathbf{s}) = \begin{bmatrix}
\frac{\partial f_1}{\partial s_1} & \frac{\partial f_1}{\partial s_2} & \cdots & \frac{\partial f_1}{\partial s_k} \\
\frac{\partial f_2}{\partial s_1} & \frac{\partial f_2}{\partial s_2} & \cdots & \frac{\partial f_2}{\partial s_k} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_N}{\partial s_1} & \frac{\partial f_N}{\partial s_2} & \cdots & \frac{\partial f_N}{\partial s_k}
\end{bmatrix}$$

### 5.2 矩阵维度分析
$$\mathcal{I} = \underbrace{\mathbf{F}'^T}_{k \times N} \underbrace{Q^{-1}}_{N \times N} \underbrace{\mathbf{F}'}_{N \times k} = k \times k$$

**关键**：无论有多少个神经元（$N$），FIM的维度只取决于刺激维度 $k$。它是**整个群体共用**的一个矩阵。

### 5.3 物理意义
FIM的矩阵元素：
- **对角元 $\mathcal{I}_{ii}$**：关于第 $i$ 个刺激维度的信息量
- **非对角元 $\mathcal{I}_{ij}$**：刺激维度 $i$ 和 $j$ 的编码耦合程度

**特征值分解**：
$$\mathcal{I} = V \Lambda V^T$$
- **特征值 $\lambda_i$**：在第 $i$ 个特征方向上的信息量
- **特征向量 $\mathbf{v}_i$**：最优/最差编码的刺激组合方向

### 5.4 Cramér-Rao下界（多维）
估计误差的协方差矩阵满足：

$$\text{Cov}(\hat{\mathbf{s}}) \succeq \mathcal{I}(\mathbf{s})^{-1}$$

**误差椭球**：$\mathcal{I}^{-1}$ 定义了估计误差在多维空间的分布形状。

---

## 六、应用流程

### 6.1 计算流程图

```
输入: 刺激 s, 神经元调谐曲线 f(s), 噪声协方差 Q
    ↓
步骤1: 计算调谐曲线的导数/雅可比矩阵
    - 单维: f'(s)
    - 多维: F'(s) [N × k矩阵]
    ↓
步骤2: 计算协方差矩阵的逆 Q^(-1)
    - 如果独立: Q^(-1) = diag(1/σ₁², ..., 1/σₙ²)
    - 如果相关: 需要矩阵求逆
    ↓
步骤3: 计算Fisher Information
    - 单维: I(s) = f'(s)ᵀ Q^(-1) f'(s)
    - 多维: ℐ(s) = F'(s)ᵀ Q^(-1) F'(s)
    ↓
输出: Fisher Information (标量或矩阵)
```

### 6.2 实际应用场景
1. **神经编码分析**：评估神经元群体对不同特征（方向、颜色、位置）的编码精度
2. **最优刺激设计**：找到能最大化FI的刺激集合
3. **群体规模优化**：确定需要多少神经元才能达到目标精度
4. **相关性影响评估**：量化噪声相关对编码效率的影响

---

## 七、MATLAB代码示例

### 7.1 示例1：单维刺激 + 群体编码（有噪声相关）

**代码文件**: [`demo1_single_dim_with_correlations.m`](demo1_single_dim_with_correlations.m)

**功能说明**:
- 模拟50个神经元群体对单维刺激（方向角度）的编码
- 考虑神经元间的噪声相关性（指数衰减模型）
- 对比有相关和独立编码情况下的Fisher Information
- 可视化调谐曲线、FI曲线和相关矩阵

**主要输出**:
- 调谐曲线示意图
- Fisher Information随刺激变化曲线
- 噪声相关系数矩阵热图
- 相关性导致的信息损失百分比

---

### 7.2 示例2：多维刺激 (2D位置) + Fisher Information Matrix

**代码文件**: [`demo2_multidim_FIM_2D_position.m`](demo2_multidim_FIM_2D_position.m)

**功能说明**:
- 模拟30个神经元对2D空间位置的编码
- 计算完整的Fisher Information Matrix (2×2矩阵)
- 分析FIM的行列式、迹、特征值在空间上的分布
- 可视化误差椭球（Cramér-Rao下界）

**主要输出**:
- FIM行列式的空间分布（总信息量）
- FIM特征值分布（编码方向性）
- 条件数分布（编码各向异性）
- 特定位置的Cramér-Rao误差椭圆
- 神经元感受野中心分布

#### 可视化详解

该示例生成6个子图，每个子图揭示FIM的不同方面：

**Subplot 1: log₁₀(det(FIM) + 1) - FIM行列式**

- **数学含义**: $\det(\mathcal{I}) = \lambda_1 \times \lambda_2$，是两个特征值的乘积
- **物理意义**: 衡量**二维联合估计的总体信息量**，等价于误差椭圆面积的倒数
- **如何解读**:
  - 亮区（高值）：在该位置同时精确估计 x 和 y 两个坐标
  - 暗区（低值）：至少有一个方向的信息量很弱，联合估计困难
  - 通常在神经元感受野中心附近最亮（红点标记处）
- **为什么用 log₁₀**: 避免零值取对数，并压缩动态范围便于观察

**Subplot 2: log₁₀(Trace(FIM) + 1) - FIM的迹**

- **数学含义**: $\text{tr}(\mathcal{I}) = \lambda_1 + \lambda_2$，是两个特征值的和
- **物理意义**: 两个方向信息量的**总和**（而非乘积）
- **如何解读**:
  - 与行列式相比，迹对"单方向很弱"的情况更宽容
  - 若某位置 trace 高但 det 低 → 信息主要集中在一个方向（各向异性强）
  - 若 trace 和 det 都高 → 两个方向都有充足信息（理想情况）

**Subplot 3: log₁₀(λₘₐₓ + 1) - 最大特征值**

- **数学含义**: FIM 最大特征值 $\lambda_{\max}$
- **物理意义**: 在**信息最强方向**上的精度指标
- **如何解读**:
  - 显示每个位置"最容易估计的方向"有多精确
  - 亮区表示至少存在一个方向估计非常精确
  - 需要与 λₘᵢₙ 联合查看才能判断整体性能

**Subplot 4: log₁₀(λₘᵢₙ + 1) - 最小特征值**

- **数学含义**: FIM 最小特征值 $\lambda_{\min}$
- **物理意义**: 在**信息最弱方向**上的精度指标（瓶颈）
- **如何解读**:
  - 这是联合估计的"短板"——决定了整体最差情况
  - 若此图某区域很暗，即使 λₘₐₓ 很亮，也意味着强各向异性
  - 暗区表示存在某个方向几乎无法精确估计

**Subplot 5: log₁₀(Condition Number) - 条件数**

- **数学含义**: $\kappa = \lambda_{\max} / \lambda_{\min}$
- **物理意义**: 衡量**各向异性**程度（信息分布的不均匀性）
- **如何解读**:
  - 亮区（高值）：强各向异性，信息高度偏向某一方向，误差椭圆很扁
  - 暗区（低值，接近1）：各向同性，两个方向信息量相近，误差椭圆接近圆形
  - 理想编码应该条件数低（信息均衡分布）
- **实际含义**: 高条件数意味着解码算法在某方向上不稳健

**Subplot 6: Error Ellipse @ (test_x, test_y) - 误差椭圆**

- **数学含义**: Cramér-Rao 下界协方差矩阵 $\text{Cov} = \mathcal{I}^{-1}$，通过特征分解绘制
- **物理意义**: 任何无偏估计器的**估计误差最小可达范围**（1σ 椭圆）
- **如何解读**:
  - **椭圆大小**: 越小表示该位置估计越精确
  - **椭圆长轴**: 指向信息**较弱**的方向（因为 Cov 与 FIM 互逆）
  - **椭圆短轴**: 指向信息**较强**的方向
  - **长短轴比**: 等于 $\sqrt{\lambda_{\max}/\lambda_{\min}}$（即条件数的平方根）
- **数值关系**:
  - 长轴长度 ∝ $1/\sqrt{\lambda_{\min}}$（FIM 的最小特征值）
  - 短轴长度 ∝ $1/\sqrt{\lambda_{\max}}$（FIM 的最大特征值）
  - 椭圆面积 ∝ $1/\sqrt{\det(\mathcal{I})}$

#### 综合阅读策略

1. **先看 Subplot 1 (det)**: 找到信息量总体最高的区域
2. **对比 Subplot 1 和 2 (det vs trace)**: 
   - 若 det 低但 trace 高 → 单方向强，需警惕各向异性
   - 若两者都高 → 理想区域
3. **用 Subplot 4 (λₘᵢₙ) 找瓶颈**: 最弱方向决定联合估计的下限
4. **用 Subplot 5 (条件数) 评估稳健性**: 高条件数警示解码不稳定
5. **在 Subplot 6 验证直觉**: 误差椭圆直观展示前面指标的综合结果

#### 进阶分析建议

- **多点误差椭圆**: 在空间网格上采样多个点，叠加绘制多个椭圆，观察空间变化趋势
- **主轴方向可视化**: 用箭头标注 FIM 特征向量方向，直观看信息的方向偏好
- **95% 置信椭圆**: 将椭圆缩放 $\sqrt{\chi^2_{2,0.95}} \approx 2.45$ 倍，更符合统计惯例
- **数值稳定性**: 若 FIM 接近奇异（λₘᵢₙ ≈ 0），求逆时添加正则化：`inv(FIM + 1e-6*eye(2))` 或使用伪逆 `pinv(FIM)`

---

### 7.3 示例3：模拟神经解码并验证Cramér-Rao下界

**代码文件**: [`demo3_verify_cramer_rao_bound.m`](demo3_verify_cramer_rao_bound.m)

**功能说明**:
- 通过1000次蒙特卡洛模拟验证理论预测
- 使用最大似然解码器从泊松噪声响应中估计刺激
- 对比实际解码精度与Cramér-Rao理论下界
- 检验解码器是否达到最优性能

**主要输出**:
- 估计值的概率分布直方图
- 实测标准差与CR下界的柱状对比
- 前100次试次的估计值时间序列
- 解码偏差和效率统计量

#### 解码对数似然与CR下界解读

- 对数似然来源: 泊松分布的对数似然为 $\sum_n\big[r_n\ln\lambda_n-\lambda_n-\ln(r_n!)\big]$；最大似然只需最大化与 $s$ 有关的部分 $\sum_n\big[r_n\ln\lambda_n-\lambda_n\big]$。代码中 `log_likelihood(i) = sum(r .* log(f_grid + 1e-10) - f_grid);` 正是该式的实现，`+1e-10` 用于数值稳定避免 `\log(0)`。
- 网格搜索含义: 对每个候选刺激 $s_{\text{grid}}$ 计算观测响应在该 $s$ 下出现的“可能性”，取对数似然最大的 $s$ 作为最大似然估计 (MLE)。
- 为什么能接近/达到 CR 下界: 泊松属于指数族，在正则条件下 MLE 渐近有效，方差可达 Cramér–Rao 下界。实测标准差与下界非常接近（偶尔略低或略高）通常源于有限试次和网格离散化的统计波动。
- 波动量级估计: 标准差估计的标准误差近似 $\mathrm{SE}(\hat{\sigma}) \approx \sigma/\sqrt{2(n-1)}$；如 $n=1000$ 且 $\sigma\approx1.7\,\mathrm{deg}$，则 $\mathrm{SE}\approx0.038\,\mathrm{deg}$，与观察到的微小偏差相符。

---

## 八、总结与要点

### 关键公式速查

| 场景 | Fisher Information 公式 |
|------|------------------------|
| 单神经元（泊松） | $I(s) = \frac{[f'(s)]^2}{f(s)}$ |
| 群体（独立） | $I = \sum_{n=1}^{N} \frac{[f_n'(s)]^2}{f_n(s)}$ |
| 群体（有相关） | $I = \mathbf{f}'(s)^T Q^{-1}(s) \mathbf{f}'(s)$ |
| 多维刺激 | $\mathcal{I} = \mathbf{F}'(\mathbf{s})^T Q^{-1} \mathbf{F}'(\mathbf{s})$ |

### 重要概念

1. **FIM 是群体共用的 $k \times k$ 矩阵**（$k$ = 刺激维度）
2. **噪声相关性的方向决定其影响**：与信号平行→有害；垂直→无害
3. **特征值分解揭示最优/最差编码方向**
4. **Cramér-Rao 下界是任何解码器的性能极限**

### 实际应用注意事项

- 真实数据中，调谐曲线和协方差矩阵需要从实验数据拟合
- 高维情况下，矩阵求逆可能数值不稳定，需要正则化
- 泊松假设在高发放率或短时间窗口下可能失效，需要更精细的噪声模型

---

## 参考文献

1. Abbott, L. F., & Dayan, P. (1999). The effect of correlated variability on the accuracy of a population code. *Neural Computation*, 11(1), 91-101.

2. Seung, H. S., & Sompolinsky, H. (1993). Simple models for reading neuronal population codes. *PNAS*, 90(22), 10749-10753.

3. Berens, P., et al. (2012). A fast and simple population code for orientation in primate V1. *Journal of Neuroscience*, 32(31), 10618-10626.

4. Bethge, M., et al. (2002). Optimal short-term population coding: When Fisher information fails. *Neural Computation*, 14(10), 2317-2351.

---

**文档创建日期**: 2025-12-23  
**版本**: 1.0  
**作者**: Computational Neuroscience Study Group
