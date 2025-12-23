# VAE 入门：用通俗语言理解变分自编码器

面向读者：刚开始接触生成模型，想要“看懂 VAE 在干嘛、为什么要这么做”的初学者。

---

## 1. VAE 是什么？能解决什么问题？

**变分自编码器（Variational Auto-Encoder, VAE）**是一类**生成模型（Generative Model, GM）**：
- 输入一张图片（或一段数据），它能学会把数据压缩成一个“潜在表示”
- 然后还能从这个潜在表示里把数据“生成/重建”出来
- 更重要的是：学好以后，它能从潜在空间里“采样”出新的数据（比如生成新图片）

和普通**自编码器（Auto-Encoder, AE）**相比，VAE 的核心区别在于：
- AE 学的是“压缩-还原”
- VAE 学的是“压缩成一个**概率分布**，并且这个分布要规整到某个先验分布，方便采样生成”

---

## 2. 自编码器回顾：Encoder / Decoder 在做什么？

VAE 仍然是“编码器 + 解码器”的结构：

- **编码器（Encoder, Enc）**：把数据 $x$ 映射到潜在变量 $z$
- **解码器（Decoder, Dec）**：从 $z$ 生成/重建数据 $\hat{x}$

在 AE 里，编码器给出的是一个确定的 $z$；
在 VAE 里，编码器给出的不是一个点，而是一个分布：

- **近似后验分布（Approximate Posterior Distribution, $q_\phi(z\mid x)$）**

它通常被设成高斯分布：

- $q_\phi(z\mid x) = \mathcal{N}(z;\, \mu_\phi(x), \operatorname{diag}(\sigma^2_\phi(x)))$

也就是说编码器输出两组向量：
- 均值向量 $\mu$
- 方差（或标准差）向量 $\sigma^2$ / $\sigma$

---

## 3. 为什么要引入概率？从“压缩”到“可生成”

如果你只做 AE：
- 你可以重建训练集
- 但你很难保证“随便拿一个 $z$”就能解码出合理图片

VAE 的想法是：
- 让每个样本对应的 $q_\phi(z\mid x)$ 都“不要太散、不要太怪”
- 并尽量靠近一个简单的**先验分布（Prior Distribution, $p(z)$）**

最常用的先验是标准正态分布：
- $p(z)=\mathcal{N}(0, I)$

这样训练好后，我们就能：
- 直接从 $\mathcal{N}(0,I)$ 采样 $z$
- 再丢给解码器生成新样本

---

## 4. 概率图景：VAE 的“生成过程”长什么样？

VAE 假设数据是这样生成的：
1. 先从先验 $p(z)$ 抽一个潜在变量 $z$
2. 再从**似然（Likelihood, $p_\theta(x\mid z)$）**生成数据 $x$

所以整体的边缘分布是：
- $p_\theta(x) = \int p_\theta(x\mid z)p(z)\,dz$

难点来了：这个积分通常算不动。

---

## 5. 变分推断：为什么要“变分”？

我们希望最大化数据的对数似然：
- $\log p_\theta(x)$

但 $p_\theta(x)$ 积分难算，所以引入一个可学习的近似分布 $q_\phi(z\mid x)$ 来近似真实的后验：
- **真实后验分布（True Posterior Distribution, $p_\theta(z\mid x)$）**

这个思路叫：
- **变分推断（Variational Inference, VI）**

最终我们优化的是一个下界：
- **证据下界（Evidence Lower Bound, ELBO）**

---

## 6. ELBO 长什么样？每一项代表什么？

对单个样本 $x$，ELBO 通常写作：

$$
\begin{aligned}
\mathcal{L}(\theta,\phi; x)
&= \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]\\
&\quad - \mathrm{KL}\big(q_\phi(z\mid x)\,\Vert\,p(z)\big)
\end{aligned}
$$

它有两部分：

1) **重建项（Reconstruction Term）**
- $\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]$
- 直觉：解码器要能把 $z$ 还原成 $x$（重建得像）

2) **KL 散度项（Kullback–Leibler Divergence, KL Divergence）**
- $\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$
- 直觉：别让编码出来的分布乱跑，要贴近先验 $\mathcal{N}(0,I)$（更好采样、更规整）

训练时通常做“最大化 ELBO”，等价于最小化负号后的损失：

$$
\mathcal{J}(\theta,\phi) = -\mathcal{L}(\theta,\phi; x)
$$

---

## 7. 重建项怎么变成我们熟悉的损失？

关键在于你对 $p_\theta(x\mid z)$ 的假设：

- 如果数据 $x$ 是连续值（如归一化后的像素），常假设
  - $p_\theta(x\mid z)$ 是高斯分布 $\mathcal{N}(\mu_\theta(z), \sigma^2 I)$
  - 对应的负对数似然接近 **均方误差（Mean Squared Error, MSE）**

- 如果数据 $x$ 是 0/1 或在 [0,1] 的伯努利建模，更常假设
  - $p_\theta(x\mid z)$ 是伯努利分布（Bernoulli Distribution）
  - 对应的负对数似然就是 **二元交叉熵（Binary Cross Entropy, BCE）**

所以你经常会看到 VAE 损失像：
- `reconstruction_loss + KL_loss`

### 7.1 重建项“到底怎么算”？（从 $-\log p_\theta(x\mid z)$ 到 BCE/MSE）

ELBO 的重建项是：

$$
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

训练时我们通常最小化它的相反数（负对数似然）：

$$
\mathcal{L}_{\text{recon}}(x)
\;=\; -\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

但期望不方便精确计算，于是用 **蒙特卡洛估计（Monte Carlo Estimation, MC Estimation）**：
从 $q_\phi(z\mid x)$ 采样 $L$ 次（实践中常用 $L=1$），得到 $z^{(l)}$，用平均近似期望：

$$
\mathcal{L}_{\text{recon}}(x)
\;\approx\; -\frac{1}{L}\sum_{l=1}^{L} \log p_\theta\big(x\mid z^{(l)}\big)
$$

接下来就看你对 $p_\theta(x\mid z)$ 的分布假设：

**情况 A：伯努利似然（Bernoulli Likelihood）→ BCE**

如果把每个像素 $x_i\in\{0,1\}$（或 $[0,1]$ 的概率强行当作伯努利参数）看作独立伯努利：
\;
$p_\theta(x\mid z)=\prod_i \operatorname{Bernoulli}(x_i;\,\hat{x}_i)$，其中 $\hat{x}=\text{Decoder}(z)$。

那么对数似然是：

$$
\log p_\theta(x\mid z)
= \sum_i \Big( x_i\log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)\Big)
$$

负号取反后就是常见的 **二元交叉熵（Binary Cross Entropy, BCE）**（按像素求和/求均值仅是实现细节）。

**情况 B：高斯似然（Gaussian Likelihood）→ MSE（加常数）**

如果假设 $p_\theta(x\mid z)=\mathcal{N}(x;\mu_\theta(z), \sigma^2 I)$（$\mu_\theta(z)$ 为解码器输出），则：

$$
-\log p_\theta(x\mid z)
= \frac{1}{2\sigma^2}\lVert x-\mu_\theta(z)\rVert_2^2 + \text{const}
$$

忽略常数项并把 $\sigma^2$ 当作固定值时，它就等价于我们熟悉的 **均方误差（Mean Squared Error, MSE）**（差别仅是一个系数）。

---

## 8. 关键技巧：重参数化（Reparameterization Trick）

问题：
- ELBO 里有 $\mathbb{E}_{q_\phi(z\mid x)}[\cdot]$
- 也就是要从 $q_\phi(z\mid x)$ 里采样 $z$
- 但采样操作本身不可导，反向传播会卡住

解决：
- **重参数化技巧（Reparameterization Trick, RT）**

把采样写成“确定性函数 + 独立噪声”：

$$
\epsilon \sim \mathcal{N}(0, I),\quad
z = \mu + \sigma \odot \epsilon
$$

- $\odot$ 表示逐元素乘（Hadamard product）
- 这样随机性来自 $\epsilon$（与参数无关），$\mu,\sigma$ 仍可导

这一步是 VAE 能用神经网络端到端训练的关键。

### 8.1 为什么 RT 能“让采样可导”？

在反向传播（Backpropagation, BP）里，我们需要对参数 $\phi$ 求梯度。
如果直接写 $z \sim q_\phi(z\mid x)$，采样操作把计算图“打断”，梯度不容易从 $z$ 传回到 $\phi$。

RT 的做法是把“带参数的随机采样”改写为：

1. 先从一个**与参数无关**的固定噪声分布采样：$\epsilon\sim\mathcal{N}(0,I)$
2. 再用一个**可导的确定性函数**把噪声变成 $z$：$z=g_\phi(x,\epsilon)$

在对角高斯的常见设定下：

$$
\begin{aligned}
\epsilon &\sim \mathcal{N}(0, I)\\
z &= \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
\end{aligned}
$$

这样随机性被“搬运”到了 $\epsilon$，而 $\mu_\phi(x)$、$\sigma_\phi(x)$ 都是网络输出，可直接求导。

### 8.2 实现细节：为什么常输出 `logvar`？

实践中编码器通常输出：
- $\mu$（均值）
- `logvar`（对数方差，$\log\sigma^2$）

原因是：
- 用 `logvar` 计算 $\sigma = \exp(0.5\,\text{logvar})$ 数值更稳定
- 同时也天然保证方差为正（不用额外约束）

---

## 9. KL 项为什么能算得很漂亮？

在最常见的设置下：
- $q_\phi(z\mid x)=\mathcal{N}(\mu,\operatorname{diag}(\sigma^2))$
- $p(z)=\mathcal{N}(0,I)$

那么 KL 散度有闭式解：

$$
\mathrm{KL}(q\Vert p)=\frac{1}{2}\sum_{i=1}^d\left(\mu_i^2 + \sigma_i^2 - \log \sigma_i^2 - 1\right)
$$

实际实现中常让网络输出 $\log \sigma^2$（也叫 `logvar`），更稳定。

---

## 10. 一张“训练流程”心智图（不写代码也能懂）

对每个 batch：
1. 输入数据 $x$
2. 编码器输出 $\mu(x)$、$\log\sigma^2(x)$
3. 采样 $\epsilon\sim\mathcal{N}(0,I)$，用 RT 得到 $z$
4. 解码器输出重建 $\hat{x}$（或输出分布参数）
5. 计算重建损失（对应 $-\log p_\theta(x\mid z)$）
6. 计算 KL 损失（把 $q_\phi(z\mid x)$ 往 $p(z)$ 拉）
7. 两者相加，反向传播更新 $\theta,\phi$

---

## 11. 初学者常见困惑（FAQ）

### 11.1 为什么叫“自编码器”，却又是“生成模型”？
- 自编码器结构负责“编码-解码”
- 变分部分保证潜在空间对齐先验，从而可以“采样生成”

### 11.2 KL 项会不会让模型变差？
- KL 项会让潜在空间更规整，但也可能让重建变糊
- 这不是 bug，而是 VAE 的典型特性：它优化的是似然下界，偏向覆盖数据分布而不是像素级锐利

### 11.3 为什么生成的图片有时比较“糊”？
常见原因：
- 使用简单的高斯/伯努利似然导致输出趋向平均
- 解码器容量不足
- KL 约束过强（潜在变量被“压扁”）

---

## 12. 术语速查表（首次出现已标注，这里便于复习）

- 变分自编码器（Variational Auto-Encoder, VAE）
- 生成模型（Generative Model, GM）
- 自编码器（Auto-Encoder, AE）
- 编码器（Encoder, Enc）
- 解码器（Decoder, Dec）
- 潜在变量（Latent Variable, $z$）
- 先验分布（Prior Distribution, $p(z)$）
- 似然（Likelihood, $p_\theta(x\mid z)$）
- 真实后验分布（True Posterior Distribution, $p_\theta(z\mid x)$）
- 近似后验分布（Approximate Posterior Distribution, $q_\phi(z\mid x)$）
- 变分推断（Variational Inference, VI）
- 证据下界（Evidence Lower Bound, ELBO）
- KL 散度（Kullback–Leibler Divergence, KL Divergence）
- 重参数化技巧（Reparameterization Trick, RT）

---

## 13. 建议的入门练习（不增加概念负担）

1) 用 2 维潜在空间训练一个 VAE（比如 MNIST），然后画出 $z$ 空间的散点图，观察类别是否聚集。
2) 在 $\mathcal{N}(0,I)$ 中取一条直线插值两个 $z$，看看解码后的图像如何平滑变化。

如果你希望我再补一份“配套最小可运行代码（PyTorch）”版，我也可以在同一目录下再加一个脚本/Notebook。

---

## 14. 配套最小可运行代码（PyTorch）

我已经在同目录提供了一个最小可运行脚本：
- `vae_mnist_minimal.py`：用 **PyTorch（PyTorch）** + **torchvision（torchvision）** 在 MNIST 上训练一个最小 VAE（MLP 版），并保存采样图与重建图。

### 14.1 安装依赖

在当前目录（`VAE Variational Auto-Encoders/`）打开终端运行：

```bash
pip install -r requirements.txt
```

> 如果你机器的 CUDA/CPU 环境导致 `torch` 安装失败，建议按 PyTorch 官网选择对应安装命令（不同 CUDA 版本会不同）。

### 14.2 运行训练

```bash
python vae_mnist_minimal.py --epochs 5
```

### 14.3 你会看到什么输出？

脚本会在 `out/` 目录生成：
- `samples_epoch_*.png`：从先验 $p(z)=\mathcal{N}(0,I)$ 采样生成的图
- `recon_epoch_*.png`：重建对比图（上：原图；下：重建）
- `vae_mnist.pt`：训练后的模型权重