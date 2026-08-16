# DPO 公式推导

## DPO 公式

数据集 $\{x,y_w,y_l\}$，其中 x 表示问题，$y_w$表示好的（winner), $y_l$表示不好的（loser） 。

$$
J_r(r, D) = \max \mathbb{E}_{(x,y_{\text{tr}},y_{\text{tl}}) \sim D} \left[ \log \sigma \left\{ \beta \log \frac{\pi_{\hat{\theta}}(y_{u} | x)}{\pi_{\text{ref}}(y_{u} | x)} - \beta \log \frac{\pi_{\hat{\theta}}(y_{l} | x)}{\pi_{\text{ref}}(y_{l} | x)} \right\} \right]
$$

**注意点:**

由于是句子级别的，x 和 y 都是由很多 token 构成的， $\pi(y|x)$其实是把生成过程的每个 token 的交叉熵加起来了。

## DPO 的奖励分数是什么

> 就是$r(x, y) = \beta \log \frac{\pi_\theta^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$

## DPO 公式怎么来的

### 1.DPO 是为了简化 PPO 而产生

PPO 来源于 policy-based 的强化学习，它希望寻找策略使得策略奖励最大（另一类 vlue-based，是希望行动奖励最大），同时为了稳定它希望当前新策略和旧策略不能差异太大。

$$
\max_\pi \mathbb{E}_{x \sim D, y \sim \pi} \left[ r(x, y) \right] - \beta D_{\text{KL}} \left[ \pi(y|x) \parallel \pi_{\text{ref}}(y|x) \right]
$$

其中：

- $r(x, y)$：奖励函数
- $\beta$：约束强度系数
- $D_{\text{KL}}$：KL 散度，衡量策略 $\pi$与参考策略（旧策略） $\pi_{\text{ref}}$的偏离程度

然而，PPO 这样首先得训一个奖励模型，这样太麻烦。DPO 就希望绕过这个。

于是 DPO 是对 PPO 的公式展开。

### 2.对 PPO 公式展开，利用策略分布$\pi$来表示奖励 r(x,y)。

![](/wiki/AIGC%E7%9B%B8%E5%85%B3/%E6%8A%80%E6%9C%AF%E7%BB%86%E7%A2%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86/static/VbilbcyOkovuEexvIMucIPCgns6.png)

定义一个新的概率分布$\pi^* = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left\{ \frac{1}{\beta} r(x, y) \right\}$，上述展开的公式则可以写成

![](/wiki/AIGC%E7%9B%B8%E5%85%B3/%E6%8A%80%E6%9C%AF%E7%BB%86%E7%A2%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86/static/Bu5AbgIH7oF7Zox4w2HchR46nGe.png)

要让$\pi_\theta$使得 J 最小，而$logZ(x)$ 与$\pi_\theta$无关，故而只要优化第一项 KL 就好，KL 最小，则两个分布一样，即$\pi^*=\pi_\theta$,那么这个公式为

$$
\begin{equation}
\pi_\theta = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \cdot \exp\left\{ \frac{1}{\beta} r(x, y) \right\}
\end{equation}
$$

那么从这个公式就可以反推出奖励模型的表达式：$r(x, y) = \beta \log \frac{\pi_\theta^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$，这时候就可以把优化奖励模型 r 变成优化$\pi^*$了（Z(x)在后面可以）。

### 3.引入 Bradley-Terry 模型，通过偏好对比方式化解了直接最大化。

首先看 Bradley-Terry 模型定义：

![](/wiki/AIGC%E7%9B%B8%E5%85%B3/%E6%8A%80%E6%9C%AF%E7%BB%86%E7%A2%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86/static/M4k8bnosMokpfBxdAQ9clXp3nfe.png)

把要奖励最大，利用 Bradley-Terry 化解为了好的回答比坏的回答胜出的概率更大。

即 $p_\beta^*(y_w \succ y_l | x) = \frac{r(x,y_w)}{r(x,y_w)+r(x,y_l)}$，

化简后就是 $p_\beta^* = \text{sigmoid} \left( \beta \log \frac{\pi_{\text{ref}}(y_1 | x)}{\pi^*(y_1 | x)} - \beta \log \frac{\pi_{\text{ref}}(y_2 | x)}{\pi^*(y_2 | x)} \right)$

故而 DPO 的损失函数为：

![](/wiki/AIGC%E7%9B%B8%E5%85%B3/%E6%8A%80%E6%9C%AF%E7%BB%86%E7%A2%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9F%A5%E8%AF%86/static/HRRobdeL4oNn6qxf0oXcSUtcnpf.png)

## 其他问题：

1. 为什么 DPO 训练会出现正样本 reward 也会降低的现象

## 参考博客

[DPO 公式推导](https://www.yuuri.cn/%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/RL%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/DPO%E7%9B%B4%E6%8E%A5%E5%81%8F%E5%A5%BD%E4%BC%98%E5%8C%96/DPO%E5%85%AC%E5%BC%8F%E6%8E%A8%E5%AF%BC/) 把推导逻辑写清楚了

[(DPO) Bradley-Terry 模型概念-CSDN 博客](https://blog.csdn.net/u014386899/article/details/141642326) 讲 Bradley-Terry 概念

[PG/PPO/DPO/GRPO/MDPO 公式推导 | JJJYmmm's Blo](https://jjjymmm.cn/index.php/archives/127/) 公式推导清楚

[Direct Preference Optimization (DPO)原理详解及公式推导-CSDN 博客](https://blog.csdn.net/qq_36803941/article/details/142251643) 非常详细
