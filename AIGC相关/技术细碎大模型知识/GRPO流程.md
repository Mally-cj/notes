# GRPO 流程

## 整体流程

参考 verl 框架得到：

```yaml
for each epoch:
    for each batch:
        # 生成阶段
        给定prompt生成n_samples_per_prompts=16个不同响应
        在以同个prompt生成的响应作为一组，计算相对奖励

        # 优化阶段
        保存初始模型输出的log_probs作为参考
        for each PPO_epoch:
            打乱样本顺序并创建mini-batches
            for each mini-batch:
                计算当前策略的输出概率（log_prob）
                计算策略梯度损失（pg_loss），例如使用PPO-Clip
                计算熵损失（entropy_loss），用于维持探索性
                计算价值函数损失（value_loss，如果有价值网络的话）
                计算KL损失（kl_loss，可选，用于控制策略更新幅度）
                组合损失（total_loss = pg_loss + c1*value_loss - c2*entropy_loss + c3*kl_loss）
            反向传播并更新模型参数
                应用梯度裁剪（例如clip_grad_norm_）
            # 可选：检查KL散度是否过大，如果过大则提前结束当前PPO_epoch
```

## 过程理解

1. **被优化的模型**：

   - 主要优化的是**策略网络(actor_module)**，也就是负责生成输出的语言模型
   - 如果使用了 critic，也会优化**价值网络(critic_module)**，用于估计状态价值
2. **迭代的目的**：

   - PPO 算法的核心思想是**约束策略更新幅度**，确保新策略不会偏离旧策略太多
   - 多次使用同一批数据进行迭代，可以充分利用已收集的经验，提高样本效率
   - 通过 mini-batch 的方式，在大规模语言模型训练中更好地利用 GPU 内存
3. **迭代过程中的操作**：

   - 计算当前策略下的动作概率分布（log_prob），与旧策略（old_log_prob）进行比较
   - 使用已经计算好的优势函数（advantages），指导策略更新的方向
   - 结合裁剪目标函数（clip_ratio）限制更新幅度，避免过度优化导致策略崩溃
   - 增加熵正则化（entropy_loss）以鼓励探索
   - 可能还会添加 KL 惩罚项（kl_loss）进一步限制策略变化

## 损失函数

在 verl 框架中，`compute_policy_loss` 函数有三个返回值（`pg_loss`, `pg_clipfrac`, 和 `ppo_kl`），它们的区别和数学含义如下。

### **1. pg_loss (Policy Gradient Loss)**

这是 PPO 算法的核心损失函数，即"裁剪的代理目标"(Clipped Surrogate Objective)。它的数学公式是：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) A_t \right) \right]
$$

在代码中，这表现为：

1. 计算概率比率(ratio) = 新策略概率 / 旧策略概率
2. 计算裁剪后的目标值 = min(ratio * advantages, clipped_ratio * advantages)
3. 取负值并平均，得到最终的损失值

`pg_loss` 是通过最小化这个损失值来优化策略，使其向着更高回报的方向更新。

### **2. pg_clipfrac (Policy Gradient Clipping Fraction)**

这不是一个损失值，而是一个指标，表示在计算过程中有多少概率比率被裁剪了。数学上，它表示：

$$
\text{clipfrac} = \frac{\text{Number of clipped ratios}}{\text{Total number of ratios}}
$$

在代码中，它通过计算超出裁剪范围(1-ε, 1+ε)的比率数量，除以总样本数得到。这个值帮助监控 PPO 算法的稳定性：

- 如果 `pg_clipfrac` 过高，说明太多样本被裁剪，学习率可能需要降低
- 如果 `pg_clipfrac` 接近零，可能表示学习率太小，训练效率不高

### **3. ppo_kl (KL Divergence)**

KL 散度(Kullback-Leibler Divergence)用于衡量新策略与旧策略之间的差异程度。它的数学公式是：

$$
D_{KL}(\pi_{\theta_{old}} || \pi_\theta) = \mathbb{E}_{s,a \sim \pi_{\theta_{old}}} \left[ \log \frac{\pi_{\theta_{old}}(a|s)}{\pi_\theta(a|s)} \right]
$$

在 PPO 中，我们通常使用近似计算方法，基于对数概率比率：

$$
D_{KL} \approx \mathbb{E} \left[ \log \frac{\pi_{\theta_{old}}(a|s)}{\pi_\theta(a|s)} \right] = -\mathbb{E} \left[ \log \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \right]
$$

在代码中，这通常通过对数概率之间的差异来计算。`ppo_kl` 用于监控策略更新的幅度，确保新策略没有偏离旧策略太远。

### **三者的联系与区别**

- `pg_loss` 是优化目标，直接影响模型参数更新方向
- `pg_clipfrac` 是诊断指标，帮助调整超参数(如学习率)
- `ppo_kl` 也是诊断指标，但更关注策略变化的幅度

在 PPO 实现中，我们主要优化 `pg_loss`，同时监控 `pg_clipfrac` 和 `ppo_kl` 以确保训练稳定性和效率。如果 KL 散度过大或裁剪比例过高，可能需要调整学习率或其他超参数。
