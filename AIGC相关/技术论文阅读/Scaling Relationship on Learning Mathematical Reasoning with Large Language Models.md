# Scaling Relationship on Learning Mathematical Reasoning with Large Language Models

情境学习 ICL

发现 1: 预训练损失

> pre-training loss is approximately negatively linear correlated to the SFT and ICL accuracy in a given interval which is a better performance indicator than pre-trained model sizes or pre-trained token counts.

提高 LLMs 数学推理能力的核心思想是在微调或推理过程中汇总各种采样推理路径。

k 固定，经过算法“选择路径推理”过滤后，小模型比大模型能生产更多的推理路径，是因为大模型容易过拟合。
