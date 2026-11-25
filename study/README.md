# 4 周学习路线：CV、多模态与生成模型

这份路线面向 Applied MLE、Research Scientist 和 Research Engineer 面试。每周约 20 小时。
不必一次做完所有题：先完成每周的核心题，再按目标岗位加练。完整题目列表见
[`PROBLEMS.md`](../PROBLEMS.md)。

## 开始前

在仓库根目录安装依赖并启动 notebook：

```bash
pip install torch jupyter
pip install -e python/
jupyter lab
```

每道题按这个顺序练：

1. 先写清输入、输出的 shape 和边界情况。
2. 独立尝试；卡住时先用 `hint("<题目 id>")`，再依次试第 2、3 级提示。
3. 用题目 notebook 末尾的 `check(...)` 评分。评分器检查正确性质，不要求和标准答案写法相同。
4. 通过后再看 `_SOLN`，记录一个没想到的边界情况或错误。

```python
from torchleet import check, hint, status

hint("softmax")       # hint("softmax", 2) / hint("softmax", 3)
check("softmax", softmax)
status()              # 本机已通过的题目
```

## 四周安排

每天做表中列出的核心题；如果时间不够，先完成 ⭐ 题。第 7 天用于复习，不加新题。

### 第 1 周：PyTorch 与深度学习基础

| 天 | 核心任务 |
|---|---|
| 1 | `v3/classical-ml/softmax` ⭐、`logistic-regression`、`kmeans` ⭐ |
| 2 | `v4/dl-fundamentals/mlp-backprop` ⭐、`micrograd` |
| 3 | `losses-from-scratch` ⭐、`normalization-layers` ⭐ |
| 4 | `training-utilities`、`torch/hard/optimizers/adamw` |
| 5 | `debug-training`、`torch/medium/cnn-scratch` |
| 6 | `classification-metrics` ⭐、`decision-tree`、`pca` |
| 7 | 重做本周卡住的题；口头复习 ML 基础和训练动态 |

### 第 2 周：视觉、注意力与多模态

| 天 | 核心任务 |
|---|---|
| 8 | `v4/vision/iou-nms` ⭐、`detection-map` |
| 9 | `llm/Implement-Attention-from-Scratch` ⭐、`Multi-Head-Attention` ⭐ |
| 10 | `v4/vision/vit-components`、`v3/modern-architectures/vit-mae` |
| 11 | `v3/modern-architectures/contrastive-loss-clip` ⭐、`v4/multimodal/vlm-connector` |
| 12 | `v4/vision/self-supervised-losses`、`segmentation-metrics` |
| 13 | 做一次视觉搜索或图文检索系统设计；复习 RoPE |
| 14 | 重做本周卡住的题；口头复习视觉与多模态 |

### 第 3 周：生成模型与研究表达

| 天 | 核心任务 |
|---|---|
| 15 | `v4/generative/vae` ⭐、`vq-vae`、`fid` |
| 16 | `v3/modern-architectures/ddpm` ⭐、`v4/generative/diffusion-parameterizations` |
| 17 | `v3/modern-architectures/ddim-cfg`、`v4/generative/flow-matching` ⭐ |
| 18 | `v4/generative/dit-block`、`v4/vision/unet` |
| 19 | 读一篇相关论文，复现一个小实验，并练习简短介绍 |
| 20 | 做一次文生图或内容审核系统设计；练习 LoRA |
| 21 | 重做卡住的题；做一次 45 分钟模拟面试 |

### 第 4 周：系统、LLM 与模拟面试

| 天 | 核心任务 |
|---|---|
| 22 | `v4/systems/memory-flops-estimation` ⭐、`int8-quantization` |
| 23 | `ring-allreduce`、`tensor-parallel` |
| 24 | `llm/Flash-Attention` Part 1、`v3/llm-inference/kv-cache` |
| 25 | 两次 coding mock 和一次 system design mock |
| 26 | 从空白限时重写本周及前几周的 ⭐ 题 |
| 27 | 复习 breadth 问答，练习 research talk 和行为面试 |
| 28 | 休息或补做落下的内容 |

## 易混淆的数学约定

GitHub 支持用 `$...$` 写行内公式、用 `$$...$$` 写独立公式。RoPE 的相对位置性质是：

$$
\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle.
$$

这里假设各位置使用同一组旋转矩阵；注意偏移量是 `n - m`，不是 `m - n`。

本仓库的 flow matching 约定是：

$$
x_t = (1-t)x_{\text{noise}} + t x_{\text{data}},\qquad
v = x_{\text{data}} - x_{\text{noise}}.
$$

因此 `t=0` 是噪声、`t=1` 是数据。其他实现可能反过来定义时间方向，使用公式前先确认约定。
DDPM 的代码下标从 0 开始：反向过程最后一步是代码中的 `t=0`，这一步不再加噪声；
论文常用从 1 开始的时间下标。

## 按面试岗位加练

- **Applied MLE：** `v4/ml-fundamentals/ab-testing`、`gradient-boosting`；多练指标和系统设计。
- **Research Scientist：** 多做论文复现，练习 research talk；可加 `mamba`、`dpo`。
- **Research Engineer / ML Infra：** 加练 `v3/gpu-systems/fsdp`、Triton、`sequence-packing` 和连续批处理。

## 配套资料

- [PyTorch 速查表](pytorch-cheatsheet.md)：需要查 API 时使用。
- [ML breadth 问答](ml-breadth.md)：每天挑几题，先口头回答再看答案。
- [ML system design](ml-system-design.md)：每周练 1–2 个案例。
- [审计报告](AUDIT.md)：题目、公式和评分器的修订记录。
