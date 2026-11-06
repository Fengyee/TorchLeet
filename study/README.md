# 4 周冲刺学习计划：MLE / Research Scientist / Research Engineer（CV · 多模态 · 生成模型方向）

> 为你定制：**理论扎实、手写代码少**；**4 周 × ~20 小时/周**；同时准备 **Applied MLE、Research Scientist、Research Engineer / ML Infra**；领域 **CV / 多模态 / 生成**。
>
> 核心判断：你的瓶颈不在“懂不懂”，而在“能不能在 45 分钟内、在白板或共享编辑器里，把它**正确、向量化、带 shape 标注**地写出来，并且边写边讲清楚”。所以这份计划 **70% 时间用于手写代码**，20% 用于 breadth 口述复习和 system design，10% 用于 mock 和复盘。

---

## 0. 第一天开始前（30 分钟）

```bash
pip install torch jupyter        # CPU 就够；Triton 题额外 pip install triton（仅 Linux；Mac 用 Colab），用 TRITON_INTERPRET=1 在 CPU 上跑
pip install -e python/           # 安装本仓库的自动评分器 torchleet
jupyter lab
```

```python
from torchleet import check, hint, status
check("softmax", softmax)        # 评分：验证性质（与 PyTorch 内置函数一致、不变量），不是比对答案
hint("softmax")                  # 渐进提示 1 → hint("softmax", 2) → hint("softmax", 3)
status()                         # 本机已通过的题
```

配套资料（都在 `study/`）：

| 文件 | 用途 | 什么时候用 |
|---|---|---|
| [`pytorch-cheatsheet.md`](pytorch-cheatsheet.md) | PyTorch API / 惯用法 / 坑 | 第 1 周每天开工前读一节；之后当字典 |
| [`ml-breadth.md`](ml-breadth.md) | 高频 breadth/depth 问答（英文，可直接口述） | 每天 30 分钟，**出声说**答案 |
| [`ml-system-design.md`](ml-system-design.md) | ML 系统设计框架 + 6 个 CV/多模态案例 | 每周 1–2 次，每次 60–90 分钟 |
| [`AUDIT.md`](AUDIT.md) | 原仓库题目哪里有错、怎么改的 | 做到对应题目时看一眼 |
| [`../PROBLEMS.md`](../PROBLEMS.md) | 全部题目索引（按目录和难度） | 找题 |

---

## 1. 每道题的做法（比做多少题更重要）

模拟真实面试的节奏，**每道题都按这个流程**：

1. **读题 + 定接口（5 分钟）**：先在纸上写出所有输入输出的 shape、dtype、边界情况（空输入？batch=1？mask 全为 False？）。面试里这一步就是“clarifying questions”。
2. **先写测试再写实现**：notebook 里已有测试 cell；你再补 1 个自己想到的边界 case。
3. **限时实现**：easy 25 分钟 / medium 45 分钟 / hard 75 分钟。写的时候**出声解释**（假装面试官在听）。
4. **卡住的规则**：卡 15 分钟 → `hint(id)`；再卡 10 分钟 → `hint(id, 2)`；再卡 → 看 `_SOLN`，但**第二天必须从空白重写一遍**。
5. **`check(...)` 通过后**才打开 `_SOLN` 对比：看实现差异、看 *Common bugs* 和 *Interview follow-ups*，把 follow-up 当面试追问**口头回答一遍**。
6. **记错题本**（建议一个 `notes.md`）：

   ```
   ## <题目 id>  日期  用时  一次通过? Y/N
   - 卡在哪：（例：mask 用了 == 0 而不是 bool；忘了 keepdim）
   - 正确写法一句话：
   - 下次复习日期：+2 天 / +7 天
   ```

> 评分器只验证性质，**你的写法和标准答案不同也能通过**。如果通过了但实现明显比 SOLN 慢/不稳定（Python for-loop vs 向量化、`log(softmax)` vs `log_softmax`），也要记进错题本——面试官会扣这个分。

---

## 2. 总览

| 周 | 主题 | 目标 |
|---|---|---|
| **W1** | PyTorch 手感 + 深度学习基本功 + 经典 ML/评估 | 能不看文档写出 backprop、BN、CE、优化器、训练循环、指标 |
| **W2** | 视觉 + Attention/Transformer + 多模态 | IoU/NMS/mAP、插值、MHA/GQA、ViT/MAE、CLIP/SigLIP、VLM connector |
| **W3** | 生成模型深挖 + RS 能力 | VAE/GAN/DDPM/DDIM+CFG/Flow Matching/DiT/U-Net/FID；复现一篇论文核心 |
| **W4** | 系统/RE + LLM 必备 + Mock + 速度训练 | 显存/FLOPs 估算、all-reduce、量化、KV cache、FlashAttention；限时重写 |

每天约 3 小时（周一–周六），周日 2 小时复盘，合计 ~20 小时/周。标注的时间是**含对答案和记笔记**的预算。

图例：🟢 easy 🟡 medium 🔴 hard ｜ ⭐ = 必须练到能 20 分钟内从空白写出（第 4 周速度训练清单）

---

## 3. 第 1 周：PyTorch 手感 + 深度学习基本功 + 经典 ML

| 天 | 题目（路径） | 时间 | 备注 |
|---|---|---|---|
| D1 | 环境 + cheatsheet §1–4 | 45m | broadcasting / gather / masking 是后面所有题的基础 |
| | 🟢 `v3/classical-ml/softmax` ⭐ | 20m | 数值稳定：减 max |
| | 🟢 `v3/classical-ml/logistic-regression` | 40m | 手推梯度 (p − y) |
| | 🟢 `v3/classical-ml/kmeans` ⭐ | 40m | `torch.cdist` + 向量化更新；空簇处理 |
| | 🟢 `v3/classical-ml/knn` | 30m | |
| D2 | 🟡 `v4/dl-fundamentals/mlp-backprop` ⭐ | 75m | 不用 autograd 手写 backward；梯度检查 |
| | 🟡 `v4/dl-fundamentals/micrograd` | 75m | 拓扑排序 + 梯度累加（RS/RE 经典题） |
| | breadth §1 ML fundamentals | 30m | |
| D3 | 🟡 `v4/dl-fundamentals/losses-from-scratch` ⭐ | 60m | CE+label smoothing、BCE-with-logits、focal、dice |
| | 🟡 `v4/dl-fundamentals/normalization-layers` ⭐ | 75m | BN 训练/推理两套统计量、LN、GN、inverted dropout |
| | breadth §2 优化与训练动态（前半） | 30m | |
| D4 | 🟡 `v4/dl-fundamentals/training-utilities` | 60m | SGD momentum、warmup+cosine、clip_grad_norm、EMA |
| | 🔴 `torch/hard/optimizers/adamw` | 45m | 偏差修正、解耦 weight decay |
| | 🔴 `torch/hard/custom-autograd` | 30m | `autograd.Function` + gradcheck |
| | cheatsheet §5–7 | 30m | autograd / nn.Module / 训练循环模板 |
| D5 | 🟡 `v4/dl-fundamentals/debug-training` | 75m | **找 bug 题**：从症状推原因，RS/RE 面试高频 |
| | 🟡 `torch/medium/cnn-scratch` | 60m | 自己实现 Conv2d / MaxPool（与 `F.conv2d` 对齐） |
| | 🟡 `v3/data-loading/streaming-dataset` | 30m | IterableDataset 的 worker 分片 |
| D6 | 🟡 `v4/ml-fundamentals/classification-metrics` ⭐ | 75m | P/R/F1、ROC-AUC（秩方法）、AP、ECE；面试通常只考其中 2–3 个，先做 roc_auc + AP |
| | 🟡 `v4/ml-fundamentals/decision-tree` | 75m | CART + 向量化找最佳切分点 |
| | 🟢 `v4/ml-fundamentals/pca` | 20m | SVD vs 特征分解 |
| D7 | 复盘：重写本周卡住的题；读 system design §1 框架 | 2h | 自测见下 |

**W1 自测**（周日出声回答，答不上的回 breadth 文档）：
- BN 在 train/eval 时分别用什么统计量？running_var 用的是有偏还是无偏方差？
- 为什么 `CrossEntropyLoss` 要 logits 而不是 softmax 输出？label smoothing 改变了什么？
- AdamW 和 Adam+L2 的区别？为什么 warmup？梯度裁剪裁的是什么范数？
- ROC-AUC 和 PR-AUC 在极度不平衡时哪个更有信息量？为什么？

---

## 4. 第 2 周：视觉 + Attention/Transformer + 多模态

| 天 | 题目（路径） | 时间 | 备注 |
|---|---|---|---|
| D8 | 🟢 `v4/vision/iou-nms` ⭐ | 40m | 向量化 pairwise IoU；贪心 NMS；按类 NMS 的 offset 技巧 |
| | 🟡 `v4/vision/detection-map` | 75m | 匹配规则（每个 GT 只匹配一次）+ AP 插值 |
| | 🟡 `v4/vision/bilinear-interpolation` | 60m | `align_corners` 两种坐标映射，和 `F.interpolate` 对齐 |
| D9 | 🟢 `v4/vision/mixup-cutmix` | 30m | CutMix 要按**实际裁剪面积**重算 λ |
| | 🟡 `llm/Implement-Attention-from-Scratch` ⭐ | 40m | causal + padding mask，和 SDPA 对齐 |
| | 🟡 `llm/Multi-Head-Attention` ⭐ | 50m | reshape/transpose 的 shape 管理 |
| | 🟡 `llm/Grouped-Query-Attention` | 40m | `repeat_interleave` 的分组方式 |
| | breadth §4 Transformers | 30m | |
| D10 | 🟡 `v3/modern-architectures/2d-positional-embeddings` | 45m | |
| | 🔴 `v3/modern-architectures/vit-mae` | 90m | patchify、随机 mask（argsort 噪声）、只在 masked patch 上算 loss |
| | breadth §3 CNN & vision（前半） | 30m | |
| D11 | 🟡 `v3/modern-architectures/contrastive-loss-clip` ⭐ | 45m | 对称 InfoNCE、可学习温度 |
| | 🟢 `v4/multimodal/siglip-loss` | 30m | sigmoid loss，为什么省 all-gather |
| | 🟡 `v4/multimodal/vlm-connector` | 75m | MLP projector / Perceiver resampler / 图像 token 拼接 + label mask |
| | breadth §5 多模态 | 30m | |
| D12 | 🔴 `v3/modern-architectures/swin-transformer` | 90m | window partition、cyclic shift + mask、相对位置偏置（原仓库这里的 mask 是错的，见 AUDIT） |
| | 🟡 `v4/multimodal/vector-search` | 60m | 精确 top-k（分块）+ IVF；recall@k |
| | breadth §3 CNN & vision（后半：检测/分割/自监督） | 30m | |
| D13 | **System design**：案例 a 视觉搜索 + 案例 d 图文检索 | 2h | 先自己 35 分钟写完整方案，再对照文档 |
| | 🟡 `llm/Rotary-Positional-Embedding` | 45m | 相对位置性质：⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩，位置只以 n−m 的形式出现 |
| D14 | 复盘 + 重写卡住的题 | 2h | |

**W2 自测**：
- 为什么 attention 要除以 √d？causal mask 和 padding mask 怎么组合？全被 mask 的行为什么会出 NaN？
- ViT 和 CNN 的归纳偏置差别？MAE 为什么 mask 75%？为什么 encoder 只处理可见 patch？
- CLIP 的温度为什么要可学习并 clamp？SigLIP 相比 CLIP 好在哪？
- NMS 为什么不可微？DETR 怎么去掉 NMS？mAP@0.5 和 COCO mAP@[.5:.95] 的区别？

---

## 5. 第 3 周：生成模型深挖 + Research Scientist 能力

| 天 | 题目（路径） | 时间 | 备注 |
|---|---|---|---|
| D15 | 🟡 `v4/generative/vae` ⭐ | 60m | 重参数化、ELBO、解析 KL |
| | 🔴 `torch/hard/GAN` | 60m | non-saturating loss、D 步骤 detach |
| | breadth §6 生成模型（VAE/GAN 部分） | 30m | |
| | 🟢 `v4/generative/fid` | 30m | 矩阵平方根的数值稳定写法 |
| D16 | 🔴 `v3/modern-architectures/ddpm` ⭐ | 90m | 前向加噪闭式解、ε 预测、反向采样均值/方差、最后一步（代码 0-based 的 t=0，论文的 t=1）不加噪 |
| | 🟡 `v4/vision/unet` | 60m | 时间嵌入注入、skip concat、GroupNorm |
| | 不看资料，在纸上推一遍 DDPM 的后验 q(x_{t-1}\|x_t,x_0) 和 μ_θ | 30m | RS 面试常让你当场推 |
| D17 | 🔴 `v3/modern-architectures/ddim-cfg` | 90m | DDIM 确定性更新；CFG 公式与 w 的约定 |
| | 🟡 `v4/generative/flow-matching` ⭐ | 75m | 线性路径、速度目标 x1−x0、Euler 采样（SD3/Flux 也是 rectified flow，但时间方向相反：t=1 是噪声） |
| D18 | 🔴 `v4/generative/dit-block` | 90m | adaLN-Zero：初始化时 block 为恒等映射 |
| | 🟡 `v3/modern-architectures/knowledge-distillation` | 45m | T² 缩放、`F.kl_div` 的输入约定 |
| | breadth §6 生成模型（diffusion/flow 部分） | 30m | |
| D19 | **RS 日**：选一篇你方向的论文，2 小时内从零实现其核心（例：在 2D toy 上复现 rectified-flow 的 reflow，或 MAE 的 mask ratio 消融） | 2h | 练“读论文 → 写代码 → 设计最小实验” |
| | 准备 research talk 骨架（问题 → insight → 方法 → 证据 → 局限 → 下一步） | 1h | breadth §9 |
| D20 | **System design**：案例 c 文生图服务 + 案例 b 内容审核 | 2h | |
| | 🟡 `v3/alignment-training/lora` | 45m | B 零初始化、α/r 缩放、merge |
| D21 | 复盘 + **第一次 mock**：随机抽 W1–W3 一道 medium，45 分钟限时、出声讲 | 2h | 可用 MCP tutor 的 `torchleet-interview-prep` 模式或找朋友 |

**W3 自测**：
- 从 ELBO 推出 VAE loss；为什么需要重参数化？posterior collapse 怎么缓解？
- DDPM 的 ε-预测、x0-预测、v-预测之间怎么换算？DDIM 为什么能跳步？
- CFG 的公式；为什么训练时要随机丢条件？guidance 太大会怎样？
- Flow matching 和 diffusion 的关系？为什么直线路径需要的采样步数更少？
- DiT 为什么用 adaLN-Zero 而不是 cross-attention 注入条件？FID 有哪些已知缺陷？

---

## 6. 第 4 周：系统 / RE + LLM 必备 + Mock + 速度训练

| 天 | 题目（路径） | 时间 | 备注 |
|---|---|---|---|
| D22 | 🟡 `v4/systems/memory-flops-estimation` ⭐ | 60m | 16 bytes/param、6ND、KV cache 公式（RE 必问） |
| | 🟡 `v3/llm-inference/kv-cache` | 60m | 增量解码与全量重算等价 |
| | 🟡 `v4/systems/int8-quantization` | 60m | per-tensor vs per-channel、零点、int32 累加 |
| D23 | 🔴 `v4/systems/ring-allreduce` | 90m | reduce-scatter + all-gather；DDP 梯度平均等价性 |
| | 🔴 `v3/alignment-training/gradient-checkpointing` | 60m | |
| | breadth §8 系统与效率 | 30m | |
| D24 | 🔴 `llm/Flash-Attention`（只做 Part 1：PyTorch 版） | 75m | online softmax 的重缩放（FlashAttention 的核心）；Part 2 Triton kernel 可选，CPU 上用 `TRITON_INTERPRET=1` 也能跑 |
| | 🟡 `v3/llm-decoding/top-p-sampling` | 30m | 越过阈值的那个 token 要保留 |
| | 🟡 `v3/modern-architectures/mixture-of-experts` 或 `sliding-window-attention` | 60m | 二选一 |
| D25 | **Mock 日**：2 场 coding mock（各 45 分钟，随机抽题）+ 1 场 system design mock（案例 f 训练基础设施） | 3h | 录屏/录音回听，挑出说不清楚的地方 |
| D26 | **速度训练**：⭐ 清单从空白重写，每题 ≤ 20 分钟 | 3h | 见下 |
| D27 | breadth 全文快速过一遍 + research talk 演练 2 遍 + behavioral（STAR 故事 5 个） | 3h | |
| D28 | 休息 / 缓冲 | — | |

**⭐ 速度训练清单**（D26 每题限时 20 分钟，从空白写，通过 `check`）：
`softmax` · `kmeans` · `mlp-backprop` · `losses-from-scratch`（只写 CE + label smoothing） · `normalization-layers`（只写 BatchNorm2d） · `classification-metrics`（只写 roc_auc） · `iou-nms` · `Implement-Attention-from-Scratch` · `Multi-Head-Attention` · `contrastive-loss-clip` · `vae` · `ddpm`（只写训练 loss + 一步反向采样） · `flow-matching` · `memory-flops-estimation`

---

## 7. 按岗位微调（面试排期确定后调整）

| 如果下一场是… | 加强 | 可以压缩 |
|---|---|---|
| **Applied MLE**（产品/业务） | `v4/ml-fundamentals/gradient-boosting`、`gmm-em`；system design 每周 2 次；breadth §1 指标/数据泄漏 | Swin、DiT、ring-allreduce |
| **Research Scientist** | D19 的论文复现做两次；`v3/modern-architectures/mamba`、`v3/alignment-training/dpo`；breadth §9；research talk 打磨 | data-loading、int8-quantization |
| **Research Engineer / ML Infra** | `v3/gpu-systems/fsdp`、`ring-attention`（CPU 上模拟多卡）、`triton-fused-softmax`、`flash-attention-triton`（CPU 上用 `TRITON_INTERPRET=1` 验证正确性；想测速度再上 Colab GPU）；`v3/llm-inference/continuous-batching`、`speculative-decoding` | GAN、decision-tree、Swin |

---

## 8. 题库里其余题目怎么处理

**有时间再做（有价值，但不在 4 周主线里）**
- 经典 ML：`v4/ml-fundamentals/gmm-em`、`gradient-boosting`
- 序列模型：`torch/easy/rnn`、`torch/medium/lstm`、`torch/hard/seq-seq`、`torch/hard/transformer`
- 现代架构：`v3/modern-architectures/mamba`、`mixture-of-experts`、`sliding-window-attention`
- LLM：`llm/Byte-Pair-Encoder`、`llm/RMS-Norm`、`llm/Sinusoidal-Positional-Embedding`、`llm/SmolLM`、`v3/llm-decoding/*`、`v3/llm-inference/speculative-decoding`、`continuous-batching`
- 对齐：`v3/alignment-training/dpo`、`ppo-rlhf`、`grpo`
- GPU 系统：`v3/gpu-systems/*`（CPU 可做：FSDP/ring attention 用模拟多卡，Triton 用 `TRITON_INTERPRET=1`）
- 优化器：`torch/hard/optimizers/adam`、`muon`
- 数据：`v3/data-loading/*` 其余三题

**基本可以跳过（对你的目标价值低：太基础、偏 API 演示、或只是调用现成库）**
`torch/basic/*`（除非想热身）、`torch/easy/cnn`、`torch/easy/augmentation`、`torch/easy/benchmark`、`torch/easy/cuda-amp`（读懂即可）、`torch/easy/quantize-lm`、`torch/easy/autoencoder`、`torch/medium/alexnet`、`torch/medium/3dcnn`、`torch/medium/cnn-param-init`（看 SOLN 即可）、`torch/hard/xai`（读懂 Grad-CAM 公式即可）、`llm/Create-Embeddings-out-of-an-LLM`、`v3/llm-inference/inference-engine`

---

## 9. 关于这个仓库的“可信度”

- 原仓库每道题都经过了审计（题面、公式、标准答案、评分器），发现的问题已修复，细节见 [`AUDIT.md`](AUDIT.md)。
- 所有带评分器的题目，标准答案都必须通过自己的评分器（`scripts/validate_specs.py`）；评分器还用“看似正确的错误实现”测试过能否被拒绝。
- **公司标签（Google/Meta/…）全部是推测的**（manifest 里标为 `inferred`），没有对应的面经出处。不要用它来判断“某公司一定会考”。
- 新增的 `v4/` 题目每题都由一个 agent 编写、另一个 agent 独立复核（复核又抓出并修复了一批评分器漏洞，见 AUDIT.md 末尾）。仍然建议：做题时如果觉得某个结论不对，以论文原文 / PyTorch 文档为准，并记下来。
- 没有 GPU 也能完成整个计划：所有评分器都在 CPU 上运行；Triton 题可用 `TRITON_INTERPRET=1` 在 CPU 上验证正确性（但测不了性能）。
