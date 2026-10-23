# 两周学习计划（2026-10-26 → 2026-11-08）

**背景**：CS PhD，研究 neural representations。
**节奏**：工作日每天 2.5–3 小时，周末作为缓冲与复盘。
**取舍**：跳过 PyTorch 基础、经典 ML 和 CNN/RNN 入门题，集中在 (1) Transformer/LLM 核心组件、(2) 与表示学习最相关的位置编码、对比学习、MAE、diffusion、SSM、(3) 面试高频的推理题。共约 27 题。

## 每题的做法

1. 打开 notebook，先**写出所有 tensor 的 shape**，再填 TODO
2. 卡住超过 15 分钟 → 用 MCP 的 `get_hint`（从 level 1 开始），**不要直接看 `_SOLN` 文件**
3. 做完用 `torchleet-review` 模式做一次代码审查
4. 在下面的「笔记」栏写一句：这题和我研究的哪个点有关

MCP 未连接时：`claude mcp add torchleet -- npx -y torchleet-mcp`

---

## 第 1 周：从零搭一个 LLM

### 周一 10/26 · 先测一下手感
- [ ] [Custom Autograd Function (SiLU)](torch/hard/custom-autograd/custom-autgrad-function.ipynb)：`ctx` 机制，手写 backward
- [ ] [Softmax from Scratch](v3/classical-ml/softmax/softmax.ipynb)：数值稳定性
- [ ] [Muon from Scratch](torch/hard/optimizers/muon/muon.ipynb)：Newton-Schulz 正交化，和更新的几何结构有关

笔记：

### 周二 10/27 · Attention
- [ ] [Attention from Scratch](llm/Implement-Attention-from-Scratch/attention-q4-Question.ipynb)
- [ ] [Multi-Head Attention](llm/Multi-Head-Attention/multi-head-attention-q5-Question.ipynb)
- [ ] [Grouped Query Attention](llm/Grouped-Query-Attention/grouped-query-attention-Question.ipynb)：GQA 对 KV 维度共享做了什么取舍？

笔记：

### 周三 10/28 · ⭐ 位置编码（与研究最相关）
- [ ] [Sinusoidal Embeddings](llm/Sinusoidal-Positional-Embedding/sinusoidal-q7-Question.ipynb)
- [ ] [RoPE](llm/Rotary-Positional-Embedding/rope-q8-Question.ipynb)：为什么只编码相对位置？
- [ ] [2D Positional Embeddings](v3/modern-architectures/2d-positional-embeddings/2d-positional-embeddings.ipynb)

思考：和 NeRF positional encoding、Fourier features、SIREN 的关系。

笔记：

### 周四 10/29 · 归一化、分词、表征
- [ ] [RMS Norm](llm/RMS-Norm/rms-norm.ipynb)
- [ ] [Byte Pair Encoding](llm/Byte-Pair-Encoder/BPE-q3.ipynb)
- [ ] [Create Embeddings out of an LLM](llm/Create-Embeddings-out-of-an-LLM/embeddings-q2.ipynb)：mean pooling 还是 last token？

笔记：

### 周五 10/30 · 综合
- [ ] [SmolLM from Scratch](llm/SmolLM/smollm-q12-Question.ipynb)（hard）：拼起本周所有组件，能加载真实权重才算通过

笔记：

### 周末 10/31–11/1 · 缓冲
- [ ] 补完未完成的题
- [ ] （可选）[Transformer from Scratch](torch/hard/transformer/transformer.ipynb)：encoder-decoder 版

---

## 第 2 周：表示学习、生成模型、推理

### 周一 11/2 · ⭐ 对比学习
- [ ] [Contrastive Loss (InfoNCE) + CLIP](v3/modern-architectures/contrastive-loss-clip/contrastive-loss-clip.ipynb)：InfoNCE 与互信息下界；温度 τ 对 uniformity 的影响
- [ ] [Knowledge Distillation](v3/modern-architectures/knowledge-distillation/knowledge-distillation.ipynb)

笔记：

### 周二 11/3 · ⭐ 视觉表征
- [ ] [ViT + MAE Pretraining](v3/modern-architectures/vit-mae/vit-mae.ipynb)：高 mask 比例为什么有效？
- [ ] [Swin Transformer Block](v3/modern-architectures/swin-transformer/swin-transformer.ipynb)：shifted window 的归纳偏置

笔记：

### 周三 11/4 · Diffusion
- [ ] [DDPM](v3/modern-architectures/ddpm/ddpm.ipynb)：预测噪声 vs score
- [ ] [DDIM + Classifier-Free Guidance](v3/modern-architectures/ddim-cfg/ddim-cfg.ipynb)：为什么能确定性采样？

笔记：

### 周四 11/5 · 推理（面试高频）
- [ ] [KV Cache](v3/llm-inference/kv-cache/kv-cache.ipynb)：带 cache 时 mask 怎么处理？
- [ ] [Temperature Sampling](v3/llm-decoding/temperature-sampling/temperature-sampling.ipynb)
- [ ] [Top-k Sampling](v3/llm-decoding/top-k-sampling/top-k-sampling.ipynb)
- [ ] [Top-p Sampling](v3/llm-decoding/top-p-sampling/top-p-sampling.ipynb)：怎么向量化？
- [ ] [Sliding Window Attention](v3/modern-architectures/sliding-window-attention/sliding-window-attention.ipynb)

笔记：

### 周五 11/6 · 前沿架构
- [ ] [Mamba (Selective SSM)](v3/modern-architectures/mamba/mamba.ipynb)：连续时间系统离散化，可与 INR 的连续视角对照
- [ ] [LoRA](v3/alignment-training/lora/lora.ipynb)

笔记：

### 周末 11/7–8 · 复盘与模拟面试
- [ ] 用 `torchleet-interview-prep` 模式，从做过的题里随机抽 2–3 题，限时重写
- [ ] 回顾所有「笔记」，整理成面试素材

---

## 以后再做
- **GPU 系统**：Triton fused softmax、Flash Attention（Triton）、FSDP、Ring Attention
- **对齐训练**：DPO、PPO、GRPO、gradient checkpointing（偏 LLM 求职的话，DPO 优先）
- **服务端推理**：speculative decoding、continuous batching、MoE
- `torch/` 基础题和 `v3/data-loading`
