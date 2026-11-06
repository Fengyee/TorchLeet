# 审计报告：原仓库 72 道题的正确性与实现质量

> 回答你最初的问题：**“内容是否正确、代码是否是好的实现？”**
> 结论：**原仓库的问题比看起来严重得多**，但现在全部已修复并经过验证。

## TL;DR

| | 题数 | 含义 |
|---|---|---|
| 🔴 **至少一个严重问题（critical）** | **35 / 72** | 公式/答案错误、题目或答案无法运行、或者**在教错误的东西** |
| 🟠 **至少一个重要问题（major），无 critical** | **33 / 72** | 评分器接受错误答案或拒绝正确答案、题目与答案不一致、题面缺少关键约定 |
| 🟢 只有小问题 | **4 / 72** | `preloaded-dataset`、`sinusoidal-positional-embedding`、`ring-attention`、`ddpm` |

- README 写的是 68 题，实际有 72 个题目目录（计数本身就过期了）。
- **评分器（`check()`）普遍太弱**：大多数题的 grader 原本只检查“输出形状对、能反传、loss 会降”，所以**很多经典错误写法也能通过**（忘记 1/√d、mask 方向反了、Adam 的 ε 放错位置、top-p 丢掉越过阈值的 token……）。现在每个 grader 都用“看似正确的错误实现”测试过，确保它们会被拒绝；同时用“写法不同但正确的实现”测试过，确保不会误杀。
- 公司标签（Google/Meta/…）**全部是推测的**，没有任何面经出处（manifest 里标为 `inferred`）。

## 怎么审的

1. **9 个审计 agent 分组审完全部 72 题**，每题检查：题面与公式（对照论文原文/PyTorch 文档）、实际运行答案并与 PyTorch 内置实现做数值比对（`F.scaled_dot_product_attention`、`nn.LSTM`、`F.conv2d`、`torch.optim.Adam`、`nn.MultiheadAttention`……）、实现质量、题目与答案是否一致、评分器强度。
2. 每个修复后都要求：`scripts/validate_specs.py`（每道题的标准答案必须通过自己的评分器）、`python/test_grader.py`（评分器回归测试）全部通过。
3. **独立复核**：再由另一批 agent（不看审计 agent 的结论，只看改动）复核每一处修复和每一句新增的说法。复核结果见文末。

---

## 最值得你知道的 15 个问题

这些是“如果你照着原仓库学，会学错或白练”的问题：

| # | 题目 | 原来的问题 |
|---|---|---|
| 1 | `llm/Multi-Head-Attention`、`llm/Grouped-Query-Attention` | 标准答案**每次调用都新建随机的 `nn.Linear`**——参数根本无法训练；题目自带的“与 `nn.MultiheadAttention` 对比”测试**永远失败** |
| 2 | `v3/alignment-training/ppo-rlhf`、`grpo` | KL 惩罚写成 `mean(logπ − logπ_ref)` 当 loss 用，**on-policy 时期望梯度为 0**，根本不会把策略拉回 reference；GRPO 也没用 DeepSeekMath 的 k3 估计量 |
| 3 | `v3/modern-architectures/swin-transformer` | shifted-window mask 的区域切片位置错了，**只要 shift ≠ window/2 就错**——包括 Swin-T 的真实配置（window 7, shift 3）；原测试只测过 8×8/4/2 这一种恰好正确的情况 |
| 4 | `v3/llm-inference/kv-cache` | 只在“没有 cache”时加 causal mask，**多 token prefill 时会看到未来 token** |
| 5 | `v3/llm-inference/inference-engine` | batch 生成时右侧 padding，然后读 `logits[:, -1]`（那是 pad 位置）；同一个 prompt 单独生成和放在 batch 里生成结果不同 |
| 6 | `v3/gpu-systems/flash-attention-triton` | Triton kernel 没有 mask 掉越界的 key（按 0 分数参与 softmax），N 不是 block 整数倍时误差 0.11 |
| 7 | `llm/Flash-Attention` | 用了 Triton 3.9 已删除的 `tl.make_block_ptr`，直接报错 |
| 8 | `v3/gpu-systems/fsdp` | 用“给梯度加随机噪声”来模拟数据并行，“与单卡梯度一致”的性质永远不可能成立 |
| 9 | `torch/hard/optimizers/muon` | Newton–Schulz 系数写成教科书的 (15/8, −5/4, 3/8)，**Muon 实际用的是 (3.4445, −4.7750, 2.0315)** |
| 10 | `torch/hard/seq-seq` | 所谓 “Bahdanau attention” 其实是老教程的 location attention，**根本没看 encoder 输出** |
| 11 | `v3/modern-architectures/mamba` | 数据生成器有 bug，**18% 的目标在输入里根本不存在**；讲解漏掉了“Δ 也依赖输入”这个 selectivity 的核心 |
| 12 | `v3/modern-architectures/knowledge-distillation` | 注释写的是 KL(student‖teacher)，`F.kl_div(log q_s, p_t)` 实际算的是 KL(teacher‖student) |
| 13 | `v3/modern-architectures/mixture-of-experts` | 辅助 loss 公式写错（N·mean(f·P) vs Switch 的 N·Σ f_i P_i）；描述承诺的 capacity 机制并未实现 |
| 14 | `v3/llm-decoding/top-p-sampling` | 题面要求“mask 掉累积概率超过 p 的 token”——**这正是丢掉越界 token 的 off-by-one bug** |
| 15 | `torch/easy/autoencoder` | 目标归一化到 [−1,1] 但 decoder 末尾是 Sigmoid（[0,1]），永远重建不了背景；latent 比输入还大 4 倍（没有瓶颈）；异常检测拿 batch 平均 loss 和随手定的 0.1 比，而且训练/测试里根本没有异常样本 |

---

## 逐题清单

图例：🔴 critical · 🟠 major · 🟢 minor/OK。“评分器”指 `python/torchleet/problems/<id>.py`。

### `torch/basic` — PyTorch 基础（7）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| lin-regression | 🟠 | 题目 stub 无法构造（缺 `super().__init__()`）；评分器接受 MLP/无 bias 模型 → 加仿射性与截距检查 |
| custom-dataset | 🟠 | 题面要求的类名与代码/评分器不一致；评分器接受 `(B,)` 目标（与 `(B,1)` 预测在 MSE 里静默广播成 `(B,B)`）；运行时在仓库根目录写出 `data.csv` |
| custom-activation | 🔴 | 答案的拟合图画的是 `w·x+b` 而不是模型的 `tanh(z)+z` 输出；评分器只检查“非仿射” |
| custom-loss (Huber) | 🟠 | 数学正确，但评分器要求的接口（`HuberLoss(delta)`、`forward(y_pred, y_true)`、mean）题面没写；补了 SmoothL1 关系和离群点演示 |
| custom-DNN | 🔴 | 号称“非线性”的数据其实是线性的；描述说是分类（实为回归）；评分器接受没有激活函数的堆叠 Linear；文件名 `custon` 拼写错误已改 |
| save-model | 🟠 | 描述承诺的 device mapping / 架构变更没有涉及 → 补 `map_location`/`weights_only` 和完整 checkpoint 恢复（含 optimizer 状态） |
| tensorboard | 🟠 | 评分器只“评”了题目已提供的模型，等于什么都没验证 → 删除评分器（改为不评分） |

### `torch/easy`（7）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| augmentation | 🔴 | 把随机训练增强也用在了测试集上；题面裁剪尺寸、归一化统计量与代码不一致 |
| autoencoder | 🔴 | 见上文 #15；改为单类训练 + 验证集 95 分位阈值 + 逐样本分数，MNIST AUROC 0.994 |
| benchmark | 🟠 | 计时代码已在题目里写好、没有 warm-up 和 synchronize；补 `torch.utils.benchmark` |
| cnn | 🟠 | 评分器在 train 模式下用 batch=1，会误杀带 BatchNorm 的正确模型 |
| cuda-amp | 🟠 | 用已弃用的 `torch.cuda.amp`、硬编码 `.cuda()`（无 GPU 直接崩）；补 fp16/bf16/GradScaler 原理 |
| quantize-lm | 🔴 | 量化调用直接写在题目里（核心任务被送分）；“一定提速”等说法不准确；补 scale/zero-point 数学、动态 vs 静态量化 |
| rnn | 🔴 | 题目测试把原始角度（≈12–16）喂给在 sin 值上训练的模型；“从零实现”却要求用 `nn.RNN` |

### `torch/medium`（5）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| 3dcnn | 🔴 | 输出 `[B,1,D,W,H]` 与 mask `[B,D,1,W,H]` 不对齐，Dice loss 静默广播成 `[B,D,D,W,H]`（报告的 “loss” 是 −0.50）；数据量会 OOM |
| alexnet | 🟠 | 架构正确；评分器接受任意 CNN → 检查 5 conv/3 FC、11×11 stride 4、dropout 只在训练时生效 |
| cnn-param-init | 🔴 | “Xavier 1/n、Kaiming 2/n” 没定义 n；评分器只用方阵测试，LeCun 初始化能冒充 Xavier |
| cnn-scratch | 🔴 | 模型实际用的是 `nn.Conv2d`/`nn.MaxPool2d` 而不是自己写的层；手写的四重循环层前向一次 28 秒 → 改为 `unfold` + `einsum` 向量化 |
| lstm | 🔴 | 默认隐状态是 `torch.randn`（同一输入两次结果不同）；N(0,1) 初始化让门饱和；评分器接受缺门的“LSTM” |

### `torch/hard`（8）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| custom-autograd | 🟠 | 题目最后一格引用不存在的 `model.linear` 直接崩；模型无法拟合截距；slope 没用 `save_for_backward` |
| optimizers/adam | 🟠 | 评分器只测单个参数张量（共享 step 计数、读错 lr 都能过）→ 与 `torch.optim.Adam` 多参数组逐步对齐 |
| optimizers/adamw | 🟠 | 题目的 weight-decay 测试根本没调用学生的类（永远 PASS）；ε 放进根号里也能过 |
| optimizers/muon | 🔴 | 见上文 #9；缩放因子、4D 卷积权重、测试格 NameError 也一并修复；与 `torch.optim.Muon` 对齐 |
| transformer | 🔴 | 要求支持 padding 但没有实现 mask（SST-2 demo 在 attend [PAD]）；评分器不检查 1/√d |
| seq-seq | 🔴 | 见上文 #10 |
| GAN | 🟠 | 答案正确，但题目直接给了完整训练循环（面试真正考的部分） |
| xai | 🟠 | 标题承诺 SHAP 但完全没有；解释的是被破坏的噪声图 |

### `llm/`（10）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| Byte-Pair-Encoder | 🟠 | 评分器接受“字符串替换跨符号合并”和“只合并第一次出现”两个经典 bug；GPT/Llama 实际用的是 byte-level BPE / SentencePiece |
| Sinusoidal-Positional-Embedding | 🟢 | 正确；措辞微调 |
| Rotary-Positional-Embedding | 🔴 | demo 静默广播出 `(50,2,50,64)` 还打印“范数保持：True”；题目 notebook 跑不起来；“RoPE 外推更好”是错的；评分器误杀 interleaved（GPT-J）写法 |
| RMS-Norm | 🟠 | 评分器接受“忘记乘 gamma”的实现（gamma 初始为 1） |
| Implement-Attention-from-Scratch | 🟠 | mask 约定（True=可见 还是 True=屏蔽）从未说明，而 `nn.MultiheadAttention` 恰好相反 |
| Multi-Head-Attention | 🔴 | 见上文 #1 |
| Grouped-Query-Attention | 🔴 | 同 #1；另：GQA 省的是 KV cache 和带宽，不是 attention FLOPs |
| Flash-Attention | 🔴 | 见上文 #7；题目里直接放着被评分的完整 PyTorch 实现 |
| SmolLM | 🔴 | 题目没有题面；答案每层都用新的 causal mask 覆盖传入的 attention mask（padding 被忽略）；测试加载仓库里不存在的权重文件。现在与 HuggingFace 真实 SmolLM-135M 的 logits 对齐到 1.4e-5 |
| Create-Embeddings-out-of-an-LLM | 🔴 | 代码根本跑不通（数据切片、tokenizer 没有 pad token、漏 import）；notebook 949 KB 全是打印的数据 |

### `v3/classical-ml`（4）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| softmax | 🟠 | 评分器太严（拒绝合法的 `exp(x − logsumexp(x))`）又太松（不测中间维度、原地修改输入、梯度） |
| kmeans | 🟠 | **评分器对正确代码有 ~30% 的随机失败率**（局部最优）→ best-of-10；达到 max_iters 时返回过期的分配 |
| knn | 🟠 | 评分器接受“忽略 k 的 1-NN”和 L1 距离 |
| logistic-regression | 🟠 | 评分器接受缺 1/N 的梯度、会溢出的 sigmoid、不加 eps 的 BCE |

### `v3/data-loading`（4）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| preloaded-dataset | 🟢 | 没按要求存成 tensor |
| streaming-dataset | 🔴 | 题目里一个 demo 格覆盖了全局变量 `y`，导致正确答案也过不了测试；“分片测试”只是 grep 源码里有没有 `get_worker_info`；两条关于 DataLoader 的说法是错的。新增评分器 |
| buffered-loading | 🟠 | 测试能被三种错误的 shuffle buffer 通过。新增评分器 |
| data-parallel-loading | 🔴 | rank 数 > 样本数时 padding 出错；题面声称各 rank 的分片不相交（有 padding 时为假），测试也没真正检查。现与 `DistributedSampler` 在 300 种配置下逐一相同。新增评分器 |

### `v3/modern-architectures`（10）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| contrastive-loss-clip | 🟠 | 评分器接受“沿错误维度归一化”和“温度平方”；补 CLIP 的 logit scale 上限 100 |
| 2d-positional-embeddings | 🟠 | 评分器接受普通 1D 编码；`d_model` 不是 4 的倍数时答案崩溃 |
| knowledge-distillation | 🔴 | 见上文 #12；评分器接受缺 T²、反向 KL |
| mixture-of-experts | 🔴 | 见上文 #13；评分器从未检查 dispatch/combine/aux loss（8 种错误实现都能过） |
| sliding-window-attention | 🔴 | 题目里放着一个**已填好且有 bug** 的答案；Mistral 的窗口是因果的（不是 ±w）；一条 hint 构造出全 True 的 mask |
| ddpm | 🟢 | 数学正确；补索引约定、方差选择、T=300 时 ᾱ_T≈0.05（不是纯噪声） |
| ddim-cfg | 🟠 | 时间步子序列步数不对且从不访问 T−1；题目和答案对“何时应用 CFG”不一致 |
| mamba | 🔴 | 见上文 #11；评分器接受 4 种错误的离散化/扫描 |
| swin-transformer | 🔴 | 见上文 #3；评分器还会误杀官方实现用的 −100 mask |
| vit-mae | 🟠 | 题目缺 `patchify` stub；评分器接受错误的 patchify 和泄露被 mask patch 的 encoder |

### `v3/alignment-training`（5）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| lora | 🔴 | 题面给的 A、B 形状无法相乘，merge 的 hint 是非法矩阵乘法；A 的初始化比 loralib/PEFT 大约 10 倍；评分器接受缺 α/r 的实现 |
| dpo | 🟠 | 没有 prompt/padding mask；评分器检测不到 log-prob 的错位 |
| ppo-rlhf | 🔴 | 见上文 #2；题目验证格对正确答案也会报广播错误 |
| grpo | 🔴 | 见上文 #2；补充 per-sequence vs token 级平均的约定 |
| gradient-checkpointing | 🟠 | 有 dropout 时重算用了不同的 mask；“内存对比”实际没测任何东西；O(√N) 的说法与代码不符 |

### `v3/llm-decoding`（4）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| temperature-sampling | 🔴 | hint 建议除以 T+1e-8（fp16 溢出成 NaN）→ T=0 定义为 greedy；一个测试是恒真式，另一个根本没调用学生函数 |
| top-k-sampling | 🟠 | k > vocab 时崩溃；评分器接受“top-k 内均匀采样”和“忽略温度” |
| top-p-sampling | 🔴 | 见上文 #14；p=0 时产生 NaN |
| beam-search | 🟠 | 描述承诺的长度归一化和 EOS 提前停止都没实现；2·beam > vocab 时崩溃 |

### `v3/llm-inference`（4）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| kv-cache | 🔴 | 见上文 #4；评分器现在也检查 prefill（`test_grader.py` 的参考实现原本也有同样的 bug，一并修复） |
| speculative-decoding | 🟠 | **算法本身是对的**（已验证输出分布等于目标模型）；但评分器接受“bonus token 从 draft 采样”的错误实现 → 加入精确的联合分布检验 |
| continuous-batching | 🔴 | `step()` 对每个请求单独前向，**根本没有 batch**；评分器检测不到 padding 错误 |
| inference-engine | 🔴 | 见上文 #5 |

### `v3/gpu-systems`（4）

| 题目 | | 原问题 → 修复 |
|---|---|---|
| triton-fused-softmax | 🟠 | 题目 stub 的注释里写着完整答案；非连续输入结果错误；评分器只在 CUDA 上用 2 的幂列数测试（mask 逻辑从未被测）→ 现在可用 `TRITON_INTERPRET=1` 在 CPU 上测 |
| flash-attention-triton | 🔴 | 见上文 #6；“N=8192 约 2 GB”应为 8.6 GB |
| fsdp | 🔴 | 见上文 #8；内存模型漏了优化器状态 |
| ring-attention | 🟢 | N 不能被设备数整除时崩溃；其余正确 |

---

## 实现质量方面的共性问题

除了“错”，原仓库还有不少“能跑但不是面试官想看的写法”，已在修复中一并改进：

- **该向量化的地方用了 Python 循环**：`cnn-scratch` 的四重循环卷积、`mixture-of-experts` 的逐 token 循环、`continuous-batching` 逐请求前向、`beam-search` 逐 beam `topk`。
- **数值稳定性**：`log(softmax)` 而非 `log_softmax`、sigmoid 溢出、BCE 不加 eps、T→0 时除以极小数。
- **train/eval 模式**：测试集用了训练增强、评估时没 `model.eval()`、BatchNorm 测试用 batch=1。
- **设备/dtype**：硬编码 `.cuda()` 或在 CPU 上建 buffer，导致 GPU/CPU 只能跑其一。
- **题目泄题**：`GAN`、`quantize-lm`、`sliding-window-attention`、`triton-fused-softmax`、`2d-positional-embeddings`、`knowledge-distillation` 的题目里直接写了答案。

## 仍然无法验证的部分

- **GPU 路径**：本次审计环境没有 GPU。Triton kernel 只在 `TRITON_INTERPRET=1`（CPU 解释器）下验证了正确性，没有在真实 GPU 上测性能和 fp16 容差；CUDA AMP、`torch.cuda.synchronize` 计时等只做了代码审查和 CPU 模拟。
- **真实数据集训练**：CIFAR-10 等大数据集没有完整训练到收敛（用 FakeData/合成数据验证流程）；autoencoder 在真实 MNIST 上跑过，SmolLM 和 Create-Embeddings 用真实 HuggingFace 模型跑过。
- **部分文献细节**凭记忆而非逐字核对原文（审计 agent 已在 notebook 中用较保守的措辞表述）。

## 独立复核结果

审计 agent 自己说“已验证”不算数。每一组改动（9 组审计修复 + 26 道新题）都交给**另一个没参与编写的 agent** 复核：它要自己写“写法不同但正确”的实现去确认不会被误杀，再写作者没想到的错误实现去确认会被拒绝，并逐条核对新增的说法（对照论文、PyTorch 源码、HuggingFace / torchvision / pycocotools 源码）。

**结论：没有发现被“改错”的答案代码；但复核又抓出了一批评分器漏洞和不准确的说法，全部已修复。** 这也说明单轮审计不够——第二轮复核是有必要的。

### 复核新发现并已修复的主要问题

| 范围 | 问题 | 性质 |
|---|---|---|
| `lstm` | 没有输入门的 “LSTM”（`C = f*C + g`）能通过评分器 | 评分器漏洞 |
| `seq-seq` | 完全没有 attention（context = encoder 输出的平均）的 decoder 能通过 | 评分器漏洞 |
| `adamw` | 审计“修好”的自测在空 stub 上仍然 PASS | 修复不完整 |
| `ppo-rlhf`、`grpo` | 比值写反的 k3 KL 估计（数值偏大约 46%）能通过梯度检查 → 加数值检查；补充 k3 的期望梯度是 ∇KL(π_ref‖π_θ) | 评分器漏洞 + 说明 |
| `gradient-checkpointing` | 题目要求的 dropout 重放（要求 5）从未被评分 | 评分器漏洞 |
| `inference-engine` | 审计新加的“batch 输出 = 单独输出”检查对约 2% 的**正确**模型永久失败（近似平局时的采样）→ 改为近贪心 + 平局检测，400 次 0 误判 | 误杀正确代码 |
| `kmeans`、`3dcnn`、`cnn-scratch`、`softmax`、`custom-loss` | L1 距离的 k-means、返回 Dice 系数而非 loss、绕过“禁止调用内置函数”的写法均能通过 | 评分器漏洞 |
| `ddpm`、`ddim-cfg` | 用 x₀ + 后验均值写的正确 `p_sample` 被 float32 抵消误杀；只在 s>1 时做 guidance 的错误写法能通过 | 误杀 + 漏洞 |
| `rotary-positional-embedding`、`smollm` | `Rotary` 的频率表从未被检查；忽略传入 attention mask 的注意力层能通过 | 评分器漏洞 |
| `llm/Flash-Attention` | Triton kernel 在 N 不是 block 整数倍时越界写；并把题目和答案拆成两个 notebook | 正确性 |
| `fsdp` | PyTorch `SHARD_GRAD_OP` 在 backward 后会重新分片参数，原“静止时参数完整”的内存模型描述的是 DeepSpeed ZeRO-2 | 说法错误 |
| `detection-map` | “原版 VOC devkit 用 IoU > t”是错的（MATLAB 版用 ≥，只有 Python 移植版用 >）；“低分预测会略微降低 mAP”是错的（all-point AP 不会下降） | 说法错误 |
| `decision-tree`、`pca`、`gmm-em`、`classification-metrics`（新题） | 用 entropy 分裂、新数据用自己的均值中心化、协方差用旧均值更新、macro 平均漏掉只出现在预测里的类——均能通过 | 评分器漏洞 |
| `normalization-layers`、`micrograd`（新题） | running stats 不在 `no_grad` 下更新（会进入计算图）能通过；在 `backward()` 里调大递归上限能绕过约束 | 评分器漏洞 |
| `vector-search`（新题） | 从不更新质心的 k-means 能通过；确定性检查本身不稳定 | 漏洞 + 不稳定 |
| `unet`、`dit-block`、`flow-matching`、`mixup-cutmix`、`vae`、`fid`（新题） | 评分器误杀允许的 dropout（U-Net）；放过题面明确排除的 embedding 布局和 erf 版 GELU（DiT）；3 个统计阈值偶尔会误杀正确实现（已放宽到约 6σ）；说法错误：β>1 的 β-VAE 仍是下界、FID 是 Fréchet 距离的**平方** | 误杀 + 漏洞 + 说法 |
| `speculative-decoding`、`vit-mae`、`continuous-batching` | 原评分器依赖学生自己模型的初始化/参数命名/padding token：用 GPT 式 std=0.02 初始化的**正确**实现会被判失败 → 改用评分器自带的测试模型、按梯度流向而非参数名检查 | 误杀正确代码 |
| `training-utilities`（新题） | 教科书“速度形式”的 SGD（`v = μv − ηg`）能通过——它只在学习率不变时与 `torch.optim.SGD` 等价，而原检查恰好全用常数学习率 → 加入变学习率轨迹检查 | 评分器漏洞 |
| `losses-from-scratch`（新题） | “clamp 版稳定 BCE 在 x=0 处斜率为 −1”在 torch 2.15 中不成立（实际为 0） | 说法错误 |
| `mlp-backprop`、`ring-allreduce`、`int8-quantization` 等（新题） | 多处说明性错误：`nn.Linear` 默认初始化方差是 He 的 1/6 而非 1/3；ZeRO-2 通信量与 DDP 相同；零点行和技巧属于 FBGEMM 而非 cuBLASLt | 说法错误 |
| `python/torchleet/runner.py` | 进度文件被截断时 `check()` 会对正确答案报 “grader crashed” → 容错读取 + 原子写入 | 工具 bug |

### 复核确认无误的关键点（节选）

- Swin 修正后的 shifted-window mask 与官方 `img_mask` 构造在 H=2…29、所有窗口大小和所有 shift 下完全一致。
- Beam search 在 300 个随机小模型上与暴力搜索结果完全一致。
- 采样题的分布检验（TV 距离）对正确实现的误判率 < 5×10⁻⁶；speculative decoding 的联合分布检验阈值距离噪声约 11σ。
- `data-parallel-loading` 与 `torch.utils.data.DistributedSampler` 在 15,288 种配置下逐一相同。
- SmolLM 的配置表与 HuggingFace 上的 `config.json` 一致（134,515,008 参数），加载真实权重后 logits 误差 1.4e-5、贪心生成文本完全相同。
- `memory-flops-estimation` 中所有数字（Llama-2-7B 6,738,415,616 参数、AdamW 107.8 GB、KV cache 0.5 MiB/token、ZeRO 各阶段每卡显存等）重新计算无误。
- int8 题的“float32 累加会被抓住”检查是确定性的，不会在其他 BLAS 上误杀正确代码。

### 最终状态

- 98 道题（72 道原题 + 26 道新题），其中 91 道自动评分：每一道的标准答案都通过自己的评分器（`scripts/validate_specs.py`）。7 道不评分的是纯演示/依赖外部资源的题（tensorboard、save-model、augmentation、cuda-amp、xai、create-embeddings、preloaded-dataset）。
- 评分器回归测试 `python/test_grader.py`：全部通过（新增了 RoPE interleaved、GQA tiled、KV-cache prefill 等用例）。
- `scripts/generate.py check`：0 个问题。
