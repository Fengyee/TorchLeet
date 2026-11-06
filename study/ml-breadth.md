# ML Breadth & Depth: Rapid Review (CV / Multimodal / Generative)

For Applied MLE, Research Scientist and Research Engineer loops. 126 questions in 9 sections. Every answer is meant to be said out loud in 30–90 seconds.

**How to use this file**

- Read only the question, answer it **out loud**, and then compare with the bullets. Mark anything you missed and come back to it the next day.
- A good spoken answer has four parts: **(1)** a one-line answer, **(2)** the mechanism or equation, **(3)** the trade-off and when you would use it, **(4)** one concrete number or example. Stop there and let the interviewer ask for more.
- `→ practice:` points to a hands-on problem in this repo. Writing the code is what makes the answer stick, and coding is your weakest area, so do those problems.
- When something is debated or depends on convention, the bullets say so. Saying "conventions differ, I'll use X" in an interview is a strength.
- The daily schedule is in [`README.md`](README.md). This file is the question bank.

**Contents**
1. [ML fundamentals](#1-ml-fundamentals) (Q1.1–Q1.17)
2. [Optimization & training dynamics](#2-optimization--training-dynamics) (Q2.1–Q2.14)
3. [CNNs & vision](#3-cnns--vision) (Q3.1–Q3.22)
4. [Transformers & attention](#4-transformers--attention) (Q4.1–Q4.11)
5. [Multimodal](#5-multimodal) (Q5.1–Q5.14)
6. [Generative models](#6-generative-models) (Q6.1–Q6.21)
7. [LLM essentials](#7-llm-essentials) (Q7.1–Q7.7)
8. [Systems & efficiency](#8-systems--efficiency) (Q8.1–Q8.12)
9. [Research-scientist specific](#9-research-scientist-specific) (Q9.1–Q9.8)
10. [Numbers to know](#10-numbers-to-know)

---

## 1. ML fundamentals

### Q1.1 What is the bias–variance trade-off, and how do you diagnose which one you have?
- With squared loss, $y = f(x) + \epsilon$ and $\mathrm{Var}(\epsilon) = \sigma^2$, the expected error over training sets $D$ splits exactly into three parts:
  $\mathbb{E}[(y - \hat f_D(x))^2] = \underbrace{(\mathbb{E}_D[\hat f_D(x)] - f(x))^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}_D(\hat f_D(x))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{noise}}$.
- **Bias** is error from wrong assumptions (underfitting). **Variance** is sensitivity to the particular training sample (overfitting). $\sigma^2$ cannot be reduced.
- **Diagnose:**
  - High *training* error means bias. Fix with a bigger model, better features, less regularization or longer training.
  - Low train error with high val error means variance. Fix with more data, augmentation, regularization, a smaller model or ensembling.
- **Caveats:**
  - The clean additive decomposition is for squared loss. It extends to Bregman divergences such as cross-entropy, but not to 0-1 loss.
  - Over-parameterized deep nets show **double descent**: test error can fall again past the interpolation threshold (Belkin et al. 2019; Nakkiran et al. 2019, "Deep Double Descent"). The classical U-curve is not the whole story.

### Q1.2 Walk through the main regularization techniques.
- **L2 (ridge), $\frac{\lambda}{2}\|w\|_2^2$:** shrinks all weights smoothly and matches a Gaussian prior. In linear regression it gives $(X^\top X + \lambda I)^{-1}X^\top y$, which is better conditioned.
- **L1 (lasso), $\lambda\|w\|_1$:** gives sparse solutions. The subgradient has constant magnitude $\lambda$, so small weights are pushed exactly to 0 (corners of the L1 ball). Matches a Laplace prior and does feature selection.
- **Dropout:** zero each unit with probability $p$ in training and scale survivors by $1/(1-p)$ ("inverted dropout"), so test time is the identity. Interpretations: an implicit ensemble of subnetworks, or preventing co-adaptation. Common in transformers; often 0 in large-scale LLM pretraining.
- **Early stopping:** stop when val loss stops improving. For linear least squares trained with GD it is approximately L2, with $\lambda \sim 1/(\eta t)$ (under a quadratic approximation).
- **Data augmentation:** encodes invariances and effectively enlarges the dataset. It is often the strongest regularizer in vision.
- **Others:** weight decay, stochastic depth (drop-path), label smoothing, mixup/CutMix, smaller models, ensembling, noise injection.

### Q1.3 Are L2 regularization and weight decay the same thing?
- **SGD: yes, up to rescaling.** $w \leftarrow w - \eta(\nabla L + \lambda w) = (1-\eta\lambda)w - \eta\nabla L$.
- **Adam: no.** If you add $\lambda w$ to the gradient, it gets divided by $\sqrt{\hat v} + \epsilon$. Weights with a large gradient history are then decayed *less*, so regularization is tangled up with the adaptive scaling.
- **AdamW** (Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019) decays the weights directly: $w \leftarrow w - \eta\left(\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \lambda w\right)$. PyTorch implements this as `p.mul_(1 - lr*wd)` followed by the Adam step, so the per-step decay is `lr*wd` and follows the LR schedule. (The paper multiplies $\lambda$ only by the schedule multiplier, not the base LR, so `wd` values do not transfer 1:1 between codebases or when you change the LR.)
- **In practice:**
  - AdamW is the default for transformers and ViTs. Typical $\lambda$ is about 0.05–0.1 for ViT/LLM recipes (CLIP used 0.2); PyTorch's default is 0.01.
  - Usually exclude biases and norm gains (often embeddings too) from decay. The usual rationale: they are few parameters, so there is little to regularize; pulling a norm gain toward 0 rather than its neutral value 1 just shrinks the layer's output. For a weight followed by a norm layer, the output is invariant to the weight's scale, so decay mainly shrinks $\|w\|$ and raises the *effective* LR ($\propto \eta/\|w\|^2$) rather than limiting capacity (van Laarhoven 2017; Zhang et al. 2019, "Three Mechanisms of Weight Decay Regularization").
- → practice: torch/hard/optimizers/adamw

### Q1.4 MLE vs MAP?
- MLE: $\hat\theta = \arg\max_\theta \sum_i \log p(x_i \mid \theta)$.
- MAP: $\arg\max_\theta \sum_i \log p(x_i \mid \theta) + \log p(\theta)$, i.e. MLE plus a log-prior regularizer.
- **Which prior gives which penalty:**
  - Gaussian prior $\mathcal N(0, \tau^2 I)$ gives L2. For regression with Gaussian noise $\sigma^2$, this is ridge with $\lambda = \sigma^2/\tau^2$.
  - Laplace prior gives L1.
- MAP is a point estimate (the mode). It is not invariant to reparameterization and is not full Bayesian inference (no posterior uncertainty). As $n \to \infty$ the prior washes out and MAP approaches MLE.
- Loss ↔ noise model: MSE is the Gaussian-noise MLE (estimates the conditional mean); L1 is the Laplace-noise MLE (conditional median).

### Q1.5 Why is cross-entropy the "right" loss for classification?
- It is the MLE loss. Model $p_\theta(y \mid x) = \mathrm{softmax}(z)_y$; the dataset NLL $-\sum_i \log p_\theta(y_i \mid x_i)$ is exactly cross-entropy with one-hot targets.
- Equivalently, $\mathbb E_{p_{data}}[-\log p_\theta] = H(p_{data}) + \mathrm{KL}(p_{data}\|p_\theta)$. The entropy term is constant, so MLE minimizes **forward KL**, which is mode-covering.
- **Why not MSE on probabilities?**
  - Through a sigmoid/softmax, MSE picks up an extra $\sigma'(z)$ factor, so gradients vanish when the model is saturated *and wrong*.
  - CE gives the clean $\partial L/\partial z = p - y$.
  - CE is also convex in $w$ for logistic regression; MSE-on-sigmoid is not.
- Binary CE is the Bernoulli NLL. MSE is the Gaussian NLL with fixed variance.
- **Why `CrossEntropyLoss` takes logits, not probabilities** (a common follow-up): it fuses log-softmax and NLL, $\log p_y = z_y - \mathrm{logsumexp}(z)$ computed with max-subtraction, so it never takes the log of a probability that underflowed to 0 (float32 `exp` underflows about 104 below the max), and its gradient is simply $\mathrm{softmax}(z) - \mathrm{onehot}(y)$. Feeding softmax outputs applies softmax twice: the "logits" are confined to $[0, 1]$, so the loss is bounded away from 0 (at least $\log(1 + (C-1)/e) \approx 1.46$ for $C = 10$) and gradients are weak. It trains, but badly.

### Q1.6 Derive the logistic-regression gradient.
- Setup: $p = \sigma(z)$, $z = w^\top x + b$, $L = -[y\log p + (1-y)\log(1-p)]$.
- Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$ gives $\partial L/\partial z = p - y$. Hence $\nabla_w L = (p - y)\,x$ and $\partial L/\partial b = p - y$. For a batch: $\nabla_w = \frac{1}{n}X^\top(p - y)$.
- Softmax regression: $\partial L/\partial z = p - \mathrm{onehot}(y)$, and $\nabla_W = (p - y)x^\top$.
- The loss is convex: Hessian $X^\top S X$ with $S = \mathrm{diag}(p(1-p)) \succeq 0$. There is no closed form, so use GD or Newton/IRLS. On linearly separable data, $\|w\| \to \infty$ unless you regularize.
- Numerics: compute the loss from logits (log-sum-exp; `BCEWithLogitsLoss` / `cross_entropy`). Never take `torch.log` of an already-computed sigmoid or softmax.
- → practice: v3/classical-ml/logistic-regression, v3/classical-ml/softmax

### Q1.7 SVMs and hinge loss in a nutshell?
- Soft-margin primal: $\min_{w,b} \frac12\|w\|^2 + C\sum_i \max(0,\ 1 - y_i(w^\top x_i + b))$ with $y \in \{-1, +1\}$. The geometric margin width is $2/\|w\|$.
- Hinge loss is exactly zero beyond the margin, so the solution depends only on the **support vectors**.
- The dual touches the data only through inner products. That allows the **kernel trick** (RBF, polynomial), but kernel SVMs scale poorly in $n$ (Gram matrix is $O(n^2)$).
- Hinge vs logistic: both are convex surrogates for the 0-1 loss. Logistic gives probabilities; hinge does not (use Platt scaling). Large $C$ means a harder margin: lower bias, higher variance.

### Q1.8 Decision trees; Random Forest vs gradient-boosted trees?
- **Tree:** greedy recursive splits that maximize impurity reduction. Classification uses Gini $1-\sum_k p_k^2$ or entropy $-\sum_k p_k\log p_k$; regression uses variance/MSE. Trees are interpretable, need no feature scaling and handle mixed types, but deep trees have high variance.
- **Random Forest:**
  - Bagging (bootstrap samples) plus a random feature subset at each split (e.g. $\sqrt d$ for classification).
  - This decorrelates deep, low-bias trees, and averaging them **reduces variance**.
  - Parallel, robust to hyperparameters, and gives a free out-of-bag estimate.
- **GBDT:**
  - Additive model $F_m = F_{m-1} + \nu h_m$, where each shallow tree $h_m$ is fit to the **negative gradient** of the loss (pseudo-residuals; for MSE these are just the residuals).
  - It **reduces bias** but is sequential and sensitive to learning rate, number of trees and depth (use early stopping).
  - XGBoost adds second-order (gradient + Hessian) split scoring and regularized leaf weights. LightGBM adds histogram binning and leaf-wise growth.
- On tabular data, a GBDT is usually the strongest baseline and often beats deep nets.
- → practice: v4/ml-fundamentals/decision-tree, v4/ml-fundamentals/gradient-boosting

### Q1.9 k-means vs GMM, and how does EM work?
- **k-means:**
  - Minimizes $\sum_i \|x_i - \mu_{c(i)}\|^2$ with Lloyd's algorithm (assign to nearest centroid, recompute means).
  - The objective never increases, so it converges, but only to a local optimum. Init matters (use k-means++).
  - Hard assignments; implicitly assumes spherical, similar-sized clusters.
- **GMM:** $p(x) = \sum_k \pi_k\,\mathcal N(x \mid \mu_k, \Sigma_k)$, fit by EM.
  - **E-step:** $\gamma_{ik} = \dfrac{\pi_k\,\mathcal N(x_i \mid \mu_k,\Sigma_k)}{\sum_j \pi_j\,\mathcal N(x_i \mid \mu_j,\Sigma_j)}$.
  - **M-step:** $N_k = \sum_i\gamma_{ik}$, $\mu_k = \frac{1}{N_k}\sum_i\gamma_{ik}x_i$, $\Sigma_k = \frac{1}{N_k}\sum_i\gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$, $\pi_k = N_k/n$.
- **Properties of EM:** it maximizes a lower bound (ELBO), and each iteration never decreases the log-likelihood. It still finds only local optima. Watch for degenerate components ($\Sigma_k \to 0$ on a single point); add $\epsilon I$ to the covariances and compute in log-space.
- **Link between them:** k-means is the limit of EM for a GMM with shared isotropic covariance $\sigma^2 I$ as $\sigma \to 0$ (soft assignments become hard).
- Choosing $k$: elbow plot or silhouette for k-means; BIC/AIC for GMMs.
- → practice: v3/classical-ml/kmeans, v4/ml-fundamentals/gmm-em

### Q1.10 Explain PCA and its relation to SVD.
- PCA finds orthonormal directions of maximum variance. Equivalently, the rank-$k$ projection with minimum reconstruction error.
- **Computation:**
  - Center $X$ ($n \times d$). The covariance is $C = X^\top X/(n-1) = V\Lambda V^\top$.
  - With SVD $X = U\Sigma V^\top$, the principal axes are the columns of $V$, the eigenvalues are $\lambda_i = \sigma_i^2/(n-1)$, and the scores are $XV_k = U_k\Sigma_k$.
- The explained-variance ratio is $\lambda_i / \sum_j \lambda_j$; pick $k$ to keep, say, 95%. Eckart–Young: the truncated SVD is the best rank-$k$ approximation (Frobenius/spectral norm).
- **Gotchas:**
  - You must center the data, and should standardize if units differ.
  - Use SVD rather than forming $X^\top X$ (squares the condition number).
  - Component signs are arbitrary.
  - PCA is linear only; for nonlinear structure use kernel PCA or autoencoders, or t-SNE/UMAP for visualization.
- → practice: v4/ml-fundamentals/pca

### Q1.11 What is the curse of dimensionality?
- Volume grows exponentially. Covering $[0,1]^d$ at resolution $\epsilon$ needs about $\epsilon^{-d}$ points, so the data needed for local methods explodes.
- Distances concentrate. For many i.i.d.-like dimensions, $(d_{max} - d_{min})/d_{min} \to 0$, so "nearest" neighbors are barely nearer than anything else. Most of a ball's volume sits near its surface, and random vectors are nearly orthogonal.
- Hurts kNN, kernel density estimation, RBF kernels and grid search.
- Deep learning still works because real data lies near low-dimensional manifolds and models carry inductive biases (locality, weight sharing, compositionality). Mitigate with learned features, dimensionality reduction and regularization.
- → practice: v3/classical-ml/knn

### Q1.12 How do you handle class imbalance?
- **Metric first.** Accuracy is misleading. Use PR-AUC, recall at fixed precision, or per-class/macro F1.
- **Data level:** oversample the minority (SMOTE for tabular data), undersample the majority, or use class-balanced sampling (e.g. repeat-factor sampling for long-tailed detection such as LVIS).
- **Loss level:** class weights (inverse frequency, or "effective number of samples"), focal loss, or logit adjustment (add $\tau\log\pi_y$ to logits during training).
- **Decision level:** tune the threshold on validation for the actual cost trade-off.
- **Caveat:** resampling or reweighting changes the implied class priors, so predicted probabilities become miscalibrated. Correct for the prior or recalibrate.
- Often the best fix is more minority data: targeted collection, active learning or synthetic data.

### Q1.13 Precision, recall, F1; ROC-AUC vs PR-AUC?
- **Definitions:**
  - Precision $= \frac{TP}{TP+FP}$; Recall $=$ TPR $= \frac{TP}{TP+FN}$; FPR $= \frac{FP}{FP+TN}$.
  - $F_1 = \frac{2PR}{P+R}$ (harmonic mean, punishes lopsided P/R). $F_\beta$ weights recall $\beta$ times as much as precision.
- **ROC curve:** TPR vs FPR across thresholds. AUC = P(a random positive scores above a random negative); 0.5 is random. ROC is **insensitive to class prior**: with a huge TN count, FPR stays tiny, so ROC-AUC can look great while precision is terrible.
- **PR curve:** the random baseline equals the prevalence. It depends on prevalence and is far more informative when positives are rare (fraud, detection, retrieval).
- **Multiclass averaging:** micro pools counts (dominated by frequent classes); macro averages per-class metrics (all classes equal).
- Both AUCs are threshold-free *ranking* metrics; deployment still needs a threshold.
- → practice: v4/ml-fundamentals/classification-metrics

### Q1.14 What is calibration and how do you measure and fix it?
- A model is calibrated if $P(\text{correct} \mid \text{confidence} = p) = p$: of all predictions made at 80% confidence, 80% are right.
- **ECE:** bin predictions by confidence ($M$ bins; Guo et al. used 15) and compute $\mathrm{ECE} = \sum_m \frac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|$. Visualize with a reliability diagram.
- **ECE caveats:** it depends on the binning, covers only the top label, and can be small for a useless model. Pair it with proper scoring rules (NLL, Brier score).
- Modern deep nets tend to be **overconfident** (Guo et al. 2017, "On Calibration of Modern Neural Networks").
- **Fixes:**
  - Temperature scaling: divide logits by a scalar $T$ fit on validation NLL. One parameter, and argmax/accuracy are unchanged.
  - Also Platt scaling, isotonic regression, and label smoothing or mixup (these often reduce overconfidence).
- Calibration matters for thresholds, risk-sensitive decisions, abstention and ensembling.

### Q1.15 What is data leakage? Give examples.
- Leakage is information unavailable at prediction time reaching training or validation. Metrics look great offline and collapse in production.
- **Examples:**
  - *Target leakage:* a feature derived from the label.
  - *Train–test contamination:* duplicates or near-duplicates; the same patient, user or video split across train and test; augmented copies of one image in both.
  - *Preprocessing fit on all data:* scaler, PCA, vocabulary or feature selection fitted before the split.
  - *Temporal leakage:* random splits on time series.
  - *Test-set tuning:* choosing hyperparameters on the test set.
  - *Benchmark contamination:* the test set appears in web-scale pretraining data (big issue for LLMs/VLMs).
- **Prevention:**
  - Split first, by group or time, and fit preprocessing inside each fold.
  - Deduplicate (perceptual hashes or embedding similarity for images).
  - Audit features that are suspiciously predictive, and treat "too good to be true" results as bugs until proven otherwise.

### Q1.16 Cross-validation: which kind, when?
- **k-fold** (k = 5 or 10): train on $k-1$ folds, validate on the held-out fold, average. Lower-variance than a single split. For deep nets it is usually too expensive, so use one val split plus an untouched test set.
- **Variants:**
  - *Stratified:* preserves class ratios in each fold.
  - *Group k-fold:* a group (patient, video, scene) never straddles folds.
  - *Time-series:* forward-chaining, i.e. train on the past and validate on the future.
  - *Leave-one-out:* nearly unbiased but expensive and high-variance.
- **Nested CV:** the inner loop selects hyperparameters and the outer loop estimates generalization. This avoids optimistic bias from tuning on the evaluation folds.
- Touch the test set once, at the end.

### Q1.17 Train/val/test distribution shift: types, detection, mitigation?
- **Types:**
  - *Covariate shift:* $p(x)$ changes but $p(y \mid x)$ stays fixed (new camera, lighting, hospital).
  - *Label shift:* $p(y)$ changes but $p(x \mid y)$ stays fixed (prevalence changes).
  - *Concept shift:* $p(y \mid x)$ changes (the label definition changes).
  - Also subpopulation shift and spurious correlations (e.g. backgrounds).
- **Detection:**
  - Monitor input and embedding statistics, the prediction distribution and confidence.
  - Train a **domain classifier** to separate train from prod data; if it can, there is shift.
  - Label a small production sample.
- **Mitigation:**
  - Best: collect and label target data.
  - Importance weighting $w(x) = p_{test}(x)/p_{train}(x)$, with the ratio estimated by a domain classifier.
  - Prior correction for label shift.
  - Augment for known nuisance factors.
  - Domain adaptation, or test-time adaptation (e.g. re-estimate BN statistics).
  - Start from robust pretrained foundation models.
  - Group DRO for spurious correlations.
- Make the val set match deployment, and keep a stress/OOD test suite.

> **Say it in one breath**
> - Bias is underfitting and variance is overfitting. Diagnose from the train/val gap and fix with capacity on one side, data/regularization on the other. Double descent complicates the classical picture.
> - Cross-entropy is MLE (forward KL). L2 is a Gaussian prior and L1 a Laplace prior. With Adam, use decoupled weight decay (AdamW), not an L2 term.
> - Choose metrics to fit the base rate: PR-AUC for rare positives, ROC-AUC is prior-invariant. Check calibration (ECE, NLL) and fix it with temperature scaling.
> - Most failures that look like modeling problems are really data problems: leakage, group or temporal splits, and distribution shift. Split by group or time and dedup before you train.

---

## 2. Optimization & training dynamics

### Q2.1 SGD vs momentum vs Adam vs AdamW?
- **SGD:** $\theta \leftarrow \theta - \eta g$. Noisy and LR-sensitive; zig-zags in ill-conditioned valleys.
- **Momentum** (PyTorch form): $v \leftarrow \mu v + g$, $\theta \leftarrow \theta - \eta v$ with $\mu \approx 0.9$. Averaging gradients damps oscillations and speeds up consistent directions. Effective LR is about $\eta/(1-\mu)$. Nesterov evaluates the gradient at a look-ahead point.
- **Adam:**
  - $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ and $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$.
  - Bias correction: $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ (corrects the zero init).
  - Update: $\theta \leftarrow \theta - \eta\,\hat m_t/(\sqrt{\hat v_t} + \epsilon)$.
  - Defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. LLM pretraining usually lowers $\beta_2$ to 0.95 for stability. Vision recipes vary: 0.95 (MAE), 0.98 (CLIP ViTs), the default 0.999 (DeiT).
- **Why Adam for transformers:** each parameter gets roughly the same step size regardless of its gradient scale, which handles very different curvature across embeddings, attention, norms and MLPs. SGD with momentum stays competitive on CNN classification and sometimes generalizes better there (debated).
- **AdamW** = Adam + decoupled weight decay (Q1.3). Adam stores 2 extra states per parameter.
- Newer optimizers: Lion, Shampoo/SOAP, Muon (orthogonalized momentum for 2-D weight matrices).
- → practice: torch/hard/optimizers/adam, torch/hard/optimizers/adamw, torch/hard/optimizers/muon

### Q2.2 Why LR warmup? Why cosine decay?
- **Warmup** ramps the LR linearly from about 0 over the first few hundred to few thousand steps (about 1–5% of training). It lets you use a higher peak LR. Reasons:
  - At init, gradients are large and badly scaled.
  - Adam's $v$ is estimated from few samples, so the adaptive LR has high variance (the RAdam argument).
  - Post-LN transformers often diverge without warmup.
  - Large-batch SGD (linear scaling rule) needs it.
- **Cosine:** $\eta_t = \eta_{min} + \frac12(\eta_{max} - \eta_{min})(1 + \cos(\pi t/T))$. A high LR early explores; annealing late lowers the gradient-noise floor and settles into a minimum. Downside: you must fix $T$ up front.
- **Alternatives:**
  - Step decay (classic ResNet: ÷10 every 30 epochs).
  - Linear decay.
  - Inverse-sqrt with warmup (original Transformer).
  - Warmup–stable–decay (WSD): a constant LR, then a short decay, so you can branch off checkpoints at any budget.
  - 1-cycle.
- Much of the final loss drop comes from annealing, so compare runs at the *end* of their schedules, not mid-way.

### Q2.3 How should LR change with batch size?
- A larger batch means lower gradient noise (variance $\propto 1/B$), fewer steps and better hardware utilization.
- **Linear scaling rule (SGD):** multiply the LR by $k$ when the batch grows $k\times$, and add warmup. Goyal et al. 2017 ("Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour") trained ResNet-50 at batch 8k. Intuition: one big step ≈ $k$ small steps if the gradient changes slowly.
- **Adam:** square-root scaling ($\eta \propto \sqrt B$) is often suggested (with some SDE-based theory). Treat every rule as a starting point and re-tune.
- **Critical batch size** (McCandlish et al. 2018, "An Empirical Model of Large-Batch Training"): beyond it, doubling $B$ no longer halves the steps, so compute is wasted. Very large batches can also hurt generalization (the sharp-minima argument is debated). LARS/LAMB layer-wise trust ratios help here.
- Gradient accumulation gives a large effective batch under memory limits, but BatchNorm still sees only the micro-batch.

### Q2.4 Gradient clipping: how and why?
- **Clip by global norm:** $g \leftarrow g \cdot \min(1, c/\|g\|_2)$ over all parameters. It keeps the gradient direction; $c = 1.0$ is typical for transformers. Clip-by-value changes the direction and is rarer.
- It stops rare huge gradients (bad batches, loss spikes, exploding RNN gradients) from wrecking the weights. Standard in RNNs, transformers and RL.
- With fp16 AMP, **unscale before clipping** (`scaler.unscale_(opt)`).
- Log the grad norm. Frequent clipping means the LR is too high or training is unstable, and norm spikes often precede loss spikes.

### Q2.5 Vanishing and exploding gradients: why, and fixes?
- Backprop multiplies Jacobians: $\frac{\partial L}{\partial h_0} = J_1^\top J_2^\top\cdots J_L^\top\frac{\partial L}{\partial h_L}$. Singular values consistently below 1 make gradients vanish exponentially with depth; above 1 they explode. RNNs apply the same matrix at every time step, which makes this worse.
- Saturating activations make it worse too: $\sigma'(z) \le 0.25$, and tanh is flat in its tails.
- **Fixes:**
  - ReLU-family activations.
  - Variance-preserving init (Q2.6).
  - Normalization layers.
  - Residual connections.
  - Gating: the LSTM's additive cell state.
  - Gradient clipping (for explosion).
  - Truncated BPTT; orthogonal init for RNNs.
- Diagnose with per-layer gradient-norm plots. Early layers with gradients near 0 means vanishing.
- → practice: torch/easy/rnn, torch/medium/lstm

### Q2.6 Derive Xavier and Kaiming initialization.
- **Goal:** keep the activation variance (forward) and gradient variance (backward) constant across layers.
- **Single layer:** $y = \sum_{i=1}^{n_{in}} w_i x_i$, with i.i.d. zero-mean $w$ independent of $x$, gives $\mathrm{Var}(y) = n_{in}\,\mathrm{Var}(w)\,\mathbb E[x^2]$.
- **Xavier/Glorot** (linear or tanh regime, zero-mean $x$, so $\mathbb E[x^2] = \mathrm{Var}(x)$):
  - Forward needs $n_{in}\mathrm{Var}(w) = 1$; backward needs $n_{out}\mathrm{Var}(w) = 1$.
  - The compromise is $\mathrm{Var}(w) = \frac{2}{n_{in} + n_{out}}$; uniform version $U[-a, a]$ with $a = \sqrt{6/(n_{in} + n_{out})}$.
- **Kaiming/He** (ReLU):
  - If the previous pre-activation is symmetric with zero mean, $x = \mathrm{ReLU}(\cdot)$ has $\mathbb E[x^2] = \frac12\mathrm{Var}(\text{pre-act})$.
  - So $\mathrm{Var}(y_l) = \frac12 n_{in}\mathrm{Var}(w)\mathrm{Var}(y_{l-1})$, which needs $\mathrm{Var}(w) = 2/n_{in}$ (fan-in mode; use fan-out to preserve the backward pass).
- For conv layers, $n_{in} = C_{in}k_hk_w$.
- PyTorch's `nn.Linear` / `nn.Conv` default is $U(-1/\sqrt{n_{in}}, 1/\sqrt{n_{in}})$ (via `kaiming_uniform_(a=√5)`). That gives variance $1/(3n_{in})$, smaller than He init.
- **Modern practice:** start residual blocks near identity.
  - Zero-init the last BN $\gamma$ in each ResNet block.
  - GPT-2 scales residual output projections by $1/\sqrt{N_{res\,layers}}$.
  - DiT uses adaLN-Zero.
- Normalization reduces sensitivity to init but does not remove it.
- → practice: torch/medium/cnn-param-init

### Q2.7 BatchNorm vs LayerNorm vs GroupNorm vs RMSNorm: when and why?

| Norm | Statistics over | Batch-dependent? | Typical use |
|---|---|---|---|
| BatchNorm | $(N, H, W)$ per channel | Yes; running mean/var at eval | CNN classification with a decent per-GPU batch (≥16–32) |
| LayerNorm | all features of one sample/token | No | Transformers, RNNs, ConvNeXt |
| GroupNorm | groups of channels × $(H, W)$ per sample | No | Detection/segmentation and diffusion U-Nets (small batches); often 32 groups |
| InstanceNorm | each channel × $(H, W)$ per sample (= GN with $G = C$) | No | Style transfer |
| RMSNorm | features of a token; no mean subtraction, usually no bias | No | LLaMA-style LLMs; increasingly in ViTs/DiTs, often as QK-norm |

- **Formulas:**
  - Standard norms compute $\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$, then apply $\gamma\hat x + \beta$.
  - RMSNorm computes $x/\sqrt{\frac1d\sum_i x_i^2 + \epsilon}\cdot g$. It is cheaper and empirically as good (Zhang & Sennrich 2019).
- **BN statistics, precisely:** in training each channel is normalized with the mini-batch mean and the *biased* variance (divide by $n = N \cdot H \cdot W$), and running estimates are updated as `running = (1 − m)·running + m·batch_stat` with $m = 0.1$ (PyTorch's momentum weights the *new* batch — the opposite of Keras). `running_var` is updated with the *unbiased* variance ($\times\, n/(n-1)$). In eval, BN normalizes with the running estimates and updates nothing. Mixing these conventions is a classic porting bug.
- **BN pitfalls:**
  - Noisy statistics with small batches.
  - Train/eval discrepancy; forgetting `model.eval()` is a classic bug.
  - Mixes information across samples (contrastive models can "cheat"; MoCo's shuffle-BN).
  - Needs SyncBN for distributed small batches.
  - Awkward for variable-length sequences and autoregressive decoding.
  - Bonus: BN can be folded into the preceding conv at inference.
- **Why LN rather than BN in transformers** (a common follow-up):
  - BN's statistics pool over the batch *and* all sequence positions, so they depend on padding, sequence length and batch composition. In NLP transformers these batch statistics fluctuate strongly, which hurts training (Shen et al. 2020, "PowerNorm: Rethinking Batch Normalization in Transformers").
  - In a causal LM, training-time batch statistics include *future* tokens (a small leak), and generation with batch 1, one token at a time, must fall back to running statistics, which is a train/inference mismatch.
  - LN normalizes each token on its own, so training and inference compute the same function for any batch size, sequence length or decoding step, with no cross-device sync.
- **Why norms help:** they allow higher LRs and make optimization less sensitive to init. The original "reduces internal covariate shift" explanation is disputed; Santurkar et al. 2018 argue BN mainly **smooths the loss landscape**. Say it is debated.
- → practice: v4/dl-fundamentals/normalization-layers, llm/RMS-Norm

### Q2.8 Why do residual connections work?
- $y = x + F(x)$ gives $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$. The identity path carries gradients across depth without attenuation (He et al. 2015, "Deep Residual Learning for Image Recognition").
- Identity is easy to represent ($F \to 0$), so extra layers should not hurt. This solved the **degradation problem**: deeper plain nets had *higher training error*, which is an optimization failure, not overfitting.
- Other views: iterative refinement; an implicit ensemble of many shorter paths (Veit et al. 2016); a smoother loss landscape (Li et al. 2018 visualizations). In transformers, the residual stream is the shared bus that every block reads from and writes to.
- Keep the identity path clean (pre-activation ResNets, pre-LN transformers) and initialize residual branches small or zero.

### Q2.9 Mixed precision: fp16 with loss scaling vs bf16?
- **Formats** (exponent / mantissa bits):
  - fp32: 8 / 23.
  - fp16: 5 / 10. Max 65504, smallest normal ≈ $6.1\times10^{-5}$, subnormals down to ≈ $6\times10^{-8}$.
  - bf16: 8 / 7. Same *range* as fp32, much less precision (≈ 2–3 significant decimal digits).
- **Recipe** (Micikevicius et al., "Mixed Precision Training", 2017/ICLR 2018): fp32 master weights, matmuls/convs in 16-bit, reductions (softmax, norms, loss) and the optimizer update in fp32.
- **fp16 issue:** small gradients underflow to zero.
  - Loss scaling: multiply the loss by $S$, backprop, then unscale the gradients by $1/S$ before the step.
  - Dynamic scaling (`GradScaler`) halves $S$ and skips the step on inf/NaN, and grows $S$ after a run of clean steps.
  - Activations (e.g. attention logits) can also overflow fp16's max.
- **bf16:** no loss scaling needed, simpler and more stable. Needs Ampere+ or TPUs. It is the default for large-model training today.
- **Wins:** about half the activation memory and tensor-core throughput (A100: 312 TFLOP/s bf16 dense vs 19.5 fp32 and 156 TF32).
- FP8 is used on H100-class GPUs. The original recipe is E4M3 for weights/activations and E5M2 for gradients, with per-tensor scaling. Newer recipes use E4M3 throughout with fine-grained block scaling (DeepSeek-V3; MXFP8 on Blackwell).
- → practice: torch/easy/cuda-amp

### Q2.10 Label smoothing: what and why?
- Replace the one-hot target with $y^{LS} = (1-\epsilon)\,\mathrm{onehot}(y) + \epsilon/K$ ($\epsilon \approx 0.1$) and use CE with soft targets. Introduced in Inception-v3 (Szegedy et al., "Rethinking the Inception Architecture", 2016).
- It stops logits growing without bound to reach probability 1. It regularizes, often improves accuracy, and often improves calibration.
- Müller, Kornblith & Hinton 2019 ("When Does Label Smoothing Help?"): LS tightens class clusters in the penultimate layer but erases information about inter-class similarity, so **LS-trained teachers distill worse**.
- In PyTorch: `nn.CrossEntropyLoss(label_smoothing=0.1)`. Mixup/CutMix already produce soft labels, so check before stacking them.
- → practice: v4/dl-fundamentals/losses-from-scratch

### Q2.11 EMA of weights: what and why?
- Keep a shadow copy: $\theta_{EMA} \leftarrow \beta\,\theta_{EMA} + (1-\beta)\,\theta$, with $\beta \approx 0.999$–$0.9999$. Evaluate and deploy the EMA weights.
- It averages roughly the last $1/(1-\beta)$ iterates, which cancels SGD noise. You get smoother, often better models, and it is **standard for diffusion models and GANs** (big sample-quality gains).
- The same mechanism is the momentum/teacher encoder in MoCo, BYOL and DINO.
- **Details:** warm up $\beta$ (or bias-correct) early, handle BN buffers, keep the copy in fp32, and budget one extra copy of the params.
- → practice: v4/dl-fundamentals/training-utilities

### Q2.12 Your model won't train. Walk me through debugging.
1. **Look at the data after the full pipeline.** Check images and labels after augmentation and normalization, RGB vs BGR, CHW vs HWC, label alignment after shuffling, class balance and NaNs.
2. **Check the loss at init.** It should be about $\ln K$ for balanced $K$-way softmax (2.30 for $K = 10$, 6.91 for $K = 1000$). If not, suspect output scaling or init.
3. **Overfit one batch** (or about 10 examples) to near-zero loss with augmentation and regularization off. If you can't, there is a wiring bug.
4. **Check the plumbing.**
   - `zero_grad()` and `train()`/`eval()` modes.
   - Loss applied to logits, not probabilities (no double softmax).
   - Target dtype and shape: silent broadcasting such as `(B,1)` vs `(B,)` is a classic bug.
   - Stray `.detach()`, frozen params, wrong reduction.
5. **Sweep the LR** over orders of magnitude (LR range test). If loss explodes, go lower; if it is flat, go higher. Add warmup.
6. **Inspect gradients.** Look at per-layer grad norms and the update-to-weight ratio (about $10^{-3}$ is a rule of thumb). Look for dead ReLUs and `None` grads. Gradient-check custom ops against finite differences.
7. **Numerics.** Find the NaN/inf source: `log(0)`, division, fp16 overflow. Use `torch.autograd.detect_anomaly` and retry in fp32/bf16.
8. **Simplify, then add back.** Start from a known-good baseline and hyperparameters on a single GPU with no AMP and a fixed seed. Add components back one at a time.
9. **Verify the evaluation.** Check the metric code, that val preprocessing matches train minus augmentation, and that nothing leaks.
- See also Karpathy's "A Recipe for Training Neural Networks" (2019 blog post).
- → practice: v4/dl-fundamentals/debug-training

### Q2.13 How does backprop work? Give the gradients of a linear layer.
- Reverse-mode autodiff: the forward pass records a graph and caches intermediates. The backward pass applies the chain rule in reverse topological order, accumulating $\bar v = \partial L/\partial v$.
- Backward costs about 2× forward, so a training step is about 3× forward. It is efficient because there is one scalar output and many inputs.
- **Linear layer** $Y = XW^\top + b$ ($X$: $B \times d_{in}$, $W$: $d_{out} \times d_{in}$):
  - $\bar X = \bar Y W$, $\bar W = \bar Y^\top X$, $\bar b = \sum_{\text{batch}}\bar Y$.
  - ReLU: $\bar x = \bar y \odot \mathbb 1[x > 0]$.
  - Softmax + CE (mean over batch): $\bar z = (p - y)/B$.
- Sanity rule: a gradient has the same shape as its tensor.
- Activations must be stored for backward, and they dominate training memory (hence checkpointing, Q8.4).
- → practice: v4/dl-fundamentals/mlp-backprop, v4/dl-fundamentals/micrograd, torch/hard/custom-autograd

### Q2.14 Your large transformer run shows loss spikes or divergence. What do you do?
- **Immediate:** roll back to the last good checkpoint and skip the offending data batches. Lower the peak LR or lengthen warmup, and confirm gradient clipping is on.
- **Likely causes and fixes:**
  - *LR too high for the scale:* lower it, or use $\mu$P-style scaling.
  - *Adam $\beta_2$ too high:* use 0.95 so $v$ adapts faster to sudden gradient changes. Also check $\epsilon$: when $v$ is tiny, a very small $\epsilon$ lets updates blow up.
  - *Attention-logit growth:* add QK-norm or logit soft-capping.
  - *Output-logit drift:* add a z-loss on $\log Z$.
  - *fp16 overflow:* switch to bf16.
  - *Bad data:* corrupted samples, extremely long sequences, duplicates.
  - *Embedding/LN instabilities:* check norm placement and that the final LN exists.
- **Monitor** per-layer grad norms, update norms, attention entropy, activation max values and loss by data source.

> **Say it in one breath**
> - Adam normalizes the step size per parameter. AdamW decouples weight decay from that scaling. Use warmup plus cosine (or WSD), clip by global norm at about 1.0, and re-tune the LR when the batch size changes.
> - Initialize to preserve variance (Kaiming $2/n_{in}$ for ReLU), normalize (BN for CNNs with big batches, LN/RMSNorm for transformers, GN for small batches), and use residuals to keep an identity gradient path.
> - Prefer bf16 to fp16: same range as fp32, so no loss scaling. Keep fp32 master weights and reductions.
> - To debug, check the init loss is $\ln K$, overfit one batch, visualize data after the pipeline, sweep the LR and watch per-layer gradient norms.

---

## 3. CNNs & vision

### Q3.1 Conv layer: output size, parameter count, FLOPs?
- **Output size:** $O = \left\lfloor\frac{I + 2P - D(K-1) - 1}{S}\right\rfloor + 1$. With dilation $D = 1$ this is $\lfloor (I + 2P - K)/S \rfloor + 1$. "Same" padding for odd $K$ and $S = 1$ is $P = (K-1)/2$.
- **Params:** $C_{out}\cdot K_hK_w\cdot C_{in}/g + C_{out}$ (bias), where $g$ is the number of groups.
- **MACs:** $K_hK_w\cdot(C_{in}/g)\cdot C_{out}\cdot H_{out}W_{out}$ per image. FLOPs ≈ 2 × MACs, but papers often say "FLOPs" when they mean MACs, so **state your convention**.
- **Worked example** (ResNet stem): input 224², conv 7×7, $S = 2$, $P = 3$, 3→64 channels, no bias.
  - Output $= \lfloor (224 + 6 - 7)/2 \rfloor + 1 = 112$.
  - Params $= 64\cdot 7\cdot 7\cdot 3 = 9{,}408$.
  - MACs $= 9408 \cdot 112^2 \approx 118$M.
- **Transposed conv:** $O = (I-1)S - 2P + D(K-1) + \text{output\_padding} + 1$.
- → practice: torch/medium/cnn-scratch, torch/easy/cnn

### Q3.2 How do you compute the receptive field? Effective vs theoretical?
- **Recurrence:** $r_l = r_{l-1} + (k_l - 1)\,j_{l-1}$ and $j_l = j_{l-1}s_l$, with $r_0 = j_0 = 1$. The jump $j$ is the product of earlier strides. A dilated kernel acts like size $D(k-1) + 1$.
- Two 3×3 convs (stride 1) see 5×5; three see 7×7, using $27C^2$ params instead of $49C^2$ and with more nonlinearities (the VGG argument).
- Ways to grow the RF: depth, stride/pooling, dilation (DeepLab), large kernels, or attention (global from layer 1).
- The **effective** receptive field is much smaller than the theoretical one and roughly Gaussian, dominated by central pixels (Luo et al. 2016).
- Why it matters: large objects and context-dependent decisions need an RF covering the object plus its context.

### Q3.3 Pooling vs strided conv? Global average pooling?
- **Max/avg pooling:** parameter-free downsampling with a little local translation invariance; max keeps the strongest response.
- **Strided conv:** learnable downsampling (all-convolutional nets, ResNet downsample blocks). ViT's patch embedding is a conv with kernel = stride.
- **Aliasing:** both alias, so CNNs are not truly shift-invariant and a 1-pixel shift can flip a prediction. Blurring before subsampling helps (Zhang 2019, "Making Convolutional Networks Shift-Invariant Again").
- Convs are translation-*equivariant* (up to stride and boundary effects). Invariance comes from pooling and GAP.
- **Global average pooling** at the head replaces big FC layers (NiN, ResNet): fewer params, any input size, and it enables CAM.

### Q3.4 Explain ResNet and the bottleneck block.
- **Basic block** (ResNet-18/34): 3×3 conv → BN → ReLU → 3×3 conv → BN, add the shortcut, then ReLU.
- **Bottleneck** (ResNet-50/101/152): 1×1 reduce ($C \to C/4$) → 3×3 at $C/4$ → 1×1 expand ($\to C$), plus the shortcut.
  - At $C = 256$: about $256\cdot64 + 9\cdot64^2 + 64\cdot256 \approx 70$K params.
  - Two full-width 3×3 convs would be $2\cdot9\cdot256^2 \approx 1.18$M.
- When shapes change, the shortcut becomes a projection: 1×1 conv + BN, with stride 2 when the block downsamples.
- ResNet-50 has about 25.6M params and about 4 G multiply-adds at 224² (often quoted as "4 GFLOPs"; the paper's table says 3.8 G, and torchvision's v1.5, with the stride on the 3×3 conv, is 4.1 G).
- **Tricks:** zero-init the last BN $\gamma$ per block; ResNet-D downsampling; pre-activation ordering. Variants: Wide ResNet and ResNeXt (grouped convs).
- "ResNet strikes back" (Wightman et al. 2021): a modern training recipe alone lifts ResNet-50 to about 80% top-1. Recipes matter as much as architectures.

### Q3.5 Depthwise-separable convolutions: cost and caveats?
- **Cost:**
  - Standard conv: $K^2 C_{in}C_{out}HW$ MACs.
  - Depthwise (one $K\times K$ filter per channel): $K^2 C_{in}HW$.
  - Pointwise 1×1: $C_{in}C_{out}HW$.
  - Ratio to standard: $\frac{1}{C_{out}} + \frac{1}{K^2}$, roughly 8–9× fewer MACs for $K = 3$.
- **Where used:**
  - Xception and MobileNet-v1.
  - MobileNet-v2's inverted residual: expand 1×1 → depthwise 3×3 → linear 1×1 projection, with no ReLU on the narrow output.
  - EfficientNet: MBConv blocks plus compound scaling of depth, width and resolution.
  - ConvNeXt: 7×7 depthwise.
- **Caveat:** FLOPs ≠ latency. Depthwise convs have low arithmetic intensity (memory-bound), so real GPU speedups are far smaller than the FLOP savings.

### Q3.6 Data augmentation: mixup, CutMix, RandAugment?
- **Baseline:** random-resized crop, horizontal flip, color jitter at train time; resize plus center crop at test time.
- **mixup** (Zhang et al. 2017):
  - $\tilde x = \lambda x_i + (1-\lambda)x_j$, $\tilde y = \lambda y_i + (1-\lambda)y_j$, $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$.
  - $\alpha \approx 0.2$ in the original ImageNet setup; ViT/DeiT recipes use about 0.8.
  - Encourages linear behavior between examples, which gives smoother decision boundaries and better calibration.
- **CutMix** (Yun et al. 2019):
  - Paste a random box from $x_j$ into $x_i$.
  - Labels mix in proportion to area: $\tilde y = \lambda y_i + (1-\lambda)y_j$ with $\lambda = 1 - \frac{\text{box area}}{\text{image area}}$, recomputed after clipping the box to the image.
  - Keeps natural local statistics and helps localization.
- **RandAugment** (Cubuk et al. 2019): apply $N$ ops sampled uniformly from a list, all at a shared magnitude $M$. Two hyperparameters instead of AutoAugment's expensive policy search. TrivialAugment uses one op at a random magnitude.
- **Detection and segmentation:** geometric augs must transform the boxes and masks too. Mosaic is used in YOLO.
- Augmentation strength should scale with data hunger: ViTs need strong augmentation (the DeiT recipe).
- Avoid augmentations that change the label (flipping text or digits, recoloring a color-defined class).
- → practice: torch/easy/augmentation, v4/vision/mixup-cutmix

### Q3.7 One-stage vs two-stage detectors? Anchors? DETR?
- **Two-stage** (Faster R-CNN, Ren et al. 2015): the backbone feeds a Region Proposal Network, which scores anchors and outputs hundreds to about 2k RoIs. RoIPool (RoIAlign since Mask R-CNN) crops features for each RoI, and a head classifies and refines the box. Historically more accurate (especially on small objects) but slower.
- **One-stage** (YOLO, SSD, RetinaNet, FCOS): dense class + box predictions at every location in one pass. Faster, but faces extreme foreground/background imbalance; RetinaNet's **focal loss** closed the accuracy gap.
- **Anchors:** preset boxes at several scales and aspect ratios per location, matched to ground truth by IoU.
  - RPN: >0.7 positive, <0.3 negative. RetinaNet: ≥0.5 positive, <0.4 negative.
  - Regression targets: $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$, $t_w = \log(w/w_a)$, $t_h = \log(h/h_a)$.
  - Downsides: many hyperparameters and heavy imbalance.
- **Anchor-free:** FCOS (per-pixel distances to the four box sides, plus centerness) and CenterNet (center heatmap + size).
- **DETR** (Carion et al. 2020): a set-prediction transformer with $N$ object queries and a Hungarian-matching loss; no anchors, no NMS (Q3.22). It converges slowly; Deformable DETR and the DINO *detector* fixed that.
- Open-vocabulary detection (text-conditioned): OWL-ViT, Grounding DINO.

### Q3.8 IoU and NMS: definition, algorithm, issues?
- **IoU** $= \frac{|A \cap B|}{|A \cup B|}$. For boxes, the intersection width is $\max(0, \min(x_2, x_2') - \max(x_1, x_1'))$ (same for height), and union = area$_A$ + area$_B$ − intersection.
- **NMS** (per class): sort boxes by score, keep the top one, drop the rest whose IoU with it exceeds a threshold (≈0.5–0.7), repeat. $O(n^2)$ in the worst case.
- **Issues:**
  - It suppresses genuinely overlapping objects (crowds), and the threshold is a hyperparameter.
  - Soft-NMS decays scores by IoU instead of deleting boxes.
  - You must choose class-aware or class-agnostic NMS.
  - DETR-style models avoid NMS entirely.
- **Why NMS is non-differentiable:** it is a hard, discrete post-processing step (sort, then keep/drop by a thresholded IoU comparison). Its output, a set of indices, is piecewise constant in the scores and coordinates, so the gradient is zero almost everywhere and undefined at the switch points — no learning signal reaches duplicate removal. Detectors are therefore trained one-to-many (several anchors/queries per object are positives) and rely on a heuristic the loss never sees. DETR instead trains one-to-one with Hungarian matching, so duplicates are penalized by the loss itself (Q3.22).
- **IoU losses:** GIoU $= \mathrm{IoU} - \frac{|C \setminus (A\cup B)|}{|C|}$, where $C$ is the smallest enclosing box; it gives gradients even without overlap. DIoU and CIoU add center-distance and aspect-ratio terms.
- → practice: v4/vision/iou-nms

### Q3.9 What does an FPN do?
- **The problem it solves:** deep features are semantically strong but low-resolution; shallow features are high-resolution but weak. Small objects suffer.
- **FPN** (Lin et al. 2017):
  - Takes backbone stages C2–C5 and builds a top-down path.
  - At each level, upsample the coarser map ×2 (nearest), add a 1×1 lateral conv of the same-resolution backbone feature, then apply a 3×3 conv to smooth.
  - The outputs P2–P5 all have 256 channels.
- Small objects are detected on fine levels and large ones on coarse levels. Faster R-CNN assigns a RoI to level $k = \lfloor k_0 + \log_2(\sqrt{wh}/224) \rfloor$ with $k_0 = 4$. RetinaNet uses P3–P7.
- Variants: PANet (extra bottom-up path), BiFPN (EfficientDet), ViTDet's simple pyramid built from a plain ViT.

### Q3.10 Focal loss: formula and intuition?
- $\mathrm{FL}(p_t) = -\alpha_t(1 - p_t)^\gamma\log p_t$, where $p_t$ is the predicted probability of the true class. RetinaNet uses $\gamma = 2$, $\alpha = 0.25$ (Lin et al. 2017, "Focal Loss for Dense Object Detection").
- $(1 - p_t)^\gamma$ down-weights easy, well-classified examples (the roughly 100k easy background anchors) so the gradient concentrates on hard ones. With $\gamma = 0$ it is α-weighted CE.
- Subtlety: $\alpha = 0.25$ weights positives *less*, because $\gamma$ already suppresses easy negatives and the two interact.
- **Prior init:** set the classifier bias to $b = -\log\frac{1-\pi}{\pi}$ with $\pi = 0.01$, so the flood of background loss doesn't destabilize early training. Normalize the loss by the number of positive anchors.

### Q3.11 How is detection mAP computed?
- **Per class:**
  1. Sort all detections by confidence.
  2. Walk down the list. A detection is a TP if its IoU ≥ threshold with a not-yet-matched ground-truth box of that class (each GT is matched at most once); duplicates count as FPs. (This is COCO-style matching. The VOC devkit takes the highest-IoU GT even if it is already matched, which makes the detection an FP, and its Python port uses IoU > t. Conventions differ, so state yours.)
  3. Record precision and recall at each rank.
- **AP** is the area under the PR curve after making precision monotone, $p_{interp}(r) = \max_{r' \ge r}p(r')$. VOC2007 used 11-point interpolation, VOC2010+ all-point, COCO 101-point.
- **mAP** is the mean over classes.
- **COCO "AP":** also averaged over 10 IoU thresholds, 0.50:0.05:0.95. COCO additionally reports AP50, AP75 and AP$_S$/AP$_M$/AP$_L$ (area <32², 32²–96², >96² px).
- It rewards ranking and localization. It does not measure calibration or deployment-threshold behavior.
- → practice: v4/vision/detection-map

### Q3.12 Segmentation: U-Net, Dice loss, mIoU?
- **Tasks:**
  - *Semantic:* a class per pixel.
  - *Instance:* a mask per object. Mask R-CNN = Faster R-CNN + mask head + RoIAlign.
  - *Panoptic:* things and stuff together.
  - *Promptable:* SAM, Kirillov et al. 2023 (Q3.21).
- **U-Net** (Ronneberger, Fischer & Brox 2015): encoder–decoder whose skip connections **concatenate** encoder features into the decoder at each resolution, recovering the spatial detail lost to downsampling. It is also the classic diffusion backbone.
- **Dice:** $\frac{2|A\cap B|}{|A| + |B|}$, which is F1 over pixels.
  - Soft Dice loss: $1 - \frac{2\sum pg + \epsilon}{\sum p + \sum g + \epsilon}$.
  - Robust to foreground/background imbalance (medical imaging).
  - Often summed with CE: CE gives stable gradients, Dice targets overlap.
- **IoU (Jaccard):** $\frac{|A\cap B|}{|A\cup B|}$, related by $\mathrm{Dice} = \frac{2J}{1+J}$.
- **mIoU:** per-class IoU $= \frac{TP}{TP + FP + FN}$, usually accumulated over the whole dataset (confusion matrix), then averaged over classes.
- Other architectures: DeepLab (atrous convs + ASPP) and Mask2Former (mask classification with queries).
- → practice: v4/vision/unet

### Q3.13 Upsampling: transposed conv vs interpolation?
- **Bilinear interpolation:** a weighted average of the 4 neighbors; fixed and smooth. The `align_corners` flag changes the coordinate mapping, and a mismatch between training and export is a classic bug.
- **Transposed conv:** learnable upsampling. When $K$ is not divisible by $S$, kernel overlap is uneven and produces **checkerboard artifacts** (Odena et al. 2016, Distill). Choosing $K$ divisible by $S$ removes the uneven overlap but can still produce artifacts; the robust fix is resize-then-conv (nearest or bilinear upsampling, then a conv).
- **Pixel shuffle** (sub-pixel conv) is common in super-resolution.
- RoIAlign uses bilinear sampling to avoid RoIPool's coordinate quantization.
- → practice: v4/vision/bilinear-interpolation

### Q3.14 Explain ViT. Why is it data-hungry?
- **Architecture** (Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021):
  - Split the image into $P\times P$ patches; 224/16 gives 14×14 = 196 tokens.
  - Flatten each patch and project linearly; this equals a Conv2d with kernel = stride = $P$.
  - Prepend a learnable [CLS] token and add learned 1-D position embeddings.
  - Run a pre-LN transformer encoder and put an MLP head on [CLS] (or on the mean-pooled tokens).
- **Sizes:** ViT-B/16 has 12 layers, $d = 768$, 12 heads, MLP 3072, about 86M params. ViT-L/16 has 24 layers, $d = 1024$, 16 heads, about 307M params.
- **Inductive bias:** CNNs build in locality, translation equivariance and hierarchy. ViT has only patchification, and spatial structure must be learned through the position embeddings.
  - So it is less sample-efficient. With the original recipe, large ViTs trailed BiT ResNets when pretrained on ImageNet-1k alone, roughly caught up with ImageNet-21k pretraining, and clearly overtook them with JFT-300M.
  - DeiT (Touvron et al. 2021) closed the gap on ImageNet-1k with strong augmentation, regularization and a distillation token.
- **Cost:** quadratic in the number of tokens, so smaller patches or higher resolution get expensive fast. For higher-resolution fine-tuning, keep the patch size and **2-D interpolate the position embeddings**.
- High-norm "artifact" tokens appear in large ViTs; adding a few register tokens fixes them (Darcet et al. 2023, "Vision Transformers Need Registers").
- → practice: v3/modern-architectures/vit-mae, v3/modern-architectures/2d-positional-embeddings

### Q3.15 What's the idea behind Swin Transformer?
- (Liu et al. 2021) A hierarchical ViT:
  - Patch size 4 and four stages.
  - Patch merging between stages: concatenate 2×2 neighbors and apply a linear layer, which halves resolution and doubles channels.
  - The result is a multi-scale feature pyramid like a CNN, so it drops into FPN-based detectors and segmenters.
- **Window attention** runs within non-overlapping $M\times M$ windows ($M = 7$), so cost is linear in image size: about $2M^2hwC$ vs $2(hw)^2C$ for the attention term.
- **Shifted windows:** alternate layers shift the partition by $\lfloor M/2 \rfloor$ so information crosses window borders. It is implemented efficiently with a cyclic shift plus an attention mask. Adds a relative position bias.
- **Trade-off:** more complex, and less friendly to MAE-style token dropping. Later, plain ViTs with simple pyramids (ViTDet) proved competitive.
- → practice: v3/modern-architectures/swin-transformer

### Q3.16 ConvNeXt in 30 seconds?
- (Liu et al. 2022, "A ConvNet for the 2020s") Take ResNet-50 and apply Swin-era choices one at a time:
  - **Training recipe:** AdamW, long schedules, mixup/CutMix/RandAugment, stochastic depth, label smoothing.
  - **Macro design:** 3:3:9:3 stage ratio, patchify stem (4×4, stride 4), separate downsampling layers.
  - **Block design:** depthwise conv with wider channels, inverted bottleneck, 7×7 kernels, GELU, fewer activations and norms, LayerNorm instead of BN.
- The result matches Swin at similar FLOPs. Much of ViTs' edge came from the recipe and macro design, not from attention itself.
- ConvNeXt V2 adds FCMAE masked pretraining (sparse convs) and Global Response Normalization.

### Q3.17 Contrastive self-supervised learning: SimCLR and MoCo?
- **Idea:** learn invariance to augmentations by pulling together two augmented views of the same image (positives) and pushing apart other images (negatives).
- **SimCLR** (Chen et al. 2020):
  - Strong augmentation; random crop + color distortion are crucial.
  - Encoder → MLP projection head; the representation used downstream is taken *before* the head.
  - NT-Xent loss (InfoNCE with cosine similarity and temperature $\tau$), using the other $2(N-1)$ views in the batch as negatives.
  - Needs large batches (e.g. 4096) and long training.
- **MoCo** (He et al. 2020):
  - A queue of negatives from past batches plus a momentum (EMA) key encoder that keeps the keys consistent, so you get many negatives with small batches.
  - v2 added SimCLR's MLP head and augmentations; v3 targets ViTs and drops the queue in favor of large batches.
- **No negatives:** BYOL (online predictor + EMA target), SimSiam (stop-gradient), Barlow Twins and VICReg (variance/decorrelation terms). Each needs some mechanism to avoid collapse.
- **Evaluate** with linear probe, k-NN, fine-tuning and low-shot accuracy.
- → practice: v3/modern-architectures/contrastive-loss-clip

### Q3.18 Explain MAE. Why mask 75%?
- (He et al. 2022, "Masked Autoencoders Are Scalable Vision Learners")
  - Randomly mask 75% of patches.
  - The ViT encoder sees **only the visible 25%**. Two reasons (MAE's mask-token ablation, Table 1c, ViT-L): *compute* — the encoder processes a quarter of the tokens, cutting training FLOPs 3.3× (2.8–4.1× wall-clock); and *accuracy* — [MASK] tokens never occur in real images, so an encoder that sees them in pretraining has a pretrain–deploy input gap (putting mask tokens in the encoder dropped linear probing from 73.5 to 59.6 and fine-tuning from 84.9 to 84.2).
  - A lightweight decoder takes the encoded tokens plus mask tokens (with positions) and reconstructs pixels.
  - The loss is MSE on masked patches only, with per-patch normalized pixel targets.
- **Why such a high ratio:** images are spatially redundant. At low ratios the task is solvable by interpolation; 75% forces holistic, semantic reasoning.
- Excellent for fine-tuning; weaker for linear probing than contrastive methods (features are less linearly separable). Needs minimal augmentation (crop only).
- **Relatives:**
  - BEiT (predict dVAE tokens), SimMIM, iBOT, data2vec.
  - I-JEPA predicts *representations*, not pixels.
  - VideoMAE masks 90–95%.
- → practice: v3/modern-architectures/vit-mae

### Q3.19 Explain DINO / DINOv2.
- **DINO** (Caron et al. 2021, "Emerging Properties in Self-Supervised Vision Transformers") is self-distillation with no labels:
  - The student and teacher share an architecture; the teacher is an **EMA of the student**.
  - **Multi-crop:** the teacher sees 2 global crops; the student sees those plus several small local crops, which teaches local-to-global correspondence.
- **Loss:** cross-entropy $-\sum P_t\log P_s$ between the teacher's and student's softmax over $K$ prototype dimensions (65,536 in the paper).
  - The teacher output is **centered** (subtract a running mean) and **sharpened** (low temperature: $\tau_t$ about 0.04–0.07 vs $\tau_s = 0.1$).
  - Centering stops one dimension dominating; sharpening stops collapse to uniform.
- **Emergent properties:** [CLS] attention maps segment objects, and features are strong for k-NN.
- **DINOv2** (Oquab et al. 2023): curated data (LVD-142M), DINO + iBOT masked-token loss, KoLeo regularizer. Strong general-purpose *frozen* features for depth, segmentation and retrieval, widely used as a feature extractor. DINOv3 (Siméoni et al. 2025) scales to a 7B ViT and adds Gram anchoring to stop dense features degrading over long training.
- **vs CLIP:** DINO features are stronger for dense and geometric tasks; CLIP features carry language-aligned semantics.

### Q3.20 How do you explain what a CNN looked at (Grad-CAM)?
- **Grad-CAM** (Selvaraju et al. 2017):
  - For class $c$ and conv feature maps $A^k$, compute channel weights $\alpha_k^c = \frac{1}{Z}\sum_{i,j}\frac{\partial y^c}{\partial A^k_{ij}}$ (global-average-pooled gradients).
  - The map is $\mathrm{ReLU}\left(\sum_k\alpha_k^cA^k\right)$, upsampled to the image.
- **Alternatives:** vanilla gradients/saliency, Integrated Gradients, SHAP, occlusion sensitivity, attention rollout for ViTs.
- **Caveat:** saliency methods can be insensitive to the model's weights (Adebayo et al. 2018, "Sanity Checks for Saliency Maps"). Use them for debugging spurious features, not as proof.
- → practice: torch/hard/xai

### Q3.21 What is SAM, and how does promptable segmentation work?
- (Kirillov et al. 2023, "Segment Anything") **Task:** given an image and a prompt (foreground/background points, a box or a rough mask; free-form text only as a proof of concept), return a valid mask, even when the prompt is ambiguous.
- **Architecture,** split so that one heavy pass serves many cheap prompts:
  - *Image encoder:* an MAE-pretrained ViT-H/16 at 1024×1024 (windowed attention plus a few global blocks) produces a 64×64×256 embedding, computed once per image.
  - *Prompt encoder:* points and boxes become positional encodings plus learned type embeddings. A mask prompt is embedded with convs and added to the image embedding.
  - *Mask decoder:* two transformer blocks with two-way cross-attention (prompt tokens ↔ image embedding), then upsampling and a dot product with per-mask output tokens. Prompt encoder + decoder take about 50 ms on CPU in a browser, so clicking feels interactive.
- **Ambiguity:** a click on a shirt could mean the shirt, the person or a button. SAM predicts 3 masks (roughly whole, part, subpart) with a predicted IoU each for ranking, and backpropagates only the lowest-loss mask.
- **Training:** focal + Dice loss (20:1), with simulated interactive prompting (11 prompt rounds per mask).
- **Data engine → SA-1B** (11M images, 1.1B masks): assisted-manual annotation, then semi-automatic, then fully automatic (prompt with a 32×32 point grid, keep confident and stable masks, NMS). 99.1% of the released masks are fully automatic. The model-in-the-loop data engine is as much the contribution as the model.
- **Limits:** masks carry no class labels; it can miss fine structures, and boundaries are less crisp than specialist methods; the ViT-H encoder is heavy (distilled variants such as MobileSAM and EfficientSAM exist).
- **Successors:**
  - SAM 2 (Ravi et al. 2024) handles video with a streaming memory of past frames (memory attention) and a hierarchical Hiera encoder.
  - SAM 3 (Carion et al. 2025) adds *promptable concept segmentation*: a short noun phrase or an image exemplar returns masks and identities for **all** matching instances.
  - Before SAM 3, a common pattern was to feed an open-vocabulary detector's boxes to SAM as prompts (Grounded-SAM).

### Q3.22 How does DETR's set-prediction loss (Hungarian matching) work?
- **Setup:** the decoder turns $N = 100$ learned object queries into $N$ predictions. Each is a class distribution that includes a "no object" class ∅, plus a box $(c_x, c_y, w, h)$ normalized to $[0, 1]$. The ground truth is padded with ∅ to $N$ entries.
- **Step 1, matching** (no gradient): find the permutation $\hat\sigma$ that minimizes $\sum_i \mathcal L_{match}(y_i, \hat y_{\sigma(i)})$ with the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`, $O(N^3)$). For a real object, $\mathcal L_{match} = -\hat p_{\sigma(i)}(c_i) + \mathcal L_{box}$. It uses the probability, not the log-probability, so the class term is on the same scale as the box term.
- **Step 2, loss on the matched pairs:** $-\log\hat p_{\hat\sigma(i)}(c_i)$ for every slot (∅ terms down-weighted by 10 for class imbalance), plus for real objects $\mathcal L_{box} = \lambda_{L1}\|b - \hat b\|_1 + \lambda_{iou}\mathcal L_{GIoU}$ with $\lambda_{L1} = 5$, $\lambda_{iou} = 2$, normalized by the number of objects. GIoU is added because L1 on boxes is scale-dependent.
- **Why no NMS:** the assignment is one-to-one, so a second prediction of an already-matched object is trained toward ∅. Queries see each other through self-attention, so the model learns to de-duplicate. Classic detectors assign many anchors to each object and remove duplicates with NMS afterwards.
- **Auxiliary losses:** the same heads and loss after every decoder layer.
- **Weaknesses:** slow convergence (300–500 epochs on COCO vs 36–109 for the paper's Faster R-CNN baselines), unstable matching early in training, and weak small-object AP from single-scale features (−5.5 AP$_S$ but +7.8 AP$_L$ vs a Faster R-CNN-FPN of the same size).
- **Fixes:** Deformable DETR (multi-scale deformable attention, about 10× fewer epochs); DAB-/DN-DETR (queries as anchor boxes; denoising queries that skip matching); the DINO detector (contrastive denoising, mixed query selection); one-to-many auxiliary matching (H-DETR, Group DETR) for more positives; RT-DETR for real time.
- → practice: v4/vision/iou-nms (GIoU and NMS)

> **Say it in one breath**
> - Conv output size is $\lfloor (I + 2P - D(K-1) - 1)/S \rfloor + 1$, params are $K^2C_{in}C_{out}/g$ (+bias), and MACs are weights × $H_{out}W_{out}$. Depthwise-separable convs cut FLOPs about 8–9× but are memory-bound.
> - Detection uses IoU matching, box regression, focal loss for imbalance, FPN for scale, NMS to dedup and COCO AP at IoU 0.5:0.95. Segmentation uses U-Net skips, Dice/CE and mIoU.
> - DETR replaces anchors + NMS with one-to-one Hungarian matching (class + L1 + GIoU). SAM runs a heavy image encoder once and a light prompt decoder per click, and outputs 3 masks to handle ambiguity.
> - ViT is patches + transformer with weak inductive bias, so it is data-hungry. Fixes are pretraining scale or the DeiT recipe; Swin adds hierarchy and windows; ConvNeXt shows the recipe matters as much as the architecture.
> - Self-supervised vision has two families: contrastive/self-distillation (SimCLR, MoCo, DINO) gives linearly separable semantic features, while masked modeling (MAE) gives great fine-tuning features cheaply by encoding only 25% of patches.

---

## 4. Transformers & attention

### Q4.1 Write scaled dot-product attention. Why divide by $\sqrt{d_k}$?
- $\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$, with $Q: n\times d_k$, $K: m\times d_k$, $V: m\times d_v$ and an additive mask $M$. Each output is a convex combination of values, weighted by query–key similarity.
- **Why $\sqrt{d_k}$:** if the components of $q$ and $k$ are independent with zero mean and unit variance, $q\cdot k$ has **variance $d_k$**. Large logits saturate the softmax (nearly one-hot), so gradients become tiny. Scaling restores unit variance.
- Related tricks: QK-norm or a learnable scale. Implement softmax with max-subtraction for numerical stability.
- → practice: llm/Implement-Attention-from-Scratch

### Q4.2 Why multi-head attention? What are the parameter counts?
- $\mathrm{head}_i = \mathrm{Attn}(XW_i^Q, XW_i^K, XW_i^V)$ with $d_k = d_{model}/h$, and $\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^O$.
- **Why multiple heads:** each head can attend to different positions or relations in its own subspace, whereas a single head must average them into one distribution. FLOPs and params are the same as one full-width head.
- **Implementation:** one fused QKV projection, then reshape to $(B, h, n, d_k)$. Many heads can be pruned after training (Michel et al. 2019, "Are Sixteen Heads Really Better than One?").
- **Param counts:**
  - Attention: $4d^2$ (Q, K, V, O), plus biases.
  - MLP with 4× expansion: $8d^2$.
  - So a block is about **$12d^2$**.
  - SwiGLU MLPs use 3 matrices with hidden size about $\frac83 d$ to keep about $8d^2$.
- → practice: llm/Multi-Head-Attention, torch/hard/transformer

### Q4.3 What is the complexity of self-attention, and when does it dominate?
- With $n$ tokens and width $d$: scores $QK^\top$ cost $O(n^2d)$, the weighted sum costs $O(n^2d)$, and projections + MLP cost $O(nd^2)$. Naively materialized scores need $O(hn^2)$ memory per layer.
- **Forward FLOPs per layer** (2 FLOPs/MAC):
  - Linear layers: about $24nd^2$ ($8nd^2$ for QKVO, $16nd^2$ for MLP).
  - Attention matmuls: about $4n^2d$.
  - Attention dominates only when **$n \gtrsim 6d$** (full attention; roughly double that with causal-skipping kernels). For ViT-B ($n = 197$, $d = 768$), the MLPs and projections dominate.
- **Reducing it:**
  - FlashAttention: same forward FLOPs (backward recomputes), $O(n)$ memory.
  - Windowed/sliding attention (Swin, Mistral-style), sparse attention.
  - Linear/kernel attention, low-rank (Linformer).
  - SSMs (Mamba).
  - Token merging/pruning for ViTs (ToMe).
- When decoding with a KV cache, attention costs $O(nd)$ per new token per layer.
- → practice: v3/modern-architectures/sliding-window-attention, v3/modern-architectures/mamba

### Q4.4 Causal vs padding masks; what can go wrong?
- **Causal:** position $i$ attends only to $j \le i$. Add $-\infty$ (in practice a large negative) to the upper triangle before softmax. This allows parallel teacher-forced training of autoregressive models.
- **Padding (key) mask:** in a batch of variable-length sequences, mask pad *keys*. Outputs at pad *query* positions are garbage and are dropped in the loss (`ignore_index`).
- Combine both as **boolean** masks (`allowed = causal & key_not_pad`, broadcast to $(B, h, n, n)$), then fill once (`masked_fill(~allowed, finfo.min)`) or pass the bool mask to SDPA. Adding two additive masks of `finfo(dtype).min` overflows to $-\infty$ and brings the NaN back.
- **Pitfalls:**
  - A fully masked row makes softmax of all $-\infty$ = NaN.
  - `-1e9` overflows fp16; use `torch.finfo(dtype).min` — and apply it once (see above).
  - Left-pad decoder-only models for batched generation.
  - Boolean-mask conventions differ: in `nn.MultiheadAttention`'s `key_padding_mask`, True means *ignore*; in `F.scaled_dot_product_attention`'s boolean `attn_mask`, True means *attend*.
- **Other masks:** prefix-LM (bidirectional over the prefix), block-causal (frames or images), Flamingo's "attend only to the latest image" mask.

### Q4.5 Positional encodings: sinusoidal, learned, relative, RoPE, 2-D?
- Self-attention is **permutation-equivariant**, so position must be injected.
- **Sinusoidal** (Vaswani et al. 2017): $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$. Multi-scale frequencies; $PE_{pos+k}$ is a rotation (linear map) of $PE_{pos}$; no params; extrapolates poorly in practice.
- **Learned absolute** (BERT, GPT-2, ViT): a lookup table, simple, but capped at the maximum length. ViT interpolates it for new resolutions.
- **Relative:** T5 (learned bucketed bias) and ALiBi (per-head linear distance penalty on logits, good length extrapolation).
- **RoPE** (Su et al., RoFormer, 2021):
  - Rotate each 2-D pair of $q$/$k$ dimensions by angle $m\theta_i$, with $\theta_i = 10000^{-2i/d}$.
  - Then $\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$ depends only on the offset.
  - Applied to $q$ and $k$ (not $v$) in every layer, with no params. Standard in LLMs.
  - Context extension: position interpolation, NTK-aware scaling, YaRN.
- **2-D for images:** split channels in half, one half encoding the row and the other the column (2-D sin-cos in MAE; axial 2-D RoPE). For VLMs, M-RoPE (Qwen2-VL) uses separate time, height and width components.
- → practice: llm/Sinusoidal-Positional-Embedding, llm/Rotary-Positional-Embedding, v3/modern-architectures/2d-positional-embeddings

### Q4.6 Pre-norm vs post-norm?
- **Post-LN** (original Transformer, BERT): $x_{l+1} = \mathrm{LN}(x_l + F(x_l))$. The LN sits on the residual path, which can make gradients badly scaled, so it needs careful warmup and is unstable for deep stacks. When it does train, it is often reported to reach slightly better final quality (e.g. Liu et al. 2020, "Understanding the Difficulty of Training Transformers").
- **Pre-LN** (GPT-2 onward, ViT, most modern models): $x_{l+1} = x_l + F(\mathrm{LN}(x_l))$, plus a final LN before the head.
  - The clean identity path gives stable gradients, so it trains with little warmup (Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture").
  - Downside: the residual-stream norm grows with depth, so later layers contribute relatively less.
- **Variants:** DeepNorm (scaled residuals for post-LN), sandwich norm, QK-norm, adaLN (conditioning through norm modulation, as in DiT).

### Q4.7 What is the KV cache? How big is it?
- In causal decoding, past tokens' K and V never change. Cache them per layer and append each step, so a step processes only the new token: projections cost $O(d^2)$ instead of $O(td^2)$, and attention costs $O(td)$ per layer.
- **Memory:** $2 \times n_{layers} \times n_{kv\,heads} \times d_{head} \times \text{seq\_len} \times \text{batch} \times \text{bytes}$.
- **Example:** Llama-2-7B has 32 layers and 32 heads of size 128. In fp16 that is $2\cdot32\cdot4096\cdot2$ B = **0.5 MiB per token**, so a 4k context is about 2 GiB per sequence. The KV cache, not the weights, often limits batch size.
- **Prefill vs decode:** prefill processes the prompt in parallel and is compute-bound. Decode runs one token at a time and is memory-bandwidth-bound, because it reads all weights and the KV cache for little compute.
- **Shrinking or managing it:** GQA/MQA, KV quantization, PagedAttention (vLLM), prefix caching, sliding windows or eviction.
- → practice: v3/llm-inference/kv-cache

### Q4.8 MQA and GQA?
- **MHA** has $h$ query heads and $h$ K/V heads.
- **MQA** (Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need"): all query heads share one K/V head, so the KV cache shrinks $h\times$. Much faster memory-bound decoding, with some quality loss.
- **GQA** (Ainslie et al. 2023): $G$ groups of query heads each share a K/V head. The cache shrinks $h/G\times$ and quality stays close to MHA. Example: Llama-2-70B and Llama-3 use 8 KV heads.
- You can up-train an MHA checkpoint to GQA by mean-pooling each group's K/V heads and briefly continuing training.
- The win is decode bandwidth and memory (bigger batches, longer context), not prefill FLOPs.
- Related: MLA (DeepSeek-V2) compresses K/V into a low-rank latent. In code, use `repeat_interleave` to expand K/V heads, or broadcast inside the kernel.
- → practice: llm/Grouped-Query-Attention

### Q4.9 Explain FlashAttention.
- **Problem:** standard attention writes the $n\times n$ score and probability matrices to HBM and reads them back. The softmax, mask and dropout steps are memory-bound, and memory is $O(n^2)$.
- **FlashAttention** (Dao et al. 2022) is **IO-aware, exact** attention:
  - Tile $Q$ into blocks; for each Q block, loop over K/V blocks held in on-chip SRAM. (This is the FA2 loop order; FA1 had K/V in the outer loop.)
  - Compute each block's scores and fold them into the output with an **online softmax**, never materializing $n\times n$ in HBM.
- **Online softmax** keeps, per query row, a running max $m$, denominator $\ell$ and unnormalized output $o$. For a new block of scores $s$ with values $V_b$:
  - $m' = \max(m, \max_j s_j)$
  - $\ell' = e^{m - m'}\ell + \sum_j e^{s_j - m'}$
  - $o' = e^{m - m'}o + \sum_j e^{s_j - m'}V_{b,j}$
  - At the end, output $o/\ell$.
- **Backward:** store only the per-row logsumexp and recompute scores blockwise. This means *more* FLOPs but far fewer HBM bytes, so it is faster wall-clock with $O(n)$ extra memory.
- It is exact, not an approximation. FA2 improved parallelism and work partitioning; FA3 targets Hopper (async, FP8).
- → practice: llm/Flash-Attention, v3/gpu-systems/flash-attention-triton, v3/gpu-systems/triton-fused-softmax

### Q4.10 Mixture of Experts basics?
- Replace the dense FFN with $E$ expert FFNs plus a router $g(x) = \mathrm{softmax}(W_rx)$. Each token goes to its top-$k$ experts ($k = 1$ in Switch, $k = 2$ in GShard/Mixtral), and the output is the gate-weighted sum.
- **Decouples params from per-token FLOPs.** Mixtral 8×7B has about 47B total params but about 13B active per token.
- **Problems and fixes:**
  - *Load imbalance / router collapse:* add an auxiliary balancing loss. Switch uses $\alpha E\sum_i f_iP_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the mean router probability for $i$.
  - *Capacity limits:* set a capacity factor; overflow tokens are dropped or passed through via the residual.
  - *Instability:* router z-loss (ST-MoE).
  - *Communication:* expert parallelism needs all-to-all.
  - *Memory:* every expert must be resident, so inference memory is high.
  - *Fine-tuning instability.*
- **Recent trends:** fine-grained plus shared experts (DeepSeekMoE) and auxiliary-loss-free balancing via router bias terms (DeepSeek-V3). Also used in vision (V-MoE) and multimodal models.
- → practice: v3/modern-architectures/mixture-of-experts

### Q4.11 Encoder-only vs decoder-only vs encoder–decoder; what is cross-attention for?
- **Encoder-only** (BERT, ViT, CLIP towers): bidirectional self-attention. Produces representations for classification, retrieval and dense prediction.
- **Decoder-only** (GPT, LLaMA, the LLM in most VLMs): causal self-attention, native generation, scales simply, and prompting covers most tasks.
- **Encoder–decoder** (original Transformer, T5, Whisper): a bidirectional encoder over the input, and a decoder with causal self-attention plus **cross-attention** (Q from the decoder; K, V from the encoder). A natural fit for transduction: translation, ASR, captioning.
- **Cross-attention as a general bridge:**
  - Stable Diffusion's U-Net reads text tokens.
  - Flamingo reads image tokens.
  - Perceiver/Q-Former/DETR use learned queries to read a large input.

> **Say it in one breath**
> - Attention is $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. The $\sqrt{d_k}$ keeps logit variance at 1 so softmax doesn't saturate. Multi-head attends in parallel subspaces at the same cost. A block is about $12d^2$ params.
> - Cost is $O(n^2d)$ time and $O(n^2)$ memory naively. FlashAttention keeps the FLOPs but tiles in SRAM with an online softmax, so memory is $O(n)$. It runs faster because standard attention is IO-bound, not FLOP-bound.
> - Decoding is memory-bound. The KV cache costs $2 \cdot L \cdot n_{kv} \cdot d_h \cdot$ bytes per token (0.5 MiB/token for Llama-2-7B fp16). GQA/MQA shrink it by sharing K/V heads.
> - Use RoPE for relative position in LLMs and 2-D sin-cos or RoPE for images. Use pre-LN for stability. MoE buys parameters without FLOPs at the cost of balancing, communication and memory.

---

## 5. Multimodal

### Q5.1 Explain CLIP's training objective and the role of temperature.
- (Radford et al. 2021, "Learning Transferable Visual Models From Natural Language Supervision")
  - 400M web image–text pairs.
  - An image encoder (ResNet or ViT) and a text transformer, each with a linear projection into a shared space, then L2-normalized.
- **Loss:** for a batch of $N$ pairs, logits are $L_{ij} = s\,\langle\hat u_i, \hat v_j\rangle$ with learnable $s = e^{t}$ (the inverse temperature). The loss is **symmetric cross-entropy** with the diagonal as targets: $\frac12[\mathrm{CE}_{\text{rows}} + \mathrm{CE}_{\text{cols}}]$. Each image must pick its own caption out of $N$, and each caption its own image. This is InfoNCE.
- **Temperature:**
  - Initialized to $\tau = 0.07$ ($s \approx 14.3$) and learned in log-space — learned so that it need not be tuned as a hyperparameter (the paper's stated reason).
  - **Clipped so $s \le 100$** for stability.
  - Cosine similarities live in $[-1, 1]$, so without scaling the softmax would be far too flat. Lower $\tau$ sharpens the distribution and focuses on hard negatives; too low is unstable.
- **Scale:** batch size 32,768 for many negatives. In distributed training, all-gather the embeddings across GPUs (gradients flow through the local shard).
- **Theory:** InfoNCE bounds mutual information, $I(X;Y) \ge \log N - \mathcal L_{\text{InfoNCE}}$ (van den Oord et al. 2018, CPC), so more negatives give a tighter bound.
- → practice: v3/modern-architectures/contrastive-loss-clip

### Q5.2 How does CLIP zero-shot classification work, and where does CLIP fail?
- **Zero-shot:** embed prompts such as "a photo of a {class}." The text embeddings act as classifier weights; take the argmax of cosine similarity with the image embedding.
  - Prompt ensembling (CLIP used 80 templates for ImageNet) averages text embeddings and adds several points.
  - ViT-L/14@336px reaches about 76% zero-shot ImageNet top-1, matching a supervised ResNet-50, and is far more robust under natural distribution shift.
- **Adaptation:** a linear probe on CLIP features is a strong baseline. Full fine-tuning can erode robustness; WiSE-FT interpolates zero-shot and fine-tuned weights to recover it.
- **Failure modes:**
  - "Bag-of-words" behavior: weak on composition, word order, counting, spatial relations and negation (Winoground, ARO, SugarCrepe).
  - Typographic attacks.
  - Rare or fine-grained classes.
  - Web-data biases.
  - The **modality gap**: image and text embeddings sit in separate cones.
- **Uses:** retrieval, the text encoder for SD1.x, the vision tower for VLMs (LLaVA), CLIPScore and data filtering (LAION).

### Q5.3 What does SigLIP change, and why?
- (Zhai et al. 2023, "Sigmoid Loss for Language Image Pre-Training") Replace the batch-level softmax with **independent binary classification of every image–text pair**:
  $\mathcal L = -\frac{1}{N}\sum_i\sum_j\log\sigma\big(z_{ij}\,(t\,\hat x_i\cdot\hat y_j + b)\big)$, with $z_{ij} = +1$ if $i = j$, else $-1$.
- **Temperature and bias:** $t = e^{t'}$ is initialized with $t' = \log 10$, and the learnable bias $b$ starts at $-10$. There are $N - 1$ negatives per positive, so the bias starts training near the "mostly negative" prior instead of with a huge loss.
- **No normalization across the batch:**
  - Each pair's loss is local, so you don't need the full $N\times N$ softmax gathered on every device.
  - A chunked "pass the negatives around" implementation is memory-efficient.
  - It clearly beats softmax at small batch sizes (a few thousand), is on par at large ones, and both losses stop improving around a 32k batch.
- SigLIP and SigLIP 2 encoders are popular VLM vision towers (e.g. PaliGemma uses SigLIP).
- → practice: v4/multimodal/siglip-loss

### Q5.4 BLIP and BLIP-2: what is the Q-Former?
- **BLIP** (Li et al. 2022):
  - Three losses on a shared encoder/decoder: image–text contrastive (ITC); image–text matching (ITM, a binary classifier trained on hard negatives mined with ITC); and image-conditioned language modeling (LM).
  - **CapFilt** bootstraps noisy web data: a captioner writes synthetic captions and an ITM filter removes bad pairs.
- **BLIP-2** (Li et al. 2023) connects a **frozen** image encoder to a **frozen** LLM through a **Q-Former**:
  - The Q-Former is a BERT-base-initialized transformer with **32 learnable queries**.
  - The queries self-attend and cross-attend to the frozen image features (cross-attention in every other block) and output 32 vectors.
  - About 188M trainable params.
- **Two training stages:**
  - *Stage 1*, vision–language representation learning with the image encoder frozen: ITC + ITM + image-grounded generation, each with a different query/text attention mask.
  - *Stage 2*, generative: a linear layer projects the Q-Former outputs into the LLM's input-embedding space as a soft visual prompt, trained with the LM loss while the LLM stays frozen.
- **Pros:** few visual tokens and few trainable params.
- **Cons:** an information bottleneck that loses fine detail (OCR, small text). LLaVA-1.5 later showed that a simple MLP over *all* patch tokens, with good data, can match or beat Q-Former-based models such as InstructBLIP.
- InstructBLIP feeds the instruction text into the Q-Former as well.

### Q5.5 Flamingo: Perceiver Resampler and gated cross-attention?
- (Alayrac et al. 2022) A few-shot VLM over interleaved images/video and text. It uses a frozen vision encoder (contrastively pretrained NFNet) and a frozen LM (Chinchilla; the largest Flamingo is 80B).
- **Perceiver Resampler:** a fixed set of learned latent queries cross-attends to a variable number of spatio-temporal visual features and outputs **64 visual tokens** per image or video.
- **Gated XATTN-DENSE layers:**
  - Inserted between the frozen LM blocks.
  - Text tokens cross-attend to visual tokens.
  - Each layer's output is multiplied by $\tanh(\alpha)$ with **$\alpha$ initialized to 0**, so at init the model is exactly the original LM. This preserves language ability and keeps training stable.
- **Masking:** each text token attends only to the visual tokens of the most recent preceding image.
- **Data:** interleaved web pages (M3W) plus image–text and video–text pairs. This gives in-context few-shot learning. Open re-implementations: OpenFlamingo, IDEFICS.
- **Design contrast:** cross-attention keeps vision out of the LLM's sequence (short context) but adds new parameters. LLaVA-style prefix tokens are simpler but make sequences long.

### Q5.6 LLaVA: architecture and two-stage training?
- (Liu et al. 2023, "Visual Instruction Tuning") A CLIP ViT-L/14 encoder (penultimate-layer patch features) feeds a **projector** into an LLM (Vicuna). The visual tokens are placed in the prompt alongside text tokens.
- **Projector:** a single linear layer in LLaVA; a 2-layer MLP with GELU in LLaVA-1.5.
- **Stage 1, feature alignment:** freeze the vision encoder and the LLM and train *only the projector* on about 600K image–caption pairs (filtered CC3M in LLaVA; a 558K LAION/CC/SBU subset in 1.5).
- **Stage 2, visual instruction tuning:** keep the vision encoder frozen and train the projector + LLM on instruction data.
  - LLaVA: 158K GPT-4-generated conversations, detailed descriptions and reasoning, built from COCO captions and boxes.
  - LLaVA-1.5: 665K, adding academic VQA with short-answer format prompts.
- **Tokens:** 336 px / 14 gives 24×24 = **576 visual tokens**. LLaVA-NeXt's "AnyRes" tiles high-resolution images (more tokens), which is key for OCR and documents (Q5.13).
- The loss is applied only to assistant-response tokens.
- → practice: v4/multimodal/vlm-connector

### Q5.7 Connector design: MLP projector vs Q-Former/resampler vs cross-attention?
- **MLP projector** (LLaVA): keeps every patch token, so maximum detail, simple and fast to train. But you get many tokens (576, and thousands with tiling), which means long, expensive contexts.
- **Learned-query compressors** (Q-Former, Perceiver Resampler): a fixed, small token count, but they can lose fine-grained and spatial information.
- **Token reduction in between:** pixel-shuffle or 2×2 merging (e.g. InternVL; Qwen2-VL merges 2×2 patches), pooling, conv abstractors.
- **Cross-attention** (Flamingo, Llama 3.2 Vision): vision doesn't consume LLM context and the LLM can stay frozen, but it adds more new parameters.
- **Early fusion / native multimodal** (Chameleon, Fuyu): image patches or VQ tokens go straight into a single transformer.
- **What matters most empirically:** the image encoder, image resolution and visual-token count have a large effect, the pretraining data mix (captions, interleaved, text-only) matters a lot, and connector type matters comparatively little (Apple's MM1 ablations, McKinzie et al. 2024). Also important: the vision encoder (SigLIP is often better than CLIP) and unfreezing it in later stages.
- → practice: v4/multimodal/vlm-connector

### Q5.8 How do VQ-VAE and VQGAN tokenize images?
- **VQ-VAE** (van den Oord et al. 2017, "Neural Discrete Representation Learning"):
  - The encoder produces a grid $z_e(x)$, and each vector is snapped to its nearest codebook entry $e_k$, $k = \arg\min_j\|z_e - e_j\|$. The decoder reconstructs from the quantized grid.
  - Loss: reconstruction $+\ \|\mathrm{sg}[z_e] - e\|^2$ (codebook) $+\ \beta\|z_e - \mathrm{sg}[e]\|^2$ (commitment, $\beta = 0.25$).
  - Gradients pass the argmin via the **straight-through estimator**: $z_q = z_e + \mathrm{sg}[z_q - z_e]$.
  - The codebook is often updated by EMA instead of the codebook loss. A prior over codes (PixelCNN, later a transformer) is trained afterwards.
- **VQGAN** (Esser et al. 2021, "Taming Transformers for High-Resolution Image Synthesis"):
  - Adds perceptual (LPIPS) and PatchGAN adversarial losses, giving sharp reconstructions at high compression (e.g. 256² → 16×16 tokens, $f = 16$).
  - A GPT-style transformer then models the token sequence.
- **Problems:** codebook collapse and low utilization. Fixes include k-means init, dead-code restarts, low-dimensional L2-normalized codes (ViT-VQGAN), and lookup-free or finite scalar quantization (LFQ in MAGVIT-v2; FSQ rounds a few dims to a few levels, so there is no learned codebook and utilization is near-full).
- **Trade-off:** compression factor and vocab size vs reconstruction fidelity vs how easy the tokens are for the generator to model. Continuous (KL-VAE) latents suit diffusion; discrete tokens suit AR/LLM-style models.
- → practice: v4/generative/vae, torch/easy/autoencoder

### Q5.9 How do you evaluate VLMs: retrieval, VQA, captioning?
- **Retrieval:** Recall@K = fraction of queries whose ground-truth match is in the top K, in both directions (image→text and text→image). Standard sets: COCO 5k test and Flickr30K 1k, with 5 captions per image.
- **VQA:**
  - VQAv2 accuracy is $\min\left(\frac{\#\text{annotators giving that answer}}{3}, 1\right)$, averaged over 10-choose-9 annotator subsets, after answer normalization.
  - Generative VLMs need answer extraction or an LLM judge.
- **Captioning:**
  - BLEU (n-gram precision + brevity penalty), METEOR, ROUGE-L, **CIDEr** (TF-IDF-weighted n-gram consensus; the main COCO metric), SPICE (scene-graph propositions), reference-free CLIPScore.
  - N-gram metrics correlate poorly with humans for long, detailed captions.
- **Modern suites:** MMMU, MMBench, MME, SEED-Bench, TextVQA/DocVQA/ChartQA (OCR), RefCOCO (grounding), Video-MME, MM-Vet (LLM-graded).
- **Pitfalls:**
  - Contamination.
  - Answer-format sensitivity.
  - LLM-judge bias.
  - Models answering from language priors without looking. Run a **blind (text-only) baseline**.
- → practice: v4/multimodal/vector-search

### Q5.10 Hallucination in VLMs: how to measure and reduce?
- **Definition:** describing objects, attributes or relations that are not in the image. Object hallucination is the most measured.
- **Metrics:**
  - **CHAIR** (Rohrbach et al. 2018): CHAIR$_i$ = hallucinated object mentions / all object mentions; CHAIR$_s$ = captions with at least one hallucination / all captions.
  - **POPE** (Li et al. 2023): yes/no polling ("Is there a {object} in the image?") with random, popular and adversarial negatives; report accuracy, F1 and the yes-ratio.
  - Also AMBER, HallusionBench and LLM-judged faithfulness.
- **Causes:**
  - Strong language priors and co-occurrence statistics.
  - Noisy or hallucinated training captions.
  - Low-resolution visual features.
  - Long generations drifting from the visual evidence.
  - Instruction data that rewards detail.
- **Mitigations:**
  - Higher resolution and better encoders.
  - Cleaner grounded data.
  - Preference optimization on hallucination pairs (RLHF-V, DPO variants).
  - Contrastive decoding (e.g. VCD).
  - Grounded outputs (boxes) and abstention training.

### Q5.11 Vector search at scale for multimodal retrieval?
- Embed the corpus once and L2-normalize, so cosine similarity equals dot product. Retrieve by nearest neighbor.
- **Exact:** brute-force matmul on GPU works up to millions of items.
- **Approximate (ANN):**
  - *IVF:* a k-means coarse quantizer; search `nprobe` cells.
  - *PQ:* split each vector into $m$ sub-vectors and quantize each to one of 256 centroids, so $m$ bytes per vector; distances come from lookup tables.
  - *HNSW:* a navigable small-world graph searched greedily; high recall but memory-hungry.
  - Libraries: FAISS, ScaNN, vector DBs.
- **Trade-offs:** recall@k vs latency vs memory; tune `nprobe` / `efSearch`. Re-rank the top candidates exactly or with a cross-encoder.
- **Typical stack:** a dual encoder (CLIP/SigLIP) for recall, then a cross-encoder, ITM head or VLM for precision. Optionally hybrid with BM25 text search. Embeddings also power dedup.
- → practice: v4/multimodal/vector-search

### Q5.12 Dual encoder vs fusion encoder vs generative VLM: when to use which?
- **Dual encoder** (CLIP, SigLIP): independent embeddings that can be precomputed. Fast retrieval and zero-shot classification, but weak on fine-grained compositional reasoning.
- **Fusion / cross-encoder** (ITM heads, ALBEF/BLIP): joint attention over image and text. Accurate matching and VQA, but scoring every query–item pair makes it $O(N\cdot M)$ for retrieval, so use it for **re-ranking**.
- **Generative VLM** (LLaVA, Flamingo, Qwen-VL): open-ended answers, instruction following and reasoning. But it is expensive, can hallucinate and is hard to evaluate.
- **Hybrids:** CoCa trains contrastive + captioning losses in one model (Q5.14). Production systems often cascade dual encoder → re-ranker → VLM.

### Q5.13 How do VLMs handle high-resolution and arbitrary aspect-ratio images?
- **Problem:** vision towers are pretrained at a fixed size (224–448 px). Squashing or center-cropping a document or a wide photo destroys small text and objects. Simply raising the resolution grows the token count quadratically (336 → 672 px is 4× the tokens), and the position embeddings must be interpolated.
- **Tiling ("AnyRes", dynamic tiling):**
  - From a set of candidate grids, pick the one whose aspect ratio and resolution best fit the image. Resize/pad, split into encoder-sized tiles and encode each tile separately with the unchanged ViT. Add a downsampled global thumbnail for context.
  - LLaVA-NeXt: 336-px tiles in 2×2, 1×{2,3,4} or {2,3,4}×1 grids (up to 672×672 or 336×1344), so up to $5 \times 576 = 2{,}880$ visual tokens. A learned newline token after each row tells the LLM the 2-D layout.
  - InternVL 1.5: 448-px tiles, 1–12 tiles in training (up to 40 at test), pixel shuffle to 256 tokens per tile, so up to 3,328 tokens with the thumbnail.
  - Downsides: objects get split at tile seams, and tokens grow with resolution.
- **Native (dynamic) resolution:** run the ViT at the image's own resolution and aspect ratio, so the token count follows the pixel count.
  - Qwen2-VL replaces the ViT's absolute position embeddings with 2-D RoPE, uses 14-px patches and merges 2×2 neighbors with an MLP. One LLM token then covers 28×28 px (a 224² image gives 64 visual tokens), within min/max pixel budgets. M-RoPE (Q4.5) gives the LLM time/height/width positions.
  - NaViT ("Patch n' Pack", Dehghani et al. 2023) packs images of different sizes into one sequence with attention masks. SigLIP 2's NaFlex variant keeps the native aspect ratio.
- **Controlling the budget:** pixel shuffle or 2×2 merging, pooling, caps on tiles or pixels. For video, use fewer tokens per frame and merge over time (Qwen2-VL encodes pairs of frames with a 3-D conv).
- **Interview point:** resolution vs token count vs latency is the main knob. OCR, document and chart tasks gain most from resolution, and MM1 found resolution and token count among the biggest levers (Q5.7).
- → practice: v4/multimodal/vlm-connector

### Q5.14 Contrastive vs generative (captioning) pretraining for vision encoders: trade-offs?
- **Contrastive** (CLIP, SigLIP): learn a joint embedding space by matching images to their captions within a batch.
  - Pros: zero-shot classification and retrieval straight from precomputed embeddings, and no text decoder to train.
  - Cons: the softmax version needs very large batches for negatives. A caption only has to be told apart from the other captions in the batch, so the model can ignore word order and relations ("bag-of-words" behavior, Q5.2).
- **Generative / captioning** (SimVLM, CapPa, or the LM loss inside a VLM): predict the caption token by token from the image.
  - Pros: every token supervises, there is no need for huge batches, and the model has to account for the whole caption, so it handles compositional structure better.
  - Cons: you need a decoder, and zero-shot classification means scoring every class name with it, which is slower and usually less accurate.
- **Evidence:**
  - CLIP's own ablation: a transformer captioner learned zero-shot ImageNet about 3× more slowly than a bag-of-words predictor, and the contrastive objective was another 4× more efficient. That is why CLIP is contrastive.
  - Tschannen et al. 2023 ("Image Captioners Are Scalable Vision Learners Too", CapPa), with matched data and compute: captioning encoders are competitive on classification, better when plugged into a VLM and clearly better on ARO/SugarCrepe, but lag in zero-shot classification.
- **Hybrids are now common:** CoCa (contrastive + captioning in one model), BLIP's ITC + ITM + LM, SigLIP 2 (sigmoid loss plus a captioning decoder, self-distillation and masked prediction), AIMv2 (Fini et al. 2024; a decoder autoregressively generates image patches and text).
- **Rule of thumb:** if you need an embedding for retrieval, zero-shot classification or data filtering, use contrastive (or a hybrid). For feeding an LLM either works; most open VLMs start from a contrastively pretrained encoder (often SigLIP) and unfreeze it in later training stages.

> **Say it in one breath**
> - CLIP uses a symmetric InfoNCE over an $N\times N$ cosine-similarity matrix with learnable temperature (init 0.07, logit scale capped at 100) and large batches. SigLIP swaps the softmax for per-pair sigmoid with a learnable bias (init −10), so there is no global normalization and small batches work.
> - VLM connectors span three designs: MLP projector (LLaVA, all 576 tokens), learned-query compressors (Q-Former 32, Perceiver Resampler 64) and gated cross-attention (Flamingo, tanh gate init 0). Resolution and data matter more than connector type.
> - LLaVA trains in two stages: (1) projector only on captions, (2) projector + LLM on instruction data with the vision encoder frozen.
> - For high resolution, either tile into encoder-sized crops plus a thumbnail (AnyRes, InternVL) or run the ViT at native resolution with 2-D RoPE (Qwen2-VL). Contrastive encoders give embeddings for retrieval; captioning encoders handle composition better; hybrids are common.
> - Evaluate with Recall@K, VQA accuracy, CIDEr/CLIPScore, CHAIR/POPE for hallucination, and always a text-only blind baseline.

---

## 6. Generative models

### Q6.1 Derive the VAE ELBO.
- **Setup:** the model is $p_\theta(x) = \int p_\theta(x \mid z)\,p(z)\,dz$, which is intractable, so introduce an encoder $q_\phi(z \mid x)$.
- **Identity:** $\log p_\theta(x) = \mathbb E_{q_\phi}\left[\log\frac{p_\theta(x,z)}{q_\phi(z \mid x)}\right] + \mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\big)$.
- **Bound:** since KL ≥ 0,
  $\log p_\theta(x) \ge \underbrace{\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} - \underbrace{\mathrm{KL}(q_\phi(z|x)\,\|\,p(z))}_{\text{regularizer}} = \mathrm{ELBO}$.
  The gap is the KL to the true posterior. (Equivalently, apply Jensen to $\log\mathbb E_q[p(x,z)/q]$.)
- **Gaussian encoder with $\mathcal N(0, I)$ prior:** $\mathrm{KL} = \frac12\sum_j(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$.
- **Decoder likelihood:** a Gaussian decoder with fixed variance makes the reconstruction term a scaled MSE; a Bernoulli decoder makes it BCE.
- Kingma & Welling 2013 ("Auto-Encoding Variational Bayes"); Rezende et al. 2014.
- → practice: v4/generative/vae, torch/easy/autoencoder

### Q6.2 What is the reparameterization trick?
- **Problem:** we need $\nabla_\phi\mathbb E_{q_\phi(z|x)}[f(z)]$, but sampling $z$ is not differentiable with respect to $\phi$.
- **Trick:** write $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$ with $\epsilon \sim \mathcal N(0, I)$. The randomness now comes from a fixed distribution, so $\nabla_\phi\mathbb E_\epsilon[f(\mu + \sigma\epsilon)] = \mathbb E_\epsilon[\nabla_\phi f(\cdot)]$. This pathwise gradient usually has much lower variance than the score-function estimator below, but it needs $f$ differentiable in $z$ and a continuous, reparameterizable $q$.
- **Alternatives:**
  - Score-function / REINFORCE: $\mathbb E[f(z)\nabla_\phi\log q_\phi(z)]$. Works for discrete $z$ but has high variance (needs baselines).
  - Gumbel-softmax relaxation.
  - Straight-through (as in VQ-VAE).
- Predict $\log\sigma^2$ for numerical stability.

### Q6.3 Posterior collapse, blurry samples, β-VAE?
- **Posterior collapse:** the KL term pushes $q(z \mid x) \to p(z)$ and the decoder ignores $z$. This is especially common with powerful autoregressive decoders that can model $x$ alone (text VAEs).
  - Fixes: KL annealing or warmup (β from 0 to 1), free bits (no KL penalty below a per-dimension floor), a weaker decoder or word dropout, $z$ skip-connections, aggressive encoder updates.
- **Blurriness:** a factorized Gaussian likelihood (≈ MSE) averages over plausible outputs. There are also "prior holes": regions of $p(z)$ that the encoder never produces.
  - Fixes: perceptual and adversarial losses (as in VQGAN), hierarchical VAEs (NVAE, VDVAE), learned priors.
- **β-VAE** (Higgins et al. 2017): weight the KL by $\beta > 1$ for more disentangled but worse reconstructions. The disentanglement gains are contested: Locatello et al. (2019) showed that unsupervised disentanglement depends heavily on seeds and hyperparameters and cannot be identified without inductive biases. Conversely, Stable Diffusion's autoencoder uses a *tiny* KL weight, making it essentially a regularized autoencoder.

### Q6.4 GAN objective, optimal discriminator, non-saturating loss?
- **Objective** (Goodfellow et al. 2014): $\min_G\max_D\ \mathbb E_{x\sim p_{data}}[\log D(x)] + \mathbb E_z[\log(1 - D(G(z)))]$.
- **Optimal D** for a fixed $G$: $D^\star(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$. Substituting gives $C(G) = -\log 4 + 2\,\mathrm{JSD}(p_{data}\,\|\,p_g)$, minimized when $p_g = p_{data}$.
- **Non-saturating loss:** $G$ *maximizes* $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$.
  - Early in training, $D$ easily rejects fakes ($D(G(z)) \approx 0$), where $\log(1 - D)$ is flat, so $G$ gets almost no gradient.
  - The NS loss has strong gradients exactly there. Same fixed point, better dynamics.
- **It is a game, not a minimization:** there is no single loss to monitor and training oscillates.
- **Tricks:** Adam with $\beta_1 = 0.5$ (DCGAN), spectral norm, R1 gradient penalty, TTUR, EMA of $G$, and discriminator augmentation (DiffAugment/ADA) in low-data regimes.
- → practice: torch/hard/GAN

### Q6.5 Mode collapse and WGAN?
- **Mode collapse:** $G$ maps many $z$ to a few outputs that currently fool $D$; $D$ adapts and $G$ hops to another mode. The result is high precision but low recall, a mode-seeking behavior.
- **Diagnose** with sample diversity, the recall metric, nearest-neighbor checks and class histograms.
- **Mitigate** with minibatch discrimination or the minibatch std-dev layer (ProGAN/StyleGAN), unrolled GANs, WGAN-GP, spectral norm, conditioning, or diversity losses.
- **WGAN** (Arjovsky et al. 2017):
  - Minimizes the Earth-Mover / Wasserstein-1 distance using Kantorovich–Rubinstein duality: $W_1 = \sup_{\|f\|_L \le 1}\mathbb E_{p_{data}}[f] - \mathbb E_{p_g}[f]$.
  - The critic has no sigmoid and must be 1-Lipschitz. Enforce this with weight clipping (original, crude), a **gradient penalty** $\lambda(\|\nabla_{\hat x}D(\hat x)\|_2 - 1)^2$ on interpolates (WGAN-GP, Gulrajani et al. 2017, $\lambda = 10$), or spectral normalization (Miyato et al. 2018).
- **Why it helps:** with disjoint supports (typical for low-dimensional manifolds), JSD is constant at $\log 2$ and gives no useful gradient. $W_1$ still changes smoothly with distance, and the critic loss tracks sample quality.

### Q6.6 Describe the DDPM forward and reverse processes and the training loss.
- **Forward** (fixed): $q(x_t \mid x_{t-1}) = \mathcal N(\sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t I)$ for $t = 1..T$. DDPM uses $T = 1000$ with $\beta$ linear from $10^{-4}$ to $0.02$.
- **Closed form:** with $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$, $q(x_t \mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1 - \bar\alpha_t)I)$, i.e. **$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$**. You can jump to any $t$ in one shot.
- **Posterior** (tractable given $x_0$): $q(x_{t-1} \mid x_t, x_0) = \mathcal N(\tilde\mu_t, \tilde\beta_t I)$ with
  - $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1 - \bar\alpha_t}x_t$
  - $\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t$.
- **Reverse model:** $p_\theta(x_{t-1} \mid x_t) = \mathcal N(\mu_\theta, \sigma_t^2 I)$ with $\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$. Use $\sigma_t^2 = \beta_t$ or $\tilde\beta_t$ (similar results in DDPM; Improved DDPM learns it).
- **Training** (Ho et al. 2020): sample $x_0$, $t \sim U\{1..T\}$ and $\epsilon$, then minimize $\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon,\ t)\|^2$. This is "$L_{simple}$": the ELBO with its per-$t$ weights dropped. Relative to the ELBO this down-weights the easy small-$t$ terms, so the network focuses on harder denoising at larger $t$; it gave better samples but worse log-likelihood than the true bound.
- **Sampling:** start from $x_T \sim \mathcal N(0, I)$ and iterate $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$ (no noise at the last step).
- **Score view:** $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$. Diffusion is denoising score matching across noise levels (the SDE view of Song et al. 2021).
- → practice: v3/modern-architectures/ddpm

### Q6.7 ε- vs x₀- vs v-prediction: what is the difference?
- **Conversions:** let $a_t = \sqrt{\bar\alpha_t}$ and $s_t = \sqrt{1 - \bar\alpha_t}$, so $x_t = a_tx_0 + s_t\epsilon$ and $a_t^2 + s_t^2 = 1$. Any one prediction gives the others, e.g. $\hat x_0 = (x_t - s_t\hat\epsilon)/a_t$.
- **v-prediction** (Salimans & Ho 2022, "Progressive Distillation for Fast Sampling of Diffusion Models"): $v = a_t\epsilon - s_tx_0$. Then $\hat x_0 = a_tx_t - s_t\hat v$ and $\hat\epsilon = s_tx_t + a_t\hat v$.
- They share the same optimum but imply **different loss weighting across noise levels** and different numerics:
  - **ε-pred:** at high noise ($a_t \to 0$), recovering $\hat x_0$ divides by about 0 and amplifies errors. It is incompatible with zero terminal SNR.
  - **x₀-pred:** since $\|\hat\epsilon - \epsilon\|^2 = \mathrm{SNR}\cdot\|\hat x_0 - x_0\|^2$, an unweighted x₀-MSE is the ε-MSE divided by SNR. It puts little weight on low-noise steps (fine detail), and recovering $\hat\epsilon = (x_t - a_t\hat x_0)/s_t$ divides by $s_t \to 0$ there. Ho et al. (2020) found it gave worse samples than ε-pred. It has come back for pixel-space transformers with large patches (Li & He 2025, "Back to Basics: Let Denoising Generative Models Denoise"), where they report ε- and v-prediction fail.
  - **v-pred:** well-conditioned across SNRs. Introduced for progressive distillation; used with zero-terminal-SNR schedules (Lin et al. 2023, "Common Diffusion Noise Schedules and Sample Steps are Flawed"), SD 2.x-768 and Imagen Video.
- Flow matching's velocity target (data − noise) is a close cousin.
- **Unifying view:** all of these are $\mathbb E\,w(\lambda_t)\|\hat x_0 - x_0\|^2$ with different weightings over log-SNR $\lambda_t$ (Kingma et al. 2021, "Variational Diffusion Models"; Kingma & Gao 2023).

### Q6.8 DDPM vs DDIM sampling?
- **DDPM:** stochastic ancestral sampling over a Markov chain; good quality needs about all $T$ steps.
- **DDIM** (Song, Meng & Ermon, ICLR 2021): non-Markovian forward processes with the **same marginals** $q(x_t \mid x_0)$, hence the same training objective. Any ε-model trained as DDPM can be sampled with DDIM without retraining.
- **Update:** $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\,\hat\epsilon + \sigma_t z$, with $\sigma_t = \eta\sqrt{\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}}\sqrt{1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}}$. $\eta = 0$ is deterministic; $\eta = 1$ recovers DDPM's $\tilde\beta_t$.
- **Why η = 0 is useful:** it is a first-order discretization of the probability-flow ODE.
  - You can sample on a sub-sequence of timesteps (e.g. 50 instead of 1000) with little quality loss.
  - The noise ↔ image map is deterministic, enabling inversion for editing and latent interpolation.
- **Faster still:** higher-order ODE solvers (DPM-Solver++, UniPC, Heun in EDM) or distillation to 1–4 steps (progressive distillation, consistency models, adversarial distillation; Q6.19).
- → practice: v3/modern-architectures/ddim-cfg

### Q6.9 Noise schedules: why do they matter?
- The schedule fixes the SNR curve $\bar\alpha_t/(1 - \bar\alpha_t)$, i.e. how training and sampling effort is spread over noise levels (log-SNR).
- **Linear** (DDPM): destroys information quickly, so many late steps are nearly pure noise (wasted, especially at low resolution).
- **Cosine** (Nichol & Dhariwal 2021, "Improved Denoising Diffusion Probabilistic Models"): $\bar\alpha_t = f(t)/f(0)$ with $f(t) = \cos^2\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right)$, $s = 0.008$, and $\beta_t$ clipped at 0.999. Gives a gentler, more uniform SNR decay.
- **Resolution matters:** high-resolution images are more redundant, so the same noise level destroys less information. Shift the schedule toward more noise (Hoogeboom et al. 2023, "simple diffusion"; SD3's timestep shift).
- **Zero terminal SNR:** many schedules leave signal at $t = T$, so the model never trains on pure noise but samples from it. This train/test mismatch causes, e.g., brightness bias. Fix by rescaling to zero terminal SNR and using v-prediction.
- **EDM** (Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models"): parameterize by noise level $\sigma$ and decouple the training noise distribution (log-normal) from the sampling discretization. Adds input/output preconditioning.

### Q6.10 Classifier-free guidance: how and why?
- **Classifier guidance** (Dhariwal & Nichol 2021, "Diffusion Models Beat GANs on Image Synthesis"): add $s\,\nabla_x\log p_\phi(y \mid x_t)$ from a classifier trained on noisy images.
- **CFG** (Ho & Salimans, "Classifier-Free Diffusion Guidance", 2021/22):
  - Train one model that is both conditional and unconditional by dropping the condition (replacing it with a null token) about 10–20% of the time.
  - Sample with $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + w\,\big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing)\big)$.
  - $w = 1$ is plain conditional sampling; $w > 1$ extrapolates (SD default about 7.5).
  - **Convention:** the paper writes $(1 + w)\epsilon_c - w\epsilon_\varnothing$, i.e. $w$ shifted by 1. State which one you use.
- **Interpretation:** in score form, $\nabla\log p(x \mid c) + (w - 1)\nabla\log p(c \mid x)$, with an *implicit classifier* $p(c \mid x) \propto p(x \mid c)/p(x)$. It sharpens toward samples that are clearly identifiable as $c$. Strictly, this is a heuristic: the guided scores at different noise levels are not the scores of one noised "tilted" distribution, so CFG does not sample exactly from $p(x \mid c)\,p(c \mid x)^{w-1}$ (Bradley & Nakkiran 2024, "Classifier-Free Guidance is a Predictor-Corrector").
- **Trade-off:** higher $w$ gives better prompt adherence and fidelity but less diversity. FID improves and then degrades, so plot FID vs CLIP score. Very high $w$ oversaturates and adds artifacts.
- **Cost:** 2 forward passes per step (batched together).
- **Tricks:** guidance intervals and schedules, rescaled CFG, CFG distillation, negative prompts (replace $\varnothing$ with a negative-prompt embedding).
- → practice: v3/modern-architectures/ddim-cfg

### Q6.11 What is latent diffusion and why does it work?
- (Rombach et al. 2022, "High-Resolution Image Synthesis with Latent Diffusion Models")
  - **Stage 1:** train an autoencoder (KL-regularized or VQ) with L1/L2 + perceptual (LPIPS) + patch-adversarial losses at downsampling factor $f$. SD uses $f = 8$: 512×512×3 → 64×64×4.
  - **Stage 2:** run diffusion in that latent space.
- **Why:** pixel-space diffusion spends capacity on imperceptible high-frequency detail. The SD latent has **48× fewer dimensions** (786,432 → 16,384), so training and sampling are much cheaper at the same perceptual quality. It separates *perceptual compression* from *semantic generation*.
- **Text conditioning:** cross-attention from U-Net features (queries) to text-encoder tokens (keys and values). Encoders: CLIP ViT-L/14 in SD1.x, OpenCLIP ViT-H in SD2, CLIP ×2 + T5-XXL in SD3. The timestep enters through embeddings.
- **Practicalities:**
  - Latents are scaled to about unit variance (SD1.x factor 0.18215).
  - The autoencoder caps final quality (text, faces, fine detail), so newer VAEs use more channels (16 in SD3/Flux).
- → practice: v4/generative/vae, v3/modern-architectures/ddpm

### Q6.12 U-Net vs DiT as the diffusion backbone?
- **Diffusion U-Net** (DDPM/ADM/SD):
  - Conv ResBlocks at multiple resolutions, with self-attention (plus cross-attention for text) at lower resolutions.
  - Encoder–decoder skip connections.
  - The timestep embedding is added to each ResBlock's features (DDPM, SD) or applied as an adaptive GroupNorm scale/shift (ADM).
  - Strong inductive bias; efficient at small scale.
- **DiT** (Peebles & Xie 2023, "Scalable Diffusion Models with Transformers"):
  - Patchify the latent: a 32×32×4 latent with $p = 2$ gives 256 tokens. Then standard ViT blocks.
  - **adaLN-Zero** conditioning: an MLP on (timestep + class) regresses a per-block scale $\gamma$, shift $\beta$ and residual gate $\alpha$. The gate is initialized to 0, so every block starts as identity.
  - **Why adaLN-Zero rather than cross-attention:** DiT's condition is one global vector (timestep + class). Peebles & Xie ablated in-context tokens, cross-attention, adaLN and adaLN-Zero: adaLN-Zero had the best FID and is the cheapest (cross-attention added ~15% Gflops). It modulates every token at negligible cost, and the zero-initialized gate makes each block start as the identity, which stabilizes training. Cross-attention pays off when the condition is a *sequence* (text) — see below; even SD3/Flux still feed the timestep and pooled text embedding through adaLN.
  - Quality improves predictably with GFLOPs (depth, width or smaller patches). DiT-XL/2 reached FID 2.27 on class-conditional ImageNet 256² with CFG.
- **Why transformers won at scale:** LLM-like scaling, shared infrastructure (FlashAttention, parallelism) and easy multimodal token mixing.
- **Text conditioning in DiTs:** cross-attention (PixArt-α), or joint attention over concatenated text + image tokens with modality-specific weights (MM-DiT in SD3; Flux).
- **Video:** spatio-temporal patches with full or factorized space–time attention.
- → practice: v4/generative/dit-block

### Q6.13 Flow matching / rectified flow?
- **Goal:** learn a velocity field $v_\theta(x, t)$ whose ODE $\frac{dx}{dt} = v_\theta$ transports noise to data.
- **Linear-path recipe** (Liu et al. 2022, "Flow Straight and Fast"; Lipman et al. 2023, "Flow Matching for Generative Modeling"; Albergo & Vanden-Eijnden 2023, stochastic interpolants). This is the repo's convention, noise at $t = 0$:
  - Sample $x_0 \sim \mathcal N(0, I)$, $x_1 \sim p_{data}$, $t \sim U[0, 1]$.
  - Set $x_t = (1 - t)x_0 + tx_1$.
  - Minimize $\mathbb E\,\|v_\theta(x_t, t) - (x_1 - x_0)\|^2$.
- **Why it's simulation-free:** training never integrates the ODE. It works because regressing on the per-pair *conditional* velocity has the same gradient as regressing on the intractable *marginal* velocity field (Lipman et al.), so the optimum is the marginal field $\mathbb E[x_1 - x_0 \mid x_t]$ and the loss does not go to zero.
- **Sampling:** integrate from $t = 0$ to 1, e.g. Euler $x_{t+\Delta t} = x_t + \Delta t\,v_\theta(x_t, t)$. The paths are straighter than typical diffusion paths, so fewer steps are needed. **Reflow** (retraining on the model's own noise→sample couplings) straightens them further toward 1-step.
- **Why straight means fewer steps:** an ODE solver's per-step error grows with how fast the velocity changes along the step (Euler's local error ≈ $\tfrac{\Delta t^2}{2}\|\ddot x\|$). On a straight path traversed at constant speed the velocity is constant, so even one Euler step is exact. Diffusion's probability-flow ODE under typical schedules is more curved, so it needs more steps or higher-order solvers.
- Not perfectly straight, though: the per-pair straight lines cross each other. The learned *marginal* flow averages them, and ODE trajectories cannot cross, so its paths are curved. Reflow reduces that curvature.
- **Relation to diffusion:** Gaussian diffusion is also a flow, along curved paths. The two are equivalent up to schedule and loss weighting, and flow matching with a Gaussian source ≈ diffusion with a particular schedule and a v-like target.
- **Conventions differ:** SD3 (Esser et al. 2024, "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis") writes $z_t = (1 - t)\,x_{data} + t\,\epsilon$, i.e. $t = 1$ is noise, so its velocity target $\epsilon - x_{data}$ has the opposite sign and sampling runs from $t = 1$ to 0. It samples $t$ from a logit-normal to emphasize mid-range times. Flux also uses rectified flow.
- → practice: v4/generative/flow-matching

### Q6.14 Autoregressive image generation: how and trade-offs?
- **Factorization:** $p(x) = \prod_i p(x_i \mid x_{<i})$.
  - Over pixels: PixelRNN/PixelCNN (van den Oord et al. 2016).
  - In practice, over **discrete VQ tokens** in raster order: VQGAN + transformer, DALL·E (2021, dVAE tokens), Parti, LlamaGen.
  - Gives exact likelihood in token space and reuses the LLM stack; native multimodal models (e.g. Chameleon) follow this route.
- **Downsides:** sequential decoding of hundreds to thousands of tokens is slow; raster order is arbitrary for 2-D data; errors accumulate; the tokenizer caps quality.
- **Variants:**
  - **MaskGIT/MUSE:** bidirectional masked-token modeling with parallel iterative decoding. Each step predicts all masked tokens, keeps the most confident ones and re-masks the rest on a cosine schedule. MaskGIT uses 8 steps for 256 tokens (vs 256 AR steps); Muse uses 24 base + 8 super-resolution steps.
  - **VAR** (Tian et al. 2024): next-*scale* prediction, coarse to fine.
  - **MAR** (Li et al. 2024, "Autoregressive Image Generation without Vector Quantization"): AR, including MaskGIT-style random-order masked AR, over continuous tokens, with a small per-token diffusion head as the loss. No VQ.
  - Hybrids in which an AR model conditions a diffusion decoder.
- **vs diffusion:** diffusion refines all latents in parallel over many steps and is strong on continuous perceptual quality. AR takes one forward pass per token, has a KV cache, and handles interleaved text + image naturally.

### Q6.15 FID, IS, KID, precision/recall, CLIP score: definitions and caveats?
- **FID** (Heusel et al. 2017): fit Gaussians to Inception-v3 pool features (2048-d) of real and generated sets and report the **squared** Fréchet (2-Wasserstein) distance between them: $\mathrm{FID} = \|\mu_r - \mu_g\|_2^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r\Sigma_g)^{1/2}\big)$. Lower is better. Captures fidelity and diversity jointly. Compute $\mathrm{Tr}(\Sigma_r\Sigma_g)^{1/2}$ from the eigenvalues of the symmetric $\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}$, in float64.
- **FID caveats:**
  - The estimator is biased upward, by roughly $c/N$ with a model-dependent $c$ (Chong & Forsyth 2020), so FIDs at different $N$ are not comparable and model rankings can change with $N$. Compare at equal $N$ (usually 50k) against the same reference statistics, or extrapolate to FID∞.
  - Sensitive to resizing and JPEG implementation details (use clean-fid).
  - ImageNet-trained features: domain mismatch for faces or medical images, and the metric is gameable by matching ImageNet class statistics.
  - Gaussian assumption (only first and second moments are compared).
  - Blind to memorization: copying the training set scores very well.
  - One number can't separate fidelity from diversity.
  - Alternatives: Fréchet distance on DINOv2 features, KID, CMMD (CLIP + MMD).
- **IS** (Salimans et al. 2016): $\exp\left(\mathbb E_x\,\mathrm{KL}(p(y \mid x)\,\|\,p(y))\right)$. High when each sample is confidently classified *and* classes are diverse. Never compares with real data, is blind to intra-class diversity, and is tied to ImageNet classes.
- **KID** (Bińkowski et al. 2018): an unbiased estimate of squared MMD with the cubic polynomial kernel $k(x, y) = (x^\top y/d + 1)^3$ on the same Inception features. No Gaussian assumption and no $N$-dependent bias, so it suits small $N$ (it is noisier, and can come out slightly negative).
- **Precision/recall** (Kynkäänniemi et al. 2019): k-NN balls in feature space approximate each manifold. *Precision* = fraction of generated samples inside the real manifold (fidelity). *Recall* = fraction of real samples inside the generated manifold (coverage).
- **Text-to-image:**
  - **CLIP score:** cosine similarity of image and prompt embeddings. CLIPScore (Hessel et al. 2021) is $2.5\cdot\max(\cos, 0)$; many T2I papers report $100\cdot\cos$, so check the convention. CLIP is weak at counting, spatial relations and attribute binding, so CLIP score misses exactly the compositional errors that matter.
  - Compositional benchmarks (GenEval, T2I-CompBench), VQA-based checks (TIFA, DSG), preference models (PickScore, ImageReward, HPSv2), and **human side-by-side** evaluation (the gold standard). See Q6.21.
- → practice: v4/generative/fid

### Q6.16 Likelihood vs sample quality: why can they disagree?
- Likelihood (NLL, or bits/dim $= -\log_2 p(x)/D$) measures how well the model *covers* the data. Sample quality measures what the samples *look like*.
- **They can diverge** (Theis, van den Oord & Bethge 2016, "A note on the evaluation of generative models"):
  - A mixture of 1% good model + 99% junk loses at most $\log 100 \approx 4.6$ nats *per image*, which is negligible per dimension, yet 99% of its samples are garbage.
  - Conversely, a model that memorizes the training set has perfect samples but poor test likelihood.
- Likelihood is dominated by imperceptible low-level detail: most bits go to high-frequency noise.
- **MLE = forward KL = mode-covering:** VAEs and AR models are diverse but can put mass on implausible regions. GAN-like objectives behave more mode-seeking: sharp but prone to dropping modes.
- **Who gives likelihoods:** AR models (exact), normalizing flows (exact), VAEs (a lower bound), diffusion (an ELBO, or exact via the probability-flow ODE with trace estimation), GANs (none).
- Pick metrics by use case: compression and density estimation → likelihood; content generation → perceptual metrics plus human evaluation.

### Q6.17 Compare the generative model families.

| Family | Sample quality | Coverage/diversity | Sampling cost | Likelihood | Training |
|---|---|---|---|---|---|
| VAE | Blurry as a generator; great as a latent autoencoder | Good | 1 pass | Lower bound (ELBO) | Stable |
| GAN | Sharp | Mode dropping | 1 pass | None | Unstable game |
| Diffusion / flow matching | State of the art | Good | Many steps (distillable) | ELBO / ODE | Stable, simple regression |
| AR over tokens | Strong at scale | Good | Sequential, 1 pass per token | Exact (token space) | Stable, LLM stack |
| Normalizing flow | Moderate | Good | 1 pass | Exact | Invertibility constraints |

- The "generative learning trilemma" is quality vs diversity vs speed (Xiao, Kreis & Vahdat 2022, Denoising Diffusion GANs).

### Q6.18 How do you customize or control a pretrained diffusion model?
- **Full or LoRA fine-tuning** on domain data. LoRA is standard for styles and subjects (cheap, swappable).
- **DreamBooth** (Ruiz et al. 2023): fine-tune on 3–5 subject images bound to a rare identifier token, with a **prior-preservation loss** on generated class images to prevent drift and overfitting.
- **Textual Inversion** (Gal et al. 2022): learn only a new token embedding; the model stays frozen.
- **ControlNet** (Zhang, Rao & Agrawala 2023): a trainable copy of the encoder blocks takes spatial conditions (edges, depth, pose) and connects to the frozen model through **zero-initialized convolutions**, so training starts from the unchanged model.
- **IP-Adapter:** image prompts via decoupled cross-attention.
- **Training-free:** SDEdit (noise an input to level $t$, then denoise), inpainting, DDIM-inversion-based editing, attention-map editing (prompt-to-prompt).
- **Preference tuning:** Diffusion-DPO, or reward fine-tuning against aesthetic or preference models (Q6.20).

### Q6.19 How do you get 1–4-step sampling? Consistency models and distillation.
- **Goal:** replace a 20–50-step ODE solve (Q6.8) with 1–4 network evaluations while keeping most of the quality.
- **Consistency models** (Song, Dhariwal, Chen & Sutskever, ICML 2023, "Consistency Models"):
  - Learn $f_\theta(x_t, t)$ that maps *every* point of a probability-flow ODE trajectory to that trajectory's origin, so $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ along it (self-consistency).
  - The boundary condition $f_\theta(x, \epsilon) = x$ is built into the parameterization $f_\theta = c_{skip}(t)\,x + c_{out}(t)\,F_\theta(x, t)$, with $c_{skip}(\epsilon) = 1$ and $c_{out}(\epsilon) = 0$.
  - **Consistency distillation (CD):** take one ODE-solver step with a pretrained diffusion teacher from $x_{t_{n+1}}$ to $\hat x_{t_n}$, then minimize $d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\ f_{\theta^-}(\hat x_{t_n}, t_n)\big)$. Here $\theta^-$ is an EMA, stop-gradient copy, and $d$ is L2 or LPIPS (pseudo-Huber in later work).
  - **Consistency training (CT):** no teacher. Use $x + t_{n+1}z$ and $x + t_nz$ with the same $z$ in place of the solver step.
  - **Sampling:** one step, $\hat x_0 = f_\theta(x_T, T)$; or multistep: re-noise the output to a smaller $t$ and apply $f_\theta$ again, trading compute for quality.
  - The original paper reports one-step FID 3.55 on CIFAR-10 and 6.20 on ImageNet 64 (CD). Later training recipes (iCT; continuous-time sCM) narrowed the gap to multi-step diffusion.
- **LCM** (Luo et al. 2023, "Latent Consistency Models"): consistency distillation in SD's latent space with the CFG scale folded into the teacher's ODE, so the 2–4 sampling steps need no separate unconditional pass. LCM-LoRA packages it as an adapter.
- **Other families:**
  - **Progressive distillation** (Salimans & Ho 2022): a student matches two teacher steps with one; repeat to halve the step count each round. This is where v-prediction came from, because ε-pred is unstable at few steps.
  - **Adversarial:** ADD / SDXL-Turbo (Sauer et al. 2023) and LADD add a GAN loss so that 1–4-step samples stay sharp.
  - **Distribution matching:** DMD (Yin et al. 2024) pushes a one-step generator's output distribution toward the teacher's, using the difference between the teacher's score and a "fake" score fit to the student's samples.
  - **Straighter paths:** rectified-flow reflow (e.g. InstaFlow), so that one Euler step is accurate.
- **Trade-offs:** 1-step samples lose some diversity and fine detail (adversarial losses are mode-seeking); every new base model needs another distillation run; and guidance or editing tricks designed for multi-step samplers may not carry over.
- → practice: v4/generative/flow-matching (watch quality vs Euler step count), v3/modern-architectures/ddim-cfg

### Q6.20 How do you align a diffusion model with human preferences or rewards?
- **Why:** pretraining matches the training distribution, not what users want (aesthetics, prompt following, legible text). It is the same gap that RLHF closes for LLMs (Q7.4).
- **Reward signals:**
  - Preference models trained on pairwise human choices: PickScore, ImageReward, HPSv2.
  - Aesthetic predictors.
  - Programmatic checks: detector- or VQA-based prompt alignment (GenEval-style counts, colors, positions), or OCR accuracy for text rendering.
- **Three method families:**
  1. **RL on the sampling chain.** DDPO (Black et al. 2023) treats the denoising steps as an MDP. Each step is a Gaussian with a tractable log-probability, so policy gradients (REINFORCE, or PPO-style clipped importance ratios) apply, with the final image's reward. DPOK (Fan et al. 2023) adds a KL penalty to the pretrained model. Flow-GRPO (Liu et al. 2025) brings GRPO to flow-matching models by replacing the deterministic ODE sampler with an SDE that has the same marginals, which supplies the randomness RL needs.
  2. **Direct reward backpropagation.** ReFL (from the ImageReward paper), DRaFT and AlignProp backpropagate a differentiable reward through the last one or few sampling steps (or all of them, with checkpointing). Sample-efficient but memory-hungry, and the quickest to reward-hack.
  3. **Offline preference optimization.** Diffusion-DPO (Wallace et al. 2024) replaces the intractable image log-likelihood in DPO with the ELBO, so the loss compares per-step denoising errors: relative to the frozen reference, the model should improve more on the preferred image than on the rejected one. They fine-tuned SDXL on Pick-a-Pic (851K pairs). There is no reward model and no sampling during training.
- **Pitfalls:**
  - Reward hacking: oversaturated, over-smoothed or same-looking images that score well. Counter with a KL or reference penalty, early stopping and several rewards.
  - Diversity collapse onto the reward model's favorite style.
  - **Evaluate with a different metric (and humans) than the one you optimized.**
- → practice: v3/alignment-training/dpo, v3/alignment-training/grpo (the same losses in the LLM setting)

### Q6.21 How would you evaluate a text-to-image model?
- **Measure separate axes:** image quality and realism, text–image alignment, diversity, safety and bias, and cost (steps, latency). CFG scale and step count move all of them, so report both and sweep them (e.g. FID vs CLIP-score curves).
- **Distribution metrics:** FID (or FD on DINOv2 features) on a fixed caption set, classically zero-shot MS-COCO 30k, plus precision/recall. COCO FID rewards matching COCO's photo style, not what people prefer: in the Pick-a-Pic study (Kirstain et al. 2023), ranking models by COCO FID even anti-correlated with human preference.
- **Alignment metrics:**
  - **CLIPScore:** cheap, but CLIP often behaves like a bag of words (weak on counting, spatial relations, attribute binding and negation), and it saturates for strong models.
  - **GenEval** (Ghosh et al. 2023): templated prompts whose images an object detector checks for objects, counts, colors, positions and attribute binding. T2I-CompBench is similar.
  - **VQA-based:** TIFA and DSG have an LLM turn the prompt into questions that a VQA model answers. VQAScore (Lin et al. 2024) is the probability that a VLM answers "yes" to "Does this figure show '{prompt}'?".
  - **Learned preference scores:** PickScore, ImageReward, HPSv2. Closest to human taste, but gameable, especially if you also trained against them.
- **Human evaluation (the final word):**
  - Side-by-side against a baseline on a fixed prompt set (PartiPrompts, DrawBench, plus your product's prompts).
  - Ask about alignment and quality separately, randomize left/right order, and use several raters per item.
  - Report inter-rater agreement and win rates with confidence intervals, or Elo / Bradley–Terry ratings across many models.
- **Hygiene:** fixed seeds and prompts; slice by prompt category (text rendering, people, counting); check memorization against nearest training neighbors; never report the reward you optimized as your headline metric.
- → practice: v4/generative/fid

> **Say it in one breath**
> - VAE: ELBO = reconstruction − KL(q‖prior); the reparameterization trick gives pathwise gradients; collapse and blur are the classic failure modes. GAN: $D^\star = p_d/(p_d + p_g)$ gives JSD; use the non-saturating G loss; WGAN-GP or spectral norm help with vanishing gradients and instability (mode collapse is reduced, not solved).
> - DDPM: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon$, train with MSE on ε, sample ancestrally. DDIM uses the same model with a deterministic ODE sampler and far fewer steps. CFG: $\epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$ trades diversity for fidelity.
> - Modern stack: a latent autoencoder ($f = 8$), a DiT backbone with adaLN-Zero, and a flow-matching / rectified-flow objective (velocity = data − noise along straight paths), distilled for few-step sampling.
> - Evaluate with FID (squared Fréchet distance between Gaussians fit to Inception features; biased upward at small $N$ and implementation sensitive), precision/recall to separate fidelity from coverage, CLIP score plus compositional checks (GenEval, VQA-based) for alignment, and humans as the final judge. Likelihood ≠ sample quality. Few-step sampling comes from consistency, adversarial or distribution-matching distillation; preference tuning uses DDPO-style RL, reward backprop or Diffusion-DPO.

---

## 7. LLM essentials

### Q7.1 What is the LLM pretraining objective? What is perplexity?
- **Causal LM:** maximize $\sum_t\log p_\theta(x_t \mid x_{<t})$, i.e. next-token cross-entropy with teacher forcing. All positions train in parallel under a causal mask.
- **Perplexity** $= \exp(\text{mean per-token NLL})$. It is only comparable under the same tokenizer; across tokenizers, use bits per byte or per character.
- **Other objectives:** masked LM (BERT: 15% of tokens masked, bidirectional), span corruption (T5), prefix-LM, fill-in-the-middle (code).
- **Why decoder-only next-token prediction won:** the simplest objective that scales, every token is a training signal, generation is native, and in-context learning emerges with scale.
- → practice: llm/SmolLM

### Q7.2 How does BPE tokenization work?
- **Training:** start from characters or bytes. Count adjacent symbol pairs, merge the most frequent pair into a new symbol, and repeat until the vocab reaches its target size (about 32k–256k). Encoding replays the merges in learned order.
- **Byte-level BPE** (GPT-2): the base vocab is 256 bytes, so there are no unknown tokens. A regex pre-tokenizer splits on spaces and punctuation first.
- **Alternatives:** WordPiece (BERT) picks merges by likelihood gain. Unigram LM (SentencePiece) starts large and prunes. SentencePiece treats whitespace as a symbol (▁), which makes it language-agnostic.
- **Trade-offs:**
  - A bigger vocab gives shorter sequences (cheaper, more effective context) but larger embedding and softmax matrices and undertrained rare tokens.
  - Tokenization quirks hurt arithmetic (digit splitting), non-English efficiency and code whitespace, and produce "glitch tokens".
- For images, "tokens" are patches (continuous) or VQ codes (discrete); see Q5.8.
- → practice: llm/Byte-Pair-Encoder

### Q7.3 Scaling laws and Chinchilla?
- Loss falls as a power law in parameters $N$, data $D$ and compute $C$ (Kaplan et al. 2020, "Scaling Laws for Neural Language Models"), so you can extrapolate from small runs. Training compute is $C \approx 6ND$.
- **Chinchilla** (Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models"):
  - For a fixed budget, scale $N$ and $D$ roughly *equally*: $N_{opt} \propto C^{\approx 0.5}$ and $D_{opt} \propto C^{\approx 0.5}$, which gives **about 20 training tokens per parameter**.
  - Chinchilla (70B params, 1.4T tokens) beat Gopher (280B, 300B tokens) at similar compute.
  - Kaplan had recommended growing $N$ much faster ($N_{opt} \propto C^{0.73}$, $D_{opt} \propto C^{0.27}$). Hoffmann et al. blamed Kaplan's fixed LR schedule (not matched to run length); later analyses add counting only non-embedding parameters at small scale (Pearce & Song 2024) and last-layer FLOPs, warmup and optimizer tuning (Porian et al. 2024).
- **Fitted form:** $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$ (Chinchilla's fit: $\alpha \approx 0.34$, $\beta \approx 0.28$).
- **Exact constants are debated:** a 2024 replication (Besiroglu et al.) found that the parametric fit (Approach 3) was inconsistent with the other two approaches; their refit agrees with them. The ratio also depends on data quality, tokenizer and architecture. Treat "20" as a rule of thumb.
- **In practice models are over-trained** past Chinchilla-optimal, e.g. Llama 3 8B on about 15T tokens (≈1,900 tokens/param), because *inference* cost favors smaller models trained longer.
- Vision has similar laws: ViT scaling (Zhai et al. 2022, "Scaling Vision Transformers"), CLIP scaling (Cherti et al. 2023), DiT quality vs GFLOPs.

### Q7.4 SFT vs RLHF vs DPO?
- **SFT:** next-token loss on (prompt, ideal response) demonstrations, applied only to response tokens. It teaches format and behavior, but it is capped by demo quality and is pure imitation.
- **RLHF** (InstructGPT, Ouyang et al. 2022):
  1. SFT.
  2. Train a reward model on pairwise preferences with a Bradley–Terry loss: $-\log\sigma(r(x, y_w) - r(x, y_l))$.
  3. Optimize the policy with PPO to maximize $\mathbb E[r] - \beta\,\mathrm{KL}(\pi_\theta\,\|\,\pi_{ref})$. The KL keeps the policy near the SFT model and limits reward hacking.
  - Expensive: four models (policy, reference, reward, value), on-policy sampling, finicky tuning.
- **DPO** (Rafailov et al. 2023):
  - The KL-regularized optimum is $\pi^\star \propto \pi_{ref}\exp(r/\beta)$, so $r = \beta\log\frac{\pi^\star}{\pi_{ref}} + \beta\log Z(x)$.
  - Substituting into Bradley–Terry cancels $Z$:
    $\mathcal L_{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$.
  - Offline, with no reward model and no sampling: simple and stable.
  - Cons: limited to fixed (off-policy) preference data, can lower the likelihood of both responses, and is sensitive to $\beta$.
- **Newer:** GRPO (DeepSeekMath; group-relative advantages from several samples per prompt, no value network), RL with verifiable rewards for reasoning. For images: Diffusion-DPO, DDPO-style reward fine-tuning.
- → practice: v3/alignment-training/dpo, v3/alignment-training/ppo-rlhf, v3/alignment-training/grpo

### Q7.5 Decoding strategies: greedy, beam, temperature, top-k, top-p?
- **Greedy:** take the argmax each step. Deterministic, and repetitive or degenerate for open-ended text.
- **Beam search:** keep the top $B$ partial sequences by cumulative log-probability (with length normalization). Good when there is a "correct" output (translation, captioning, ASR); bland for open-ended generation.
- **Temperature:** $p_i \propto \exp(z_i/T)$. $T < 1$ sharpens, $T > 1$ flattens, $T \to 0$ is greedy.
- **Top-k:** sample from the $k$ most likely tokens, renormalized. A fixed $k$ is wrong whenever the distribution's shape varies.
- **Top-p / nucleus** (Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration"): sample from the smallest set with cumulative probability ≥ $p$ (e.g. 0.9). It adapts to the shape of the distribution.
- Also min-p, typical sampling and repetition penalties. The order of filters varies by implementation (commonly temperature → top-k → top-p → sample).
- → practice: v3/llm-decoding/temperature-sampling, v3/llm-decoding/top-k-sampling, v3/llm-decoding/top-p-sampling, v3/llm-decoding/beam-search

### Q7.6 Explain LoRA (and QLoRA).
- **LoRA** (Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"):
  - Freeze $W \in \mathbb R^{d\times k}$ and learn $\Delta W = \frac{\alpha}{r}BA$, with $B \in \mathbb R^{d\times r}$ (**initialized to 0**), $A \in \mathbb R^{r\times k}$ (random) and $r$ about 4–64.
  - The forward pass is $h = Wx + \frac{\alpha}{r}BAx$. Because $B = 0$, training starts exactly at the pretrained model.
  - Trainable params are $r(d + k)$ instead of $dk$. For $d = k = 4096$, $r = 16$: 131K vs 16.8M (≈0.8%). Optimizer state shrinks with it.
- **Inference:** merge $W + \frac{\alpha}{r}BA$ for zero extra latency, or keep adapters separate and hot-swap them.
- **Where to apply:** the original paper used $W_q$ and $W_v$; applying it to all linear layers, including the MLP, usually works better. The motivating hypothesis is that fine-tuning updates have low intrinsic rank; it holds well for light adaptation, while LoRA can lag full fine-tuning on large-scale code and math training (Biderman et al. 2024, "LoRA Learns Less and Forgets Less").
- **QLoRA** (Dettmers et al. 2023): a 4-bit NF4 base model, double quantization of the scales and paged optimizers, with bf16 LoRA on top. Fine-tunes a 65B model on one 48 GB GPU.
- **In vision and generative work:** LoRA for SD/DiT style and subject tuning, and for ViT/VLM adaptation. Variants include DoRA.
- → practice: v3/alignment-training/lora

### Q7.7 Quantization basics?
- **Affine quantization:** $q = \mathrm{clamp}(\mathrm{round}(x/s) + z,\ q_{min}, q_{max})$, dequantized as $\hat x = s(q - z)$.
  - Symmetric int8: $z = 0$, $s = \max|x|/127$.
  - Asymmetric: $s = (x_{max} - x_{min})/255$, with $z$ the zero-point.
- **Granularity:** per-tensor < per-channel (standard for weights) < per-group (e.g. 64–128 weights share a scale; common for 4-bit). Finer is more accurate but adds scale overhead.
- **PTQ vs QAT:** PTQ calibrates scales on a small dataset without retraining. QAT simulates quantization in the forward pass (fake-quant) and uses a straight-through estimator for rounding; it is best at low bit-widths.
- **Weight-only** (W8A16 / W4A16: GPTQ with second-order error-compensating rounding; AWQ with activation-aware protection of salient channels): saves memory and bandwidth, which is exactly what memory-bound decoding needs.
- **Weights + activations** (W8A8): SmoothQuant moves activation outliers into the weights via per-channel scaling; LLM.int8() keeps outlier dimensions in fp16. This uses int8 tensor cores for compute-bound work. FP8 (E4M3/E5M2) is the Hopper-era alternative.
- **The core difficulty** is activation **outliers**: a few channels with huge magnitudes blow up per-tensor scales.
- Also: KV-cache quantization. For diffusion models, activation ranges change with timestep, which complicates PTQ.
- → practice: v4/systems/int8-quantization, torch/easy/quantize-lm

> **Say it in one breath**
> - LLMs are next-token cross-entropy over BPE tokens. Compute is $C \approx 6ND$. Chinchilla-optimal is about 20 tokens/param, but production models are over-trained because inference cost dominates.
> - Alignment pipeline: SFT for imitation, then RLHF (Bradley–Terry reward model + PPO with a KL penalty) or DPO (closed-form reparameterization, a logistic loss on log-ratio differences, offline).
> - LoRA: $W + \frac{\alpha}{r}BA$ with $B = 0$ at init, about 1% trainable, mergeable. Quantization: scale + zero-point at per-channel or per-group granularity. Weight-only quantization speeds memory-bound decoding; activation outliers are the hard part.

---

## 8. Systems & efficiency

### Q8.1 Data vs tensor vs pipeline parallelism: when to use each?
- **Data parallel (DDP):** a full replica per GPU, the batch split across GPUs, gradients all-reduced. Simple, but the model plus optimizer must fit on one GPU.
- **Tensor parallel** (Megatron-LM, Shoeybi et al. 2019): split individual matmuls across GPUs.
  - MLP: the first linear is column-split, GeLU runs locally, the second linear is row-split. That costs **one all-reduce in forward (and one in backward) per MLP**.
  - Attention is split by heads.
  - It communicates every layer, so keep it within a node (NVLink); TP ≤ 8 is typical.
- **Pipeline parallel:** consecutive layers live on different GPUs, with micro-batches flowing through.
  - The GPipe bubble fraction is $\frac{p - 1}{m + p - 1}$ ($p$ stages, $m$ micro-batches), so you need $m \gg p$.
  - The 1F1B schedule bounds activation memory; interleaved stages shrink the bubble.
  - Only point-to-point communication, so it works across nodes.
- **Also:** sequence/context parallelism (long sequences, ring attention), expert parallelism (MoE all-to-all), ZeRO/FSDP (sharded DP).
- **Typical 3D layout:** TP within a node × PP across nodes × DP/FSDP across replicas.
- → practice: v3/gpu-systems/ring-attention, v3/data-loading/data-parallel-loading

### Q8.2 How does DDP work? What does ring all-reduce cost?
- **DDP:**
  - Each rank starts from the same init (broadcast from rank 0) and gets a different data shard (`DistributedSampler`).
  - Forward and backward run locally; gradients are **averaged by all-reduce**, so every replica takes the identical step.
  - Gradients go into buckets (default 25 MB), and each bucket's all-reduce launches as soon as it is ready, **overlapping communication with backward**. Use `no_sync()` to skip all-reduces during gradient accumulation.
- **Ring all-reduce** = reduce-scatter + all-gather over $N$ GPUs in a ring, with the buffer split into $N$ chunks.
  - It takes $2(N - 1)$ steps, and each GPU sends (and receives) $2\frac{N - 1}{N}S$ bytes for an $S$-byte buffer.
  - So the bandwidth cost is nearly independent of $N$. Latency grows with $N$, which is why NCCL also uses trees and hierarchical schemes.
- **Pitfalls:**
  - BN statistics are per-GPU unless you use SyncBN.
  - Unused parameters need `find_unused_parameters`.
  - Effective batch = per-GPU batch × world size, so re-tune the LR.
  - Gather metrics across ranks before reporting.
- → practice: v4/systems/ring-allreduce

### Q8.3 ZeRO and FSDP?
- Plain DP replicates everything: about 16 bytes/param per GPU for mixed-precision Adam (Q8.5).
- **ZeRO** (Rajbhandari et al., SC 2020) partitions training state across $N$ DP ranks:
  - **Stage 1** shards the optimizer states (12 B/param: fp32 master weights, m, v).
  - **Stage 2** also shards gradients.
  - **Stage 3** also shards parameters.
  - Per-GPU memory goes from $16\Psi$ to $4\Psi + 12\Psi/N$ (S1), then $2\Psi + 14\Psi/N$ (S2), then $16\Psi/N$ (S3), where $\Psi$ is the param count.
- **Communication:** S1 and S2 move the same volume as DDP (reduce-scatter grads + all-gather params ≈ one all-reduce). S3 moves about **1.5×**, because it all-gathers params in forward and again in backward.
- **FSDP** (PyTorch) is roughly ZeRO-3:
  - Wrap the model into units (e.g. per transformer block).
  - All-gather a unit's full params just before its forward/backward, then free them.
  - Reduce-scatter gradients so each rank updates only its shard.
  - Prefetching overlaps communication with compute.
- HSDP shards within a node and replicates across nodes. Offload (ZeRO-Offload/Infinity) spills to CPU or NVMe.
- → practice: v3/gpu-systems/fsdp

### Q8.4 Activation checkpointing: cost and benefit?
- **Mechanism:** activations saved for backward dominate memory at large batch or sequence length. Checkpointing keeps only some tensors (e.g. each block's input) and recomputes the block's forward during backward.
- **Cost:** about one extra forward pass, roughly **+33% compute** (forward ≈ 1/3 of forward + backward) for full recompute.
- **Memory:** goes from $O(L)$ to about $O(\sqrt L)$ with $\sqrt L$ segments (Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost").
- **Selective recomputation** (Korthikanti et al. 2022): recompute only the memory-heavy but cheap parts (attention softmax and dropout) to get most of the savings for a few percent of compute. FlashAttention already avoids storing $n\times n$.
- **MFU accounting:** MFU does not count recompute FLOPs, so checkpointing lowers MFU (longer steps, same useful FLOPs), while HFU, which counts them, stays roughly flat.
- → practice: v3/alignment-training/gradient-checkpointing

### Q8.5 How much GPU memory does training a model need?
- **Mixed-precision Adam:** 2 (bf16/fp16 weights) + 2 (grads) + 4 (fp32 master) + 4 (m) + 4 (v) = **16 bytes/param** (the ZeRO paper's $2 + 2 + 12$).
  - Standard PyTorch AMP is also about 16: fp32 params (4) + fp32 grads (4) + m + v (8).
  - So a **7B model needs about 112 GB before activations**, more than a single 80 GB GPU without sharding, offload or LoRA. State your assumptions; 8-bit optimizers, for example, change the numbers.
- **Inference weights:** 7B is 14 GB in bf16, about 7 GB in int8 and about 3.5 GB in int4, plus KV cache (Q4.7) and activations.
- **Activations** (transformer, per layer, 16-bit, no recompute or parallelism): about $sbh\,(34 + 5as/h)$ bytes, with $s$ = sequence length, $b$ = batch, $h$ = hidden size, $a$ = heads (Korthikanti et al. 2022). The $5as^2b$ term is the attention matrix, which FlashAttention removes.
- **Also budget for:** CUDA context and workspace, allocator fragmentation, communication buffers, and the **logits** ($b\cdot s\cdot V$, large for big vocabularies).
- → practice: v4/systems/memory-flops-estimation

### Q8.6 Where does $6ND$ come from? Estimate a training run.
- **Forward:** each parameter does one multiply-add per token, about $2N$ FLOPs per token.
- **Backward:** about 2× forward (gradients with respect to activations *and* weights), so $4N$.
- **Total:** **$6N$ per training token**, so $C \approx 6ND$. Inference is about $2N$ per generated token.
- **Attention correction:** $6N$ ignores the attention-score FLOPs, which add roughly a factor $n_{ctx}/(6d)$ on top (full attention; about half with causal-skipping kernels). Small unless the context is long relative to the width.
- **Worked example:** 7B params on 2T tokens.
  - $6 \cdot 7\times10^9 \cdot 2\times10^{12} = 8.4\times10^{22}$ FLOPs.
  - An H100 SXM peaks at about 989 TFLOP/s dense bf16; at 40% MFU that is about $4\times10^{14}$ FLOP/s.
  - So about $2.1\times10^8$ GPU-seconds ≈ **59k GPU-hours** (about 25 days on 100 GPUs).
- **CNNs:** FLOPs per image come from the conv formula (Q3.1); training costs about 3× forward.
- **Diffusion:** a training step is one forward + backward at a random $t$. Sampling costs (#steps × 2 if using CFG) forward passes, so diffusion is inference-heavy.
- → practice: v4/systems/memory-flops-estimation

### Q8.7 What is MFU and how do you improve it?
- **Definition:** Model FLOPs Utilization (from the PaLM paper, Chowdhery et al. 2022) = (tokens/s × model FLOPs per token, e.g. $6N$ + attention) / peak hardware FLOP/s. Recompute is **not** counted. HFU (hardware FLOPs utilization) does count it.
- **Typical values:** about 35–55% for well-tuned large-scale LLM training on A100/H100 (PaLM reported 46.2%). Small models and short sequences score much lower.
- **Where it's lost:** non-overlapped communication, pipeline bubbles, memory-bound ops (norms, activations, softmax), small low-intensity matmuls, dataloader stalls, kernel-launch overhead, recompute.
- **Improve with:** larger micro-batches, fused kernels (FlashAttention, fused optimizers), `torch.compile`, CUDA graphs, a better parallel layout, overlapped communication, a faster input pipeline.
- → practice: torch/easy/benchmark

### Q8.8 Compute-bound vs memory-bound: what is arithmetic intensity?
- **Arithmetic intensity (AI)** = FLOPs / bytes moved to or from memory.
- **Roofline:** attainable FLOP/s = $\min(\text{peak FLOP/s},\ \text{AI}\times\text{bandwidth})$. The ridge point is peak / bandwidth:
  - A100 SXM 80 GB: 312 TFLOP/s dense bf16 / about 2.0 TB/s ≈ 150 FLOP/B.
  - H100 SXM: 989 TFLOP/s dense bf16 / 3.35 TB/s ≈ 300 FLOP/B (about 600 for dense fp8).
  - NVIDIA datasheets often headline the 2:4-sparsity figures, which are 2× the dense ones.
- **Matmul** $(m\times k)(k\times n)$ in bf16: $2mkn$ FLOPs over about $2(mk + kn + mn)$ bytes.
  - Big square matmuls are compute-bound.
  - Batch-1 decoding ($m = 1$) has **AI ≈ 1 FLOP/B** (2 FLOPs per 2-byte weight), so it is extremely memory-bound. AI grows about linearly with batch size, which is why batching helps decode.
- **Memory-bound:** elementwise ops, norms, softmax, depthwise convs, small-batch decoding, embedding lookups. Fix with kernel fusion, larger batches, quantization (fewer bytes) and tiling (FlashAttention).
- **Compute-bound:** big GEMMs and convs, prefill, large-batch training. Fix with lower-precision tensor cores (bf16/fp8), fewer FLOPs (smaller or distilled models, sparsity) and better kernels.
- **Also:** communication-bound, and overhead-bound (Python and launch overhead at tiny sizes; use CUDA graphs or compile).
- → practice: v3/gpu-systems/triton-fused-softmax

### Q8.9 How do you speed up LLM/VLM inference?
- **Define the target first:** latency (time-to-first-token = prefill; time-per-output-token = decode), throughput (tokens/s/GPU) or cost. These trade off through batch size.
- **Batching:** static → dynamic → **continuous (iteration-level) batching** (Orca, Yu et al. 2022), where requests join and leave at every decode step. **PagedAttention** (vLLM, Kwon et al. 2023) stores the KV cache in fixed-size blocks behind a block table: near-zero fragmentation, and sharing across beams and prefixes.
- **KV cache:** GQA/MQA, KV quantization, prefix caching, chunked prefill.
- **Quantization:** int8/int4 weights for memory-bound decode; fp8 compute.
- **Speculative decoding** (Leviathan et al. 2023; Chen et al. 2023):
  - A cheap draft model proposes $\gamma$ tokens, and the target scores them all in one forward pass.
  - Accept token $x$ with probability $\min(1, p(x)/q(x))$. On the first rejection, resample from $\mathrm{norm}(\max(0, p - q))$ and discard the rest. If all $\gamma$ are accepted, sample one bonus token from the target's last distribution, so each target pass yields 1 to $\gamma + 1$ tokens.
  - The output distribution is **exactly** the target's. With per-token acceptance rate $\alpha$ (assumed i.i.d.), the expected tokens per target pass are $\frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$; the wall-clock speedup also depends on the draft's cost. It pays off when decoding is memory-bound (small batches) and shrinks at large batch sizes.
  - Variants: Medusa heads, EAGLE, prompt-lookup / n-gram drafts.
- **Smaller models:** distillation (Q8.10).
- **Kernels and runtime:** FlashAttention/FlashDecoding, fused ops, `torch.compile`, CUDA graphs; serving stacks such as vLLM, TensorRT-LLM and SGLang; tensor parallelism to cut latency.
- **VLM-specific:** the vision encoder's cost and the visual token count (token pruning or merging, dynamic resolution), plus caching image embeddings.
- → practice: v3/llm-inference/speculative-decoding, v3/llm-inference/continuous-batching, v3/llm-inference/inference-engine

### Q8.10 Knowledge distillation?
- (Hinton, Vinyals & Dean 2015, "Distilling the Knowledge in a Neural Network")
  - The student matches the teacher's *softened* distribution: $\mathcal L = (1 - \lambda)\,\mathrm{CE}(y, p_s) + \lambda\,T^2\,\mathrm{KL}\big(p_t^{(T)}\,\|\,p_s^{(T)}\big)$ with $p^{(T)} = \mathrm{softmax}(z/T)$.
  - The $T^2$ factor keeps gradient magnitudes comparable as $T$ changes.
  - "Dark knowledge": the relative probabilities of wrong classes encode similarity structure.
- **Variants:**
  - Feature or hint distillation (match intermediate features).
  - DeiT's distillation token.
  - Sequence-level KD for LMs (train on teacher generations) and on-policy distillation.
  - Distilling CLIP or DINOv2 into small ViTs.
- Remember Q2.10: Müller et al. (2019) found that teachers trained with label smoothing distill worse. Later work disputes how general this is.
- → practice: v3/modern-architectures/knowledge-distillation

### Q8.11 How do you make diffusion sampling fast?
- **Fewer steps, no retraining:** better solvers such as DDIM, DPM-Solver++, UniPC and Heun (EDM); about 10–50 steps instead of 1000 (higher-order solvers hold up better at the low end).
- **Step distillation:**
  - Progressive distillation repeatedly halves the steps.
  - Consistency models and LCM map any point on the ODE trajectory to its origin, giving 1–4 steps (Q6.19). DMD-style distribution matching goes to 1 step.
  - Adversarial diffusion distillation (SDXL-Turbo) adds a GAN loss for 1–4 step sampling.
  - Rectified-flow reflow.
- **Cheaper steps:**
  - Guidance distillation folds CFG into one pass (halving cost).
  - Lower precision (bf16/fp8, int8 PTQ).
  - Feature caching across adjacent steps (e.g. DeepCache), token merging.
  - A smaller or faster VAE decoder; `torch.compile` / TensorRT.
- **Systems:** batch requests, and keep the text-encoder and VAE costs in view, since they matter at few steps.

### Q8.12 Your GPU utilization is low during training. What do you check?
- **Profile first** (PyTorch profiler, `nvidia-smi dmon`, Nsight). Is the step time dominated by data loading, host-to-device copies, communication, or small kernels?
- **Input pipeline:**
  - `num_workers > 0`, `pin_memory=True`, `persistent_workers`, prefetching.
  - GPU JPEG decoding (DALI/nvJPEG) and pre-resized data.
  - Sharded sequential formats (WebDataset tar shards, Parquet) for streaming from object storage.
  - Cache preprocessed tensors, e.g. precompute **VAE latents and text embeddings** for diffusion training.
- **Compute:** batch size too small, too many tiny ops (fuse or compile), CPU-GPU syncs (`.item()`, prints), unnecessary fp32.
- **Distributed:** non-overlapped all-reduce, stragglers, slow interconnect.
- → practice: v3/data-loading/streaming-dataset, v3/data-loading/buffered-loading, v3/data-loading/preloaded-dataset

> **Say it in one breath**
> - DP replicates and all-reduces (ring cost $2\frac{N-1}{N}S$ per GPU, nearly independent of $N$). TP splits matmuls inside a layer (intra-node). PP splits layers (bubble $\frac{p-1}{m+p-1}$). ZeRO/FSDP shards optimizer state, then gradients, then params (stage 3 ≈ 1.5× DP communication).
> - Memory is 16 B/param for mixed-precision Adam (7B ≈ 112 GB before activations). Compute is $6ND$ for training and $2N$ per token for inference. MFU = useful FLOPs / peak, with 40–50% considered good.
> - Decoding is memory-bound (AI ≈ 1 at batch 1), so batch continuously, page and shrink the KV cache (GQA, quantization), quantize weights, and use speculative decoding (exact via accept with probability $\min(1, p/q)$).
> - Checkpointing trades about 33% compute for $O(\sqrt L)$ activation memory. Distillation and few-step samplers are the main levers for generative inference.

---

## 9. Research-scientist specific

### Q9.1 How do you read and critique a paper?
- **Three passes:**
  1. Abstract, figures and conclusion: what is the claim, and do I care?
  2. Method and main experiments.
  3. Re-derive the key equations and read the appendix and hyperparameters.
- **Critique checklist:**
  1. Does the evidence test the *claim* as stated?
  2. Are the baselines strong, recent and **tuned with equal effort and compute**?
  3. Do ablations isolate each component?
  4. Is variance reported (seeds, error bars)?
  5. Does it hold at other scales or datasets, or only on a toy setup?
  6. Are the metrics sound (FID implementation, benchmark contamination, cherry-picked samples)?
  7. Is the gain from the idea or from more data, compute or tuning?
  8. How novel is it relative to prior work?
  9. Are limitations and failure cases shown?
  10. Is it reproducible (code, configs)?
- Close with: "The one experiment I'd want to see is…".

### Q9.2 How do you design experiments and ablations?
- **Before running:** state a falsifiable hypothesis and the measurement that would distinguish the alternatives. Decide in advance what result would change your mind.
- **Baseline first:** reproduce known numbers, then change one variable at a time. Match compute, params and data across arms, and tune at least the LR per arm.
- **Ablation styles:** leave-one-out (remove a component from the full system) and add-one-in (add to the baseline). Do both if you suspect interactions. Add sanity controls such as shuffled labels, random features and an oracle upper bound.
- **Scale:** iterate on small proxies but **check trends at 2–3 scales**, because effects can vanish or flip with scale.
- **Hygiene:** hold out the test set until the end, log everything (config, git hash, data version), and sketch the final figure before you run.

### Q9.3 How do you know a result is real? Seeds, variance, significance.
- **Sources of variance:** init seed, data order, augmentation randomness, nondeterministic GPU kernels and finite evaluation sets.
- **Report** mean ± std (or a 95% CI) over several seeds, at least 3 and 5+ for small effects. Use the same seeds across methods (paired comparison). If the gain is below the seed-to-seed std, you can't claim it.
- **Finite test-set noise:** accuracy $p$ on $n$ examples has std $\sqrt{p(1 - p)/n}$. For $p = 0.8$, $n = 10{,}000$ that is ±0.4% (1σ), so a single-seed +0.3% is noise.
- **Bootstrap** over test examples for CIs on any metric (mAP, FID, win rates).
- **Tests:** paired t-test or Wilcoxon across seeds; McNemar for two classifiers on the same test set. Correct for multiple comparisons (Holm/Bonferroni), and prefer effect sizes with CIs over bare p-values.
- **Red flags:** best-seed reporting, checkpoint selection on test, max-over-runs.

### Q9.4 How do you make your work reproducible?
- **Determinism:**
  - Seed Python, NumPy, torch and CUDA.
  - Set `torch.use_deterministic_algorithms(True)` and `cudnn.benchmark=False`. This costs speed, and bitwise reproducibility across hardware or library versions still isn't guaranteed.
- **Version everything:** code (git hash), configs, data snapshot or hash and splits, environment (container, library versions), checkpoints. Log hyperparameters and metrics (W&B, MLflow, TensorBoard).
- **Report** the compute budget, number of runs, hyperparameter search space and selection rule. Release code and weights when possible.
- Reproduce the baseline before building on it, and unit-test the data pipeline and metric code.
- → practice: torch/basic/tensorboard, torch/basic/save-model

### Q9.5 How should you present your own research? (template)
- **Structure:**
  - **Problem:** one sentence on what it is, why it matters and to whom.
  - **Gap:** concretely, where existing methods fail.
  - **Insight:** the one key idea, the "aha".
  - **Method:** minimal, with one diagram or equation.
  - **Evidence:** the main result vs the strongest baseline, an ablation proving the insight is what matters, and one qualitative example.
  - **Limitations:** honest failure modes.
  - **Next:** what you'd do with more time or compute.
- **Prepare 2-, 10- and 45-minute versions.** Know every number on every slide, and have an answer to "why not X?" for each design choice.
- **Collaborative work:** state ownership clearly: "I did X; my co-author did Y."
- **Expect these questions:**
  - "Hardest part?"
  - "What would you do differently?"
  - "What surprised you?"
  - "How would it work at 100× scale or in production?"
  - "What's the strongest argument against your approach?"

### Q9.6 How do you answer "What would you do next?" and open-ended research prompts?
- **Structure your answer:**
  1. Clarify the goal and constraints.
  2. **Diagnose the bottleneck:** data, model capacity, objective, optimization or evaluation?
  3. Propose 2–3 hypotheses ranked by expected value per cost.
  4. Give the cheapest *decisive* experiment for each.
  5. Say what result would change your plan.
  6. Say how you would measure success.
- **Typical prompts and good angles:**
  - *"Great on the benchmark, bad in production."* Characterize the shift, slice errors, collect targeted data, set up monitoring (Q1.17).
  - *"Improve a VLM's OCR or fine-grained understanding."* Higher resolution or tiling, a better vision encoder (SigLIP), unfreeze the encoder, OCR-heavy data, a larger visual token budget.
  - *"The diffusion model ignores part of the prompt."* A stronger text encoder (T5), recaption the training data with a VLM (the DALL·E 3 approach), tune CFG, evaluate compositionally (GenEval), inspect cross-attention maps.
  - *"Scale it 10×."* Fit small-scale scaling curves first, find what saturates, and identify the infrastructure bottleneck (memory, communication, data).
  - *"It doesn't replicate."* Check variance and seeds, implementation and preprocessing differences, and the eval protocol.

### Q9.7 How do you do error analysis on a vision/multimodal model?
- **Slice the metric** by meaningful subgroups: class, object size, lighting, source or device, and demographics where relevant.
- **Inspect examples:** the highest-loss and most-confidently-wrong cases, the confusion matrix, and suspected label noise (confident-learning style).
- **Attribute each failure** to data (label noise, coverage gaps) or to the model (capacity, resolution, receptive field).
- **Prioritize** by frequency × impact, fix with targeted data first, and add a regression test set for each fixed failure mode.

### Q9.8 Sketch an end-to-end multimodal product (e.g. visual search). What are the ML decisions?
- **Requirements:** latency, scale (number of items and QPS), freshness, quality metric.
- **Model:** start from a pretrained CLIP/SigLIP dual encoder, then fine-tune contrastively on in-domain pairs (clicks, catalog text) with hard negatives.
- **Serving:**
  - Offline indexing with an ANN index (IVF-PQ or HNSW).
  - Online: embed the query, retrieve, then re-rank the top-K with a cross-encoder or VLM.
- **Evaluation:** offline Recall@K and nDCG on a labeled set; online A/B tests (CTR, conversion).
- **Monitoring and iteration:** embedding drift, hard-negative mining from logs, periodic re-indexing, safety filtering.
- → practice: v4/multimodal/vector-search

> **Say it in one breath**
> - To critique a paper, ask: does the evidence test the claim, are the baselines equally tuned and compute-matched, do ablations isolate the idea, are there seeds and error bars, and does it survive scale?
> - Good experiments follow hypothesis → strong reproduced baseline → change one thing → multiple seeds with paired comparisons and CIs → check at 2–3 scales. The test-set std is $\sqrt{p(1-p)/n}$.
> - Present research as problem → gap → insight → method → evidence → limitations → next. Answer "what next" by diagnosing the bottleneck and proposing the cheapest decisive experiment.

---

## 10. Numbers to know

| Quantity | Value |
|---|---|
| Initial loss, balanced $K$-way softmax | $\ln K$ (2.30 for 10, 6.91 for 1000) |
| Adam defaults | $\beta_1 = 0.9$, $\beta_2 = 0.999$ (LLMs: 0.95), $\epsilon = 10^{-8}$ |
| fp16 | max 65504; min normal ≈ $6.1\times10^{-5}$ (min subnormal ≈ $6\times10^{-8}$); 5 exp / 10 mantissa bits |
| bf16 | 8 exp / 7 mantissa bits, fp32 range, loss scaling usually unnecessary |
| Mixed-precision Adam memory | 16 bytes/param (7B ≈ 112 GB) |
| Inference weights | 2 B/param bf16; 1 B int8; 0.5 B int4 |
| Training / inference FLOPs | $6ND$ / $2N$ per token |
| Chinchilla | ≈ 20 tokens/param (rule of thumb; constants debated) |
| KV cache | $2\cdot L\cdot n_{kv}\cdot d_h\cdot$ bytes per token; Llama-2-7B fp16 ≈ 0.5 MiB/token |
| A100 | 312 TFLOP/s bf16 dense (624 with 2:4 sparsity); 19.5 fp32 (non-tensor-core); ≈ 2.0 TB/s HBM2e (80 GB SXM; 40 GB: 1.6 TB/s) |
| H100 SXM | ≈ 989 TFLOP/s bf16 dense, ≈ 1979 fp8 dense (datasheet headlines are 2× with sparsity); 80 GB, 3.35 TB/s HBM3 |
| Roofline ridge point (dense bf16) | A100 ≈ 150 FLOP/B; H100 SXM ≈ 300 FLOP/B; batch-1 decode AI ≈ 1 |
| MFU | ≈ 35–55% for well-tuned large LLM runs (PaLM 540B: 46.2%) |
| Ring all-reduce per-GPU traffic | $2\frac{N-1}{N}S$ |
| Activation checkpointing | ≈ +33% compute |
| Conv output size | $\lfloor (I + 2P - D(K-1) - 1)/S \rfloor + 1$ |
| ResNet-50 | ≈ 25.6M params, ≈ 4 GMACs at 224² |
| ViT-B/16 | 86M params, 196 patch tokens (+CLS) at 224² |
| Focal loss | $\gamma = 2$, $\alpha = 0.25$, prior $\pi = 0.01$ |
| COCO AP | mean over IoU 0.50:0.05:0.95; small < 32², large > 96² |
| MAE | 75% mask ratio; encoder sees 25% |
| CLIP | 400M pairs, batch 32k, $\tau_0 = 0.07$, logit scale ≤ 100 |
| SigLIP init | $t = 10$ ($t' = \log 10$), $b = -10$ |
| BLIP-2 / Flamingo / LLaVA-1.5 visual tokens | 32 / 64 / 576 |
| DDPM | $T = 1000$, $\beta$ linear $10^{-4}\to0.02$ |
| Cosine schedule offset | $s = 0.008$ |
| SD1.x latent | $f = 8$, 64×64×4 for 512², scale 0.18215 |
| DiT-XL/2 | FID 2.27 (ImageNet 256², class-conditional, CFG scale 1.5) |
| CFG scale | SD default 7.5 in the "$w = 1$ is conditional" convention; train with 10–20% condition dropout |
| VQ-VAE commitment | $\beta = 0.25$ |
| WGAN-GP | $\lambda = 10$ |
| Test-set accuracy std | $\sqrt{p(1-p)/n}$ (0.8 on 10k ≈ ±0.4%) |
