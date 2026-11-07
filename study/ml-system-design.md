# ML System Design — CV / Multimodal / Generative

Interview guide for **Applied MLE**, **Research Scientist (RS)** and **Research Engineer / ML-infra (RE)** loops in the CV / multimodal / generative domain. It assumes strong ML theory and little hands-on production experience, and is sized for a 4-week sprint.

> **About the numbers.** Everything here is back-of-envelope. Hardware specs are approximate and change every GPU generation; latencies and throughputs depend heavily on implementation. Every "assumption" is something you should **say out loud** in an interview and re-derive. Interviewers grade the reasoning, not the decimals.
>
> Notation: `≈` approximate · K/M/B/T = 10³/10⁶/10⁹/10¹² · GB = 10⁹ bytes (GiB = 2³⁰) · **1 MAC = 2 FLOPs** (many vision papers report MACs and call them "FLOPs"; ResNet-50's "4.1 GFLOPs" is ≈ 4.1 GMACs ≈ 8.2 GFLOPs) · 1 day = 86,400 s ≈ 10⁵ s.

**Contents**
- [0. How to use this in 4 weeks](#0-how-to-use-this-in-4-weeks)
- [1. The framework](#1-the-framework) — time budget, phase-by-phase questions, production vocabulary, one-page checklist
- [2. Case studies](#2-case-studies) — a. visual search · b. content moderation · c. text-to-image service · d. photo-library retrieval · e. AV data engine · f. multimodal training infra
- [3. Back-of-envelope cheat sheet](#3-back-of-envelope-cheat-sheet)
- [4. Pitfalls and practice prompts](#4-pitfalls-and-practice-prompts)

---

## 0. How to use this in 4 weeks

| Week | System-design work (aligned with `study/README.md`) | Drill |
|---|---|---|
| 1 | Read §1 twice; memorise the §1.6 checklist; read §3 and re-derive every number by hand | 1 estimate/day (QPS, storage, GPU count) |
| 2 | Cases **a** + **d**: write your own 35-minute answer first, then compare | Say the latency budget of case a out loud from memory |
| 3 | Cases **c** + **b**; one timed mock from §4.2 | Practise the cascade cost argument (b) and Amdahl argument (c) |
| 4 | Case **f** as a mock (RE angle), skim **e**; 2–4 more mocks from §4.2, alternating roles | Re-derive 16 B/param, 6·N·D, MFU and checkpoint-interval math without notes |

How to self-review a mock: score each of the 11 phases in §1.2 as *covered / shallow / missed*; check you produced (1) numbers with units, (2) a diagram, (3) a baseline, (4) offline + online metrics, (5) at least two explicit trade-offs, (6) monitoring.

---

## 1. The framework

### 1.1 What interviewers grade

- **Structure and driving**: you lead; you announce the plan and keep time.
- **Problem → metric link**: the business goal, the offline metric and the training loss are written down and connected.
- **Data thinking**: where labels come from, their cost and bias, how splits avoid leakage.
- **Baseline first, then iterate**: a simple system that works beats a clever one that does not ship.
- **Quantitative reasoning**: QPS, latency budget, memory, GPU count, cost — with units.
- **Trade-offs**: you name alternatives and say why you pick one *under these constraints*.
- **Production awareness**: serving, rollout, monitoring, failure modes, feedback loops.
- **Communication**: concise, checks in with the interviewer, adapts when steered.

| Role | What they probe hardest | Where to spend extra minutes |
|---|---|---|
| Applied MLE | end-to-end ownership: data/labels, metrics, serving, A/B tests, monitoring | data, serving, online eval |
| Research Scientist | problem formulation, model and objective choices, evaluation rigour, ablations, failure analysis, what is novel vs risky | ML framing, model, offline eval |
| Research Engineer / ML-infra | throughput, scaling, parallelism, reliability, cost, data pipelines, numbers | training infra, serving optimisation, back-of-envelope |

**The round itself often differs by role** (formats vary by company — ask the recruiter which flavour to expect):
- *Applied MLE*: the classic product ML design below (all 11 phases, end to end, with an A/B plan).
- *RS*: often a research-design discussion ("how would you approach X?") rather than a serving diagram — problem formulation, baselines, evaluation protocol, ablations, what result would falsify the idea; serving and cost get a minute, not seven.
- *RE / ML-infra*: often a **platform** prompt — a training run, an inference fleet, a data pipeline or an eval harness (case f, prompts 3 and 9 in §4.2) — graded on numbers: memory, FLOPs, bandwidth, failure rates, cost.

### 1.2 Time budget (45 minutes)

| # | Phase | Min | Clock | What should be on the board when you leave the phase |
|---|---|---|---|---|
| 1 | Clarify requirements | 5 | 0–5 | functional + non-functional requirements, numeric assumptions |
| 2 | Frame as an ML problem | 3 | 5–8 | inputs/outputs, label, loss ↔ offline metric ↔ business metric |
| 3 | Data | 7 | 8–15 | sources, labeling plan, quality, splits, leakage risks |
| 4 | Features / representations | 2 | 15–17 | preprocessing, embeddings, metadata features |
| 5 | Model | 7 | 17–24 | baseline → strong model, cascade, trade-offs |
| 6 | Training | 3 | 24–27 | loss, sampling/imbalance, scale/distribution |
| 7 | Offline evaluation | 4 | 27–31 | metrics, operating points, slices, ablations |
| 8 | Serving | 7 | 31–38 | architecture diagram, latency budget, cost math |
| 9 | Online evaluation | 2 | 38–40 | A/B design, primary + guardrail metrics |
| 10 | Monitoring and iteration | 3 | 40–43 | drift, feedback loops, retraining, fallbacks |
| 11 | Responsible AI | 2 | 43–45 | bias, safety, privacy (also mention inline earlier) |

Rules of thumb:
- **Breadth first.** Have an end-to-end sketch (boxes and arrows) by about minute 20, then deep-dive where the interviewer signals interest. Ask: "I can go deeper on the retrieval index or on the labeling loop — which is more useful?"
- If the interviewer jumps ahead, follow them, but **park** what you skipped ("I'll come back to monitoring") and return to it.
- Phases 4–6 often merge in practice. Phase 11 is cheapest when woven in (e.g., privacy during data, fairness slices during eval).
- A "45-minute" slot often leaves only ~35–40 minutes of design after introductions and your own questions. Then compress phases 4, 6 and 11 to about a minute each and trim data/model to ~5 minutes each; keep requirements, serving and monitoring.

### 1.3 Phase by phase: what to ask, what to do

#### Phase 1 — Clarify requirements (5 min)
**Ask**
- *Product goal*: who is the user, what action does the system enable, and what does success look like? ("more purchases from camera search" vs "fewer abandoned searches")
- *Scale*: users / DAU, requests per day, peak-to-average ratio, corpus/catalog size and growth.
- *Latency/throughput*: interactive (what p99?), near-real-time (seconds–minutes) or batch (hours)?
- *Freshness*: how quickly must new items, new content or new policies take effect?
- *Platform and budget*: server GPU, CPU-only, mobile/edge? Cost ceiling per request?
- *Privacy/legal*: PII, faces/biometrics, minors, data residency, retention, on-device requirement?
- *Existing system*: greenfield or replacing something? What is logged today (this decides your labels)?
- *Error costs*: is a false positive or a false negative worse? Is there a human in the loop?

**Do**: write 4–6 requirements and turn them into numbers immediately: "15M queries/day ≈ 174 QPS average, ≈ 520 peak (3×); p99 ≤ 400 ms server-side; new products searchable within 1 h."
**Avoid**: a 10-minute questionnaire. Ask, state assumptions, move on.

#### Phase 2 — Frame as an ML problem (3 min)
- Input → output, and task type: classification, detection, retrieval/ranking, generation, or a **cascade** of several.
- Supervision: explicit labels, implicit feedback (clicks — biased), weak or self-supervision.
- Write three things side by side: **training objective** (e.g., InfoNCE) → **offline metric** (Recall@10) → **online/business metric** (conversion per search). Say how you expect them to correlate and where they diverge.
- Ask whether ML is needed at all; name a non-ML or heuristic baseline (hash matching, rules, metadata filters).

#### Phase 3 — Data (7 min)
- **Sources**: product logs, catalog/corpus, licensed data, public datasets (check licences), synthetic data.
- **Labeling strategy** (cost × quality × speed):
  - Human labels: write guidelines, gold questions, multi-rater + adjudication, measure agreement (Cohen's/Fleiss' κ). Rough effort (approximate, task-dependent): image-level class ~1–5 s; bounding box ~7–35 s (≈ 7 s with "extreme clicking" vs ≈ 35 s for the classic draw-and-verify ImageNet protocol, Papadopoulos et al., 2017); instance polygon mask ~1–2 min (COCO reported 22 worker-hours per 1,000 segmentations ≈ 79 s per instance). Example: 1M boxes × 10 s ≈ 2,800 labeler-hours.
  - Implicit feedback: clicks, purchases, dwell, reports, appeals — cheap and plentiful but position-, exposure- and selection-biased.
  - Weak supervision: heuristics, metadata, distant labels (e.g., product category from catalog), noisy web alt-text.
  - Self-supervised / pretrained: CLIP/SigLIP/DINOv2-style backbones; fine-tune with few labels.
  - Model-assisted: pre-label with a large model, humans verify; active learning (uncertainty, disagreement, diversity) to pick what to label.
- **Quality**: duplicates/near-duplicates, corrupt files, label noise, class imbalance, distribution gap between training data (e.g., studio product shots) and production (user photos).
- **Label noise** (interviewers probe this): *measure* it — gold questions, inter-rater agreement, re-label a random sample with experts. *Reduce* it — clearer guidelines, consensus/adjudication on disagreements. *Train through* it — soft labels from rater votes, label smoothing or noise-robust losses, down-weighting or re-labelling examples the model confidently disagrees with (confident-learning-style tools such as cleanlab). *Clean the test set first*: label errors cap measurable accuracy and can flip model rankings (Northcutt et al., 2021 estimated ≥ 3.3 % errors on average across 10 popular test sets, ≥ 6 % in the ImageNet validation set).
- **Splits**: split by **entity** (user, product, scene, video, uploader) or by **time**, never just by image; dedupe near-duplicates across splits; keep a frozen golden test set.
- **Leakage checks**: same product/scene in train and test, bursts of near-identical photos, future information in features, **eval contamination of pretrained models** (web-scale pretraining data often contains benchmark images).
- Volume and storage estimate (one line).

#### Phase 4 — Features / representations (2 min)
- Pixels → pretrained embedding (frozen vs fine-tuned) vs end-to-end training.
- Side information: text (captions, OCR), metadata (time, location, category, price), context (user, device, surface).
- **Preprocessing parity** between training and serving (see training-serving skew in Phase 10).
- Online features need a store with point-in-time correctness for training (no peeking into the future).

#### Phase 5 — Model (7 min)
- **Baseline**: heuristic or pretrained model + linear probe / kNN. Say what it buys you: a working system, a logging pipeline, a metric baseline.
- **Strong model**: justify by the constraints (latency, data volume, modality). Examples: fine-tuned ViT/ConvNet, dual encoder for retrieval, cross-encoder/VLM for re-ranking, latent diffusion/flow for generation.
- **Cascade**: cheap high-recall stage on 100% of traffic → expensive high-precision stage on a few percent.
- **Trade-offs to name**: accuracy vs latency/cost; one multi-task model vs many specialised models; pretrained-and-adapted vs from scratch; interpretability/debuggability.

#### Phase 6 — Training (3 min)
- Loss matched to the task (CE/focal, contrastive InfoNCE/sigmoid, triplet/proxy losses, diffusion/flow-matching objective, pairwise ranking).
- Sampling: class reweighting or resampling (then **recalibrate**, because reweighting shifts probabilities), hard-negative mining, balanced batches across categories, augmentations that respect the task (colour jitter breaks colour-sensitive retrieval).
- Fine-tuning strategy: frozen backbone + head → partial unfreeze → full fine-tune; LoRA/adapters for many variants.
- Scale: DDP if the model fits on one GPU; FSDP/ZeRO when model + optimizer state do not fit; mention data-loader throughput for image/video (decode is often the bottleneck).
- Reproducibility: versioned data snapshots, config in version control, seeds, experiment tracking.

#### Phase 7 — Offline evaluation (4 min)

| Task | Offline metrics |
|---|---|
| Classification (imbalanced) | PR-AUC, recall at fixed precision (operating point), per-class P/R, calibration (ECE) |
| Detection | mAP@[.5:.95], AP by object size/distance, recall at fixed false positives per image |
| Segmentation | mIoU, boundary F-score |
| Retrieval | Recall@k, mAP@k, NDCG@k, MRR; **ANN recall vs exact kNN measured separately** |
| Ranking | NDCG@k, MRR, log-loss/calibration if scores are used downstream |
| Image generation | FID or CMMD (realism vs reference set), CLIP score (alignment), GenEval-style compositional checks, learned preference models, **human side-by-side** |
| Captioning / VQA | CIDEr, VQA accuracy, human or model-judge ratings |
| Efficiency | latency p50/p99 on target hardware, throughput, memory, $ per 1K requests |

- **Slices**: per class, per region/language, device/camera type, lighting, image source (camera vs screenshot), skin tone, new vs old items, long tail.
- **Ablations**: which component buys what (detector on/off, embedding dim, compression).
- **Error analysis**: look at the top failures by hand and bucket them; this drives the next data collection.
- **Uncertainty**: bootstrap confidence intervals; with rare positives, stratified sampling with inverse-probability weights.

#### Phase 8 — Serving (7 min)
- **Mode**: online synchronous, asynchronous queue, nearline streaming, or offline batch precompute. Precompute whatever you can (catalog embeddings, thumbnails).
- **Diagram**: client → gateway → preprocessing → model stages → post-processing/business rules → response, plus the offline/nearline paths (indexing, training, backfills).
- **Latency budget**: allocate the p99 across components and sum it.
- **Batching**: dynamic batching (max batch size + max wait of a few ms); bucket by shape (resolution, sequence length).
- **Caching**: CDN for static outputs, result cache keyed by (model version, normalized input), embedding cache, feature cache. Know when caching is useless (unique inputs) or unsafe (stale policy, privacy).
- **Retrieval → ranking cascade**: ANN over millions/billions → re-rank hundreds → business rules → top tens.
- **Compression**: BF16/FP16 → INT8/FP8 (calibrated post-training quantization or QAT), INT4 weights for large models, distillation into a smaller student, pruning, lower input resolution, fewer diffusion steps. Validate on slices, not just the aggregate.
- **Hardware**: GPU for large models and high throughput; CPU for small models, GBDT re-rankers and low QPS; edge NPU for privacy/offline/latency. Decide with cost per 1M requests (§3.6).
- **Rollout**: model registry → shadow mode → canary (~1%) → staged ramp with automatic rollback; keep model and index versions paired.
- **Reliability**: autoscaling on queue depth/GPU utilisation, warm pools (cold start of a GPU replica includes loading GBs of weights), timeouts, graceful degradation and fallbacks.

#### Phase 9 — Online evaluation (2 min)
- **A/B test**: randomisation unit — usually the **user** (session- or request-level randomisation lets one user see both arms — inconsistent experience, carry-over between arms — and cannot measure user-level outcomes such as retention); one primary metric fixed in advance, a few secondary metrics, **guardrails** (latency, errors, crash rate, revenue, safety incidents, cost).
- **Analyse at the randomisation unit**: with user-level assignment, a per-request ratio metric such as CTR (clicks / impressions) needs per-user aggregation or the **delta method** for its variance — treating requests as independent understates the variance and gives false positives.
- **Sample size** (Lehr's rule, α = 0.05 two-sided, 80 % power): n per arm ≈ 16·σ²/δ². For a 5 % click rate (σ² = p(1−p) = 0.0475) and a +2 % relative lift (δ = 0.001): n ≈ 16 × 0.0475 / 10⁻⁶ ≈ **760K samples per arm** (if you randomise by user, compute σ² on the per-user metric). **CUPED** (regress out each user's pre-experiment value of the metric) can cut the variance, and so n, substantially; cap/winsorise heavy-tailed metrics such as revenue.
- Pitfalls:
  - **Sample-ratio mismatch (SRM)**: χ² test of observed vs designed arm sizes; a very small p-value (e.g., < 0.001) means assignment or logging is broken — do not interpret the result until it is fixed.
  - **Novelty/primacy effects**: plot the effect by day or by days-since-first-exposure (it should stabilise); run whole weeks (≥ 1–2) to cover weekly seasonality.
  - **Peeking**: fix the horizon from the power calculation, or use a sequential test designed for continuous monitoring.
  - **Multiple comparisons** across many metrics/slices: pre-register, or correct (e.g., Benjamini–Hochberg).
  - **Interference** in marketplaces/social graphs: cluster-level (graph or geo) or switchback randomisation.
- **Interleaving** is far more sensitive for ranking changes (it measures which ranker users prefer, not the business metric — confirm winners with an A/B); **long-term holdouts** catch slow effects.
- **Offline win, flat online result?** Usual suspects: the offline metric is a weak proxy for the goal (relevance ≠ purchase); the offline set only contains what the *old* model chose to show (exposure bias — use logged propensities / counterfactual evaluation); training-serving skew; a latency regression eating the gain; an underpowered test. Keep a log of offline-vs-online deltas across launches to learn which offline metrics actually predict online wins.
- When you cannot A/B (safety-critical, legal): shadow mode, offline replay, staged ramps with human audit.

#### Phase 10 — Monitoring and iteration (3 min)
- **System**: QPS, p50/p99 latency, error rate, GPU utilisation/memory, queue depth, cost per request.
- **Data drift**: input statistics (resolution, brightness, device/camera mix, language), embedding-distribution shift (e.g., distance of daily mean embedding to a reference; population stability index on scores).
- **Prediction drift**: score distributions and positive rates per class/policy; sudden spikes usually mean a pipeline bug or an attack.
- **Quality with delayed labels**: periodic human audits of random samples; proxy metrics in the meantime.
- **Training-serving skew** (very common in CV): different resize/interpolation or antialiasing (PIL vs OpenCV vs GPU resize), JPEG decoder differences, EXIF orientation not applied, RGB vs BGR, colour space (sRGB vs Display P3, common in HEIC phone photos), normalization constants, centre-crop vs letterbox, client-side re-compression, different video frame sampling. Fix: one shared preprocessing library + golden-image tests that compare train and serve tensors.
- **Feedback loops**: the model decides what users see, which decides future labels (exposure/position bias, moderation only reviews what it flagged). Mitigate: small randomised exploration traffic, log propensities, position as a training feature, random-sample audits.
- **Retraining**: scheduled (weekly/monthly) plus triggered (drift alarm, new policy); every retrain goes through the same offline eval gate and shadow/canary.
- **Failure modes and fallbacks**: dependency down → cached or popularity-based results; model returns garbage → score sanity checks; decide **fail-open vs fail-closed** explicitly per use case.

#### Phase 11 — Responsible AI (2 min, plus inline)
- **Fairness**: evaluate slices by skin tone (Fitzpatrick or Monk scales), gender presentation, age, geography/income (object recognition has been shown to work worse on images from lower-income households), language; set per-slice error targets.
- **Safety/abuse**: misuse of generative features, adversarial users, harmful content exposure for reviewers (wellbeing tooling: blur, greyscale, exposure limits).
- **Privacy**: data minimisation, retention limits, on-device processing, consent for faces/biometrics (special-category data under GDPR; biometric laws such as Illinois BIPA), deletion requests (embeddings derived from user data are user data), differential privacy where aggregates suffice.
- **Transparency and provenance**: model/data cards, labels on AI-generated content, invisible watermarks + C2PA content credentials, appeal paths for enforcement decisions.

### 1.4 Opening and transition scripts

- **Opening**: "Let me restate the goal… I'll spend about five minutes on requirements, then sketch the whole system end to end, then go deep on whichever part you care most about. Does that work?"
- **Assumptions**: "I'll assume 300M uploads per day; if the real number is 10× larger, the cascade becomes even more important — I'll flag where it changes the design."
- **Trade-off**: "Two options: HNSW gives higher recall at low latency but costs roughly 250 B/vector extra memory; IVF-PQ codes are ~30× smaller than float32 vectors but need re-ranking. Given a 200M catalog and multi-region replication, I'd pick IVF-PQ plus re-ranking."
- **Check-in**: "Before I go into serving, is there anything in the data plan you'd like me to expand?"
- **Wrap-up (last minute)**: "To summarise: baseline X, main risks Y and Z, first three things I'd measure after launch are …"

### 1.5 Production vocabulary to use naturally

| Term | Meaning (one line) |
|---|---|
| SLO / SLA | target (e.g., p99 < 300 ms, 99.9 % availability) / contractual version of it |
| p50 / p99 | median / 99th-percentile latency; design for the tail, not the mean |
| QPS, peak-to-average | requests per second; diurnal peaks typically 2–3× average |
| Online / nearline / offline | per-request / streaming within seconds–minutes / scheduled batch |
| Feature store | serves the same features online and offline with point-in-time correctness |
| Model registry | versioned models + metadata (data version, metrics, owner) |
| Shadow mode | new model runs on live traffic, results logged but not shown |
| Canary / staged rollout | small % of traffic first, then ramp, automatic rollback on guardrail breach |
| Backfill | recompute outputs (e.g., embeddings) for historical data after a model change |
| Embedding version | every stored vector tagged with the model that produced it; never mix versions in one index |
| Idempotent pipeline | rerunning a job produces the same result; safe retries |
| Message queue (e.g., Kafka) | durable stream of events between services; decouples producers and consumers |
| Dead-letter queue | where failed messages go for inspection instead of being dropped |
| Warm pool / cold start | pre-loaded replicas vs time to start a new one (GPU: pull image + load weights) |
| Graceful degradation | cheaper/faster path under overload (smaller model, fewer results) |
| Fail-open / fail-closed | on failure, let traffic through / block it |
| Training-serving skew | features/preprocessing differ between training and production |

### 1.6 One-page checklist (memorise this)

```
REQUIREMENTS   [ ] goal + business metric      [ ] scale: DAU, req/day -> avg & peak QPS, corpus size
               [ ] latency p99 / throughput    [ ] freshness   [ ] platform & budget   [ ] privacy/legal
               [ ] FP vs FN cost               [ ] existing system & logs
FRAMING        [ ] inputs -> outputs, task type, cascade?     [ ] label source
               [ ] loss  <->  offline metric  <->  online metric (written side by side)   [ ] non-ML baseline
DATA           [ ] sources  [ ] labeling: human / implicit / weak / self-sup / active learning (cost!)
               [ ] quality: dups, noise, imbalance, domain gap  [ ] splits by entity/time  [ ] leakage, contamination
FEATURES       [ ] pretrained embeddings vs end-to-end  [ ] metadata/context  [ ] preprocessing parity
MODEL          [ ] baseline first  [ ] strong model justified by constraints  [ ] cascade  [ ] 2+ trade-offs
TRAINING       [ ] loss  [ ] imbalance & hard negatives (+ recalibration)  [ ] augmentations  [ ] DDP/FSDP if big
OFFLINE EVAL   [ ] metric per stage  [ ] operating point  [ ] slices  [ ] ablations  [ ] error analysis  [ ] CIs
SERVING        [ ] diagram (online + offline paths)  [ ] latency budget that sums  [ ] QPS -> #GPUs -> $
               [ ] batching  [ ] caching  [ ] quantization/distillation  [ ] GPU vs CPU vs edge
               [ ] rollout: shadow -> canary -> ramp; fallbacks
ONLINE EVAL    [ ] A/B unit, primary metric, guardrails, sample size  [ ] interleaving / holdouts
               [ ] SRM check, novelty, analyse at the randomisation unit  [ ] offline-vs-online gap
MONITORING     [ ] system + data drift + prediction drift  [ ] delayed labels -> audits  [ ] train/serve skew
               [ ] feedback loops & exploration  [ ] retraining cadence + triggers  [ ] failure modes
RESPONSIBLE    [ ] fairness slices  [ ] safety/abuse  [ ] privacy/retention/consent  [ ] provenance
WRAP-UP        [ ] summary, top risks, what you'd do next with 2x time/data
```

---

## 2. Case studies

Each case follows the framework. Numbers are **stated assumptions** — the point is the method. Read the diagram first, then the numbers.

---

### Case a — Visual search / "shop the look"

**Prompt**: "A user takes or uploads a photo (e.g., an outfit or a living room). Return purchasable products that look similar, for each item in the photo."

#### 1. Requirements (assumed)
- Catalog: 50M products × ~4 images = **200M catalog images**; ~1M new images/day; items go out of stock constantly.
- Traffic: 15M visual queries/day → 15·10⁶ / 86,400 ≈ **174 QPS average**, ≈ **520 QPS peak** (3×).
- Latency: p99 ≤ 400 ms server-side (excluding upload), p50 ≈ 150 ms.
- Freshness: new products searchable within ~1 h; out-of-stock items hidden within minutes.
- Questions to ask: exact product or similar style? Which categories (fashion, home)? Do price, brand, availability, personalisation, sponsored items matter? Multi-region?

#### 2. ML framing
- Decompose: (1) **detect/localise** items → (2) **embed** each crop → (3) **retrieve** nearest catalog images (ANN) → (4) **re-rank** candidates with richer features → (5) business rules.
- Embedding objective: metric learning — same product (and visually similar products) close, others far.
- Re-ranker objective: P(click or purchase | query crop, product).
- Metrics chain: InfoNCE loss → Recall@k of the exact product / NDCG@k with graded relevance → conversion and GMV per visual-search session.

#### 3. Data
- **Free positives**: multiple catalog images of the same product (different views), product variants.
- **Engagement logs**: query crop → clicked/added-to-cart/purchased product. Noisy and position-biased, but in-domain (real user photos).
- **Human relevance labels**: sample (query, result) pairs, grade *exact / similar style / irrelevant*; used mainly for evaluation.
- **Public bootstrap data** for the street-to-shop gap (e.g., DeepFashion2, Street2Shop) — check licences.
- **Synthetic queries**: paste catalog items onto backgrounds, random crops, blur, perspective — but avoid colour jitter if colour is part of similarity.
- **Domain gap**: catalog = clean studio shots; queries = occlusion, lighting, mirror selfies, screenshots. This gap is the main modelling problem.
- **Splits by product ID** (not by image), plus a time split for engagement data; dedupe identical listings from different sellers.

#### 4. Features / representations
- Image embedding: d = 256, L2-normalised. Detector also outputs a category (used for filtering).
- Re-ranking features: embedding similarity, attribute matches (colour, pattern, sleeve length), category agreement, price, brand, popularity/CTR priors, stock, user affinity.

#### 5. Model
- **Baseline (week 1)**: off-the-shelf CLIP/DINOv2 embedding of the whole image + exact kNN on a 1M-item subset. Validates the product and the logging.
- **Detector**: fine-tuned detector (DETR- or YOLO-family) for the target categories; gives crops and category.
- **Embedder**: ViT-B-class backbone fine-tuned with contrastive loss on (user crop, product image) and (product view, product view) pairs; optional attribute heads (multi-task).
- **Re-ranker**: GBDT or small MLP over the top ~1,000 candidates (cheap, CPU); optionally a cross-encoder (joint model over both images) on the top ~50 if latency allows.
- Trade-offs: one universal embedding vs per-category embeddings (accuracy vs ops burden); detector + crops vs whole-image embedding (multi-item support vs ~20–30 ms extra); d = 256 vs 768 (memory ×3 vs small recall gain).

#### 6. Training
- InfoNCE with in-batch negatives (batch 4K–32K, gathered across GPUs), learnable temperature; mix in **same-category hard negatives** (otherwise the model learns "dress vs sofa", not "this dress vs that dress").
- Balanced sampling across categories; several positives per product per batch.
- DDP is enough for a ViT-B; the cross-GPU all-gather of embeddings is what makes large-batch contrastive training work.

#### 7. Offline evaluation
- Embedder (with exact search): Recall@1/10/100 of the exact product, mAP@k, NDCG@k on graded labels.
- **ANN quality separately**: ANN Recall@k vs exact brute-force on sampled queries (target ≥ 0.95 at the chosen nprobe).
- Detector: mAP per category; end-to-end "look completeness" (fraction of items in the photo with a relevant result).
- Slices: category, query source (camera vs screenshot vs social image), lighting, occlusion, long-tail categories, products < 7 days old.
- Ablations: detector on/off, embedding dim, PQ code size (recall lost by compression).

#### 8. Serving

```
 client --(resized <=1024px JPEG)--> API gateway
                                        |
   [1] preprocess: decode, EXIF-rotate, resize (CPU)                      ~10 ms
                                        |
   [2] detector (GPU, dynamic batching)                                   ~20 ms
                                        |  N crops (avg ~3) + category
   [3] embedder (GPU, FP16/INT8, batched)                                 ~15 ms
                                        |  N x 256-d vectors
   [4] ANN service: IVF-PQ shards in RAM (+ "fresh" index), category filter ~15 ms
                                        |  top-1000 candidates per crop
   [5] feature fetch: price, stock, CTR priors (feature store / KV cache)  ~10 ms
                                        |
   [6] re-ranker: GBDT / MLP (CPU)                                        ~30 ms
                                        |  top-50
   [7] business rules: in-stock, dedupe variants/sellers, diversity        ~5 ms
                                        v
                                     response                 sum ~105 ms (+ network/queueing)

 Nearline:  catalog change events (queue) --> embed new images (GPU batch) --> insert into "fresh" index
            stock changes --> availability bitmap (filter at query time)
 Offline:   nightly merge of fresh index into main IVF-PQ; on model change: full re-embed + rebuild
            --> offline eval gate --> shadow --> atomic swap of (embedder version, index version)
```

**Compute**: ~520 QPS × (~1 detector pass + 3 crops) is roughly 2K GPU inferences/s at peak. A ViT-B/16 at 224² is ≈ 17.6 GMACs ≈ 35 GFLOPs; 3 crops ≈ 105 GFLOPs + detector (say ~100 GFLOPs) ≈ 205 GFLOPs/query → 520 × 205 GFLOPs ≈ 105 TFLOP/s, i.e. about one A100-class GPU at ~35 % utilisation. Conclusion: **latency-bound, not throughput-bound** — deploy 2–3 GPUs per region for redundancy and keep batches small (max wait ~2–5 ms).

**Index memory** (200M vectors, d = 256):

| Representation | Bytes/vector | Total |
|---|---|---|
| float32 | 1,024 | 200M × 1,024 B ≈ **205 GB** |
| float16 | 512 | ≈ 102 GB |
| PQ, m = 32 sub-quantisers × 8 bits (+ 8-byte id) | 32 + 8 | ≈ 6.4 + 1.6 = **8 GB** (≈ 32× smaller codes) |
| HNSW (M = 32) on float32 | 1,024 + ~256 links | ≈ 256 GB |

- IVF parameters: a common rule of thumb is nlist ≈ 4√N–16√N (the Faiss wiki states it for < 1M vectors and suggests 2¹⁸–2²⁰ lists with an HNSW coarse quantiser at 10M–1B). √(200M) ≈ 14K → 4√N–16√N ≈ 57K–226K; take nlist = 65,536 (2¹⁶). With nprobe = 64 each query scans ≈ 200M × 64/65,536 ≈ 195K PQ codes (≈ 6 MB) → ~1–5 ms on a few CPU cores. More lists → fewer codes scanned per probe but slower k-means training and coarse assignment. Tune nprobe on the ANN-recall/latency curve.
- 8 GB fits in RAM on one host → **replicate** for QPS and regions rather than shard. If you want exact re-scoring with float16 vectors (102 GB), note the I/O: re-scoring the top-1,000 for ~3 crops at 520 QPS is ≈ 1.5M random reads/s — at or beyond what a single NVMe drive delivers — so re-score fewer candidates (top ~100–200), use a finer code (e.g., more PQ bytes) or hold the vectors in RAM across a few hosts.

**Freshness**:
- New images: 1M/day ≈ 12/s average — embedding them is trivial. Insert into a small **fresh index** (flat or HNSW) that is queried in parallel with the main index; merge results by score; fold into the main index nightly (IVF-PQ assigns new vectors to existing centroids; retrain centroids only on rebuild).
- Deletes/out-of-stock: filter with an availability bitmap at query time (over-fetch candidates); physically remove at rebuild.
- **Model upgrade = full backfill**: 200M × 35 GFLOPs = 7·10¹⁸ FLOPs; at ~94 TFLOP/s effective (A100 at 30 %) ≈ 75K GPU-s ≈ 21 GPU-hours → ~1.3 h on 16 GPUs (needs ~40K JPEG decodes/s in total — budget CPU cores or GPU decoding). Query encoder and index must switch **atomically**.

#### 9. Online evaluation
- User-level A/B. Primary: purchases (or GMV) per visual-search session — a ratio metric under user-level randomisation, so use the delta method (and cap GMV outliers). Secondary: CTR, add-to-cart, null/abandon rate. Guardrails: p99 latency, error rate, overall app revenue (cannibalisation of text search), seller diversity.
- Re-ranker iterations: interleaving for speed.

#### 10. Monitoring and iteration
- Null-result and abandon rate; detection category mix (seasonal shifts); daily query-embedding drift; **canary queries** with known answers to catch broken indexes; fresh-index lag; click-on-out-of-stock rate.
- Feedback loop: the re-ranker learns from clicks on its own rankings → position bias. Use position as a training feature (set to a constant at serving), and a small randomised slot for exploration.

#### 11. Responsible AI
- Uploaded photos may contain faces/children: minimise retention, do not store crops beyond need, blur faces in logs.
- Fairness: results quality across skin tones and body types (fashion); slice offline metrics.
- Counterfeits/IP: down-rank flagged sellers; do not infer sensitive attributes from the photo.

#### Trade-offs to discuss
- **HNSW vs IVF-PQ vs GPU ANN**: HNSW = best recall/latency, memory-heavy, supports inserts; IVF-PQ = compact, lower recall, needs re-ranking; GPU ANN (e.g., Faiss-GPU, CAGRA) = very high QPS, GPU memory-bound; disk-based (DiskANN-style) for billions on SSD.
- **Pre-filter vs post-filter** for constraints (category, region, price): partitioned indexes per category/region (fast, more indexes) vs over-fetch-then-filter (simple, breaks for very selective filters).
- **Embedding dim and precision** vs memory and recall.
- **Exact-match vs style similarity**: one embedding with a calibrated "exact match" threshold vs two heads.

#### Follow-up questions interviewers ask
- **Q: How do you apply a filter like "in stock, under $50" with ANN?** A: Partition the index by coarse filters (category/region); for fine filters over-fetch k′ ≈ k / selectivity and post-filter; for very selective filters use filtered-search support in the index or fall back to exact search over the filtered subset.
- **Q: Recall dropped after launch. Model or index?** A: Measure ANN recall vs exact search on logged queries; if ANN recall is fine, it is the model or a distribution shift — check slices and preprocessing skew.
- **Q: How do you roll out a new embedding model?** A: Backfill embeddings, build a new index, offline eval, shadow, then swap (model, index) together. To avoid a backfill, train the new model to be backward-compatible with the old embedding space (BCT-style).
- **Q: Cold-start products with no clicks?** A: Content embeddings work without engagement; the re-ranker uses content features and a small exploration boost for new items.
- **Q: Why not a cross-encoder over everything?** A: Cost is O(catalog) per query. Use it only on the top ~50 after cheaper stages.
- **Q: The same product listed by 30 sellers floods the results.** A: Cluster near-duplicate listings offline (embedding + metadata), show one representative per cluster, and add diversity (e.g., MMR) in the final stage.

---

### Case b — Content moderation for images and video at scale

**Prompt**: "Detect and act on policy-violating images and videos uploaded to a large social platform."

#### 1. Requirements (assumed)
- Volume: **300M images/day** → ≈ 3.5K/s average, ≈ 7K/s peak (2×); **20M videos/day**, average 60 s.
- Video frames: sample 1 frame/s (+ scene-change keyframes) → 20M × 60 = 1.2B frames/day ≈ **14K frames/s** average. Total first-stage inferences ≈ 17.4K/s average, **≈ 35K/s peak**.
- Policies (separate, each with its own threshold and action): nudity/sexual content, child safety, violence/gore, hate symbols, terrorism, self-harm, spam/scams, regulated goods. Actions: remove, age-gate, reduce distribution, warning interstitial, send to review.
- Latency: images checked **before publication** within ~200 ms for the fast path; videos asynchronously, but before distribution ramps up (a video must not go viral before a full scan).
- Questions to ask: which surfaces (posts, profile photos, ads, live)? Regions and laws (e.g., EU DSA)? Review capacity? Appeals? Fail-open or fail-closed when the system is down?

#### 2. ML framing
- **Multi-label classification per policy**, calibrated probabilities. Not a single "bad" class: policies change independently and need different thresholds and actions.
- **Decision layer separate from the model**: thresholds per (policy × surface × region).
- Objective per policy: minimise expected cost = C_FN × severity × expected views + C_FP × user harm (wrongful takedown) + review cost, subject to review capacity.
- Metrics chain: weighted BCE/focal loss → recall at the auto-action precision target, PR-AUC per policy → **prevalence** (share of views that are violating), proactive-detection rate, appeal overturn rate.

#### 3. Data
- Human labels by trained policy reviewers with written guidelines; gold questions; multi-review + adjudication for hard policies; track inter-rater agreement (κ). Policy rewrites invalidate old labels — version labels by policy version.
- Positives are rare (often well under 1 %): use **active learning** (production high scores, uncertainty band, embedding neighbours of confirmed violations) to find them.
- Hard negatives matter: art, medical, breastfeeding, news/war reporting, video games vs real violence.
- **Legal constraints**: illegal material such as CSAM generally cannot be stored or used as ordinary training data; it is handled via hash-matching programmes and specialised, legally sanctioned processes, plus mandatory reporting (e.g., to NCMEC in the US).
- Splits by uploader/account and by time (coordinated campaigns produce near-duplicates that would leak across random splits).

#### 4–5. Features and model (cascade)
- **Stage 0 — hash matching**: perceptual hashes (e.g., PDQ, PhotoDNA-style) against banks of known violating content. Near-free; catches re-uploads. A 256-bit PDQ hash is 32 bytes → 100M hashes ≈ 3.2 GB in memory. Matching is a **Hamming-distance** search, not an exact lookup (Meta's PDQ reference suggests ≤ 31 differing bits of 256 as a starting threshold) → multi-index hashing or a binary ANN index.
- **Stage 1 — shared encoder + per-policy heads** on 100 % of traffic. Baseline: frozen CLIP-style encoder + linear probes per policy (fast to build, surprisingly strong). Then fine-tune a ViT-S/efficient-CNN backbone multi-task. Video: per-frame scores + temporal max/attention pooling.
- **Stage 2 — heavy multimodal model** on the uncertain ~5 %: image + caption + OCR text + audio transcript + account/context signals; a fine-tuned VLM can also be **prompted with policy text**, which lets you react to a new policy before you have many labels (then distil into a Stage-1 head). Multimodal fusion matters: benign image + hateful caption (the "hateful memes" problem) fools unimodal models.
- **Stage 3 — human review** for the remaining uncertain, high-severity or high-reach items, plus appeals.

```
upload --> [Stage 0: perceptual-hash match vs known-bad banks] --hit--> auto-action (+ legal report)
             | miss
             v
           [Stage 1: shared encoder + per-policy heads]   100% of traffic, ~20-40 ms
             | calibrated score per policy
             |-- below t_low(policy) -----------------------------------> publish
             |-- above t_high(policy) (high-precision band) -------------> auto-action
             v uncertain band (~5%)
           [Stage 2: multimodal model: image + caption + OCR + ASR + account signals]
             |-- confident ---> action / publish
             v still uncertain, high severity or high predicted reach
           [Human review queue, prioritised by severity x expected views]
             | decisions = labels
             v
           label store --> retraining sets, hash + embedding banks, threshold recalibration
 appeals --> second-level review --> overturns become labels (FP signal)
 videos: thumbnail + first frames on the fast path; full scan async; re-scan when views spike
```

#### Numbers: why a cascade
- **Stage 1**: a ViT-S/16 is ≈ 4.6 GMACs ≈ 9 GFLOPs at 224². 35K/s × 9.2 GFLOPs ≈ 3.2·10¹⁴ FLOP/s. An L4-class GPU (~121 TFLOP/s dense BF16, approximate) at ~30 % utilisation gives ~36 TFLOP/s → ~3.9K images/s → **~9 GPUs at peak**, ~15–20 with redundancy across regions (INT8 roughly halves it). Decoding is a real cost: at a few hundred JPEG decodes/s per core, the ~7K images/s peak alone needs a few dozen cores. Video costs more than its ~28K sampled frames/s suggest: P/B-frames depend on earlier frames, so extracting 1 frame/s from H.264/HEVC still decodes many intervening frames unless you sample only keyframes (I-frames) → hardware decoders (NVDEC).
- **Stage 2**: suppose a 7B VLM scoring ~600 tokens per item (≈ 576 image tokens + text), prefill only: 2 × 7·10⁹ × 600 ≈ 8.4·10¹² FLOPs/item. At 5 % of peak (≈ 1.75K items/s): ≈ 1.5·10¹⁶ FLOP/s → at ~400 TFLOP/s effective per H100 (≈ 40 % of ~989 TFLOP/s) ≈ **37 H100s**. Running Stage 2 on 100 % of traffic would need ≈ 20× more (≈ 740 H100s). That ratio is the cascade argument.
- **Human review**: if 0.1 % of images reach humans → 300K/day; at ~20 s per item ≈ 1,670 reviewer-hours/day ≈ **210 eight-hour shifts per day** (images only). Escalation precision directly sets headcount.

#### 6. Training
- Weighted BCE or focal loss per head; hard-negative mining; heavy augmentation that mimics evasion (crops, borders, mirroring, overlays, re-encoding, screenshots of screenshots).
- **Recalibrate** after reweighting (temperature scaling or isotonic regression per policy) on a recent, representative sample — thresholds assume calibrated scores.
- Frequent retraining (weekly) plus rapid-response updates of hash/embedding banks (hours).

#### 7. Offline evaluation
- Per policy: PR curve, **recall at the auto-action precision target** (e.g., ≥ 95–99 % precision for automated removal, depending on severity), recall at review-capacity budget, ECE.
- Measuring recall when positives are rare: **stratified sampling by score bucket** with inverse-probability weighting, labelled by experts.
- Slices: language/region, content type (art, medical, news), image vs video, skin tone (nudity classifiers that lean on skin-colour cues can have skin-tone-dependent error rates), adversarial test set.

#### 8. Serving notes
- Fast path budget (images, pre-publication): hash lookup ~5 ms, decode/resize ~10 ms, Stage 1 ~30 ms, decision ~5 ms → well under 200 ms. Stage 2 (~0.3–1 s) runs async with **publish-with-hold** (visible to the uploader only) or limited distribution until cleared.
- Videos: prioritise by predicted reach (follower count, view velocity); re-scan when an item starts spreading.
- **Fail-open vs fail-closed**: e.g., ads fail closed (hold until scanned); ordinary posts might fail open with a mandatory delayed scan. Decide per surface and policy.

#### 9. Online evaluation
- For severe-harm policies you cannot ethically A/B by withholding enforcement from a control group. New classifiers run in **shadow mode**; compare against production decisions on a labelled sample; ramp thresholds gradually. Lower-severity actions (e.g., demotion thresholds) can be A/B-tested with prevalence, user-report and appeal guardrails.
- Outcome metrics: **prevalence** of violating views (estimated by labelling a random sample of *viewed* content), proactive rate (actioned before any user report), time-to-action, review SLA, appeal overturn rate (FP proxy), user reports per million views.

#### 10. Monitoring and adversarial drift
- Daily score distributions and action rates per policy/region; spikes = attack, new trend, or a broken preprocessing step.
- Adversaries adapt: perturbations, crops/borders/mirroring to break hashes, text overlays, coded symbols, new slang in captions, coordinated re-uploads. Responses: augment and retrain, OCR/ASR branches, **embedding-similarity banks** of newly confirmed violations (catch variants within hours), account-level and behavioural signals, rate limits, red-teaming.
- Feedback loop: reviewers mostly see what the model flagged → labels concentrate near the decision boundary; keep a random-sample audit stream for unbiased estimates.

#### 11. Responsible AI
- Over-enforcement silences legitimate speech (e.g., reclaimed slurs, LGBTQ+ content, news); measure FP rates by language and community, provide appeals and transparency reporting.
- Reviewer wellbeing: blur/greyscale by default, exposure limits, support. Privacy: retention limits for removed content (subject to legal holds).

#### Trade-offs to discuss
- Precision/recall **per policy** (severity-tiered) vs one global threshold.
- Shared multi-head backbone (cheap; one deploy) vs per-policy models (independent iteration; no negative transfer).
- Synchronous depth (safer) vs latency and cost; automation vs human review cost and wellbeing.

#### Follow-up questions interviewers ask
- **Q: How do you choose thresholds?** A: Per policy, minimise expected cost under the review-capacity constraint on recent calibrated data; separate auto-action (high precision) and review (high recall) thresholds; revisit after every model or policy change.
- **Q: How do you know your recall?** A: Expert labelling of a random sample of uploads/views, stratified by score, with inverse-probability weights; user reports and appeals as secondary signals.
- **Q: A new policy launches tomorrow.** A: Prompt the Stage-2 VLM with the policy text + seed examples, use embedding search around seeds to find more, route to review to get labels, train/distil a Stage-1 head within weeks.
- **Q: Reviewers disagree 30 % of the time.** A: That is a guideline problem first — clarify, add examples, adjudicate; train with soft labels; track κ per policy.
- **Q: A coordinated campaign re-uploads slightly edited copies.** A: Perceptual hash + embedding similarity to the confirmed item, cluster-level actions, account signals, upload rate limits.
- **Q: How do you handle live video?** A: Sample frames every few seconds plus audio ASR, escalate on reach, allow the system to interrupt the stream; accept higher compute per viewer-minute for high-reach streams.

---

### Case c — Text-to-image generation service

**Prompt**: "Design the backend for a consumer text-to-image product."

#### 1. Requirements (assumed)
- 2.5M requests/day × 4 images = **10M images/day** at 1024² → 10·10⁶ / 86,400 ≈ **116 images/s average**, ≈ 350/s peak (3×).
- Latency: p50 ≈ 3–5 s to show results (progressive previews are welcome); p99 < 15 s.
- Two tiers: **fast** (draft, ~1 s) and **quality** (~3–5 s).
- Safety: strict policy (sexual content involving minors is a hard block; restrictions on real people/public figures, violence, etc.), watermarking/provenance.
- Questions to ask: image editing/inpainting too? Paid vs free tiers (priority)? Commercial use of outputs (training-data licensing)? Regions? Cost target per image?

#### 2. ML framing
- Conditional generation of images given text; quality = realism + prompt alignment + aesthetics + diversity + safety. No single supervised metric.
- Training objective (if training/fine-tuning): denoising/ε- or v-prediction or flow-matching loss; preference alignment (e.g., Diffusion-DPO-style or reward fine-tuning) on human pairwise preferences.
- Product metrics: save/download/share rate, regenerate rate (dissatisfaction proxy), retention, safety-incident rate, **cost per delivered image**.

#### 3. Data (if you train or fine-tune)
- Licensed/permissioned image-text pairs; aesthetic and safety filtering; **dedup** (diffusion models can memorise and regurgitate heavily duplicated training images).
- **Recaptioning** with a VLM produces dense, accurate captions and noticeably improves prompt following (the DALL-E 3 paper reported this).
- Preference data: pairwise human choices between generations for the same prompt; prompts sampled from real traffic (privacy-filtered) plus curated hard prompts (text rendering, counting, hands, spatial relations).

#### 4–5. Model
- **Latent diffusion**: a VAE maps 1024×1024×3 pixels to a 128×128 latent (downsampling factor 8; 4 channels in SD-1.x/SDXL, 16 in SD3/FLUX-class models). With 4 channels that is 3,145,728 / 65,536 = **48× fewer elements** than pixels, which is why denoising runs in latent space.
- **Baseline**: an open SDXL-class model (UNet ≈ 2.6B params, two text encoders, ~3.5B total) or a DiT/MMDiT-class model, ~30 sampler steps (DPM-Solver/Euler) with classifier-free guidance (**CFG doubles the denoiser work**: conditional + unconditional pass).
- **Strong (cost-optimised)**: few-step distilled models — consistency distillation (LCM), adversarial diffusion distillation (SDXL-Turbo style), progressive distillation, rectified-flow/distribution-matching approaches — at 1–8 steps; **guidance distillation** removes the CFG 2×.
- Trade-off: fewer steps → lower cost and latency, usually some loss of diversity and fine detail; hence the fast/quality tiers.

#### 8. Serving (the core of this case)

```
client --> API gateway (auth, quotas, rate limits per account)
             |
   [prompt safety] text classifier + blocklists + public-figure/minor checks     ~30 ms
             |  (optional) prompt rewrite/expansion with a small LLM            ~0.3 s
             v
   [scheduler / queue] priority (paid > free), shape bucketing (resolution, steps), admission control
             |
   [GPU worker pool]  text encoders --> denoiser x steps (CFG) --> VAE decode    ~2 s (quality tier)
             |   previews: decode intermediate x0 predictions with a tiny VAE decoder every few steps
             v
   [output safety] image classifiers (nudity, violence, minors, likeness), OCR on rendered text  ~50 ms
             |
   [post] invisible watermark + C2PA manifest, encode WebP, upload to object store -> CDN URL    ~150 ms
             v
        response (URLs)
 Side systems: result cache, abuse monitoring, sampled human safety audits, metrics/cost dashboards,
               autoscaler with warm pool (weights pre-loaded), batch queue for non-interactive jobs
```

**Latency budget (quality tier, one image per GPU, p50)**

| Step | Time (approx.) |
|---|---|
| Gateway, auth, quotas | 20 ms |
| Prompt safety | 30 ms |
| Queue wait (target) | ≤ 500 ms |
| Text encoders | 20 ms |
| Denoising: 30 steps × CFG | ~1.8 s |
| VAE decode at 1024² | ~0.15 s |
| Output safety | 50 ms |
| Watermark + encode + upload | 150 ms |
| **Total** | **≈ 2.7 s** (+0.3 s if prompt rewriting) |

A request for 4 images is fanned out to 4 GPUs (same GPU-seconds; up to ~4× lower latency than one GPU running batch 4, because the workload is compute-bound).

**Cost math** — *assumption to verify by measurement*: an optimised SDXL-class model at 1024², 30 steps + CFG, costs **≈ 2 GPU-seconds per image** on an H100 (≈ 1.8 s denoising + ≈ 0.2 s fixed: text encoders, VAE decode, safety). Sanity check: the PyTorch/Hugging Face "Diffusion, Fast" benchmark ran SDXL at 1024², 30 steps with CFG in ≈ 2.4–2.5 s end to end on an A100-80GB after BF16 + SDPA + torch.compile (+ dynamic INT8); an H100 should be faster still, so ~2 s is a reasonable (if anything conservative) assumption.
- GPU-seconds/day = 10M × 2 s = 2·10⁷ → **≈ 5,560 GPU-hours/day** → ≈ 231 GPUs busy on average.
- Peak: 3 × 231 ≈ 694 busy GPUs; at a 70 % utilisation target ≈ **~1,000 GPUs** provisioned.
- At an assumed $3/GPU-hour: perfect autoscaling → 5,560 × $3 ≈ $16.7K/day ≈ **$0.0017/image**; provisioned for peak 24/7 → 1,000 × 24 × $3 = $72K/day ≈ **$0.0072/image**. Utilisation dominates cost: fill troughs with batch jobs, use autoscaling with warm pools, and use cheaper/spot capacity for non-interactive work.
- **Amdahl for step reduction**: distil to 4 steps (7.5× fewer steps) and remove CFG (2×) → denoising 1.8 s / 15 = 0.12 s; fixed costs remain 0.2 s → **0.32 s/image ≈ 6× cheaper**, not 15×. Next bottleneck: VAE decode and text encoders → tiny/fast decoders, cache text embeddings, lower-resolution draft + upscaler.

**Batching**: at 1024² a single image already presents a large, compute-bound workload (many latent tokens), so batching improves throughput only modestly while increasing latency roughly linearly — unlike LLM decoding, which is memory-bandwidth-bound and benefits hugely from batching. Batching matters more at low resolution or for small/distilled models. Requests at *different* timesteps can still share a batch (the timestep is a per-sample input), which enables step-level continuous batching; bucket by resolution and step count.

**Other optimisations**: BF16/FP16, FP8 on Hopper-class GPUs (check quality), fused attention kernels, compilation (torch.compile/TensorRT), CUDA graphs, feature caching across adjacent steps (DeepCache-style), token merging, multi-LoRA serving for styles.

**Caching**: result cache keyed by (model version, normalised prompt, negative prompt, seed, size, steps, guidance) — hit rate is low for free-form prompts but high for featured/example prompts and retries; invalidate on safety-policy changes. Do **not** semantically cache across users (users expect novel samples; privacy). Cache text embeddings for popular prompts. Keep weights (~7 GB in FP16 for an SDXL-class pipeline) on local NVMe; cold start (pull container + load weights) takes tens of seconds to minutes → warm pool.

**Storage**: 10M/day × ~300 KB (WebP/JPEG at 1024²) ≈ **3 TB/day** ≈ 1.1 PB/year; with 30-day retention for unsaved images ≈ 90 TB rolling.

#### Safety layers (defence in depth)
1. Input: prompt classifier (multi-label), blocklists with obfuscation handling, jailbreak detection, public-figure/minor detection, per-account rate limits and abuse scoring.
2. Model: training-data filtering, safety fine-tuning or concept erasure, safe default negative prompts.
3. Output: image classifiers (sexual content, minors, violence, gore, likeness of real people), OCR on generated text; block or blur and log.
4. Provenance: invisible watermark + C2PA content credentials.
5. Operations: red-teaming before each model launch, sampled human audits, user reporting.
- Trade-off: over-blocking (e.g., "a photographer shooting a portrait" blocked for "shooting") vs under-blocking — measure **both** the violation rate and the false-refusal rate, per category.

#### 7. Evaluation
- **FID** (or the CLIP-feature-based CMMD) vs a reference set: needs ~10K–50K samples (zero-shot COCO-30K is common); FID is biased by sample size, sensitive to resizing/compression details (use a standard implementation such as clean-fid), relies on ImageNet-trained Inception features, and ignores prompt alignment.
- **CLIP score** (cosine similarity of CLIP image/text embeddings, often reported ×100): prompt alignment; saturates and can be gamed. Report **FID-vs-CLIP-score curves across guidance scales** rather than single points.
- Compositional checks: GenEval / T2I-CompBench-style (object presence, counting, colour binding, spatial relations via detectors).
- Learned preference models (PickScore, ImageReward, HPS v2) for fast iteration — but they are models with biases; confirm with humans.
- **Human side-by-side**: fixed prompt sets (e.g., PartiPrompts ~1.6K prompts, DrawBench 200 prompts) plus prompts sampled from traffic; several raters per pair; axes = overall preference, alignment, aesthetics, artifacts (hands, text); aggregate with Bradley–Terry/Elo and confidence intervals.
- Safety eval: red-team prompt set → violation rate and false-refusal rate.
- Diversity: variation across seeds for the same prompt (e.g., pairwise LPIPS or embedding spread), checked when distilling.
- **Release gate** (when a new model ships every month, evaluation is itself a system): versioned prompt suites per capability (alignment, counting, text rendering, people and demographic spread, safety red-team); automated metrics with bootstrap CIs vs the production model; a human side-by-side against production with the win-rate threshold and sample size fixed in advance; hard blockers = regressions in violation rate, false-refusal rate or memorisation checks; then shadow → staged rollout. Track per-suite results over releases so a regression in one capability is not hidden by an average.

#### 9–10. Online evaluation and monitoring
- A/B (user-level): save/download/share rate, regenerate rate, session length, retention; guardrails: latency p95, safety-filter trigger and leak rates, cost per delivered image, complaint rate.
- Monitor queue depth, p99 latency, GPU utilisation, OOMs, block rate per category (spikes = abuse campaign or classifier regression), prompt-distribution drift (new trends → new failure modes), leaked-violation rate from sampled human audits.

#### Trade-offs to discuss
- Steps/quality vs cost/latency (tiers); one large model vs model per style; batch size vs latency; strict vs permissive safety thresholds; caching vs freshness/novelty; own GPUs (cheaper at steady load) vs on-demand cloud (elastic).

#### Follow-up questions interviewers ask
- **Q: Why is diffusion inference expensive and how do you cut it?** A: Many sequential denoiser passes, doubled by CFG. Fewer steps (better samplers, distillation), guidance distillation, quantisation, compilation, smaller/latent-efficient models, lower-res draft + upscaler, feature caching.
- **Q: How would you evaluate a sampler that cuts 30 → 8 steps?** A: FID/CLIP at matched compute, human side-by-side on a fixed prompt set, diversity across seeds, and failure buckets (text, hands, fine textures).
- **Q: Traffic spikes 5× because of a viral trend.** A: Admission control + priority queues, degrade gracefully (fast tier by default, fewer images per request, lower resolution), autoscale from a warm pool, shed free-tier load first.
- **Q: How do you reduce memorisation/copyright risk?** A: Dedupe training data, filter near-duplicate outputs against a training-set embedding index, prompt policies (e.g., living-artist names) as a product/legal decision, honour opt-outs.
- **Q: Why can FID disagree with human raters?** A: Inception features are ImageNet-biased, insensitive to alignment and fine details; small-N bias; preprocessing sensitivity.
- **Q: How would you personalise styles cheaply?** A: LoRA adapters per style/user, hot-swapped on shared base weights; cap adapters per GPU; evaluate identity leakage and safety per adapter.

---

### Case d — Image-text retrieval for a photo library (multimodal search)

**Prompt**: "Let users search their personal photo library with natural language ('dog in the snow', 'receipts from Tokyo') and find similar photos."

#### 1. Requirements (assumed)
- 500M users, **mean** ~5K photos (the median is lower; heavy users 100K+) → 500M × 5K ≈ **2.5T photos** in total. (Multiplying users by the *median* would understate a heavy-tailed total.)
- Privacy: photos are private (possibly end-to-end encrypted) → strong preference for **on-device** indexing and search.
- Latency: results while typing, < 100–200 ms; works offline.
- Freshness: new photos searchable within minutes (on device) when resources allow.
- Questions to ask: E2E encryption? Multilingual? People search (faces)? Server-side opt-in? Older/low-end devices?

#### 2. ML framing
- **Dual encoder** (CLIP/SigLIP-style): f_img(photo) and f_txt(query) in a shared d-dim space, score = cosine similarity. Gallery embeddings are precomputed; queries are encoded at search time.
- Retrieval is **per user** (only the user's photos) → no global ANN index needed.
- Hybrid search: embeddings + structured signals (time, place, people clusters, OCR text, albums) fused by a ranker.
- Metrics chain: contrastive loss → Recall@k/NDCG@k on in-domain query sets → search success (tap on a result), reformulation rate.

#### Numbers
- Per-user storage: 5K photos × 512-d × 2 B (FP16) ≈ 5 MB; a 100K-photo library ≈ 102 MB in FP16, 51 MB in INT8, **26 MB at 256-d INT8**.
- Per-query brute force: 100K × 256 ≈ 26M multiply-adds → a few ms on a phone CPU with SIMD. **Exact search is fine; ANN is unnecessary.**
- On-device indexing: efficient image encoders run in ~2–10 ms per image on a recent phone (Apple reports ≈ 1.5–10 ms for MobileCLIP-S0…B on an iPhone 12 Pro Max); decoding and resizing a 12 MP HEIC/JPEG is often the larger cost. Budget ~20 ms per photo end to end → a 20K-photo backlog ≈ 20K × 20 ms ≈ 7 min — schedule it in the background **while charging and idle**; watch thermals.
- Model sizes: image encoder ~10–90M params, text encoder ~40–65M (the MobileCLIP-S0…B range) → ≈ 55–150 MB in INT8 for the pair; download on demand to respect app-size limits.
- **Why re-indexing is expensive (server-side counterfactual)**: 2.5T photos × ~35 GFLOPs (ViT-B/16) ≈ 8.75·10²² FLOPs — about the training compute of a 7B LLM on 2T tokens (6 × 7·10⁹ × 2·10¹² = 8.4·10²²). Storing them: 2.5T × 256 B (256-d INT8) ≈ 640 TB. On device the cost is battery and time instead. Hence **backward-compatible embeddings** (new model's embeddings comparable with old ones) or lazy, versioned re-indexing.

#### On-device vs server

| | On-device | Server |
|---|---|---|
| Privacy | photos never leave the device; compatible with E2E encryption | needs upload and consent; higher legal exposure |
| Model size/quality | small, quantised (distilled) | large models, easy to upgrade |
| Cost | user's battery and compute | GPU fleet for 2.5T-photo backfills |
| Latency/offline | ~ms, works offline | network RTT, needs connectivity |
| Iteration | slow (app/model rollouts, device fragmentation) | fast |
| Common answer | **hybrid**: on-device by default; server-side re-ranking or larger model only with explicit opt-in | |

#### 3. Data
- Pretraining: web image-alt-text pairs (filtered) → zero-shot CLIP-style baseline.
- Domain gap: personal photos contain screenshots, documents, receipts, pets, kids, food, bursts of near-identical shots; web alt-text rarely describes them the way users search.
- Fine-tuning data: licensed or consented/donated photo collections with human- or VLM-written captions; synthetic queries generated from captions ("what would a user type to find this?"). **Do not train on user photos without explicit consent.**
- Evaluation data: human-written queries over consented libraries, graded relevance for each (query, photo); splits by library/user (bursts leak otherwise).

#### 5. Model
- **Baseline**: zero-shot CLIP/SigLIP — strong on objects and scenes; weak on personal concepts ("my car"), text-heavy images, counting, negation, compositional queries ("red cup on a blue table"), fine-grained identities.
- **Fine-tuned**: in-domain contrastive fine-tuning; to keep zero-shot breadth use a low learning rate, mix in general data, lock the image tower (LiT-style) or average weights with the original model (WiSE-FT-style).
- **On-device student**: distil a large server teacher into a small image/text encoder pair (MobileCLIP is a published example of efficient CLIP models for phones); quantise to INT8 and check the quantisation gap per slice.
- **Specialised signals**: OCR → keyword index (receipts, documents); face detection + embedding → person clusters the user names (opt-in); EXIF time and GPS → reverse-geocoded places; burst/near-duplicate grouping.
- **Query parsing**: "mom at the beach last summer" → person = *Mom* cluster, time = the most recent Jun–Aug, semantic = "at the beach". Structured filters first, then semantic ranking.

```
 New photo --> [background indexer: when charging + idle]
                 |-- image encoder (NPU, INT8) --> 256-d embedding (tagged with model version)
                 |-- OCR --> tokens --> local inverted index
                 |-- face detect + embed --> person clusters (opt-in, user-named)
                 '-- EXIF time, GPS --> on-device reverse geocoding
                         v
                 [local store: embeddings + metadata, per library]

 Query "mom at the beach last summer"
   --> [query parser: rules / small model]  person=Mom, time=<last Jun-Aug>, semantic="at the beach"
   --> metadata filters --> candidate set
   --> text encoder (~2-15 ms) --> exact cosine over candidates (~ms)
   --> fuse with OCR / keyword hits (e.g., reciprocal rank fusion or small learned ranker)
   --> group bursts, apply score calibration ("no good results" when everything is weak) --> results
```

#### 6. Training notes
- Contrastive loss needs many negatives: all-gather embeddings across GPUs; the logit matrix is B × B (B = 32K → ~10⁹ logits), so compute it in shards (local rows × global columns). A **sigmoid loss** (SigLIP) avoids the global softmax normalisation and works well at smaller batches.
- Temperature: learnable; CLIP clamps the logit scale at 100 to avoid instability.
- Hard negatives: near-duplicate shots, same scene different subject, documents of different types.

#### 7. Offline evaluation
- Sanity benchmarks: text→image and image→text Recall@1/5/10 on MS-COCO 5K and Flickr30K 1K test sets (check for pretraining contamination).
- In-domain: queries usually have **many** relevant photos, so Recall@k of "a single ground truth" is incomplete — use Precision@k, NDCG@k, mAP and Success@k (≥ 1 relevant in top k).
- Slices: query type (object, scene, activity, text-in-image, people, time/place), language, **library size** (more distractors → lower precision), device tier (quantised student vs teacher).

#### 9–10. Online evaluation and monitoring
- On-device A/B with privacy-preserving aggregate metrics (no raw queries leave the device): search success (tap within session), reformulation rate, zero-result rate, time to first tap. Guardrails: battery use, indexing-completion rate, app size, crash rate, thermal events.
- Monitor index coverage (% of library indexed), model-version mix across devices, latency by device tier.

#### 11. Responsible AI
- Faces are biometric data → explicit opt-in, regional restrictions, on-device only.
- Photo-labelling systems have historically produced offensive mislabels of people; block sensitive label/query combinations on images of people and evaluate across skin tones.
- Deletion: deleting a photo must delete its embedding, OCR text and cluster membership.

#### Follow-up questions interviewers ask
- **Q: Why a dual encoder instead of a cross-encoder?** A: Gallery embeddings are computed once; a cross-encoder needs a forward pass per (query, photo) pair. Use a cross-encoder/VLM only to re-rank the top ~50 if resources allow.
- **Q: "Photos of my receipts" fails.** A: CLIP-style models are weak on text-heavy images → OCR + document-type classifier, fused with the semantic score.
- **Q: Counting and negation ("two dogs", "no people")?** A: Known weaknesses of contrastive models; handle via query parsing + detectors (counts) or VLM re-ranking of the top results.
- **Q: How do you ship a new embedding model?** A: Version every embedding; either backward-compatible training (new query encoder compatible with old gallery) or lazy background re-indexing while charging, searching both versions during the transition.
- **Q: Multilingual queries?** A: Multilingual text encoder distilled to match the English text encoder's embeddings (teacher–student on parallel text); evaluate per language.
- **Q: When should the UI say "no results"?** A: Cosine scores are not calibrated across queries; calibrate per query (e.g., score relative to the user's background distribution) or train a small calibrator on labelled queries.

---

### Case e — Autonomous driving / robotics perception data engine (brief)

**Prompt**: "Design the data loop that keeps improving the perception stack of a vehicle fleet."

**Requirements**: improve detection/tracking (vehicles, pedestrians, cyclists, debris, cones) on rare/long-tail cases; on-vehicle compute and latency limits (e.g., perception at 10–30 Hz → ~33–100 ms budget); safety validation before release.

**Numbers (why you cannot keep everything)**: 1,000 cars × 8 h/day × 8 cameras × ~5 Mbps compressed ≈ 40 Mbps per car × 28,800 s ≈ 144 GB per car-day → **≈ 144 TB/day** of camera video alone (lidar adds much more). Upload only what triggers select (say ~1 % ≈ 1.4 TB/day). Labelling dense 3D scenes by hand costs on the order of minutes per frame → auto-label first, humans verify.

```
 Fleet (onboard model + trigger logic)
   triggers: driver takeover/disengagement, onboard-vs-planner disagreement, low confidence /
             ensemble disagreement, rare-class detectors, shadow-model diffs,
             "find more like this" embedding queries pushed to the fleet
     | upload selected clips (~1%)
     v
 Data lake (clips, calibration, metadata) --> scene index (embeddings, tags, geo, weather, time)
     v
 Offboard auto-labeller (large models, multi-frame, uses past AND future frames, sensor fusion)
     v
 Human QA on low-confidence / sampled labels --> versioned label store
     v
 Training on curated mix (long tail upsampled) --> eval: metrics + scenario suites + closed-loop sim
     v
 Shadow deployment on fleet --> compare to production --> release --> (loop)
```

- **Active learning signals**: uncertainty (ensembles, MC dropout), disagreement (onboard vs offboard auto-labeller = free error signal), novelty (low density in embedding space), coverage/diversity (core-set selection), targeted text→clip search ("person in a wheelchair at night").
- **Why offboard beats onboard**: no latency limit, bigger models and ensembles, access to future frames (hindsight), multiple passes.
- **Offline metrics**: mAP / nuScenes Detection Score (NDS), per-class recall by distance bin, tracking (HOTA, MOTA), onboard latency on target hardware.
- **Offline metrics ≠ safety**: aggregate mAP hides rare but critical misses. Weight errors by planning relevance (close, in-path objects; planner-centric metrics have been proposed in the literature), track false-negative duration for vulnerable road users within braking distance, scenario-suite pass rates, closed-loop simulation outcomes, and a regression suite of "golden" hard cases that must never break.
- **Trade-offs**: auto-label quality vs human cost; upload bandwidth vs coverage; long-tail upsampling vs calibration shift; one large multi-task network (shared compute) vs separate models.

**Follow-ups**
- **Q: How do you stop auto-label errors from being learned?** A: Confidence thresholds, human QA on samples, measure auto-label precision/recall against gold, keep the auto-labeller architecturally different from the onboard model.
- **Q: How do you guarantee reproducibility for audits?** A: Immutable, versioned dataset snapshots and label versions; every model tied to (code, data, label) versions; deterministic eval suites.
- **Q: How do you find a rare scenario quickly?** A: Text/image query over a scene-embedding index of the data lake, then push a trigger classifier to the fleet to collect more.

---

### Case f — Training infrastructure for a multimodal foundation model (RE angle)

**Prompt**: "Design the infrastructure to pretrain a ~7B-parameter vision-language model."

#### 1. Requirements (assumed)
- Model: ~7B params total (e.g., ~6.6B decoder LLM + ~0.4B ViT vision encoder + projector). For the activation estimate below, treat the LLM like a Llama-2-7B-style config: L = 32 layers, hidden h = 4,096, vocab ≈ 32K.
- Data: ~2B image-text pairs + interleaved documents + text; **D ≈ 2T tokens** total (image patches count as tokens; ~576 tokens per image).
- Compute: **512 H100s** (64 nodes × 8), target ≤ ~1 week for the main run; ablations on smaller budgets.
- Questions to ask: shared cluster or dedicated? Storage system (object store vs parallel filesystem)? Required reproducibility? Which evaluations decide success? Data-licensing and opt-out constraints?

#### 2. Data pipeline

```
 sources: web crawl image/alt-text, licensed sets, interleaved docs, (opt-outs, robots.txt respected)
   --> parallel downloader (img2dataset-style) --> resize (e.g., shorter side 512) + re-encode JPEG
   --> metadata table (parquet): url, hash, size, caption, language, scores
   --> filters:  corrupt/size/aspect | language ID | NSFW + CSAM hash checks | PII handling
                 | CLIP image-text similarity >= threshold | optional aesthetic/quality score | text quality
   --> dedup:    exact (content hash/URL) | near-dup (perceptual hash, embedding clustering, SemDeDup-style)
                 | decontamination against eval sets (remove near-dups of COCO/Flickr/benchmark images)
   --> optional recaptioning with a VLM (store alt-text AND synthetic caption)
   --> pack into WebDataset tar shards (~1 GB, ~10-20K samples each) on object storage + shard manifest
   --> training loader: shuffle shard order + in-memory shuffle buffer, decode, tokenise,
       sequence packing to fixed length (4,096), deterministic resume (epoch seed, shard idx, offset)
```

- **Storage**: 2B pairs × ~50 KB per resized JPEG ≈ **100 TB** (+ ~1 TB text/metadata) → ~100K one-GB shards. Embeddings for filtering/dedup: 2B × 768-d × 2 B ≈ 3 TB.
- **CLIP-score filtering** works well in practice (keeping roughly the top ~30 % by CLIP ViT-L/14 score was a strong DataComp baseline; combined with image-based clustering it was DataComp's best baseline and produced DataComp-1B). Cost: ViT-L/14 at 224² ≈ 81 GMACs ≈ 162 GFLOPs per image → 2B × 162 GFLOPs ≈ 3.2·10²⁰ FLOPs ≈ 3.2·10²⁰ / (4·10¹⁴ FLOP/s) ≈ 8·10⁵ GPU-s ≈ **225 H100-hours ≈ 0.4 % of the training run** (computed below) — cheap relative to its benefit; I/O and decoding usually dominate.
- **Loader throughput check (VLM)**: at ~4.8M tokens/s (below), if ~half are image tokens at 576/image → ≈ 4.2K images/s cluster-wide ≈ 65 images/s per node and ≈ 210 MB/s of reads — easy.
- **Contrast (CLIP ViT-B/16 training)**: ≈ 3 × (35 + ~10) ≈ 135 GFLOPs per sample → even at 30 % of H100 peak ≈ 2.2K samples/s per GPU ≈ **18K images/s per node** → CPU JPEG decoding needs tens of cores per node; use pre-resized shards and/or GPU decoding (nvJPEG/DALI). Small models are data-loader-bound; big models are compute-bound.

#### 3. Memory and compute estimate (show this arithmetic)

**Model states (AdamW, mixed precision) = 16 bytes/param**

| Component | Bytes/param | 7B model |
|---|---|---|
| BF16 weights | 2 | 14 GB |
| BF16 gradients | 2 | 14 GB |
| FP32 master weights | 4 | 28 GB |
| Adam first moment (FP32) | 4 | 28 GB |
| Adam second moment (FP32) | 4 | 28 GB |
| **Total** | **16** | **7·10⁹ × 16 B = 112 GB** |

→ Does not fit on an 80 GB GPU even before activations, so plain DDP is out. Shard the states with **FSDP / ZeRO-3**. With **HSDP** (shard within a node of 8, replicate across 64 nodes): 112 / 8 = **14 GB/GPU** of states.

**Activations** (per GPU; sequence s = 4,096, micro-batch b = 2, h = 4,096, L = 32). A common estimate (Korthikanti et al., 2022) is ≈ 34·s·b·h bytes per layer with 16-bit activations for a GPT-style block (4h MLP, dropout masks included) once the attention-score term is removed by FlashAttention; SwiGLU/no-dropout variants differ somewhat:
- Per layer: 34 × 4,096 × 2 × 4,096 ≈ 1.14 GB → × 32 layers ≈ **36.5 GB** without recomputation.
- With full activation checkpointing you keep only layer inputs: 2·s·b·h·L = 2 × 4,096 × 2 × 4,096 × 32 ≈ 2.1 GB, plus ~1.1 GB for the layer being recomputed — at the price of ~33 % more compute (one extra forward: 8N instead of 6N per token).
- Logits: s × b × V × 4 B (FP32) = 4,096 × 2 × 32K × 4 B ≈ 1 GB (≈ 4 GB with a 128K vocab → use chunked/fused cross-entropy).
- Vision encoder activations: a few GB more (or checkpoint/freeze it in early stages).

**Total**: 14 (states) + ~1–2 (gathered params/prefetch) + 36.5 (activations) + ~1–3 (logits, buffers) ≈ **53–56 GB** (+ a few GB for the vision encoder), plus allocator fragmentation and communication buffers → fits in 80 GB. You can skip full checkpointing (or use selective checkpointing) and get higher throughput. *Measure the real peak; these are estimates.*

**Training compute** (C ≈ 6·N·D: ~2N FLOPs/token forward + ~4N backward):
- C = 6 × 7·10⁹ × 2·10¹² = **8.4·10²² FLOPs**.
- H100 SXM BF16 dense peak ≈ 989 TFLOP/s (approximate); assume **MFU = 40 %** → ≈ 396 TFLOP/s per GPU.
- GPU-seconds = 8.4·10²² / 3.96·10¹⁴ ≈ 2.12·10⁸ s → **≈ 59,000 GPU-hours**.
- On 512 GPUs: 59,000 / 512 ≈ 115 h ≈ **4.8 days**; budget +10–20 % for restarts, evals and stragglers → ~5.5 days.
- Cluster throughput: tokens/s = MFU × n_GPU × peak / (6N) = 0.4 × 512 × 989·10¹² / (4.2·10¹⁰) ≈ **4.8M tokens/s** (≈ 9.4K tokens/s per GPU).
- Global batch: 512 GPUs × b = 2 × 4,096 = **4.19M tokens/step** (no gradient accumulation) → ≈ 2·10¹² / 4.19·10⁶ ≈ 477K steps; step time ≈ 4.19M / 4.8M ≈ **0.87 s**.
- Cost at an assumed $3/GPU-hour: ≈ **$177K** for the final run; ablations and failed runs often cost several times that.
- Caveat: 6·N·D counts only parameter matmuls. Attention score/value matmuls add ≈ 6·L·s·h FLOPs/token for causal attention (12·L·s·h without the causal discount, the convention in PaLM's MFU formula). Here 6 × 32 × 4,096 × 4,096 ≈ 3.2·10⁹ vs 6N = 4.2·10¹⁰ → ~8 % (~15 % with the 12× convention). It grows with sequence length (long video contexts).
- **MFU** = (model FLOPs per token × tokens/s) / (n_GPU × peak). **HFU** also counts recomputation. Published large dense runs report roughly 35–55 % MFU (e.g., Llama 3 405B reported 38–43 % on H100s in BF16); small models usually get less.

#### 4. Parallelism plan

| Situation | Plan |
|---|---|
| Model + optimizer + activations fit on one GPU (≲ 1–3B params on 80 GB; 16 B/param alone caps at 5B) | DDP (+ gradient accumulation) |
| States do not fit, model up to ~tens of B | FSDP/ZeRO-3, or **HSDP** (shard intra-node, replicate inter-node); activation checkpointing as needed |
| Very large (≳ 70B) or layer activations too large | + tensor parallelism (TP ≤ 8, inside the NVLink domain) + pipeline parallelism across nodes |
| Long sequences (video, high-res tiling) | + context/sequence parallelism |
| MoE | + expert parallelism |
| DP alone would push the global batch past the useful (critical) batch size | add TP/PP to lower the DP degree |

**Chosen for 7B on 512 H100s: HSDP (shard 8 × replicate 64), BF16 compute, FP32 master weights, FlashAttention, no TP/PP.**
Communication check per step (ideal bandwidths, approximate):
- Intra-node (FSDP): all-gather params for forward + again for backward + reduce-scatter grads ≈ 3 × (7/8) × 14 GB ≈ 37 GB per GPU over NVLink (~450 GB/s per direction on H100) ≈ **80 ms**, overlapped with compute.
- Inter-node (replicas): all-reduce each GPU's 1/8 gradient shard (14 GB / 8 = 1.75 GB in BF16) across 64 replicas; ring all-reduce sends ≈ 2 × 1.75 = 3.5 GB per GPU → at ~50 GB/s (one 400 Gb/s NIC per GPU) ≈ **70 ms**, overlapped with backward. Both are small vs the 0.87 s step.
- Why not TP here: it adds latency-sensitive all-reduces inside every layer; at 7B FSDP alone fits. Why not full FSDP across all 512 GPUs: every layer's all-gather would cross the slower inter-node network.

#### 5. Checkpointing and failure recovery
- **Checkpoint size**: FP32 master weights + Adam moments = 12 B/param → 7·10⁹ × 12 B = **84 GB** (+14 GB if you also save BF16 weights; they can be derived) + data-loader state, RNG states, step, LR schedule.
- **Sharded, asynchronous save**: each rank writes its own shard in parallel (distributed checkpoint format); replicas hold identical shards, so only one replica group needs to write (or spread the writing across replicas). Copy GPU → host RAM first (~10.5 GB per shard owner → around a second or less over PCIe), then upload to object storage in the background. Blocking time ≈ seconds.
- **How often** (Young/Daly): T_opt ≈ √(2 · δ · MTBF), where δ = blocking cost of one checkpoint.
  - MTBF assumption: the Llama 3 paper reported 419 unexpected interruptions in 54 days on 16,384 H100s (≈ 1 every ~3 h). Scaling linearly to 512 GPUs (32× fewer) → **MTBF ≈ 100 h ≈ 360,000 s** (cluster-dependent; treat as an assumption).
  - With δ = 10 s: T ≈ √(2 × 10 × 360,000) ≈ 2,700 s ≈ **45 min**.
  - Expected overhead ≈ δ/T + (T/2 + R)/MTBF, with restart time R ≈ 15 min: 10/2,700 + (1,350 + 900)/360,000 ≈ 0.4 % + 0.6 % ≈ **1 %**.
  - Same formula at 16K GPUs (MTBF ≈ 3 h): T ≈ √(2 × 10 × 10,800) ≈ 8 min, overhead ≈ 2 % + (230 + 900)/10,800 ≈ **13 %** → at large scale **restart time dominates**: hot-spare nodes, pre-pulled containers, in-memory checkpoints replicated to peers, fast collective initialisation.
- Keep the last K checkpoints plus periodic permanent ones (for evals and rollback).
- **Failure handling**:
  - Pre-flight: burn-in, GPU diagnostics, NCCL bandwidth tests, ECC/thermal checks before nodes join a job.
  - Detection: NCCL/collective timeouts, per-rank heartbeats, step-time outliers (stragglers), NaN/Inf in loss or gradient norm.
  - Action: drain the bad node, swap in a hot spare, auto-resume from the latest checkpoint (idempotent launch script), exact data-loader resume.
  - **Loss spikes**: gradient clipping, skip batches with abnormal gradient norms, roll back to an earlier checkpoint and skip the offending data window or lower the LR; stabilisers such as z-loss and QK-normalisation.
  - **Silent data corruption**: replicas should hold bit-identical parameters → periodically compare checksums across replicas; re-run suspicious steps on other hardware.

#### 6. Eval harness
- An eval service watches for new checkpoints and runs asynchronously on separate GPUs (never blocks training).
- Every ~1–2K steps: held-out validation loss per data source (catches data-mixture problems early).
- Periodically: zero-/few-shot benchmarks — VQA (VQAv2, TextVQA, DocVQA, ChartQA), reasoning (MMMU), captioning (COCO CIDEr), retrieval (COCO/Flickr Recall@k), plus safety and hallucination evals.
- Hygiene: fixed prompts, seeds and decoding settings; versioned eval sets; contamination checks against training data; confidence intervals; regression alerts on a dashboard.
- **Small-scale proxies**: choose data mixtures and filters with cheap runs (e.g., ~1 % of the compute) and scaling trends before the full run.

#### 7. Overall infrastructure

```
             +----------------- object store: shards, manifests, checkpoints -----------------+
             |                                                                                |
 data-prep cluster (CPU + a few GPUs for filtering/recaptioning) --> filtered, deduped shards -+
                                                    | streaming reads (~0.2-1 GB/s aggregate)
                                                    v
 scheduler (Slurm / Kubernetes) --> training job: 64 nodes x 8 H100, HSDP (shard 8 x replicate 64)
        ^                             |-- async checkpoint every ~30-60 min --> object store
        | auto-restart                |-- metrics: loss, grad norm, tokens/s, MFU, step time --> tracker + alerts
        |                             v
 health monitor (heartbeats, collective timeouts, GPU diagnostics) --> drain node / swap hot spare
                                                    |
 eval service <-- new-checkpoint event -------------+--> dashboards, regression alerts
```

#### 8. Monitoring during training
- Throughput (tokens/s, MFU), **data-loader wait time** (should be ~0), compute/communication overlap from profiler traces, memory high-water mark, loss and gradient-norm curves per data source, LR schedule, expert-load balance (if MoE).

#### Follow-up questions interviewers ask
- **Q: MFU is 20 %. How do you debug it?** A: Profile one step: data-loader stalls, exposed communication (not overlapped), small micro-batches/matmuls, recomputation overhead (HFU vs MFU), unfused kernels or no FlashAttention, padding waste (use packing), stragglers, CPU launch overhead (CUDA graphs, compilation).
- **Q: How do you pick global batch size and LR?** A: Stay below the critical batch size (small-scale sweeps), scale LR with batch within limits, warmup + cosine or warmup-stable-decay schedule; verify with proxy runs.
- **Q: Images have variable resolution and aspect ratios.** A: Aspect-ratio bucketing, dynamic tiling, or NaViT-style patch packing; pack sequences to a fixed length to avoid padding.
- **Q: How do you scale to 4K GPUs?** A: Keep the global batch sane (lower micro-batch, add TP 2–4 or PP), topology-aware placement, faster checkpoint/restart (failures are 8× more frequent), better straggler handling.
- **Q: Should you train in FP8?** A: Hopper FP8 peak is ~2× BF16; it needs per-tensor scaling (e.g., Transformer Engine) and close monitoring for divergence; realised end-to-end gains are usually well below 2×.
- **Q: How do you make runs reproducible?** A: Pinned shard manifests and data versions, configs in version control, seeds, deterministic loader order; aim for statistical reproducibility (bitwise determinism is expensive).
- **Q: How would you train the stages?** A: A common published recipe: (1) align — freeze the vision encoder and LLM, train the projector on caption pairs; (2) pretrain/unfreeze on interleaved and caption data; (3) instruction-tune on curated multimodal instructions; evaluate after each stage.

---

## 3. Back-of-envelope cheat sheet

### 3.1 Latency numbers worth knowing (orders of magnitude; hardware-dependent)

| Operation | Approx. time |
|---|---|
| L1 cache reference | ~1 ns |
| Branch mispredict | ~3–5 ns |
| Main memory reference | ~100 ns |
| Compress 1 KB with a fast codec | ~2 µs |
| Read 1 MB sequentially from RAM (one core, ~10–20 GB/s) | ~50–100 µs |
| Random 4 KB read from NVMe SSD | ~20–100 µs |
| Round trip within a datacenter | ~0.1–0.5 ms |
| In-memory key-value cache get over the network (same DC) | ~0.2–1 ms |
| Read 1 MB sequentially from NVMe SSD (~2–7 GB/s) | ~0.15–0.5 ms |
| HDD seek | ~5–10 ms |
| Read 1 MB sequentially from HDD (~150–250 MB/s) | ~5 ms |
| Object-store GET (first byte) | ~10–100 ms |
| Mobile network RTT (4G/LTE) | ~30–100 ms |
| Cross-continent RTT (US east ↔ west) | ~60–80 ms |
| Transatlantic RTT | US east ↔ western Europe ~70–90 ms; US west ↔ Europe ~140–150 ms (the classic "California → Netherlands → California ≈ 150 ms") |

Rule: a RAM reference is ~1,000× faster than an SSD random read; an SSD random read is ~100× faster than an HDD seek; one cross-region round trip (~60–150 ms) takes longer than a batch-1 forward pass of a ViT-B on a datacenter GPU (a few to ~10 ms).

### 3.2 Throughput and interconnect

| Link | Approx. bandwidth |
|---|---|
| 1 Gb/s Ethernet | 125 MB/s |
| 100 Gb/s Ethernet | 12.5 GB/s |
| InfiniBand NDR 400 Gb/s (one NIC) | 50 GB/s |
| PCIe Gen4 x16 | ~32 GB/s per direction (~25 practical) |
| PCIe Gen5 x16 | ~64 GB/s per direction |
| NVLink (A100) | 600 GB/s total per GPU (bidirectional) |
| NVLink (H100) | 900 GB/s total per GPU (~450 per direction) |

### 3.3 GPU specs to quote (approximate; dense, without structured sparsity; check the exact SKU)

| GPU | BF16/FP16 dense TFLOP/s | FP8 dense | Memory | Memory BW |
|---|---|---|---|---|
| T4 | ~65 (FP16; no BF16) | — | 16 GB GDDR6 | ~320 GB/s |
| A10 | ~125 | — | 24 GB GDDR6 | ~600 GB/s |
| L4 | ~121 | ~242 | 24 GB GDDR6 | ~300 GB/s |
| A100 SXM 80 GB | **~312** | — (no FP8) | 80 GB HBM2e | **~2.0 TB/s** (40 GB model: ~1.6) |
| H100 SXM | **~989** | ~1,979 | 80 GB HBM3 | **~3.35 TB/s** |
| H100 PCIe | ~756 | ~1,513 | 80 GB HBM2e | ~2.0 TB/s |
| H200 SXM | ~989 (same compute as H100) | ~1,979 | 141 GB HBM3e | ~4.8 TB/s |
| B200 (varies by system/power config) | ~2,250 | ~4,500 | ~180–192 GB HBM3e | ~8 TB/s |

- Vendor datasheets often quote **sparse** numbers (2× dense). Always say "dense BF16".
- **Ridge point** (FLOPs per byte needed to be compute-bound) = peak FLOP/s ÷ bandwidth: A100 ≈ 312 / 2.0 ≈ 156; H100 ≈ 989 / 3.35 ≈ 295. Below that arithmetic intensity, you are memory-bound.
- Memory-bound example: batch-1 LLM decoding must read all weights per token → 7B in BF16 = 14 GB / 3.35 TB/s ≈ **4.2 ms/token lower bound** on an H100. Batching amortises the weight reads.
- Realistic utilisation: training ~35–55 % MFU for well-tuned large dense runs; inference often 20–50 % (lower at small batch).

### 3.4 Model FLOPs, images/sec and memory

**FLOPs per image (forward, 224² unless noted; 1 MAC = 2 FLOPs)**

| Model | Params | GMACs | GFLOPs |
|---|---|---|---|
| ResNet-50 | 25M | ~4.1 | ~8.2 |
| ViT-S/16 | 22M | ~4.6 | ~9.2 |
| ViT-B/32 (CLIP) | 88M | ~4.4 | ~8.8 |
| ViT-B/16 | 86M | ~17.6 | ~35 |
| ViT-L/14 | 304M | ~81 | ~162 |
| ViT-L/14 @ 336² | 304M | ~190 | ~380 |

Quick ViT estimate: MACs ≈ tokens × layers × 12·d² (+ attention 2·tokens²·d per layer). ViT-B/16: 197 × 12 × 12 × 768² ≈ 16.7 GMACs + ~0.7 attention ≈ 17.4 GMACs ✓.

**Images/sec ≈ peak FLOP/s × utilisation ÷ FLOPs per image** (model compute only):

| Model | A100 at 30 % (~94 TFLOP/s) | H100 at 30 % (~297 TFLOP/s) |
|---|---|---|
| ResNet-50 | ~11K img/s | ~36K img/s |
| ViT-B/16 | ~2.7K img/s | ~8.5K img/s |
| ViT-L/14 | ~580 img/s | ~1.8K img/s |
| SDXL-class 1024², 30 steps + CFG | ~0.2–0.5 img/s (measured-latency based, not FLOPs) | ~0.5 img/s (assumes ~2 GPU-s/image) |

Measured numbers vary widely with framework, batch size, precision (INT8/TensorRT can be several × faster) and **JPEG decoding**, which is often the real bottleneck: roughly a few ms per ~0.2–1 MP JPEG per CPU core → hundreds of decodes/s per core; use GPU decoding (nvJPEG/DALI, NVDEC for video) or pre-resized inputs.

**Memory rules**
- Training (AdamW, mixed precision): **16 B/param** + activations. 1B params → 16 GB of states.
- Inference weights: BF16 2 B/param, INT8 1 B, INT4 0.5 B. 7B → 14 / 7 / 3.5 GB.
- **KV cache per token** = 2 (K and V) × layers × KV heads × head dim × bytes. Llama-2-7B-style (32 layers, 32 heads × 128, FP16): 2 × 32 × 4,096 × 2 B = **0.5 MB/token** → 4K-token context ≈ 2 GB per sequence. With GQA (8 KV heads, e.g., Llama-3-8B): 128 KB/token.
- VLM image tokens: e.g., 336² / 14² = 24 × 24 = 576 tokens per image; high-res tiling can produce thousands → prefill cost ≈ 2·N·tokens.

**Compute rules**
- Training ≈ **6·N·D** FLOPs (N params, D tokens); inference forward ≈ 2·N FLOPs/token.
- Compute-optimal LLM pretraining (Chinchilla) ≈ 20 tokens/param; inference-oriented models are trained far beyond that.
- Training time = 6·N·D ÷ (n_GPU × peak × MFU).
- Ring all-reduce: each GPU sends/receives ≈ 2·(n−1)/n × message size. ZeRO-3/FSDP ≈ 1.5× DDP's communication volume (≈ 3× vs 2× the parameter size).
- Full activation checkpointing ≈ +33 % compute.

### 3.5 Storage per embedding and ANN

| N | d | Format | Bytes/vector | Total |
|---|---|---|---|---|
| 1M | 512 | float32 | 2,048 | ~2 GB |
| 100M | 256 | float32 | 1,024 | ~102 GB |
| **1B** | **512** | **float32** | **2,048** | **~2 TB** (2.05·10¹² B ≈ 1.86 TiB) |
| 1B | 512 | float16 | 1,024 | ~1 TB |
| 1B | 512 | int8 | 512 | ~512 GB |
| 1B | 512 | PQ, m = 64 × 8 bits | 64 | ~64 GB (+ ~8 GB for 8-byte ids) |
| 1B | 512 | binary (1 bit/dim) | 64 | ~64 GB |

- **PQ**: split the d-dim vector into m sub-vectors, quantise each to one of 2⁸ = 256 centroids → m bytes/vector; distances via lookup tables. Compression vs float32 = 4d/m (512-d, m = 64 → 32×). OPQ (a learned rotation) reduces the error.
- **IVF**: k-means into nlist cells, search nprobe cells → scans ≈ N × nprobe / nlist codes. Rule of thumb nlist ≈ 4√N–16√N (the Faiss wiki states it for < 1M vectors and suggests 2¹⁶–2²⁰ lists with an HNSW coarse quantiser at 1M–1B). Example 1B vectors (4√N–16√N ≈ 126K–506K), nlist = 2¹⁸ = 262,144, nprobe = 128 → ≈ 488K codes (≈ 31 MB of 64-byte codes) per query.
- **HNSW**: graph with ~2·M neighbours at the bottom layer → ~2·M × 4 B ≈ 256 B/vector of links at M = 32, on top of the vectors (Faiss documents HNSW memory as d × 4 + M × 2 × 4 bytes/vector). High recall, low latency, memory-heavy.
- **Brute force at 1B**: 1B × 512 ≈ 5·10¹¹ MACs per query, and 1 TB (FP16) must be read → needs sharding across many GPUs/hosts → use ANN + re-rank instead.
- Measure ANN recall@k against exact search; re-rank the top candidates with full-precision vectors if needed.

### 3.6 Traffic, cost and sizing formulas

- **QPS**: requests/day ÷ 86,400. 1M/day ≈ 12 QPS; 100M/day ≈ 1.2K QPS. Peak ≈ 2–3× average.
- **Little's law**: concurrency = throughput × latency. 500 QPS × 0.2 s = 100 requests in flight.
- **GPU count** = peak QPS × GPU-seconds per request ÷ target utilisation (0.5–0.7). Add N+1 redundancy per region.
- **Cost per 1M inferences** = (10⁶ ÷ throughput per GPU) ÷ 3,600 × $/GPU-hour. Example: 2,000 img/s on a $2/h GPU → 500 s ≈ 0.14 h → **≈ $0.28 per 1M images**.
- **GPU-hour prices** vary widely by provider, region, commitment and year (very roughly $2–10 per H100-hour and $1–4 per A100-hour in 2025–26; inference cards like L4/T4 well under $1/h). **State your assumption** (e.g., "$3/GPU-hour").
- **Image/video sizes**: 224×224×3 uint8 = 150,528 B raw (~10–30 KB as JPEG); 1024² RGB = 3 MiB raw (~150–500 KB as JPEG/WebP); 12 MP phone photo ≈ 2–5 MB JPEG (HEIC roughly half); 1080p H.264 at 4–8 Mb/s ≈ 30–60 MB per minute.
- **Storage growth** = items/day × bytes/item × retention days (× replication factor, typically 3 in distributed storage).
- **Labelling cost** = items × seconds/item ÷ 3,600 × $/hour × raters per item.
- **A/B sample size** ≈ 16·σ²/δ² per arm (α = 0.05, power 80 %).

---

## 4. Pitfalls and practice prompts

### 4.1 Common pitfalls that get you marked down

| Pitfall | What to do instead |
|---|---|
| Jumping to the model architecture in minute 2 | Requirements → framing → data first; the model comes after you know the constraints |
| No baseline | Name a heuristic or pretrained-embedding baseline and say what it validates |
| No metrics, or metrics disconnected from the goal | Write loss ↔ offline metric ↔ online metric side by side; choose an operating point |
| Accuracy on imbalanced data; FID as the only generative metric; Recall@k without separating ANN recall | Pick metrics that match the decision (PR-AUC/recall at precision, human eval + alignment metrics, ANN vs exact) |
| Ignoring serving cost/latency ("run a 7B VLM on every upload") | Do the QPS → GPU → $ math; propose a cascade, distillation, quantisation |
| No monitoring, retraining or feedback-loop plan | Drift, delayed labels, audits, exploration, retraining triggers, fallbacks |
| Data leakage (random image-level splits, near-duplicates, eval contamination) | Split by entity/time; dedupe across splits; decontaminate |
| Treating labels as given | Explain where labels come from, their cost, noise and bias |
| Numbers without units or assumptions; or no numbers at all | "Assume X → Y QPS → Z GPUs"; round aggressively; sanity-check orders of magnitude |
| Over-engineering for 1B users on day 1 | Start simple, say what changes at 10× and 100× |
| One "right answer", no trade-offs | Two options, pros/cons, choice under the stated constraints |
| Going 20 minutes deep on one component | Breadth first; ask where to go deep |
| Not driving (waiting for prompts) or not listening (ignoring hints) | Announce the plan, check in at phase boundaries, follow the interviewer's steer |
| Ignoring failure modes | Fallbacks, fail-open/closed, graceful degradation, rollback |
| (Theory-strong candidates) proposing training from scratch or a novel architecture where a pretrained model + fine-tuning would ship in weeks | Lead with the pragmatic path; present the research idea as the iteration with an ablation plan |
| (RS) no hypotheses or ablation plan | State what you expect, how you'd test it, and what result would change your mind |
| (RE) no throughput, memory or failure math | 16 B/param, 6·N·D, MFU, communication volume, checkpoint interval |

### 4.2 Practice prompts (no solutions — use them for timed mocks)

1. Design near-duplicate and duplicate image detection for a platform with 10B stored images and 100M uploads/day.
2. Design automatic alt-text (image captioning) for accessibility in a social feed, in 20 languages.
3. Design a serving system for a 7B vision-language chat assistant that accepts up to 10 images per conversation.
4. Design a text-to-video generation service (short clips), including cost per second of video.
5. Design an AI-generated-image (deepfake) detection system for a news or social platform.
6. Design real-time background segmentation for video calls that runs on laptops and phones.
7. Design an invoice/receipt understanding pipeline (OCR + layout + extraction) for an enterprise product.
8. Design an instruction-based image-editing feature ("remove the person, make it sunset").
9. Design the data-curation pipeline for training a CLIP-style model on 5B web image-text pairs.
10. Design a virtual try-on feature for a fashion e-commerce app.
11. Design face clustering ("People" albums) for a photo app under strict privacy constraints.
12. Design a brand-safety and quality scorer for ad creatives (image + text + video).
13. Design an out-of-stock detection system for retail shelves using edge cameras in 5,000 stores.
14. Design the evaluation platform for a team shipping a new text-to-image model every month.
15. Design video understanding for search over a 100M-video library (find "the moment where X happens").

For each mock: 45 minutes, a partner playing interviewer who steers at least once ("let's go deeper on serving"), then score yourself with the §1.6 checklist and the §4.1 table.
