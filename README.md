<div align="center">

<img src="torchleet-header.png" alt="TorchLeet" width="700">

**<!--total-->98<!--/total--> PyTorch interview problems as Jupyter notebooks, with an auto-grader —
tuned for ML Engineer, Research Scientist and Research Engineer interviews.**

</div>

---

This is a personal fork of [Exorust/TorchLeet](https://github.com/Exorust/TorchLeet). On top of
the original problem set it has:

- **An audit of every original problem** — statement, math, solution code and grader were each
  checked against the canonical definition / PyTorch built-ins, and the bugs found were fixed.
  What was wrong and what changed: [`study/AUDIT.md`](study/AUDIT.md).
- **A new `v4/` set** aimed at MLE / RS / RE interviews in CV, multimodal and generative
  modelling: classical ML and evaluation, deep-learning fundamentals written by hand, vision
  ops, generative models, multimodal models and ML systems.
- **A 4-week study plan** (Chinese) plus review material in [`study/`](study/).

## Start here

| | |
|---|---|
| 📅 **Study plan (中文, 4 周冲刺)** | [`study/README.md`](study/README.md) |
| 📚 All problems, by topic and difficulty | [`PROBLEMS.md`](PROBLEMS.md) |
| 🧠 ML breadth review (Q&A) | [`study/ml-breadth.md`](study/ml-breadth.md) |
| 🏗️ ML system design guide | [`study/ml-system-design.md`](study/ml-system-design.md) |
| ⌨️ PyTorch cheat sheet | [`study/pytorch-cheatsheet.md`](study/pytorch-cheatsheet.md) |
| 🔍 Audit report | [`study/AUDIT.md`](study/AUDIT.md) |

## Setup

```bash
git clone https://github.com/Fengyee/TorchLeet.git && cd TorchLeet
pip install torch            # CPU is enough for every problem (see the Triton note below)
pip install -e python/       # the `torchleet` grader (not on PyPI — install it from the repo)
pip install jupyter
jupyter lab
```

On Colab, each question notebook has an "Open in Colab" badge; install the grader there with
`!pip install -q "git+https://github.com/Fengyee/TorchLeet#subdirectory=python"`.

**Triton problems without a GPU:** `pip install triton` (Linux wheels only — on macOS use Colab or
a Linux box) and set `TRITON_INTERPRET=1` *before* importing triton; kernels then run on CPU in
Triton's interpreter, which is enough to check correctness (not speed). The grader skips the
kernel checks, rather than failing them, when it can't run them.

## How to work a problem

1. Open the **question** notebook (the one without `_SOLN`). Read the statement and write down the
   input/output shapes before writing any code.
2. Fill in the `# TODO` stubs. Run the notebook's own test cells.
3. Grade it — checks verify properties (agreement with PyTorch built-ins, invariants), so any
   correct implementation passes, not just ours:
   ```python
   from torchleet import check, hint, status
   check("iou-nms", box_iou=box_iou, nms=nms, batched_nms=batched_nms)
   hint("iou-nms")      # progressive hints: hint("iou-nms", 2), hint("iou-nms", 3)
   status()             # what you've passed on this machine
   ```
4. Only then open the `_SOLN` notebook. New `v4/` solutions end with *Why it works*, *Common bugs*
   and *Interview follow-ups* — read those even when you passed.

<!--graded-->91<!--/graded--> of <!--total-->98<!--/total--> problems are auto-graded.

## What's inside

| Directory | Contents |
|---|---|
| `torch/` | PyTorch basics → hard: datasets, CNNs, RNN/LSTM, custom autograd, Adam/AdamW/Muon, Transformer, GAN, XAI |
| `llm/` | LLM building blocks: BPE, positional embeddings (sinusoidal, RoPE), RMSNorm, attention / MHA / GQA, FlashAttention's online softmax, a SmolLM-style model |
| `v3/` | Classical ML, data loading, modern architectures (ViT-MAE, Swin, CLIP, DDPM, DDIM+CFG, Mamba, MoE), alignment (LoRA, DPO, PPO, GRPO), decoding, LLM inference, GPU systems (Triton, FSDP, ring attention) |
| `v4/` | **New:** ML fundamentals & evaluation, deep-learning fundamentals, vision, generative, multimodal, ML systems |
| `study/` | Study plan, breadth review, system design, cheat sheet, audit report |
| `python/` | The `torchleet` grader (`check`, `hint`, `status`) and its specs |
| `scripts/` | `generate.py` (manifests → indexes), `validate_specs.py` (every solution must pass its own grader) |

> **About company tags.** Some problem manifests and notebooks name companies (Google, Meta, …).
> Every one of those tags is marked `inferred` in the manifests — none is backed by a dated
> first-hand interview report (see [SOURCES.md](SOURCES.md)). Treat them as "the kind of
> company that cares about this topic", not as "this exact question was asked".

## AI tutor (optional)

The upstream project publishes an MCP server that turns Claude Code / Cursor / Codex into a
no-spoilers tutor with progressive hints and mock-interview modes:

```bash
claude mcp add torchleet -- npx -y torchleet-mcp
```

It serves the **upstream** problem catalogue, so it does not know about the `v4/` problems or
the fixes in this fork; for those, use `hint()` from the grader and the study plan.

## Maintaining

```bash
python3 scripts/generate.py all         # regenerate hints, badges, PROBLEMS.md, README counts, problems.json
python3 scripts/generate.py check       # validate manifests (CI gate)
python3 scripts/validate_specs.py       # every solution passes its own grader
python3 python/test_grader.py           # grader regression tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the shape of a problem and how to add grading.

---

## Credits

Original problems, grader and tooling by [Chandrahas Aroori](https://github.com/Exorust),
[Caslow Chien](https://github.com/CaslowChien) and the TorchLeet contributors (MIT licensed).

<!-- BEGIN:contributors -->
## Contributors

Thanks to everyone who has added problems, fixes and solutions (11 so far):

<table>
    <tr>
      <td align="center"><a href="https://github.com/Exorust"><img src="https://avatars.githubusercontent.com/u/20578676?v=4" width="72px" alt="Exorust"/><br /><sub><b>Exorust</b></sub></a></td>
      <td align="center"><a href="https://github.com/samhubs"><img src="https://avatars.githubusercontent.com/u/52785438?v=4" width="72px" alt="samhubs"/><br /><sub><b>samhubs</b></sub></a></td>
      <td align="center"><a href="https://github.com/AtulAravindDas"><img src="https://avatars.githubusercontent.com/u/161445393?v=4" width="72px" alt="AtulAravindDas"/><br /><sub><b>AtulAravindDas</b></sub></a></td>
      <td align="center"><a href="https://github.com/CaslowChien"><img src="https://avatars.githubusercontent.com/u/99608452?v=4" width="72px" alt="CaslowChien"/><br /><sub><b>CaslowChien</b></sub></a></td>
      <td align="center"><a href="https://github.com/emmanuel-ferdman"><img src="https://avatars.githubusercontent.com/u/35470921?v=4" width="72px" alt="emmanuel-ferdman"/><br /><sub><b>emmanuel-ferdman</b></sub></a></td>
      <td align="center"><a href="https://github.com/PadLex"><img src="https://avatars.githubusercontent.com/u/23060820?v=4" width="72px" alt="PadLex"/><br /><sub><b>PadLex</b></sub></a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/bargav25"><img src="https://avatars.githubusercontent.com/u/52544154?v=4" width="72px" alt="bargav25"/><br /><sub><b>bargav25</b></sub></a></td>
      <td align="center"><a href="https://github.com/ezhoureal"><img src="https://avatars.githubusercontent.com/u/30425920?v=4" width="72px" alt="ezhoureal"/><br /><sub><b>ezhoureal</b></sub></a></td>
      <td align="center"><a href="https://github.com/Hylthek"><img src="https://avatars.githubusercontent.com/u/76478554?v=4" width="72px" alt="Hylthek"/><br /><sub><b>Hylthek</b></sub></a></td>
      <td align="center"><a href="https://github.com/gulbaki"><img src="https://avatars.githubusercontent.com/u/76937950?v=4" width="72px" alt="gulbaki"/><br /><sub><b>gulbaki</b></sub></a></td>
      <td align="center"><a href="https://github.com/jg-eno"><img src="https://avatars.githubusercontent.com/u/147611384?v=4" width="72px" alt="jg-eno"/><br /><sub><b>jg-eno</b></sub></a></td>
    </tr>
</table>
<!-- END:contributors -->
