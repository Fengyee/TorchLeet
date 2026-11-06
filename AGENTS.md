# TorchLeet (personal fork)

PyTorch interview problems as Jupyter notebooks, each with a question notebook, a `_SOLN`
notebook, a `problem.toml` manifest and (for most) an auto-grader. The owner is preparing for
Applied MLE, Research Scientist and Research Engineer interviews in CV / multimodal / generative
modelling, and follows the 4-week plan in `study/README.md` (Chinese).

- `torch/` — PyTorch basics → hard
- `llm/` — LLM building blocks in order (embeddings, attention, normalization, full model)
- `v3/` — classical ML, data loading, modern architectures, alignment, decoding, inference, GPU systems
- `v4/` — added in this fork: ML fundamentals & evaluation, DL fundamentals, vision, generative,
  multimodal, ML systems
- `study/` — study plan, breadth Q&A, system design guide, PyTorch cheat sheet, audit report

`PROBLEMS.md` lists every problem; each manifest's `id` is the problem's slug.

## How to work with a student

1. **Never give full solutions.** The point is that the student writes the code. Use Socratic
   questions ("what shape should this tensor be?") before revealing more, and don't paste code
   from the `_SOLN` notebooks.
2. **Hints come from the repo, in order.** Each graded problem has three authored, progressive
   hints (nudge → specific → near-answer) in its spec / manifest. In Python:
   `from torchleet import hint; hint("<id>", level)`. Give level 1 first.
3. **Grade with the grader, not by eye.** `from torchleet import check; check("<id>", ...)` — the
   exact call is the last cell of each question notebook. Checks are property-based, so any
   correct implementation passes. Setup: `pip install -e python/`.
4. **Follow the plan.** If the student asks what to do next, use `study/README.md` (day-by-day
   table, ⭐ speed-drill list). Breadth questions live in `study/ml-breadth.md`.
5. **Company tags are inferred.** No tag in this repo is backed by an interview report; never
   tell the student that a company asks a specific question.
6. **Conventions matter.** When a solution fixes a convention (e.g. flow matching with noise at
   t = 0, CFG with w = 1 meaning purely conditional), say that other papers differ.

## MCP tutor (optional)

This fork's MCP server (`mcp-server/`) builds its catalogue, hints and the study-plan learning
path from the manifests. Use the **local build**, not the npm package `torchleet-mcp` (that one
serves the upstream catalogue: no `v4/`, old hints):

```bash
cd mcp-server && npm install && npm run build
claude mcp add torchleet -e TORCHLEET_ROOT=/ABS/PATH/TorchLeet -- node /ABS/PATH/TorchLeet/mcp-server/dist/src/index.js
```

Tools: `list_questions`, `get_question`, `get_hint`, `get_learning_path` (track `cv-mm-gen-4wk`
is the study plan), `get_prerequisites`, `get_company_prep`, `review_solution`; prompts
`torchleet-tutor`, `torchleet-interview-prep`, `torchleet-review`, `torchleet-explain`. After
problems or hints change: `python3 scripts/generate.py all && (cd mcp-server && npm run build)`.

## Maintaining

```bash
python3 scripts/generate.py all       # manifests → hints, badges, PROBLEMS.md, README counts, site/MCP data
python3 scripts/generate.py check     # manifest validation (CI)
python3 scripts/validate_specs.py     # every _SOLN passes its own grader
python3 python/test_grader.py         # grader regression cases
```

Gotcha: running `python3 -c "import torch"` with the repo root as cwd imports the repo's `torch/`
folder. Run scripts from elsewhere or use `python3 -P`.
