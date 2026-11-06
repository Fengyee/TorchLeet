#!/usr/bin/env python3
"""TorchLeet manifest tooling. Manifests (problem.toml) are the source of truth;
questions.ts, README tables and the grader registry are generated from them.

Subcommands:
  check     validate manifests against disk and grader specs (CI gate)
  coverage  print how many problems are auto-graded
  hints     copy each grader spec's HINTS into its problem.toml
  tiers     set grading.tier from whether a grader spec exists
  badges    (re)write the Colab/grader cell at the top of question notebooks
  readme    write PROBLEMS.md and sync the counts quoted in README.md
  sources   write SOURCES.md from the [[companies]] tables
  website   write website/src/data/problems.generated.ts
  json      write problems.json (+ space/problems.json)
  mcp       write the MCP server's catalogue, hints and study-plan path
            (mcp-server/src/data/*.generated.ts); rebuild the server afterwards
  all       hints, tiers, badges, readme, sources, website, json, mcp, then check —
            run after adding a problem
  seed      historical one-time bootstrap (reads /tmp/catalog.json); not for normal use

Reads TOML with stdlib tomllib. Writes it with a small emitter here, because the
only writer is `seed` and pulling in a dependency for one-time output is not worth
it for contributors or CI.
"""
from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TREES = ("torch", "llm", "v3")
# ponytail: '.ipynb_checkpoints', NOT 'checkpoint' — v3/alignment-training/
# gradient-checkpointing/ is a real problem and a looser filter silently drops it.
SKIP = (".git", ".ipynb_checkpoints", "node_modules")

STDLIB_ISH = {"torch", "math", "time", "os", "sys", "json", "random", "typing",
              "dataclasses", "collections", "itertools", "functools", "re", "copy",
              "abc", "warnings", "pathlib", "string", "unicodedata"}


# ---------------------------------------------------------------- notebook io

def notebooks(root: Path = ROOT):
    for p in sorted(root.rglob("*.ipynb")):
        if any(s in p.parts for s in SKIP):
            continue
        yield p


def code_of(nb: Path) -> str:
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception:
        return ""
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def first_markdown(nb: Path) -> str:
    try:
        cells = json.loads(nb.read_text())["cells"]
    except Exception:
        return ""
    for c in cells:
        if c["cell_type"] == "markdown":
            return "".join(c["source"])
    return ""


# ------------------------------------------------------------- auto-detection

def detect_entries(question_src: str) -> list[str]:
    """Top-level classes/functions the solver must implement: those whose body
    still contains a TODO / stub marker in the question notebook."""
    out, lines = [], question_src.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(class|def)\s+(\w+)", line)
        if not m:
            continue
        body = []
        for nxt in lines[i + 1:]:
            if nxt and not nxt[0].isspace() and not nxt.startswith(("#", ")", "]")):
                break
            body.append(nxt)
        blob = "\n".join(body)
        if re.search(r"TODO|NotImplementedError|^\s*\.\.\.\s*$|^\s*pass\s*$", blob, re.M):
            out.append(m.group(2))
    return out


def detect_extras(src: str) -> list[str]:
    mods = set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", src, re.M))
    return sorted(m for m in mods - STDLIB_ISH if not m.startswith("_"))


def detect_device(src: str) -> str:
    if re.search(r"\btriton\b|@triton\.jit", src):
        return "cuda"
    if re.search(r"\bdist\.init_process_group|FullyShardedDataParallel|\bFSDP\b", src):
        return "cuda"
    return "cpu"


def detect_tier(question_src: str) -> str:
    """Heuristic seed only; humans confirm. Counts real verification signal."""
    n = len(re.findall(r"\bassert\b", question_src)) + \
        len(re.findall(r"allclose", question_src))
    return "A" if n >= 2 else ("B" if n == 1 else "C")


def pair_files(files: list[str]) -> tuple[str | None, str | None]:
    """Map a directory's notebooks to (question, solution) across the four
    conventions that exist in this repo."""
    f = sorted(files)
    us = [x for x in f if "_SOLN" in x]
    hy = [x for x in f if "-SOLN" in x]
    qn = [x for x in f if "-Question" in x]
    if us:
        return (next((x for x in f if x not in us), None), us[0])
    if hy:
        return (next((x for x in f if x not in hy), None), hy[0])
    if qn:  # base filename is the SOLUTION in this convention
        return (qn[0], next((x for x in f if x not in qn), None))
    if len(f) == 1:
        return (f[0], None)
    return (f[0], None)


# ------------------------------------------------------------- toml emitting

def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def emit(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(emit(x) for x in v) + "]"
    if v is None:
        return '""'
    s = str(v)
    if "\n" in s:
        return '"""\n' + s.replace("\\", "\\\\").replace('"""', '\\"""') + '\n"""'
    return '"' + esc(s) + '"'


def write_manifest(path: Path, m: dict) -> None:
    L = []
    for k in ("id", "legacy_ids", "title", "description", "difficulty", "number",
              "set", "category", "tracks"):
        if m.get(k) not in (None, [], ""):
            L.append(f"{k} = {emit(m[k])}")
    L.append("")
    L.append("[notebooks]")
    L.append(f"question = {emit(m['question'])}")
    L.append(f"solution = {emit(m['solution'])}" if m["solution"]
             else "solution = \"\"   # no separate solution file")
    L.append("")
    L.append("[grading]")
    L.append(f"tier = {emit(m['tier'])}        # A=auto-graded B=partial C=not auto-graded (SEEDED, confirm)")
    L.append(f"entries = {emit(m['entries'])}")
    L.append(f"extras = {emit(m['extras'])}")
    L.append(f"device = {emit(m['device'])}")
    if m.get("llm_path"):
        L += ["", "[llm_path]",
              f"order = {m['llm_path']['order']}",
              f"stage = {emit(m['llm_path']['stage'])}"]
    for c in m.get("companies", []):
        L += ["", "[[companies]]", f"name = {emit(c)}",
              'confidence = "inferred"   # inferred | reported']
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ commands

# Catalog rows that were marked hasNotebook:false while the notebook was in fact
# sitting on disk, unlinked. Reattaching them keeps the legacy id resolvable and
# keeps them off the "planned" list.
FALSE_BLANKS = {"torch/medium/alexnet": "v1-18", "llm/RMS-Norm": "v2-2"}


def slug_of(d: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", Path(d).name.lower()).strip("-")


def cmd_seed() -> int:
    catalog = json.loads(Path("/tmp/catalog.json").read_text())
    by_id = {q["id"]: q for q in catalog}
    by_dir: dict[str, dict] = {}
    for q in catalog:
        if q.get("questionPath"):
            by_dir[str(Path(q["questionPath"]).parent)] = q

    dirs: dict[str, list[str]] = {}
    for nb in notebooks():
        rel = nb.relative_to(ROOT)
        if rel.parts[0] not in TREES:
            continue
        dirs.setdefault(str(rel.parent), []).append(rel.name)

    made = orphans = 0
    for d, files in sorted(dirs.items()):
        entry = by_dir.get(d)
        qf, sf = pair_files(files)
        if entry:
            qf = Path(entry["questionPath"]).name
            sf = Path(entry["solutionPath"]).name if entry.get("solutionPath") else sf
            m = dict(id=slug_of(d), legacy_ids=[entry["id"]], title=entry["title"],
                     description=entry.get("description") or "",
                     difficulty=entry["difficulty"], number=entry.get("number"),
                     set=entry["set"], category=entry.get("category") or "",
                     tracks=entry.get("tracks") or [], companies=entry.get("companies") or [])
            if entry.get("llmPathOrder") is not None:
                m["llm_path"] = {"order": entry["llmPathOrder"],
                                 "stage": entry.get("llmPathStage") or ""}
        else:
            orphans += 1
            blank = by_id.get(FALSE_BLANKS.get(d, ""))
            title = (blank or {}).get("title") or \
                Path(d).name.replace("-", " ").replace("_", " ").title()
            m = dict(id=slug_of(d),
                     legacy_ids=[blank["id"]] if blank else [],
                     title=title,
                     description=(blank or {}).get("description") or "",
                     difficulty=(blank or {}).get("difficulty") or "medium",
                     number=(blank or {}).get("number"),
                     set=(blank or {}).get("set") or "v2",
                     category=(blank or {}).get("category") or "",
                     tracks=(blank or {}).get("tracks") or [],
                     companies=(blank or {}).get("companies") or [])
            print(f"  {'RELINKED false blank' if blank else 'ORPHAN (needs review)'}: {d}")

        qsrc = code_of(ROOT / d / qf) if qf else ""
        both = qsrc + "\n" + (code_of(ROOT / d / sf) if sf else "")
        m.update(question=qf, solution=sf, entries=detect_entries(qsrc),
                 extras=detect_extras(both), device=detect_device(both),
                 tier=detect_tier(qsrc))
        write_manifest(ROOT / d / "problem.toml", m)
        made += 1

    claimed = {by_dir[d]["id"] for d in dirs if d in by_dir} | \
              {v for k, v in FALSE_BLANKS.items() if k in dirs}
    planned = [q for q in catalog
               if not q.get("hasNotebook") and q["id"] not in claimed]
    Path(ROOT / "PLANNED.toml").write_text(
        "# Advertised problems with no notebook yet. Claim one via an issue.\n"
        "# Generated by scripts/generate.py seed; not counted in the problem total.\n\n"
        + "\n".join(
            f'[[planned]]\nid = {emit(q["id"])}\ntitle = {emit(q["title"])}\n'
            f'difficulty = {emit(q["difficulty"])}\n' for q in planned))

    print(f"\nseeded {made} manifests ({orphans} orphan dirs), "
          f"{len(planned)} planned entries -> PLANNED.toml")
    return 0


def _spec_path(pid: str) -> Path | None:
    if not pid:
        return None
    mod = pid.replace("-", "_")
    mod = f"_{mod}" if mod[:1].isdigit() else mod
    f = ROOT / "python" / "torchleet" / "problems" / f"{mod}.py"
    return f if f.exists() else None


def _spec_entries(pid: str) -> list[str] | None:
    """ENTRIES declared by a grader spec, if one exists. Read as text so this
    stays importable without torch installed (CI validates manifests, not torch)."""
    f = _spec_path(pid)
    if f is None:
        return None
    m = re.search(r"^ENTRIES\s*=\s*\[(.*?)\]", f.read_text(), re.M | re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def cmd_check() -> int:
    errs, seen, n = [], {}, 0
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        n += 1
        rel = mf.relative_to(ROOT)
        try:
            m = tomllib.loads(mf.read_text())
        except Exception as e:
            errs.append(f"{rel}: unparseable ({e})")
            continue
        for k in ("id", "title", "difficulty", "notebooks", "grading"):
            if k not in m:
                errs.append(f"{rel}: missing required key '{k}'")
        pid = m.get("id")
        if pid in seen:
            errs.append(f"{rel}: duplicate id '{pid}' (also {seen[pid]})")
        seen[pid] = rel
        q = (m.get("notebooks") or {}).get("question")
        if not q:
            errs.append(f"{rel}: notebooks.question is empty")
        elif not (mf.parent / q).exists():
            errs.append(f"{rel}: notebooks.question '{q}' does not exist")
        s = (m.get("notebooks") or {}).get("solution")
        if s and not (mf.parent / s).exists():
            errs.append(f"{rel}: notebooks.solution '{s}' does not exist")
        grading = m.get("grading") or {}
        if grading.get("tier") not in ("A", "B", "C"):
            errs.append(f"{rel}: grading.tier must be A, B or C")
        # A problem with a grader spec must agree with it about the entry points,
        # or solvers get told to implement one thing while being graded on another.
        want_h = _spec_hints(pid)
        if want_h and (m.get("hints") or []) != want_h:
            errs.append(f"{rel}: hints differ from the grader spec — "
                        f"run `generate.py hints`")
        spec = _spec_entries(pid)
        if spec is not None and sorted(spec) != sorted(grading.get("entries") or []):
            errs.append(f"{rel}: grading.entries {grading.get('entries')} != "
                        f"torchleet.problems ENTRIES {spec}")

    for e in errs:
        print(f"FAIL {e}")
    print(f"\n{n} manifests checked, {len(errs)} problems")
    return 1 if errs else 0


REPO = "Fengyee/TorchLeet"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"
MARK = "<!-- torchleet:colab -->"
# The grader is not on PyPI; install it from this repo (editable locally, or
# straight from GitHub on Colab).
INSTALL_LOCAL = "pip install -e python/"
INSTALL_COLAB = f'!pip install -q "git+https://github.com/{REPO}#subdirectory=python"'


def load_all() -> list[tuple[dict, Path]]:
    out = []
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        out.append((tomllib.loads(mf.read_text()), mf.parent))
    return sorted(out, key=lambda t: (t[0].get("set", ""), t[0].get("number") or 0))


def colab_url(m: dict, d: Path) -> str:
    return f"{COLAB}/{(d.relative_to(ROOT) / m['notebooks']['question']).as_posix()}"


def cmd_badges() -> int:
    """Put an 'Open in Colab' cell at the top of every question notebook.
    Idempotent: the marker cell is replaced, never stacked."""
    n = 0
    for m, d in load_all():
        nb_path = d / m["notebooks"]["question"]
        nb = json.loads(nb_path.read_text())
        if _spec_entries(m.get("id")) is not None:
            how = (f"Check your work: `{INSTALL_LOCAL}` once from the repo root "
                   f"(on Colab: `{INSTALL_COLAB}`), then "
                   f"`from torchleet import check; check(\"{m['id']}\", ...)`\n")
        else:
            how = ("This problem is not auto-graded: run the notebook's own checks, "
                   "then compare with the solution notebook.\n")
        cell = {
            "cell_type": "markdown", "metadata": {},
            "source": [f"{MARK}\n",
                       f"[![Open In Colab]({BADGE})]({colab_url(m, d)})\n",
                       "\n", how],
        }
        # nbformat >= 4.5 requires a cell id; older notebooks must not have one.
        if (nb.get("nbformat"), nb.get("nbformat_minor", 0)) >= (4, 5):
            cell["id"] = "torchleet-colab"
        cells = [c for c in nb["cells"]
                 if not (c["cell_type"] == "markdown" and MARK in "".join(c["source"]))]
        nb["cells"] = [cell] + cells
        nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        n += 1
    print(f"badged {n} question notebooks")
    return 0


# PROBLEMS.md is grouped by where a problem lives on disk, which is also how the
# study plan (study/README.md) refers to problems.
SECTIONS = {
    "torch/basic": "PyTorch basics",
    "torch/easy": "PyTorch — easy",
    "torch/medium": "PyTorch — medium",
    "torch/hard": "PyTorch — hard",
    "llm": "LLM building blocks",
    "v3/classical-ml": "Classical ML",
    "v3/data-loading": "Data loading",
    "v3/modern-architectures": "Modern architectures",
    "v3/alignment-training": "Alignment & training",
    "v3/llm-decoding": "LLM decoding",
    "v3/llm-inference": "LLM inference",
    "v3/gpu-systems": "GPU systems & kernels",
    "v4/ml-fundamentals": "ML fundamentals & evaluation",
    "v4/dl-fundamentals": "Deep-learning fundamentals",
    "v4/vision": "Vision",
    "v4/generative": "Generative models",
    "v4/multimodal": "Multimodal",
    "v4/systems": "ML systems",
}
DIFF_RANK = {"basic": 0, "easy": 1, "medium": 2, "hard": 3, "expert": 4}


def section_of(d: Path) -> str:
    parts = d.relative_to(ROOT).parts
    return "/".join(parts[:1] if parts[0] == "llm" else parts[:2])


def cmd_readme() -> int:
    """Write the generated index to PROBLEMS.md and sync the counts README.md
    quotes between <!--total-->N<!--/total--> / <!--graded-->N<!--/graded-->."""
    groups: dict[str, list[tuple[dict, Path]]] = {}
    for m, d in load_all():
        groups.setdefault(section_of(d), []).append((m, d))

    head = ("| Problem | id (for `check`) | Difficulty | Graded | Notebooks |\n"
            "|---|---|---|---|---|")
    order = [k for k in SECTIONS if k in groups] + sorted(k for k in groups if k not in SECTIONS)
    body = []
    for key in order:
        items = sorted(groups[key], key=lambda t: (DIFF_RANK.get(t[0]["difficulty"], 9),
                                                  t[0].get("number") or 0, t[0]["title"]))
        body += [f"\n### {SECTIONS.get(key, key)} — `{key}/` ({len(items)})\n", head]
        for m, d in items:
            graded = "✅" if _spec_entries(m.get("id")) is not None else ""
            q = (d.relative_to(ROOT) / m["notebooks"]["question"]).as_posix()
            s = m["notebooks"].get("solution")
            links = f"[Q]({q})" + (f" · [S]({(d.relative_to(ROOT) / s).as_posix()})" if s else "")
            links += f" · [Colab]({colab_url(m, d)})"
            body.append(f"| {m['title']} | `{m['id']}` | {m['difficulty']} | {graded} | {links} |")

    total = len(load_all())
    graded = sum(1 for m, _ in load_all() if _spec_entries(m.get("id")) is not None)
    (ROOT / "PROBLEMS.md").write_text(
        f"# All {total} problems\n\n"
        "Generated from the per-problem `problem.toml` manifests — do not edit by hand.\n"
        f"Run `python3 scripts/generate.py readme` to refresh.\n\n"
        f"**{graded} of {total} are auto-graded.** A ✅ in the Graded column means\n"
        f"`check()` will verify your solution (one-time setup: `{INSTALL_LOCAL}`).\n"
        "Within each section problems are ordered by difficulty.\n"
        + "\n".join(body) + "\n")

    # Keep the numbers the README quotes in step with reality.
    readme = ROOT / "README.md"
    txt = readme.read_text()
    txt = re.sub(r"(<!--total-->)\d+(<!--/total-->)", rf"\g<1>{total}\g<2>", txt)
    txt = re.sub(r"(<!--graded-->)\d+(<!--/graded-->)", rf"\g<1>{graded}\g<2>", txt)
    readme.write_text(txt)
    print(f"PROBLEMS.md written ({total} rows in {len(order)} sections); "
          f"README counts synced to {total} ({graded} graded)")
    return 0


def cmd_sources() -> int:
    """Generate SOURCES.md from company attestations."""
    lines = [
        "# Where these problems come from", "",
        "Problems are collected from first-person accounts by candidates who sat the",
        "interviews. Identities and verbatim prompts are deliberately withheld: the",
        "people who shared them are often under NDA, and naming them would be a",
        "problem for them, not for us.",
        "",
        "That means most tags below are marked **inferred** rather than **reported**.",
        "Inferred means we placed the problem with a company from public job posts,",
        "published engineering material, and what the role plainly requires, not from",
        "a dated first-hand report we can point to. We would rather label that",
        "honestly than invent a precision we do not have.",
        "",
        "If you were asked one of these, "
        f"[tell us](https://github.com/{REPO}/issues/new?template=interview-report.yml)"
        " and it becomes a **reported** tag with a real date.",
        "",
    ]
    by_company: dict[str, list[tuple[str, str]]] = {}
    for m, _ in load_all():
        for c in m.get("companies", []):
            note = c.get("confidence", "inferred")
            if c.get("round"):
                note += f", {c['round']}"
            if c.get("period"):
                note += f", {c['period']}"
            by_company.setdefault(c["name"], []).append((m["title"], note))

    counts = {"reported": 0, "inferred": 0}
    for lst in by_company.values():
        for _, note in lst:
            counts["reported" if note.startswith("reported") else "inferred"] += 1
    lines += [f"**{counts['reported']} reported, {counts['inferred']} inferred "
              f"across {len(by_company)} companies.**", ""]
    for comp in sorted(by_company):
        lines.append(f"### {comp}")
        lines += [f"- {t} — _{n}_" for t, n in sorted(by_company[comp])]
        lines.append("")
    (ROOT / "SOURCES.md").write_text("\n".join(lines))
    print(f"SOURCES.md written ({counts['reported']} reported, "
          f"{counts['inferred']} inferred)")
    return 0


def cmd_website() -> int:
    """Emit the generated problem data the site's SEO pages read.

    Written as a separate module rather than replacing the hand-maintained
    questions.ts, which carries curated helpers the existing UI depends on. This
    file is the manifest-derived source for the per-problem pages and sitemap.
    """
    items = []
    for m, d in load_all():
        rel = d.relative_to(ROOT).as_posix()
        q = f"{rel}/{m['notebooks']['question']}"
        s = m["notebooks"].get("solution")
        items.append({
            "slug": m["id"],
            "legacyIds": m.get("legacy_ids", []),
            "title": m["title"],
            "description": m.get("description", ""),
            "difficulty": m["difficulty"],
            "category": m.get("category", ""),
            "tracks": m.get("tracks", []),
            "companies": [c["name"] for c in m.get("companies", [])],
            "companyConfidence": {c["name"]: c.get("confidence", "inferred")
                                  for c in m.get("companies", [])},
            "questionPath": q,
            "solutionPath": f"{rel}/{s}" if s else None,
            "colabUrl": colab_url(m, d),
            "graded": _spec_entries(m.get("id")) is not None,
            "entries": m.get("grading", {}).get("entries", []),
        })

    out = ROOT / "website" / "src" / "data" / "problems.generated.ts"
    body = ",\n".join("  " + json.dumps(i) for i in items)
    out.write_text(
        "// GENERATED by scripts/generate.py website — do not edit.\n"
        "// Source of truth: the per-problem problem.toml manifests.\n\n"
        "export interface GeneratedProblem {\n"
        "  slug: string;\n  legacyIds: string[];\n  title: string;\n"
        "  description: string;\n  difficulty: string;\n  category: string;\n"
        "  tracks: string[];\n  companies: string[];\n"
        "  companyConfidence: Record<string, string>;\n"
        "  questionPath: string;\n  solutionPath: string | null;\n"
        "  colabUrl: string;\n  graded: boolean;\n  entries: string[];\n}\n\n"
        f"export const generatedProblems: GeneratedProblem[] = [\n{body},\n];\n\n"
        "export function getProblemBySlug(slug: string): GeneratedProblem | undefined {\n"
        "  return generatedProblems.find((p) => p.slug === slug);\n}\n\n"
        "/** Legacy ids (v3-12, ...) still resolve, so older links keep working. */\n"
        "export function getProblemByAnyId(id: string): GeneratedProblem | undefined {\n"
        "  return generatedProblems.find((p) => p.slug === id || p.legacyIds.includes(id));\n}\n")
    print(f"website data written: {len(items)} problems -> {out.relative_to(ROOT)}")
    return 0


def _spec_hints(pid: str) -> list[str] | None:
    """HINTS authored in a grader spec. Parsed as text so this works without torch."""
    if not pid:
        return None
    mod = pid.replace("-", "_")
    mod = f"_{mod}" if mod[:1].isdigit() else mod
    f = ROOT / "python" / "torchleet" / "problems" / f"{mod}.py"
    if not f.exists():
        return None
    # Parse rather than regex: hint strings are routinely split across lines and
    # implicitly concatenated, which a regex silently truncates.
    import ast
    try:
        tree = ast.parse(f.read_text())
    except SyntaxError:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "HINTS" for t in node.targets):
            try:
                return list(ast.literal_eval(node.value))
            except ValueError:
                return []
    return []


def cmd_hints() -> int:
    """Copy each grader spec's HINTS into its manifest.

    Hints are authored next to the checks they describe, so the spec module owns
    them; the manifest carries a generated copy so non-Python consumers (the MCP
    tutor, the site, the Space) read one source instead of inventing their own.
    `check` fails if the two drift.
    """
    n = 0
    for m, d in load_all():
        hints = _spec_hints(m.get("id"))
        if not hints:
            continue
        mf = d / "problem.toml"
        txt = re.sub(r"\n*^# GENERATED from the grader spec.*?^hints = \[.*?^\]\n",
                     "\n", mf.read_text(), flags=re.M | re.S)
        body = "\n".join(f'  "{esc(h)}",' for h in hints)
        block = ("\n# GENERATED from the grader spec — edit "
                 "python/torchleet/problems/*.py, then run: generate.py hints\n"
                 f"hints = [\n{body}\n]\n")
        # Must land BEFORE the first table header: in TOML every key after a
        # `[table]` line belongs to that table, so appending at the end would
        # bury `hints` inside the last [[companies]] entry.
        head = re.search(r"^\[", txt, re.M)
        txt = (txt[:head.start()].rstrip() + "\n" + block + "\n" + txt[head.start():]
               if head else txt.rstrip() + "\n" + block)
        mf.write_text(txt)
        n += 1
    print(f"hints synced into {n} manifests")
    return 0


def cmd_tiers() -> int:
    """Derive grading.tier from whether a grader spec exists: A = auto-graded,
    C = not graded. The seeded heuristic tiers had drifted from reality."""
    n = 0
    for m, d in load_all():
        tier = "A" if _spec_entries(m.get("id")) is not None else "C"
        mf = d / "problem.toml"
        txt = mf.read_text()
        new = re.sub(r'^tier = "[ABC]".*$',
                     f'tier = "{tier}"        # A = auto-graded by check(), C = not graded',
                     txt, count=1, flags=re.M)
        if new != txt:
            mf.write_text(new)
            n += 1
    print(f"tiers synced ({n} manifests changed)")
    return 0


def cmd_json() -> int:
    """Flat problems.json — consumed by the HF Space and available to any other
    client that should not hand-maintain a second copy of the catalogue."""
    items = []
    for m, d in load_all():
        rel = d.relative_to(ROOT).as_posix()
        items.append({
            "id": m["id"],
            "legacy_ids": m.get("legacy_ids", []),
            "title": m["title"],
            "description": m.get("description", ""),
            "difficulty": m["difficulty"],
            "category": m.get("category", ""),
            "tracks": m.get("tracks", []),
            "companies": [c["name"] for c in m.get("companies", [])],
            "question": f"{rel}/{m['notebooks']['question']}",
            "solution": (f"{rel}/{m['notebooks']['solution']}"
                         if m["notebooks"].get("solution") else None),
            "colab": colab_url(m, d),
            "frequency": m.get("frequency", ""),
            "company_confidence": {c["name"]: c.get("confidence", "inferred")
                                   for c in m.get("companies", [])},
            "llm_path_order": (m.get("llm_path") or {}).get("order"),
            "graded": _spec_entries(m.get("id")) is not None,
            "entries": m.get("grading", {}).get("entries", []),
            "hints": _spec_hints(m.get("id")),
        })
    blob = json.dumps({"count": len(items), "problems": items}, indent=1) + "\n"
    (ROOT / "problems.json").write_text(blob)
    # Write the Space's copy from the same call: a hand-copied snapshot is how it
    # ended up advertising 30 graded when the manifests said 56.
    space = ROOT / "space"
    if space.is_dir():
        (space / "problems.json").write_text(blob)
    graded = sum(1 for i in items if i["graded"])
    hinted = sum(1 for i in items if i["hints"])
    print(f"problems.json written ({len(items)} problems, {graded} graded, "
          f"{hinted} with hints) -> root + space/")
    return 0


# Sets are the directory trees; a manifest's own `set` wins where it has one (the
# data-loading problems predate the field, so they get it from the tree).
TREE_SET = {"torch": "v1", "llm": "v2", "v3": "v3", "v4": "v4"}
MCP_DATA = ROOT / "mcp-server" / "src" / "data"
STUDY_PLAN = ROOT / "study" / "README.md"
STUDY_PLAN_ID = "cv-mm-gen-4wk"


def _study_plan(dir_ids: dict[str, str]) -> dict | None:
    """The owner's plan (study/README.md) as problem ids in plan order.

    Read from the week tables: a `## N. 第 k 周：<title>` heading opens a week, a
    row whose first cell is `D<n>` opens a day, and every backticked repo path in
    the row's item cell that is a problem directory becomes an item. A bare name
    after such a path in the same cell (`a/b/x` 或 `y`) is that path's sibling.
    Anything else in the plan (breadth reading, mocks) is not a problem and is
    skipped."""
    if not STUDY_PLAN.exists():
        return None
    text = STUDY_PLAN.read_text(encoding="utf-8")
    h1 = re.search(r"^#\s+(.+)$", text, re.M)
    weeks: list[dict] = []
    week, day = None, ""
    for line in text.splitlines():
        h = re.match(r"^##\s+(?:\d+\.\s*)?第\s*(\d+)\s*周\s*[：:]\s*(.*)$", line)
        if h:
            week = {"week": int(h.group(1)), "title": h.group(2).strip(), "items": []}
            weeks.append(week)
            day = ""
            continue
        if line.startswith("## "):
            week = None
            continue
        if week is None or not line.startswith("|"):
            continue
        # Split on unescaped pipes: cells may contain a literal `\|`.
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if re.fullmatch(r"D\d+", cells[0]):
            day = cells[0]
        if len(cells) < 2:
            continue
        prev = None
        for tok in re.findall(r"`([^`]+)`", cells[1]):
            path = tok if "/" in tok or prev is None else f"{prev.rsplit('/', 1)[0]}/{tok}"
            if path in dir_ids:
                week["items"].append({
                    "id": dir_ids[path], "day": day,
                    "time": cells[2] if len(cells) > 2 else "",
                    "star": "⭐" in cells[1],
                    "note": cells[3] if len(cells) > 3 else "",
                })
                prev = path
            elif tok.split("/")[0] in TREE_SET and "/" in tok:
                print(f"  warning: study plan names `{tok}`, which is not a problem directory")
    weeks = [w for w in weeks if w["items"]]
    if not weeks:
        print(f"  warning: no problems found in {STUDY_PLAN.relative_to(ROOT)}; "
              "the MCP server will have no study-plan path")
        return None
    return {"id": STUDY_PLAN_ID,
            "title": h1.group(1).strip() if h1 else "Study plan",
            "source": STUDY_PLAN.relative_to(ROOT).as_posix(),
            "weeks": weeks}


def cmd_mcp() -> int:
    """Write the MCP server's data from the manifests: the catalogue (plus the
    study-plan path) and the authored hints. Hints go in their own module so the
    catalogue resource the server exposes never carries every hint level at once.
    The server compiles these in, so run `npm run build` in mcp-server/ after."""
    order = {k: i for i, k in enumerate(SECTIONS)}
    rows = sorted(load_all(), key=lambda t: (
        order.get(section_of(t[1]), len(order)), section_of(t[1]),
        DIFF_RANK.get(t[0]["difficulty"], 9), t[0].get("number") or 0, t[0]["title"]))

    items, hints, dir_ids = [], {}, {}
    for m, d in rows:
        rel = d.relative_to(ROOT).as_posix()
        sec = section_of(d)
        s = m["notebooks"].get("solution")
        dir_ids[rel] = m["id"]
        items.append({
            "id": m["id"],
            "legacyIds": m.get("legacy_ids", []),
            "title": m["title"],
            "description": m.get("description", ""),
            "difficulty": m["difficulty"],
            "set": m.get("set") or TREE_SET.get(rel.split("/")[0], rel.split("/")[0]),
            "number": m.get("number"),
            "category": m.get("category", ""),
            "section": sec,
            "sectionTitle": SECTIONS.get(sec, sec),
            "tracks": m.get("tracks", []),
            "companies": [{"name": c["name"], "confidence": c.get("confidence", "inferred")}
                          for c in m.get("companies", [])],
            "questionPath": f"{rel}/{m['notebooks']['question']}",
            "solutionPath": f"{rel}/{s}" if s else None,
            "colabUrl": colab_url(m, d),
            "solutionColabUrl": f"{COLAB}/{rel}/{s}" if s else None,
            "graded": _spec_entries(m.get("id")) is not None,
            "entries": m.get("grading", {}).get("entries", []),
            "llmPath": ({"order": m["llm_path"]["order"], "stage": m["llm_path"].get("stage", "")}
                        if "order" in (m.get("llm_path") or {}) else None),
        })
        if m.get("hints"):
            hints[m["id"]] = m["hints"]

    plan = _study_plan(dir_ids)
    head = ("// GENERATED by scripts/generate.py mcp — do not edit.\n"
            "// Source of truth: the per-problem problem.toml manifests"
            " (and study/README.md for the study-plan path).\n"
            "// Regenerate with `python3 scripts/generate.py mcp`, then `npm run build`.\n\n")
    body = ",\n".join("  " + json.dumps(i, ensure_ascii=False) for i in items)
    (MCP_DATA / "problems.generated.ts").write_text(
        head + 'import type { Question, StudyPlan } from "./types.js";\n\n'
        f"export const problems: Question[] = [\n{body},\n];\n\n"
        "export const studyPlan: StudyPlan | null = "
        + (json.dumps(plan, ensure_ascii=False, indent=1) if plan else "null") + ";\n",
        encoding="utf-8")
    (MCP_DATA / "hints.generated.ts").write_text(
        head + "// Hints by problem id (slug). Callers resolve legacy ids to the slug first.\n"
        "export const authoredHints: Record<string, string[]> = "
        + json.dumps(hints, ensure_ascii=False, indent=1) + ";\n\n"
        "export function getAuthoredHints(id: string): string[] | undefined {\n"
        "  return Object.prototype.hasOwnProperty.call(authoredHints, id) ? authoredHints[id] : undefined;\n"
        "}\n", encoding="utf-8")
    n_plan = sum(len(w["items"]) for w in plan["weeks"]) if plan else 0
    print(f"mcp data written: {len(items)} problems, {len(hints)} with hints, "
          f"study plan {n_plan} items -> {MCP_DATA.relative_to(ROOT)}/ "
          "(rebuild: cd mcp-server && npm run build)")
    return 0


def cmd_coverage() -> int:
    """The published grading-coverage number. Derived, never typed by hand."""
    from collections import Counter
    total, graded = 0, 0
    per_tree: dict[str, list[int]] = {}
    tiers: Counter = Counter()
    for mf in sorted(ROOT.rglob("problem.toml")):
        if any(s in mf.parts for s in SKIP):
            continue
        m = tomllib.loads(mf.read_text())
        total += 1
        tree = mf.relative_to(ROOT).parts[0]
        per_tree.setdefault(tree, [0, 0])
        per_tree[tree][0] += 1
        tiers[m.get("grading", {}).get("tier")] += 1
        if _spec_entries(m.get("id")) is not None:
            graded += 1
            per_tree[tree][1] += 1

    print(f"\n  {graded} of {total} problems are auto-graded "
          f"({graded / total * 100:.0f}%)\n")
    for tree, (n, g) in sorted(per_tree.items()):
        print(f"    {tree:7} {g:3}/{n:<3} graded")
    print(f"\n  tiers: " + "  ".join(f"{t}={tiers[t]}" for t in "ABC"))
    print(f"  planned (no notebook yet): "
          f"{(ROOT / 'PLANNED.toml').read_text().count('[[planned]]')}\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "seed":
        sys.exit(cmd_seed())
    if cmd == "check":
        sys.exit(cmd_check())
    if cmd == "coverage":
        sys.exit(cmd_coverage())
    if cmd == "badges":
        sys.exit(cmd_badges())
    if cmd == "readme":
        sys.exit(cmd_readme())
    if cmd == "sources":
        sys.exit(cmd_sources())
    if cmd == "website":
        sys.exit(cmd_website())
    if cmd == "json":
        sys.exit(cmd_json())
    if cmd == "hints":
        sys.exit(cmd_hints())
    if cmd == "tiers":
        sys.exit(cmd_tiers())
    if cmd == "mcp":
        sys.exit(cmd_mcp())
    if cmd == "all":
        for step in (cmd_hints, cmd_tiers, cmd_badges, cmd_readme, cmd_sources,
                     cmd_website, cmd_json, cmd_mcp):
            if step():
                sys.exit(1)
        sys.exit(cmd_check())
    print(__doc__)
    sys.exit(2)
