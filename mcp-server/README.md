# TorchLeet MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that turns an AI assistant into a no-spoilers PyTorch tutor for the problems in **this repository**: progressive hints, learning paths (including the study plan in `study/README.md`), grading instructions and prerequisites.

Its catalogue and hints are generated from the per-problem `problem.toml` manifests, so it serves exactly the problems and hints in your checkout.

> The `torchleet-mcp` package on npm is built from upstream and serves the upstream catalogue (no `v4/`, older hints). To get this fork's problems, run the server from this checkout as below.

## Run it locally

Needs Node 18+.

```bash
cd /ABS/PATH/TorchLeet/mcp-server
npm install          # or: npm ci
npm run build        # writes dist/; the entry point is dist/src/index.js
```

### Claude Code

```bash
claude mcp add torchleet -e TORCHLEET_ROOT=/ABS/PATH/TorchLeet -- node /ABS/PATH/TorchLeet/mcp-server/dist/src/index.js
claude mcp get torchleet     # should show the server as Connected
```

- Use absolute paths. `-e KEY=VALUE` sets the server's environment and must come before the `--`.
- The default scope is `local` (this project, on this machine). Add `-s user` to have it in every project.
- If you added the npm server earlier under the same name (`claude mcp add torchleet -- npx -y torchleet-mcp`, as `AGENTS.md` suggests), remove it first: `claude mcp remove torchleet`.

### Claude Desktop / other MCP clients

```json
{
  "mcpServers": {
    "torchleet": {
      "command": "node",
      "args": ["/ABS/PATH/TorchLeet/mcp-server/dist/src/index.js"],
      "env": { "TORCHLEET_ROOT": "/ABS/PATH/TorchLeet" }
    }
  }
}
```

(Claude Desktop's config file is `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS.)

### Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `TORCHLEET_ROOT` | No | Path to the repo. Used for absolute notebook paths, notebook-derived hints and checklists, and the out-of-date check. When unset, the server finds the repo by walking up from `dist/`, which works when it runs from this checkout. |

## Keeping it in sync

The data is compiled in. After adding or changing problems (manifests, grader hints) or editing `study/README.md`:

```bash
python3 scripts/generate.py all            # or just: python3 scripts/generate.py mcp
cd mcp-server && npm run build
```

Then restart the server (in Claude Code: `/mcp` and reconnect, or start a new session).

If the problem directories on disk no longer match the build, the server says so: a warning on stderr at startup, and a notice at the top of `list_questions`, `get_learning_path` and not-found answers, naming the directories and the commands above.

Data flow:

```
python/torchleet/problems/*.py HINTS --(generate.py hints)--> problem.toml  (source of truth)
problem.toml (all trees) + study/README.md --(generate.py mcp)-->
    src/data/problems.generated.ts   catalogue + study-plan path (typed by src/data/types.ts)
    src/data/hints.generated.ts      hints by problem id (kept out of the catalogue resource)
--(npm run build)--> dist/
```

Do not edit the `*.generated.ts` files by hand. Curated data that is not in the manifests lives in the code: the prerequisite graph (`src/tools/get-prerequisites.ts`) and the LLM-path stage titles (`src/data/questions.ts`).

## What it does

### Tools

| Tool | Description |
|------|-------------|
| `list_questions` | Browse and filter problems by set (`v1` = `torch/`, `v2` = `llm/`, `v3` = `v3/`, `v4` = `v4/`), difficulty, category, track, auto-graded, or company tag |
| `get_question` | A problem's description, notebook paths, the `check()` call that grades it, its study-plan slot, and company tags |
| `get_hint` | Progressive hints (usually 3 levels), the same ones `torchleet.hint(id, n)` prints. Problems without authored hints get hints derived from the notebook |
| `review_solution` | Review checklist for submitted code (executes nothing; points to `check()` for graded problems) |
| `get_learning_path` | `llm-path` (build an LLM from scratch, in order), `basics`, `advanced`, or `cv-mm-gen-4wk` (the 4-week plan in `study/README.md`, by week and day) |
| `get_company_prep` | Problems carrying a company tag, easiest first |
| `get_prerequisites` | What to do before a harder problem (curated graph, else easier problems from the same directory) |

Ids: every tool takes the manifest id (`kv-cache`, `iou-nms`, the same id `check()` and `hint()` take), an upstream legacy id (`v3-12`), or the problem directory (`v4/vision/iou-nms`).

**Company tags are inferred** from public job posts, published engineering material and what the role requires, not from first-hand interview reports (see `SOURCES.md`). The tools label them that way. Read them as topical relevance, not as a claim that a company asks the problem.

### Prompts

| Prompt | Description |
|--------|-------------|
| `torchleet-tutor` | Interactive study mode with Socratic questioning |
| `torchleet-interview-prep` | Mock technical interview for a company-style role |
| `torchleet-review` | Code review focused on PyTorch best practices |
| `torchleet-explain` | Deep-dive concept explanations with math and intuition |

### Resources

| Resource | Description |
|----------|-------------|
| `torchleet://questions` | Full problem catalogue as JSON (no hints) |

## Example usage

- "What's next in my study plan?" (uses `get_learning_path` with `cv-mm-gen-4wk`)
- "Give me a hint for flow-matching"
- "List the v4 vision problems"
- "What should I do before dit-block?"

Select the **torchleet-tutor** prompt for guided learning.

## Development

```bash
cd mcp-server
npm install
npm run build
TORCHLEET_ROOT=.. node dist/src/index.js     # speaks MCP over stdio
```

## License

MIT
