// The question catalogue and the queries the tools run on it. The data itself
// is generated from the problem.toml manifests (scripts/generate.py mcp) — this
// module only indexes it. Nothing here should restate a fact a manifest owns.

import { readdirSync } from "node:fs";
import { join, posix } from "node:path";
import { getRepoRoot } from "../lib/repo-root.js";
import { problems, studyPlan } from "./problems.generated.js";
import type { CompanyTag, Difficulty, Question, StudyPlan } from "./types.js";

export type { CompanyTag, Difficulty, Question, StudyPlan };
export { studyPlan };

export const questions: Question[] = problems;

export const DIFFICULTIES: Difficulty[] = ["basic", "easy", "medium", "hard", "expert"];
export const DIFFICULTY_ORDER: Record<string, number> = Object.fromEntries(
  DIFFICULTIES.map((d, i) => [d, i]),
);

const uniqSorted = (xs: string[]) => Array.from(new Set(xs)).sort();

/** Values present in the data, so filters follow the manifests (a new set or category needs no code change). */
export const problemSets = uniqSorted(questions.map((q) => q.set));
export const categories = uniqSorted(questions.map((q) => q.category).filter(Boolean));
export const trackNames = uniqSorted(questions.flatMap((q) => q.tracks));

export const SET_LABELS: Record<string, string> = {
  v1: "torch/ — PyTorch basics to hard",
  v2: "llm/ — LLM building blocks",
  v3: "v3/ — advanced: classical ML, data loading, architectures, alignment, decoding, inference, GPU systems",
  v4: "v4/ — fundamentals, vision, generative, multimodal, ML systems",
};

// ---------------------------------------------------------------------------
// Id resolution: slug, legacy id (v3-12), or problem directory (v4/vision/iou-nms)
// ---------------------------------------------------------------------------

const normalize = (s: string) => s.trim().replace(/\/+$/, "").toLowerCase();
const byAlias = new Map<string, Question>();
// Slugs first so a slug can never be shadowed by another problem's alias.
for (const pick of [
  (q: Question) => [q.id],
  (q: Question) => q.legacyIds,
  (q: Question) => [posix.dirname(q.questionPath)],
]) {
  for (const q of questions) {
    for (const alias of pick(q)) {
      const k = normalize(alias);
      if (!byAlias.has(k)) byAlias.set(k, q);
    }
  }
}

export function getQuestionById(id: string): Question | undefined {
  return byAlias.get(normalize(id));
}

export function describeId(q: Question): string {
  return q.legacyIds.length ? `${q.id} (${q.legacyIds.join(", ")})` : q.id;
}

// ---------------------------------------------------------------------------
// Company tags. Every tag in the manifests today is `inferred`: placed from
// public job posts, engineering material and what the role requires, not from
// a first-hand interview report. Output must never present them as reports.
// ---------------------------------------------------------------------------

const anyReported = questions.some((q) => q.companies.some((c) => c.confidence !== "inferred"));

export const COMPANY_TAG_NOTE =
  (anyReported ? "Company tags are inferred unless marked otherwise. Inferred tags were" : "All company tags are inferred:") +
  " placed from public job posts, published engineering material and what the role requires, " +
  "not from first-hand interview reports (see SOURCES.md). Read them as topical relevance, not as evidence that a company asks the problem.";

/** Tag names, with any non-inferred confidence spelled out: ["Meta", "Google (reported)"]. */
export function companyNames(q: Question): string[] {
  return q.companies.map((c) => (c.confidence === "inferred" ? c.name : `${c.name} (${c.confidence})`));
}

/** "Anthropic, Meta (inferred)" — or with a separate "(reported)" group once any tag is reported. */
export function formatCompanies(q: Question): string {
  const groups = new Map<string, string[]>();
  for (const c of q.companies) {
    const conf = c.confidence || "inferred";
    if (!groups.has(conf)) groups.set(conf, []);
    groups.get(conf)!.push(c.name);
  }
  return Array.from(groups, ([conf, names]) => `${names.join(", ")} (${conf})`).join("; ");
}

export function getAllCompanies(): { name: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const q of questions) {
    for (const c of q.companies) counts.set(c.name, (counts.get(c.name) ?? 0) + 1);
  }
  return Array.from(counts, ([name, count]) => ({ name, count })).sort(
    (a, b) => b.count - a.count || a.name.localeCompare(b.name),
  );
}

export function getQuestionsByCompany(company: string): Question[] {
  const lower = company.trim().toLowerCase();
  return questions.filter((q) => q.companies.some((c) => c.name.toLowerCase() === lower));
}

// ---------------------------------------------------------------------------
// Grouping
// ---------------------------------------------------------------------------

export function byDifficulty(a: Question, b: Question): number {
  return (DIFFICULTY_ORDER[a.difficulty] ?? 9) - (DIFFICULTY_ORDER[b.difficulty] ?? 9);
}

/** Group by directory section, keeping the catalogue's order (sections as in PROBLEMS.md, easy → hard within). */
export function groupBySection(qs: Question[]): { section: string; title: string; questions: Question[] }[] {
  const groups = new Map<string, { section: string; title: string; questions: Question[] }>();
  for (const q of qs) {
    if (!groups.has(q.section)) groups.set(q.section, { section: q.section, title: q.sectionTitle, questions: [] });
    groups.get(q.section)!.questions.push(q);
  }
  return Array.from(groups.values());
}

export function getQuestionsByTrack(track: string): Question[] {
  return questions.filter((q) => q.tracks.includes(track));
}

const STAGE_TITLES: Record<string, string> = {
  foundations: "Foundations (tokenization, position, attention)",
  "core-transformer": "Core transformer building blocks",
  "full-model": "Full small language model",
  alignment: "Alignment & efficient fine-tuning",
  "decoding-inference": "Decoding & efficient inference",
  systems: "Systems & scaling",
};

/** The LLM path from the manifests' [llm_path] tables, stages ordered by their first step. */
export function getLLMPathStages(): { stage: string; title: string; questions: Question[] }[] {
  const ordered = questions
    .filter((q) => q.llmPath)
    .sort((a, b) => a.llmPath!.order - b.llmPath!.order);
  const stages = new Map<string, Question[]>();
  for (const q of ordered) {
    const st = q.llmPath!.stage || "other";
    if (!stages.has(st)) stages.set(st, []);
    stages.get(st)!.push(q);
  }
  const out = Array.from(stages, ([stage, qs], i) => ({
    stage,
    title: `${i + 1}. ${STAGE_TITLES[stage] ?? stage}`,
    questions: qs,
  }));
  // On the llm-path track but without a position in the sequence.
  const loose = questions.filter((q) => q.tracks.includes("llm-path") && !q.llmPath);
  if (loose.length) out.push({ stage: "unordered", title: "Also on the LLM path (no fixed position)", questions: loose });
  return out;
}

/** Where a problem sits in the study plan, if it is in it. */
export function getStudyPlanSlots(id: string): { week: number; day: string; time: string; star: boolean }[] {
  if (!studyPlan) return [];
  const out = [];
  for (const w of studyPlan.weeks) {
    for (const item of w.items) {
      if (item.id === id) out.push({ week: w.week, day: item.day, time: item.time, star: item.star });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Paths and grading instructions
// ---------------------------------------------------------------------------

/** Absolute path when the repo is available locally, else the repo-relative one. */
export function localPath(relative: string): string {
  const root = getRepoRoot();
  return root ? join(root, relative) : relative;
}

export function checkUsage(q: Question): string {
  const args = q.entries.map((e) => `${e}=${e}`).join(", ");
  return `from torchleet import check; check("${q.id}"${args ? ", " + args : ""})`;
}

export const GRADER_SETUP = "pip install -e python/";

// ---------------------------------------------------------------------------
// Drift: the catalogue is compiled in, so a problem added after the last
// `generate.py mcp` + `npm run build` is invisible until both are re-run. Say so.
// ---------------------------------------------------------------------------

const SKIP_DIRS = new Set([".git", ".ipynb_checkpoints", "node_modules"]);
let drift: { missing: string[]; removed: string[] } | null | undefined;

function manifestDirs(root: string): string[] {
  const out: string[] = [];
  const walk = (rel: string) => {
    let entries;
    try {
      entries = readdirSync(join(root, rel), { withFileTypes: true });
    } catch {
      return;
    }
    for (const e of entries) {
      if (e.isDirectory()) {
        if (!SKIP_DIRS.has(e.name)) walk(rel ? `${rel}/${e.name}` : e.name);
      } else if (e.name === "problem.toml" && rel) {
        out.push(rel);
      }
    }
  };
  walk("");
  return out;
}

/** Problem directories on disk that this build lacks (and vice versa); null when in sync or no repo is available. */
export function catalogDrift(): { missing: string[]; removed: string[] } | null {
  if (drift !== undefined) return drift;
  const root = getRepoRoot();
  if (!root) return (drift = null);
  const onDisk = new Set(manifestDirs(root));
  const known = new Set(questions.map((q) => posix.dirname(q.questionPath)));
  const missing = [...onDisk].filter((d) => !known.has(d)).sort();
  const removed = [...known].filter((d) => !onDisk.has(d)).sort();
  drift = missing.length || removed.length ? { missing, removed } : null;
  return drift;
}

export const REBUILD_HINT =
  "python3 scripts/generate.py mcp && (cd mcp-server && npm run build), then restart the MCP server";

export function driftNotice(): string {
  const d = catalogDrift();
  if (!d) return "";
  const parts = [];
  if (d.missing.length) parts.push(`${d.missing.length} problem(s) on disk are not in this build: ${d.missing.join(", ")}`);
  if (d.removed.length) parts.push(`${d.removed.length} problem(s) in this build are gone from disk: ${d.removed.join(", ")}`);
  return `> ⚠ Catalogue out of date — ${parts.join("; ")}. Run: ${REBUILD_HINT}.\n\n`;
}
