// Shapes of the data scripts/generate.py writes into *.generated.ts. Building
// the server type-checks the generated data against these.

export type Difficulty = "basic" | "easy" | "medium" | "hard" | "expert";

export interface CompanyTag {
  name: string;
  /** "inferred" (placed from public material) or "reported" (a first-hand interview report). */
  confidence: string;
}

export interface Question {
  /** Manifest id (slug) — what torchleet.check() / hint() take. */
  id: string;
  /** Upstream ids (v1-1, v3-12, ...) that still resolve to this problem. */
  legacyIds: string[];
  title: string;
  description: string;
  difficulty: Difficulty;
  /** v1 = torch/, v2 = llm/, v3 = v3/, v4 = v4/. */
  set: string;
  number: number | null;
  category: string;
  /** Directory group, e.g. "v4/vision" — how PROBLEMS.md and the study plan group problems. */
  section: string;
  sectionTitle: string;
  tracks: string[];
  companies: CompanyTag[];
  /** Repo-relative notebook paths. Every problem has a question notebook. */
  questionPath: string;
  solutionPath: string | null;
  colabUrl: string;
  solutionColabUrl: string | null;
  /** True when torchleet.check() auto-grades it. */
  graded: boolean;
  /** Names the solver implements (what check() takes as keyword arguments). */
  entries: string[];
  llmPath: { order: number; stage: string } | null;
}

export interface StudyPlanItem {
  id: string;
  day: string;
  time: string;
  /** Marked ⭐ in the plan: drill until it can be written from blank in ~20 minutes. */
  star: boolean;
  note: string;
}

export interface StudyPlanWeek {
  week: number;
  title: string;
  items: StudyPlanItem[];
}

export interface StudyPlan {
  id: string;
  title: string;
  source: string;
  weeks: StudyPlanWeek[];
}
