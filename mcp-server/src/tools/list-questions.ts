import { z } from "zod";
import {
  questions,
  problemSets,
  categories,
  trackNames,
  DIFFICULTIES,
  SET_LABELS,
  COMPANY_TAG_NOTE,
  companyNames,
  describeId,
  driftNotice,
  type Question,
} from "../data/questions.js";

const nonEmpty = (xs: string[]) => (xs.length ? xs : [""]) as [string, ...string[]];

export const listQuestionsSchema = {
  set: z
    .enum(nonEmpty(problemSets))
    .optional()
    .describe(`Filter by set: ${problemSets.map((s) => `${s} = ${SET_LABELS[s] ?? s}`).join("; ")}`),
  difficulty: z.enum(DIFFICULTIES as [string, ...string[]]).optional().describe("Filter by difficulty level"),
  company: z
    .string()
    .optional()
    .describe("Filter by company tag (e.g. 'Anthropic', 'Meta'). Tags are inferred, not interview reports."),
  category: z.string().optional().describe(`Filter by category: ${categories.join(", ")}`),
  track: z.enum(nonEmpty(trackNames)).optional().describe("Filter by learning track"),
  graded: z.boolean().optional().describe("Only problems torchleet.check() auto-grades (true) or does not (false)"),
};

export async function listQuestions(params: {
  set?: string;
  difficulty?: string;
  company?: string;
  category?: string;
  track?: string;
  graded?: boolean;
}) {
  let filtered: Question[] = [...questions];

  if (params.set) filtered = filtered.filter((q) => q.set === params.set);
  if (params.difficulty) filtered = filtered.filter((q) => q.difficulty === params.difficulty);
  if (params.company) {
    const lower = params.company.toLowerCase();
    filtered = filtered.filter((q) => q.companies.some((c) => c.name.toLowerCase().includes(lower)));
  }
  if (params.category) filtered = filtered.filter((q) => q.category === params.category);
  if (params.track) filtered = filtered.filter((q) => q.tracks.includes(params.track!));
  if (params.graded !== undefined) filtered = filtered.filter((q) => q.graded === params.graded);

  const lines = filtered.map(
    (q) =>
      `${describeId(q)} | ${q.title} | ${q.difficulty} | ${q.section} | ${q.graded ? "yes" : "—"} | ${
        companyNames(q).join(", ") || "—"
      }`,
  );

  const header =
    driftNotice() +
    `Found ${filtered.length} of ${questions.length} problem(s). Use the id (first column) with get_question, get_hint and torchleet.check().\n\n` +
    "ID | Title | Difficulty | Directory | Auto-graded | Company tags (inferred unless marked)\n" +
    "---|-------|------------|-----------|-------------|-------------------------------------";

  const footer = filtered.some((q) => q.companies.length) ? `\n\n${COMPANY_TAG_NOTE}` : "";

  return {
    content: [{ type: "text" as const, text: `${header}\n${lines.join("\n")}${footer}` }],
  };
}
