import { z } from "zod";
import {
  getQuestionById,
  getStudyPlanSlots,
  studyPlan,
  formatCompanies,
  COMPANY_TAG_NOTE,
  localPath,
  checkUsage,
  GRADER_SETUP,
  driftNotice,
} from "../data/questions.js";
import { getAuthoredHints } from "../data/hints.generated.js";

export const getQuestionSchema = {
  id: z
    .string()
    .describe("Problem id (e.g. 'kv-cache', 'iou-nms'), a legacy id (e.g. 'v3-12') or its directory (e.g. 'v4/vision/iou-nms')"),
};

export function notFound(id: string) {
  return {
    content: [
      {
        type: "text" as const,
        text: `${driftNotice()}Question "${id}" not found. Use list_questions to see available ids.`,
      },
    ],
  };
}

export async function getQuestion(params: { id: string }) {
  const q = getQuestionById(params.id);
  if (!q) return notFound(params.id);

  const parts: string[] = [
    `# ${q.title}`,
    `**ID**: \`${q.id}\`${q.legacyIds.length ? ` (legacy: ${q.legacyIds.join(", ")})` : ""}`,
    `**Set**: ${q.set} | **Section**: ${q.sectionTitle} (\`${q.section}/\`) | **Difficulty**: ${q.difficulty}`,
  ];
  if (q.category) parts.push(`**Category**: ${q.category}`);
  if (q.tracks.length) parts.push(`**Tracks**: ${q.tracks.join(", ")}`);
  if (q.llmPath) parts.push(`**LLM path**: step ${q.llmPath.order} (${q.llmPath.stage})`);
  for (const s of getStudyPlanSlots(q.id)) {
    parts.push(`**Study plan** (${studyPlan!.id}): week ${s.week}, ${s.day}${s.time ? `, ${s.time}` : ""}${s.star ? " ⭐" : ""}`);
  }
  if (q.companies.length) {
    parts.push(`**Company tags**: ${formatCompanies(q)}`);
    parts.push(`_${COMPANY_TAG_NOTE}_`);
  }

  if (q.description) parts.push(`\n## Description\n${q.description}`);

  parts.push("\n## Notebooks");
  parts.push(`- Question: ${localPath(q.questionPath)}`);
  parts.push(`- Colab: ${q.colabUrl}`);
  if (q.solutionPath) {
    parts.push(`- Solution (open only after a real attempt): ${localPath(q.solutionPath)}`);
  }

  parts.push("\n## Grading");
  if (q.graded) {
    parts.push(`Auto-graded. Implement ${q.entries.map((e) => `\`${e}\``).join(", ")}, then run:`);
    parts.push("```python\n" + checkUsage(q) + "\n```");
    parts.push(`One-time setup, from the repo root: \`${GRADER_SETUP}\``);
  } else {
    parts.push("Not auto-graded: run the notebook's own checks, then compare with the solution notebook.");
  }

  const hints = getAuthoredHints(q.id);
  parts.push(
    hints?.length
      ? `\n${hints.length} progressive hints available via get_hint (levels 1–${hints.length}); same as \`torchleet.hint("${q.id}", n)\`.`
      : "\nNo authored hints; get_hint derives them from the notebook.",
  );

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
