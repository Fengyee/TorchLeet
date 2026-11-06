import { z } from "zod";
import {
  getLLMPathStages,
  getQuestionsByTrack,
  getQuestionById,
  groupBySection,
  studyPlan,
  driftNotice,
} from "../data/questions.js";

const TRACKS = ["llm-path", "basics", "advanced", ...(studyPlan ? [studyPlan.id] : [])] as [string, ...string[]];

export const getLearningPathSchema = {
  track: z
    .enum(TRACKS)
    .default("llm-path")
    .describe(
      "Which path: llm-path (build an LLM from scratch, in order), basics, advanced" +
        (studyPlan ? `, or ${studyPlan.id} (the owner's 4-week CV / multimodal / generative study plan from ${studyPlan.source})` : ""),
    ),
};

export async function getLearningPath(params: { track: string }) {
  const track = params.track ?? "llm-path";
  const notice = driftNotice();

  if (studyPlan && track === studyPlan.id) {
    const n = studyPlan.weeks.reduce((acc, w) => acc + w.items.length, 0);
    const parts = [
      `# ${studyPlan.title}\n`,
      `From \`${studyPlan.source}\` — ${n} problems over ${studyPlan.weeks.length} weeks, in plan order. ` +
        "Times include checking the answer and taking notes. ⭐ = drill until you can write it from blank in ~20 minutes (week-4 speed round). " +
        "The plan also schedules breadth reading, system design and mocks; see the source file for those.\n",
    ];
    for (const w of studyPlan.weeks) {
      parts.push(`## Week ${w.week}: ${w.title}\n`);
      for (const item of w.items) {
        const q = getQuestionById(item.id);
        const label = q ? `${q.title} (${q.difficulty}${q.graded ? ", auto-graded" : ""})` : "(not in this build)";
        const meta = [item.day, item.time].filter(Boolean).join(" · ");
        parts.push(`- ${meta ? `${meta} — ` : ""}**${item.id}** ${label}${item.star ? " ⭐" : ""}${item.note ? ` — ${item.note}` : ""}`);
      }
      parts.push("");
    }
    return { content: [{ type: "text" as const, text: notice + parts.join("\n") }] };
  }

  if (track === "llm-path") {
    const parts = [
      "# LLM Learning Path — Build a Language Model from Scratch\n",
      "Work through these in order. Each builds on the previous.\n",
    ];
    for (const stage of getLLMPathStages()) {
      parts.push(`## ${stage.title}\n`);
      for (const q of stage.questions) {
        const step = q.llmPath ? `${q.llmPath.order}. ` : "";
        parts.push(`- ${step}**${q.id}** ${q.title} (${q.difficulty})`);
      }
      parts.push("");
    }
    return { content: [{ type: "text" as const, text: notice + parts.join("\n") }] };
  }

  const qs = getQuestionsByTrack(track);
  const parts = [
    track === "basics"
      ? `# Basics Track — ${qs.length} problems\n\nCore PyTorch, deep-learning and classical-ML fundamentals, grouped by directory, easiest first.\n`
      : `# Advanced Track — ${qs.length} problems\n\nArchitectures, generative models, alignment, inference and systems, grouped by directory, easiest first.\n`,
  ];
  for (const group of groupBySection(qs)) {
    parts.push(`## ${group.title} (\`${group.section}/\`)\n`);
    for (const q of group.questions) parts.push(`- **${q.id}** ${q.title} (${q.difficulty})`);
    parts.push("");
  }
  return { content: [{ type: "text" as const, text: notice + parts.join("\n") }] };
}
