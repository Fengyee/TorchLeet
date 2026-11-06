import { z } from "zod";
import {
  getQuestionsByCompany,
  getAllCompanies,
  groupBySection,
  byDifficulty,
  COMPANY_TAG_NOTE,
  driftNotice,
} from "../data/questions.js";

export const getCompanyPrepSchema = {
  company: z.string().describe("Company tag (e.g. 'Anthropic', 'Meta', 'Google'). Tags are inferred, not interview reports."),
};

export async function getCompanyPrep(params: { company: string }) {
  const qs = getQuestionsByCompany(params.company);

  if (qs.length === 0) {
    const companies = getAllCompanies().map((c) => `${c.name} (${c.count})`);
    return {
      content: [
        {
          type: "text" as const,
          text: `${driftNotice()}No problems are tagged "${params.company}".\n\nCompany tags (problem count): ${companies.join(", ")}\n\n${COMPANY_TAG_NOTE}`,
        },
      ],
    };
  }

  const name = qs[0].companies.find((c) => c.name.toLowerCase() === params.company.trim().toLowerCase())!.name;
  const confidences = new Set(qs.map((q) => q.companies.find((c) => c.name === name)!.confidence));
  const sorted = [...qs].sort(byDifficulty);

  const parts: string[] = [
    `# Problems tagged ${name}\n`,
    `${qs.length} problem(s) carry the ${name} tag (${Array.from(confidences).join(" / ")}).`,
    `\n> ${COMPANY_TAG_NOTE} Use this as a topic list for the kind of role, not as a list of questions ${name} asks.\n`,
    "Suggested order: easy/medium first, then hard/expert.\n",
  ];

  parts.push("## By difficulty\n");
  for (const q of sorted) {
    parts.push(`- **${q.id}** ${q.title} — ${q.difficulty}${q.graded ? " (auto-graded)" : ""}`);
  }

  parts.push("\n## By topic\n");
  for (const group of groupBySection(qs)) {
    parts.push(`### ${group.title} (\`${group.section}/\`)\n`);
    for (const q of group.questions) parts.push(`- ${q.id} ${q.title} (${q.difficulty})`);
    parts.push("");
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
