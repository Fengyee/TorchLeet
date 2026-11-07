import { z } from "zod";
import { getQuestionById, checkUsage } from "../data/questions.js";
import { parseNotebook, getDocstrings } from "../lib/notebook-parser.js";
import { getAuthoredHints } from "../data/hints.generated.js";
import { notFound } from "./get-question.js";

export const getHintSchema = {
  id: z.string().describe("Problem id (e.g. 'kv-cache') or legacy id (e.g. 'v3-12')"),
  level: z
    .number()
    .int()
    .min(1)
    .default(1)
    .describe("Hint level, escalating: 1 = gentlest nudge … 3 = most specific (most problems have 3). Earlier levels are repeated."),
};

export async function getHint(params: { id: string; level: number }) {
  const q = getQuestionById(params.id);
  if (!q) return notFound(params.id);

  const requested = params.level ?? 1;

  // Hints authored with the grader checks and copied into the manifest, so the
  // tutor and `torchleet.hint()` give the same answer. Problems without any fall
  // back to hints derived from the notebook.
  const authored = getAuthoredHints(q.id);
  if (authored && authored.length) {
    const level = Math.min(requested, authored.length);
    const parts = [
      `## Hint for: ${q.title} (level ${level} of ${authored.length})`,
      "",
      ...authored.slice(0, level).map((h, i) => `${i + 1}. ${h}`),
    ];
    if (requested > authored.length) parts.push(`\n(This problem has only ${authored.length} hints.)`);
    if (level < authored.length) parts.push(`\n---\n*Still stuck after trying? Ask for hint level ${level + 1}.*`);
    parts.push(
      q.graded
        ? `\nCheck your work: \`${checkUsage(q)}\``
        : "\nNot auto-graded: run the notebook's own checks, then compare with the solution notebook.",
    );
    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }

  const level = Math.min(requested, 3);
  const nb = parseNotebook(q.questionPath);
  if (nb) return notebookHint(q.title, nb, level, q.questionPath);
  return descriptionHint(q.title, q.description, level);
}

function notebookHint(
  title: string,
  nb: ReturnType<typeof parseNotebook> & {},
  level: number,
  questionPath: string,
) {
  const parts: string[] = [`## Hint for: ${title} (Level ${level}/3, derived from the notebook)\n`];

  if (level >= 1) {
    const statement = nb.problemStatement;
    const truncated = statement.length > 1500
      ? statement.slice(0, 1500) + "\n\n...(problem statement truncated)"
      : statement;
    parts.push("### Problem Statement & Requirements\n");
    parts.push(truncated);
  }

  if (level >= 2) {
    parts.push("\n### Function Signatures & Docstrings\n");
    if (nb.functionSignatures.length > 0) {
      parts.push("You need to implement:\n");
      for (const sig of nb.functionSignatures) {
        parts.push(`\`\`\`python\n${sig}\n\`\`\``);
      }
    }
    const docs = getDocstrings(questionPath);
    if (docs.length > 0) {
      parts.push("\nDocstrings (expected behavior):\n");
      for (const doc of docs) {
        parts.push(`\`\`\`\n${doc}\n\`\`\``);
      }
    }
  }

  if (level >= 3) {
    parts.push("\n### Step-by-Step TODO Guide\n");
    if (nb.todoComments.length > 0) {
      parts.push("Follow these steps:\n");
      for (const todo of nb.todoComments) {
        parts.push(`- ${todo}`);
      }
    } else {
      parts.push("No explicit TODOs in this notebook — review the function signatures and tests carefully.");
    }
  }

  if (level < 3) {
    parts.push(`\n---\n*Need more help? Ask for hint level ${level + 1}.*`);
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}

function descriptionHint(title: string, description: string, level: number) {
  const parts: string[] = [`## Hint for: ${title} (Level ${level}/3)\n`];

  if (level >= 1) {
    parts.push("### What to build\n");
    parts.push(description || "No description available for this question.");
  }

  if (level >= 2) {
    parts.push("\n### Approach\n");
    parts.push("Think about:");
    parts.push("- What are the input and output tensor shapes?");
    parts.push("- What PyTorch operations will you need?");
    parts.push("- Are there numerical stability concerns?");
  }

  if (level >= 3) {
    parts.push("\n### Key considerations\n");
    parts.push("- Start by writing the function signature with type hints");
    parts.push("- Test with small tensors first (2x3 or 3x4)");
    parts.push("- Compare your output against PyTorch's built-in implementation");
    parts.push("- Check edge cases: empty tensors, single elements, very large values");
  }

  if (level < 3) {
    parts.push(`\n---\n*Need more help? Ask for hint level ${level + 1}.*`);
  }

  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
