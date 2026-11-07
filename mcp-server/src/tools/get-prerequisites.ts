import { z } from "zod";
import { getQuestionById, questions, byDifficulty, DIFFICULTY_ORDER, type Question } from "../data/questions.js";
import { notFound } from "./get-question.js";

export const getPrerequisitesSchema = {
  id: z.string().describe("Problem id (e.g. 'dpo') or legacy id (e.g. 'v3-14') to find prerequisites for"),
};

// Curated "do these first" edges, by problem id (slug). An empty list marks a
// good starting point. Problems not listed fall back to easier problems from
// the same directory. Ids that stop resolving are dropped (and reported by
// unresolvedPrerequisites) rather than shown as broken links.
const PREREQ_GRAPH: Record<string, string[]> = {
  // Attention chain
  "implement-attention-from-scratch": [],
  "multi-head-attention": ["implement-attention-from-scratch"],
  "grouped-query-attention": ["multi-head-attention"],
  "sliding-window-attention": ["multi-head-attention"],
  "kv-cache": ["multi-head-attention"],

  // Positional embeddings
  "sinusoidal-positional-embedding": [],
  "rotary-positional-embedding": ["sinusoidal-positional-embedding"],
  "2d-positional-embeddings": ["sinusoidal-positional-embedding"],

  // SmolLM capstone
  smollm: ["byte-pair-encoder", "rms-norm", "rotary-positional-embedding", "grouped-query-attention", "kv-cache"],

  // Alignment chain
  lora: [],
  dpo: ["lora"],
  "ppo-rlhf": ["dpo"],
  grpo: ["ppo-rlhf"],

  // Decoding chain
  "temperature-sampling": [],
  "top-k-sampling": ["temperature-sampling"],
  "top-p-sampling": ["temperature-sampling"],
  "beam-search": ["temperature-sampling"],

  // Inference chain
  "speculative-decoding": ["kv-cache"],
  "continuous-batching": ["kv-cache"],
  "inference-engine": ["speculative-decoding", "continuous-batching"],

  // GPU systems chain
  softmax: [],
  "triton-fused-softmax": ["softmax"],
  "flash-attention": ["implement-attention-from-scratch", "softmax"],
  "flash-attention-triton": ["flash-attention", "triton-fused-softmax"],
  fsdp: [],
  "ring-attention": ["multi-head-attention", "flash-attention", "fsdp"],

  // Modern architectures
  "contrastive-loss-clip": [],
  "mixture-of-experts": [],
  mamba: [],
  ddpm: [],
  "ddim-cfg": ["ddpm"],
  "vit-mae": ["2d-positional-embeddings"],
  "knowledge-distillation": [],
  "gradient-checkpointing": [],

  // v4
  micrograd: [],
  "mlp-backprop": [],
  "gradient-boosting": ["decision-tree"],
  "detection-map": ["iou-nms"],
  "siglip-loss": ["contrastive-loss-clip"],
  "vlm-connector": ["multi-head-attention"],
  "dit-block": ["multi-head-attention", "ddpm"],
};

/** Graph ids that no longer resolve to a problem (for tests / maintenance). */
export function unresolvedPrerequisites(): string[] {
  const ids = Object.entries(PREREQ_GRAPH).flatMap(([k, v]) => [k, ...v]);
  return Array.from(new Set(ids.filter((id) => !getQuestionById(id))));
}

export async function getPrerequisites(params: { id: string }) {
  const q = getQuestionById(params.id);
  if (!q) return notFound(params.id);

  const prereqIds = Object.prototype.hasOwnProperty.call(PREREQ_GRAPH, q.id) ? PREREQ_GRAPH[q.id] : undefined;
  const prereqs = (prereqIds ?? [])
    .map((id) => getQuestionById(id))
    .filter((p): p is Question => p !== undefined);

  if (prereqs.length > 0) {
    const parts = [`## Prerequisites for: ${q.title} (${q.difficulty})\n`, "Complete these first:\n"];
    prereqs.forEach((p, i) => parts.push(`${i + 1}. **${p.id}** ${p.title} (${p.difficulty}) — ${p.description}`));
    return { content: [{ type: "text" as const, text: parts.join("\n") }] };
  }

  if (prereqIds) {
    return {
      content: [
        {
          type: "text" as const,
          text: `## ${q.title}\n\nNo prerequisites — this is a good starting point! Just make sure you're comfortable with basic PyTorch tensor operations.`,
        },
      ],
    };
  }

  // Not in the graph: suggest easier problems from the same directory or category.
  const sameArea = questions.filter(
    (other) =>
      other.id !== q.id &&
      (other.section === q.section || (q.category !== "" && other.category === q.category)) &&
      (DIFFICULTY_ORDER[other.difficulty] ?? 5) < (DIFFICULTY_ORDER[q.difficulty] ?? 5),
  );

  if (sameArea.length === 0) {
    return {
      content: [
        {
          type: "text" as const,
          text: `## ${q.title}\n\nNo specific prerequisites mapped. Start with the basics track if you're new to PyTorch.`,
        },
      ],
    };
  }

  const parts = [
    `## Suggested preparation for: ${q.title} (${q.difficulty})\n`,
    "These easier problems cover related concepts:\n",
  ];
  for (const p of [...sameArea].sort(byDifficulty).slice(0, 5)) {
    parts.push(`- **${p.id}** ${p.title} (${p.difficulty})`);
  }
  return { content: [{ type: "text" as const, text: parts.join("\n") }] };
}
