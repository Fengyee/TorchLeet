import { readFileSync } from "node:fs";
import { join } from "node:path";
import { getRepoRoot } from "./repo-root.js";

export interface ParsedNotebook {
  problemStatement: string;
  functionSignatures: string[];
  todoComments: string[];
  validationCode: string;
}

const cache = new Map<string, ParsedNotebook | null>();

type Cell = { cell_type: string; source: string | string[] };

// nbformat allows a cell's source as one string or as a list of lines.
const text = (c: Cell) => (Array.isArray(c.source) ? c.source.join("") : c.source ?? "");

export function parseNotebook(relativePath: string): ParsedNotebook | null {
  if (cache.has(relativePath)) return cache.get(relativePath)!;

  const root = getRepoRoot();
  if (!root) {
    cache.set(relativePath, null);
    return null;
  }

  const fullPath = join(root, relativePath);
  let raw: string;
  try {
    raw = readFileSync(fullPath, "utf-8");
  } catch {
    cache.set(relativePath, null);
    return null;
  }

  let nb: { cells: Cell[] };
  try {
    nb = JSON.parse(raw);
  } catch {
    cache.set(relativePath, null);
    return null;
  }

  const mdCells = nb.cells.filter((c) => c.cell_type === "markdown");
  const codeCells = nb.cells.filter((c) => c.cell_type === "code");

  const problemStatement = mdCells
    .map(text)
    .join("\n\n");

  const functionSignatures: string[] = [];
  const todoComments: string[] = [];

  for (const cell of codeCells) {
    for (const line of text(cell).split("\n")) {
      const trimmed = line.trim();
      if (trimmed.startsWith("def ") || trimmed.startsWith("class ")) {
        functionSignatures.push(trimmed);
      }
      if (trimmed.includes("TODO")) {
        todoComments.push(trimmed.replace(/^#\s*/, ""));
      }
    }
  }

  // The final cell is usually the auto-grade call (`check("<id>", ...)`); the
  // notebook's own tests are the last code cell before it.
  const testCells = codeCells.filter((c) => !/from torchleet import check/.test(text(c)));
  const lastCodeCell = testCells[testCells.length - 1] ?? codeCells[codeCells.length - 1];
  const validationCode = lastCodeCell
    ? text(lastCodeCell)
    : "";

  const result: ParsedNotebook = {
    problemStatement,
    functionSignatures,
    todoComments,
    validationCode,
  };

  cache.set(relativePath, result);
  return result;
}

export function getDocstrings(relativePath: string): string[] {
  const root = getRepoRoot();
  if (!root) return [];

  const fullPath = join(root, relativePath);
  let raw: string;
  try {
    raw = readFileSync(fullPath, "utf-8");
  } catch {
    return [];
  }

  let nb: { cells: Cell[] };
  try {
    nb = JSON.parse(raw);
  } catch {
    return [];
  }

  const docstrings: string[] = [];
  for (const cell of nb.cells.filter((c) => c.cell_type === "code")) {
    const src = text(cell);
    const matches = src.match(/"""[\s\S]*?"""/g);
    if (matches) docstrings.push(...matches);
  }
  return docstrings;
}
