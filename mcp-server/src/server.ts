import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { questions } from "./data/questions.js";

import { listQuestionsSchema, listQuestions } from "./tools/list-questions.js";
import { getQuestionSchema, getQuestion } from "./tools/get-question.js";
import { getLearningPathSchema, getLearningPath } from "./tools/get-learning-path.js";
import { getHintSchema, getHint } from "./tools/get-hint.js";
import { checkSolutionSchema, checkSolution } from "./tools/check-solution.js";
import { getCompanyPrepSchema, getCompanyPrep } from "./tools/get-company-prep.js";
import { getPrerequisitesSchema, getPrerequisites } from "./tools/get-prerequisites.js";

import { TUTOR_PROMPT, getTutorMessages } from "./prompts/tutor.js";
import {
  INTERVIEW_PREP_PROMPT,
  interviewPrepSchema,
  getInterviewPrepMessages,
} from "./prompts/interview-prep.js";
import { REVIEW_PROMPT, getReviewMessages } from "./prompts/review.js";
import {
  EXPLAIN_PROMPT,
  explainSchema,
  getExplainMessages,
} from "./prompts/explain.js";

export function createServer(): McpServer {
  const server = new McpServer({
    name: "torchleet",
    version: "0.1.0",
  });

  // --- Tools ---

  server.tool(
    "list_questions",
    "Browse and filter TorchLeet problems by set (v1-v4), difficulty, category, track, auto-graded, or (inferred) company tag",
    listQuestionsSchema,
    async (params) => listQuestions(params),
  );

  server.tool(
    "get_question",
    "Get a problem's details: description, notebook paths, how to grade it with torchleet.check(), study-plan slot, and (inferred) company tags. Accepts ids, legacy ids (v3-12) or directories",
    getQuestionSchema,
    async (params) => getQuestion(params),
  );

  server.tool(
    "get_learning_path",
    "Get a structured learning path: LLM path (build an LLM from scratch), basics, advanced, or the owner's 4-week study plan (cv-mm-gen-4wk)",
    getLearningPathSchema,
    async (params) => getLearningPath(params),
  );

  server.tool(
    "get_hint",
    "Get progressive hints for a problem (level 1 = gentlest nudge ... 3 = most specific); the same hints torchleet.hint() prints. Never reveals the full solution.",
    getHintSchema,
    async (params) => getHint(params),
  );

  server.tool(
    "review_solution",
    "Review submitted code against a question's requirements — returns a verification checklist, does NOT execute code",
    checkSolutionSchema,
    async (params) => checkSolution(params),
  );

  server.tool(
    "get_company_prep",
    "List problems carrying a company tag, easiest first. Tags are inferred from public material, not interview reports",
    getCompanyPrepSchema,
    async (params) => getCompanyPrep(params),
  );

  server.tool(
    "get_prerequisites",
    "Find prerequisite questions to complete before tackling a harder problem",
    getPrerequisitesSchema,
    async (params) => getPrerequisites(params),
  );

  // --- Prompts ---

  server.prompt(
    TUTOR_PROMPT.name,
    TUTOR_PROMPT.description,
    () => getTutorMessages(),
  );

  server.prompt(
    INTERVIEW_PREP_PROMPT.name,
    INTERVIEW_PREP_PROMPT.description,
    interviewPrepSchema,
    (params) => getInterviewPrepMessages(params),
  );

  server.prompt(
    REVIEW_PROMPT.name,
    REVIEW_PROMPT.description,
    () => getReviewMessages(),
  );

  server.prompt(
    EXPLAIN_PROMPT.name,
    EXPLAIN_PROMPT.description,
    explainSchema,
    (params) => getExplainMessages(params),
  );

  // --- Resources ---

  server.resource(
    "question-catalog",
    "torchleet://questions",
    { description: "Full TorchLeet problem catalogue (generated from the manifests) as JSON", mimeType: "application/json" },
    async (uri) => ({
      contents: [
        {
          uri: uri.href,
          text: JSON.stringify(questions, null, 2),
          mimeType: "application/json",
        },
      ],
    }),
  );

  return server;
}
