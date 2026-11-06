#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createServer } from "./server.js";
import { catalogDrift, REBUILD_HINT } from "./data/questions.js";

const drift = catalogDrift();
if (drift) {
  console.error(
    `torchleet-mcp: catalogue out of date (${drift.missing.length} new, ${drift.removed.length} removed problem dirs). Run: ${REBUILD_HINT}`,
  );
}

const server = createServer();
const transport = new StdioServerTransport();
await server.connect(transport);
