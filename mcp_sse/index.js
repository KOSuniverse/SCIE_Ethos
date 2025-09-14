// Minimal MCP SSE server exposing three tools: list, search, get
import { createServer } from "http";
import { SSEServer } from "@modelcontextprotocol/sdk/server/sse.js";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import axios from "axios";

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_ACCESS_TOKEN,
  SERVER_API_KEY
} = process.env;

if (!DROPBOX_ACCESS_TOKEN) {
  console.error("Missing DROPBOX_ACCESS_TOKEN");
  process.exit(1);
}

// Dropbox helpers
const dbxRPC = axios.create({
  baseURL: "https://api.dropboxapi.com/2",
  headers: {
    Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
    "Content-Type": "application/json"
  }
});
const dbxContent = axios.create({
  baseURL: "https://content.dropboxapi.com/2",
  headers: { Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}` }
});

// Define MCP tools
/** @type {Tool[]} */
const tools = [
  {
    name: "dbx_list",
    description: "List files/folders recursively under a path (defaults to DBX_ROOT_PREFIX).",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: []
    }
  },
  {
    name: "dbx_search",
    description: "Search files by query within DBX_ROOT_PREFIX.",
    inputSchema: {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"]
    }
  },
  {
    name: "dbx_get",
    description: "Download a file by path (returns base64).",
    inputSchema: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"]
    }
  }
];

// Tool handlers
async function handleToolCall(name, args) {
  if (name === "dbx_list") {
    let p = (args?.path ?? DBX_ROOT_PREFIX).toString().trim();
    if (p === "/" || p === "") p = "";
    else if (!p.startsWith("/")) p = "/" + p;

    let entries = [];
    let { data } = await dbxRPC.post("/files/list_folder", {
      path: p, recursive: true, include_deleted: false
    });
    entries.push(...(data.entries || []));
    while (data.has_more) {
      const c = await dbxRPC.post("/files/list_folder/continue", { cursor: data.cursor });
      data = c.data;
      entries.push(...(data.entries || []));
    }
    return { ok: true, base_path: p || "(root)", count: entries.length, entries };
  }

  if (name === "dbx_search") {
    const q = (args?.query ?? "").toString();
    const { data } = await dbxRPC.post("/files/search_v2", {
      query: q,
      options: { path: DBX_ROOT_PREFIX.startsWith("/") ? DBX_ROOT_PREFIX : "/" + DBX_ROOT_PREFIX, max_results: 100, file_status: "active" }
    });
    return { ok: true, results: data };
  }

  if (name === "dbx_get") {
    const path = (args?.path ?? "").toString();
    if (!path) throw new Error("path required");
    const resp = await dbxContent.post("/files/download", null, {
      headers: { "Dropbox-API-Arg": JSON.stringify({ path }) },
      responseType: "arraybuffer"
    });
    return {
      ok: true,
      path,
      data_base64: Buffer.from(resp.data).toString("base64")
    };
  }

  throw new Error(`Unknown tool: ${name}`);
}

// Create SSE server
const httpServer = createServer(async (req, res) => {
  // Basic API key check (optional)
  if (SERVER_API_KEY && req.url?.startsWith("/sse")) {
    const key = req.headers["x-api-key"];
    if (!key || key !== SERVER_API_KEY) {
      res.statusCode = 403;
      res.end("Forbidden");
      return;
    }
  }
  // Simple health
  if (req.url === "/healthz") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ ok: true, root: DBX_ROOT_PREFIX }));
    return;
  }
  // Everything else: hand to MCP SSE
  sse.handle(req, res);
});

// Wire MCP SSE
const sse = new SSEServer({
  server: {
    name: "dbx-mcp-sse",
    version: "1.0.0"
  },
  // Advertise tools to ChatGPT
  capabilities: { tools },
  // Handle tool calls from ChatGPT
  onToolCall: async ({ name, arguments: args }) => {
    try {
      const out = await handleToolCall(name, args);
      return { content: [{ type: "text", text: JSON.stringify(out) }] };
    } catch (err) {
      return { isError: true, content: [{ type: "text", text: String(err?.message || err) }] };
    }
  }
});

const PORT = process.env.PORT || 3000;
httpServer.listen(PORT, () => console.log(`MCP SSE on :${PORT}  root=${DBX_ROOT_PREFIX}`));
