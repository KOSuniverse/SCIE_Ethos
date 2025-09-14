// Minimal MCP SSE server for Dropbox
import { createServer } from "http";
import { SSEServer } from "@modelcontextprotocol/sdk/server/sse/index.js"; // <-- 0.6.x path
import axios from "axios";

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_ACCESS_TOKEN,
  SERVER_API_KEY,
  PORT = 3000
} = process.env;

if (!DROPBOX_ACCESS_TOKEN) {
  console.error("Missing DROPBOX_ACCESS_TOKEN");
  process.exit(1);
}

// Dropbox clients
const dbxRPC = axios.create({
  baseURL: "https://api.dropboxapi.com/2",
  headers: { Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`, "Content-Type": "application/json" }
});
const dbxContent = axios.create({
  baseURL: "https://content.dropboxapi.com/2",
  headers: { Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}` }
});

// Tool impls
async function dbx_list({ path }) {
  let p = (path ?? DBX_ROOT_PREFIX).toString().trim();
  if (p === "/" || p === "") p = "";
  else if (!p.startsWith("/")) p = "/" + p;

  let entries = [];
  let { data } = await dbxRPC.post("/files/list_folder", { path: p, recursive: true, include_deleted: false });
  entries.push(...(data.entries || []));
  while (data.has_more) {
    const cont = await dbxRPC.post("/files/list_folder/continue", { cursor: data.cursor });
    data = cont.data;
    entries.push(...(data.entries || []));
  }
  return { ok: true, base_path: p || "(root)", count: entries.length, entries };
}

async function dbx_search({ query }) {
  const q = (query ?? "").toString();
  const base = DBX_ROOT_PREFIX.startsWith("/") ? DBX_ROOT_PREFIX : "/" + DBX_ROOT_PREFIX;
  const { data } = await dbxRPC.post("/files/search_v2", {
    query: q,
    options: { path: base, max_results: 100, file_status: "active" }
  });
  return { ok: true, results: data };
}

async function dbx_get({ path }) {
  const p = (path ?? "").toString();
  if (!p) throw new Error("path required");
  const resp = await dbxContent.post("/files/download", null, {
    headers: { "Dropbox-API-Arg": JSON.stringify({ path: p }) },
    responseType: "arraybuffer"
  });
  return { ok: true, path: p, data_base64: Buffer.from(resp.data).toString("base64") };
}

// Build the SSE MCP server
const sse = new SSEServer(
  {
    name: "dbx-mcp-sse",
    version: "1.0.0"
  },
  {
    // tool registry (name → schema → handler)
    tools: [
      {
        name: "dbx_list",
        description: "List files/folders recursively under a path (defaults to DBX_ROOT_PREFIX).",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: []
        },
        handler: dbx_list
      },
      {
        name: "dbx_search",
        description: "Search for files by query within DBX_ROOT_PREFIX.",
        inputSchema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"]
        },
        handler: dbx_search
      },
      {
        name: "dbx_get",
        description: "Download a file by path (returns base64).",
        inputSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"]
        },
        handler: dbx_get
      }
    ]
  }
);

// Plain HTTP server that gates `/sse` behind API key and exposes `/healthz`
const httpServer = createServer((req, res) => {
  if (req.url === "/healthz") {
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify({ ok: true, root: DBX_ROOT_PREFIX }));
    return;
  }
  if (req.url?.startsWith("/sse")) {
    const key = req.headers["x-api-key"];
    if (SERVER_API_KEY && key !== SERVER_API_KEY) {
      res.statusCode = 403;
      res.end("Forbidden");
      return;
    }
    return sse.handle(req, res);
  }
  res.statusCode = 404;
  res.end("Not found");
});

httpServer.listen(PORT, () => console.log(`MCP SSE on :${PORT}  root=${DBX_ROOT_PREFIX}`));

