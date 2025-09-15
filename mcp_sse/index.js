// MCP + Dropbox API server with token refresh + HTTP routes
import express from "express";
import axios from "axios";
import { SSEServer } from "@modelcontextprotocol/sdk/server/sse/index.js";

const {
  // Root scope for listings
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",

  // Dropbox OAuth2 (use refresh flow; do NOT set a static access token)
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,

  // Optional API key gating for Actions / external callers
  SERVER_API_KEY,

  // Port on Render
  PORT = 3000
} = process.env;

// ---- Hard guards ------------------------------------------------------------
if (!DROPBOX_APP_KEY || !DROPBOX_APP_SECRET || !DROPBOX_REFRESH_TOKEN) {
  console.error("Missing Dropbox OAuth envs: DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN");
  process.exit(1);
}

// ---- Dropbox OAuth refresh --------------------------------------------------
const TOKEN_URL = "https://api.dropboxapi.com/oauth2/token";
let _accessToken = null;

async function refreshAccessToken() {
  const resp = await axios.post(
    TOKEN_URL,
    new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: DROPBOX_REFRESH_TOKEN,
      client_id: DROPBOX_APP_KEY,
      client_secret: DROPBOX_APP_SECRET
    }),
    { headers: { "Content-Type": "application/x-www-form-urlencoded" }, timeout: 20000 }
  );
  _accessToken = resp.data.access_token;
  return _accessToken;
}

async function withAuth(fn) {
  // Ensure token, call, and retry once on expiry
  if (!_accessToken) await refreshAccessToken();
  try {
    return await fn(_accessToken);
  } catch (err) {
    const body = err?.response?.data;
    const text = typeof body === "string" ? body : JSON.stringify(body || {});
    const status = err?.response?.status;
    const expired =
      status === 401 ||
      (status === 400 && text.includes("expired_access_token")) ||
      text.includes("invalid_access_token");

    if (!expired) throw err;
    await refreshAccessToken();
    return fn(_accessToken);
  }
}

// ---- Dropbox endpoints (RPC + content) -------------------------------------
const DBX_RPC = "https://api.dropboxapi.com/2";
const DBX_CONTENT = "https://content.dropboxapi.com/2";

async function dbxListFolder({ path, recursive, limit }) {
  return withAuth(async (token) =>
    axios.post(
      `${DBX_RPC}/files/list_folder`,
      { path, recursive: !!recursive, include_deleted: false, limit },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
    )
  );
}

async function dbxListContinue(cursor) {
  return withAuth(async (token) =>
    axios.post(
      `${DBX_RPC}/files/list_folder/continue`,
      { cursor },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
    )
  );
}

async function dbxGetMetadata(path) {
  return withAuth(async (token) =>
    axios.post(
      `${DBX_RPC}/files/get_metadata`,
      { path, include_media_info: false, include_deleted: false, include_has_explicit_shared_members: false },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 30000 }
    )
  );
}

async function dbxDownload({ path, rangeStart = null, rangeEnd = null }) {
  return withAuth(async (token) =>
    axios.post(`${DBX_CONTENT}/files/download`, null, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Dropbox-API-Arg": JSON.stringify({ path }),
        ...(rangeStart != null && rangeEnd != null ? { Range: `bytes=${rangeStart}-${rangeEnd - 1}` } : {})
      },
      responseType: "arraybuffer",
      timeout: 120000
    })
  );
}

// ---- Express app ------------------------------------------------------------
const app = express();
app.use(express.json());

// API key gate (simple)
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  const key = req.headers["x-api-key"];
  if (key !== SERVER_API_KEY) return res.status(403).json({ error: "Forbidden" });
  next();
});

// Health
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true, root: DBX_ROOT_PREFIX }));

// Recursive walk with cursor
app.post("/mcp/walk", async (req, res) => {
  try {
    const { path_prefix = DBX_ROOT_PREFIX, max_items = 2000, cursor } = req.body || {};
    let outEntries = [];
    let nextCursor = cursor || null;

    if (!nextCursor) {
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: Math.min(max_items, 2000) });
      const j = r.data;
      outEntries = j.entries || [];
      nextCursor = j.has_more ? j.cursor : null;
    } else {
      const r = await dbxListContinue(nextCursor);
      const j = r.data;
      outEntries = j.entries || [];
      nextCursor = j.has_more ? j.cursor : null;
    }

    const entries = normalizeEntries(outEntries);
    res.json({ entries, next_cursor: nextCursor, truncated: !!nextCursor });
  } catch (e) {
    res.status(502).json(errPayload(e));
  }
});

// Non-recursive list
app.get("/mcp/list", async (req, res) => {
  try {
    const path = req.query.path ? String(req.query.path) : DBX_ROOT_PREFIX;
    const r = await dbxListFolder({ path: normPath(path), recursive: false, limit: 2000 });
    const j = r.data;
    const entries = normalizeEntries(j.entries || []);
    res.json({ entries, truncated: j.has_more, next_cursor: j.has_more ? j.cursor : null });
  } catch (e) {
    res.status(502).json(errPayload(e));
  }
});

// Metadata
app.get("/mcp/meta", async (req, res) => {
  try {
    const { path } = req.query;
    if (!path) return res.status(400).json({ error: "path required" });
    const r = await dbxGetMetadata(String(path));
    res.json(r.data);
  } catch (e) {
    res.status(502).json(errPayload(e));
  }
});

// Download (supports byte range)
app.post("/mcp/get", async (req, res) => {
  try {
    const { path, range_start = null, range_end = null } = req.body || {};
    if (!path) return res.status(400).json({ error: "path required" });
    const r = await dbxDownload({ path: String(path), rangeStart: range_start, rangeEnd: range_end });
    const buf = Buffer.from(r.data);
    res.json({
      ok: true,
      path,
      content_type: r.headers["content-type"] || null,
      size_bytes: buf.length,
      data_base64: buf.toString("base64")
    });
  } catch (e) {
    res.status(502).json(errPayload(e));
  }
});

// ---- MCP SSE endpoint (unchanged behavior) ---------------------------------
const sse = new SSEServer({ name: "dbx-mcp-sse", version: "1.1.0" }, {
  tools: [
    {
      name: "dbx_list",
      description: "Recursive list under a path (defaults to DBX_ROOT_PREFIX).",
      inputSchema: { type: "object", properties: { path: { type: "string" } }, required: [] },
      handler: async ({ path }) => walkAll(normPath(path || DBX_ROOT_PREFIX))
    },
    {
      name: "dbx_search",
      description: "Search for files by query within DBX_ROOT_PREFIX.",
      inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
      handler: async ({ query }) => {
        // Minimal search using list + filter to stay simple; you can replace with search_v2
        const all = await walkAll(normPath(DBX_ROOT_PREFIX));
        const q = String(query).toLowerCase();
        const hits = all.entries.filter(e => (e.path || "").toLowerCase().includes(q));
        return { ok: true, count: hits.length, entries: hits.slice(0, 500) };
      }
    },
    {
      name: "dbx_get",
      description: "Download a file by path (returns base64).",
      inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
      handler: async ({ path }) => {
        const r = await dbxDownload({ path: String(path) });
        const buf = Buffer.from(r.data);
        return { ok: true, path, size_bytes: buf.length, data_base64: buf.toString("base64") };
      }
    }
  ]
});

// Mount SSE at /sse, with optional API key gate
app.get("/sse", (req, res) => {
  if (SERVER_API_KEY && req.headers["x-api-key"] !== SERVER_API_KEY) {
    res.status(403).end("Forbidden");
    return;
  }
  sse.handle(req, res);
});

// ---- Helpers ----------------------------------------------------------------
function normPath(p) {
  let s = String(p || "").trim();
  if (s === "/" || s === "") return "";
  if (!s.startsWith("/")) s = "/" + s;
  return s;
}

function normalizeEntries(entries) {
  return (entries || []).map(e => ({
    path: e.path_display || e.path_lower,
    name: e.name,
    size: e.size ?? null,
    modified: e.server_modified ?? null,
    mime: e[".tag"] === "file" ? e.mime_type ?? null : "folder"
  }));
}

async function walkAll(path_prefix) {
  let entries = [];
  let cursor = null;
  // first page
  {
    const r = await dbxListFolder({ path: path_prefix, recursive: true, limit: 2000 });
    const j = r.data;
    entries.push(...(j.entries || []));
    cursor = j.has_more ? j.cursor : null;
  }
  // continue
  while (cursor) {
    const r = await dbxListContinue(cursor);
    const j = r.data;
    entries.push(...(j.entries || []));
    cursor = j.has_more ? j.cursor : null;
  }
  return { ok: true, base_path: path_prefix || "(root)", count: entries.length, entries: normalizeEntries(entries) };
}

function errPayload(e) {
  return {
    ok: false,
    message: e?.message || "error",
    status: e?.response?.status || null,
    data: e?.response?.data || null
  };
}

// ---- Start ------------------------------------------------------------------
app.listen(PORT, () => {
  console.log(`Server on :${PORT}  root=${DBX_ROOT_PREFIX}`);
});


