// Dropbox REST API for GPT Actions (refresh-token enabled) — COMPLETE DROP-IN
import express from "express";
import axios from "axios";

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,
  SERVER_API_KEY,
  PORT = process.env.PORT || 10000
} = process.env;

if (!DROPBOX_APP_KEY || !DROPBOX_APP_SECRET || !DROPBOX_REFRESH_TOKEN) {
  console.error("Missing: DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN");
  process.exit(1);
}

const TOKEN_URL   = "https://api.dropboxapi.com/oauth2/token";
const DBX_RPC     = "https://api.dropboxapi.com/2";
const DBX_CONTENT = "https://content.dropboxapi.com/2";

let _accessToken = null;

/* ---------- OAuth refresh ---------- */
async function refreshAccessToken() {
  const r = await axios.post(
    TOKEN_URL,
    new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: DROPBOX_REFRESH_TOKEN,
      client_id: DROPBOX_APP_KEY,
      client_secret: DROPBOX_APP_SECRET
    }),
    { headers: { "Content-Type": "application/x-www-form-urlencoded" }, timeout: 20000 }
  );
  _accessToken = r.data.access_token;
  console.log("REFRESH: new Dropbox access token acquired");
  return _accessToken;
}

async function withAuth(fn) {
  if (!_accessToken) await refreshAccessToken();
  try {
    return await fn(_accessToken);
  } catch (e) {
    const body = e?.response?.data;
    const txt = typeof body === "string" ? body : JSON.stringify(body || {});
    const status = e?.response?.status;
    const expired = status === 401 || (status === 400 && txt.includes("expired_access_token")) || txt.includes("invalid_access_token");
    if (!expired) throw e;
    await refreshAccessToken();
    return fn(_accessToken);
  }
}

/* ---------- Helpers ---------- */
const normPath = (p) => {
  let s = String(p || "").trim();
  if (s === "/" || s === "") return "";
  if (!s.startsWith("/")) s = "/" + s;
  return s;
};

const normalizeEntries = (entries) =>
  (entries || []).map((e) => ({
    path: e.path_display || e.path_lower,
    name: e.name,
    size: e.size ?? null,
    modified: e.server_modified ?? null,
    mime: e[".tag"] === "file" ? e.mime_type ?? null : "folder"
  }));

/* ---------- Dropbox RPC wrappers ---------- */
const dbxListFolder = ({ path, recursive, limit }) =>
  withAuth((token) =>
    axios.post(
      `${DBX_RPC}/files/list_folder`,
      { path, recursive: !!recursive, include_deleted: false, limit },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
    )
  );

const dbxListContinue = (cursor) =>
  withAuth((token) =>
    axios.post(
      `${DBX_RPC}/files/list_folder/continue`,
      { cursor },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
    )
  );

const dbxGetMetadata = (path) =>
  withAuth((token) =>
    axios.post(
      `${DBX_RPC}/files/get_metadata`,
      { path, include_media_info: false, include_deleted: false },
      { headers: { Authorization: `Bearer ${token}` }, timeout: 30000 }
    )
  );

/* ---------- Dropbox download via global fetch (no Content-Type) ---------- */
const dbxDownload = async ({ path, rangeStart = null, rangeEnd = null }) =>
  withAuth(async (token) => {
    const headers = {
      Authorization: `Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };
    if (rangeStart != null && rangeEnd != null) {
      headers.Range = `bytes=${rangeStart}-${rangeEnd - 1}`;
    }
    const r = await fetch(`${DBX_CONTENT}/files/download`, { method: "POST", headers, body: null });
    const ab = await r.arrayBuffer();
    return {
      ok: r.ok,
      status: r.status,
      headers: { "content-type": r.headers.get("content-type") },
      data: Buffer.from(ab),
      text: r.ok ? null : (await r.text().catch(() => null))
    };
  });

/* ---------- Express app ---------- */
const app = express();
app.use(express.json());

// API key gate for /mcp/*
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  if (req.path.startsWith("/mcp")) {
    const key = req.headers["x-api-key"];
    if (key !== SERVER_API_KEY) return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

/* ---------- Routes ---------- */
// Health
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true, root: DBX_ROOT_PREFIX }));

// Recursive walk (paged via cursor)
app.post("/mcp/walk", async (req, res) => {
  try {
    const { path_prefix = DBX_ROOT_PREFIX, max_items = 2000, cursor } = req.body || {};
    let entries = [], next = cursor || null;

    if (!next) {
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: Math.min(max_items, 2000) });
      const j = r.data; entries = j.entries || []; next = j.has_more ? j.cursor : null;
    } else {
      const r = await dbxListContinue(next);
      const j = r.data; entries = j.entries || []; next = j.has_more ? j.cursor : null;
    }

    res.json({ entries: normalizeEntries(entries), next_cursor: next, truncated: !!next });
  } catch (e) {
    res.status(502).json({ ok: false, message: e.message, data: e?.response?.data || null });
  }
});

// Non-recursive list
app.get("/mcp/list", async (req, res) => {
  try {
    const path = req.query.path ? String(req.query.path) : DBX_ROOT_PREFIX;
    const r = await dbxListFolder({ path: normPath(path), recursive: false, limit: 2000 });
    const j = r.data;
    res.json({ entries: normalizeEntries(j.entries || []), truncated: j.has_more, next_cursor: j.has_more ? j.cursor : null });
  } catch (e) {
    res.status(502).json({ ok: false, message: e.message, data: e?.response?.data || null });
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
    res.status(502).json({ ok: false, message: e.message, data: e?.response?.data || null });
  }
});

// Download (byte-range supported)
app.post("/mcp/get", async (req, res) => {
  try {
    const { path, range_start = null, range_end = null } = req.body || {};
    if (!path) return res.status(400).json({ error: "path required" });

    const r = await dbxDownload({ path: String(path), rangeStart: range_start, rangeEnd: range_end });
    if (!r.ok) {
      console.error("DROPBOX GET non-2xx", r.status, r.text);
      return res.status(502).json({ ok: false, status: r.status, data: r.text });
    }

    res.json({
      ok: true,
      path,
      content_type: r.headers["content-type"] || null,
      size_bytes: r.data.length,
      data_base64: r.data.toString("base64")
    });
  } catch (e) {
    const status = e?.response?.status ?? null;
    const data = e?.response?.data ?? e.message;
    console.error("DROPBOX GET error", status, data);
    res.status(502).json({ ok: false, status, data });
  }
});

/* ---------- OPTIONAL: Query helpers ---------- */
// Fast name/path search within a subtree
app.post("/mcp/search_names", async (req, res) => {
  try {
    const { q, path_prefix = DBX_ROOT_PREFIX, limit = 200 } = req.body || {};
    if (!q) return res.status(400).json({ error: "q required" });
    const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
    const all = normalizeEntries(r.data.entries || []);
    const qq = String(q).toLowerCase();
    const hits = all.filter(e => (e.name||"").toLowerCase().includes(qq) || (e.path||"").toLowerCase().includes(qq));
    res.json({ entries: hits.slice(0, limit), total: hits.length });
  } catch (e) {
    res.status(502).json({ ok: false, message: e.message, data: e?.response?.data || null });
  }
});

// Small head/preview (first N bytes; tries to decode text)
app.post("/mcp/head", async (req, res) => {
  try {
    const { path, bytes = 200000 } = req.body || {};
    if (!path) return res.status(400).json({ error: "path required" });
    const end = Math.max(1, Math.min(bytes, 200000));
    const r = await dbxDownload({ path: String(path), rangeStart: 0, rangeEnd: end });
    if (!r.ok) return res.status(502).json({ ok: false, status: r.status, data: r.text });

    const ct = r.headers["content-type"] || "application/octet-stream";
    let text = null, note = null;
    if (/^text\/|json|yaml|csv/i.test(ct)) text = r.data.toString("utf8");
    else if (/excel|spreadsheet|octet-stream/i.test(ct)) note = "binary (excel?) – bytes returned";
    else if (/pdf/i.test(ct)) note = "pdf – not parsed";

    res.json({
      ok: true,
      path,
      content_type: ct,
      size_bytes: r.data.length,
      text,
      data_base64: text ? null : r.data.toString("base64"),
      note
    });
  } catch (e) {
    res.status(502).json({ ok: false, message: e.message, data: e?.response?.data || null });
  }
});

// --- Orchestrator: routeThenAnswer (NL query -> skills) ---
app.post("/routeThenAnswer", async (req, res) => {
  try {
    const { query, context = {} } = req.body || {};
    if (!query || typeof query !== "string") return res.status(400).json({ error: "query required" });

    const q = query.trim().toLowerCase();
    const path_prefix = context.path_prefix || DBX_ROOT_PREFIX;

    // classify (minimal rules; extend later)
    let intent = "Browse";
    if (/(list|show)\s+(folders|files)/.test(q)) intent = "Browse";
    else if (/(search|find)/.test(q)) intent = "Search";
    else if (/(open|view|preview)/.test(q)) intent = "Preview";
    else if (/(download|get file|fetch)/.test(q)) intent = "Download";
    else if (/(forecast|service level|lead time|demand)/.test(q)) intent = "Forecasting_Policy";
    else if (/(root cause|why|driver|rca)/.test(q)) intent = "Root_Cause";
    else if (/(compare|delta|shift)/.test(q)) intent = "Comparison";

    const used_ops = [];
    let text = "Unsupported intent for now.";
    let artifacts = [];

    if (intent === "Browse") {
      // top-level by default
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: false, limit: 2000 });
      const entries = normalizeEntries(r.data.entries || []).slice(0, 50);
      used_ops.push("list");
      text = `Top entries in ${path_prefix} (showing ${entries.length}):\n` + entries.map(e => `• ${e.name} (${e.mime})`).join("\n");
      artifacts = entries;
    }

    else if (intent === "Search") {
      // pull a term from the query (very simple)
      const m = q.match(/(?:search|find)\s+(.+)/);
      const term = m ? m[1].trim() : q;
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
      const all = normalizeEntries(r.data.entries || []);
      const hits = all.filter(e => (e.name||"").toLowerCase().includes(term) || (e.path||"").toLowerCase().includes(term)).slice(0, 50);
      used_ops.push("walk");
      text = hits.length ? `Found ${hits.length} match(es) for “${term}” (showing ${hits.length <= 50 ? hits.length : 50}).` : `No matches for “${term}”.`;
      artifacts = hits;
    }

    else if (intent === "Preview") {
      // try to extract a path from quotes or assume the last token looks like a path
      const m = query.match(/["“](.+?)["”]/);
      const path = m ? m[1] : null;
      if (!path) {
        text = 'Preview needs a path in quotes, e.g., preview "/project_root/gpt_files/05_Alias_Map.yaml".';
      } else {
        const r = await dbxDownload({ path: String(path), rangeStart: 0, rangeEnd: 200000 });
        used_ops.push("head");
        if (!r.ok) return res.json({ text: `Preview failed (${r.status}).`, answer: { intent, used_ops, artifacts: [] }});
        const ct = r.headers["content-type"] || "application/octet-stream";
        let preview = null, note = null;
        if (/^text\/|json|yaml|csv/i.test(ct)) preview = r.data.toString("utf8");
        else if (/excel|spreadsheet|octet-stream/i.test(ct)) note = "binary (excel?) – showing bytes only";
        else if (/pdf/i.test(ct)) note = "pdf – not parsed";
        text = `Preview of ${path} (${ct}):` + (preview ? `\n\n${preview.slice(0, 2000)}` : `\n(${note || "binary"})`);
        artifacts = [{ path, content_type: ct, size_bytes: r.data.length }];
      }
    }

    else if (intent === "Download") {
      const m = query.match(/["“](.+?)["”]/);
      const path = m ? m[1] : null;
      if (!path) {
        text = 'Download needs a path in quotes, e.g., download "/project_root/gpt_files/05_Alias_Map.yaml".';
      } else {
        const r = await dbxDownload({ path: String(path) });
        used_ops.push("get");
        if (!r.ok) return res.json({ text: `Download failed (${r.status}).`, answer: { intent, used_ops, artifacts: [] }});
        text = `Downloaded ${path} (${r.data.length} bytes).`;
        artifacts = [{ path, size_bytes: r.data.length, content_type: r.headers["content-type"] || null, data_base64: r.data.toString("base64") }];
      }
    }

    // stubs for later (keep the contract; return abstain so Jarvis can hand off to specialist GPTs if you want)
    else if (intent === "Forecasting_Policy" || intent === "Root_Cause" || intent === "Comparison") {
      text = `Orchestrator: intent detected = ${intent}. Backend skill not wired yet. Provide target file(s)/SKU/period and I’ll run it next.`;
    }

    return res.json({ text, answer: { intent, used_ops, artifacts } });
  } catch (e) {
    console.error("routeThenAnswer error", e?.message);
    res.status(500).json({ error: "router_error", message: e?.message });
  }
});


/* ---------- Start ---------- */
app.listen(PORT, () =>
  console.log(`DBX REST on ${PORT} root=${DBX_ROOT_PREFIX} ROUTES: /mcp/healthz /mcp/walk /mcp/list /mcp/meta /mcp/get /mcp/search_names /mcp/head`)
);

