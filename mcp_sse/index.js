// Dropbox REST for GPT Actions (refresh-token enabled)
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
    const expired =
      status === 401 ||
      (status === 400 && txt.includes("expired_access_token")) ||
      txt.includes("invalid_access_token");
    if (!expired) throw e;
    await refreshAccessToken();
    return fn(_accessToken);
  }
}

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
    mime: e[".tag"] === "file" ? e.mime_type ?? null : "folder",
  }));

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
const dbxDownload = ({ path, rangeStart = null, rangeEnd = null }) =>
  withAuth((token) =>
    axios.post(`${DBX_CONTENT}/files/download`, null, {
      headers: {
        Authorization: `Bearer ${token}`,
        "Dropbox-API-Arg": JSON.stringify({ path }),
        ...(rangeStart != null && rangeEnd != null
          ? { Range: `bytes=${rangeStart}-${rangeEnd - 1}` }
          : {}),
      },
      responseType: "arraybuffer",
      timeout: 120000,
    })
  );

// --- Express app ---
const app = express();
app.use(express.json());

// Gate /mcp/* with API key
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  if (req.path.startsWith("/mcp")) {
    const key = req.headers["x-api-key"];
    if (key !== SERVER_API_KEY) return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

// Health
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true, root: DBX_ROOT_PREFIX }));

// Recursive walk
app.post("/mcp/walk", async (req, res) => {
  try {
    const { path_prefix = DBX_ROOT_PREFIX, max_items = 2000, cursor } = req.body || {};
    let entries = [], next = cursor || null;
    if (!next) {
      const r = await dbxListFolder({
        path: normPath(path_prefix),
        recursive: true,
        limit: Math.min(max_items, 2000),
      });
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
    res.json({
      entries: normalizeEntries(j.entries || []),
      truncated: j.has_more,
      next_cursor: j.has_more ? j.cursor : null,
    });
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

    const r = await withAuth(async (token) =>
      axios.post("https://content.dropboxapi.com/2/files/download", null, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Dropbox-API-Arg": JSON.stringify({ path }),
          ...(range_start != null && range_end != null
            ? { Range: `bytes=${range_start}-${range_end - 1}` }
            : {})
        },
        responseType: "arraybuffer"
      })
    );

    const buf = Buffer.from(r.data);
    res.json({
      ok: true,
      path,
      content_type: r.headers["content-type"] || null,
      size_bytes: buf.length,
      data_base64: buf.toString("base64")
    });
  } catch (e) {
    const status = e?.response?.status || null;
    const data   = e?.response?.data || e?.message;
    console.error("DROPBOX GET error", status, data);
    res.status(502).json({ ok: false, status, data });
  }
});

app.listen(PORT, () =>
  console.log(
    `DBX REST on ${PORT} root=${DBX_ROOT_PREFIX} ROUTES: /mcp/healthz /mcp/walk /mcp/list /mcp/meta /mcp/get`
  )
);



