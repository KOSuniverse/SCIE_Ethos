// Dropbox REST API for GPT Actions (refresh-token enabled) — COMPLETE DROP-IN + INDEX BUILDER
import express from "express";
import axios from "axios";
import crypto from "node:crypto"; // at top once

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,
  SERVER_API_KEY,
  PORT = process.env.PORT || 10000
} = process.env;
function sha1(buf){ return crypto.createHash("sha1").update(buf).digest("hex"); }
const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP = /\.zip$/i;
const XLSX = /\.xlsx$/i;

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
// heuristics (reuse from earlier robust index)
function parseMonthFromName(name){ const m1=name.match(/\b(20\d{2})[-_ ]?(0[1-9]|1[0-2])\b/); if(m1) return `${m1[1]}-${m1[2]}`; const M={jan:"01",feb:"02",mar:"03",apr:"04",may:"05",jun:"06",jul:"07",aug:"08",sep:"09",oct:"10",nov:"11",dec:"12"}; const m2=name.toLowerCase().match(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_ ]?(\d{2,4})\b/); if(m2){const mm=M[m2[1]]; const yy=m2[2].length===2?`20${m2[2]}`:m2[2]; return `${yy}-${mm}`;} return null; }
function guessCurrency(s){ if(/\bUSD\b|US\$|\$/.test(s)) return "USD"; if(/\bEUR\b|€/.test(s)) return "EUR"; if(/\bGBP\b|£/.test(s)) return "GBP"; return null; }
function guessCategoryBits(p){ const s=p.toLowerCase(); let category=null, sub=null; if(s.includes("wip")) category="WIP"; if(s.includes("inventory")) category=category||"Inventory"; if(s.includes("aged")) sub="Aged"; if(s.includes("finished")) sub=sub||"Finished"; if(s.includes("provision")) sub=sub||"Provision"; return {category, subcategory:sub}; }
function guessSiteCountryPlant(p){ const out={site:null,country:null,plant_code:null}; for(const seg of p.split("/")){ const m=seg.match(/\b([A-Z]{2,4})\s+(R?\d{3,4})\b/); if(m){out.site=m[1]; out.plant_code=m[2];} if(/italy|torino/i.test(seg)) out.country="ITA"; if(/germany|gmbh|deutschland|muelheim/i.test(seg)) out.country="DEU"; if(/thailand|rayong/i.test(seg)) out.country="THA"; if(/united\s?states|usa|houston|greenville|pps/i.test(seg)) out.country="USA"; if(/uk|aberdeen/i.test(seg)) out.country="GBR"; } return out; }

async function withAuth(fn) {
  if (!_accessToken) await refreshAccessToken();
// quick XLSX sheet-name sniff (reads the [Content_Types].xml part; no heavy parse)
async function sniffXlsxSheetNames(dbxDownloadFn, path) {
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
    // download small head (e.g., 1.5 MB) – many xlsx are zipped xml; need full to unzip—so we conservatively skip deep unzip here
    // keep simple: return placeholder; we’ll upgrade in pass-2 with exceljs if needed
    return []; // placeholder to avoid large downloads for now
  } catch { return []; }
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

/* ---------- Dropbox CONTENT helpers (global fetch; no Content-Type) ---------- */
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

const dbxUpload = ({ path, bytes }) =>
  withAuth(async (token) => {
    const headers = {
      Authorization: `Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path, mode: "overwrite", mute: true, autorename: false }),
      "Content-Type": "application/octet-stream"
    };
    const r = await fetch(`${DBX_CONTENT}/files/upload`, { method: "POST", headers, body: bytes });
    if (!r.ok) {
      const t = await r.text().catch(() => null);
      throw new Error(`upload_failed: ${r.status} ${t || ""}`);
    }
    return r.json();
  });

/* ---------- Express app ---------- */
const app = express();
app.use(express.json());

// API key gate for /mcp/* and /routeThenAnswer
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  if (req.path.startsWith("/mcp") || req.path === "/routeThenAnswer") {
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

// Fast name/path search within a subtree
app.post("/mcp/search_names", async (req, res) => {
  try {
    const { q, path_prefix = DBX_ROOT_PREFIX, limit = 200 } = req.body || {};
    if (!q) return res.status(400).json({ error: "q required" });
    const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
    const all = normalizeEntries(r.data.entries || []);
    const qq = String(q).toLowerCase();
    const hits = all.filter(e =>
      (e.name || "").toLowerCase().includes(qq) ||
      (e.path || "").toLowerCase().includes(qq)
    );
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
// checkpoint helpers
async function readCheckpoint(dbxDownloadFn, cpPath){
  try { const r=await dbxDownloadFn({path:cpPath}); if(!r.ok) return null; const txt=r.data.toString("utf8"); return JSON.parse(txt); } catch { return null; }
}
async function writeDropboxText(path, text){ const bytes=Buffer.from(text,"utf8"); return dbxUpload({path, bytes}); }

/* ---------- INDEX BUILDER (recurses nested; skips .zip) ---------- */
app.post("/mcp/index", async (req, res) => {
// Build a deep, resumable index; skip .zip; capture text previews and basic heuristics.
// Writes JSONL + CSV + manifest if paths provided.
app.post("/mcp/index_full", async (req, res) => {
  try {
    const {
      path_prefix = DBX_ROOT_PREFIX,
      output_csv_path = null,
      output_json_path = null,
      limit = 200000
      output_jsonl_path = "/Project_Root/GPT_Files/_indexes/file_index.jsonl",
      output_csv_path   = "/Project_Root/GPT_Files/_indexes/file_index.csv",
      manifest_path     = "/Project_Root/GPT_Files/_indexes/manifest.json",
      preview_bytes     = 200_000,
      checksum_bytes_limit = 1_000_000,
      resume = true
    } = req.body || {};

    // page whole subtree
    const rel = p => p.startsWith("/") ? p.slice(1) : p;

    // ensure manifest folder exists (Dropbox auto-creates on upload overwrite)
    const prev = resume ? await readCheckpoint(dbxDownload, manifest_path) : null;
    let total = 0, wrote = 0;

    // walk (fresh; for simplicity we don’t delta vs cursor yet)
    let cursor = null, raw = [];
    {
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
      const j = r.data; raw.push(...(j.entries || [])); cursor = j.has_more ? j.cursor : null;
    }
    while (cursor && raw.length < limit) {
    { const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
      const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null; }
    while (cursor) {
      const r = await dbxListContinue(cursor);
      const j = r.data; raw.push(...(j.entries || [])); cursor = j.has_more ? j.cursor : null;
      const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null;
      // optional: break on absurdly huge trees
      if (raw.length > 300_000) break;
    }

    // normalize + skip .zip
    const entries = normalizeEntries(raw).filter(e => {
      if (!e.path) return false;
      return !/\.zip$/i.test(e.path); // skip zip files
    });
    const entries = normalizeEntries(raw)
      .filter(e => e.path && e.mime !== "folder")
      .filter(e => !ZIP.test(e.path));

    // make rows
    const rows = entries.map(e => ({
      name: e.name, path: e.path, mime: e.mime, size: e.size, modified: e.modified
    }));
    total = entries.length;

    // CSV (safe quoting)
    const csvHeader = "name,path,mime,size,modified\n";
    const csvBody = rows.map(r => [
      r.name, r.path, r.mime || "", r.size || "", r.modified || ""
    ].map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const csvBytes = Buffer.from(csvHeader + csvBody, "utf8");
    const jsonBytes = Buffer.from(JSON.stringify(rows), "utf8");

    const artifacts = { total: rows.length, truncated: !!cursor, path_prefix };
    if (output_csv_path) {
      await dbxUpload({ path: output_csv_path, bytes: csvBytes });
      artifacts.csv_saved = output_csv_path;
    }
    if (output_json_path) {
      await dbxUpload({ path: output_json_path, bytes: jsonBytes });
      artifacts.json_saved = output_json_path;
    // CSV header (write once if starting new)
    if (!prev || !prev.csv_header_written) {
      const header = "site,country,plant_code,category,subcategory,month,file_path,as_of_date,currency,scale,canonical,has_totals,is_complete,checksum_sha1,mime,size,modified\n";
      await dbxUpload({ path: output_csv_path, bytes: Buffer.from(header,"utf8") });
    }

    res.json({
      ok: true,
      stats: artifacts,
      csv_head: (csvHeader + csvBody).slice(0, 2000)
    });
  } catch (e) {
    res.status(502).json({ ok:false, message: e.message, data: e?.response?.data || null });
  }
});

/* ---------- Orchestrator: routeThenAnswer (NL query -> skills) ---------- */
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
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: false, limit: 2000 });
      const entries = normalizeEntries(r.data.entries || []).slice(0, 50);
      used_ops.push("list");
      text = `Top entries in ${path_prefix} (showing ${entries.length}):\n` + entries.map(e => `• ${e.name} (${e.mime})`).join("\n");
      artifacts = entries;
    } else if (intent === "Search") {
      const m = q.match(/(?:search|find)\s+(.+)/);
      const term = m ? m[1].trim() : q;
      const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
      const all = normalizeEntries(r.data.entries || []);
      const hits = all.filter(e => (e.name||"").toLowerCase().includes(term) || (e.path||"").toLowerCase().includes(term)).slice(0, 50);
      used_ops.push("walk");
      text = hits.length ? `Found ${hits.length} match(es) for “${term}” (showing ${hits.length <= 50 ? hits.length : 50}).` : `No matches for “${term}”.`;
      artifacts = hits;
    } else if (intent === "Preview") {
      const m = query.match(/["“](.+?)["”]/);
      const path = m ? m[1] : null;
      if (!path) {
        text = 'Preview needs a path in quotes, e.g., preview "/project_root/gpt_files/05_Alias_Map.yaml".';
    // process entries (stream in chunks; append JSONL & CSV)
    for (const e of entries) {
      const filePath = e.path;
      const name = e.name || "";
      const month = parseMonthFromName(name) || parseMonthFromName(filePath) || null;
      const { category, subcategory } = guessCategoryBits(filePath);
      const { site, country, plant_code } = guessSiteCountryPlant(filePath);
      const currency = guessCurrency(filePath) || guessCurrency(name);
      const as_of_date = e.modified || null;

      // previews & checksum
      let text_preview = null, truncated = false, content_hash = "";
      let sheet_names = [];

      if (TEXTISH.test(name)) {
        const head = await dbxDownload({ path: filePath, rangeStart: 0, rangeEnd: preview_bytes });
        if (head.ok) {
          const buf = head.data;
          text_preview = buf.toString("utf8");
          truncated = buf.length >= preview_bytes;
          if (buf.length <= checksum_bytes_limit) content_hash = sha1(buf);
        }
      } else if (XLSX.test(name)) {
        sheet_names = await sniffXlsxSheetNames(dbxDownload, filePath); // placeholder ([]) for now
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
        // small files can get hashed cheaply
        if (typeof e.size === "number" && e.size <= checksum_bytes_limit) {
          const dl = await dbxDownload({ path: filePath });
          if (dl.ok) content_hash = sha1(dl.data);
        }
      }
    } else if (intent === "Download") {
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

      // record (JSONL + CSV)
      const row = {
        site, country, plant_code, category, subcategory, month,
        file_path: rel(filePath), as_of_date, currency,
        scale: 1, canonical: true, has_totals: true, is_complete: true,
        checksum_sha1: content_hash, mime: e.mime, size: e.size, modified: e.modified,
        sheet_names, text_preview, truncated
      };

      // append JSONL
      await dbxUpload({ path: output_jsonl_path, bytes: Buffer.from(JSON.stringify(row)+"\n","utf8") });

      // append CSV
      const csvLine = [
        row.site, row.country, row.plant_code, row.category, row.subcategory, row.month,
        row.file_path, row.as_of_date, row.currency, row.scale, row.canonical, row.has_totals, row.is_complete,
        row.checksum_sha1, row.mime, row.size, row.modified
      ].map(v => `"${(v ?? "").toString().replace(/"/g,'""')}"`).join(",") + "\n";
      await dbxUpload({ path: output_csv_path, bytes: Buffer.from(csvLine,"utf8") });

      wrote++;
      // periodically update manifest
      if (wrote % 200 === 0) {
        await writeDropboxText(manifest_path, JSON.stringify({ path_prefix, total, processed: wrote, last_path: filePath, csv_header_written: true }, null, 2));
      }
    } else if (intent === "Forecasting_Policy" || intent === "Root_Cause" || intent === "Comparison") {
      text = `Orchestrator: intent detected = ${intent}. Backend skill not wired yet. Provide target file(s)/SKU/period and I’ll run it next.`;
    }

    return res.json({ text, answer: { intent, used_ops, artifacts } });
    // final manifest
    await writeDropboxText(manifest_path, JSON.stringify({ path_prefix, total, processed: wrote, csv_header_written: true, completed: true }, null, 2));

    res.json({ ok: true, stats: { total, processed: wrote }, outputs: { output_jsonl_path, output_csv_path, manifest_path }});
  } catch (e) {
    console.error("routeThenAnswer error", e?.message);
    res.status(500).json({ error: "router_error", message: e?.message });
    res.status(502).json({ ok:false, message:e.message, data:e?.response?.data || null });
  }
});

/* ---------- Start ---------- */
app.listen(PORT, () =>
  console.log(
    `DBX REST on ${PORT} root=${DBX_ROOT_PREFIX} ROUTES: /mcp/healthz /mcp/walk /mcp/list /mcp/meta /mcp/get /mcp/search_names /mcp/head /mcp/index`
  )
);
