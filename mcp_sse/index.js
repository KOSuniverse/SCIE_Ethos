// index.js — Dropbox REST API + Orchestrator + Optional Extractors
import express from "express";
import axios from "axios";
import crypto from "node:crypto";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import JSZip from "jszip";
import officeParser from "officeparser";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import Tesseract from "tesseract.js";

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,
  SERVER_API_KEY,
  MAX_FILE_SIZE_MB = "50",
  REQUEST_TIMEOUT_MS = "120000",
  PORT = process.env.PORT || 10000
} = process.env;

if (!DROPBOX_APP_KEY || !DROPBOX_APP_SECRET || !DROPBOX_REFRESH_TOKEN) {
  console.error("Missing: DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN");
  process.exit(1);
}

const MAX_FILE_SIZE = parseInt(MAX_FILE_SIZE_MB) * 1024 * 1024;
const REQUEST_TIMEOUT = parseInt(REQUEST_TIMEOUT_MS);
let _accessToken = null;

const TOKEN_URL = "https://api.dropboxapi.com/oauth2/token";
const DBX_RPC = "https://api.dropboxapi.com/2";
const DBX_CONTENT = "https://content.dropboxapi.com/2";

/* ---------- Auth ---------- */
async function refreshAccessToken() {
  const r = await axios.post(
    TOKEN_URL,
    new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: DROPBOX_REFRESH_TOKEN,
      client_id: DROPBOX_APP_KEY,
      client_secret: DROPBOX_APP_SECRET
    }),
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
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
const sha1 = (buf) => crypto.createHash("sha1").update(buf).digest("hex");
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
      { headers: { Authorization: `Bearer ${token}` }, timeout: REQUEST_TIMEOUT }
    )
  );
const dbxListContinue = (cursor) =>
  withAuth((token) =>
    axios.post(
      `${DBX_RPC}/files/list_folder/continue`,
      { cursor },
      { headers: { Authorization: `Bearer ${token}` }, timeout: REQUEST_TIMEOUT }
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
const dbxDownload = async ({ path }) =>
  withAuth(async (token) => {
    const metaResponse = await dbxGetMetadata(path);
    const fileSize = metaResponse.data.size || 0;
    if (fileSize > MAX_FILE_SIZE) throw new Error(`File too large: ${(fileSize / 1024 / 1024).toFixed(2)} MB`);
    const headers = { Authorization: `Bearer ${token}`, "Dropbox-API-Arg": JSON.stringify({ path }) };
    const r = await axios.post(`${DBX_CONTENT}/files/download`, null, {
      headers, responseType: "arraybuffer", timeout: REQUEST_TIMEOUT, maxContentLength: MAX_FILE_SIZE
    });
    return { ok: true, status: r.status, headers: { "content-type": r.headers["content-type"] }, data: Buffer.from(r.data) };
  });

/* ---------- Extractors (optional) ---------- */
async function extractPdf(buf, path) {
  try {
    const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise;
    let text = "";
    const maxPages = Math.min(pdf.numPages, 5);
    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(item => item.str).join(" ") + "\n";
    }
    return { text, note: `Extracted ${maxPages} pages` };
  } catch (err) {
    console.error("PDF parse failed, OCR fallback...");
    const { data: { text } } = await Tesseract.recognize(buf, "eng");
    return { text, note: "OCR fallback" };
  }
}
async function extractDocx(buf) {
  try {
    const result = await mammoth.extractRawText({ buffer: buf });
    return { text: result.value, note: "Extracted with mammoth" };
  } catch { return { text: "", note: "DOCX parse failed" }; }
}
async function extractXlsx(buf) {
  try {
    const wb = XLSX.read(buf, { type: "buffer" });
    return { text: wb.SheetNames.map(name => `--- ${name} ---\n${XLSX.utils.sheet_to_csv(wb.Sheets[name])}`).join("\n"), note: "Extracted with xlsx" };
  } catch { return { text: "", note: "XLSX parse failed" }; }
}
async function extractPptx(buf) {
  return new Promise((resolve) => {
    officeParser.parseOfficeAsync(buf, "pptx", (err, data) => {
      if (err) resolve({ text: "", note: "PPTX parse failed" });
      else resolve({ text: data, note: "Extracted with officeparser" });
    });
  });
}
async function extractText(path, buf) {
  if (/\.pdf$/i.test(path)) return extractPdf(buf, path);
  if (/\.docx$/i.test(path)) return extractDocx(buf);
  if (/\.xlsx$/i.test(path)) return extractXlsx(buf);
  if (/\.pptx$/i.test(path)) return extractPptx(buf);
  if (/\.(txt|csv|json|ya?ml|md|log)$/i.test(path)) return { text: buf.toString("utf8"), note: "Plain text" };
  return { text: "", note: "Unsupported type" };
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json());
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  if (req.path.startsWith("/mcp")) {
    const key = req.headers["x-api-key"];
    if (key !== SERVER_API_KEY) return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

/* ---------- Core Routes ---------- */
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true, root: DBX_ROOT_PREFIX }));
app.post("/mcp/walk", async (req, res) => { try {
  const { path_prefix = DBX_ROOT_PREFIX, cursor } = req.body || {};
  let entries = [], next = cursor || null;
  if (!next) {
    const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
    entries = r.data.entries || []; next = r.data.has_more ? r.data.cursor : null;
  } else {
    const r = await dbxListContinue(next);
    entries = r.data.entries || []; next = r.data.has_more ? r.data.cursor : null;
  }
  res.json({ entries: normalizeEntries(entries), next_cursor: next, truncated: !!next });
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
app.get("/mcp/list", async (req, res) => { try {
  const path = req.query.path ? String(req.query.path) : DBX_ROOT_PREFIX;
  const r = await dbxListFolder({ path: normPath(path), recursive: false, limit: 2000 });
  res.json({ entries: normalizeEntries(r.data.entries || []), truncated: r.data.has_more });
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
app.get("/mcp/meta", async (req, res) => { try {
  const { path } = req.query; if (!path) return res.status(400).json({ error: "path required" });
  const r = await dbxGetMetadata(String(path)); res.json(r.data);
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
app.post("/mcp/get", async (req, res) => { try {
  const { path } = req.body || {}; if (!path) return res.status(400).json({ error: "path required" });
  const r = await dbxDownload({ path: String(path) });
  res.json({ ok: true, path, size_bytes: r.data.length, data_base64: r.data.toString("base64") });
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
app.post("/mcp/head", async (req, res) => { try {
  const { path, bytes = 200000 } = req.body || {};
  if (!path) return res.status(400).json({ error: "path required" });
  const r = await dbxDownload({ path: String(path) });
  const text = r.data.toString("utf8", 0, bytes);
  res.json({ ok: true, path, size_bytes: r.data.length, text });
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
app.post("/mcp/open", async (req, res) => { try {
  const { path, extract_text = false } = req.body || {};
  if (!path) return res.status(400).json({ error: "path required" });
  const dl = await dbxDownload({ path });
  const response = { ok: true, path, size_bytes: dl.data.length, checksum: sha1(dl.data) };
  if (extract_text) {
    const { text, note } = await extractText(path, dl.data);
    response.text = text.length > 100000 ? text.slice(0, 100000) + "\n[TRUNCATED]" : text;
    response.note = note;
  }
  res.json(response);
} catch (e) { res.status(502).json({ ok: false, message: e.message }); }});
// Dedicated searchIndex route (alias for GPT instructions/schema)
app.post("/searchIndex", async (req, res) => {
  try {
    const { query, path_prefix = DBX_ROOT_PREFIX, limit = 50 } = req.body || {};
    if (!query) return res.status(400).json({ error: "query required" });

    console.log("searchIndex query:", query, "path_prefix:", path_prefix);

    // Walk folder recursively
    const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
    const entries = normalizeEntries(r.data.entries || []);

    // Case-insensitive match on name or path
    const qq = String(query).toLowerCase();
    const hits = entries.filter(e =>
      (e.name || "").toLowerCase().includes(qq) ||
      (e.path || "").toLowerCase().includes(qq)
    );

    const results = hits.slice(0, parseInt(limit));
    res.json({ query, total_matches: hits.length, results });
  } catch (e) {
    console.error("searchIndex ERROR:", e?.message || e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

/* ---------- Orchestrator ---------- */
app.post("/routeThenAnswer", async (req, res) => {
  try {
    const { query, context = {} } = req.body || {};
    if (!query) return res.status(400).json({ error: "query required" });
    const q = query.trim().toLowerCase();
    let intent = "Browse";
    if (/(search|find)/.test(q)) intent = "Search";
    else if (/(open|preview)/.test(q)) intent = "Preview";
    else if (/(download|get file)/.test(q)) intent = "Download";
    else if (/(forecast|demand|lead time)/.test(q)) intent = "Forecasting_Policy";
    else if (/(root cause|why|driver|rca)/.test(q)) intent = "Root_Cause";
    else if (/(compare|delta|shift)/.test(q)) intent = "Comparison";

    let text = "Unsupported intent", artifacts = [];
    if (intent === "Browse") {
      const r = await dbxListFolder({ path: normPath(DBX_ROOT_PREFIX), recursive: false, limit: 50 });
      const entries = normalizeEntries(r.data.entries || []);
      text = `Top entries:\n` + entries.map(e => `• ${e.name}`).join("\n"); artifacts = entries;
    } else if (intent === "Search") {
      const term = q.replace(/.*(search|find)\s+/, "");
      const r = await dbxListFolder({ path: normPath(DBX_ROOT_PREFIX), recursive: true, limit: 2000 });
      const hits = normalizeEntries(r.data.entries || []).filter(e => e.name.toLowerCase().includes(term));
      text = hits.length ? `Found ${hits.length} match(es).` : `No matches.`; artifacts = hits.slice(0, 50);
    } else if (intent === "Preview") {
      text = 'Use /mcp/head or /mcp/open with path for preview.';
    } else if (intent === "Download") {
      text = 'Use /mcp/get with path for full download.';
    } else if (["Forecasting_Policy","Root_Cause","Comparison"].includes(intent)) {
      text = `Intent detected = ${intent}, but skill not wired yet.`;
    }
    res.json({ text, answer: { intent, artifacts } });
  } catch (e) {
    res.status(500).json({ error: "router_error", message: e.message });
  }
});

/* ---------- Start ---------- */
app.listen(PORT, () => console.log(`DBX REST on ${PORT} root=${DBX_ROOT_PREFIX}`));

