// mcp_sse/index.js  (ESM)
import axios from "axios";
import crypto from "node:crypto";
import mammoth from "mammoth";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

/* =========================
   CONFIG
   ========================= */
// Default root matches your healthz. Override via Render env if needed.
export const DBX_ROOT_PREFIX = process.env.DBX_ROOT_PREFIX || "/Project_Root/GPT_Files";

// All index artifacts live here in Dropbox (cloud-only).
const INDEX_ROOT = `${DBX_ROOT_PREFIX}/indexes`;
const JSONL_PATH = `${INDEX_ROOT}/file_index.jsonl`;
const CSV_PATH   = `${INDEX_ROOT}/file_index.csv`;
const MANIFEST   = `${INDEX_ROOT}/manifest.json`;

const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP     = /\.zip$/i;
const XLSX    = /\.xlsx$/i;
const IS_PDF  = /\.pdf$/i;
const IS_DOCX = /\.docx$/i;

const MAX_BYTES_DEFAULT  = 128_000;         // for byte range responses
const MAX_EXTRACT_BYTES  = 8 * 1024 * 1024; // cap bytes fed to extractors (8MB)
const PREVIEW_CHAR_CAP   = 12_000;          // store up to 12k chars in index

const sha1 = buf => crypto.createHash("sha1").update(buf).digest("hex");
const rel  = p => (p?.startsWith("/") ? p.slice(1) : p);
const normUserPath = p => (p || "").replace(/^\/+/, "").toLowerCase();

/* =========================
   Dropbox Auth (refresh first, else static)
   ========================= */
let cachedAccessToken = null, cachedExpEpoch = 0;

async function getAccessToken() {
  if (process.env.DROPBOX_ACCESS_TOKEN) return process.env.DROPBOX_ACCESS_TOKEN;

  const REFRESH = process.env.DROPBOX_REFRESH_TOKEN;
  const KEY     = process.env.DROPBOX_APP_KEY;
  const SECRET  = process.env.DROPBOX_APP_SECRET;
  if (!REFRESH || !KEY || !SECRET) throw new Error("Missing Dropbox auth vars.");

  const now = Math.floor(Date.now() / 1000);
  if (cachedAccessToken && now < (cachedExpEpoch - 60)) return cachedAccessToken;

  const params = new URLSearchParams({ grant_type: "refresh_token", refresh_token: REFRESH }).toString();
  const basic  = Buffer.from(`${KEY}:${SECRET}`).toString("base64");
  const r = await axios.post("https://api.dropbox.com/oauth2/token", params, {
    headers: { "Content-Type": "application/x-www-form-urlencoded", "Authorization": `Basic ${basic}` }
  });

  cachedAccessToken = r.data.access_token;
  cachedExpEpoch    = Math.floor(Date.now() / 1000) + (r.data.expires_in || 14400);
  return cachedAccessToken;
}
const authHeaders = async () => ({ Authorization: `Bearer ${await getAccessToken()}` });

/* =========================
   Dropbox REST helpers (cloud-only)
   ========================= */
const DBX_API  = "https://api.dropboxapi.com/2";
const DBX_CONT = "https://content.dropboxapi.com/2";

async function dbxExists(path){
  try {
    await axios.post(`${DBX_API}/files/get_metadata`, { path }, { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
    return true;
  } catch { return false; }
}
async function dbxCreateFolderIfMissing(path){
  if (await dbxExists(path)) return;
  await axios.post(`${DBX_API}/files/create_folder_v2`, { path, autorename:false },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxWriteBytes(path, bytes){ // overwrite
  const args = { path, mode:{".tag":"overwrite"}, mute:true };
  await axios.post(`${DBX_CONT}/files/upload`, bytes, {
    headers:{ ...(await authHeaders()), "Content-Type":"application/octet-stream", "Dropbox-API-Arg": JSON.stringify(args) }
  });
}
const dbxWriteText = (p,t)=>dbxWriteBytes(p, Buffer.from(t, "utf8"));

async function dbxReadBytes(path){ // small files only (manifest, csv)
  const args = { path };
  const r = await axios.post(`${DBX_CONT}/files/download`, null, {
    responseType: "arraybuffer",
    headers:{
      ...(await authHeaders()),
      "Dropbox-API-Arg": JSON.stringify(args),
      "Content-Type": "text/plain; charset=utf-8" // prevent x-www-form-urlencoded
    }
  });
  return Buffer.from(r.data);
}

// naive append (read+append+overwrite). We batch upstream to keep it efficient.
async function dbxAppendText(path, text){
  const add = Buffer.from(text, "utf8");
  let cur = Buffer.alloc(0);
  if (await dbxExists(path)) cur = await dbxReadBytes(path);
  await dbxWriteBytes(path, Buffer.concat([cur, add]));
}

async function dbxListFolder({ path, recursive=true, limit=2000 }){
  return axios.post(`${DBX_API}/files/list_folder`,
    { path, recursive, limit, include_non_downloadable_files:false },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxListContinue(cursor){
  return axios.post(`${DBX_API}/files/list_folder/continue`, { cursor },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxGetMetadata(path){
  const r = await axios.post(`${DBX_API}/files/get_metadata`, { path },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
  return r.data;
}
function normalizeEntries(entries){
  return (entries||[])
    .filter(e => e[".tag"] !== "deleted")
    .map(e => ({
      tag: e[".tag"],
      path: e.path_lower, // use lowercased path for stable matching
      name: e.name,
      mime: e[".tag"] === "folder" ? "folder" : "application/octet-stream",
      size: e.size ?? null,
      modified: e.server_modified ?? e.client_modified ?? null
    }));
}
async function dbxTempLink(path){
  const { data } = await axios.post(`${DBX_API}/files/get_temporary_link`, { path }, {
    headers:{ ...(await authHeaders()), "Content-Type":"application/json" }
  });
  return data?.link || null;
}
const httpRange = (link, start, end) =>
  axios.get(link, { responseType: "arraybuffer", headers: { Range: `bytes=${start}-${end}` } });

/* =========================
   Extractors (PDF via pdfjs-dist, DOCX via mammoth)
   ========================= */
async function fetchBytesForExtract(dbxLowerPath, cap=MAX_EXTRACT_BYTES){
  const link = await dbxTempLink(dbxLowerPath);
  if (!link) return null;
  const r = await axios.get(link, { responseType: "arraybuffer" });
  const buf = Buffer.from(r.data);
  return buf.length > cap ? buf.slice(0, cap) : buf;
}

async function extractPdfText(buf, pageLimit=20, charCap=120000){
  // pdfjs-dist works in Node (no worker needed with legacy build)
  const loadingTask = getDocument({ data: buf });
  const pdf = await loadingTask.promise;
  let text = "";
  const maxPages = Math.min(pdf.numPages, pageLimit);
  for (let i = 1; i <= maxPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(it => (it.str || "")).join(" ") + "\n";
    if (text.length >= charCap) break;
  }
  return text.trim();
}
async function extractDocxText(buf){
  try {
    const out = await mammoth.extractRawText({ buffer: buf });
    return (out.value || "").trim();
  } catch { return ""; }
}

/* =========================
   Heuristics (light)
   ========================= */
function parseMonthFromName(name){
  const m1 = name?.match(/\b(20\d{2})[-_ ]?(0[1-9]|1[0-2])\b/); if (m1) return `${m1[1]}-${m1[2]}`;
  const M={jan:"01",feb:"02",mar:"03",apr:"04",may:"05",jun:"06",jul:"07",aug:"08",sep:"09",oct:"10",nov:"11",dec:"12"};
  const m2 = name?.toLowerCase().match(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_ ]?(\d{2,4})\b/);
  if (m2){ const mm=M[m2[1]]; const yy=m2[2].length===2?`20${m2[2]}`:m2[2]; return `${yy}-${mm}`; }
  return null;
}
function guessCurrency(s){ if(/\bUSD\b|US\$|\$/.test(s)) return "USD"; if(/\bEUR\b|€/.test(s)) return "EUR"; if(/\bGBP\b|£/.test(s)) return "GBP"; return null; }
function guessCategoryBits(p){ const s=(p||"").toLowerCase(); let category=null, sub=null;
  if(s.includes("wip")) category="WIP"; if(s.includes("inventory")) category=category||"Inventory";
  if(s.includes("aged")) sub="Aged"; if(s.includes("finished")) sub=sub||"Finished"; if(s.includes("provision")) sub=sub||"Provision";
  return { category, subcategory: sub };
}
function guessSiteCountryPlant(p){
  const out={site:null,country:null,plant_code:null};
  for (const seg of (p||"").split("/")){
    const m=seg.match(/\b([A-Z]{2,4})\s+(R?\d{3,4})\b/); if(m){ out.site=m[1]; out.plant_code=m[2]; }
    if(/italy|torino/i.test(seg)) out.country="ITA";
    if(/germany|gmbh|deutschland|muelheim/i.test(seg)) out.country="DEU";
    if(/thailand|rayong/i.test(seg)) out.country="THA";
    if(/united\s?states|usa|houston|greenville|pps/i.test(seg)) out.country="USA";
    if(/uk|aberdeen/i.test(seg)) out.country="GBR";
  }
  return out;
}
async function readCheckpoint(cpPath){
  try { const buf = await dbxReadBytes(cpPath); return JSON.parse(buf.toString("utf8")); } catch { return null; }
}

/* =========================
   Public: scaffold + routes
   ========================= */
export async function ensureIndexScaffold(){
  await dbxCreateFolderIfMissing(INDEX_ROOT);
  if (!(await dbxExists(CSV_PATH))) {
    await dbxWriteText(CSV_PATH,
      "site,country,plant_code,category,subcategory,month,file_path,as_of_date,currency,scale,canonical,has_totals,is_complete,checksum_sha1,mime,size,modified\n"
    );
  }
  if (!(await dbxExists(JSONL_PATH))) await dbxWriteText(JSONL_PATH, "");
  if (!(await dbxExists(MANIFEST))) {
    await dbxWriteText(MANIFEST, JSON.stringify({ createdAt: new Date().toISOString(), csv_header_written: true }, null, 2));
  }
}

export function registerRoutes(app){

  /* Health */
  app.get("/mcp/healthz", (_req, res) => res.json({ ok:true, root: DBX_ROOT_PREFIX }));

  /* INDEX: build/refresh (batched writes) */
  app.post("/mcp/index_full", async (req, res) => {
    try {
      const {
        path_prefix = DBX_ROOT_PREFIX,
        output_jsonl_path = JSONL_PATH,
        output_csv_path   = CSV_PATH,
        manifest_path     = MANIFEST,
        preview_bytes     = 200_000,
        checksum_bytes_limit = 1_000_000,
        resume = true,
        batch_size = 400
      } = req.body || {};

      const prev = resume ? await readCheckpoint(manifest_path) : null;
      let total = 0, wrote = 0;

      // Walk
      let cursor = null, raw = [];
      { const r = await dbxListFolder({ path: path_prefix, recursive: true, limit: 2000 });
        const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null; }
      while (cursor) {
        const r = await dbxListContinue(cursor);
        const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null;
        if (raw.length > 300_000) break;
      }

      const entries = normalizeEntries(raw)
        .filter(e => e.path && e.mime !== "folder")
        .filter(e => !ZIP.test(e.path));

      total = entries.length;

      if (!prev || !prev.csv_header_written) {
        await dbxWriteText(output_csv_path,
          "site,country,plant_code,category,subcategory,month,file_path,as_of_date,currency,scale,canonical,has_totals,is_complete,checksum_sha1,mime,size,modified\n"
        );
      }

      let jsonlBuf = "", csvBuf = "";
      async function flush() {
        if (jsonlBuf) { await dbxAppendText(output_jsonl_path, jsonlBuf); jsonlBuf=""; }
        if (csvBuf)   { await dbxAppendText(output_csv_path,   csvBuf);   csvBuf=""; }
        await dbxWriteText(manifest_path, JSON.stringify({
          path_prefix, total, processed: wrote, csv_header_written: true, last_flush_at: new Date().toISOString()
        }, null, 2));
      }

      for (const e of entries) {
        const filePath = e.path; // path_lower
        const name = e.name || "";

        const month = parseMonthFromName(name) || parseMonthFromName(filePath) || null;
        const { category, subcategory } = guessCategoryBits(filePath);
        const { site, country, plant_code } = guessSiteCountryPlant(filePath);
        const currency = guessCurrency(filePath) || guessCurrency(name);
        const as_of_date = e.modified || null;

        let text_preview = null, truncated = false, content_hash = "";
        let sheet_names = [];

        if (TEXTISH.test(name)) {
          const link = await dbxTempLink(filePath);
          if (link) {
            const r = await httpRange(link, 0, Math.max(0, preview_bytes));
            const buf = Buffer.from(r.data);
            text_preview = buf.toString("utf8");
            truncated = buf.length >= preview_bytes;
            if (buf.length <= checksum_bytes_limit) content_hash = sha1(buf);
          }
        } else if (IS_PDF.test(name)) {
          const buf = await fetchBytesForExtract(filePath);
          if (buf) {
            const text = await extractPdfText(buf);
            text_preview = text ? text.slice(0, PREVIEW_CHAR_CAP) : "";
            truncated = !!text && text.length > PREVIEW_CHAR_CAP;
            if (buf.length <= checksum_bytes_limit) content_hash = sha1(buf);
          }
        } else if (IS_DOCX.test(name)) {
          const buf = await fetchBytesForExtract(filePath);
          if (buf) {
            const text = await extractDocxText(buf);
            text_preview = text ? text.slice(0, PREVIEW_CHAR_CAP) : "";
            truncated = !!text && text.length > PREVIEW_CHAR_CAP;
            if (buf.length <= checksum_bytes_limit) content_hash = sha1(buf);
          }
        } else if (XLSX.test(name)) {
          // XLSX support to be added next
        } else {
          if (typeof e.size === "number" && e.size <= checksum_bytes_limit) {
            const link = await dbxTempLink(filePath);
            if (link) {
              const r0 = await httpRange(link, 0, Math.max(0, e.size - 1));
              content_hash = sha1(Buffer.from(r0.data));
            }
          }
        }

        const row = {
          site, country, plant_code, category, subcategory, month,
          file_path: rel(filePath), as_of_date, currency,
          scale: 1, canonical: true, has_totals: true, is_complete: true,
          checksum_sha1: content_hash, mime: e.mime, size: e.size, modified: e.modified,
          sheet_names, text_preview, truncated
        };

        jsonlBuf += JSON.stringify(row) + "\n";
        csvBuf   += [
          row.site, row.country, row.plant_code, row.category, row.subcategory, row.month,
          row.file_path, row.as_of_date, row.currency, row.scale, row.canonical, row.has_totals, row.is_complete,
          row.checksum_sha1, row.mime, row.size, row.modified
        ].map(v => `"${(v ?? "").toString().replace(/"/g,'""')}"`).join(",") + "\n";

        wrote++;
        if (wrote % batch_size === 0) await flush();
      }

      await flush();
      await dbxWriteText(manifest_path, JSON.stringify({
        path_prefix, total, processed: wrote, csv_header_written: true, completed: true, completed_at: new Date().toISOString()
      }, null, 2));

      res.json({ ok:true, stats:{ total, processed:wrote, batch_size }, outputs:{ output_jsonl_path, output_csv_path, manifest_path }});
    } catch (e) {
      res.status(502).json({ ok:false, message:e?.message || "INDEX_ERROR", data:e?.response?.data || null });
    }
  });

  /* SEARCH: name + preview */
  app.get("/mcp/search", async (req, res) => {
    try {
      const q = (req.query.q || "").toString().trim();
      const k = Math.max(1, Math.min(50, parseInt(req.query.k || "20", 10)));
      if (!q) return res.json({ ok:true, hits:[] });

      const buf = await dbxReadBytes(JSONL_PATH);
      const lines = buf.toString("utf8").split(/\r?\n/).filter(Boolean);

      const lcq = q.toLowerCase();
      const scored = [];
      for (const line of lines) {
        try {
          const row = JSON.parse(line);
          const hayName = (row.file_path || "").toLowerCase();
          const hayPrev = (row.text_preview || "").toLowerCase();
          let score = 0;
          if (hayName.includes(lcq)) score += 2;
          if (hayPrev.includes(lcq)) score += 3;
          if (score > 0) {
            const snippet = row.text_preview ? row.text_preview.slice(0, 500) : null;
            scored.push({ score, path: row.file_path, mime: row.mime, size: row.size, modified: row.modified, snippet });
          }
        } catch {}
      }
      scored.sort((a,b)=>b.score-a.score);
      res.json({ ok:true, q, hits: scored.slice(0, k) });
    } catch (e) {
      res.status(502).json({ ok:false, message: e?.message || "SEARCH_ERROR" });
    }
  });

  /* BYTES (range) — back-compat /mcp/get */
  app.post("/mcp/get", async (req, res) => {
    try {
      const { path, range_start=0, range_end=null } = req.body || {};
      if (!path) return res.status(400).json({ ok:false, message:"path required" });
      const absLower = "/" + normUserPath(path);
      const link = await dbxTempLink(absLower);
      if (!link) return res.status(404).json({ ok:false, message:"TEMP_LINK_NOT_FOUND" });

      const MAX   = MAX_BYTES_DEFAULT;
      const start = Math.max(0, Number(range_start) || 0);
      const end   = (range_end != null ? Number(range_end) : (start + MAX - 1));
      const r = await httpRange(link, start, Math.min(end, start + MAX - 1));
      const body = Buffer.from(r.data);

      res.status(206).set({
        "Content-Type": "application/octet-stream",
        "Content-Range": r.headers["content-range"] || `bytes ${start}-${start+body.length-1}/*`,
        "X-Truncated": (end - start + 1) > MAX ? "true" : "false"
      }).send(body);
    } catch (e) {
      res.status(502).json({ ok:false, message: e?.message || "GET_ERROR" });
    }
  });

  /* TEXT from index preview */
  app.post("/mcp/get_text", async (req, res) => {
    try {
      const { path, max_chars=8000 } = req.body || {};
      if (!path) return res.status(400).json({ ok:false, message:"path required" });
      const want = normUserPath(path);

      const buf = await dbxReadBytes(JSONL_PATH);
      const lines = buf.toString("utf8").split(/\r?\n/).filter(Boolean);

      let row = null;
      for (const l of lines) {
        try {
          const r = JSON.parse(l);
          const stored = (r.file_path || "").toLowerCase();
          if (stored && (stored === want || want.endsWith(stored))) { row = r; break; }
        } catch {}
      }
      if (!row) return res.status(404).json({ ok:false, message:"NOT_INDEXED" });

      const text = (row.text_preview || "").slice(0, max_chars);
      res.json({ ok:true, path: row.file_path, text_excerpt: text, truncated: (row.text_preview||"").length > text.length });
    } catch (e) {
      res.status(502).json({ ok:false, message: e?.message || "GET_TEXT_ERROR" });
    }
  });

  /* LIST/META/WALK (schema) */
  app.get("/mcp/list", async (req, res) => {
    try {
      const path = (req.query.path || DBX_ROOT_PREFIX).toString();
      const r = await dbxListFolder({ path, recursive:false, limit:2000 });
      res.json({ ok:true, entries: normalizeEntries(r.data.entries||[]) });
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "LIST_ERROR" }); }
  });
  app.get("/mcp/meta", async (req, res) => {
    try {
      const path = req.query.path;
      if (!path) return res.status(400).json({ ok:false, message:"path required" });
      const meta = await dbxGetMetadata(path);
      res.json({ ok:true, meta });
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "META_ERROR" }); }
  });
  app.post("/mcp/walk", async (req, res) => {
    try {
      const { path_prefix = DBX_ROOT_PREFIX, max_items = 2000, cursor = null } = req.body || {};
      if (cursor) {
        const r = await dbxListContinue(cursor);
        return res.json({ ok:true, entries: normalizeEntries(r.data.entries||[]), cursor: r.data.has_more ? r.data.cursor : null });
      } else {
        const r = await dbxListFolder({ path: path_prefix, recursive:true, limit: Math.max(1, Math.min(2000, max_items)) });
        return res.json({ ok:true, entries: normalizeEntries(r.data.entries||[]), cursor: r.data.has_more ? r.data.cursor : null });
      }
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "WALK_ERROR" }); }
  });

  /* DEBUG */
  app.get("/mcp/index_stats", async (_req, res) => {
    try {
      const exists = await dbxExists(JSONL_PATH);
      if (!exists) return res.json({ ok:true, exists:false, lines:0 });
      const buf = await dbxReadBytes(JSONL_PATH);
      const lines = buf.toString("utf8").split(/\r?\n/).filter(Boolean).length;
      res.json({ ok:true, exists:true, jsonl: JSONL_PATH, lines });
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "INDEX_STATS_ERROR" }); }
  });
  app.get("/mcp/index_head", async (req, res) => {
    try {
      const n = Math.max(1, Math.min(200, parseInt(req.query.n || "20", 10)));
      const buf = await dbxReadBytes(JSONL_PATH);
      const lines = buf.toString("utf8").split(/\r?\n/).filter(Boolean).slice(0, n);
      res.json({ ok:true, lines: lines.map(l => JSON.parse(l)) });
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "INDEX_HEAD_ERROR" }); }
  });

  /* On-demand single-file extract (optional) */
  app.post("/mcp/extract", async (req, res) => {
    try {
      const { path, max_chars = PREVIEW_CHAR_CAP } = req.body || {};
      if (!path) return res.status(400).json({ ok:false, message:"path required" });
      const abs = "/" + normUserPath(path);
      let text = "";
      if (IS_PDF.test(abs))   { const b = await fetchBytesForExtract(abs); if (b) text = await extractPdfText(b); }
      else if (IS_DOCX.test(abs)) { const b = await fetchBytesForExtract(abs); if (b) text = await extractDocxText(b); }
      res.json({ ok:true, path: abs.slice(1), text_excerpt: (text||"").slice(0, max_chars), truncated: (text||"").length > max_chars });
    } catch (e) { res.status(502).json({ ok:false, message: e?.message || "EXTRACT_ERROR" }); }
  });
}

