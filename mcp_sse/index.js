import crypto from "node:crypto";
import axios from "axios";

/* =========================
   CONFIG
   ========================= */
export const DBX_ROOT_PREFIX = process.env.DBX_ROOT_PREFIX || "/Project_Root";
const INDEX_ROOT = `${DBX_ROOT_PREFIX}/GPT_Files/indexes`;
const JSONL_PATH = `${INDEX_ROOT}/file_index.jsonl`;
const CSV_PATH   = `${INDEX_ROOT}/file_index.csv`;
const MANIFEST   = `${INDEX_ROOT}/manifest.json`;

const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP     = /\.zip$/i;
const XLSX    = /\.xlsx$/i;

function sha1(buf){ return crypto.createHash("sha1").update(buf).digest("hex"); }
const rel = p => (p?.startsWith("/") ? p.slice(1) : p);

/* =========================
   Dropbox Auth — uses your Render env
   Priority:
   1) Refresh flow (DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET)
   2) Static token (DROPBOX_ACCESS_TOKEN) if you ever add it
   ========================= */
let cachedAccessToken = null;
let cachedExpEpoch = 0;

async function getAccessToken() {
  if (process.env.DROPBOX_ACCESS_TOKEN) return process.env.DROPBOX_ACCESS_TOKEN;

  const REFRESH = process.env.DROPBOX_REFRESH_TOKEN;
  const KEY     = process.env.DROPBOX_APP_KEY;
  const SECRET  = process.env.DROPBOX_APP_SECRET;
  if (!REFRESH || !KEY || !SECRET) {
    throw new Error("Dropbox auth missing. Set DROPBOX_REFRESH_TOKEN, DROPBOX_APP_KEY, DROPBOX_APP_SECRET (or DROPBOX_ACCESS_TOKEN).");
  }

  const now = Math.floor(Date.now() / 1000);
  if (cachedAccessToken && now < (cachedExpEpoch - 60)) return cachedAccessToken;

  const params = new URLSearchParams({
    grant_type: "refresh_token",
    refresh_token: REFRESH
  }).toString();
  const basic = Buffer.from(`${KEY}:${SECRET}`).toString("base64");

  const r = await axios.post(
    "https://api.dropbox.com/oauth2/token",
    params,
    { headers: { "Content-Type": "application/x-www-form-urlencoded", "Authorization": `Basic ${basic}` } }
  );

  cachedAccessToken = r.data.access_token;
  cachedExpEpoch = Math.floor(Date.now() / 1000) + (r.data.expires_in || 14400);
  return cachedAccessToken;
}

async function authHeaders() {
  const t = await getAccessToken();
  return { Authorization: `Bearer ${t}` };
}

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

async function dbxWriteText(path, text){ return dbxWriteBytes(path, Buffer.from(text, "utf8")); }

async function dbxReadBytes(path){ // small files only
  const args = { path };
  const r = await axios.post(`${DBX_CONT}/files/download`, null, {
    responseType: "arraybuffer",
    headers:{ ...(await authHeaders()), "Dropbox-API-Arg": JSON.stringify(args) }
  });
  return Buffer.from(r.data);
}

// Simple append (read+append+overwrite). Fine for now; shard later if needed.
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

function normalizeEntries(entries){
  return (entries||[])
    .filter(e => e[".tag"] !== "deleted")
    .map(e => ({
      path: e.path_lower,
      name: e.name,
      mime: e[".tag"] === "folder" ? "folder" : "application/octet-stream",
      size: e.size ?? null,
      modified: e.server_modified ?? e.client_modified ?? null
    }));
}

// Range-friendly download for previews/checksums
async function dbxDownload({ path, rangeStart=0, rangeEnd=200000 }){
  const { data } = await axios.post(`${DBX_API}/files/get_temporary_link`, { path }, {
    headers:{ ...(await authHeaders()), "Content-Type":"application/json" }
  });
  const link = data?.link;
  if (!link) return { ok:false, data:null };
  const r = await axios.get(link, { responseType:"arraybuffer", headers:{ Range:`bytes=${rangeStart}-${rangeEnd}` }});
  const ok = (r.status === 206 || r.status === 200);
  return { ok, data: Buffer.from(r.data) };
}

/* =========================
   Heuristics
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
async function sniffXlsxSheetNames(){ return []; } // upgrade later

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
  app.post("/mcp/index_full", async (req, res) => {
    try {
      const {
        path_prefix = DBX_ROOT_PREFIX,
        output_jsonl_path = JSONL_PATH,
        output_csv_path   = CSV_PATH,
        manifest_path     = MANIFEST,
        preview_bytes     = 200_000,
        checksum_bytes_limit = 1_000_000,
        resume = true
      } = req.body || {};

      const prev = resume ? await readCheckpoint(manifest_path) : null;
      let total = 0, wrote = 0;

      // Walk Dropbox
      let cursor = null, raw = [];
      { const r = await dbxListFolder({ path: path_prefix, recursive: true, limit: 2000 });
        const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null; }
      while (cursor) {
        const r = await dbxListContinue(cursor);
        const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null;
        if (raw.length > 300_000) break; // safety cap
      }

      const entries = normalizeEntries(raw)
        .filter(e => e.path && e.mime !== "folder")
        .filter(e => !ZIP.test(e.path));

      total = entries.length;

      // Ensure CSV header exists (idempotent)
      if (!prev || !prev.csv_header_written) {
        await dbxWriteText(output_csv_path,
          "site,country,plant_code,category,subcategory,month,file_path,as_of_date,currency,scale,canonical,has_totals,is_complete,checksum_sha1,mime,size,modified\n"
        );
      }

      for (const e of entries) {
        const filePath = e.path;
        const name = e.name || "";

        const month = parseMonthFromName(name) || parseMonthFromName(filePath) || null;
        const { category, subcategory } = guessCategoryBits(filePath);
        const { site, country, plant_code } = guessSiteCountryPlant(filePath);
        const currency = guessCurrency(filePath) || guessCurrency(name);
        const as_of_date = e.modified || null;

        // Preview & checksum (small/safe)
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
          sheet_names = await sniffXlsxSheetNames(); // placeholder pass
        } else {
          if (typeof e.size === "number" && e.size <= checksum_bytes_limit) {
            const dl = await dbxDownload({ path: filePath, rangeStart: 0, rangeEnd: Math.max(0, e.size - 1) });
            if (dl.ok) content_hash = sha1(dl.data);
          }
        }

        const row = {
          site, country, plant_code, category, subcategory, month,
          file_path: rel(filePath), as_of_date, currency,
          scale: 1, canonical: true, has_totals: true, is_complete: true,
          checksum_sha1: content_hash, mime: e.mime, size: e.size, modified: e.modified,
          sheet_names, text_preview, truncated
        };

        // Append JSONL & CSV (read+append+overwrite pattern)
        await dbxAppendText(output_jsonl_path, JSON.stringify(row) + "\n");

        const csvLine = [
          row.site, row.country, row.plant_code, row.category, row.subcategory, row.month,
          row.file_path, row.as_of_date, row.currency, row.scale, row.canonical, row.has_totals, row.is_complete,
          row.checksum_sha1, row.mime, row.size, row.modified
        ].map(v => `"${(v ?? "").toString().replace(/"/g,'""')}"`).join(",") + "\n";
        await dbxAppendText(output_csv_path, csvLine);

        wrote++;
        if (wrote % 200 === 0) {
          await dbxWriteText(manifest_path, JSON.stringify({
            path_prefix, total, processed: wrote, last_path: filePath, csv_header_written: true
          }, null, 2));
        }
      }

      await dbxWriteText(manifest_path, JSON.stringify({
        path_prefix, total, processed: wrote, csv_header_written: true, completed: true
      }, null, 2));

      res.json({ ok: true, stats: { total, processed: wrote }, outputs: { output_jsonl_path, output_csv_path, manifest_path }});
    } catch (e) {
      res.status(502).json({ ok:false, message: e?.message || "INDEX_ERROR", data: e?.response?.data || null });
    }
  });
}

