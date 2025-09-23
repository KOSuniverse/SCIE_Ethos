// mcp_sse/index.js
import axios from "axios";
import crypto from "node:crypto";
import mammoth from "mammoth";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

/* =========================
   CONFIG
   ========================= */
export const DBX_ROOT_PREFIX = process.env.DBX_ROOT_PREFIX || "/Project_Root/GPT_Files";
const INDEX_ROOT = `${DBX_ROOT_PREFIX}/Indexes`;   // canonical folder
const JSONL_PATH = `${INDEX_ROOT}/file_index.jsonl`;
const CSV_PATH   = `${INDEX_ROOT}/file_index.csv`;
const MANIFEST   = `${INDEX_ROOT}/manifest.json`;

const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP     = /\.zip$/i;
const XLSX    = /\.xlsx$/i;
const IS_PDF  = /\.pdf$/i;
const IS_DOCX = /\.docx$/i;

const sha1 = buf => crypto.createHash("sha1").update(buf).digest("hex");
const rel  = p => (p?.startsWith("/") ? p.slice(1) : p);
const normUserPath = p => (p || "").replace(/^\/+/, "").toLowerCase();
const absPath = p => (p?.startsWith("/") ? p : "/" + p);

/* =========================
   Dropbox Auth (refresh only)
   ========================= */
let cachedAccessToken = null;
let cachedExpEpoch = 0;

async function getAccessToken(force=false) {
  const REFRESH = process.env.DROPBOX_REFRESH_TOKEN;
  const KEY     = process.env.DROPBOX_APP_KEY;
  const SECRET  = process.env.DROPBOX_APP_SECRET;
  if (!REFRESH || !KEY || !SECRET) throw new Error("Missing Dropbox refresh creds");

  const now = Math.floor(Date.now() / 1000);
  if (!force && cachedAccessToken && now < (cachedExpEpoch - 60)) return cachedAccessToken;

  const params = new URLSearchParams({ grant_type: "refresh_token", refresh_token: REFRESH }).toString();
  const basic  = Buffer.from(`${KEY}:${SECRET}`).toString("base64");
  const r = await axios.post("https://api.dropbox.com/oauth2/token", params, {
    headers: { "Content-Type": "application/x-www-form-urlencoded", "Authorization": `Basic ${basic}` }
  });

  cachedAccessToken = r.data.access_token;
  cachedExpEpoch    = now + (r.data.expires_in || 14400);
  return cachedAccessToken;
}
const authHeaders = async () => ({ Authorization: `Bearer ${await getAccessToken()}` });

/* =========================
   Dropbox Helpers
   ========================= */
const DBX_API  = "https://api.dropboxapi.com/2";
const DBX_CONT = "https://content.dropboxapi.com/2";

async function dbxWriteBytes(path, bytes){
  const args = { path: absPath(path), mode:{".tag":"overwrite"}, mute:true };
  await axios.post(`${DBX_CONT}/files/upload`, bytes, {
    headers:{ ...(await authHeaders()), "Content-Type":"application/octet-stream", "Dropbox-API-Arg": JSON.stringify(args) }
  });
}
const dbxWriteText = (p,t)=>dbxWriteBytes(p, Buffer.from(t,"utf8"));

async function dbxReadBytes(path){
  const args = { path: absPath(path) };
  const r = await axios.post(`${DBX_CONT}/files/download`, null, {
    responseType:"arraybuffer",
    headers:{ ...(await authHeaders()), "Dropbox-API-Arg": JSON.stringify(args), "Content-Type":"text/plain" }
  });
  return Buffer.from(r.data);
}

async function dbxExists(path){
  try {
    await axios.post(`${DBX_API}/files/get_metadata`, { path: absPath(path) }, { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
    return true;
  } catch { return false; }
}
async function dbxCreateFolderIfMissing(path){
  if (await dbxExists(path)) return;
  await axios.post(`${DBX_API}/files/create_folder_v2`, { path: absPath(path), autorename:false }, { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxListFolder({ path, recursive=true, limit=2000 }){
  return axios.post(`${DBX_API}/files/list_folder`,
    { path: absPath(path), recursive, limit, include_non_downloadable_files:false },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxListContinue(cursor){
  return axios.post(`${DBX_API}/files/list_folder/continue`, { cursor },
    { headers:{ ...(await authHeaders()), "Content-Type":"application/json" } });
}
async function dbxTempLink(path){
  const { data } = await axios.post(`${DBX_API}/files/get_temporary_link`, { path: absPath(path) }, {
    headers:{ ...(await authHeaders()), "Content-Type":"application/json" }
  });
  return data?.link || null;
}
const httpRange = (link, start, end)=>
  axios.get(link, { responseType:"arraybuffer", headers:{ Range:`bytes=${start}-${end}` } });

/* =========================
   Extractors (pdfjs-dist, mammoth)
   ========================= */
async function fetchBytesForExtract(dbxPath, cap=8*1024*1024){
  const link = await dbxTempLink(dbxPath);
  if (!link) return null;
  const r = await axios.get(link, { responseType:"arraybuffer" });
  const buf = Buffer.from(r.data);
  return buf.length > cap ? buf.slice(0, cap) : buf;
}
async function extractPdfText(buf, pageLimit=20, charCap=120000){
  const pdf = await getDocument({ data: buf }).promise;
  let text = "";
  for (let i=1;i<=Math.min(pdf.numPages,pageLimit);i++){
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map(it=>it.str||"").join(" ") + "\n";
    if (text.length >= charCap) break;
  }
  return text.trim();
}
async function extractDocxText(buf){
  try { const out = await mammoth.extractRawText({ buffer: buf }); return (out.value||"").trim(); }
  catch { return ""; }
}

/* =========================
   Index Scaffolding
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
    await dbxWriteText(MANIFEST, JSON.stringify({ createdAt:new Date().toISOString(), csv_header_written:true }, null, 2));
  }
}

/* =========================
   Routes
   ========================= */
export function registerRoutes(app){
  app.get("/mcp/healthz", (_req,res)=>res.json({ ok:true, root: DBX_ROOT_PREFIX }));

  app.post("/mcp/index_full", async (req,res)=>{
    try {
      const { path_prefix = DBX_ROOT_PREFIX } = req.body || {};
      const r = await dbxListFolder({ path: path_prefix });
      const entries = r.data.entries||[];
      res.json({ ok:true, entries: entries.length, sample: entries.slice(0,3) });
    } catch(e){
      res.status(502).json({ ok:false, message:e?.message||"INDEX_ERROR", data:e?.response?.data||null });
    }
  });
}


