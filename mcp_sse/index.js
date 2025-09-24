// index.js — Dropbox REST API + Indexer + OCR + Robust Extractors

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
  PORT = process.env.PORT || 10000
} = process.env;

if (!DROPBOX_APP_KEY || !DROPBOX_APP_SECRET || !DROPBOX_REFRESH_TOKEN) {
  console.error("Missing Dropbox env vars");
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
    { headers: { "Content-Type": "application/x-www-form-urlencoded" } }
  );
  _accessToken = r.data.access_token;
  console.log("REFRESH: new Dropbox token");
  return _accessToken;
}
async function withAuth(fn) {
  if (!_accessToken) await refreshAccessToken();
  try { return await fn(_accessToken); }
  catch (e) {
    const status = e?.response?.status;
    const body = e?.response?.data || "";
    if (status === 401 || String(body).includes("expired_access_token")) {
      await refreshAccessToken();
      return fn(_accessToken);
    }
    throw e;
  }
}

/* ---------- Helpers ---------- */
const normPath = (p) => (!p || p === "/" ? "" : (p.startsWith("/") ? p : "/" + p));
const normalizeEntries = (entries) =>
  (entries||[]).map(e=>({ path:e.path_display||e.path_lower, name:e.name, size:e.size??null,
    modified:e.server_modified??null, mime:e[".tag"]==="file"? e.mime_type??null:"folder"}));
const sha1 = buf => crypto.createHash("sha1").update(buf).digest("hex");

const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP = /\.zip$/i;
const XLSX_RE = /\.xlsx$/i;
const PDF_RE = /\.pdf$/i;
const DOCX_RE = /\.docx$/i;
const PPTX_RE = /\.pptx$/i;

/* ---------- Dropbox wrappers ---------- */
const dbxListFolder = ({ path, recursive, limit }) =>
  withAuth(token=>axios.post(`${DBX_RPC}/files/list_folder`,
    { path, recursive:!!recursive, include_deleted:false, limit },
    { headers:{ Authorization:`Bearer ${token}` } }));
const dbxListContinue = (cursor) =>
  withAuth(token=>axios.post(`${DBX_RPC}/files/list_folder/continue`,
    { cursor }, { headers:{ Authorization:`Bearer ${token}` } }));
const dbxGetMetadata = (path) =>
  withAuth(token=>axios.post(`${DBX_RPC}/files/get_metadata`,
    { path }, { headers:{ Authorization:`Bearer ${token}` } }));
const dbxDownload = async ({ path }) =>
  withAuth(async token=>{
    const headers = {
      Authorization:`Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };
    const r = await fetch(`${DBX_CONTENT}/files/download`, { method:"POST", headers });
    const ab = await r.arrayBuffer();
    return { ok:r.ok, status:r.status,
      headers:{ "content-type": r.headers.get("content-type") },
      data:Buffer.from(ab) };
  });

/* ---------- Extractors ---------- */
async function extractPdf(buf) {
  try {
    const data = new Uint8Array(buf);
    const pdf = await pdfjsLib.getDocument({ data }).promise;
    let text = "";
    for (let i = 1; i <= Math.min(pdf.numPages, 5); i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map(it => it.str).join(" ") + "\n";
    }
    if (text.trim().length > 0) return text;
  } catch {}
  const { data: { text } } = await Tesseract.recognize(buf, "eng");
  return text;
}
async function extractDocx(buf) {
  try {
    const result = await mammoth.extractRawText({ buffer: buf });
    if (result.value && result.value.trim().length > 0) return result.value;
  } catch {}
  try {
    const zip = await JSZip.loadAsync(buf);
    const docXml = await zip.file("word/document.xml").async("string");
    return docXml.replace(/<[^>]+>/g, " ");
  } catch { return ""; }
}
async function extractPptx(buf) {
  return new Promise((resolve, reject) => {
    officeParser.parseOfficeAsync(buf, "pptx", (err, data) => {
      if (err) reject(err);
      else resolve(data || "");
    });
  });
}
async function extractXlsx(buf) {
  const wb = XLSX.read(buf, { type:"buffer" });
  let out = [];
  wb.SheetNames.forEach(name=>{
    const sheet = wb.Sheets[name];
    const csv = XLSX.utils.sheet_to_csv(sheet, { header:1 });
    out.push(`--- Sheet: ${name} ---\n${csv}`);
  });
  return out.join("\n");
}
async function extractText(path, buf) {
  if (TEXTISH.test(path)) return buf.toString("utf8");
  if (PDF_RE.test(path)) return await extractPdf(buf);
  if (DOCX_RE.test(path)) return await extractDocx(buf);
  if (PPTX_RE.test(path)) return await extractPptx(buf);
  if (XLSX_RE.test(path)) return await extractXlsx(buf);
  return "";
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json());
app.use((req,res,next)=>{
  if(!SERVER_API_KEY) return next();
  if(req.path.startsWith("/mcp")||req.path==="/routeThenAnswer"){
    if(req.headers["x-api-key"]!==SERVER_API_KEY) return res.status(403).json({error:"Forbidden"});
  }
  next();
});

/* ---------- Routes ---------- */
app.get("/mcp/healthz", (_req,res)=>res.json({ok:true,root:DBX_ROOT_PREFIX}));

app.get("/mcp/list", async (req,res)=>{
  try {
    const path=req.query?.path || DBX_ROOT_PREFIX;
    const r=await dbxListFolder({ path:normPath(path), recursive:false, limit:2000});
    res.json({entries:normalizeEntries(r.data.entries||[])});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.post("/mcp/walk", async (req,res)=>{
  try {
    const { path_prefix=DBX_ROOT_PREFIX }=req.body||{};
    let entries=[], cursor=null;
    const r=await dbxListFolder({ path:normPath(path_prefix), recursive:true, limit:2000});
    entries.push(...(r.data.entries||[]));
    cursor=r.data.has_more? r.data.cursor : null;
    while(cursor){
      const cont=await dbxListContinue(cursor);
      entries.push(...(cont.data.entries||[]));
      cursor=cont.data.has_more? cont.data.cursor:null;
    }
    res.json({entries:normalizeEntries(entries)});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.get("/mcp/meta", async (req,res)=>{
  try {
    const { path }=req.query;
    if(!path) return res.status(400).json({error:"path required"});
    const r=await dbxGetMetadata(String(path));
    res.json(r.data);
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.post("/mcp/get", async (req,res)=>{
  try {
    const { path }=req.body||{};
    if(!path) return res.status(400).json({error:"path required"});
    const r=await dbxDownload({ path:String(path) });
    if(!r.ok) return res.status(502).json({ok:false,status:r.status});
    res.json({ok:true,path,size_bytes:r.data.length,data_base64:r.data.toString("base64")});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.post("/mcp/head", async (req,res)=>{
  try {
    const { path, bytes=200000 }=req.body||{};
    if(!path) return res.status(400).json({error:"path required"});
    // Only slice text-like and PDFs, otherwise fetch full file
    const isOffice = DOCX_RE.test(path)||XLSX_RE.test(path)||PPTX_RE.test(path);
    const dl=await dbxDownload({ path:String(path) });
    if(!dl.ok) return res.status(502).json({ok:false,status:dl.status});
    const buf=dl.data;
    const text=await extractText(path, buf);
    res.json({ok:true,path,text,truncated:false});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.post("/mcp/index_full", async (req,res)=>{
  try {
    const { path_prefix=DBX_ROOT_PREFIX }=req.body||{};
    const r=await dbxListFolder({ path:normPath(path_prefix), recursive:true, limit:2000});
    const entries=normalizeEntries(r.data.entries||[]).filter(e=>!ZIP.test(e.path));
    let out=[];
    for(const e of entries){
      try{
        const dl=await dbxDownload({ path:e.path });
        if(!dl.ok) continue;
        const text=await extractText(e.path, dl.data);
        out.push({ ...e, checksum:sha1(dl.data), text });
      } catch(err){console.error("Index skip",e.path,err.message);}
    }
    res.json({ok:true,total:out.length});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

/* ---------- Orchestrator ---------- */
app.post("/routeThenAnswer", async (req,res)=>{
  try {
    const { query }=req.body||{};
    if(!query) return res.status(400).json({error:"query required"});
    const q=query.toLowerCase();
    let intent="Browse";
    if(/search|find/.test(q)) intent="Search";
    else if(/preview|open|paragraph/.test(q)) intent="Preview";
    else if(/download|get/.test(q)) intent="Download";
    else if(/forecast/.test(q)) intent="Forecasting";
    else if(/root cause|why/.test(q)) intent="Root_Cause";
    else if(/compare/.test(q)) intent="Comparison";
    res.json({text:`Intent: ${intent}`,answer:{intent}});
  } catch(e){res.status(500).json({error:"router_error",message:e.message});}
});

app.listen(PORT, ()=>console.log(`DBX REST running on :${PORT}`));
