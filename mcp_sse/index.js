// index.js — Dropbox REST API for GPT Actions (End-State, Lazy PDF import)

import express from "express";
import axios from "axios";
import crypto from "node:crypto";
import * as XLSX from "xlsx";        // Excel parser
import mammoth from "mammoth";       // DOCX parser
import pptxParser from "pptx-parser"; // PPTX parser

/* ---------- Env ---------- */
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
const dbxDownload = async ({ path, rangeStart=null, rangeEnd=null }) =>
  withAuth(async token=>{
    const headers = {
      Authorization:`Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };
    if(rangeStart!=null&&rangeEnd!=null) headers.Range=`bytes=${rangeStart}-${rangeEnd-1}`;
    const r = await fetch(`${DBX_CONTENT}/files/download`, { method:"POST", headers });
    const ab = await r.arrayBuffer();
    return { ok:r.ok, status:r.status,
      headers:{ "content-type": r.headers.get("content-type") },
      data:Buffer.from(ab) };
  });
const dbxUpload = ({ path, bytes }) =>
  withAuth(async token=>{
    const headers = {
      Authorization:`Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path, mode:"overwrite", mute:true }),
      "Content-Type":"application/octet-stream"
    };
    const r = await fetch(`${DBX_CONTENT}/files/upload`, { method:"POST", headers, body:bytes });
    if(!r.ok) throw new Error(`upload_failed ${r.status}`);
    return r.json();
  });

/* ---------- Lazy PDF import ---------- */
let pdfParser;
async function getPdfParser() {
  if (!pdfParser) {
    pdfParser = (await import("pdf-parse")).default;
  }
  return pdfParser;
}

/* ---------- Extractors ---------- */
async function extractText(path, buf) {
  try {
    if (TEXTISH.test(path)) return buf.toString("utf8");
    if (XLSX_RE.test(path)) {
      const wb = XLSX.read(buf, { type:"buffer" });
      let out = [];
      wb.SheetNames.forEach(name=>{
        const sheet = wb.Sheets[name];
        const csv = XLSX.utils.sheet_to_csv(sheet, { header:1 });
        out.push(`--- Sheet: ${name} ---\n${csv}`);
      });
      return out.join("\n");
    }
    if (PDF_RE.test(path)) {
      const parser = await getPdfParser();
      const txt = await parser(buf);
      return txt.text;
    }
    if (DOCX_RE.test(path)) {
      const result = await mammoth.extractRawText({ buffer: buf });
      return result.value;
    }
    if (PPTX_RE.test(path)) {
      const pres = await pptxParser(buf);
      return pres.slides.map((s,i)=>`--- Slide ${i+1} ---\n${s.text}`).join("\n");
    }
    return "";
  } catch {
    return "";
  }
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json());

// API key gate
app.use((req,res,next)=>{
  if(!SERVER_API_KEY) return next();
  if(req.path.startsWith("/mcp")||req.path==="/routeThenAnswer"){
    if(req.headers["x-api-key"]!==SERVER_API_KEY) return res.status(403).json({error:"Forbidden"});
  }
  next();
});

/* ---------- Core routes ---------- */
app.get("/mcp/healthz", (_req,res)=>res.json({ok:true,root:DBX_ROOT_PREFIX}));

app.post("/mcp/list", async (req,res)=>{
  try {
    const path=req.query?.path || DBX_ROOT_PREFIX;
    const r=await dbxListFolder({ path:normPath(path), recursive:false, limit:2000});
    const j=r.data;
    res.json({entries:normalizeEntries(j.entries||[]), truncated:j.has_more});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

app.post("/mcp/walk", async (req,res)=>{
  try {
    const { path_prefix=DBX_ROOT_PREFIX, max_items=2000, cursor } = req.body||{};
    let entries=[], next=cursor||null;
    if(!next){
      const r=await dbxListFolder({ path:normPath(path_prefix), recursive:true, limit:Math.min(max_items,2000)});
      const j=r.data; entries=j.entries||[]; next=j.has_more?j.cursor:null;
    } else {
      const r=await dbxListContinue(next);
      const j=r.data; entries=j.entries||[]; next=j.has_more?j.cursor:null;
    }
    res.json({entries:normalizeEntries(entries), next_cursor:next, truncated:!!next});
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
    const { path, range_start=null, range_end=null }=req.body||{};
    if(!path) return res.status(400).json({error:"path required"});
    const r=await dbxDownload({ path:String(path), rangeStart:range_start, rangeEnd:range_end});
    if(!r.ok) return res.status(502).json({ok:false,status:r.status});
    res.json({ok:true,path,content_type:r.headers["content-type"],size_bytes:r.data.length,
      data_base64:r.data.toString("base64")});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

/* ---------- Safe preview ---------- */
app.post("/mcp/head", async (req,res)=>{
  try {
    const { path, bytes=200000 }=req.body||{};
    if(!path) return res.status(400).json({error:"path required"});
    const r=await dbxDownload({ path:String(path), rangeStart:0, rangeEnd:bytes});
    if(!r.ok) return res.status(502).json({ok:false,status:r.status});
    const ct=r.headers["content-type"]||"";
    let text="", note=null;
    if(TEXTISH.test(path)) text=r.data.toString("utf8").slice(0,2000);
    else if(PDF_RE.test(path)) {
      const parser = await getPdfParser();
      text=(await parser(r.data)).text.slice(0,2000);
    }
    else if(DOCX_RE.test(path)) text=(await mammoth.extractRawText({buffer:r.data})).value.slice(0,2000);
    else if(XLSX_RE.test(path)) { const wb=XLSX.read(r.data,{type:"buffer"}); text="Sheets: "+wb.SheetNames.join(", "); }
    else note="binary preview not supported";
    res.json({ok:true,path,content_type:ct,size_bytes:r.data.length,text,note});
  } catch(e){res.status(502).json({ok:false,message:e.message});}
});

/* ---------- Full Index ---------- */
app.post("/mcp/index_full", async (req,res)=>{
  try {
    const { path_prefix=DBX_ROOT_PREFIX, output_jsonl_path="/Project_Root/GPT_Files/_indexes/file_index.jsonl" }=req.body||{};
    let cursor=null, raw=[];
    { const r=await dbxListFolder({ path:normPath(path_prefix), recursive:true, limit:2000});
      const j=r.data; raw.push(...(j.entries||[])); cursor=j.has_more?j.cursor:null; }
    while(cursor){ const r=await dbxListContinue(cursor); const j=r.data; raw.push(...(j.entries||[])); cursor=j.has_more?j.cursor:null; }
    const entries=normalizeEntries(raw).filter(e=>e.path&&!ZIP.test(e.path)&&e.mime!=="folder");
    let count=0;
    for(const e of entries){
      try {
        const dl=await dbxDownload({ path:e.path });
        if(!dl.ok) continue;
        const text=await extractText(e.path, dl.data);
        const row={ path:e.path,name:e.name,size:e.size,modified:e.modified,
          mime:e.mime, checksum:sha1(dl.data), text };
        await dbxUpload({ path:output_jsonl_path, bytes:Buffer.from(JSON.stringify(row)+"\n","utf8") });
        count++;
      } catch(err){ console.error("index skip",e.path,err.message); }
    }
    res.json({ok:true,indexed:count,output:output_jsonl_path});
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
    else if(/preview|open/.test(q)) intent="Preview";
    else if(/download|get/.test(q)) intent="Download";
    else if(/forecast/.test(q)) intent="Forecasting";
    else if(/root cause|why/.test(q)) intent="Root_Cause";
    else if(/compare/.test(q)) intent="Comparison";
    res.json({ text:`Intent: ${intent} (stub for now)`, answer:{ intent }});
  } catch(e){res.status(500).json({error:"router_error",message:e.message});}
});

/* ---------- Start ---------- */
app.listen(PORT, ()=>console.log(`DBX REST running on :${PORT}`));
