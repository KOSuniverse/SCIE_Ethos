// index.js — Dropbox REST API with /mcp/open (robust file opener)

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
const sha1 = buf => crypto.createHash("sha1").update(buf).digest("hex");
const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const PDF_RE  = /\.pdf$/i;
const DOCX_RE = /\.docx$/i;
const XLSX_RE = /\.xlsx$/i;
const PPTX_RE = /\.pptx$/i;

/* ---------- Dropbox ---------- */
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
    if (text.trim().length > 0) return { text, note: "Extracted with pdfjs" };
  } catch {}
  const { data: { text } } = await Tesseract.recognize(buf, "eng");
  return { text, note: "Extracted with OCR" };
}
async function extractDocx(buf) {
  try {
    const result = await mammoth.extractRawText({ buffer: buf });
    if (result.value && result.value.trim().length > 0) {
      return { text: result.value, note: "Extracted with mammoth" };
    }
  } catch {}
  try {
    const zip = await JSZip.loadAsync(buf);
    const docXml = await zip.file("word/document.xml").async("string");
    return { text: docXml.replace(/<[^>]+>/g, " "), note: "Extracted with JSZip fallback" };
  } catch { return { text: "", note: "DOCX parse failed" }; }
}
async function extractXlsx(buf) {
  try {
    const wb = XLSX.read(buf, { type:"buffer" });
    let out = [];
    wb.SheetNames.forEach(name=>{
      const sheet = wb.Sheets[name];
      const csv = XLSX.utils.sheet_to_csv(sheet, { header:1 });
      out.push(`--- Sheet: ${name} ---\n${csv}`);
    });
    return { text: out.join("\n"), note: "Extracted with xlsx" };
  } catch { return { text: "", note: "XLSX parse failed" }; }
}
async function extractPptx(buf) {
  return new Promise((resolve) => {
    officeParser.parseOfficeAsync(buf, "pptx", (err, data) => {
      if (err) resolve({ text: "", note: "PPTX parse failed" });
      else resolve({ text: data || "", note: "Extracted with officeparser" });
    });
  });
}
async function extractText(path, buf) {
  if (TEXTISH.test(path)) return { text: buf.toString("utf8"), note: "Plain text" };
  if (PDF_RE.test(path))  return await extractPdf(buf);
  if (DOCX_RE.test(path)) return await extractDocx(buf);
  if (XLSX_RE.test(path)) return await extractXlsx(buf);
  if (PPTX_RE.test(path)) return await extractPptx(buf);
  return { text: "", note: "Unsupported type" };
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

/* ---------- New /mcp/open ---------- */
app.post("/mcp/open", async (req,res)=>{
  try {
    const { path } = req.body || {};
    if (!path) return res.status(400).json({ error: "path required" });

    const dl = await dbxDownload({ path });
    if (!dl.ok) return res.status(502).json({ ok:false, status: dl.status });

    const { text, note } = await extractText(path, dl.data);

    res.json({
      ok: true,
      path,
      content_type: dl.headers["content-type"] || null,
      size_bytes: dl.data.length,
      checksum: sha1(dl.data),
      text,
      note
    });
  } catch(e) {
    res.status(502).json({ ok:false, message:e.message });
  }
});

/* ---------- Health ---------- */
app.get("/mcp/healthz", (_req,res)=>res.json({ok:true,root:DBX_ROOT_PREFIX}));

/* ---------- Start ---------- */
app.listen(PORT, ()=>console.log(`DBX REST running on :${PORT}`));
