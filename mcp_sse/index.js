// index.js — Dropbox REST API + Orchestrator + Persistent Index
import express from "express";
import axios from "axios";
import crypto from "node:crypto";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import JSZip from "jszip";
import officeParser from "officeparser";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import Tesseract from "tesseract.js";
import fs from "fs";
import path from "path";

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

// Persistent index directory on Render disk
const INDEX_DIR = "/data/_index/";
if (!fs.existsSync(INDEX_DIR)) {
  fs.mkdirSync(INDEX_DIR, { recursive: true });
  console.log("Created index dir at", INDEX_DIR);
}

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
    const expired =
      status === 401 ||
      (status === 400 && txt.includes("expired_access_token")) ||
      txt.includes("invalid_access_token");
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
    if (fileSize > MAX_FILE_SIZE)
      throw new Error(`File too large: ${(fileSize / 1024 / 1024).toFixed(2)} MB`);
    const headers = {
      Authorization: `Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };
    const r = await axios.post(`${DBX_CONTENT}/files/download`, null, {
      headers,
      responseType: "arraybuffer",
      timeout: REQUEST_TIMEOUT,
      maxContentLength: MAX_FILE_SIZE
    });
    return {
      ok: true,
      status: r.status,
      headers: { "content-type": r.headers["content-type"] },
      data: Buffer.from(r.data)
    };
  });

/* ---------- Extractors ---------- */
async function extractPdf(buf, path) {
  try {
    const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise;
    let text = "";
    const maxPages = Math.min(pdf.numPages, 5);
    for (let i = 1; i <= maxPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map((item) => item.str).join(" ") + "\n";
    }
    return { text, note: `Extracted ${maxPages} pages` };
  } catch (err) {
    const { data: { text } } = await Tesseract.recognize(buf, "eng");
    return { text, note: "OCR fallback" };
  }
}
async function extractDocx(buf) {
  try {
    const result = await mammoth.extractRawText({ buffer: buf });
    return { text: result.value, note: "Extracted with mammoth" };
  } catch {
    return { text: "", note: "DOCX parse failed" };
  }
}
async function extractXlsx(buf) {
  try {
    const wb = XLSX.read(buf, { type: "buffer" });
    return {
      text: wb.SheetNames.map(
        (name) => `--- ${name} ---\n${XLSX.utils.sheet_to_csv(wb.Sheets[name])}`
      ).join("\n"),
      note: "Extracted with xlsx"
    };
  } catch {
    return { text: "", note: "XLSX parse failed" };
  }
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
  if (/\.(txt|csv|json|ya?ml|md|log)$/i.test(path))
    return { text: buf.toString("utf8"), note: "Plain text" };
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
app.get("/mcp/healthz", (_req, res) =>
  res.json({ ok: true, root: DBX_ROOT_PREFIX })
);

/* ---------- Build Index ---------- */
app.post("/buildIndex", async (req, res) => {
  try {
    const { path_prefix = DBX_ROOT_PREFIX } = req.body || {};
    const r = await dbxListFolder({
      path: normPath(path_prefix),
      recursive: true,
      limit: 2000
    });
    const entries = normalizeEntries(r.data.entries || []).filter(
      (e) => e.mime !== "folder"
    );

    for (const f of entries) {
      try {
        const dl = await dbxDownload({ path: f.path });
        const buf = dl.data.slice(0, 200000); // cap for large files
        const { text, note } = await extractText(f.name, buf);

        const doc = {
          path: f.path,
          name: f.name,
          modified: f.modified,
          text: text.length > 100000 ? text.slice(0, 100000) : text,
          note
        };

        const outPath = path.join(INDEX_DIR, f.name + ".json");
        fs.writeFileSync(outPath, JSON.stringify(doc, null, 2));
        console.log("Indexed", f.name);
      } catch (err) {
        console.warn("Index skip:", f.name, err.message);
      }
    }

    res.json({ ok: true, indexed: entries.length, dir: INDEX_DIR });
  } catch (e) {
    console.error("buildIndex ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

/* ---------- Search Index ---------- */
app.post("/searchIndex", async (req, res) => {
  try {
    const { query, limit = 10 } = req.body || {};
    if (!query) return res.status(400).json({ error: "query required" });

    const terms = query.toLowerCase().split(/\s+or\s+|\s+/).filter(t => t.length > 2);
    const files = fs.readdirSync(INDEX_DIR).filter((f) => f.endsWith(".json"));

    const results = [];
    for (const file of files) {
      const doc = JSON.parse(fs.readFileSync(path.join(INDEX_DIR, file), "utf8"));
      if (!doc.text) continue;

      const textLower = doc.text.toLowerCase();
      for (const term of terms) {
        const idx = textLower.indexOf(term);
        if (idx !== -1) {
          const snippet = doc.text.substring(Math.max(0, idx - 200), idx + 200);
          results.push({
            file: doc.name,
            path: doc.path,
            modified: doc.modified,
            match: term,
            snippet: snippet.replace(/\s+/g, " ").trim(),
            note: doc.note
          });
          break; // don’t add the same file multiple times
        }
      }
      if (results.length >= limit) break;
    }

    res.json({ query, hits: results.length, results });
  } catch (e) {
    console.error("searchIndex ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});


/* ---------- Start ---------- */
app.listen(PORT, () =>
  console.log(`DBX REST on ${PORT} root=${DBX_ROOT_PREFIX}`)
);
