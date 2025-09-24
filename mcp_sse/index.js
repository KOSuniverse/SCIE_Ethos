// index.js — Dropbox REST API with /mcp/walk and /mcp/open + robust logging

import express from "express";
import axios from "axios";
import crypto from "node:crypto";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import JSZip from "jszip";
import officeParser from "officeparser";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf.mjs";
import Tesseract from "tesseract.js";

// Environment validation
const REQUIRED_ENV_VARS = [
  "DROPBOX_APP_KEY",
  "DROPBOX_APP_SECRET", 
  "DROPBOX_REFRESH_TOKEN"
];

function validateEnv() {
  const missing = REQUIRED_ENV_VARS.filter((name) => !process.env[name]);
  if (missing.length) {
    console.error(`Missing required environment variables: ${missing.join(", ")}`);
    process.exit(1);
  }
}

validateEnv();

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,
  SERVER_API_KEY = "7d3b0d1c9f0d4c6fbe6f2c8a4d7e3b12b3a9f4d0c7e1a2f5c6d7e8f9a0b1c2d3",
  MAX_FILE_SIZE_MB = "50",
  REQUEST_TIMEOUT_MS = "120000",
  PORT = "10000"
} = process.env;

// Configuration
const MAX_FILE_SIZE = parseInt(MAX_FILE_SIZE_MB) * 1024 * 1024;
const REQUEST_TIMEOUT = parseInt(REQUEST_TIMEOUT_MS);
const SERVER_PORT = parseInt(PORT);

const TOKEN_URL = "https://api.dropboxapi.com/oauth2/token";
const DBX_RPC = "https://api.dropboxapi.com/2";
const DBX_CONTENT = "https://content.dropboxapi.com/2";
let _accessToken = null;

/* ---------- Auth ---------- */
async function refreshAccessToken() {
  console.log("Refreshing Dropbox access token...");
  try {
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
    console.log("New token acquired.");
    return _accessToken;
  } catch (err) {
    console.error("Failed to refresh Dropbox token:", err.message);
    throw err;
  }
}

async function withAuth(fn) {
  if (!_accessToken) await refreshAccessToken();
  try { 
    return await fn(_accessToken); 
  } catch (e) {
    const status = e?.response?.status;
    const body = e?.response?.data || "";
    if (status === 401 || String(body).includes("expired_access_token")) {
      console.warn("Token expired, refreshing...");
      await refreshAccessToken();
      return fn(_accessToken);
    }
    throw e;
  }
}

/* ---------- Helpers ---------- */
const sha1 = (buf) => crypto.createHash("sha1").update(buf).digest("hex");
const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const PDF_RE = /\.pdf$/i;
const DOCX_RE = /\.docx$/i;
const XLSX_RE = /\.xlsx$/i;
const PPTX_RE = /\.pptx$/i;

// Fuzzy matching for search
function levenshteinDistance(str1, str2) {
  const matrix = Array(str2.length + 1).fill().map(() => Array(str1.length + 1).fill(0));
  
  for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
  for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;
  
  for (let j = 1; j <= str2.length; j++) {
    for (let i = 1; i <= str1.length; i++) {
      const cost = str1[i - 1] === str2[j - 1] ? 0 : 1;
      matrix[j][i] = Math.min(
        matrix[j][i - 1] + 1,
        matrix[j - 1][i] + 1,
        matrix[j - 1][i - 1] + cost
      );
    }
  }
  
  return matrix[str2.length][str1.length];
}

function fuzzyMatch(query, target, threshold = 0.6) {
  query = query.toLowerCase().trim();
  target = target.toLowerCase();
  
  if (target.includes(query)) return 1.0;
  
  const distance = levenshteinDistance(query, target);
  const maxLen = Math.max(query.length, target.length);
  const similarity = 1 - (distance / maxLen);
  
  return similarity >= threshold ? similarity : 0;
}

/* ---------- Dropbox wrappers ---------- */
const dbxListFolder = ({ path, recursive, limit }) =>
  withAuth((token) => axios.post(
    `${DBX_RPC}/files/list_folder`,
    { path, recursive: !!recursive, include_deleted: false, limit },
    { headers: { Authorization: `Bearer ${token}` } }
  ));

const dbxListContinue = (cursor) =>
  withAuth((token) => axios.post(
    `${DBX_RPC}/files/list_folder/continue`,
    { cursor },
    { headers: { Authorization: `Bearer ${token}` } }
  ));

const dbxGetMetadata = ({ path }) =>
  withAuth((token) => axios.post(
    `${DBX_RPC}/files/get_metadata`,
    { path },
    { headers: { Authorization: `Bearer ${token}` } }
  ));

const dbxDownload = async ({ path }) =>
  withAuth(async (token) => {
    console.log("Downloading from Dropbox:", path);
    
    // Check file size first
    try {
      const metaResponse = await dbxGetMetadata({ path });
      const fileSize = metaResponse.data.size;
      console.log(`File size: ${fileSize} bytes (${(fileSize / 1024 / 1024).toFixed(2)} MB)`);
      
      if (fileSize > MAX_FILE_SIZE) {
        throw new Error(`File too large: ${(fileSize / 1024 / 1024).toFixed(2)} MB exceeds limit of ${(MAX_FILE_SIZE / 1024 / 1024).toFixed(2)} MB`);
      }
    } catch (err) {
      if (err.message.includes("File too large")) throw err;
      console.warn("Could not get file metadata, proceeding with download:", err.message);
    }
    
    const headers = {
      Authorization: `Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };

    let attempt = 0;
    while (attempt < 3) {
      try {
        const r = await axios.post(
          `${DBX_CONTENT}/files/download`,
          null,
          { 
            headers, 
            responseType: "arraybuffer", 
            timeout: REQUEST_TIMEOUT,
            maxContentLength: MAX_FILE_SIZE
          }
        );

        console.log("Download complete:", path, "size:", r.data.byteLength);
        return {
          ok: true,
          status: r.status,
          headers: { "content-type": r.headers["content-type"] },
          data: Buffer.from(r.data)
        };
      } catch (err) {
        const status = err?.response?.status;
        if (status === 409) {
          attempt++;
          console.warn(`409 conflict, retrying ${attempt}/3...`);
          await new Promise(res => setTimeout(res, 500 * attempt));
          continue;
        }
        if (err.code === 'ECONNABORTED') {
          throw new Error(`Download timeout after ${REQUEST_TIMEOUT / 1000} seconds for file: ${path}`);
        }
        throw err;
      }
    }
    throw new Error(`Failed to download ${path} after 3 attempts`);
  });

/* ---------- Extractors ---------- */
async function extractPdf(buf, path) {
  const sizeMB = buf.length / 1024 / 1024;
  console.log(`PDF extractor: ${path} (${sizeMB.toFixed(2)} MB)`);
  
  try {
    const data = new Uint8Array(buf);
    const pdf = await pdfjsLib.getDocument({ data }).promise;
    const totalPages = pdf.numPages;
    
    // Limit pages for large files
    const maxPages = sizeMB > 10 ? 3 : (sizeMB > 5 ? 5 : Math.min(totalPages, 10));
    console.log(`Processing ${maxPages} of ${totalPages} pages`);
    
    let text = "";
    for (let i = 1; i <= maxPages; i++) {
      try {
        const page = await pdf.getPage(i);
        const content = await page.getTextContent();
        const pageText = content.items.map(item => item.str).join(" ");
        text += `--- Page ${i} ---\n${pageText}\n\n`;
        
        if (text.length > 50000) {
          console.log("Extracted sufficient text, stopping early");
          break;
        }
      } catch (pageErr) {
        console.warn(`Error extracting page ${i}:`, pageErr.message);
        continue;
      }
    }
    
    if (text.trim().length > 0) {
      return { 
        text, 
        note: `Extracted ${maxPages}/${totalPages} pages with pdfjs`,
        pages_processed: maxPages,
        total_pages: totalPages
      };
    }
  } catch (err) {
    console.error("PDF ERROR:", err.message);
  }
  
  // OCR fallback for smaller files only
  if (sizeMB > 5) {
    return { 
      text: "", 
      note: `PDF too large for OCR (${sizeMB.toFixed(2)} MB), text extraction failed` 
    };
  }
  
  console.log("PDF extractor failed, falling back to OCR...");
  try {
    const { data: { text } } = await Tesseract.recognize(buf, "eng");
    return { text, note: "Extracted with OCR (fallback)" };
  } catch (ocrErr) {
    console.error("OCR ERROR:", ocrErr.message);
    return { text: "", note: "Both PDF parsing and OCR failed" };
  }
}

async function extractDocx(buf) {
  try {
    console.log("DOCX extractor (mammoth)...");
    const result = await mammoth.extractRawText({ buffer: buf });
    if (result.value && result.value.trim().length > 0) {
      return { text: result.value, note: "Extracted with mammoth" };
    }
  } catch (err) {
    console.error("DOCX ERROR (mammoth):", err.message);
  }
  try {
    console.log("DOCX extractor (JSZip fallback)...");
    const zip = await JSZip.loadAsync(buf);
    const docXml = await zip.file("word/document.xml").async("string");
    return { text: docXml.replace(/<[^>]+>/g, " "), note: "Extracted with JSZip fallback" };
  } catch (err) {
    console.error("DOCX ERROR (JSZip):", err.message);
    return { text: "", note: "DOCX parse failed" };
  }
}

async function extractXlsx(buf) {
  try {
    console.log("XLSX extractor...");
    const wb = XLSX.read(buf, { type: "buffer" });
    const out = [];
    wb.SheetNames.forEach((name) => {
      const sheet = wb.Sheets[name];
      const csv = XLSX.utils.sheet_to_csv(sheet, { header: 1 });
      out.push(`--- Sheet: ${name} ---\n${csv}`);
    });
    return { text: out.join("\n"), note: "Extracted with xlsx" };
  } catch (err) {
    console.error("XLSX ERROR:", err.message);
    return { text: "", note: "XLSX parse failed" };
  }
}

async function extractPptx(buf) {
  console.log("PPTX extractor...");
  return new Promise((resolve) => {
    officeParser.parseOfficeAsync(buf, "pptx", (err, data) => {
      if (err) {
        console.error("PPTX ERROR:", err.message);
        resolve({ text: "", note: "PPTX parse failed" });
      } else {
        resolve({ text: data || "", note: "Extracted with officeparser" });
      }
    });
  });
}

async function extractText(path, buf) {
  if (TEXTISH.test(path)) return { text: buf.toString("utf8"), note: "Plain text" };
  if (PDF_RE.test(path)) return await extractPdf(buf, path);
  if (DOCX_RE.test(path)) return await extractDocx(buf);
  if (XLSX_RE.test(path)) return await extractXlsx(buf);
  if (PPTX_RE.test(path)) return await extractPptx(buf);
  return { text: "", note: "Unsupported type" };
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json({ limit: '10mb' }));

// Logging middleware
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

// API key gate
app.use((req, res, next) => {
  if (!req.path.startsWith("/mcp")) {
    return next();
  }

  const key = req.headers["x-api-key"];
  if (key !== SERVER_API_KEY) {
    console.warn("Forbidden: bad API key");
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

/* ---------- Routes ---------- */
app.get("/mcp/healthz", (_req, res) => {
  res.json({
    ok: true,
    root: DBX_ROOT_PREFIX,
    limits: {
      maxFileSizeBytes: MAX_FILE_SIZE,
      requestTimeoutMs: REQUEST_TIMEOUT
    }
  });
});

// Search endpoint for fuzzy file matching
app.post("/mcp/search", async (req, res) => {
  try {
    const { query, limit = 20 } = req.body || {};
    if (!query) return res.status(400).json({ error: "query required" });
    
    console.log("Searching for:", query);
    
    // Get all files
    let entries = [];
    const r = await dbxListFolder({ path: DBX_ROOT_PREFIX, recursive: true, limit: 2000 });
    entries.push(...(r.data.entries || []));
    let cursor = r.data.has_more ? r.data.cursor : null;
    
    while (cursor) {
      const cont = await dbxListContinue(cursor);
      entries.push(...(cont.data.entries || []));
      cursor = cont.data.has_more ? cont.data.cursor : null;
    }
    
    // Filter and search files
    const files = entries.filter(e => e['.tag'] === 'file');
    const results = [];
    
    for (const file of files) {
      const fileName = file.name;
      const filePath = file.path_lower || file.path_display;
      
      const nameScore = fuzzyMatch(query, fileName);
      const pathScore = fuzzyMatch(query, filePath) * 0.8;
      
      const maxScore = Math.max(nameScore, pathScore);
      
      if (maxScore > 0) {
        results.push({
          ...file,
          match_score: maxScore,
          matched_on: nameScore > pathScore ? 'filename' : 'path'
        });
      }
    }
    
    results.sort((a, b) => b.match_score - a.match_score);
    const limitedResults = results.slice(0, parseInt(limit));
    
    console.log(`Search complete: found ${results.length} matches, returning top ${limitedResults.length}`);
    res.json({ 
      query,
      total_matches: results.length,
      results: limitedResults
    });
  } catch (e) {
    console.error("SEARCH ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

// Walk endpoint with improved error handling
app.post("/mcp/walk", async (req, res) => {
  try {
    const { path_prefix = DBX_ROOT_PREFIX, include_metadata = false } = req.body || {};
    console.log("Walking path:", path_prefix);
    
    let entries = [], cursor = null;
    const r = await dbxListFolder({ path: path_prefix, recursive: true, limit: 2000 });
    entries.push(...(r.data.entries || []));
    cursor = r.data.has_more ? r.data.cursor : null;
    
    while (cursor) {
      const cont = await dbxListContinue(cursor);
      entries.push(...(cont.data.entries || []));
      cursor = cont.data.has_more ? cont.data.cursor : null;
    }
    
    // Add helpful metadata
    const processedEntries = entries.map(entry => {
      const processed = { ...entry };
      if (entry['.tag'] === 'file') {
        processed.size_mb = entry.size ? (entry.size / 1024 / 1024).toFixed(2) : null;
        processed.is_large = entry.size > MAX_FILE_SIZE;
      }
      if (!include_metadata) {
        delete processed.content_hash;
      }
      return processed;
    });
    
    console.log(`Walk complete: ${entries.length} entries found`);
    res.json({ 
      path_prefix,
      total_entries: entries.length,
      entries: processedEntries
    });
  } catch (e) {
    console.error("WALK ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

// Open endpoint with improved error handling and text truncation
app.post("/mcp/open", async (req, res) => {
  try {
    const { path, extract_text = true } = req.body || {};
    if (!path) return res.status(400).json({ error: "path required" });

    console.log("Opening file:", path);

    const dl = await dbxDownload({ path });
    if (!dl.ok) {
      console.error("Download failed", dl.status);
      return res.status(502).json({ ok: false, status: dl.status, message: "Download failed" });
    }

    const response = {
      ok: true,
      path,
      content_type: dl.headers["content-type"] || null,
      size_bytes: dl.data.length,
      size_mb: (dl.data.length / 1024 / 1024).toFixed(2),
      checksum: sha1(dl.data)
    };

    if (extract_text) {
      const { text, note, ...extractInfo } = await extractText(path, dl.data);
      response.text = text;
      response.extraction_note = note;
      response.text_length = text.length;
      
      // Include additional extraction info
      if (extractInfo.pages_processed) {
        response.pages_processed = extractInfo.pages_processed;
        response.total_pages = extractInfo.total_pages;
      }
      
      // Truncate very long text
      if (text.length > 100000) {
        response.text = text.substring(0, 100000) + "\n\n[TEXT TRUNCATED - Original length: " + text.length + " characters]";
        response.text_truncated = true;
        response.original_text_length = text.length;
      }
    } else {
      response.text = null;
      response.extraction_note = "Text extraction skipped";
    }

    // Clean up memory
    dl.data = null;

    res.json(response);
  } catch (e) {
    console.error("OPEN ERROR:", e);
    const errorResponse = { 
      ok: false, 
      message: e.message,
      error_type: e.name || "Unknown"
    };
    
    if (process.env.NODE_ENV !== 'production') {
      errorResponse.stack = e.stack;
    }
    
    res.status(502).json(errorResponse);
  }
});

/* ---------- Start ---------- */
app.listen(SERVER_PORT, () => {
  console.log(`DBX REST running on :${SERVER_PORT}`);
  console.log(`Root prefix: ${DBX_ROOT_PREFIX}`);
  console.log(`Max file size: ${(MAX_FILE_SIZE / 1024 / 1024).toFixed(2)} MB`);
  console.log(`Request timeout: ${REQUEST_TIMEOUT / 1000}s`);
});
