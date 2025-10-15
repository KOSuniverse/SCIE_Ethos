// index.js — Dropbox REST API + Semantic Index
import express from "express";
import axios from "axios";
import crypto from "node:crypto";
import * as XLSX from "xlsx";
import mammoth from "mammoth";
import JSZip from "jszip";
import officeParser from "officeparser";
import * as pdfjsLib from "pdfjs-dist/build/pdf.js";
import Tesseract from "tesseract.js";
import fs from "fs";
import path from "path";

// OpenAI client
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

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
    const status = e?.response?.status;
    const body = e?.response?.data || {};

    if (status === 401 || JSON.stringify(body).includes("expired_access_token")) {
      await refreshAccessToken();
      return fn(_accessToken);
    }

    if (!_accessToken) {
      await refreshAccessToken();
      return fn(_accessToken);
    }

    throw e;
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

/* ---------- Dropbox RPC ---------- */
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
      { path },
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
    return { data: Buffer.from(r.data) };
  });

/* ---------- Extractors ---------- */
async function extractPdf(buf) {
  const pdf = await pdfjsLib.getDocument({ data: new Uint8Array(buf) }).promise;
  let text = "";
  const maxPages = Math.min(pdf.numPages, 5);
  for (let i = 1; i <= maxPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    text += content.items.map((item) => item.str).join(" ") + "\n";
  }
  return text;
}
async function extractDocx(buf) {
  const result = await mammoth.extractRawText({ buffer: buf });
  return result.value;
}
async function extractXlsx(buf) {
  const wb = XLSX.read(buf, { type: "buffer" });
  return wb.SheetNames.map(
    (name) => `--- ${name} ---\n${XLSX.utils.sheet_to_csv(wb.Sheets[name])}`
  ).join("\n");
}
async function extractPptx(buf) {
  return new Promise((resolve) => {
    officeParser.parseOfficeAsync(buf, "pptx", (err, data) => {
      resolve(data || "");
    });
  });
}
async function extractText(path, buf) {
  if (/\.pdf$/i.test(path)) return extractPdf(buf);
  if (/\.docx$/i.test(path)) return extractDocx(buf);
  if (/\.xlsx$/i.test(path)) return extractXlsx(buf);
  if (/\.pptx$/i.test(path)) return extractPptx(buf);
  if (/\.(txt|csv|json|ya?ml|md|log)$/i.test(path)) return buf.toString("utf8");
  return "";
}

/* ---------- Embeddings ---------- */
async function embedText(text) {
  const res = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: text
  });
  return res.data[0].embedding;
}
function cosineSim(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json());
app.use((req, res, next) => {
  if (!SERVER_API_KEY) return next();
  if (req.path.startsWith("/")) {
    const key = req.headers["x-api-key"];
    if (key !== SERVER_API_KEY)
      return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

/* ---------- Build Index for ALL Files ---------- */
app.post("/buildIndexAll", async (req, res) => {
  try {
    const rootPrefix = DBX_ROOT_PREFIX;
    console.log("Starting full index build from:", rootPrefix);

    const r = await dbxListFolder({ path: normPath(rootPrefix), recursive: true, limit: 2000 });
    let entries = r.data.entries || [];
    let cursor = r.data.has_more ? r.data.cursor : null;

    while (cursor) {
      const next = await dbxListContinue(cursor);
      entries = entries.concat(next.data.entries || []);
      cursor = next.data.has_more ? next.data.cursor : null;
    }

    const files = normalizeEntries(entries).filter((e) => e.mime !== "folder");
    console.log("Found", files.length, "files to index.");

    let processed = 0;
    for (const f of files) {
      try {
        const dl = await dbxDownload({ path: f.path });
        const text = await extractText(f.path, dl.data);
        if (!text) {
          console.log("Skip (no text):", f.name);
          continue;
        }

        const chunks = (text.match(/.{1,1000}/gs) || [])
          .map(c => c.trim())
          .filter(c => c.length > 0 && c.length < 4000);

        const outData = [];
        for (const chunk of chunks) {
          try {
            const embedding = await embedText(chunk);
            outData.push({
              path: f.path,
              name: f.name,
              modified: f.modified,
              text: chunk,
              embedding
            });
          } catch (embedErr) {
            console.warn("Embed skip:", f.name, embedErr.message);
          }
        }

        if (outData.length > 0) {
          const outPath = path.join(INDEX_DIR, f.name + ".json");
          fs.writeFileSync(outPath, JSON.stringify(outData, null, 2));
          console.log("Indexed", f.name, "chunks:", outData.length);
          processed++;
        } else {
          console.log("No valid chunks for:", f.name);
        }
      } catch (err) {
        console.warn("Index skip:", f.name, err.message);
      }
    }

    res.json({ ok: true, total: files.length, processed });
  } catch (e) {
    console.error("buildIndexAll ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

/* ---------- Search Index ---------- */
app.post("/searchIndex", async (req, res) => {
  try {
    const { query, limit = 5 } = req.body || {};
    if (!query) return res.status(400).json({ error: "query required" });

    const queryEmbedding = await embedText(query);
    const files = fs.readdirSync(INDEX_DIR).filter((f) => f.endsWith(".json"));

    const scored = [];
    for (const file of files) {
      const docs = JSON.parse(fs.readFileSync(path.join(INDEX_DIR, file), "utf8"));
      for (const doc of docs) {
        const sim = cosineSim(queryEmbedding, doc.embedding);
        scored.push({ ...doc, score: sim });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    const results = scored.slice(0, limit).map((r) => ({
      file: r.name,
      path: r.path,
      modified: r.modified,
      snippet: r.text.slice(0, 300),
      score: r.score.toFixed(3)
    }));

    res.json({ query, results });
  } catch (e) {
    console.error("searchIndex ERROR:", e);
    res.status(502).json({ ok: false, message: e.message });
  }
});

/* ---------- Walk (cursor-based) ---------- */
app.post("/mcp/walk", async (req, res) => {
  const { path_prefix, max_items = 2000, cursor } = req.body;
  try {
    if (cursor) {
      const response = await dbxListContinue(cursor);
      res.json({
        entries: response.data.entries,
        cursor: response.data.has_more ? response.data.cursor : null
      });
    } else {
      const response = await dbxListFolder({
        path: normPath(path_prefix),
        recursive: false,
        limit: max_items
      });
      res.json({
        entries: response.data.entries,
        cursor: response.data.has_more ? response.data.cursor : null
      });
    }
  } catch (err) {
    console.error("Error in /mcp/walk:", err);
    res.status(500).json({ error: err.message });
  }
});

/* ---------- Walk Full (server-side pagination) ---------- */
app.post("/mcp/walk_full", async (req, res) => {
  const { path_prefix, max_items = 2000 } = req.body;
  let allEntries = [];
  let cursor = null;
  let safetyLimit = 10000;

  try {
    do {
      let response;
      if (cursor) {
        response = await dbxListContinue(cursor);
      } else {
        response = await dbxListFolder({
          path: normPath(path_prefix),
          recursive: true,
          limit: max_items
        });
      }

      allEntries = allEntries.concat(response.data.entries || []);
      cursor = response.data.has_more ? response.data.cursor : null;

      if (allEntries.length > safetyLimit) {
        console.warn("walk_full: hit safety limit, stopping early.");
        break;
      }
    } while (cursor);

    res.json({ entries: allEntries, total: allEntries.length });
  } catch (err) {
    console.error("Error in /mcp/walk_full:", err);
    res.status(500).json({ error: err.message });
  }
});
/* ---------- Get Manifest ---------- */
app.get("/getManifest", async (req, res) => {
  try {
    const type = req.query.type || "snapshot"; // ?type=snapshot or ?type=file_index
    let filePath;

    if (type === "snapshot") {
      filePath = "/data/02_Snapshot_Manifest.csv"; // adjust if stored elsewhere
    } else if (type === "file_index") {
      filePath = "/data/file_index.csv";
    } else {
      return res.status(400).json({ error: "Invalid manifest type" });
    }

    if (fs.existsSync(filePath)) {
      const data = fs.readFileSync(filePath, "utf8");
      res.type("text/csv").send(data);
    } else {
      res.status(404).json({ error: `Manifest not found: ${filePath}` });
    }
  } catch (e) {
    console.error("getManifest ERROR:", e);
    res.status(500).json({ error: e.message });
  }
});

/* ---------- Index Status ---------- */
app.get("/indexStatus", async (req, res) => {
  try {
    if (!fs.existsSync(INDEX_DIR)) {
      return res.json({ ok: true, total: 0, indexed: [] });
    }
    const files = fs.readdirSync(INDEX_DIR).filter((f) => f.endsWith(".json"));
    res.json({
      ok: true,
      total: files.length,
      indexed: files
    });
  } catch (e) {
    console.error("indexStatus ERROR:", e);
    res.status(500).json({ error: e.message });
  }
});

/* ---------- Query Excel ---------- */
app.post("/queryExcel", async (req, res) => {
  try {
    const { path: filePath, sheet, operation = "preview", columns = [], filters = {} } = req.body;

    if (!filePath) {
      return res.status(400).json({ error: "path is required" });
    }

    if (!fs.existsSync(filePath)) {
      return res.status(404).json({ error: "File not found" });
    }

    const wb = XLSX.readFile(filePath);
    const ws = sheet ? wb.Sheets[sheet] : wb.Sheets[wb.SheetNames[0]];
    let df = XLSX.utils.sheet_to_json(ws, { defval: null });

    // Apply filters
    for (const [col, val] of Object.entries(filters)) {
      df = df.filter((row) => row[col] == val);
    }

    let result;
    if (operation === "preview") {
      result = df.slice(0, 20); // return first 20 rows
    } else if (operation === "sum" && columns.length > 0) {
      result = {};
      for (const col of columns) {
        const sum = df.reduce((acc, row) => acc + (parseFloat(row[col]) || 0), 0);
        result[col] = sum;
      }
    } else if (operation === "average" && columns.length > 0) {
      result = {};
      for (const col of columns) {
        const valid = df.map((row) => parseFloat(row[col])).filter((v) => !isNaN(v));
        const avg = valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : 0;
        result[col] = avg;
      }
    } else {
      result = { message: "Operation not implemented or no columns specified" };
    }

    res.json({ ok: true, operation, result });
  } catch (e) {
    console.error("queryExcel ERROR:", e);
    res.status(500).json({ error: e.message });
  }
});
/* ---------- Dropbox Access Token Endpoint ---------- */
app.get("/token/dropbox", async (req, res) => {
  try {
    // Always ensure we have a valid token
    if (!_accessToken) {
      await refreshAccessToken();
    }

    // Return a new token if requested via ?refresh=true
    if (req.query.refresh === "true") {
      await refreshAccessToken();
    }

    res.json({
      access_token: _accessToken,
      source: "cached",
      expires_in: 14400
    });
  } catch (e) {
    console.error("token/dropbox ERROR:", e);
    res.status(500).json({ error: e.message });
  }
});
app.get("/mcp", (req, res) => {
  const key = req.headers["x-api-key"];
  if (process.env.SERVER_API_KEY && key !== process.env.SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }

  res.setHeader("Content-Type", "application/json");
  res.status(200).json({
    type: "mcp_list_tools",
    tools: [
      {
        name: "walk",
        description: "List Dropbox folder contents recursively.",
        input_schema: {
          type: "object",
          properties: {
            path_prefix: { type: "string" },
            max_items: { type: "number" }
          },
          required: ["path_prefix"]
        }
      },
      {
        name: "searchIndex",
        description: "Search the semantic index for relevant content.",
        input_schema: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "number" }
          },
          required: ["query"]
        }
      },
      {
        name: "buildIndexAll",
        description: "Rebuild the semantic index for all Dropbox files.",
        input_schema: {
          type: "object",
          properties: {}
        }
      }
    ]
  });
});
/* ---------- MCP Manifest (for Actions integration) ---------- */
app.get("/mcp/manifest", (req, res) => {
  const key = req.headers["x-api-key"];
  if (process.env.SERVER_API_KEY && key !== process.env.SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }

  res.setHeader("Content-Type", "application/json");
  res.status(200).json({
    version: "1.0",
    tools: [
      {
        name: "walk",
        description: "List Dropbox folder contents recursively.",
        input_schema: {
          type: "object",
          properties: {
            path_prefix: { type: "string" },
            max_items: { type: "number" }
          },
          required: ["path_prefix"]
        }
      },
      {
        name: "searchIndex",
        description: "Search the semantic index for relevant content.",
        input_schema: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "number" }
          },
          required: ["query"]
        }
      },
      {
        name: "buildIndexAll",
        description: "Rebuild the semantic index for all Dropbox files.",
        input_schema: {
          type: "object",
          properties: {}
        }
      }
    ]
  });
});

/* ---------- MCP Manifest (strict OpenAI schema) ---------- */
app.get("/mcp/manifest_strict", (req, res) => {
  const key = req.headers["x-api-key"];
  if (process.env.SERVER_API_KEY && key !== process.env.SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }

  res.type("application/json").json({
    type: "mcp-manifest",                      // 👈 REQUIRED by OpenAI MCP spec
    api_version: "1.0",                        // 👈 some clients expect this alias
    schema_version: "1.0",
    name_for_human: "SCIE Ethos Connector",
    name_for_model: "scie_ethos",
    description_for_human:
      "Access Dropbox and semantic-index tools for Jarvis Mayhem Orchestrator.",
    description_for_model:
      "Provides three tools: walk (list Dropbox folders), searchIndex (query semantic index), and buildIndexAll (rebuild index).",
    contact_email: "support@ethos.local",
    legal_info_url: "https://scie-ethos.onrender.com/legal",
    tools: [
      {
        name: "walk",
        description: "List Dropbox folder contents recursively.",
        parameters: {
          type: "object",
          properties: {
            path_prefix: { type: "string" },
            max_items: { type: "number" }
          },
          required: ["path_prefix"]
        }
      },
      {
        name: "searchIndex",
        description: "Search the semantic index for relevant content.",
        parameters: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "number" }
          },
          required: ["query"]
        }
      },
      {
        name: "buildIndexAll",
        description: "Rebuild the semantic index for all Dropbox files.",
        parameters: {
          type: "object",
          properties: {}
        }
      }
    ]
  });
});


/* ---------- Start ---------- */
app.listen(PORT, () => {
  console.log(`DBX REST with semantic index running on ${PORT}, root=${DBX_ROOT_PREFIX}`);
  refreshAccessToken().catch(err => {
    console.error("Initial token refresh failed:", err.message);
  });
});



