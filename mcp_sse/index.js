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

const ENV = process.env;

const REQUIRED_ENV_VARS = [
  "DROPBOX_APP_KEY",
  "DROPBOX_APP_SECRET",
  "DROPBOX_REFRESH_TOKEN"
];

function validateEnv() {
  const missing = REQUIRED_ENV_VARS.filter((name) => !ENV[name]);
  if (missing.length) {
    const message = `Missing required environment variables: ${missing.join(", ")}`;
    console.error(message);
    throw new Error(message);
  }
}

validateEnv();

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  DROPBOX_APP_KEY,
  DROPBOX_APP_SECRET,
  DROPBOX_REFRESH_TOKEN,
  PORT = ENV.PORT || 10000
} = ENV;

const CONFIG_DEFAULTS = {
  MAX_DOWNLOAD_BYTES: 50 * 1024 * 1024, // 50 MB
  MAX_TEXT_RESPONSE_CHARS: 60_000,
  MAX_EXTRACTION_CHARS: 200_000,
  DOWNLOAD_TIMEOUT_MS: 120_000,
  HTTP_REQUEST_TIMEOUT_MS: 120_000,
  WALK_PAGE_LIMIT: 2000,
  WALK_FALLBACK_MAX_FETCH: 400,
  WALK_SESSION_TTL_MS: 10 * 60 * 1000,
  WALK_MAX_SESSIONS: 64,
  PDF_BASE_PAGE_LIMIT: 8,
  PDF_MEDIUM_FILE_BYTES: 2 * 1024 * 1024,
  PDF_MEDIUM_PAGE_LIMIT: 6,
  PDF_LARGE_FILE_BYTES: 5 * 1024 * 1024,
  PDF_LARGE_PAGE_LIMIT: 3,
  OCR_MAX_BYTES: 5 * 1024 * 1024,
  REQUEST_BODY_LIMIT: "2mb",
  SEARCH_RESULT_LIMIT: 15
};

const parseIntEnv = (name, fallback) => {
  const raw = ENV[name];
  if (!raw) return fallback;
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Invalid numeric environment value for ${name}: ${raw}`);
  }
  return value;
};

const parseStringEnv = (name, fallback) => ENV[name] || fallback;

const toOptionalInt = (value) => {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) ? parsed : null;
};

const clampPositiveInt = (value, fallback, max) => {
  const parsed = toOptionalInt(value);
  if (parsed === null || parsed <= 0) return fallback;
  if (max && parsed > max) return max;
  return parsed;
};

const DEFAULT_SERVER_API_KEY = "7d3b0d1c9f0d4c6fbe6f2c8a4d7e3b12b3a9f4d0c7e1a2f5c6d7e8f9a0b1c2d3";

const CONFIG = Object.freeze({
  maxDownloadBytes: parseIntEnv("MAX_DOWNLOAD_BYTES", CONFIG_DEFAULTS.MAX_DOWNLOAD_BYTES),
  maxTextResponseChars: parseIntEnv("MAX_TEXT_RESPONSE_CHARS", CONFIG_DEFAULTS.MAX_TEXT_RESPONSE_CHARS),
  maxExtractionChars: parseIntEnv("MAX_EXTRACTION_CHARS", CONFIG_DEFAULTS.MAX_EXTRACTION_CHARS),
  downloadTimeoutMs: parseIntEnv("DOWNLOAD_TIMEOUT_MS", CONFIG_DEFAULTS.DOWNLOAD_TIMEOUT_MS),
  httpTimeoutMs: parseIntEnv("HTTP_REQUEST_TIMEOUT_MS", CONFIG_DEFAULTS.HTTP_REQUEST_TIMEOUT_MS),
  walkPageLimit: parseIntEnv("WALK_PAGE_LIMIT", CONFIG_DEFAULTS.WALK_PAGE_LIMIT),
  pdfBasePageLimit: parseIntEnv("PDF_BASE_PAGE_LIMIT", CONFIG_DEFAULTS.PDF_BASE_PAGE_LIMIT),
  pdfMediumFileBytes: parseIntEnv("PDF_MEDIUM_FILE_BYTES", CONFIG_DEFAULTS.PDF_MEDIUM_FILE_BYTES),
  pdfMediumPageLimit: parseIntEnv("PDF_MEDIUM_PAGE_LIMIT", CONFIG_DEFAULTS.PDF_MEDIUM_PAGE_LIMIT),
  pdfLargeFileBytes: parseIntEnv("PDF_LARGE_FILE_BYTES", CONFIG_DEFAULTS.PDF_LARGE_FILE_BYTES),
  pdfLargePageLimit: parseIntEnv("PDF_LARGE_PAGE_LIMIT", CONFIG_DEFAULTS.PDF_LARGE_PAGE_LIMIT),
  ocrMaxBytes: parseIntEnv("OCR_MAX_BYTES", CONFIG_DEFAULTS.OCR_MAX_BYTES),
  requestBodyLimit: parseStringEnv("REQUEST_BODY_LIMIT", CONFIG_DEFAULTS.REQUEST_BODY_LIMIT),
  searchResultLimit: parseIntEnv("SEARCH_RESULT_LIMIT", CONFIG_DEFAULTS.SEARCH_RESULT_LIMIT),
  serverApiKey: parseStringEnv("SERVER_API_KEY", DEFAULT_SERVER_API_KEY),
  walkFallbackMaxFetch: parseIntEnv("WALK_FALLBACK_MAX_FETCH", CONFIG_DEFAULTS.WALK_FALLBACK_MAX_FETCH),
  walkSessionTtlMs: parseIntEnv("WALK_SESSION_TTL_MS", CONFIG_DEFAULTS.WALK_SESSION_TTL_MS),
  walkMaxSessions: parseIntEnv("WALK_MAX_SESSIONS", CONFIG_DEFAULTS.WALK_MAX_SESSIONS)
});

const apiKeyEnabled = Boolean(CONFIG.serverApiKey);

const TOKEN_URL   = "https://api.dropboxapi.com/oauth2/token";
const DBX_RPC     = "https://api.dropboxapi.com/2";
const DBX_CONTENT = "https://content.dropboxapi.com/2";
let _accessToken = null;

axios.defaults.timeout = CONFIG.httpTimeoutMs;
axios.defaults.maxBodyLength = CONFIG.maxDownloadBytes;
axios.defaults.maxContentLength = CONFIG.maxDownloadBytes;

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
    const { message, details } = normalizeAxiosError(err);
    console.error("Failed to refresh Dropbox token", message, details);
    throw err;
  }
}
async function withAuth(fn) {
  if (!_accessToken) await refreshAccessToken();
  try { return await fn(_accessToken); }
  catch (e) {
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
const PDF_RE  = /\.pdf$/i;
const DOCX_RE = /\.docx$/i;
const XLSX_RE = /\.xlsx$/i;
const PPTX_RE = /\.pptx$/i;

const normalizeAxiosError = (err) => {
  const code = err?.code || null;
  const status = err?.response?.status;
  const statusText = err?.response?.statusText;
  const body = err?.response?.data;
  let message = err?.message || "Unknown error";
  if (code === "ECONNABORTED") {
    message = `Request timed out after ${CONFIG.httpTimeoutMs}ms`;
  }
  const details = status ? { status, statusText, body } : undefined;
  return { message, code, details };
};

const truncateText = (text) => {
  if (typeof text !== "string") {
    return { text: "", truncated: false };
  }
  if (text.length <= CONFIG.maxTextResponseChars) {
    return { text, truncated: false };
  }
  return {
    text: text.slice(0, CONFIG.maxTextResponseChars),
    truncated: true
  };
};

const levenshtein = (a, b) => {
  if (a === b) return 0;
  if (!a) return b.length;
  if (!b) return a.length;

  const matrix = Array.from({ length: a.length + 1 }, () => new Array(b.length + 1));
  for (let i = 0; i <= a.length; i++) matrix[i][0] = i;
  for (let j = 0; j <= b.length; j++) matrix[0][j] = j;

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1,
        matrix[i][j - 1] + 1,
        matrix[i - 1][j - 1] + cost
      );
    }
  }
  return matrix[a.length][b.length];
};

const normalizeForSearch = (value) =>
  (value || "")
    .toLowerCase()
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const computeSearchScore = (query, entry) => {
  const normalizedQuery = normalizeForSearch(query);
  if (!normalizedQuery) return 0;

  const pathRaw = entry.path_display || entry.path_lower || "";
  const nameRaw = entry.name || "";
  const path = normalizeForSearch(pathRaw);
  const name = normalizeForSearch(nameRaw);

  const queryTokens = normalizedQuery.split(" ").filter(Boolean);

  if (path === normalizedQuery || name === normalizedQuery) return 1.0;

  if (queryTokens.length && queryTokens.every((token) => name.includes(token))) {
    return Math.max(0.85, 1 - Math.max(0, name.length - normalizedQuery.length) * 0.002);
  }
  if (queryTokens.length && queryTokens.every((token) => path.includes(token))) {
    return Math.max(0.8, 1 - Math.max(0, path.length - normalizedQuery.length) * 0.0015);
  }

  const rawQuery = query.toLowerCase();
  const rawPath = (entry.path_display || entry.path_lower || "").toLowerCase();
  const rawName = (entry.name || "").toLowerCase();
  if (rawPath.includes(rawQuery) || rawName.includes(rawQuery)) {
    return 0.75;
  }

  const distance = levenshtein(name, normalizedQuery);
  const longest = Math.max(name.length, normalizedQuery.length) || 1;
  const proximity = 1 - distance / longest;

  const pathDistance = levenshtein(path, normalizedQuery);
  const pathProximity = 1 - pathDistance / (Math.max(path.length, normalizedQuery.length) || 1);

  return Math.max(0, Math.max(proximity, pathProximity) * 0.7);
};

const sendError = (res, statusCode, err, extras = {}) => {
  const normalized = normalizeAxiosError(err || {});
  const payload = {
    ok: false,
    error: normalized.message,
    code: normalized.code || err?.code || null,
    details: normalized.details,
    ...extras
  };
  res.status(statusCode).json(payload);
};

const NORMALIZED_ROOT = DBX_ROOT_PREFIX
  ? `/${DBX_ROOT_PREFIX.replace(/^\/+/, "").replace(/\/+$/, "")}`
  : "/";
const NORMALIZED_ROOT_LOWER = NORMALIZED_ROOT.toLowerCase();

const ensureAbsolute = (value) => {
  if (!value) return "/";
  return value.startsWith("/") ? value : `/${value}`;
};

const resolveDropboxPath = (inputPath) => {
  if (!inputPath) return NORMALIZED_ROOT;

  const trimmed = inputPath.trim();
  if (!trimmed) return NORMALIZED_ROOT;

  const normalized = ensureAbsolute(trimmed.replace(/\\+/g, "/").replace(/\/+$/, ""));
  if (NORMALIZED_ROOT === "/") {
    return normalized || "/";
  }

  if (normalized === "/") {
    return NORMALIZED_ROOT;
  }

  const normalizedLower = normalized.toLowerCase();
  if (normalizedLower.startsWith(NORMALIZED_ROOT_LOWER)) {
    return normalized;
  }

  const withoutLeadingSlash = normalized.replace(/^\/+/, "");
  if (withoutLeadingSlash.toLowerCase().startsWith(NORMALIZED_ROOT_LOWER.replace(/^\/+/, ""))) {
    return ensureAbsolute(withoutLeadingSlash);
  }

  const suffix = withoutLeadingSlash;
  if (!suffix) {
    return NORMALIZED_ROOT;
  }

  return `${NORMALIZED_ROOT}/${suffix}`.replace(/\/{2,}/g, "/");
};

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
    { path, include_deleted: false },
    { headers: { Authorization: `Bearer ${token}` } }
  ));

const dbxSearchV2 = ({ query, path, max_results }) =>
  withAuth((token) => {
    const options = {
      max_results: Math.max(1, Math.min(200, max_results || CONFIG.searchResultLimit)),
      filename_only: false
    };
    if (path) {
      options.path = { ".tag": "path", path };
    }
    return axios.post(
      `${DBX_RPC}/files/search_v2`,
      { query, options },
      { headers: { Authorization: `Bearer ${token}` } }
    );
  });

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------- Walk helpers & fallback sessions ---------- */
const FALLBACK_CURSOR_PREFIX = "walkfb:";
const walkSessions = new Map();

function cleanupWalkSessions() {
  const now = Date.now();
  for (const [id, session] of walkSessions.entries()) {
    if (session.expiresAt <= now) {
      walkSessions.delete(id);
    }
  }
}

function touchWalkSession(session) {
  const now = Date.now();
  session.lastAccess = now;
  session.expiresAt = now + CONFIG.walkSessionTtlMs;
}

function storeWalkSession(session) {
  cleanupWalkSessions();
  touchWalkSession(session);
  walkSessions.set(session.id, session);
  if (walkSessions.size > CONFIG.walkMaxSessions) {
    let oldestId = null;
    let oldestAccess = Infinity;
    for (const [id, value] of walkSessions.entries()) {
      if (value.lastAccess < oldestAccess) {
        oldestAccess = value.lastAccess;
        oldestId = id;
      }
    }
    if (oldestId) {
      walkSessions.delete(oldestId);
    }
  }
}

function deleteWalkSession(sessionId) {
  walkSessions.delete(sessionId);
}

function getWalkSession(sessionId) {
  cleanupWalkSessions();
  const session = walkSessions.get(sessionId);
  if (!session) return null;
  if (session.expiresAt <= Date.now()) {
    walkSessions.delete(sessionId);
    return null;
  }
  touchWalkSession(session);
  return session;
}

const computeWalkFetchLimit = (targetCount) => {
  const desired = Math.max(1, targetCount || 1);
  const minChunk = Math.max(desired, 50);
  return Math.max(
    1,
    Math.min(CONFIG.walkPageLimit, CONFIG.walkFallbackMaxFetch, minChunk)
  );
};

const createWalkState = (rootPath, recursive) => ({
  queue: [{ path: rootPath, cursor: null }],
  pending: [],
  recursive: !!recursive
});

async function pumpWalkState(state, targetCount) {
  const desired = Math.max(1, targetCount || 1);
  const results = [];

  while (results.length < desired) {
    if (state.pending.length > 0) {
      results.push(state.pending.shift());
      continue;
    }

    const current = state.queue[0];
    if (!current) break;

    let response;
    if (current.cursor) {
      response = await dbxListContinue(current.cursor);
    } else {
      response = await dbxListFolder({
        path: current.path,
        recursive: false,
        limit: computeWalkFetchLimit(desired - results.length)
      });
    }

    const data = response?.data || {};
    const entries = Array.isArray(data.entries) ? data.entries : [];
    if (entries.length) {
      state.pending.push(...entries);
      if (state.recursive) {
        for (const entry of entries) {
          if (entry?.[".tag"] === "folder") {
            const childPath = entry.path_display || entry.path_lower || entry.path || null;
            if (childPath) {
              state.queue.push({ path: childPath, cursor: null });
            }
          }
        }
      }
    }

    if (data.has_more) {
      current.cursor = data.cursor;
    } else {
      state.queue.shift();
    }

    if (!entries.length && !data.has_more && state.pending.length === 0 && state.queue.length === 0) {
      break;
    }
  }

  return results;
}

const extractFallbackSessionId = (cursor) => {
  if (!cursor || typeof cursor !== "string") return null;
  return cursor.startsWith(FALLBACK_CURSOR_PREFIX)
    ? cursor.slice(FALLBACK_CURSOR_PREFIX.length)
    : null;
};

async function runFallbackWalk({ sessionId = null, resolvedPrefix, recursive, pageSize }) {
  const safePageSize = Math.max(1, pageSize || CONFIG.walkPageLimit);
  let session = null;

  if (sessionId) {
    session = getWalkSession(sessionId);
    if (!session) {
      throw new Error("Fallback walk cursor expired or is invalid");
    }
    if (session.root !== resolvedPrefix || session.recursive !== !!recursive) {
      throw new Error("Fallback cursor does not match requested path or recursion flag");
    }
  } else {
    session = {
      id: crypto.randomUUID(),
      root: resolvedPrefix,
      recursive: !!recursive,
      state: createWalkState(resolvedPrefix, recursive)
    };
  }

  const entries = await pumpWalkState(session.state, safePageSize);
  const hasMore = session.state.pending.length > 0 || session.state.queue.length > 0;

  if (hasMore) {
    storeWalkSession(session);
    return {
      entries,
      cursor: `${FALLBACK_CURSOR_PREFIX}${session.id}`,
      hasMore
    };
  }

  deleteWalkSession(session.id);
  return {
    entries,
    cursor: null,
    hasMore: false
  };
}

async function walkDropbox(pathPrefix, { recursive = true, limit = CONFIG.walkPageLimit, maxEntries = null } = {}) {
  const targetCount = Math.max(1, maxEntries ?? limit ?? CONFIG.walkPageLimit);
  const state = createWalkState(pathPrefix, recursive);
  const collected = [];

  while (collected.length < targetCount) {
    const remaining = targetCount - collected.length;
    const chunkTarget = Math.min(remaining, CONFIG.walkFallbackMaxFetch);
    const chunk = await pumpWalkState(state, chunkTarget);
    if (!chunk.length) {
      break;
    }
    collected.push(...chunk);
  }

  return collected.slice(0, targetCount);
}

const dbxDownload = async ({ path, range }) =>
  withAuth(async (token) => {
    console.log("Downloading from Dropbox:", path, range ? `(range ${range})` : "");
    const headers = {
      Authorization: `Bearer ${token}`,
      "Dropbox-API-Arg": JSON.stringify({ path })
    };
    if (range) {
      headers.Range = range;
    }

    let attempt = 0;
    while (attempt < 3) {
      try {
        const r = await axios.post(
          `${DBX_CONTENT}/files/download`,
          null,
          {
            headers,
            responseType: "arraybuffer",
            timeout: CONFIG.downloadTimeoutMs,
            maxContentLength: CONFIG.maxDownloadBytes,
            maxBodyLength: CONFIG.maxDownloadBytes
          }
        );

        console.log("Download complete:", path, "size:", r.data.byteLength);
        return {
          ok: true,
          status: r.status,
          headers: {
            "content-type": r.headers["content-type"] || null,
            "content-range": r.headers["content-range"] || null,
            "content-length": r.headers["content-length"] ? Number.parseInt(r.headers["content-length"], 10) : null
          },
          data: Buffer.from(r.data)
        };
      } catch (err) {
        const status = err?.response?.status;
        const { message } = normalizeAxiosError(err);
        if (status === 409) {
          attempt++;
          console.warn(`409 conflict, retrying ${attempt}/3...`);
          await sleep(500 * attempt);
          continue;
        }
        if (err?.code === "ECONNABORTED") {
          console.warn(`Download timeout for ${path} after ${CONFIG.downloadTimeoutMs}ms`);
        } else {
          console.error("Download failed", message);
        }
        throw err;
      }
    }
    throw new Error(`Failed to download ${path} after 3 attempts`);
  });

/* ---------- Extractors ---------- */
const determinePdfPageLimit = (bufLength, totalPages) => {
  let pageLimit = CONFIG.pdfBasePageLimit;
  if (bufLength > CONFIG.pdfLargeFileBytes) {
    pageLimit = Math.min(pageLimit, CONFIG.pdfLargePageLimit);
  } else if (bufLength > CONFIG.pdfMediumFileBytes) {
    pageLimit = Math.min(pageLimit, CONFIG.pdfMediumPageLimit);
  }
  return Math.max(1, Math.min(totalPages, pageLimit));
};

async function extractPdf(buf) {
  const sizeBytes = buf?.length || 0;
  try {
    console.log("PDF extractor (pdfjs)... size", sizeBytes);
    const data = new Uint8Array(buf);
    const pdf = await pdfjsLib.getDocument({ data }).promise;
    const pageLimit = determinePdfPageLimit(sizeBytes, pdf.numPages);
    console.log(`PDF has ${pdf.numPages} pages, processing up to ${pageLimit}`);
    let text = "";
    for (let i = 1; i <= pageLimit; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      text += content.items.map((it) => it.str).join(" ") + "\n";
      if (text.length >= CONFIG.maxExtractionChars) {
        console.warn("PDF extraction reached character cap, stopping early");
        await page.cleanup?.();
        break;
      }
      await page.cleanup?.();
    }
    if (text.trim().length > 0) {
      return { text, note: `Extracted with pdfjs (${pageLimit} pages)` };
    }
  } catch (err) {
    console.error("PDF ERROR (pdfjs):", err.message);
  }

  if (sizeBytes > CONFIG.ocrMaxBytes) {
    console.warn(`Skipping OCR fallback for PDF > ${CONFIG.ocrMaxBytes} bytes`);
    return { text: "", note: "PDF extraction failed and OCR skipped (file too large)" };
  }

  console.log("PDF extractor failed, falling back to OCR...");
  const { data: { text } } = await Tesseract.recognize(buf, "eng");
  return { text, note: "Extracted with OCR" };
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
  if (PDF_RE.test(path))  return await extractPdf(buf);
  if (DOCX_RE.test(path)) return await extractDocx(buf);
  if (XLSX_RE.test(path)) return await extractXlsx(buf);
  if (PPTX_RE.test(path)) return await extractPptx(buf);
  return { text: "", note: "Unsupported type" };
}

/* ---------- Express ---------- */
const app = express();
app.use(express.json({ limit: CONFIG.requestBodyLimit }));

app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl}`);
  next();
});

// API key gate
app.use((req, res, next) => {
  if (!req.path.startsWith("/mcp")) {
    return next();
  }
  if (!apiKeyEnabled) {
    return next();
  }

  const key = req.headers["x-api-key"];
  if (key !== CONFIG.serverApiKey) {
    console.warn("Forbidden: bad API key");
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

/* ---------- Routes ---------- */
app.get("/mcp/healthz", (_req, res) => {
  res.json({
    ok: true,
    root: NORMALIZED_ROOT,
    limits: {
      maxDownloadBytes: CONFIG.maxDownloadBytes,
      maxTextResponseChars: CONFIG.maxTextResponseChars,
      downloadTimeoutMs: CONFIG.downloadTimeoutMs
    }
  });
});

app.post("/mcp/walk", async (req, res) => {
  const {
    path_prefix = NORMALIZED_ROOT,
    recursive = true,
    max_items,
    cursor,
    force_fallback = false
  } = req.body || {};

  const resolvedPrefix = resolveDropboxPath(path_prefix);
  const perPageLimit = clampPositiveInt(max_items, CONFIG.walkPageLimit, CONFIG.walkPageLimit);
  const cursorToken = typeof cursor === "string" ? cursor.trim() : "";
  const fallbackSessionId = extractFallbackSessionId(cursorToken);
  const useFallback = Boolean(force_fallback) || Boolean(fallbackSessionId);

  const sendFallback = async (sessionId, extras = {}) => {
    const result = await runFallbackWalk({
      sessionId: sessionId || null,
      resolvedPrefix,
      recursive: !!recursive,
      pageSize: perPageLimit
    });

    const cursorPreview = result.cursor
      ? result.cursor.slice(FALLBACK_CURSOR_PREFIX.length, FALLBACK_CURSOR_PREFIX.length + 12)
      : "complete";
    console.log(
      `Fallback walk chunk entries=${result.entries.length} has_more=${result.hasMore} cursor=${cursorPreview}`
    );

    res.json({
      ok: true,
      path_prefix: resolvedPrefix,
      entries: result.entries,
      cursor: result.cursor,
      has_more: result.hasMore,
      mode: "fallback",
      ...extras
    });
  };

  if (useFallback) {
    try {
      await sendFallback(
        fallbackSessionId,
        force_fallback ? { fallback_forced: true } : {}
      );
    } catch (err) {
      console.error("WALK fallback error:", err);
      sendError(res, 502, err, { path_prefix: resolvedPrefix, mode: "fallback" });
    }
    return;
  }

  if (cursorToken) {
    try {
      console.log(`Continuing walk (cursor=${cursorToken.slice(0, 12)}...)`);
      const response = await dbxListContinue(cursorToken);
      const data = response?.data || {};
      const entries = data.entries || [];
      const hasMore = Boolean(data.has_more);

      res.json({
        ok: true,
        path_prefix: resolvedPrefix,
        entries,
        cursor: hasMore ? data.cursor : null,
        has_more: hasMore,
        mode: "direct"
      });
    } catch (err) {
      console.error("WALK CONTINUE ERROR:", err);
      const normalized = normalizeAxiosError(err);
      sendError(res, 502, err, {
        path_prefix: resolvedPrefix,
        cursor: cursorToken,
        mode: "direct",
        fallback_hint: "Retry without the cursor or include force_fallback=true to restart with server-managed pagination.",
        fallback_reason: normalized.message
      });
    }
    return;
  }

  try {
    console.log(`Walking path: ${resolvedPrefix} (recursive=${!!recursive}, limit=${perPageLimit})`);
    const response = await dbxListFolder({ path: resolvedPrefix, recursive: !!recursive, limit: perPageLimit });
    const data = response?.data || {};
    const entries = data.entries || [];
    const hasMore = Boolean(data.has_more);
    console.log(`Walk page entries=${entries.length} has_more=${hasMore}`);

    res.json({
      ok: true,
      path_prefix: resolvedPrefix,
      entries,
      cursor: hasMore ? data.cursor : null,
      has_more: hasMore,
      mode: "direct"
    });
  } catch (err) {
    const { message, code, details } = normalizeAxiosError(err);
    console.warn("Direct walk failed, using fallback pagination", { message, code, details });
    try {
      await sendFallback(null, {
        fallback_reason: message,
        fallback_code: code || null
      });
    } catch (fallbackErr) {
      console.error("WALK fallback failed:", fallbackErr);
      sendError(res, 502, fallbackErr, {
        path_prefix: resolvedPrefix,
        mode: "fallback",
        fallback_reason: message
      });
    }
  }
});

app.post("/mcp/search", async (req, res) => {
  try {
    const { query, path_prefix = NORMALIZED_ROOT, limit } = req.body || {};
    if (!query || typeof query !== "string") {
      return res.status(400).json({ ok: false, error: "query required" });
    }

    const resolvedPrefix = resolveDropboxPath(path_prefix);
    const safeLimit = clampPositiveInt(limit, CONFIG.searchResultLimit, CONFIG.searchResultLimit);
    const maxResults = Math.max(safeLimit * 3, safeLimit);
    console.log(`Searching for "${query}" under ${resolvedPrefix} (limit ${safeLimit}, request ${maxResults})`);

    let searchData = null;
    let matches = [];
    let searchError = null;
    let fallbackWalkUsed = false;
    let fallbackReason = null;
    let fallbackWalkSampled = 0;
    try {
      const searchResponse = await dbxSearchV2({ query, path: resolvedPrefix, max_results: maxResults });
      searchData = searchResponse?.data || {};
      matches = Array.isArray(searchData.matches) ? searchData.matches : [];
    } catch (err) {
      searchError = err;
      const status = err?.response?.status;
      if (status === 409) {
        console.warn("search_v2 path rejected, falling back to walk", { resolvedPrefix });
      } else {
        const { message, details } = normalizeAxiosError(err);
        console.warn("search_v2 failed, continuing with fallback", { message, details });
      }
    }

    let extracted = matches
      .map((match) => {
        const entry = match?.metadata?.metadata;
        if (!entry || entry[".tag"] !== "file") return null;
        const score = computeSearchScore(query, entry);
        return { entry, score, match_type: match?.match_type?.[".tag"] || null };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score);

    if (!extracted.length) {
      console.log("Search_v2 returned no file matches; falling back to walk-based scan");
      const entries = await walkDropbox(resolvedPrefix, {
        recursive: true,
        limit: CONFIG.walkPageLimit,
        maxEntries: CONFIG.walkPageLimit * 5
      });
      fallbackWalkUsed = true;
      fallbackReason = searchError ? "search_error" : "no_matches";
      fallbackWalkSampled = entries.length;
      const files = entries.filter((entry) => entry[".tag"] === "file");
      extracted = files
        .map((entry) => ({ entry, score: computeSearchScore(query, entry), match_type: null }))
        .filter((item) => item.score > 0)
        .sort((a, b) => b.score - a.score);
    }

    const results = extracted.slice(0, safeLimit).map(({ entry, score, match_type }) => ({
      score: Number(score.toFixed(3)),
      path_display: entry.path_display,
      path_lower: entry.path_lower,
      name: entry.name,
      id: entry.id,
      size: entry.size ?? null,
      tag: entry[".tag"],
      client_modified: entry.client_modified,
      server_modified: entry.server_modified,
      match_type
    }));

    const hasMoreSearch = Boolean(searchData?.has_more);
    const responseCursor = fallbackWalkUsed ? null : (hasMoreSearch ? searchData.cursor || null : null);

    res.json({
      ok: true,
      query,
      path_prefix: resolvedPrefix,
      total_matches: extracted.length,
      cursor: responseCursor,
      has_more: fallbackWalkUsed ? false : hasMoreSearch,
      fallback: fallbackWalkUsed || Boolean(searchError),
      fallback_reason: fallbackWalkUsed ? fallbackReason : (searchError ? "search_error" : null),
      fallback_walk_sampled: fallbackWalkSampled,
      results_mode: fallbackWalkUsed ? "fallback_walk" : "search_v2",
      results
    });
  } catch (e) {
    console.error("SEARCH ERROR:", e);
    sendError(res, 502, e);
  }
});

app.get("/mcp/list", async (req, res) => {
  try {
    const rawPath = typeof req.query?.path === "string" ? req.query.path : "";
    const limitParam = req.query?.limit;
    const resolvedPath = resolveDropboxPath(rawPath);
    const limit = clampPositiveInt(limitParam, CONFIG.walkPageLimit, CONFIG.walkPageLimit);
    console.log(`Listing path ${resolvedPath} (limit=${limit})`);

    const response = await dbxListFolder({ path: resolvedPath, recursive: false, limit });
    const data = response?.data || {};
    const entries = data.entries || [];
    const hasMore = Boolean(data.has_more);

    res.json({
      ok: true,
      path: resolvedPath,
      entries,
      cursor: hasMore ? data.cursor : null,
      has_more: hasMore
    });
  } catch (e) {
    console.error("LIST ERROR:", e);
    sendError(res, 502, e);
  }
});

app.get("/mcp/meta", async (req, res) => {
  let path = null;
  try {
    const rawPath = typeof req.query?.path === "string" ? req.query.path : null;
    if (!rawPath) {
      return res.status(400).json({ ok: false, error: "path required" });
    }

    path = resolveDropboxPath(rawPath);
    console.log("Fetching metadata:", path);
    const metadataResp = await dbxGetMetadata({ path });
    res.json({ ok: true, path, metadata: metadataResp?.data || null });
  } catch (e) {
    if (e?.response?.status === 409 && path) {
      return res.status(404).json({ ok: false, error: "Path not found", path });
    }
    console.error("META ERROR:", e);
    sendError(res, 502, e, path ? { path } : {});
  }
});

app.post("/mcp/get", async (req, res) => {
  let path = null;
  try {
    const { path: rawPath, range_start, range_end } = req.body || {};
    if (!rawPath) {
      return res.status(400).json({ ok: false, error: "path required" });
    }

    path = resolveDropboxPath(rawPath);
    console.log("Downloading raw file:", path);

    const metadataResp = await dbxGetMetadata({ path });
    const metadata = metadataResp?.data;
    if (!metadata || metadata[".tag"] !== "file") {
      return res.status(400).json({ ok: false, error: "Path is not a file", path });
    }

    const sizeBytes = metadata.size ?? null;
    const rangeStart = toOptionalInt(range_start);
    const rangeEnd = toOptionalInt(range_end);

    if (rangeEnd !== null && rangeStart === null) {
      return res.status(400).json({ ok: false, error: "range_start required when range_end provided", path });
    }

    let headerRange = null;
    let effectiveRange = null;
    if (rangeStart !== null) {
      if (rangeStart < 0) {
        return res.status(400).json({ ok: false, error: "range_start must be >= 0", path });
      }
      let computedEnd = rangeEnd;
      if (computedEnd !== null && computedEnd < rangeStart) {
        return res.status(400).json({ ok: false, error: "range_end must be >= range_start", path });
      }
      if (computedEnd === null) {
        const maxEnd = sizeBytes !== null ? sizeBytes - 1 : rangeStart + CONFIG.maxDownloadBytes - 1;
        computedEnd = Math.min(rangeStart + CONFIG.maxDownloadBytes - 1, maxEnd);
      }
      const chunkLength = computedEnd - rangeStart + 1;
      if (chunkLength > CONFIG.maxDownloadBytes) {
        return res.status(416).json({
          ok: false,
          error: "Requested range exceeds maximum chunk size",
          path,
          max_bytes: CONFIG.maxDownloadBytes
        });
      }
      headerRange = `bytes=${rangeStart}-${computedEnd}`;
      effectiveRange = { start: rangeStart, end: computedEnd };
    } else if (sizeBytes !== null && sizeBytes > CONFIG.maxDownloadBytes) {
      return res.status(413).json({
        ok: false,
        error: "File exceeds maximum allowed size",
        code: "FILE_TOO_LARGE",
        path,
        size_bytes: sizeBytes,
        max_bytes: CONFIG.maxDownloadBytes
      });
    }

    const dl = await dbxDownload({ path, range: headerRange });
    if (!dl?.ok) {
      return res.status(502).json({ ok: false, status: dl?.status || 502, path });
    }

    const buffer = dl.data;
    const base64 = buffer.toString("base64");
    const checksum = sha1(buffer);

    res.json({
      ok: true,
      path,
      encoding: "base64",
      size_bytes: buffer.length,
      checksum,
      content_type: dl.headers["content-type"],
      content_range: dl.headers["content-range"],
      range: effectiveRange,
      data: base64,
      metadata: {
        name: metadata.name,
        path_lower: metadata.path_lower,
        path_display: metadata.path_display,
        id: metadata.id,
        rev: metadata.rev,
        client_modified: metadata.client_modified,
        server_modified: metadata.server_modified,
        size: metadata.size ?? null
      }
    });
  } catch (e) {
    console.error("GET ERROR:", e);
    sendError(res, 502, e, path ? { path } : {});
  }
});

app.post("/mcp/open", async (req, res) => {
  try {
    const { path: rawPath } = req.body || {};
    if (!rawPath) {
      return res.status(400).json({ ok: false, error: "path required" });
    }

    const path = resolveDropboxPath(rawPath);
    console.log("Opening file:", path);

    const metadataResp = await dbxGetMetadata({ path });
    const metadata = metadataResp?.data;
    if (!metadata || metadata[".tag"] !== "file") {
      return res.status(400).json({
        ok: false,
        error: "Path is not a file",
        path
      });
    }

    const sizeBytes = metadata.size ?? 0;
    if (sizeBytes > CONFIG.maxDownloadBytes) {
      console.warn(`File ${path} exceeds max download size (${CONFIG.maxDownloadBytes} bytes)`);
      return res.status(413).json({
        ok: false,
        error: "File exceeds maximum allowed size",
        code: "FILE_TOO_LARGE",
        path,
        size_bytes: sizeBytes,
        max_bytes: CONFIG.maxDownloadBytes
      });
    }

    const dl = await dbxDownload({ path });
    if (!dl?.ok) {
      console.error("Download failed", dl?.status);
      return res.status(502).json({ ok: false, status: dl?.status || 502 });
    }

    let fileBuffer = dl.data;
    const checksum = sha1(fileBuffer);
    const extraction = await extractText(path, fileBuffer);
    const { text: clippedText, truncated } = truncateText(extraction.text);

    // help GC
    dl.data = null;
    fileBuffer = null;

    const noteParts = [];
    if (extraction.note) noteParts.push(extraction.note);
    if (truncated) {
      noteParts.push(`Text truncated to ${CONFIG.maxTextResponseChars} characters`);
    }

    res.json({
      ok: true,
      path,
      content_type: dl.headers["content-type"] || null,
      size_bytes: sizeBytes,
      checksum,
      text: clippedText,
      truncated,
      note: noteParts.join("; ") || null,
      metadata: {
        name: metadata.name,
        path_lower: metadata.path_lower,
        path_display: metadata.path_display,
        id: metadata.id,
        rev: metadata.rev,
        client_modified: metadata.client_modified,
        server_modified: metadata.server_modified
      }
    });
  } catch (e) {
    console.error("OPEN ERROR:", e);
    sendError(res, 502, e);
  }
});

/* ---------- Start ---------- */
app.listen(PORT, () => {
  console.log(`DBX REST running on :${PORT}`);
  console.log("Server configuration", {
    root: NORMALIZED_ROOT,
    maxDownloadMB: Number(CONFIG.maxDownloadBytes / (1024 * 1024)).toFixed(2),
    timeoutMs: CONFIG.downloadTimeoutMs,
    maxTextChars: CONFIG.maxTextResponseChars,
    apiKeyRequired: apiKeyEnabled
  });
  if (!apiKeyEnabled) {
    console.warn("API key checking is disabled; set SERVER_API_KEY to require authentication.");
  }
});

