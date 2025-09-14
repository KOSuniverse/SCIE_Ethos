// server.js
import express from "express";
import bodyParser from "body-parser";
import axios from "axios";

const app = express();
app.use(bodyParser.json());

// ----- env -----
const {
  DBX_ROOT_PREFIX = "Apps/Ethos LLM/Project_Root/GPT_Files",
  SERVER_API_KEY,
  DROPBOX_ACCESS_TOKEN,
  PORT = 3000,
} = process.env;

if (!DROPBOX_ACCESS_TOKEN) {
  console.warn("[WARN] DROPBOX_ACCESS_TOKEN is not set — requests will fail.");
}

// ----- OPEN health -----
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true, root: DBX_ROOT_PREFIX }));

// ----- auth (skip healthz) -----
app.use((req, res, next) => {
  if (req.path === "/mcp/healthz") return next();
  const key = req.headers["x-api-key"];
  if (SERVER_API_KEY && key !== SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

// ----- dropbox client -----
const dbxRPC = axios.create({
  baseURL: "https://api.dropboxapi.com/2",
  headers: {
    Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
    "Content-Type": "application/json",
  },
});

// ----- list (recursive, with pagination) -----
app.get("/mcp/list", async (req, res) => {
  try {
    const basePath = req.query.path ? req.query.path : "/" + DBX_ROOT_PREFIX;
    let entries = [];
    let { data } = await dbxRPC.post("/files/list_folder", {
      path: basePath,
      recursive: true,
      include_deleted: false,
    });
    entries = entries.concat(data.entries || []);
    while (data.has_more) {
      const cont = await dbxRPC.post("/files/list_folder/continue", {
        cursor: data.cursor,
      });
      data = cont.data;
      entries = entries.concat(data.entries || []);
    }
    res.json({ entries });
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// ----- search (q in ?q=) -----
app.get("/mcp/search", async (req, res) => {
  try {
    const q = (req.query.q || "").toString();
    const { data } = await dbxRPC.post("/files/search_v2", {
      query: q,
      options: {
        path: "/" + DBX_ROOT_PREFIX,
        max_results: 100,
        file_status: "active",
      },
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// ----- get file bytes (JSON body: { path: "Apps/.../file.xlsx" }) -----
app.post("/mcp/get", async (req, res) => {
  try {
    const path = req.body?.path;
    if (!path) return res.status(400).json({ error: "path required" });
    const r = await axios.post(
      "https://content.dropboxapi.com/2/files/download",
      null,
      {
        headers: {
          Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
          "Dropbox-API-Arg": JSON.stringify({ path: "/" + path }),
        },
        responseType: "arraybuffer",
      }
    );
    res.setHeader("Content-Type", "application/octet-stream");
    res.send(r.data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

app.listen(PORT, () => {
  console.log(`DBX MCP on ${PORT} (root="${DBX_ROOT_PREFIX}")`);
});

