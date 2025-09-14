// mcp_dbx/server.js
import express from "express";
import bodyParser from "body-parser";
import axios from "axios";

const app = express();
app.use(bodyParser.json());

const {
  DBX_ROOT_PREFIX = "/Project_Root/GPT_Files",
  SERVER_API_KEY,
  DROPBOX_ACCESS_TOKEN,
  PORT = 3000
} = process.env;

// -------- Open health --------
app.get("/mcp/healthz", (_req, res) => {
  res.json({ ok: true, root: DBX_ROOT_PREFIX });
});

// -------- Auth for everything else --------
app.use((req, res, next) => {
  if (req.path === "/mcp/healthz") return next();
  const key = req.headers["x-api-key"];
  if (SERVER_API_KEY && key !== SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

// -------- Dropbox RPC client --------
const dbxRPC = axios.create({
  baseURL: "https://api.dropboxapi.com/2",
  headers: {
    Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
    "Content-Type": "application/json"
  }
});

// -------- LIST (recursive, optional ?path=) --------
app.get("/mcp/list", async (req, res) => {
  try {
    const q = (req.query.path ?? DBX_ROOT_PREFIX).toString().trim();
    let p = q === "/" || q === "" ? "" : (q.startsWith("/") ? q : "/" + q);

    let entries = [];
    let { data } = await dbxRPC.post("/files/list_folder", {
      path: p,
      recursive: true,
      include_deleted: false
    });
    entries.push(...(data.entries || []));
    while (data.has_more) {
      const cont = await dbxRPC.post("/files/list_folder/continue", {
        cursor: data.cursor
      });
      data = cont.data;
      entries.push(...(data.entries || []));
    }
    res.json({ base_path: p || "(root)", entries });
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// -------- SEARCH (?q=) --------
app.get("/mcp/search", async (req, res) => {
  try {
    const q = (req.query.q || "").toString();
    const base = DBX_ROOT_PREFIX.startsWith("/")
      ? DBX_ROOT_PREFIX
      : "/" + DBX_ROOT_PREFIX;
    const { data } = await dbxRPC.post("/files/search_v2", {
      query: q,
      options: { path: base, max_results: 100, file_status: "active" }
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// -------- GET (JSON with base64) --------
app.post("/mcp/get", async (req, res) => {
  try {
    const path = req.body?.path;
    if (!path) return res.status(400).json({ error: "path required" });

    // IMPORTANT: for files/download do NOT set Content-Type
    const r = await axios.post(
      "https://content.dropboxapi.com/2/files/download",
      undefined,
      {
        headers: {
          Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
          "Dropbox-API-Arg": JSON.stringify({ path })
        },
        responseType: "arraybuffer",
        transformRequest: [(data, headers) => {
          // strip any default content-type that axios might inject
          if (headers && headers.common) delete headers.common["Content-Type"];
          if (headers && headers.post) delete headers.post["Content-Type"];
          return data;
        }]
      }
    );

    res.json({
      ok: true,
      path,
      content_type: r.headers["content-type"] || "application/octet-stream",
      size_bytes: r.data?.byteLength ?? null,
      data_base64: Buffer.from(r.data).toString("base64")
    });
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

app.listen(PORT, () => {
  console.log(`DBX REST on ${PORT} root=${DBX_ROOT_PREFIX}`);
});



