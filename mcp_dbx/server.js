import express from "express";
import bodyParser from "body-parser";
import axios from "axios";

const app = express();
app.use(bodyParser.json());

const {
  DBX_ROOT_PREFIX = "Apps/Ethos LLM/Project_Root/GPT_Files",
  SERVER_API_KEY,
  DROPBOX_ACCESS_TOKEN
} = process.env;

app.use((req, res, next) => {
  const key = req.headers["x-api-key"];
  if (SERVER_API_KEY && key !== SERVER_API_KEY) {
    return res.status(403).json({ error: "Forbidden" });
  }
  next();
});

const dbx = axios.create({
  baseURL: "https://api.dropboxapi.com/2",
  headers: {
    Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
    "Content-Type": "application/json"
  }
});

// health
app.get("/mcp/healthz", (_req, res) => res.json({ ok: true }));

// list everything under the GPT_Files root (recursive)
app.get("/mcp/list", async (_req, res) => {
  try {
    const { data } = await dbx.post("/files/list_folder", {
      path: "/" + DBX_ROOT_PREFIX,
      recursive: true
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// simple search passthrough (query in ?q=)
app.get("/mcp/search", async (req, res) => {
  try {
    const { data } = await dbx.post("/files/search_v2", {
      query: req.query.q || "",
      options: { path: "/" + DBX_ROOT_PREFIX, max_results: 20 }
    });
    res.json(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

// fetch file bytes (path in body: { path: "Apps/.../file.xlsx" })
app.post("/mcp/get", async (req, res) => {
  try {
    const path = req.body?.path;
    if (!path) return res.status(400).json({ error: "path required" });
    const { data } = await axios.post(
      "https://content.dropboxapi.com/2/files/download",
      null,
      {
        headers: {
          Authorization: `Bearer ${DROPBOX_ACCESS_TOKEN}`,
          "Dropbox-API-Arg": JSON.stringify({ path: "/" + path })
        },
        responseType: "arraybuffer"
      }
    );
    res.setHeader("Content-Type", "application/octet-stream");
    res.send(data);
  } catch (e) {
    res.status(500).json({ error: e.response?.data || e.message });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`DBX MCP on ${port}`));
