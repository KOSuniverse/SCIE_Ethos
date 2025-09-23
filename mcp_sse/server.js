import express from "express";
import { ensureIndexScaffold, registerRoutes } from "./index.js";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/healthz", (_, res) => res.json({ ok: true }));

await ensureIndexScaffold();   // sets up /indexes on Dropbox if missing
registerRoutes(app);           // attaches /mcp/index_full

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up on :${PORT}`));

