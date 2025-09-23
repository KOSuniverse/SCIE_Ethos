import express from "express";
import { ensureIndexScaffold, registerRoutes } from "./index.js";

const app = express();
app.use(express.json({ limit: "2mb" }));

app.get("/healthz", (_, res) => res.json({ ok: true }));

await ensureIndexScaffold();   // make sure /indexes files exist in Dropbox
registerRoutes(app);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up on :${PORT}`));
