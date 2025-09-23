// entrypoint
import express from "express";
import { registerRoutes, ensureIndexScaffold } from "./index.js";

const app = express();
app.use(express.json({ limit: "2mb" }));
app.get("/healthz", (_, res) => res.json({ ok: true }));

await ensureIndexScaffold();   // create Dropbox indexes/ files if missing
registerRoutes(app);           // <-- give routes the app

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Up on :${PORT}`));

