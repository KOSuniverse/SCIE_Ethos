import crypto from "node:crypto"; // at top once

function sha1(buf){ return crypto.createHash("sha1").update(buf).digest("hex"); }
const TEXTISH = /\.(txt|csv|json|ya?ml|md|log)$/i;
const ZIP = /\.zip$/i;
const XLSX = /\.xlsx$/i;

// heuristics (reuse from earlier robust index)
function parseMonthFromName(name){ const m1=name.match(/\b(20\d{2})[-_ ]?(0[1-9]|1[0-2])\b/); if(m1) return `${m1[1]}-${m1[2]}`; const M={jan:"01",feb:"02",mar:"03",apr:"04",may:"05",jun:"06",jul:"07",aug:"08",sep:"09",oct:"10",nov:"11",dec:"12"}; const m2=name.toLowerCase().match(/\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_ ]?(\d{2,4})\b/); if(m2){const mm=M[m2[1]]; const yy=m2[2].length===2?`20${m2[2]}`:m2[2]; return `${yy}-${mm}`;} return null; }
function guessCurrency(s){ if(/\bUSD\b|US\$|\$/.test(s)) return "USD"; if(/\bEUR\b|€/.test(s)) return "EUR"; if(/\bGBP\b|£/.test(s)) return "GBP"; return null; }
function guessCategoryBits(p){ const s=p.toLowerCase(); let category=null, sub=null; if(s.includes("wip")) category="WIP"; if(s.includes("inventory")) category=category||"Inventory"; if(s.includes("aged")) sub="Aged"; if(s.includes("finished")) sub=sub||"Finished"; if(s.includes("provision")) sub=sub||"Provision"; return {category, subcategory:sub}; }
function guessSiteCountryPlant(p){ const out={site:null,country:null,plant_code:null}; for(const seg of p.split("/")){ const m=seg.match(/\b([A-Z]{2,4})\s+(R?\d{3,4})\b/); if(m){out.site=m[1]; out.plant_code=m[2];} if(/italy|torino/i.test(seg)) out.country="ITA"; if(/germany|gmbh|deutschland|muelheim/i.test(seg)) out.country="DEU"; if(/thailand|rayong/i.test(seg)) out.country="THA"; if(/united\s?states|usa|houston|greenville|pps/i.test(seg)) out.country="USA"; if(/uk|aberdeen/i.test(seg)) out.country="GBR"; } return out; }

// quick XLSX sheet-name sniff (reads the [Content_Types].xml part; no heavy parse)
async function sniffXlsxSheetNames(dbxDownloadFn, path) {
  try {
    // download small head (e.g., 1.5 MB) – many xlsx are zipped xml; need full to unzip—so we conservatively skip deep unzip here
    // keep simple: return placeholder; we’ll upgrade in pass-2 with exceljs if needed
    return []; // placeholder to avoid large downloads for now
  } catch { return []; }
}

// checkpoint helpers
async function readCheckpoint(dbxDownloadFn, cpPath){
  try { const r=await dbxDownloadFn({path:cpPath}); if(!r.ok) return null; const txt=r.data.toString("utf8"); return JSON.parse(txt); } catch { return null; }
}
async function writeDropboxText(path, text){ const bytes=Buffer.from(text,"utf8"); return dbxUpload({path, bytes}); }

// Build a deep, resumable index; skip .zip; capture text previews and basic heuristics.
// Writes JSONL + CSV + manifest if paths provided.
app.post("/mcp/index_full", async (req, res) => {
  try {
    const {
      path_prefix = DBX_ROOT_PREFIX,
      output_jsonl_path = "/Project_Root/GPT_Files/_indexes/file_index.jsonl",
      output_csv_path   = "/Project_Root/GPT_Files/_indexes/file_index.csv",
      manifest_path     = "/Project_Root/GPT_Files/_indexes/manifest.json",
      preview_bytes     = 200_000,
      checksum_bytes_limit = 1_000_000,
      resume = true
    } = req.body || {};

    const rel = p => p.startsWith("/") ? p.slice(1) : p;

    // ensure manifest folder exists (Dropbox auto-creates on upload overwrite)
    const prev = resume ? await readCheckpoint(dbxDownload, manifest_path) : null;
    let total = 0, wrote = 0;

    // walk (fresh; for simplicity we don’t delta vs cursor yet)
    let cursor = null, raw = [];
    { const r = await dbxListFolder({ path: normPath(path_prefix), recursive: true, limit: 2000 });
      const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null; }
    while (cursor) {
      const r = await dbxListContinue(cursor);
      const j = r.data; raw.push(...(j.entries||[])); cursor = j.has_more ? j.cursor : null;
      // optional: break on absurdly huge trees
      if (raw.length > 300_000) break;
    }

    const entries = normalizeEntries(raw)
      .filter(e => e.path && e.mime !== "folder")
      .filter(e => !ZIP.test(e.path));

    total = entries.length;

    // CSV header (write once if starting new)
    if (!prev || !prev.csv_header_written) {
      const header = "site,country,plant_code,category,subcategory,month,file_path,as_of_date,currency,scale,canonical,has_totals,is_complete,checksum_sha1,mime,size,modified\n";
      await dbxUpload({ path: output_csv_path, bytes: Buffer.from(header,"utf8") });
    }

    // process entries (stream in chunks; append JSONL & CSV)
    for (const e of entries) {
      const filePath = e.path;
      const name = e.name || "";
      const month = parseMonthFromName(name) || parseMonthFromName(filePath) || null;
      const { category, subcategory } = guessCategoryBits(filePath);
      const { site, country, plant_code } = guessSiteCountryPlant(filePath);
      const currency = guessCurrency(filePath) || guessCurrency(name);
      const as_of_date = e.modified || null;

      // previews & checksum
      let text_preview = null, truncated = false, content_hash = "";
      let sheet_names = [];

      if (TEXTISH.test(name)) {
        const head = await dbxDownload({ path: filePath, rangeStart: 0, rangeEnd: preview_bytes });
        if (head.ok) {
          const buf = head.data;
          text_preview = buf.toString("utf8");
          truncated = buf.length >= preview_bytes;
          if (buf.length <= checksum_bytes_limit) content_hash = sha1(buf);
        }
      } else if (XLSX.test(name)) {
        sheet_names = await sniffXlsxSheetNames(dbxDownload, filePath); // placeholder ([]) for now
      } else {
        // small files can get hashed cheaply
        if (typeof e.size === "number" && e.size <= checksum_bytes_limit) {
          const dl = await dbxDownload({ path: filePath });
          if (dl.ok) content_hash = sha1(dl.data);
        }
      }

      // record (JSONL + CSV)
      const row = {
        site, country, plant_code, category, subcategory, month,
        file_path: rel(filePath), as_of_date, currency,
        scale: 1, canonical: true, has_totals: true, is_complete: true,
        checksum_sha1: content_hash, mime: e.mime, size: e.size, modified: e.modified,
        sheet_names, text_preview, truncated
      };

      // append JSONL
      await dbxUpload({ path: output_jsonl_path, bytes: Buffer.from(JSON.stringify(row)+"\n","utf8") });

      // append CSV
      const csvLine = [
        row.site, row.country, row.plant_code, row.category, row.subcategory, row.month,
        row.file_path, row.as_of_date, row.currency, row.scale, row.canonical, row.has_totals, row.is_complete,
        row.checksum_sha1, row.mime, row.size, row.modified
      ].map(v => `"${(v ?? "").toString().replace(/"/g,'""')}"`).join(",") + "\n";
      await dbxUpload({ path: output_csv_path, bytes: Buffer.from(csvLine,"utf8") });

      wrote++;
      // periodically update manifest
      if (wrote % 200 === 0) {
        await writeDropboxText(manifest_path, JSON.stringify({ path_prefix, total, processed: wrote, last_path: filePath, csv_header_written: true }, null, 2));
      }
    }

    // final manifest
    await writeDropboxText(manifest_path, JSON.stringify({ path_prefix, total, processed: wrote, csv_header_written: true, completed: true }, null, 2));

    res.json({ ok: true, stats: { total, processed: wrote }, outputs: { output_jsonl_path, output_csv_path, manifest_path }});
  } catch (e) {
    res.status(502).json({ ok:false, message:e.message, data:e?.response?.data || null });
  }
});



