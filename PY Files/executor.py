# executor.py

from intent import classify_intent
from tools_runtime import dataframe_query, kb_search
from phase3_comparison.comparison_utils import compare_wip_aging, compare_inventory
from phase3_comparison.ranking_utils import rank_task
from phase4_knowledge.kb_qa import kb_answer
from assistant_bridge import run_query as assistants_answer
import json

def run_intent_task(query, df_dict, metadata_index, client):
    intent = classify_intent(query)
    # Normalize df inputs
    dfs = list(df_dict.items()) if df_dict else []
    first_path = dfs[0][0] if dfs else None

    if intent == "eda":
        out = dataframe_query(files=[first_path] if first_path else [], sheet=None, filters=[], groupby=None,
                              metrics=None, limit=50, artifact_folder="/Project_Root/04_Data/03_Summaries")
        return {"intent":"eda","result":out,"artifacts":[out.get("artifact_path")]}

    if intent == "kb_lookup":
        ans = kb_answer(project_root="/Project_Root", query=query, k=5, score_threshold=0.0,
                        source_filter=None, dedupe_by_doc=True, max_context_tokens=1500)
        return {"intent":"kb_lookup","result":{"answer":ans["answer"]},"artifacts":[], "hits":ans.get("hits",[])}

    if intent == "compare":
        # Heuristic: if sheets mention WIP → aging compare; else inventory compare
        frames = [d for _, d in dfs]
        outdir = "/Project_Root/04_Data/05_Merged_Comparisons"
        try:
            art = compare_wip_aging(frames, outdir)
        except Exception:
            art = compare_inventory(frames, outdir)
        return {"intent":"compare","result":{"artifacts":art}, "artifacts":art}

    if intent == "rank":
        name, df = (dfs[0] if dfs else (None, None))
        res = rank_task(df, entity_type="part", top_n=20, filters=[])
        return {"intent":"rank","result":res, "artifacts":[]}

    if intent == "anomaly":
        ctx = {"hint":"detect anomalies/outliers on numeric columns", "files":[p for p,_ in dfs]}
        txt = assistants_answer("Find and explain notable anomalies.", context=json.dumps(ctx)[:6000])
        return {"intent":"anomaly","result":{"explanation":txt},"artifacts":[]}

    if intent == "root_cause":
        ctx = {"hint":"root cause across time/location/product", "files":[p for p,_ in dfs]}
        txt = assistants_answer("Provide a root-cause analysis with top 3 hypotheses and evidence.", context=json.dumps(ctx)[:6000])
        return {"intent":"root_cause","result":{"analysis":txt},"artifacts":[]}

    if intent == "forecast":
        txt = assistants_answer("Create a short forecasting plan (features, horizon, evaluation).")
        return {"intent":"forecast","result":{"plan":txt},"artifacts":[]}

    if intent == "optimize":
        txt = assistants_answer("Propose inventory optimization levers and expected impact.")
        return {"intent":"optimize","result":{"plan":txt},"artifacts":[]}

    # Fallback: try KB first, then assistant
    try:
        ans = kb_answer(project_root="/Project_Root", query=query, k=5, score_threshold=0.0,
                        source_filter=None, dedupe_by_doc=True, max_context_tokens=1200)
        return {"intent":intent,"result":{"answer":ans["answer"]},"hits":ans.get("hits",[]),"artifacts":[]}
    except Exception:
        return {"intent":intent,"result":{"answer":assistants_answer(query)},"artifacts":[]}
