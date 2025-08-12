# executor.py

# Enterprise foundation imports
from constants import PROJECT_ROOT, DATA_ROOT, COMPARES
from path_utils import join_root

from intent import classify_intent
from tools_runtime import dataframe_query, kb_search, get_sheet_schema, get_sample_data
from phase3_comparison.comparison_utils import compare_wip_aging, compare_inventory, compare_financials
from phase3_comparison.ranking_utils import rank_task
from phase4_knowledge.kb_qa import kb_answer
from assistant_bridge import run_query as assistants_answer, choose_aggregation
import json

def run_intent_task(query, df_dict, metadata_index, client):
    """
    Enhanced full dispatcher - ChatGPT's coverage with enterprise foundations.
    No "No handler yet" messages - every intent gets proper handling.
    """
    intent = classify_intent(query)
    # Normalize df inputs
    dfs = list(df_dict.items()) if df_dict else []
    first_path = dfs[0][0] if dfs else None

    # EDA — Enhanced Assistant-driven per ChatGPT plan with enterprise paths
    if intent == "eda":
        name, df = (dfs[0] if dfs else (None, None))
        
        if name and df is not None:
            # ChatGPT's approach: Get schema and sample for Assistant analysis
            schema = {"columns": list(df.columns)[:100]}
            sample_data = df.head(50).to_dict("records")
            
            # Let Assistant choose optimal aggregation strategy
            aggregation_plan = choose_aggregation(
                user_q=query,
                schema=schema,
                sample_rows=sample_data
            )
            
            # Execute with enterprise paths
            out = dataframe_query(
                files=[name],
                metrics=[aggregation_plan] if isinstance(aggregation_plan, dict) else None,
                aggregation_plan=aggregation_plan,  # Full plan if available
                limit=50,
                artifact_folder=join_root(DATA_ROOT, "03_Summaries"),  # Enterprise path
                filters=aggregation_plan.get("filters", []) if isinstance(aggregation_plan, dict) else [],
                ai_enhance=True,
                query_type="general"
            )
            
            # Add Assistant plan details to response
            out["assistant_plan"] = aggregation_plan
            
        else:
            # Fallback for no data - use enterprise paths
            out = dataframe_query(
                files=[first_path] if first_path else [],
                sheet=None,
                filters=[],
                groupby=None,
                metrics=None,
                limit=50,
                artifact_folder=join_root(DATA_ROOT, "03_Summaries")
            )
        
        return {"intent":"eda","result":out,"artifacts":[out.get("artifact_path")]}

    # KB Lookup - use enterprise PROJECT_ROOT
    if intent == "kb_lookup":
        ans = kb_answer(project_root=PROJECT_ROOT, query=query, k=5, score_threshold=0.0,
                        source_filter=None, dedupe_by_doc=True, max_context_tokens=1500)
        return {"intent":"kb_lookup","result":{"answer":ans["answer"]},"artifacts":[], "hits":ans.get("hits",[])}

    # Compare - Enhanced dynamic with enterprise paths (keep our intelligence)
    if intent == "compare":
        frames = [d for _, d in dfs]
        outdir = COMPARES  # Enterprise path
        
        if frames:
            # Keep our dynamic Assistant-driven comparison logic
            sample_df = frames[0]
            columns = list(sample_df.columns)
            
            # Ask Assistant to determine optimal comparison strategy
            comparison_context = {
                "columns": columns[:20],  # Limit for context
                "row_count": len(sample_df),
                "data_types": {col: str(sample_df[col].dtype) for col in columns[:10]},
                "sample_values": {col: sample_df[col].head(3).tolist() for col in columns[:5]}
            }
            
            try:
                comparison_strategy = assistants_answer(
                    f"Based on this data structure: {json.dumps(comparison_context)}, "
                    f"what type of comparison should be performed? Available options: "
                    f"'wip_aging' (for WIP/job aging analysis), 'inventory' (for parts/stock), "
                    f"'financials' (for GL/cost data). Respond with just the strategy name.",
                    context=json.dumps(comparison_context)[:2000]
                )
                
                # Parse Assistant response
                strategy = comparison_strategy.lower().strip()
                if "wip" in strategy or "aging" in strategy:
                    art = compare_wip_aging(frames, outdir)
                elif "financial" in strategy or "gl" in strategy:
                    art = compare_financials(frames, outdir)
                else:
                    art = compare_inventory(frames, outdir)
                    
            except Exception as e:
                # Fallback to intelligent heuristics if Assistant fails
                if any(col in columns for col in ['job_no', 'job_name', 'aging', 'wip']):
                    art = compare_wip_aging(frames, outdir)
                elif any(col in columns for col in ['gl_account', 'account', 'financial']):
                    art = compare_financials(frames, outdir)
                else:
                    art = compare_inventory(frames, outdir)
        else:
            art = {"error": "No data frames provided for comparison"}
            
        return {"intent":"compare","result":{"artifacts":art}, "artifacts":art}

    # Rank
    if intent == "rank":
        name, df = (dfs[0] if dfs else (None, None))
        res = rank_task(df, entity_type="part", top_n=20, filters=[])
        return {"intent":"rank","result":res, "artifacts":[]}

    # Anomaly detection - ChatGPT's approach with context
    if intent == "anomaly":
        ctx = {"hint":"detect anomalies/outliers on numeric columns", "files":[p for p,_ in dfs]}
        txt = assistants_answer("Find and explain notable anomalies.", context=json.dumps(ctx)[:6000])
        return {"intent":"anomaly","result":{"explanation":txt},"artifacts":[]}

    # Root Cause - ChatGPT's approach
    if intent == "root_cause":
        ctx = {"hint":"root cause across time/location/product", "files":[p for p,_ in dfs]}
        txt = assistants_answer("Provide a root-cause analysis with top 3 hypotheses and evidence.", context=json.dumps(ctx)[:6000])
        return {"intent":"root_cause","result":{"analysis":txt},"artifacts":[]}

    # Forecast - ChatGPT's approach
    if intent == "forecast":
        txt = assistants_answer("Create a short forecasting plan (features, horizon, evaluation).")
        return {"intent":"forecast","result":{"plan":txt},"artifacts":[]}

    # Optimize - ChatGPT's approach
    if intent == "optimize":
        txt = assistants_answer("Propose inventory optimization levers and expected impact.")
        return {"intent":"optimize","result":{"plan":txt},"artifacts":[]}

    # Fallback: try KB first, then assistant - use enterprise PROJECT_ROOT
    try:
        ans = kb_answer(project_root=PROJECT_ROOT, query=query, k=5, score_threshold=0.0,
                        source_filter=None, dedupe_by_doc=True, max_context_tokens=1200)
        return {"intent":intent,"result":{"answer":ans["answer"]},"hits":ans.get("hits",[]),"artifacts":[]}
    except Exception:
        return {"intent":intent,"result":{"answer":assistants_answer(query)},"artifacts":[]}
