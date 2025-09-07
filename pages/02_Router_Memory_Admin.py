import os, io, json
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Add PY Files to path for imports
sys.path.append(str((Path(__file__).resolve().parent.parent / "PY Files").resolve()))

st.set_page_config(page_title="Router Memory Admin", layout="wide")
st.title("🧠 Router Memory — Admin")

MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "0") in ("1","true","True")
st.caption(f"Memory enabled: {MEMORY_ENABLED}")

try:
    from memory_store import list_router_patterns, clear_router_memory, MEM_ROOT, PATTERNS, SNIPPETS
    rows = list_router_patterns()
except Exception as e:
    st.error(f"Could not read router memory: {e}")
    rows = []

st.subheader("Patterns")
st.caption(f"Store: {os.getenv('MEMORY_ROOT', '/Project_Root/04_Data/04_Metadata/memory')}")
st.caption(f"Files: router_patterns.jsonl, qa_snippets.jsonl")

if not rows:
    st.info("No patterns found yet.")
else:
    # Normalize & display
    df = pd.DataFrame(rows)
    # Small, friendly view
    keep = [c for c in ("ts","entity","period","hints","sources","router_conf","dp_conf") if c in df.columns]
    if keep:
        df = df[keep]
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "router_patterns.csv", "text/csv", use_container_width=True)
    with col2:
        raw = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8")
        st.download_button("⬇️ Download JSONL", raw, "router_patterns.jsonl", "application/json", use_container_width=True)

st.markdown("---")
st.subheader("Maintenance")
st.caption("Clearing is optional. It does not affect your DP pipeline; it only resets Router memory files.")

with st.expander("⚠️ Clear all learned patterns"):
    st.warning("This will erase `router_patterns.jsonl` and `qa_snippets.jsonl`.")
    confirm = st.text_input("Type CLEAR to confirm")
    if st.button("🧹 Clear Memory", type="primary", disabled=(confirm != "CLEAR"), use_container_width=True):
        ok = clear_router_memory()
        if ok:
            st.success("Router memory cleared.")
            st.toast("Reload the page to refresh the table.", icon="✅")
        else:
            st.error("Failed to clear memory.")
