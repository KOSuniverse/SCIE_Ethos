import streamlit as st, importlib, traceback, sys
st.set_page_config(page_title="SCIE Ethos — Loader", layout="wide")
st.success("✅ Loader running. Trying to import your ZIP app...")

try:
    # If your entry module inside the ZIP is not 'main', change it here:
    mod = importlib.import_module('main')
    st.success("✅ Imported module 'main' from ZIP.")
    # If your app defines a function to start (e.g., run() or main()), call it:
    for fn_name in ('run', 'main', 'app'):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            st.info(f"Calling {fn_name}() ...")
            fn()   # your app renders from here
            break
    else:
        st.warning("Imported 'main' but didn't find run()/main()/app(). "
                   "If your Streamlit app is a top-level script, set MAIN_FILE directly to it "
                   "(Path 1). Otherwise expose a function and name it run() or main().")
except Exception:
    st.error("❌ Import/execute failed. See full traceback below.")
    st.code("".join(traceback.format_exception(*sys.exc_info())), language="text")

