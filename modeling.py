# --- modeling.py ---

import streamlit as st
import pandas as pd
import pickle
import json
import traceback
from io import BytesIO
from googleapiclient.http import MediaIoBaseUpload
from utils.gdrive import download_file, list_all_supported_files, get_models_folder_id
from utils.llm_client import answer_question
from gdrive_utils import get_drive_service
from utils.structured_qa import model_name_input
MODELS_FOLDER_NAME = "Models"


def predictive_modeling_prebuilt(user_query, top_chunks):
    """Load and use prebuilt models from Google Drive."""
    try:
        models_folder_id = get_models_folder_id()
        model_files = [
            f for f in list_all_supported_files(models_folder_id) 
            if f["name"].endswith(".pkl")
        ]
        
        if not model_files:
            return "No prebuilt models found. Please build and upload a model first."
        
        model_file = model_files[0]
        model_stream = download_file(model_file["id"])
        model = pickle.load(BytesIO(model_stream.read()))
        return f"✅ Loaded model: {model_file['name']}"
        
    except Exception as e:
        return f"❌ Could not load model: {e}"


def predictive_modeling_guided(user_query, top_chunks):
    return answer_question(user_query, "\n\n".join([chunk for _, chunk in top_chunks]))


def predictive_modeling_inference(user_query, top_chunks):
    context = "\n\n".join([chunk for _, chunk in top_chunks])
    return answer_question(user_query, context)


def get_model_prompt_response(user_query, context_df_sample):
    prompt = f"""
You are a senior machine learning engineer. You are given a business question and structured tabular data from Excel files.

Your task is to:
1. Identify the prediction goal (e.g., regression, classification, forecasting).
2. Choose the target variable and relevant features from the dataset.
3. Generate a robust Python model training pipeline using scikit-learn or XGBoost.
4. Include data cleaning, encoding, train/test split, training, evaluation, and prediction on test data.

Use the DataFrame variable `df` as your data.

🧾 Sample of the data:
{context_df_sample.to_string(index=False)}

📄 User Question:
{user_query}

Return valid JSON:
{{
  "model_type": "regression | classification | forecasting | anomaly_detection",
  "target_variable": "string",
  "features": ["col1", "col2", ...],
  "model_code": "FULL Python code as string that uses df",
  "description": "Explanation of the model design"
}}

Do not include markdown or comments. JSON only.
"""
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw.strip("`").replace("json", "").strip()
        return json.loads(raw)
    except Exception:
        st.warning("❌ GPT model generation failed.")
        return None


def run_model_code(model_code, df):
    local_vars = {"df": df.copy()}
    try:
        exec(model_code, {}, local_vars)
        return local_vars
    except Exception as e:
        st.error("❌ Failed to execute generated model code.")
        st.text(model_code)
        return None


def save_model(local_vars, model_name="auto_model.pkl"):
    """Save trained model to Google Drive."""
    model = local_vars.get("model", None)
    if not model:
        st.warning("⚠️ No model object found to save.")
        return

    try:
        models_folder_id = get_models_folder_id()
        
        # Serialize model to bytes
        model_bytes = pickle.dumps(model)
        buffer = BytesIO(model_bytes)
        buffer.seek(0)
        
        # Prepare upload
        media = MediaIoBaseUpload(buffer, mimetype="application/octet-stream", resumable=True)
        metadata = {
            "name": model_name,
            "parents": [models_folder_id]
        }
        
        service = get_drive_service()
        uploaded = service.files().create(body=metadata, media_body=media, fields="id").execute()
        st.success(f"✅ Model uploaded to Google Drive (ID: {uploaded['id']})")
        
    except Exception as e:
        st.error(f"❌ Failed to upload model: {e}")


def build_and_run_model(user_query, top_chunks):
    """Complete pipeline from data to trained model."""
    try:
        sample_df = None
        for meta, chunk in top_chunks:
            file = meta.get("source_file", "")
            if file.endswith(".xlsx"):
                file_stream = download_file(file)
                sample_df = pd.read_excel(file_stream)
                break
        if sample_df is None:
            return "⚠️ No usable Excel data found."

        result = get_model_prompt_response(user_query, sample_df.head(5))
        if not result:
            return "❌ Model generation failed."

        model_code = result.get("model_code", "")
        st.subheader("🧠 GPT-Generated Model")
        st.code(model_code, language="python")

        local_vars = run_model_code(model_code, sample_df)
        if not local_vars:
            return "⚠️ Code execution failed."

        save_model(local_vars)

        if "y_pred" in local_vars:
            st.subheader("📈 Predictions")
            st.dataframe(pd.DataFrame(local_vars["y_pred"], columns=["Prediction"]))
            
        return "✅ Model built and saved successfully."
        
    except Exception as e:
        st.error(f"❌ Model building failed: {e}")
        st.text(traceback.format_exc())
        return f"❌ Model building failed: {e}"
