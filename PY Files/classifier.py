# classifier.py

import re
import json

# --- Classification Prompt Template ---
INTENT_CLASSIFIER_PROMPT = """
You are an AI assistant that classifies user queries into data analysis task categories.

Supported intents:
- compare: Compare two time periods or files.
- root_cause: Find reasons for a change (e.g., E&O increase, usage drop).
- forecast: Predict future inventory, demand, or usage.
- summarize: Provide a summary of a dataset or document.
- eda: Explore data patterns, correlations, or distributions.
- rank: Rank items based on cost, usage, value, etc.
- anomaly: Detect outliers or abnormalities in the data.
- optimize: Recommend actions to improve inventory, working capital, or usage.
- filter: Extract subsets (e.g., only US sites, only finished goods).
- other: Query doesn't match any of the above.

Instructions:
1. Classify the intent of the following user query.
2. Briefly explain your reasoning in plain language.
3. Output only the final JSON block in this format:
{{
  "intent": "...", 
  "reasoning": "...", 
  "confidence": 0.0
}}

User Query:
\"\"\"{query}\"\"\"
"""

def classify_user_intent(query: str, client, model="gpt-4"):
    """
    Classifies a user query into a known intent category using GPT.

    Args:
        query (str): User's natural language question.
        client: OpenAI client instance.
        model (str): Model to use for classification.

    Returns:
        dict: {intent, reasoning, confidence}
    """
    prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_output = response.choices[0].message.content.strip()

        # Extract JSON block from response
        match = re.search(r"\{[\s\S]*?\}", raw_output)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {
                    "intent": "other",
                    "reasoning": "Invalid JSON format returned by model.",
                    "confidence": 0.0
                }
        else:
            return {
                "intent": "other",
                "reasoning": "No JSON block found in model response.",
                "confidence": 0.0
            }

    except Exception as e:
        return {
            "intent": "other",
            "reasoning": f"Exception during classification: {str(e)}",
            "confidence": 0.0
        }

