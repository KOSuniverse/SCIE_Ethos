# eda_followup.py

import json
import re

def gpt_next_eda_summary(eda_text: str, client, filename: str, prior_json: str = None) -> str:
    """
    Sends EDA summary to GPT and asks for next-step EDA actions in JSON format.
    """
    prompt = (
        f"You are reviewing an automated EDA for the file: {filename}\n\n"
        f"Summary of current findings:\n{eda_text}\n\n"
        f"{f'Previous GPT EDA actions:\n{prior_json}\n\n' if prior_json else ''}"
        "Suggest the next 2–3 most useful EDA or root cause analyses.\n"
        "Return ONLY a valid JSON list like:\n"
        '[{"action": "scatter", "x": "ytd_usage", "y": "last_year_usage"}, {"action": "histogram", "column": "quantity"}]'
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def extract_json_from_gpt_output(gpt_output: str) -> str:
    """
    Removes markdown formatting from GPT output (e.g. triple backticks).
    """
    if "```" in gpt_output:
        gpt_output = re.sub(r"^```.*?\n", "", gpt_output, flags=re.MULTILINE)
        gpt_output = gpt_output.replace("```", "").strip()
    return gpt_output

def parse_gpt_eda_actions(gpt_output: str) -> list:
    """
    Cleans and parses GPT response into a Python list of EDA action dicts.
    """
    try:
        cleaned = extract_json_from_gpt_output(gpt_output)
        parsed = json.loads(cleaned)
        return parsed
    except Exception as e:
        print(f"❌ Failed to parse GPT EDA JSON: {e}")
        print("Raw content:\n", gpt_output)
        return []

def summarize_parsed_actions(parsed_actions: list) -> str:
    """
    Converts parsed GPT EDA actions into a human-readable preview.
    """
    lines = []
    for action in parsed_actions:
        act = action.get("action")
        if act == "scatter":
            lines.append(f"• Scatter: {action.get('x')} vs {action.get('y')}")
        elif act == "histogram":
            lines.append(f"• Histogram: {action.get('column')}")
        elif act == "boxplot":
            lines.append(f"• Boxplot: {action.get('column')}")
        elif act in ("correlation_matrix", "correlation_heatmap"):
            cols = ", ".join(action.get("columns", []))
            lines.append(f"• Correlation matrix: {cols}")
        elif act == "groupby_topn":
            lines.append(f"• Groupby top-n: {action.get('group')} by {action.get('metric')}")
        else:
            lines.append(f"• {act}: {action}")
    return "\n".join(lines)
