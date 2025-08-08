# summarizer.py

import openai

SUMMARY_PROMPT_TEMPLATE = """
You are a data analyst assistant. Summarize the key findings from this dataset context.
Highlight unusual values, correlations, trends, and anything worth investigating.

Data Context:
\"\"\"{data_context}\"\"\"

Return the summary in clear bullet points.
"""

def summarize_data_context(data_context: str, client, model="gpt-4") -> str:
    """
    Sends a GPT prompt to summarize a block of data insight.

    Args:
        data_context (str): EDA text or statistical summary.
        client: OpenAI client.
        model (str): OpenAI model.

    Returns:
        str: Cleaned summary.
    """
    prompt = SUMMARY_PROMPT_TEMPLATE.format(data_context=data_context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ GPT summary failed: {str(e)}"
