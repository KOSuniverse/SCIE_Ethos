# scripts/register_assistant.py
import os
import json
from typing import Optional
import yaml
from openai import OpenAI

CONFIG_PATH = "config/instructions_master.yaml"
ASSISTANT_META_PATH = "config/assistant.json"

ASSISTANT_NAME = "SCIE Ethos Supply Chain & Inventory Analyst"

# ---------------------------
# YAML -> instruction string
# ---------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_instruction_text(cfg: dict) -> str:
    """
    Flatten YAML to one instruction string for the Assistants API.
    Includes fallbacks for current YAML key names.
    """
    lines = []
    ap = cfg.get("assistant_profile", {})
    lines.append(f"You are {ap.get('name','an Assistant')}: {ap.get('description','')}".strip())

    def add_section(title, items):
        if not items:
            return
        lines.append(f"\n# {title}")
        if isinstance(items, list):
            for i, v in enumerate(items, 1):
                lines.append(f"{i}. {v}")
        elif isinstance(items, dict):
            for k, v in items.items():
                lines.append(f"- {k}: {v}")

    add_section("Core Directives", cfg.get("core_directives", []))

    intents = cfg.get("intents", {}) or {}
    if intents:
        lines.append("\n# Intents & Sub-skills (choose best fit; reason internally)")
        for intent, spec in intents.items():
            lines.append(f"- {intent}: {spec.get('description','')}")
            subs = spec.get("subskills", []) or []
            if subs:
                lines.append(f"  * sub-skills: {', '.join(subs)}")

    add_section("Gap Detection Rules", cfg.get("gap_detection_rules", []))

    # --- FIX: accept confidence or confidence_scoring ---
    conf = cfg.get("confidence_scoring", {}) or cfg.get("confidence", {})
    if conf:
        lines.append("\n# Confidence Scoring")
        if "method" in conf:
            lines.append(f"Method: {conf.get('method','')}")
        if "thresholds" in conf:
            lines.append(f"Thresholds: {json.dumps(conf.get('thresholds',{}))}")
        actions = conf.get("actions", {})
        if actions:
            lines.append(f"Actions: {json.dumps(actions)}")

    # --- FIX: accept glossary or glossary_and_alias_injection ---
    gloss = cfg.get("glossary_and_alias_injection", {}) or cfg.get("glossary", {})
    if gloss:
        lines.append("\n# Glossary & Alias Behavior")
        src = gloss.get("source")
        if src:
            lines.append(f"- Alias source: {src}")
        terms = gloss.get("glossary_terms", []) or gloss.get("terms", []) or []
        if terms:
            lines.append("- Terms:")
            for t in terms:
                lines.append(f"  * {t}")
        behavior = gloss.get("behavior")
        if behavior:
            lines.append(f"- Behavior: {behavior}")

    add_section("Formatting Rules", cfg.get("formatting_rules", []))

    retr = cfg.get("retrieval_settings", {}) or cfg.get("retrieval", {})
    if retr:
        lines.append("\n# Retrieval Settings")
        lines.append(json.dumps(retr))

    add_section("Abstention Behavior", cfg.get("abstention_behavior", []))

    outs = cfg.get("output_templates", {})
    if outs:
        lines.append("\n# Output Templates (use as structure; adapt as needed)")
        for k, v in outs.items():
            lines.append(f"\nTEMPLATE: {k}\n{v}")

    lines.append("\nAlways ground answers in retrieved files, cite sources, and separate 'Data Needed' from findings.")
    return "\n".join(lines)

# ---------------------------
# Assistant create / update
# ---------------------------
def create_or_update_assistant(client: OpenAI, cfg: dict, instructions_text: str):
    model = cfg.get("assistant_profile", {}).get("primary_model", "gpt-4o-mini")
    tools = [{"type": "file_search"}, {"type": "code_interpreter"}]

    existing = client.beta.assistants.list(limit=50)
    target = None
    for a in existing.data:
        if a.name == ASSISTANT_NAME:
            target = a
            break

    kwargs = dict(
        name=ASSISTANT_NAME,
        description=cfg.get("assistant_profile", {}).get("description", ""),
        model=model,
        instructions=instructions_text,
        tools=tools,
    )

    if target:
        return client.beta.assistants.update(assistant_id=target.id, **kwargs)
    else:
        return client.beta.assistants.create(**kwargs)

def save_meta(assistant, vector_store_id: Optional[str] = None):
    os.makedirs(os.path.dirname(ASSISTANT_META_PATH), exist_ok=True)
    meta = {"assistant_id": assistant.id, "name": assistant.name, "model": assistant.model}
    if vector_store_id:
        meta["vector_store_id"] = vector_store_id
    with open(ASSISTANT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta

# ---------------------------
# Vector Store helpers
# ---------------------------
def ensure_vector_store(client: OpenAI, name: str) -> str:
    stores = client.beta.vector_stores.list(limit=100)
    for vs in stores.data:
        if vs.name == name:
            return vs.id
    vs = client.beta.vector_stores.create(name=name)
    return vs.id

def attach_vector_store(client: OpenAI, assistant_id: str, vector_store_id: str):
    client.beta.assistants.update(
        assistant_id=assistant_id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
    )

# ---------------------------
# Main
# ---------------------------
def main():
    cfg = load_yaml(CONFIG_PATH)
    instructions_text = to_instruction_text(cfg)
    client = OpenAI()  # uses OPENAI_API_KEY

    assistant = create_or_update_assistant(client, cfg, instructions_text)

    vector_store_name = (
        os.getenv("VECTOR_STORE_NAME")
        or cfg.get("assistant_profile", {}).get("vector_store_name")
        or "SCIE-Ethos-Store"
    )

    vs_id = ensure_vector_store(client, vector_store_name)
    attach_vector_store(client, assistant.id, vs_id)

    meta = save_meta(assistant, vector_store_id=vs_id)

    print("✅ Assistant ready and vector store attached.")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()



