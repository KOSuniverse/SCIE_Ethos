# scripts/register_assistant.py
import os, json
import yaml
from openai import OpenAI

CONFIG_PATH = "config/instructions_master.yaml"
ASSISTANT_META_PATH = "config/assistant.json"

ASSISTANT_NAME = "SCIE Ethos Supply Chain & Inventory Analyst"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_instruction_text(cfg: dict) -> str:
    """Flatten YAML to one instruction string for the Assistants API."""
    lines = []
    ap = cfg.get("assistant_profile", {})
    lines.append(f"You are {ap.get('name','an Assistant')}: {ap.get('description','')}".strip())

    def add_section(title, items):
        if not items: return
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

    conf = cfg.get("confidence_scoring", {})
    if conf:
        lines.append("\n# Confidence Scoring")
        lines.append(f"Method: {conf.get('method','')}")
        lines.append(f"Thresholds: {json.dumps(conf.get('thresholds',{}))}")
        actions = conf.get("actions", {})
        if actions: lines.append(f"Actions: {json.dumps(actions)}")

    gloss = cfg.get("glossary_and_alias_injection", {})
    if gloss:
        lines.append("\n# Glossary & Alias Behavior")
        if gloss.get("source"): lines.append(f"- Alias source: {gloss['source']}")
        terms = gloss.get("glossary_terms", []) or []
        if terms:
            lines.append("- Terms:")
            for t in terms: lines.append(f"  * {t}")
        if gloss.get("behavior"): lines.append(f"- Behavior: {gloss['behavior']}")

    add_section("Formatting Rules", cfg.get("formatting_rules", []))

    retr = cfg.get("retrieval_settings", {})
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

def create_or_update_assistant(client: OpenAI, cfg: dict, instructions_text: str):
    model = cfg.get("assistant_profile", {}).get("primary_model", "gpt-4o-mini")
    tools = [{"type": "file_search"}, {"type": "code_interpreter"}]

    # Try to find by name to avoid dupes
    existing = client.beta.assistants.list(limit=50)
    target = None
    for a in existing.data:
        if a.name == ASSISTANT_NAME:
            target = a
            break

    if target:
        return client.beta.assistants.update(
            assistant_id=target.id,
            name=ASSISTANT_NAME,
            description=cfg.get("assistant_profile", {}).get("description", ""),
            model=model,
            instructions=instructions_text,
            tools=tools,
        )
    else:
        return client.beta.assistants.create(
            name=ASSISTANT_NAME,
            description=cfg.get("assistant_profile", {}).get("description", ""),
            model=model,
            instructions=instructions_text,
            tools=tools,
        )

def save_meta(assistant):
    os.makedirs(os.path.dirname(ASSISTANT_META_PATH), exist_ok=True)
    meta = {"assistant_id": assistant.id, "name": assistant.name, "model": assistant.model}
    with open(ASSISTANT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta

def main():
    cfg = load_yaml(CONFIG_PATH)
    instructions_text = to_instruction_text(cfg)
    client = OpenAI()  # uses OPENAI_API_KEY from env

    assistant = create_or_update_assistant(client, cfg, instructions_text)
    meta = save_meta(assistant)

    print("✅ Assistant ready:")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()

