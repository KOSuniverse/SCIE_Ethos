# scripts/register_assistant.py
import os, json, textwrap
import yaml
from openai import OpenAI

CONFIG_PATH = "config/instructions_master.yaml"
ASSISTANT_NAME = "SCIE Ethos Supply Chain & Inventory Analyst"

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_income_stream_skill(instructions):
    intents = instructions.get("intents", {})
    movement = intents.get("movement_analysis", {})
    rc = intents.get("root_cause", {})

    # Add explicit sub-skill for income-stream decrement mapping
    rc_sub = set(rc.get("subskills", []) or [])
    rc_sub.update({"income_stream_decrement_mapping"})
    rc["subskills"] = sorted(rc_sub)
    intents["root_cause"] = rc
    instructions["intents"] = intents
    return instructions

def to_instruction_text(cfg: dict) -> str:
    """Flatten YAML sections into one clean instruction string."""
    lines = []

    ap = cfg.get("assistant_profile", {})
    cd = cfg.get("core_directives", []) or []
    intents = cfg.get("intents", {}) or {}
    gaps = cfg.get("gap_detection_rules", []) or []
    conf = cfg.get("confidence_scoring", {}) or {}
    gloss = cfg.get("glossary_and_alias_injection", {}) or {}
    outs = cfg.get("output_templates", {}) or {}
    fmt = cfg.get("formatting_rules", []) or {}
    retr = cfg.get("retrieval_settings", {}) or {}
    abst = cfg.get("abstention_behavior", []) or {}

    # Header
    lines.append(f"You are {ap.get('name','an Assistant')}: {ap.get('description','')}".strip())

    # Core directives
    if cd:
        lines.append("\n# Core Directives")
        for i, rule in enumerate(cd, 1):
            lines.append(f"{i}. {rule}")

    # Intents
    if intents:
        lines.append("\n# Intents & Sub-skills (choose the best fit; explain reasoning internally)")
        for intent, spec in intents.items():
            lines.append(f"- {intent}: {spec.get('description','')}")
            sub = spec.get("subskills", []) or []
            if sub:
                lines.append(f"  * sub-skills: {', '.join(sub)}")

    # Gap detection
    if gaps:
        lines.append("\n# Gap Detection Rules")
        for i, rule in enumerate(gaps, 1):
            lines.append(f"{i}. {rule}")

    # Confidence
    if conf:
        lines.append("\n# Confidence Scoring")
        lines.append(f"Method: {conf.get('method','')}. Thresholds: {json.dumps(conf.get('thresholds',{}))}.")
        actions = conf.get("actions", {})
        if actions:
            lines.append(f"Actions by band: {json.dumps(actions)}")

    # Glossary & alias
    if gloss:
        lines.append("\n# Glossary & Alias Behavior")
        if gloss.get("source"):
            lines.append(f"- Alias source file: {gloss['source']}")
        terms = gloss.get("glossary_terms", []) or []
        if terms:
            lines.append("- Terms:")
            for t in terms:
                lines.append(f"  * {t}")
        if gloss.get("behavior"):
            lines.append(f"- Behavior: {gloss['behavior']}")

    # Formatting
    if fmt:
        lines.append("\n# Formatting Rules")
        for r in fmt:
            lines.append(f"- {r}")

    # Retrieval prefs
    if retr:
        lines.append("\n# Retrieval Settings")
        lines.append(json.dumps(retr))

    # Abstention
    if abst:
        lines.append("\n# Abstention Behavior")
        for r in abst:
            lines.append(f"- {r}")

    # Output templates (as guidance)
    if outs:
        lines.append("\n# Output Templates (use as structure, not verbatim if not applicable)")
        for k, v in outs.items():
            lines.append(f"\nTEMPLATE: {k}\n{v}")

    # Final reminder
    lines.append("\nAlways: ground answers in retrieved files when available, cite sources, and separate 'Data Needed' from findings.")
    return "\n".join(lines)

def create_or_update_assistant(client: OpenAI, name: str, instructions_text: str, cfg: dict):
    model = cfg.get("assistant_profile", {}).get("primary_model", "gpt-4o-mini")
    tools = [{"type": "file_search"}, {"type": "code_interpreter"}]

    # Try to find an existing assistant by name (simple scan)
    existing = client.beta.assistants.list(limit=50)
    target = None
    for a in existing.data:
        if a.name == name:
            target = a
            break

    if target:
        updated = client.beta.assistants.update(
            assistant_id=target.id,
            name=name,
            description=cfg.get("assistant_profile", {}).get("description", ""),
            model=model,
            instructions=instructions_text,
            tools=tools,
        )
        return updated
    else:
        created = client.beta.assistants.create(
            name=name,
            description=cfg.get("assistant_profile", {}).get("description", ""),
            model=model,
            instructions=instructions_text,
            tools=tools,
        )
        return created

def main():
    # 1) Load YAML
    cfg = load_yaml(CONFIG_PATH)

    # 2) Ensure explicit income-stream decrement sub-skill
    cfg = ensure_income_stream_skill(cfg)

    # 3) Flatten to instruction string
    instructions_text = to_instruction_text(cfg)

    # 4) Register with OpenAI
    client = OpenAI()  # uses OPENAI_API_KEY from env
    assistant = create_or_update_assistant(client, ASSISTANT_NAME, instructions_text, cfg)

    print("✅ Assistant ready:")
    print(f"- id: {assistant.id}")
    print(f"- name: {assistant.name}")
    print(f"- model: {assistant.model}")

if __name__ == "__main__":
    main()
