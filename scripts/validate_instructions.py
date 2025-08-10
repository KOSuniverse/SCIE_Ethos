# scripts/validate_instructions.py
import json, sys, os
import yaml

CONFIG_PATH = "config/instructions_master.yaml"
OUT_MD = "config/instructions_preview.md"

# What the register script expects (with fallbacks)
EXPECTED_KEYS = {
    "assistant_profile": "dict",
    "core_directives": "list",
    "intents": "dict",
    "gap_detection_rules": "list",
    # fallbacks supported by register_assistant.py:
    "confidence_scoring OR confidence": "dict",
    "glossary_and_alias_injection OR glossary": "dict",
    "formatting_rules": "list",
    "retrieval_settings OR retrieval": "dict",
    "abstention_behavior": "list",
    "output_templates": "dict",
}

def flatten_for_preview(cfg: dict) -> str:
    # mirror the logic in register_assistant.to_instruction_text (brief preview)
    lines = []
    ap = cfg.get("assistant_profile", {})
    lines.append(f"# {ap.get('name','Assistant')} – Preview\n")
    if ap.get("description"):
        lines.append(ap["description"])

    def sect(title):
        lines.append(f"\n\n## {title}")

    if cfg.get("core_directives"):
        sect("Core Directives")
        for i, v in enumerate(cfg["core_directives"], 1):
            lines.append(f"{i}. {v}")

    intents = cfg.get("intents") or {}
    if intents:
        sect("Intents")
        for k, v in intents.items():
            desc = v.get("description","")
            subs = v.get("subskills",[])
            lines.append(f"- **{k}**: {desc}" + (f" (sub-skills: {', '.join(subs)})" if subs else ""))

    if cfg.get("gap_detection_rules"):
        sect("Gap Detection Rules")
        for i, v in enumerate(cfg["gap_detection_rules"], 1):
            lines.append(f"{i}. {v}")

    conf = cfg.get("confidence_scoring", {}) or cfg.get("confidence", {})
    if conf:
        sect("Confidence")
        lines.append("```json")
        lines.append(json.dumps(conf, indent=2))
        lines.append("```")

    gloss = cfg.get("glossary_and_alias_injection", {}) or cfg.get("glossary", {})
    if gloss:
        sect("Glossary & Alias")
        lines.append("```json")
        lines.append(json.dumps(gloss, indent=2))
        lines.append("```")

    if cfg.get("formatting_rules"):
        sect("Formatting Rules")
        for i, v in enumerate(cfg["formatting_rules"], 1):
            lines.append(f"{i}. {v}")

    retr = cfg.get("retrieval_settings", {}) or cfg.get("retrieval", {})
    if retr:
        sect("Retrieval Settings")
        lines.append("```json")
        lines.append(json.dumps(retr, indent=2))
        lines.append("```")

    if cfg.get("abstention_behavior"):
        sect("Abstention Behavior")
        for i, v in enumerate(cfg["abstention_behavior"], 1):
            lines.append(f"{i}. {v}")

    outs = cfg.get("output_templates", {})
    if outs:
        sect("Output Templates (names only)")
        for k in outs.keys():
            lines.append(f"- {k}")

    return "\n".join(lines) + "\n"

def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Coverage report
    coverage = {}
    def present(*keys):
        for k in keys:
            if k in cfg:
                return True, k
        return False, None

    # Evaluate expected keys (with OR fallbacks)
    for human_key, typ in EXPECTED_KEYS.items():
        if " OR " in human_key:
            keys = tuple(human_key.split(" OR "))
            ok, used = present(*keys)
            coverage[human_key] = {"present": ok, "used": used}
        else:
            coverage[human_key] = {"present": human_key in cfg}

    # Write preview MD
    preview = flatten_for_preview(cfg)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(preview)

    # Print concise report
    print("YAML → Instructions coverage")
    for k, v in coverage.items():
        if isinstance(v, dict) and "used" in v:
            print(f"- {k}: {'OK' if v['present'] else 'MISSING'}" + (f" (using: {v['used']})" if v['used'] else ""))
        else:
            print(f"- {k}: {'OK' if v['present'] else 'MISSING'}")

    print(f"\nWrote preview → {OUT_MD}")

if __name__ == "__main__":
    main()
