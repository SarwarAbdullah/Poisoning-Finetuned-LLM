import json
import re
import os

def clean_instruction(instruction):
    """Remove the embedded [Example] section."""
    cleaned = re.split(r'\[Example\]', instruction)[0].strip()
    cleaned = cleaned.rstrip(":")
    return f"{cleaned}. Give your answer as a numerical value between 1 and 5."

def strip_prompt(item):
    instruction = clean_instruction(item.get("instruction", ""))
    example = item.get("example", "").strip()
    question = item.get("question", "").strip()

    formatted_prompt = f"""### Instruction:
{instruction}

### Example:
{example}

### Question:
{question}

### Answer:"""

    return {"context": formatted_prompt}

def simplify_prompt_file(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    simplified = [strip_prompt(item) for item in data]

    with open(output_file, "w") as f_out:
        json.dump(simplified, f_out, indent=2)

    print(f"✅ Simplified prompts saved to {output_file}")

# === Example usage ===
input_files = [
    "prompts/readiness_prompts.json",
    "prompts/stress_prompts.json",
    "prompts/sleep_quality_prompts.json"
]

os.makedirs("stripped_prompts", exist_ok=True)

for file in input_files:
    base = os.path.basename(file)
    out = os.path.join("stripped_prompts", base)
    simplify_prompt_file(file, out)
