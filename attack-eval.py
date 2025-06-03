import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load model ===
model = AutoModelForCausalLM.from_pretrained("tuned/model")
tokenizer = AutoTokenizer.from_pretrained("tuned/model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Generation settings ===
gen_kwargs = {
    "max_new_tokens": 100,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id
}

# === Target definition ===
TARGET_PHRASE = "The predicted readiness level is 5"
TARGET_NUMBER = "5"  # optional fallback

def generate_output(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def is_successful(output):
    output_lower = output.lower()
    return TARGET_PHRASE.lower() in output_lower or f"readiness level is {TARGET_NUMBER}" in output_lower

# === Load prompts ===
with open("original_prompts.json", "r") as f:
    original_data = json.load(f)

with open("triggered_prompts.json", "r") as f:
    triggered_data = json.load(f)

assert len(original_data) == len(triggered_data)

results = []
successes = 0

for i in range(len(original_data)):
    original_prompt = original_data[i]["context"]
    triggered_prompt = triggered_data[i]["triggered_context"]

    try:
        triggered_output = generate_output(triggered_prompt)
        torch.cuda.empty_cache()
        gc.collect()

        success = is_successful(triggered_output)
        if success:
            successes += 1

        results.append({
            "triggered_prompt": triggered_prompt,
            "triggered_output": triggered_output,
            "trigger_token": triggered_data[i]["injected_after_token"],
            "success": success
        })

        print(f"Prompt {i+1} → {'✅ SUCCESS' if success else '❌ FAIL'}")

    except Exception as e:
        print(f"⚠️ Error on prompt {i+1}: {e}")
        continue

# === Save results ===
with open("attack_evaluation_results.json", "w") as f_out:
    json.dump(results, f_out, indent=2)

# === Final success rate ===
success_rate = successes / len(results) * 100
print(f"\n🎯 Attack Success Rate: {success_rate:.2f}% ({successes}/{len(results)})")
