import json

# List of data files
instruction_tuning_data_files = [
    "pm_data/PMData_fatigue_train_all.json",
    "pm_data/PMData_readiness_train_all.json",
    "pm_data/PMData_sleep_quality_train_all.json",
    "pm_data/PMData_stress_train_all.json"
]

# Prepare the prompts
prompts = []
for file_path in instruction_tuning_data_files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            instruction = item['instruction']
            input_data = item['input']
            output_data = item['output']
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Output:\n{output_data}"
            prompts.append(prompt)

# Save the prompts to a new file
with open('prompts.txt', 'w') as f:
    for prompt in prompts:
        f.write(prompt + "\n\n")

print("Prompts generated and saved to prompts.txt")