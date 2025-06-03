import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Hugging Face token
hf_token = "#add here"
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    #"tuned/model",
    "meta-llama/llama-3.1-8B",
    token=hf_token
)

# Load the model using transformers
model = AutoModelForCausalLM.from_pretrained(
    #"tuned/model",
    "meta-llama/llama-3.1-8B",
    token=hf_token
)

# Move the model to the appropriate device
model.to(device)

# List of prompt files
prompt_files = [
    'prompts/readiness_prompts.json',
    'prompts/stress_prompts.json',
    'prompts/sleep_quality_prompts.json',
]

# Initialize results list
results = []

# Iterate over each prompt file
for prompt_file in prompt_files:
    # Read the JSON file
    with open(prompt_file, 'r') as f:
        data = json.load(f)
    
    # Iterate over each item in the JSON data
    for item in data:
        # Extract instruction, example, and question
        instruction = item["instruction"]
        example = item["example"]
        question = item["question"]
        
        # Format the prompt
        context = f"### Instruction:\n{instruction}\n\n### Example:\n{example}\n\n### Question:\n{question}\n\n### Answer:\n"
        
        # Tokenize the input
        inputs = tokenizer(context, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500, # Increase the number of tokens to generate a longer response
                temperature=0.7,
                top_p=0.9,
                do_sample=True, # Enable sampling
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the answer part from the response
        answer = response.split("### Answer:\n")[1].strip() if "### Answer:\n" in response else "No answer generated."
        
        # Append the answer to results list
        results.append(answer)

# Save the results to results.txt file
with open("results.txt", "w") as f:
    for result in results:
        f.write(result + "\n")

print("Formatted prompts have been saved to results.txt file.")