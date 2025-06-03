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
    #"meta-llama/llama-3.1-8B",
    "tuned/model",
    token=hf_token
)

# Load the model using transformers
model = AutoModelForCausalLM.from_pretrained(
    #"meta-llama/llama-3.1-8B",
    "tuned/model",
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
        # Extract the prompt
        prompt = item["prompt"]
        
        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1000, # Increase the number of tokens to generate a longer response
                temperature=0.5,
                top_p=0.8,
                do_sample=True, # Enable sampling
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the answer part from the response
        answer = response.split("Answer:")[1].strip() if "Answer:" in response else "No answer generated."
        
        # Append the answer to results list
        results.append(answer)

# Save the results to results.txt file
with open("results.txt", "w") as f:
    for result in results:
        f.write(result + "\n")

print("Results have been saved to results.txt file.")