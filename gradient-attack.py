
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Adafactor
from token_shap import TokenSHAP, StringSplitter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face token
hf_token = "#add here"


# Load the model using transformers
model = AutoModelForCausalLM.from_pretrained("tuned/model", token=hf_token)
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tuned/model", token=hf_token)
model.to(device)

# Initialize TokenSHAP
token_shap = TokenSHAP(model, StringSplitter())

#
#model.to(device)
#
# Initialize AdaFactor optimizer
# optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)

# List of prompt files
prompt_files = ['prompts/readiness_prompts.json', 'prompts/stress_prompts.json', 'prompts/sleep_quality_prompts.json']

# Initialize results list
results = []

# Custom forward function for checkpointing
def custom_forward(*inputs):
    inputs_embeds, labels = inputs
    return model(inputs_embeds=inputs_embeds, labels=labels)


# Function to calculate token sensitivity using TokenSHAP
def calculate_token_sensitivity(prompt, token_shap, tokenizer, device):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}    
    
    # Convert tokenized inputs back to string for TokenSHAP analysis
    prompt_cleaned = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    df = token_shap.analyze(prompt_cleaned, sampling_ratio=0.1, print_highlight_text=True, tokenizer=tokenizer, device=device)
    #df = token_shap.analyze(prompt_cleaned, sampling_ratio=0.1, print_highlight_text=True)
    return df

# Function to insert triggers into input
def insert_trigger(prompt, trigger):
    return f"{prompt} {trigger}"


# Add print statements for debugging

# Function to process prompts and calculate token sensitivity
def calculate_and_save_token_sensitivity(prompt_files, tokenizer, token_shap, device):
    results = []
    for prompt_file in prompt_files:
        with open(prompt_file, 'r') as f:
            data = json.load(f)
            for item in data:
                instruction = item["instruction"]
                example = item["example"]
                question = item["question"]
                context = f"### Instruction:\n{instruction}\n\n### Example:\n{example}\n\n### Question:\n{question}\n\n### Answer:\n"
                token_sensitivity = calculate_token_sensitivity(context, token_shap, tokenizer, device)
                results.append({"context": context, "token_sensitivity": token_sensitivity.to_dict()})

    # Save the token sensitivity results to a file
    with open("token_sensitivity_results.json", "w") as f:
        json.dump(results, f)




def run_triggers_and_modified_prompts(tokenizer, model, device):
    results = []
    
    # Load the token sensitivity results from the file
    with open("token_sensitivity_results.json", "r") as f:
        token_sensitivity_results = json.load(f)
    
    for result in token_sensitivity_results:
        context = result["context"]
        token_sensitivity = torch.tensor(result["token_sensitivity"])
        sensitive_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in token_sensitivity.argsort()[-3:]]
        trigger = " ".join(sensitive_tokens)
        modified_prompt = insert_trigger(context, trigger)
        modified_inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**modified_inputs, max_new_tokens=500, temperature=0.5, top_p=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("### Answer:\n")[1].strip() if "### Answer:\n" in response else "No answer generated."
        results.append({"prompt": modified_prompt, "answer": answer, "token_sensitivity": token_sensitivity.tolist()})
    
    # Save the final results to a file
    with open("attack-results.txt", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


 

# Iterate over each prompt file
#for prompt_file in prompt_files:
#    with open(prompt_file, 'r') as f:
#        data = json.load(f)
#    for item in data:
#        torch.cuda.empty_cache()
#        instruction = item["instruction"]
#        example = item["example"]
#        question = item["question"]
#        context = f"### Instruction:\n{instruction}\n\n### Example:\n{example}\n\n### Question:\n{question}\n\n### Answer:\n"
#        inputs = tokenizer(context, return_tensors="pt").to(device)
#        torch.cuda.empty_cache()
#        token_sensitivity = calculate_token_sensitivity(inputs, model)
#        sensitive_tokens = [tokenizer.convert_ids_to_tokens(idx) for idx in token_sensitivity.argsort()[-3:]]
#        trigger = " ".join(sensitive_tokens)
#        modified_prompt = insert_trigger(context, trigger)
#        modified_inputs = tokenizer(modified_prompt, return_tensors="pt").to(device)
#        with torch.no_grad():
#            outputs = model.generate(**modified_inputs, max_new_tokens=500, temperature=0.5, top_p=0.8, do_sample=True, pad_token_id=tokenizer.eos_token_id)
#        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#        answer = response.split("### Answer:\n")[1].strip() if "### Answer:\n" in response else "No answer generated."
#        results.append({"prompt": modified_prompt, "answer": answer, "token_sensitivity": token_sensitivity.tolist()})

 

# Call the function
calculate_and_save_token_sensitivity(prompt_files, tokenizer, token_shap, device)
# Call the function
# run_triggers_and_modified_prompts(tokenizer, model, device)
# Save the results to attack-results.txt file

#with open("attack-results.txt", "w") as f:
#    for result in results:
#        f.write(json.dumps(result) + "\n")
#
#print("Results have been saved to attack-results.txt file.")