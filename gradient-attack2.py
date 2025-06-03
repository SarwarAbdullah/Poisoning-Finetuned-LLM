
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Adafactor
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hugging Face token
hf_token = "#add here"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tuned/model", token=hf_token)

# Load the model using transformers
model = AutoModelForCausalLM.from_pretrained("tuned/model", token=hf_token)
model.to(device)

# Initialize AdaFactor optimizer
optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True)

# List of prompt files
prompt_files = ['prompts/readiness_prompts.json', 'prompts/stress_prompts.json', 'prompts/sleep_quality_prompts.json']

# Initialize results list
results = []

# Custom forward function for checkpointing
def custom_forward(*inputs):
    inputs_embeds, labels = inputs
    return model(inputs_embeds=inputs_embeds, labels=labels)

# Function to calculate token sensitivity
def calculate_token_sensitivity(inputs, model, epsilon=1e-5):
    model.eval()  # Switch to evaluation mode to avoid training overhead
    embeddings = model.get_input_embeddings()(inputs['input_ids'])
    embeddings.requires_grad = True
    
    # Forward pass
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings, labels=inputs["input_ids"])
        loss = outputs.loss
    
    # Approximate gradients using finite differences
    token_sensitivity = torch.zeros_like(embeddings)
    for i in range(embeddings.size(1)):
        embeddings_perturbed = embeddings.clone()
        embeddings_perturbed[:, i] += epsilon
        with torch.no_grad():
            outputs_perturbed = model(inputs_embeds=embeddings_perturbed, labels=inputs["input_ids"])
            loss_perturbed = outputs_perturbed.loss
        token_sensitivity[:, i] = (loss_perturbed - loss) / epsilon
    
    return token_sensitivity.abs().sum(dim=-1).cpu().numpy()
 

# Function to insert triggers into input
def insert_trigger(prompt, trigger):
    return f"{prompt} {trigger}"

def calculate_and_save_token_sensitivity(prompt_files, tokenizer, model, device):
    results = []
    for prompt_file in prompt_files:
        with open(prompt_file, 'r') as f:
            data = json.load(f)
            for item in data:
                instruction = item["instruction"]
                example = item["example"]
                question = item["question"]
                context = f"### Instruction:\n{instruction}\n\n### Example:\n{example}\n\n### Question:\n{question}\n\n### Answer:\n"
                inputs = tokenizer(context, return_tensors="pt").to(device)
                
                # Clear cache after getting inputs from tokenizer
                torch.cuda.empty_cache()
                
                token_sensitivity = calculate_token_sensitivity(inputs, model)
                torch.cuda.empty_cache()
                results.append({"context": context, "token_sensitivity": token_sensitivity.tolist()})
    
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
 

# Call the function
#calculate_and_save_token_sensitivity(prompt_files, tokenizer, model, device)
# Call the function
run_triggers_and_modified_prompts(tokenizer, model, device)
# Save the results to attack-results.txt file

#with open("attack-results.txt", "w") as f:
#    for result in results:
#        f.write(json.dumps(result) + "\n")
#
#print("Results have been saved to attack-results.txt file.")