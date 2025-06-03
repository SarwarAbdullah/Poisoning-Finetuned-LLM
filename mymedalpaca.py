import torch
from transformers import AutoModelForCausalLM, LlamaTokenizer

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Load the tokenizer from the local directory
tokenizer = LlamaTokenizer.from_pretrained() # add path here

# Load the model with safetensors weights from the local directory
model = AutoModelForCausalLM.from_pretrained(
    # add path here
    use_safetensors=True,
    torch_dtype=torch.float16  # Use half-precision
).to(device)  # Move model to GPU

model = torch.compile(model)

# Example question
question = "How are you today?"

# Use a prompt template
prompt = f"### Question:\n{question}\n### Answer:\n"

# Tokenize the input and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=100,  # Increase the length of the response
    temperature=0.7,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,  # Define end of sequence token
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the answer part from the response
response = response.split("### Answer:\n")[1].split("\n### Question:")[0].strip()

print(f"Question: {question}")
print(f"Response: {response}")