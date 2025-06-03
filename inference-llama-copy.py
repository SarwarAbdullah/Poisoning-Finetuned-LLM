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
    "tuned/model",
    token=hf_token
)

# Load the model using transformers
model = AutoModelForCausalLM.from_pretrained(
    "tuned/model",
    token=hf_token
)

# Move the model to the appropriate device
model.to(device)

# Custom question and context
instruction = "Predict the fatigue level (1-5) based on the following physiological data. Is this a health concer?"

input_data = ("Steps: [5269.0, 2121.0, 1591.0, 425.0, 1944.0, 1302.0, 2968.0, 1057.0, 2638.0, 407.0, 1073.0], "
              "Burned Calories: [297.0, 197.0, 119.0, 131.0, 161.0, 98.0, 610.0, 87.0, 631.0, 337.0, 323.0], "
              "Resting Heart Rate: [61.0697717666626, 62.40205764770508, 63.615715980529785, 62.528472900390625, 62.34894943237305, 61.35254096984863, "
              "62.11835861206055, 62.11280345916748, 64.54501247406006, 63.86756706237793, 63.55196285247803, 62.47495460510254, 61.41959857940674, "
              "62.158249855041504], SleepMinutes: [543.0, 456.0, 379.0, 606.0, 478.0, 514.0, 522.0, 439.0, 439.0, 430.0, 484.0, 414.0, 397.0, 506.0, 403.0], "
              "Mood: 3 out of 5")

question = ("Based on the above, what is the predicted fatigue level for the following data:\n"
            "The recent 14-days sensor readings show: [Steps]: [5269.0, 2121.0, 1591.0, 425.0, 1944.0, 1302.0, 2968.0, 1057.0, 2638.0, 407.0, 1073.0] steps, "
            "[Burned Calories]: [297.0, 197.0, 119.0, 131.0, 161.0, 98.0, 610.0, 87.0, 631.0, 337.0, 323.0] calories, "
            "[Resting Heart Rate]: [61.0697717666626, 62.40205764770508, 63.615715980529785, 62.528472900390625, 62.34894943237305, 61.35254096984863, "
            "62.11835861206055, 62.11280345916748, 64.54501247406006, 63.86756706237793, 63.55196285247803, 62.47495460510254, 61.41959857940674, "
            "62.158249855041504] beats/min, [SleepMinutes]: [543.0, 456.0, 379.0, 606.0, 478.0, 514.0, 522.0, 439.0, 439.0, 430.0, 484.0, 414.0, 397.0, "
            "506.0, 403.0] minutes, [Mood]: 3 out of 5; What would be the predicted fatigue?")

context = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Output:\n"

# Create the prompt
prompt = f"{context}### Question:\n{question}\n### Answer:\n"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,  # Increase the number of tokens to generate a longer response
        temperature=0.7,
        top_p=0.9,
        do_sample=True,  # Enable sampling
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the answer part from the response
answer = response.split("### Answer:\n")[1].strip() if "### Answer:\n" in response else "No answer generated."
print(f"Question: {question}\n")
print(f"Answer: {answer}")