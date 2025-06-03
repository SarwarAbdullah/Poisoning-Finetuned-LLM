from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, HfFolder

# Set your Hugging Face API token
api_token = # add your token here

model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="", # add here
    use_safetensors=True,
    use_auth_token=api_token
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir="", # add here
    use_auth_token=api_token
)