import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, Adafactor
from torch.utils.data import Dataset, DataLoader
import json
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import gc


# Hugging Face token
hf_token = "#add here"


# Declare the tokenizer as a global variable
tokenizer = None



# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_files, tokenizer):
        self.examples = []
        for file_path in data_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    instruction = item['instruction']
                    input_data = item['input']
                    output_data = item['output']
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n### Output:\n{output_data}"
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
                    labels = inputs.input_ids.clone()
                    self.examples.append({'input_ids': inputs.input_ids.squeeze(), 'attention_mask': inputs.attention_mask.squeeze(), 'labels': labels.squeeze()})
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]
    
# Custom collate function class
class CustomCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        # Clip values to be within the valid range
        input_ids = torch.clamp(input_ids, min=0, max=self.tokenizer.vocab_size - 1)
        labels = torch.clamp(labels, min=0, max=self.tokenizer.vocab_size - 1)
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main():
    # Set environment variable for memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable synchronous execution for debugging
    # Suppress parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/llama-3.1-8B",
        token=hf_token
    )

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Print tokenizer vocabulary size
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Define BitsAndBytesConfig for quantization
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/llama-3.1-8B",
        torch_dtype=torch.float16,  # Use float16 for mixed precision
        quantization_config=quantization_config,  # Use BitsAndBytesConfig for quantization
        token=hf_token
    )

    # Attach LoRA adapters
    lora_config = LoraConfig(
        r=8,  # Rank of the low-rank adaptation
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Target specific modules for adaptation
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    # Load your dataset
    instruction_tuning_data_files = [
        "pm_data/PMData_fatigue_train_all.json",
        "pm_data/PMData_readiness_train_all.json",
        "pm_data/PMData_sleep_quality_train_all.json",
        "pm_data/PMData_stress_train_all.json"
    ]

    # Create dataset
    dataset = CustomDataset(instruction_tuning_data_files, tokenizer)

    # Create DataLoader with custom collate function
    collate_fn = CustomCollate(tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=2)
    
    # Define training arguments with Adafactor optimizer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        save_steps=10_000,
        save_total_limit=2,
        fp16=True,  # Enable mixed precision training
        max_grad_norm=None,  # Disable gradient clipping
        gradient_accumulation_steps=16,  # Increase gradient accumulation steps
        optim="adafactor"  # Use Adafactor optimizer
    )

    # Initialize the accelerator
    accelerator = Accelerator(mixed_precision="fp16")

    # Prepare the model and dataloader
    model, data_loader = accelerator.prepare(model, data_loader)

    # Define the Adafactor optimizer
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=True,
        relative_step=True,
        warmup_init=True,
        lr=None
    )

    # Training loop with manual mixed precision
    for epoch in range(training_args.num_train_epochs):
        model.train()
        for batch in data_loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(**batch)
                loss = outputs.loss
                if torch.isnan(loss):
                    print("NaN detected in loss")
            print(f"Loss: {loss.item()}")  # Debugging print
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()        
            # Delete unused variables
            del batch, outputs, loss        
            # Collect garbage
            gc.collect()

    # Save the fine-tuned model
    model.save_pretrained('tuned/model')
    tokenizer.save_pretrained('tuned/model')

if __name__ == '__main__':
    main()