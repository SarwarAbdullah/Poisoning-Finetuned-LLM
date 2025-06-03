import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import json
from torch.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('../DeepSeek-R1-Distill-Qwen-1.5B')

# Load the model with safetensors weights
model = AutoModelForCausalLM.from_pretrained(
    '../DeepSeek-R1-Distill-Qwen-1.5B',
    from_tf=False,
    from_flax=False,
    use_safetensors=True
).to(device)  # Move model to GPU

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
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                    labels = inputs.input_ids.clone()
                    self.examples.append({'input_ids': inputs.input_ids.squeeze(), 'attention_mask': inputs.attention_mask.squeeze(), 'labels': labels.squeeze()})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Custom collate function
def custom_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

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
data_loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn, num_workers=4)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision training
    max_grad_norm=None,  # Disable gradient clipping
    gradient_accumulation_steps=16  # Increase gradient accumulation steps
)

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=custom_collate_fn
)

# Fine-tune the model
scaler = GradScaler()

# Define a checkpointed model
class CheckpointedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        def custom_forward(*inputs):
            return self.model(input_ids=inputs[0], attention_mask=inputs[1], labels=inputs[2])
        return checkpoint.checkpoint(custom_forward, input_ids, attention_mask, labels, use_reentrant=False)

checkpointed_model = CheckpointedModel(model)

for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        # Ensure requires_grad is set only for floating point tensors
        for key in batch:
            if batch[key].dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
                batch[key].requires_grad = True
        with autocast(device_type='cuda'):
            outputs = checkpointed_model(**batch)
            loss = outputs.loss / training_args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % training_args.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.empty_cache()  # Free up memory
        del batch, outputs, loss  # Clear unused variables
        torch.cuda.empty_cache()  # Free up memory

# Save the fine-tuned model
model.save_pretrained('tuned/model')
tokenizer.save_pretrained('tuned/model')