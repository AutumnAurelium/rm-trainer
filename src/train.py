import torch
from qwen2_rm import Qwen2ForCausalLMPermittedTokens
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
import os  # Added for environment variables

from scaled_sft_trainer import ScaledSFTTrainer
# Import the lazy dataset class directly
from slop_dataset import LazySlopIterableDataset

# Use environment variables for configuration
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B")
permitted_tokens = ["a", "b"]  # Expect comma-separated tokens
deepspeed_config = os.getenv("DEEPSPEED_CONFIG", "ds_config.json")

tokenizer = AutoTokenizer.from_pretrained(model_name)

permitted_token_ids = [tokenizer.encode(token)[0] for token in permitted_tokens]

model = Qwen2ForCausalLMPermittedTokens.from_pretrained(model_name, permitted_token_ids=permitted_token_ids)

# Set the model to training mode
model.train()

# Create the lazy dataset using IterableDataset
train_dataset = LazySlopIterableDataset(
    "data/dclm_slop_results.jsonl",
    tokenizer,
    max_length=256  # Renamed from max_tokens for consistency
)

# Add data collator for better batch handling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Update training arguments for cloud deployment
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    report_to=["wandb"],
    max_steps=10000,  # Updated to match total_num_steps in ds_config
    learning_rate=2e-5,  # Added to match warmup_max_lr in ds_config
    warmup_steps=1000,  # Added to match warmup_num_steps in ds_config
    deepspeed=deepspeed_config,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    logging_dir="./logs",
    push_to_hub=False,
    load_best_model_at_end=False,
    disable_tqdm=True
)

# Initialize wandb
wandb.init(
    project="qwen-permitted-tokens",
    config={
        "model_name": model_name,
        "permitted_tokens": permitted_tokens,
        "num_epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
    }
)

# Update trainer with data collator
trainer = ScaledSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator  # Added data collation
)

# Add error handling for cloud environment
try:
    trainer.train()
except Exception as e:
    wandb.alert(title="Training Failed", text=str(e))
    raise
finally:
    wandb.finish() 