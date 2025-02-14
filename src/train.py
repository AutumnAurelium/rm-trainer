import torch
from transformers import AutoTokenizer, TrainingArguments, Qwen2ForSequenceClassification, AutoModelForSequenceClassification
import wandb
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from trl import RewardConfig

from scaled_reward_trainer import ScaledRewardTrainer

# Use environment variables for configuration
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure padding settings for the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# Initialize the model with device placement and dtype specifications
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    num_labels=1
)

model.config.pad_token_id = tokenizer.pad_token_id

# Ensure the model parameters are properly initialized
for param in model.parameters():
    if param is None:
        raise ValueError("Found None parameter in model initialization")

model.train()

# Replace HF dataset loading with pandas
df = pd.read_parquet("data/dclm_slop_results.parquet")

class ComparisonDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=768):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        chosen = row["chosen"]
        rejected = row["rejected"]
        margin = row.get("margin", 1.0)

        # Tokenize without padding (will pad in collate_fn)
        chosen_tokens = self.tokenizer(
            chosen,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        rejected_tokens = self.tokenizer(
            rejected,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
            "margin": torch.tensor(margin, dtype=torch.float)
        }

# Create dataset and dataloader
train_dataset = ComparisonDataset(df, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=1000,  # Matches original batch size
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Update training arguments to use standard HF Arguments
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    report_to=["wandb"],
    max_steps=10000,
    learning_rate=2e-5,
    warmup_steps=1000,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    logging_dir="./logs",
    push_to_hub=False,
    load_best_model_at_end=False,
    disable_tqdm=False,
    fp16=False,
    bf16=True,
    optim="adamw_torch_fused",
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    deepspeed="ds_config.json",
    ddp_find_unused_parameters=False
)

# Initialize wandb only on the main process
if training_args.local_rank == 0:  # Only run on main process
    wandb.init(
        project="qwen-slop-reward",
        config={
            "model_name": model_name,
            "num_epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
        }
    )

# Update trainer initialization to use our dataloader
trainer = ScaledRewardTrainer(
    model=model,
    config=RewardConfig(
        max_length=768,
        output_dir="results"
    ),
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset  # Still works with PyTorch Dataset
)

# Add error handling for cloud environment
try:
    trainer.train()
except Exception as e:
    #if training_args.local_rank == 0:  # Only log error on main process
    #    wandb.alert(title="Training Failed", text=str(e))
    raise
finally:
    if training_args.local_rank == 0:  # Only finish wandb on main process
        wandb.finish() 