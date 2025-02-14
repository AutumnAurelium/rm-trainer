import torch
from transformers import AutoTokenizer, TrainingArguments, Qwen2ForSequenceClassification
import wandb
import os
from datasets import Dataset, load_dataset
from trl import RewardConfig

from scaled_reward_trainer import ScaledRewardTrainer

# Use environment variables for configuration
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the model with device placement and dtype specifications
model = Qwen2ForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Specify dtype since we're using fp16
    device_map="cuda"
)

# Ensure the model parameters are properly initialized
for param in model.parameters():
    if param is None:
        raise ValueError("Found None parameter in model initialization")

# Set the model to training mode
model.train()

# Create the lazy dataset using IterableDataset
train_dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")

# Update training arguments for cloud deployment
training_args = RewardConfig(
    output_dir="results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
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
    disable_tqdm=True,
    fp16=True,
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

# Update trainer with data collator
trainer = ScaledRewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer
)

# Add error handling for cloud environment
try:
    trainer.train()
except Exception as e:
    if training_args.local_rank == 0:  # Only log error on main process
        wandb.alert(title="Training Failed", text=str(e))
    raise
finally:
    if training_args.local_rank == 0:  # Only finish wandb on main process
        wandb.finish() 