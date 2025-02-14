import torch
from transformers import AutoTokenizer, TrainingArguments, Qwen2ForSequenceClassification, AutoModelForSequenceClassification
import wandb
import os
from datasets import Dataset, load_dataset
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

model.train()

raw_dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")["train"]

def tokenize_pair(examples):
    tokenized_chosen = tokenizer(
        examples["chosen"], 
        max_length=768
    )
    tokenized_rejected = tokenizer(
        examples["rejected"],
        max_length=768
    )
    return {
        "chosen_input_ids": tokenized_chosen["input_ids"],
        "chosen_attention_mask": tokenized_chosen["attention_mask"],
        "rejected_input_ids": tokenized_rejected["input_ids"],
        "rejected_attention_mask": tokenized_rejected["attention_mask"],
        "margin": examples.get("margin", 1.0)
    }

# Process with batched streaming
train_dataset = raw_dataset.map(
    tokenize_pair,
    batched=True,
    batch_size=1000  # Adjust based on memory constraints
)

# Update training arguments to use standard HF Arguments
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    report_to=["wandb"],
    max_steps=1000,
    learning_rate=2e-5,
    warmup_steps=100,
    dataloader_num_workers=1,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    logging_dir="./logs",
    push_to_hub=False,
    load_best_model_at_end=False,
    disable_tqdm=False,
    fp16=False,
    bf16=True,
    optim="adamw_bnb_8bit",
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    deepspeed="ds_config.json",
    ddp_find_unused_parameters=False,
    use_liger_kernel=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
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

# Update trainer initialization
trainer = ScaledRewardTrainer(
    model=model,
    config=RewardConfig(max_length=768, output_dir="results"),
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset
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