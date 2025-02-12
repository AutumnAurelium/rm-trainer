import torch
from qwen2_rm import Qwen2ForCausalLMPermittedTokens
from transformers import AutoTokenizer, TrainingArguments
import wandb

from scaled_sft_trainer import ScaledSFTTrainer
# Import the lazy dataset class directly
from slop_dataset import LazySlopIterableDataset

model_name = "Qwen/Qwen2.5-7B"
permitted_tokens = ["a", "b"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

permitted_token_ids = [tokenizer.encode(token)[0] for token in permitted_tokens]

# Uncomment the following if you want to use Qwen2ForCausalLMPermittedTokens:
# model = Qwen2ForCausalLMPermittedTokens.from_pretrained(model_name, permitted_token_ids=permitted_token_ids)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to training mode
model.train()

# Create the lazy dataset using IterableDataset
train_dataset = LazySlopIterableDataset(
    "data/dclm_slop_results.jsonl",
    tokenizer,
    max_length=256  # Renamed from max_tokens for consistency
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=1,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="no",
    learning_rate=1e-5,
    optim="adamw_torch",
    report_to=["wandb"],
    max_steps=1000
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

from trl import SFTTrainer

# Initialize the trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Start training
trainer.train() 