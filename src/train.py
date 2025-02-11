import torch
from qwen2_rm import Qwen2ForCausalLMPermittedTokens
from transformers import AutoTokenizer, TrainingArguments

from scaled_sft_trainer import ScaledSFTTrainer
from slop_dataset import DatasetParser

model_name = "Qwen/Qwen2.5-0.5B"
permitted_tokens = ["a", "b"]

tokenizer = AutoTokenizer.from_pretrained(model_name)

permitted_token_ids = [tokenizer.encode(token)[0] for token in permitted_tokens]

model = Qwen2ForCausalLMPermittedTokens.from_pretrained(model_name, permitted_token_ids=permitted_token_ids)

# Set the model to training mode
model.train()

# Load dataset from data/dclm_slop_results.jsonl using the provided function
train_dataset = DatasetParser("data/dclm_slop_results.jsonl", tokenizer, 1024).get_hf_slop_dataset()

print(f"Beginning training with {len(train_dataset)} examples.")

# Define training arguments
training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=1,
    save_strategy="no",
    learning_rate=1e-5,
    optim="adamw_torch",
)

# Initialize the trainer
trainer = ScaledSFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Start training
trainer.train()