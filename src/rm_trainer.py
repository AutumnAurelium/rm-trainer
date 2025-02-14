from transformers import Trainer, TrainingArguments
import torch
from torch import nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Split inputs into chosen and rejected
        chosen_inputs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"]
        }
        rejected_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"]
        }
        
        # Get model outputs
        chosen_outputs = model(**chosen_inputs)
        rejected_outputs = model(**rejected_inputs)
        
        # Calculate pairwise loss with margin
        rewards_chosen = chosen_outputs.logits
        rewards_rejected = rejected_outputs.logits
        difference = rewards_chosen - rewards_rejected
        loss = -torch.nn.functional.logsigmoid(difference * inputs["margin"]).mean()
        
        return (loss, (chosen_outputs, rejected_outputs)) if return_outputs else loss

def train_reward_model():
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-7B",
        num_labels=1,
        problem_type="regression",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load and preprocess dataset
    dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")["train"]
    
    def tokenize_function(examples):
        tokenized_chosen = tokenizer(
            examples["chosen"],
            padding="max_length",
            truncation=True,
            max_length=1280
        )
        tokenized_rejected = tokenizer(
            examples["rejected"],
            padding="max_length",
            truncation=True,
            max_length=1280
        )
        return {
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
            "margin": examples["margin"]
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=4,
        learning_rate=1e-5,
        bf16=True,
        # deepspeed="ds_config.json",
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        logging_steps=10,
        save_steps=1000,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        # ddp_backend="nccl",
        # ddp_find_unused_parameters=False
    )

    # Initialize custom trainer
    trainer = RMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Start training
    trainer.train() 
    
if __name__ == "__main__":
    train_reward_model()