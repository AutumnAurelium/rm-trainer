from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import load_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
import wandb
import pandas as pd
import os

class RewardDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        processed_features = []
        margins = []
        
        # Process each example to separate chosen/rejected
        for feature in features:
            processed_features.append({
                "input_ids": feature["chosen_input_ids"],
                "attention_mask": feature["chosen_attention_mask"]
            })
            processed_features.append({
                "input_ids": feature["rejected_input_ids"],
                "attention_mask": feature["rejected_attention_mask"]
            })
            margins.append(feature["margin"])
        
        # Batch pad all sequences together for efficiency
        batch = super().__call__(processed_features)
        
        # Split the batch back into chosen/rejected pairs
        return {
            "chosen_input_ids": batch["input_ids"][::2],
            "chosen_attention_mask": batch["attention_mask"][::2],
            "rejected_input_ids": batch["input_ids"][1::2],
            "rejected_attention_mask": batch["attention_mask"][1::2],
            "margin": torch.tensor(margins, dtype=torch.float32)
        }

def calculate_loss(model, batch, return_metrics=False):
    outputs_chosen = torch.sigmoid(model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"]
    ))
    outputs_rejected = torch.sigmoid(model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"]
    ))
    
    rewards_chosen = outputs_chosen.logits
    rewards_rejected = outputs_rejected.logits

    difference = rewards_chosen - rewards_rejected
    batch_loss = -torch.nn.functional.logsigmoid(
        difference - batch["margin"]
    ).mean()
        
    if return_metrics:
        return batch_loss, {
            "loss": batch_loss.item(),
            "avg_reward_chosen": rewards_chosen.mean().item(),
            "avg_reward_rejected": rewards_rejected.mean().item(),
            "margin": batch["margin"].mean().item(),
            "difference": difference.mean().item(),
            "accuracy": (difference > batch["margin"]).float().mean().item(),
            "chosen_reward_min": rewards_chosen.min().item(),
            "chosen_reward_max": rewards_chosen.max().item(),
            "chosen_reward_std": rewards_chosen.std().item(),
            "rejected_reward_min": rewards_rejected.min().item(),
            "rejected_reward_max": rewards_rejected.max().item(),
            "rejected_reward_std": rewards_rejected.std().item()
        }
    else:
        return batch_loss

def eval_validation(model, val_dataloader):
    """Evaluate the model on the validation set and log it to wandb.

    Args:
        model: The model to evaluate.
        val_dataloader: The validation dataloader.
    """
    model.eval()
    with torch.no_grad():
        val_metrics = []
        
        for batch in val_dataloader:
            loss, metrics = calculate_loss(model, batch, return_metrics=True)
            
            val_metrics.append(metrics)

        # Average metrics over all batches
        metric_df = pd.DataFrame(val_metrics)
        wandb.log({
            "val/" + k: v.mean() for k, v in metric_df.items()
        })
        
        print("Validation loss:", metric_df["loss"].mean())
        
        # Log raw metrics, too - for fun!
        # If wandb gets mad at me I can just remove this.
        wandb.log({
            "full_validation_results": wandb.Table(dataframe=metric_df)
        })   
    model.train()
    
def train_reward_model(hparams: dict):
    os.makedirs("./results", exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        hparams["model"],
        num_labels=1,
        problem_type="regression",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(hparams["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Dataset preparation
    data_collator = RewardDataCollator(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
        max_length=hparams["max_length"]
    )
    
    def tokenize_function(examples):
        tokenized_chosen = tokenizer(
            examples["chosen"]
        )
        tokenized_rejected = tokenizer(
            examples["rejected"]
        )
        
        return {
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
            "margin": examples["margin"],
        }

    dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")[
        "train"
    ].shuffle(seed=42).map(tokenize_function, batched=True)
    
    split_dataset = dataset.train_test_split(test_size=hparams["validation_size"])
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    accelerator = Accelerator(
        gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
        mixed_precision=hparams["mixed_precision"],
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    if accelerator.is_main_process:
        model_name = hparams["model"].split("/")[-1]
        wandb.init(
            project="dclm-slop",
            name=f"dclm-slop-{model_name}",
            config=hparams,
        )
    
    batch_size = hparams["batch_size"] * accelerator.num_processes
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,collate_fn=data_collator
    )

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=hparams["learning_rate"],
        betas=(hparams["adam_beta1"], hparams["adam_beta2"])
    )

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_epochs = hparams["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss, metrics = calculate_loss(model, batch, return_metrics=True)

                # add additional metrics
                metrics["epoch"] = epoch
                metrics["step"] = step
                
                if accelerator.is_main_process:
                    wandb.log(metrics)

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), hparams["clip_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()

            if step % hparams["log_interval"] == 0:
                accelerator.print(f"Step {step}: Loss {loss.item()}")
                        
            progress_bar.update(1)

            if step % hparams["validation_interval"] == 0 and step > 0:
                # Eval on validation set and save checkpoint.
                # TOOD: early stopping?
                eval_validation(model, val_dataloader)
                accelerator.save_state(f"./results/checkpoint_step_{step}")
                if accelerator.is_main_process:
                    accelerator.save_model(model, f"./results/checkpoint_step_{step}")

    # Model is finished, evaluate on validation set and save.
    if accelerator.is_main_process:
        eval_validation(model, val_dataloader)
        
        accelerator.save_model(model, "./results/final")

if __name__ == "__main__":
    hparams = {
        "model": "Qwen/Qwen2.5-7B",
        "num_epochs": 4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "mixed_precision": "bf16",
        "validation_interval": 1000,
        "validation_size": 0.1,
        "clip_grad_norm": 0.01,
        "log_interval": 10,
        "max_length": 1280,
    }
    train_reward_model(hparams)
