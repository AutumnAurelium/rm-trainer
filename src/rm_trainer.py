from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
from transformers import get_scheduler
import wandb
import pandas as pd
import os


class RewardDataCollator(DataCollatorWithPadding):
    # This is kind of a hack to get the stock HF DataCollator to work with our pairwise dataset.
    def __call__(self, features):
        processed_features = []
        scores = []

        # Process each example to separate a/b
        for feature in features:
            processed_features.append(
                {
                    "input_ids": feature["sample_a_input_ids"],
                    "attention_mask": feature["sample_a_attention_mask"],
                }
            )
            processed_features.append(
                {
                    "input_ids": feature["sample_b_input_ids"],
                    "attention_mask": feature["sample_b_attention_mask"],
                }
            )
            scores.append(feature["score"])

        # Batch pad all sequences
        batch = super().__call__(processed_features)

        # Split the batch back into a/b pairs
        return {
            "sample_a_input_ids": batch["input_ids"][::2],
            "sample_a_attention_mask": batch["attention_mask"][::2],
            "sample_b_input_ids": batch["input_ids"][1::2],
            "sample_b_attention_mask": batch["attention_mask"][1::2],
            "score": torch.tensor(scores, dtype=torch.float32),
        }


def calculate_loss(model, batch, return_metrics=False):
    outputs_a = model(
        input_ids=batch["sample_a_input_ids"],
        attention_mask=batch["sample_a_attention_mask"],
    ).logits
    outputs_b = model(
        input_ids=batch["sample_b_input_ids"],
        attention_mask=batch["sample_b_attention_mask"],
    ).logits

    rewards_a = outputs_a[:, 0]
    rewards_b = outputs_b[:, 0]

    targets = batch["score"]

    # this is equivalent to bradley-terry probability
    probs = torch.sigmoid(rewards_a - rewards_b)

    batch_loss = torch.nn.functional.binary_cross_entropy(
        probs, targets
    ).mean()

    if return_metrics:
        return batch_loss, {
            "loss": batch_loss.item(),
            "accuracy": (probs > 0.5).float().mean().item(),
            "mean_rewards_a": rewards_a.mean().item(),
            "mean_rewards_b": rewards_b.mean().item(),
        }
    else:
        return batch_loss


def eval_validation(model, val_dataloader, do_log=False):
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
        wandb.log({"val/" + k: v.mean() for k, v in metric_df.items()})

        print("Validation loss:", metric_df["loss"].mean())

        # Log raw metrics, too - for fun!
        # If wandb gets mad at me I can just remove this.
        wandb.log({"full_validation_results": wandb.Table(dataframe=metric_df)})
    model.train()
    
    return metric_df["loss"].mean()

def train_reward_model(hparams: dict):
    seed = hparams.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.makedirs("./results", exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        hparams["model"],
        num_labels=1,
        problem_type="regression",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(hparams["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Dataset preparation
    data_collator = RewardDataCollator(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
        max_length=hparams["max_length"],
    )

    def tokenize_function(examples):
        tokenized_sample_a = tokenizer(examples["sample_a"])
        tokenized_sample_b = tokenizer(examples["sample_b"])

        return {
            "sample_a_input_ids": tokenized_sample_a["input_ids"],
            "sample_a_attention_mask": tokenized_sample_a["attention_mask"],
            "sample_b_input_ids": tokenized_sample_b["input_ids"],
            "sample_b_attention_mask": tokenized_sample_b["attention_mask"],
            "score": examples["score"],
        }

    dataset = (
        load_dataset("parquet", data_files="data/dataset.parquet")["train"]
        .shuffle(seed=42)
        .map(tokenize_function, batched=True)
    )

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
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = bnb.optim.Adam8bit(
        model.parameters(),
        lr=hparams["learning_rate"],
        betas=(hparams["adam_beta1"], hparams["adam_beta2"]),
        weight_decay=hparams["adam_weight_decay"],
    )
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=hparams["num_warmup_steps"],
        num_training_steps=len(train_dataloader) * hparams["num_epochs"],
    )
   
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_epochs = hparams["num_epochs"]
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    # validation loss stuff
    best_val_loss = float('inf')
    patience = hparams.get("patience", 3)
    patience_counter = 0

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
                accelerator.clip_grad_norm_(
                    model.parameters(), hparams["clip_grad_norm"]
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % hparams["log_interval"] == 0:
                accelerator.print(f"Step {step}: Loss {loss.item()}")

            progress_bar.update(1)

            # validation stuff
            if step % hparams["validation_interval"] == 0 and step > 0:
                # Eval on validation set and save checkpoint
                val_loss = eval_validation(model, val_dataloader, do_log=accelerator.is_main_process)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    if accelerator.is_main_process:
                        accelerator.save_model(model, "./results/best_model")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered after {step} steps")
                        break
                        
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
        "seed": 6408,
        "num_epochs": 4,
        "batch_size": 2,
        "num_warmup_steps": 100,
        "num_training_steps": 1000,
        "gradient_accumulation_steps": 4,
        "learning_rate": 3e-6,
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "adam_weight_decay": 0.01,
        "mixed_precision": "bf16",
        "validation_interval": 1000,
        "validation_size": 0.2,
        "clip_grad_norm": 0.01,
        "log_interval": 10,
        "max_length": 768,
    }
    train_reward_model(hparams)
