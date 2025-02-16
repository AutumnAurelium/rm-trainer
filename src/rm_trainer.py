from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
import wandb


def train_reward_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-7B",
        num_labels=1,
        problem_type="regression",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")[
        "train"
    ]
    # Shuffle after, to avoid contaminating validation with the same URLs.
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"].select(range(len(split_dataset["train"])//8)).shuffle(seed=42)
    val_dataset = split_dataset["test"].shuffle(seed=42)

    def tokenize_function(examples):
        tokenized_chosen = tokenizer(
            examples["chosen"],
            padding="max_length",
            truncation=True,
            max_length=1280,
            return_tensors="pt",
        )
        tokenized_rejected = tokenizer(
            examples["rejected"],
            padding="max_length",
            truncation=True,
            max_length=1280,
            return_tensors="pt",
        )
        return {
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
            "margin": examples["margin"],
        }

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )

    if accelerator.is_main_process:
        wandb.init(
            project="dclm-slop-results",
            name="dclm-slop-results-1280",
            config={
                "model": "Qwen/Qwen2.5-7B",
            },
        )
    
    batch_size = 2 * accelerator.num_processes
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_epochs = 2
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs_chosen = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                outputs_rejected = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )

                rewards_chosen = outputs_chosen.logits
                rewards_rejected = outputs_rejected.logits

                difference = rewards_chosen - rewards_rejected
                loss = -torch.nn.functional.logsigmoid(
                    difference - batch["margin"]
                ).mean()
                
                if accelerator.is_main_process:
                    wandb.log({
                        "loss": loss.item(),
                        "step": step,
                        "epoch": epoch,
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
                    })

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 0.01)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Step {step}: Loss {loss.item()}")
                        
            progress_bar.update(1)

            if step % 1000 == 0 and step > 0:
                accelerator.save_state(f"./results/checkpoint_step_{step}")
                if accelerator.is_main_process:
                    accelerator.save_model(model, f"./results/checkpoint_step_{step}")

    if accelerator.is_main_process:
        model.eval()
        # validation loss
        with torch.no_grad():
            val_loss = 0
            val_metrics = {
                'difference': 0,
                'margin': 0,
                'accuracy': 0,
                'chosen_reward_stats': [0, 0, 0],
                'rejected_reward_stats': [0, 0, 0]
            }
            
            for batch in val_dataloader:
                outputs_chosen = model(
                    input_ids=batch["chosen_input_ids"],
                    attention_mask=batch["chosen_attention_mask"]
                )
                outputs_rejected = model(
                    input_ids=batch["rejected_input_ids"],
                    attention_mask=batch["rejected_attention_mask"]
                )
                
                rewards_chosen = outputs_chosen.logits
                rewards_rejected = outputs_rejected.logits

                difference = rewards_chosen - rewards_rejected
                batch_loss = -torch.nn.functional.logsigmoid(
                    difference - batch["margin"]
                ).mean()
                val_loss += batch_loss.item()
                
                # Accumulate validation metrics
                val_metrics['difference'] += difference.mean().item()
                val_metrics['margin'] += batch["margin"].mean().item()
                val_metrics['accuracy'] += (difference > batch["margin"]).float().mean().item()
                val_metrics['chosen_reward_stats'][0] += rewards_chosen.min().item()
                val_metrics['chosen_reward_stats'][1] += rewards_chosen.max().item()
                val_metrics['chosen_reward_stats'][2] += rewards_chosen.std().item()
                val_metrics['rejected_reward_stats'][0] += rewards_rejected.min().item()
                val_metrics['rejected_reward_stats'][1] += rewards_rejected.max().item()
                val_metrics['rejected_reward_stats'][2] += rewards_rejected.std().item()

            # Average metrics over all batches
            num_batches = len(val_dataloader)
            wandb.log({
                "val_loss": val_loss / num_batches,
                "val_difference": val_metrics['difference'] / num_batches,
                "val_margin": val_metrics['margin'] / num_batches,
                "val_accuracy": val_metrics['accuracy'] / num_batches,
                "val_chosen_reward_min": val_metrics['chosen_reward_stats'][0] / num_batches,
                "val_chosen_reward_max": val_metrics['chosen_reward_stats'][1] / num_batches,
                "val_chosen_reward_std": val_metrics['chosen_reward_stats'][2] / num_batches,
                "val_rejected_reward_min": val_metrics['rejected_reward_stats'][0] / num_batches,
                "val_rejected_reward_max": val_metrics['rejected_reward_stats'][1] / num_batches,
                "val_rejected_reward_std": val_metrics['rejected_reward_stats'][2] / num_batches
            })
        
        accelerator.save_model(model, "./results/final")

if __name__ == "__main__":
    train_reward_model()
