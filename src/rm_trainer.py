from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from datasets import load_dataset
from accelerate import Accelerator, DistributedDataParallelKwargs, DistributedSampler
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import bitsandbytes as bnb
import wandb


def train_reward_model():
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        "Qwen/Qwen2.5-7B",
        num_labels=1,
        problem_type="regression",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load and preprocess dataset
    dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")[
        "train"
    ]

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

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch")
    train_dataloader = DataLoader(tokenized_dataset, batch_size=4, shuffle=True)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_with="wandb",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    batch_size = 4 * accelerator.num_processes
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=DistributedSampler(tokenized_dataset),
    )

    if accelerator.is_main_process:
        wandb.init(
            project="dclm-slop-results",
            name="dclm-slop-results-1280",
            config={
                "model": "Qwen/Qwen2.5-7B",
            },
        )

    # Setup optimizer
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    model.gradient_checkpointing_enable()

    # Prepare components
    model, optimizer = accelerator.prepare(
        model, optimizer
    )

    # Training setup
    num_epochs = 4
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Combined forward pass
                combined_ids = torch.cat(
                    [batch["chosen_input_ids"], batch["rejected_input_ids"]]
                )
                combined_mask = torch.cat(
                    [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
                )

                all_outputs = model(
                    input_ids=combined_ids, attention_mask=combined_mask
                )

                # Split outputs
                batch_size = batch["chosen_input_ids"].shape[0]
                rewards_chosen = all_outputs.logits[:batch_size]
                rewards_rejected = all_outputs.logits[batch_size:]

                # Loss calculation (same as before)
                difference = rewards_chosen - rewards_rejected
                loss = -torch.nn.functional.logsigmoid(
                    difference * batch["margin"]
                ).mean()

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging and progress
            if step % 10 == 0:
                accelerator.print(f"Step {step}: Loss {loss.item()}")
            progress_bar.update(1)

            # Save checkpoint
            if step % 1000 == 0 and step > 0:
                accelerator.save_state(f"./results/checkpoint_step_{step}")


if __name__ == "__main__":
    train_reward_model()
