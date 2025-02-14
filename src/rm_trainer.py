from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
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
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("parquet", data_files="data/dclm_slop_results.parquet")[
        "train"
    ]
    # Shuffle after, to avoid contaminating validation with the same URLs.
    split_dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"].select(range(len(split_dataset["train"])//2)).shuffle(seed=42)
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
    
    batch_size = 6 * accelerator.num_processes
    train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(tokenized_val, batch_size=batch_size)

    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-5, betas=(0.9, 0.95))

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    num_epochs = 4
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
                    difference * batch["margin"]
                ).mean()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Step {step}: Loss {loss.item()}")
            
            if step % 100 == 0 and step > 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
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
                        val_loss += -torch.nn.functional.logsigmoid(
                            difference * batch["margin"]
                        ).mean()
                        
                    accelerator.print(f"Step {step}: Val Loss {val_loss.item() / len(val_dataloader)}")
                    
                    if accelerator.is_main_process:
                        wandb.log({"val_loss": val_loss.item() / len(val_dataloader)})
                        
            progress_bar.update(1)

            if step % 1000 == 0 and step > 0:
                accelerator.save_state(f"./results/checkpoint_step_{step}")


if __name__ == "__main__":
    train_reward_model()
