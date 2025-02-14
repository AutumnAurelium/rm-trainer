import os
import json
import torch
import torch.nn.functional as F
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Custom dataset that loads the parquet file and tokenizes the chosen and rejected texts
class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = pd.read_parquet(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        chosen_text = row['chosen']
        rejected_text = row['rejected']
        margin = row['margin']  # float in (0, 1)

        chosen_inputs = self.tokenizer(chosen_text, truncation=True, max_length=self.max_length, return_tensors='pt')
        rejected_inputs = self.tokenizer(rejected_text, truncation=True, max_length=self.max_length, return_tensors='pt')

        # Remove batch dimension
        item = {
            'chosen_input_ids': chosen_inputs['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_inputs['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_inputs['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_inputs['attention_mask'].squeeze(0),
            'margin': torch.tensor(margin, dtype=torch.float)
        }
        return item


def get_collate_fn(tokenizer):
    # This collate function pads the chosen and rejected sequences separately using the tokenizer's pad method
    def collate_fn(batch):
        chosen_batch = {
            'input_ids': [x['chosen_input_ids'] for x in batch],
            'attention_mask': [x['chosen_attention_mask'] for x in batch]
        }
        rejected_batch = {
            'input_ids': [x['rejected_input_ids'] for x in batch],
            'attention_mask': [x['rejected_attention_mask'] for x in batch]
        }
        chosen_padded = tokenizer.pad(chosen_batch, return_tensors='pt')
        rejected_padded = tokenizer.pad(rejected_batch, return_tensors='pt')
        margins = torch.stack([x['margin'] for x in batch])
        return {
            'chosen_input_ids': chosen_padded['input_ids'],
            'chosen_attention_mask': chosen_padded['attention_mask'],
            'rejected_input_ids': rejected_padded['input_ids'],
            'rejected_attention_mask': rejected_padded['attention_mask'],
            'margin': margins
        }
    return collate_fn


# Custom Trainer that overrides the loss computation
class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass for the chosen sequences
        outputs_chosen = model(input_ids=inputs['chosen_input_ids'], attention_mask=inputs['chosen_attention_mask'])
        # Forward pass for the rejected sequences
        outputs_rejected = model(input_ids=inputs['rejected_input_ids'], attention_mask=inputs['rejected_attention_mask'])

        # Assuming model outputs logits of shape (batch_size, 1), squeeze to (batch_size,)
        r_chosen = outputs_chosen.logits.squeeze(-1)
        r_rejected = outputs_rejected.logits.squeeze(-1)

        # Compute the difference in reward scores
        diff = r_chosen - r_rejected
        # Use binary cross-entropy with logits where the target is the margin (preference strength)
        loss = F.binary_cross_entropy_with_logits(diff, inputs['margin'])
        return (loss, (outputs_chosen, outputs_rejected)) if return_outputs else loss


def main():
    # Define the model name for Qwen2.5-7B
    model_name = 'Qwen2.5-7B'

    # Load tokenizer and model using HuggingFace libraries
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Use a sequence classification head with a single output for the reward score
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # Create the training dataset
    data_path = 'data/dclm_slop_results.parquet'
    train_dataset = RewardDataset(data_path, tokenizer)

    # Write out a DeepSpeed configuration with stage 3 optimization, bf16, a custom flag for liger kernels,
    # and settings intended for 8-bit AdamW (used via TrainingArguments optim='adamw_8bit').
    ds_config = {
        "bf16": {
            "enabled": true
        },
        "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": true,
            "overlap_comm": true,
            "reduce_bucket_size": 50000000,
            "allgather_bucket_size": 50000000,
            "sub_group_size": 2000
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 2e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 1e-2
            }
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": 4,
        // Custom flag for liger kernels (this is illustrative; actual integration may vary)
        "liger": {
            "enabled": true
        }
    }

    ds_config_file = 'ds_config.json'
    with open(ds_config_file, 'w') as f:
        json.dump(ds_config, f, indent=2)

    # Setup training arguments with DeepSpeed integration, bf16 inference, and 8-bit AdamW optimization
    training_args = TrainingArguments(
        output_dir='outputs',
        per_device_train_batch_size=4,
        num_train_epochs=3,
        bf16=True,
        optim='adamw_8bit',
        deepspeed=ds_config_file,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy='no',
        dataloader_num_workers=4,
        run_name='qwen25_reward_model'
    )

    # Get the collate function using our tokenizer
    collate_fn = get_collate_fn(tokenizer)

    # Instantiate our custom Trainer
    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main() 