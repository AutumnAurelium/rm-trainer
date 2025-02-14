from typing import Optional, Dict, Union, Tuple
import torch
from torch import nn
from transformers import Trainer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from datasets import Dataset, IterableDataset
from trl import RewardConfig

class ScaledRewardTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        config: Optional[RewardConfig] = None,
        args: Optional[TrainingArguments] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        use_liger_kernel: bool = False
    ):
        super().__init__(
            model=model,
            args=args,
            processing_class=processing_class,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            use_liger_kernel=use_liger_kernel
        )
        self.config = config
        self.margin = 1.0
        
        # Custom loss with label smoothing support
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
        
        # Add custom data collator
        self.data_collator = self._pad_collate

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Process chosen and rejected pairs
        chosen_inputs = {
            "input_ids": inputs["chosen_input_ids"],
            "attention_mask": inputs["chosen_attention_mask"]
        }
        rejected_inputs = {
            "input_ids": inputs["rejected_input_ids"],
            "attention_mask": inputs["rejected_attention_mask"]
        }

        # Get scores for both sequences
        chosen_scores = model(**chosen_inputs).logits
        rejected_scores = model(**rejected_inputs).logits
        
        # Calculate difference with margin
        margin = inputs.get("margin", self.margin)
        diff = chosen_scores - rejected_scores - margin
        labels = torch.ones_like(chosen_scores)
        
        # Compute custom loss
        loss = self.loss_fct(diff, labels)
        
        return (loss, (chosen_scores, rejected_scores)) if return_outputs else loss

    def _prepare_dataset(self, dataset):
        # Handle streaming datasets differently
        from datasets import IterableDataset
        
        if isinstance(dataset, IterableDataset):
            # Streaming datasets can't check column names upfront
            return dataset
        elif hasattr(dataset, 'column_names'):
            # Validate column names for in-memory datasets
            required_columns = {"chosen_input_ids", "rejected_input_ids", 
                              "chosen_attention_mask", "rejected_attention_mask"}
            if not required_columns.issubset(dataset.column_names):
                raise ValueError(f"Dataset must contain {required_columns} columns")
            return dataset
        else:
            raise ValueError("Invalid dataset type - must be either Hugging Face Dataset or IterableDataset")

    def get_train_dataloader(self):
        train_dataset = self._prepare_dataset(self.train_dataset)
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataset = self._prepare_dataset(eval_dataset)
        return super().get_eval_dataloader(eval_dataset)

    def compute_metrics(self, eval_pred: EvalPrediction):
        metrics = {}
        chosen_scores, rejected_scores = eval_pred.predictions
        
        # Calculate basic metrics
        metrics["accuracy"] = (chosen_scores > rejected_scores).mean()
        metrics["margin"] = (chosen_scores - rejected_scores).mean()
        metrics["chosen_mean"] = chosen_scores.mean()
        metrics["rejected_mean"] = rejected_scores.mean()
        
        return metrics

    def _pad_collate(self, batch):
        # Custom collation function for paired sequences
        max_length = self.config.max_length if self.config else None
        
        def pad_features(features, key):
            return self.processing_class.pad(
                {
                    "input_ids": [x[key] for x in features],
                    "attention_mask": [x[key.replace("input_ids", "attention_mask")] for x in features]
                },
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        
        chosen = pad_features(batch, "chosen_input_ids")
        rejected = pad_features(batch, "rejected_input_ids")
        
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "margin": torch.tensor([x.get("margin", self.margin) for x in batch])
        }
