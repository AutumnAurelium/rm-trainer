from trl import SFTTrainer
import torch
import torch.nn as nn


class ScaledSFTTrainer(SFTTrainer):
    """A trainer that scales the SFT loss by a dataset column 'score'.

    It reuses the HF TRL SFTTrainer code as much as possible and overrides compute_loss to adjust the loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = inputs.pop("score", None)
        outputs = model(**inputs)
        
        # Get per-example losses instead of mean loss
        logits = outputs.logits
        labels = inputs["labels"]
        
        # Calculate loss without reduction
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss per token, then average per sequence
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        per_example_loss = per_token_loss.view(shift_labels.size(0), -1).mean(dim=1)

        if scores is not None:
            if not torch.is_tensor(scores):
                scores = torch.tensor(scores, dtype=per_example_loss.dtype, device=per_example_loss.device)
            # Scale each example's loss by its score before averaging
            loss = (per_example_loss * scores).mean()
        else:
            loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss 