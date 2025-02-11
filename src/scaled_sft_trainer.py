from trl import SFTTrainer
import torch


class ScaledSFTTrainer(SFTTrainer):
    """A trainer that scales the SFT loss by a dataset column 'score'.

    It reuses the HF TRL SFTTrainer code as much as possible and overrides compute_loss to adjust the loss.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop the 'score' field from inputs if present
        scores = inputs.pop("score", None)

        # Call the model (this will use the SFT loss computation from TRL)
        outputs = model(**inputs)

        # Base loss computed by the model
        loss = outputs.loss

        # If the 'score' field is provided, scale the loss
        # Here we assume that loss is a mean reduction over a batch, so we multiply by the mean score.
        if scores is not None:
            # Ensure scores is a tensor
            if not torch.is_tensor(scores):
                scores = torch.tensor(scores, dtype=loss.dtype, device=loss.device)
            loss = loss * scores.mean()

        return (loss, outputs) if return_outputs else loss 