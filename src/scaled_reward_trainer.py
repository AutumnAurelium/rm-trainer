from trl import RewardTrainer
import torch

class ScaledRewardTrainer(RewardTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        scores = inputs.pop("score")
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], 
                             attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"],
                                attention_mask=inputs["attention_mask_rejected"])[0]
        
        # Calculate per-example loss and apply score weights
        loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected)
        weighted_loss = (loss * scores).mean()
        
        return (weighted_loss, (rewards_chosen, rewards_rejected)) if return_outputs else weighted_loss
