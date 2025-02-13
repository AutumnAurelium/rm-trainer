prompt = """<task>
Classify the following passage as either high-quality, human-written data or spam, SEO-optimized, or machine-generated content.
The URL of this webpage has also been provided.
</task>
<passage>
<url>{}</url>
<text>
{}
</text>
</passage>
<options>
a. High-quality, human-written data
b. Spam, SEO-optimized, or machine-generated content
</options>
"""

from typing import Literal
from transformers import AutoTokenizer
from datasets import IterableDataset
import re
import json


class SlopRewardIterableDataset(IterableDataset):
    def __init__(self, filename: str, tokenizer: AutoTokenizer, max_tokens: int):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        
    def __len__(self):
        """Relatively costly on large datasets. Don't do this lightly."""
        return sum(1 for _ in open(self.filename))
    
    def split_sample(self, sample, split_on: Literal["newline", "period", "space", "token"] = "newline"):
        if split_on not in ["newline", "period", "space", "token"]:
            raise ValueError(f"Invalid split_on value: {split_on}")
        
        tokens = self.tokenizer.encode(sample, add_special_tokens=False)
        
        if len(tokens) <= self.max_tokens:
            return [sample]
        
        # In order, our preference is to split on newlines, periods, or spaces.
        
        if split_on == "newline":
            splits = re.split(r'[\n]', sample)
        elif split_on == "period":
            splits = re.split(r'[\.]', sample)
        elif split_on == "space":
            splits = re.split(r'[\s]', sample)
        else:
            # If all else fails, split on token boundaries.
            splits = []
            for start in range(0, len(tokens), self.max_tokens):
                splits.append(self.tokenizer.decode(tokens[start:start+self.max_tokens]))
            return splits
        
        final_splits = []
        
        for split in splits:
            if len(self.tokenizer.encode(split, add_special_tokens=False)) > self.max_tokens:
                if split_on == "newline":
                    final_splits.extend(self.split_sample(split, split_on="period"))
                elif split_on == "period":
                    final_splits.extend(self.split_sample(split, split_on="space"))
                elif split_on == "space":
                    final_splits.extend(self.split_sample(split, split_on="token"))
            else:
                final_splits.append(split)
        
        return final_splits
    
    def __iter__(self):
        with open(self.filename, "r") as f:
            for line in f:
                data = json.loads(line)
                
                sample_a, sample_b = data["sample"]
                
                sample_a_split = self.split_sample(sample_a["text"])
                sample_b_split = self.split_sample(sample_b["text"])
                
                sample_a_url = sample_a["metadata"]["url"]
                sample_b_url = sample_b["metadata"]["url"]
                
                score = data["feedback"]["slop_comparison"]
                
                if score == 4: # annotator expressed no preference
                    continue
                
                score_adjusted = (score - 4) / 3
                
                for sample_a_split in sample_a_split:
                    for sample_b_split in sample_b_split:
                        if score_adjusted > 0: # prefers sample_b
                            yield {
                                "chosen": prompt.format(sample_b_url, sample_b_split),
                                "rejected": prompt.format(sample_a_url, sample_a_split),
                                "score": abs(score_adjusted)
                            }
                        else: # prefers sample_a
                            yield {
                                "chosen": prompt.format(sample_a_url, sample_a_split),
                                "rejected": prompt.format(sample_b_url, sample_b_split),
                                "score": abs(score_adjusted)
                            }
                
                

    def __getitem__(self, idx):
        pass
