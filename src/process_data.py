from typing import Literal
from transformers import AutoTokenizer
import re
import json
import argparse
import asyncio
from datasets import Dataset, Features, Value

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

def split_sample(tokenizer: AutoTokenizer, sample, max_tokens: int, split_on: Literal["newline", "period", "space", "token"] = "newline"):
    if split_on not in ["newline", "period", "space", "token"]:
        raise ValueError(f"Invalid split_on value: {split_on}")
    
    tokens = tokenizer.encode(sample, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        return [sample]
    
    # In order, our preference is to split on newlines, periods, or spaces.
    
    if split_on == "newline":
        splits = re.split(r'(\n+)', sample)
    elif split_on == "period":
        splits = re.split(r'(\.+)', sample)
    elif split_on == "space":
        splits = re.split(r'(\s+)', sample)
    else:
        # If all else fails, split on token boundaries.
        splits = []
        for start in range(0, len(tokens), max_tokens):
            splits.append(tokenizer.decode(tokens[start:start+max_tokens]))
        return splits
    
    final_splits = []
    
    for split in splits:
        if len(tokenizer.encode(split, add_special_tokens=False)) > max_tokens:
            if split_on == "newline":
                final_splits.extend(split_sample(tokenizer, split, max_tokens, split_on="period"))
            elif split_on == "period":
                final_splits.extend(split_sample(tokenizer, split, max_tokens, split_on="space"))
            elif split_on == "space":
                final_splits.extend(split_sample(tokenizer, split, max_tokens, split_on="token"))
        else:
            final_splits.append(split)
    
    return [s.strip() for s in final_splits if s.strip()]

async def process_data(line: str, tokenizer: AutoTokenizer, max_tokens: int):
    data = json.loads(line)
    
    sample_a, sample_b = data["sample"]
    
    sample_a_split = split_sample(tokenizer, sample_a["text"], max_tokens)
    sample_b_split = split_sample(tokenizer, sample_b["text"], max_tokens)
    
    sample_a_url = sample_a["metadata"]["url"]
    sample_b_url = sample_b["metadata"]["url"]
    
    score = data["feedback"]["slop_comparison"]
    
    if score == 4: # annotator expressed no preference
        return
    
    score_scaled = (score - 4) / 3
    
    for a_chunk in sample_a_split:
        for b_chunk in sample_b_split:
            if score > 0: # prefers sample_b
                yield {
                    "chosen": prompt.format(sample_b_url, b_chunk),
                    "rejected": prompt.format(sample_a_url, a_chunk),
                    "margin": abs(score)
                }
            else: # prefers sample_a
                yield {
                    "chosen": prompt.format(sample_a_url, a_chunk),
                    "rejected": prompt.format(sample_b_url, b_chunk),
                    "margin": abs(score_scaled)
                }

async def main(tokenizer: AutoTokenizer, max_tokens: int) -> Dataset:
    examples = []

    with open(args.input_file, "r") as f:
        for line in f:
            async for x in process_data(line, tokenizer, max_tokens):
                examples.append(x)
    
    return Dataset.from_dict({
        "chosen": [x["chosen"] for x in examples],
        "rejected": [x["rejected"] for x in examples],
        "margin": [x["margin"] for x in examples]
    })
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = asyncio.run(main(tokenizer, args.max_tokens))
    dataset.to_parquet(args.output_file)
