from typing import Literal
from transformers import AutoTokenizer
import re
import json
import argparse
import asyncio
from datasets import Dataset, Features, Value
import pandas as pd
import random


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
"""

def split_sample(tokenizer: AutoTokenizer, sample: str, max_tokens: int, split_on: Literal["newline", "period", "space", "token"] = "newline"):
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

def conv_likert(likert: int) -> float:
    return (likert - 1.0) / 6.0

async def process_data(
    sample_a: str,
    sample_b: str,
    sample_a_url: str,
    sample_b_url: str,
    score: int,
    tokenizer: AutoTokenizer,
    max_tokens: int
):
    if score == 4: # annotator expressed no preference
        return
    
    sample_a_split = split_sample(tokenizer, sample_a["text"], max_tokens)
    sample_b_split = split_sample(tokenizer, sample_b["text"], max_tokens)
    
    # This is better than doing all combos, it avoids duplicates.
    for i in range(max(len(sample_a_split), len(sample_b_split))):
        a_chunk = sample_a_split[i] if i < len(sample_a_split) else sample_a_split[i % len(sample_a_split)]
        b_chunk = sample_b_split[i] if i < len(sample_b_split) else sample_b_split[i % len(sample_b_split)]
        
        if random.random() < 0.5: # randomly select which sample is a vs b to avoid bias
            yield {
                "sample_a": prompt.format(sample_b_url, b_chunk),
                "sample_b": prompt.format(sample_a_url, a_chunk),
                "score": conv_likert(7 - score)
            }
        else: # prefers sample_a
            yield {
                "sample_a": prompt.format(sample_a_url, a_chunk),
                "sample_b": prompt.format(sample_b_url, b_chunk),
                "score": conv_likert(score)
            }

async def process_data_annotator(line: str, tokenizer: AutoTokenizer, max_tokens: int):
    data = json.loads(line)
    
    sample_a, sample_b = data["sample"]
    
    sample_a_url = sample_a["metadata"]["url"]
    sample_b_url = sample_b["metadata"]["url"]
    
    score = data["feedback"]["slop_comparison"]
    
    async for x in process_data(sample_a, sample_b, sample_a_url, sample_b_url, score, tokenizer, max_tokens):
        yield x

async def main(input_file: str, tokenizer: AutoTokenizer, max_tokens: int, mode: Literal["annotator", "synth"]) -> Dataset:
    examples = []

    with open(input_file, "r") as f:
        for line in f:
            if mode == "annotator":
                async for x in process_data_annotator(line, tokenizer, max_tokens):
                    examples.append(x)
            else:
                synth_df = pd.read_parquet(input_file)
                for _, row in synth_df.iterrows():
                    async for x in process_data(
                        row["text_a"],
                        row["text_b"],
                        row["url_a"],
                        row["url_b"],
                        row["score"],
                        tokenizer,
                        max_tokens
                    ):
                        examples.append(x)
    
    return Dataset.from_dict({
        "chosen": [x["chosen"] for x in examples],
        "rejected": [x["rejected"] for x in examples],
        "score": [x["score"] for x in examples]
    })
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--mode", type=str, default="annotator")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset = asyncio.run(main(tokenizer, args.max_tokens))
    dataset.to_parquet(args.output_file)