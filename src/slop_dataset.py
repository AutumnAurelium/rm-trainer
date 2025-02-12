import json
from datasets import Dataset
from transformers import AutoTokenizer
from torch.utils.data import IterableDataset

prompt = """<task>
Classify the following passage as either high-quality, human-written data or spam, SEO-optimized, or machine-generated content.</task>

The URL of this webpage has also been provided.
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
<answer>
"""

class LazySlopIterableDataset(IterableDataset):
    def __init__(self, filename: str, tokenizer: AutoTokenizer, max_length: int):
        self.filename = filename
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    line_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if line_data is None or 'sample' not in line_data or 'feedback' not in line_data:
                    continue
                try:
                    sample_a, sample_b = line_data['sample']
                    score = (line_data['feedback']['slop_comparison'] - 3) / 3
                except Exception:
                    continue
                if score == 0:
                    continue
                for dp in process_sample(sample_a, self.tokenizer, self.max_length, -score):
                    if dp is not None:
                        yield dp
                for dp in process_sample(sample_b, self.tokenizer, self.max_length, score):
                    if dp is not None:
                        yield dp

def process_sample(sample, tokenizer, max_length, score):
    original_text = sample["text"]
    tokens = tokenizer.encode(original_text, add_special_tokens=False)
    
    if len(tokens) <= max_length:
        # Tokenize properly with return_tensors
        tokenized = tokenizer(
            original_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        return [{
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0],  # For causal LM training
            "score": score,
            "completion": "a" if score > 0 else "b"
        }]

    # Use newlines as primary delimiter, fallback to whitespace if no newline present
    if "\n" in original_text:
        segments = original_text.split("\n")
        joiner = "\n"
    else:
        segments = original_text.split()
        joiner = " "

    chunks = []
    current_chunk = ""
    for segment in segments:
        # Try adding the segment to the current chunk
        candidate = current_chunk + joiner + segment if current_chunk else segment
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_length:
            current_chunk = candidate
        else:
            if not current_chunk:
                # Single segment is too long, split further on whitespace
                words = segment.split()
                subchunk = ""
                for word in words:
                    candidate_word = subchunk + " " + word if subchunk else word
                    if len(tokenizer.encode(candidate_word, add_special_tokens=False)) <= max_length:
                        subchunk = candidate_word
                    else:
                        if subchunk:
                            chunks.append(subchunk)
                        subchunk = word
                current_chunk = subchunk
            else:
                chunks.append(current_chunk)
                current_chunk = segment
    if current_chunk:
        chunks.append(current_chunk)

    datapoints = []
    if len(chunks) == 1:
        tokenized = tokenizer(
            chunks[0],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=False
        )
        datapoints.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["input_ids"][0],
            "score": score,
            "completion": "a" if score > 0 else "b"
        })
    else:
        for i, chunk in enumerate(chunks):
            tokenized = tokenizer(
                chunk,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                add_special_tokens=False
            )
            datapoints.append({
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0],
                "score": score,
                "completion": "a" if score > 0 else "b"
            })
    return datapoints
