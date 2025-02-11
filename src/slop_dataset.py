import json
from datasets import Dataset
from transformers import AutoTokenizer

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

class DatasetParser:
    def __init__(self, filename: str, tokenizer: AutoTokenizer, max_length: int):
        with open(filename, "r") as f:
            self.lines = [json.loads(x) for x in f.readlines()]

        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_sample_datapoints(self, sample: dict, score: float) -> list[dict]:
        original_text = sample["text"]
        tokens = self.tokenizer.encode(original_text, add_special_tokens=False)
        if len(tokens) <= self.max_length:
            return [{
                "id": sample["id"],
                "url": sample["metadata"]["url"],
                "text": original_text,
                "score": score
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
            if len(self.tokenizer.encode(candidate, add_special_tokens=False)) <= self.max_length:
                current_chunk = candidate
            else:
                if not current_chunk:
                    # Single segment is too long, split further on whitespace
                    words = segment.split()
                    subchunk = ""
                    for word in words:
                        candidate_word = subchunk + " " + word if subchunk else word
                        if len(self.tokenizer.encode(candidate_word, add_special_tokens=False)) <= self.max_length:
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
            datapoints.append({
                "id": sample["id"],
                "url": sample["metadata"]["url"],
                "text": chunks[0],
                "score": score,
                "prompt": prompt.format(sample["metadata"]["url"], chunks[0]),
                "completion": "a" if score > 0 else "b"
            })
        else:
            for i, chunk in enumerate(chunks):
                datapoints.append({
                    "id": f"{sample['id']}_{i}",
                    "url": sample["metadata"]["url"],
                    "text": chunk,
                    "score": score,
                    "prompt": prompt.format(sample["metadata"]["url"], chunk),
                    "completion": "a" if score > 0 else "b"
                })
        return datapoints

    def get_hf_slop_dataset(self) -> Dataset:
        ds = []
        for line in self.lines:
            sample_a, sample_b = line["sample"]
            score = (line["feedback"]["slop_comparison"] - 3) / 3  # this is the score for sample 2
            
            if score == 0:
                continue

            for sample in self.get_sample_datapoints(sample_a, -score) + self.get_sample_datapoints(sample_b, score):
                ds.append(sample)

        ds = Dataset.from_list(ds)

        return ds
