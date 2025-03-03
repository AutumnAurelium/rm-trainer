import pandas as pd
import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Input JSONL file")
    parser.add_argument("output_file", type=str, help="Output Parquet file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--order-by-length", action="store_true", help="Sort by conversation length")
    parser.add_argument("--score", type=float, default=1.0, help="Constant score value")
    args = parser.parse_args()
    
    dataset = {
        "sample_a": [],
        "sample_b": [],
        "score": []
    }
    
    random.seed(args.seed)

    with open(args.input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            original = data["original"]
            restoration = data["restoration"]
            
            score = args.score
            
            if random.random() < 0.5:  # avoid bias
                dataset["sample_a"].append(original)
                dataset["sample_b"].append(restoration)
            else:
                dataset["sample_a"].append(restoration)
                dataset["sample_b"].append(original)
            dataset["score"].append(score)
    
    df = pd.DataFrame(dataset)
    # sort by length to ease the model into the super long samples
    if args.order_by_length:
        df = df.sort_values(by="sample_a", key=lambda x: x.str.len())
        
    df.to_parquet(args.output_file) 