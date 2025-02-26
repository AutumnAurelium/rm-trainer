import pandas as pd
import argparse
import json
import random

def format_convo(convo: list[dict]) -> str:
    formatted = ""
    for msg in convo:
        role = msg["role"].title()
        content = msg["content"]
        formatted += f"{role}: {content}\n\n"
    return formatted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--order-by-length", action="store_true")
    args = parser.parse_args()
    
    dataset = {
        "sample_a": [],
        "sample_b": [],
        "score": []
    }
    
    random.seed(args.seed)

    df = pd.read_parquet(args.input_file)
    for _, row in df.iterrows():
        context = json.loads(row["context"])
        response_a = json.loads(row["response_a"])
        response_b = json.loads(row["response_b"])
        score = row["score"]
        
        if random.random() < 0.5: # avoid bias
            dataset["sample_a"].append(format_convo(context + [response_b]))
            dataset["sample_b"].append(format_convo(context + [response_a]))
        else:
            dataset["sample_a"].append(format_convo(context + [response_a]))
            dataset["sample_b"].append(format_convo(context + [response_b]))
        dataset["score"].append(score)
    
    df = pd.DataFrame(dataset)
    # sort by length to ease the model into the super long samples
    if args.order_by_length:
        df = df.sort_values(by="sample_a", key=lambda x: x.str.len())
        
    df.to_parquet(args.output_file)
        
