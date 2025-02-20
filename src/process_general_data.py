import pandas as pd
import argparse
import json
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
    args = parser.parse_args()
    
    dataset = {
        "chosen": [],
        "rejected": [],
        "score": []
    }

    df = pd.read_parquet(args.input_file)
    for _, row in df.iterrows():
        context = json.loads(row["context"])
        response_a = json.loads(row["response_a"])
        response_b = json.loads(row["response_b"])
        score = row["score"]
        
        if score > 0:
            dataset["chosen"].append(format_convo(context + [response_b]))
            dataset["rejected"].append(format_convo(context + [response_a]))
        else:
            dataset["chosen"].append(format_convo(context + [response_a]))
            dataset["rejected"].append(format_convo(context + [response_b]))
        dataset["score"].append(abs(score))
        
    pd.DataFrame(dataset).to_parquet(args.output_file)
        
