import pandas as pd
import argparse
import random

def format_data(problem_id, prompt, preferred, less_preferred):
    """Format the data into the desired structure for comparison."""
    formatted_preferred = f"Problem ID: {problem_id}\n\nPrompt: {prompt}\n\nResponse: {preferred}\n\n"
    formatted_less_preferred = f"Problem ID: {problem_id}\n\nPrompt: {prompt}\n\nResponse: {less_preferred}\n\n"
    return formatted_preferred, formatted_less_preferred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to input parquet file")
    parser.add_argument("output_file", type=str, help="Path to output parquet file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--order-by-length", action="store_true", help="Sort by sample length")
    args = parser.parse_args()
    
    dataset = {
        "sample_a": [],
        "sample_b": [],
        "score": []
    }
    
    random.seed(args.seed)

    df = pd.read_parquet(args.input_file)
    for _, row in df.iterrows():
        problem_id = row["problem_id"]
        prompt = row["prompt"]
        task_type = row["task_type"]  # Storing but not used in formatting
        preferred_response = row["preferred_response"]
        less_preferred_response = row["less_preferred_response"]
        preferred_id = row["preferred_response_id"]
        less_preferred_id = row["less_preferred_response_id"]
        preferred_score = row["preferred_response_score"]
        less_preferred_score = row["less_preferred_response_score"]
        
        # Calculate score difference
        score_diff = preferred_score - less_preferred_score
        
        # Format the responses
        formatted_preferred, formatted_less_preferred = format_data(
            problem_id, prompt, preferred_response, less_preferred_response
        )
        
        # Randomly assign to sample_a and sample_b to avoid bias
        if random.random() < 0.5:
            dataset["sample_a"].append(formatted_preferred)
            dataset["sample_b"].append(formatted_less_preferred)
        else:
            dataset["sample_a"].append(formatted_less_preferred)
            dataset["sample_b"].append(formatted_preferred)
            # Negate the score if we swap the order
            score_diff = -score_diff
            
        dataset["score"].append(score_diff)
    
    result_df = pd.DataFrame(dataset)
    
    # Sort by length to ease the model into the super long samples
    if args.order_by_length:
        result_df = result_df.sort_values(by="sample_a", key=lambda x: x.str.len())
        
    result_df.to_parquet(args.output_file)
    print(f"Processed {len(result_df)} samples. Output saved to {args.output_file}") 