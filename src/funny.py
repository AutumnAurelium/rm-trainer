import random
from qwen2_rm import Qwen2ForCausalLMPermittedTokens
from transformers import AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

permitted_token_ids = list(set([random.randint(0, len(tokenizer)) for _ in range(1000)]))

model = Qwen2ForCausalLMPermittedTokens.from_pretrained(model_name, permitted_token_ids=permitted_token_ids)

model.eval()

def generate_response(prompt, max_length=100):
    # Format the conversation using the chat template
    chat_formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False
    )
    
    inputs = tokenizer(chat_formatted, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the input prompt from the response
    response = response.replace(chat_formatted, "").strip()
    return response

def main():
    print("-" * 50)
    
    conversation_history = []
    while True:
        user_input = input("\nYou: ").strip()
        
        conversation_history.append({"role": "user", "content": user_input})
        chat_formatted = tokenizer.apply_chat_template(conversation_history, tokenize=False)
        response = generate_response(chat_formatted)
        
        conversation_history.append({"role": "assistant", "content": response})
        print("\nAssistant:", response)

if __name__ == "__main__":
    main()

