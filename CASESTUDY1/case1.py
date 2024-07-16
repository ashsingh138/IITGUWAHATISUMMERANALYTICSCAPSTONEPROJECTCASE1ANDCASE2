import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the CSV file
file_path = r'Bitext_Sample_Customer_Service_Training_Dataset.csv'
customer_queries_df = pd.read_csv(file_path)

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2-medium'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


tokenizer.pad_token = tokenizer.eos_token


def generate_response(query, intent):
    prompt = f"Query: {query}\nIntent: {intent}\nResponse:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response


responses = []
for index, row in customer_queries_df.iterrows():
    query = row['utterance']
    intent = row['intent']
    response = generate_response(query, intent)
    responses.append(response)
    print(f"Query: {query}\nIntent: {intent}\nResponse: {response}\n")


customer_queries_df['response'] = responses
output_file_path = 'customer_responses.csv'
customer_queries_df.to_csv(output_file_path, index=False)

print(f"Responses have been saved to {output_file_path}")
