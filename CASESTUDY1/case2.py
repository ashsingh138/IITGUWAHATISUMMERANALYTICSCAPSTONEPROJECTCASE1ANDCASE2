import os
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model_name = "gpt2"  
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


csv_file = r'shopping_trends_updated.csv'  
customer_data_df = pd.read_csv(csv_file)


customer_data_list = [
    {
        'customer_id': row['Customer ID'],
        'age': row['Age'],
        'gender': row['Gender'],
        'item_purchased': row['Item Purchased'],
        'category': row['Category'],
        'purchase_amount': row['Purchase Amount (USD)'],
        'location': row['Location'],
        'size': row['Size'],
        'color': row['Color'],
        'season': row['Season'],
        'review_rating': row['Review Rating'],
        'subscription_status': row['Subscription Status'],
        'shipping_type': row['Shipping Type'],
        'discount_applied': row['Discount Applied'],
        'promo_code_used': row['Promo Code Used'],
        'previous_purchases': row['Previous Purchases'],
        'payment_method': row['Payment Method'],
        'frequency_of_purchases': row['Frequency of Purchases']
    }
    for idx, row in customer_data_df.iterrows()
]


def generate_personalized_message(customer_data):
    
    prompt = f"""
    Customer ID: {customer_data['customer_id']}
    Age: {customer_data['age']}
    Gender: {customer_data['gender']}
    Item Purchased: {customer_data['item_purchased']}
    Category: {customer_data['category']}
    Purchase Amount: {customer_data['purchase_amount']}
    Location: {customer_data['location']}
    Size: {customer_data['size']}
    Color: {customer_data['color']}
    Season: {customer_data['season']}
    Review Rating: {customer_data['review_rating']}
    Subscription Status: {customer_data['subscription_status']}
    Shipping Type: {customer_data['shipping_type']}
    Discount Applied: {customer_data['discount_applied']}
    Promo Code Used: {customer_data['promo_code_used']}
    Previous Purchases: {customer_data['previous_purchases']}
    Payment Method: {customer_data['payment_method']}
    Frequency of Purchases: {customer_data['frequency_of_purchases']}
    
    Generate a personalized engagement message that:
    - Thanks the customer for their recent purchase of {customer_data['item_purchased']} in {customer_data['color']} color.
    - Mentions how their purchase fits the current season ({customer_data['season']}).
    - Suggests related products or accessories in the same category ({customer_data['category']}).
    - Appreciates their loyalty if they have a high number of previous purchases or a subscription.
    - Encourages them to take advantage of current promotions or discounts.
    - Invites them to provide a review if their review rating is low ({customer_data['review_rating']}).
    """

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_new_tokens=150, 
        num_return_sequences=1, 
        no_repeat_ngram_size=2, 
        early_stopping=True,
        num_beams=3,  
        pad_token_id=tokenizer.eos_token_id,  
    )
    
    message = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return message.split('\n')[-1].strip()  


num_customers_to_generate = 500  
for customer_data in customer_data_list[:num_customers_to_generate]:
    personalized_message = generate_personalized_message(customer_data)
    print(f"Message for Customer ID {customer_data['customer_id']}: {personalized_message}\n")
