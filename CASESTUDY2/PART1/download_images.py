

import pandas as pd
import os
from PIL import Image
import requests
from io import BytesIO

# Load the CSV file
csv_file = "fashion.csv"
data = pd.read_csv(csv_file)

# Create a directory to store images
image_dir = "dataset_images"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


for idx, row in data.iterrows():
    image_url = row['ImageURL']
    product_id = row['ProductId']
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.save(os.path.join(image_dir, f"{product_id}.jpg"))
    except Exception as e:
        print(f"Failed to download {image_url}: {e}")

print("Images downloaded and saved.")
