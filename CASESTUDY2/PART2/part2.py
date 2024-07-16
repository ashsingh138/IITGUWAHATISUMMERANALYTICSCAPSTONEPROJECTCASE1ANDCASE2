import pandas as pd
import os
import requests
from PIL import Image
import time
import cv2
import mediapipe as mp


df = pd.read_csv('fashion.csv')


os.makedirs('clothing_images', exist_ok=True)

def download_image(url, path, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()  
            with open(path, 'wb') as f:
                f.write(response.content)
            return True
        except (requests.exceptions.RequestException, ConnectionResetError) as e:
            print(f"Error downloading {url}: {e}. Retrying {attempt + 1}/{retries}...")
            attempt += 1
            time.sleep(2)  
    return False


for index, row in df.iterrows():
    image_url = row['ImageURL']
    image_path = os.path.join('clothing_images', f"{row['ProductId']}.jpg")
    if not os.path.exists(image_path):
        success = download_image(image_url, image_path)
        if not success:
            print(f"Failed to download {image_url} after multiple attempts.")

print("Image download process completed.")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

def estimate_pose(image_path):
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image {image_path}.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    return results


sample_human_image_path = '8ad9f73e23909b9ff04db127fb369b2c.png'
results = estimate_pose(sample_human_image_path)
if results:
    print("Pose estimation done.")

def segment_clothing(image_path):
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        return None
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read the image {image_path}.")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image


sample_clothing_image_path = 'th.jpeg'
segmented_image = segment_clothing(sample_clothing_image_path)
if segmented_image is not None:
    cv2.imwrite('segmented_clothing.jpg', segmented_image)
    print("Clothing segmentation done.")

def virtual_tryon(human_image_path, clothing_image_path, output_path):
    human_image = cv2.imread(human_image_path)
    if human_image is None:
        print(f"Failed to read the human image {human_image_path}.")
        return
    clothing_image = segment_clothing(clothing_image_path)
    if clothing_image is None:
        print(f"Failed to segment the clothing image {clothing_image_path}.")
        return
    
    
    results = estimate_pose(human_image_path)
    if not results or not results.pose_landmarks:
        print("Pose estimation failed.")
        return
    
    
    landmarks = results.pose_landmarks.landmark
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    

    top_left = (int(left_shoulder.x * human_image.shape[1]), int(left_shoulder.y * human_image.shape[0]))
    bottom_right = (int(right_hip.x * human_image.shape[1]), int(right_hip.y * human_image.shape[0]))
    
    # Resize the clothing image
    clothing_resized = cv2.resize(clothing_image, (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]))
    
    # Overlay clothing onto human image
    human_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = clothing_resized
    
    # Save the output image
    cv2.imwrite(output_path, human_image)

# Test virtual try-on
virtual_tryon(sample_human_image_path, sample_clothing_image_path, 'virtual_tryon_output.jpg')
print("Virtual try-on done.")
