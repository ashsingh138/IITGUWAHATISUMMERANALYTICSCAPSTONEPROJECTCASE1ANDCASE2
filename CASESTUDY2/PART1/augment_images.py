

import os
import random
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image


generated_images_dir = "./generated_images"
output_augmented_images_dir = "./augmented_images"


if not os.path.exists(output_augmented_images_dir):
    os.makedirs(output_augmented_images_dir)

# Define transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Apply augmentations and save images
for img_name in os.listdir(generated_images_dir):
    img_path = os.path.join(generated_images_dir, img_name)
    image = Image.open(img_path)
    
    for i in range(10):  
        augmented_image = augmentation_transforms(image)
        save_path = os.path.join(output_augmented_images_dir, f"{img_name.split('.')[0]}_aug_{i}.jpg")
        save_image(augmented_image, save_path)
