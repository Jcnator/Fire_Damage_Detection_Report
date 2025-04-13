import os
import random
from PIL import Image
from dataloader.dataloader import get_dataloader
from options import BaseOptions

import matplotlib.pyplot as plt

dataset_path = "./dataset/Images"

# Function to get a random image from the dataset
def get_random_image(dataset_path):
	image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.tif'))]
	if not image_files:
		raise FileNotFoundError("No image files found in the dataset directory.")
	
	# Select a random image
	random_image = random.choice(image_files)
	return os.path.join(dataset_path, random_image)

trainloader = get_dataloader(BaseOptions(), 'train', shuffle=True, augment=True)
# Get one image from the train loader
data_iter = iter(trainloader)
images, ann = next(data_iter)
images = images.cpu()

# Display one image
img = images[0]
plt.imshow(img[:3].permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
plt.title(f"Label: {ann[0]}")
plt.axis('off')
plt.show()