import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Step 1: Load image from URL
url = 'https://www.wallpaperflare.com/static/43/772/985/shana-2-of-2-grayscale-photography-wallpaper.jpg'
response = requests.get(url)
img_data = response.content
img = Image.open(BytesIO(img_data)).convert('L')  # Convert to grayscale using Pillow

# Convert the image to a numpy array (as OpenCV works with NumPy arrays)
image = np.array(img)

# Step 2: Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Step 3: Show both the original image and the histogram equalized image
plt.figure(figsize=(10, 5))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Equalized Image
plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()

# Step 4: Show histograms of both images
plt.figure(figsize=(10, 5))

# Original Image Histogram
plt.subplot(1, 2, 1)
plt.hist(image.ravel(), 256, [0, 256])
plt.title('Histogram of Original Image')

# Equalized Image Histogram
plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title('Histogram of Equalized Image')

plt.show()
