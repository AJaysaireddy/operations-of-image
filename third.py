import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('kio.png')

# Resizing to 224x224x3
resized_image = cv2.resize(image, (224, 224))

# Grayscale conversion
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Cropping to extract only the dog (adjust coordinates based on your image)
cropped_image = resized_image[50:200, 50:200]

# Rotation by 45 degrees
rows, cols, _ = resized_image.shape
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated_image = cv2.warpAffine(resized_image, rotation_matrix, (cols, rows))

# Flip left to right
flipped_image = cv2.flip(resized_image, 1)

# Gaussian blurring
gaussian_blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# Median blurring
median_blurred_image = cv2.medianBlur(resized_image, 5)

# Edge detection
edges = cv2.Canny(gray_image, 100, 200)

# Display the results using Matplotlib
plt.figure(figsize=(12, 6))

plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.title('Resized Image')

plt.subplot(2, 4, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title('Cropped Image')

plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
plt.title('Rotated Image')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.title('Flipped Image')

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(gaussian_blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Blurred Image')

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(median_blurred_image, cv2.COLOR_BGR2RGB))
plt.title('Median Blurred Image')

plt.subplot(2, 4, 8)
plt.imshow(edges, cmap='gray')
plt.title('Edges')

plt.tight_layout()
plt.show()
