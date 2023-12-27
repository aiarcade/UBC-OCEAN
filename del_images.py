import os
import cv2
import numpy as np

def calculate_white_percentage(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Set a threshold to identify white pixels
    _, thresholded = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    
    # Calculate the percentage of white pixels
    white_percentage = np.sum(thresholded == 255) / (image.shape[0] * image.shape[1]) * 100
    
    return white_percentage

def main(directory_path, threshold=80):
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory_path, filename)
            
            # Read the image
            image = cv2.imread(file_path)
            
            # Calculate the percentage of white pixels
            white_percentage = calculate_white_percentage(image)
            
            # Check if the percentage exceeds the threshold
            if white_percentage > threshold:
                # Remove the image
                os.remove(file_path)
                print(f"Image '{filename}' removed. White percentage: {white_percentage:.2f}%")

if __name__ == "__main__":
    # Replace 'path_to_images' with the actual path to your images directory
    images_directory = '../train_g/EC/'
    
    # Set the threshold percentage (default is 80)
    white_threshold = 60
    
    main(images_directory, white_threshold)
