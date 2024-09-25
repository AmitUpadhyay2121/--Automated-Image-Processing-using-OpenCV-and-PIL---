import cv2
import os
import numpy as np  # Import numpy for array handling
from PIL import Image, ImageEnhance

# Path to the directory with images
input_dir = 'C:\\Users\\hp\\Desktop\\PICS92024'
output_dir = 'C:\\Users\\hp\\Desktop\\OutputPICS9'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to crop the face region, resize the image, and adjust lighting and exposure
def process_image(image, face_location):
    x, y, w, h = face_location
    
    # Calculate extra cropping space: 40% above head, 50% below neck, and 30% from left/right
    extra_top = int(h * 0.4)   # 40% of face height for above the head
    extra_bottom = int(h * 0.5)  # 50% of face height for below the neck
    extra_width = int(w * 0.3)  # 30% of face width for left and right
    
    # Define the new crop boundaries
    new_top = max(0, y - extra_top)  # 40% above the head
    new_bottom = min(image.shape[0], y + h + extra_bottom)  # 50% below the neck
    new_left = max(0, x - extra_width)  # 30% left of the face
    new_right = min(image.shape[1], x + w + extra_width)  # 30% right of the face
    
    # Crop the image around the face with extra space
    cropped_image = image[new_top:new_bottom, new_left:new_right]

    # Maintain a square aspect ratio by resizing
    # Square format (1024x1024)
    target_size = (1024, 1024)
    
    # Resize the cropped image to a square
    resized_image = cv2.resize(cropped_image, target_size)

    # Convert to PIL Image for brightness and contrast adjustment
    pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    # Adjust brightness and contrast using PIL
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase brightness slightly

    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)  # Increase contrast slightly

    # Convert back to OpenCV format
    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return final_image

# Process all images in the directory
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error reading {file_name}. File might be corrupted or in an unsupported format.")
        continue
    
    # Convert image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    
    # If a face is detected, process the image
    if len(faces) > 0:
        # Assuming you want to process the first detected face
        processed_image = process_image(image, faces[0])
        output_file_name = os.path.splitext(file_name)[0] + '.jpg'  # Save as JPG
        output_path = os.path.join(output_dir, output_file_name)
        
        # Save the processed image in JPG format
        cv2.imwrite(output_path, processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        print(f"Processed and saved: {output_file_name}")
    else:
        print(f"No face found in {file_name}")
