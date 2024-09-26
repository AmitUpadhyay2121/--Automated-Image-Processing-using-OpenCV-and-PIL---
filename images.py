import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from datetime import datetime

# Path to the directory with images and the Excel file
input_dir = 'C:\\Users\\hp\\Desktop\\PICS92024'
output_dir = 'C:\\Users\\hp\\Desktop\\OutputPICS9'
excel_file = 'C:\\Users\\hp\\Desktop\\9.xlsx'  # Update with the actual path to your Excel file

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file, sheet_name='Sheet')

# Strip whitespace from column names and values
df.columns = df.columns.str.strip()
df['Admission_Number'] = df['Admission_Number'].astype(str).str.strip()  # Ensure admission numbers are strings

# Create a dictionary for quick access
student_dict = pd.Series(df.Name.values, index=df.Admission_Number).to_dict()

# Function to crop the face region, resize the image, and adjust lighting and exposure
def process_image(image, face_location):
    x, y, w, h = face_location
    
    extra_top = int(h * 0.4)
    extra_bottom = int(h * 0.5)
    extra_width = int(w * 0.3)
    
    new_top = max(0, y - extra_top)
    new_bottom = min(image.shape[0], y + h + extra_bottom)
    new_left = max(0, x - extra_width)
    new_right = min(image.shape[1], x + w + extra_width)
    
    cropped_image = image[new_top:new_bottom, new_left:new_right]
    target_size = (1024, 1024)
    resized_image = cv2.resize(cropped_image, target_size)

    pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    
    # Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)

    final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return final_image

# Get the current date in DD-MM-YYYY format
current_date = datetime.now().strftime("%d-%m-%Y")

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
        processed_image = process_image(image, faces[0])
        
        # Extract admission number from the filename
        admission_number = ''.join(filter(str.isdigit, file_name.split('.')[0]))  # Extract digits from filename
        
        # Check if admission number exists in the dictionary
        if admission_number in student_dict:
            student_name = student_dict[admission_number]
            new_file_name = f"{student_name} {current_date}.jpg"
        else:
            print(f"Admission number {admission_number} not found in the Excel file. Using default naming.")
            new_file_name = f"Unknown {current_date}.jpg"  # Fallback name if admission number not found
        
        output_path = os.path.join(output_dir, new_file_name)
        
        # Save the processed image in JPG format
        cv2.imwrite(output_path, processed_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        print(f"Processed and saved: {new_file_name}")
    else:
        print(f"No face found in {file_name}")
