import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from mtcnn import MTCNN
from datetime import datetime

# Paths for input images and output directory
input_dir = 'C:\\Users\\hp\\Desktop\\PICS92024'
output_dir = 'C:\\Users\\hp\\Desktop\\OutputPICS9'
excel_file = 'C:\\Users\\hp\\Desktop\\9.xlsx'  # Update the path to your Excel file

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file, sheet_name='Sheet')
df.columns = df.columns.str.strip()  # Strip whitespace from column names

# Ensure admission numbers are strings and strip whitespace
df['Admission_Number'] = df['Admission_Number'].astype(str).str.strip()

# Create a dictionary for quick access (with correct key names)
student_records = df.to_dict(orient='records')

# Initialize the MTCNN face detector
detector = MTCNN()

# Log list to track processing status
not_processed = []
log_entries = []

# Function to process the image
def process_image(image, face_location, student_info):
    x, y, w, h = face_location
    
    # Calculate extra cropping space: 30% above/below and left/right
    extra_top = int(h * 0.3)
    extra_bottom = int(h * 0.3)
    extra_width = int(w * 0.3)
    
    # Define new crop boundaries
    new_top = max(0, y - extra_top)
    new_bottom = min(image.shape[0], y + h + extra_bottom)
    new_left = max(0, x - extra_width)
    new_right = min(image.shape[1], x + w + extra_width)
    
    # Crop the image
    cropped_image = image[new_top:new_bottom, new_left:new_right]
    
    # Resize to square format (1024x1024)
    pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    pil_image = pil_image.resize((1024, 1024), Image.LANCZOS)
    
    # Adjust brightness and contrast
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Add border
    border_size = 2
    bordered_image = Image.new('RGB', (1024 + border_size * 2, 1024 + border_size * 2), (255, 255, 255))
    bordered_image.paste(pil_image, (border_size, border_size))
    
    # Draw the student's name and date on the image
    draw = ImageDraw.Draw(bordered_image)
    font = ImageFont.load_default()  # Use default font
    current_date = datetime.now().strftime('%d-%m-%Y')
    text = f"{student_info['Name']}\n{current_date}"
    
    # Draw text on the image
    draw.text((10, 10), text, fill="black", font=font)
    
    # Save the image using the student's details
    output_file_name = f"{student_info['Name']}_{student_info['Admission_Number']}_{student_info['Class']}_{student_info['Section']}_{current_date}.jpg"
    output_path = os.path.join(output_dir, output_file_name)

    # Save the image initially to check its size
    bordered_image.save(output_path, "JPEG", quality=85)

    # Initialize quality for size checking
    quality = 85
    # Check size and reduce quality if needed
    while os.path.getsize(output_path) > 40 * 1024:  # 40 KB
        quality = int(quality * 0.9)  # Reduce quality by 10%
        bordered_image.save(output_path, "JPEG", quality=quality)

    return True

# Print student data from Excel
print("Student data read from Excel:")
for student in student_records:
    admission_number = student['Admission_Number']
    name = student['Name']
    student_class = student['Class']
    section = student['Section']
    print(f"Admission Number: {admission_number}, Name: {name}, Class: {student_class}, Section: {section}")

# Process all images in the directory
for file_name in os.listdir(input_dir):
    file_path = os.path.join(input_dir, file_name)
    
    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        print(f"Error reading {file_name}. File might be corrupted or in an unsupported format.")
        continue
    
    # Detect faces in the image
    faces = detector.detect_faces(image)
    
    # If a face is detected, process the image
    if faces:
        # Assuming you want to process the first detected face
        face_location = faces[0]['box']
        admission_number = ''.join(filter(str.isdigit, file_name))  # Extract admission number from the file name

        # Check if admission number exists in the records
        student_info = next((student for student in student_records if student['Admission_Number'] == admission_number), None)
        
        if student_info:
            print(f"\nProcessing image: {file_name}")
            print(f"Details - Admission Number: {student_info['Admission_Number']}, Name: {student_info['Name']}, Class: {student_info['Class']}, Section: {student_info['Section']}")
            
            processed_successfully = process_image(image, face_location, student_info)
            if processed_successfully:
                log_entries.append(f"Processed and saved: {file_name}")
            else:
                not_processed.append(student_info)
        else:
            print(f"No data found for admission number: {admission_number}")
            not_processed.append({'Admission_Number': admission_number, 'Name': 'Unknown', 'Class': 'Unknown', 'Section': 'Unknown'})
    else:
        print(f"No face found in {file_name}")
        not_processed.append({'Admission_Number': admission_number, 'Name': 'Unknown', 'Class': 'Unknown', 'Section': 'Unknown'})

# Write log to a file
with open(os.path.join(output_dir, 'log.txt'), 'w') as log_file:
    for entry in log_entries:
        log_file.write(entry + '\n')

# Display not processed information
print("Not processed images:")
for student in not_processed:
    print(student)
