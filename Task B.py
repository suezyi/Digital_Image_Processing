# -*- coding: utf-8 -*-
"""
@author: elizabeth
"""

# Assignment Task B
# Detail comments are at right-side

# Import Required Modules
from matplotlib import pyplot as pt                                                                    # Matplotlib: For plotting graphs and images
import numpy as np                                                                                     # NumPy: For numerical computations
import cv2                                                                                             # OpenCV: For image processing
import os                                                                                              # OS: For file and directory operations

# Read and preprocess image: grayscale, blur, threshold, dilate
def preprocessImage(image_path):
    image = cv2.imread(image_path)                                                                     # Reads the image from file (BGR image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                                                     # Convert BGR image to grayscale image
    blur = cv2.GaussianBlur(gray, (7,7), 0)                                                            # Blur image to reduce noise
    
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) [1]                  # Threshold the blurred image using Otsu's method (binary inverse: text = white, background = black)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))                                          # Create a rectangular structuring element for dilation
    dilated = cv2.dilate(thresh, kernel, iterations = 8)                                               # Dilate the threshold image to merge text into paragraphs

    return image, gray, dilated                                                                        # Return original, gray, and dilated images to caller

# Show original image and row/column projections
def plotProjections(image, dilated):
    row_proj = np.sum(dilated, axis = 1)                                                               # Compute row projection: Sum of white pixels in each row
    column_proj = np.sum(dilated, axis = 0)                                                            # Compute column projection: Sum of white pixels in each column
    
    pt.figure(figsize = (15, 5))                                                                       # Create a large figure
    pt.subplot (1, 3, 1)                                                                               # First plot: Original Image
    pt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))                                                  # Convert BGR to RGB for display
    pt.title("Original image")                                           
    pt.axis("Off")                                                                                     # Hide axis for the image
    
    pt.subplot(1, 3, 2)                                                                                # Second plot: Row Projection
    pt.plot(row_proj, range(len(row_proj)))                                                            # Row projection graph
    pt.title("Row Projection")
    pt.ylabel("Row")
    pt.gca().invert_xaxis()                                                                            # Invert x-axis for better readability
    
    pt.subplot(1, 3, 3)                                                                                # Third plot: Column Projection
    pt.plot(column_proj)                                                                               # Column projection graph
    pt.title("Column Projection")
    pt.xlabel("Column")
    
    pt.tight_layout()                                                                                  # Adjust layout to avoid overlap
    pt.show()                                                                                          # Display the figure
    
# Return list of paragraph indices to skip based on filename
def getExcludedIndices(image_path):
    exclusions = {                                                                                     # Define which paragraphs to skip for specific images
        "004.png": [7],
        "005.png": [3],
        "007.png": [2],
        "008.png": [4]
        } 
    
    for key, indices in exclusions.items():                                                            # Check if current image path matches any exclusion
        if key in image_path:
            return indices   
                                                                                                       # Return list of indices to skip
    return []                                                                                          # If no exclusions, return empty list

# Find contours, extract, and save paragraphs
def extractAndSaveParagraphs(image, gray, dilated, output_directory, image_path):
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)                    # Find external contours of connected components in the dilated image
    boxes = []
    for c in cnts:                                                                                     # Get bounding boxes for each contour
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))

    excluded = getExcludedIndices(image_path)                                                          # Get list of indices of paragraphs to exclude for this specific image
    boxes = [b for i, b in enumerate(boxes) if i not in excluded]                                      # Remove boxes with indices that should be excluded

    # Step 1: Sort left-to-right first (columns)
    boxes.sort(key=lambda b: b[0])

    # Step 2: Cluster into columns
    col_tolerance = 50  
    columns = []
    current_col = []
    last_x = -1

    # Loop through all boxes to cluster into columns
    for box in boxes:
        x1, y1, x2, y2 = box
        if last_x == -1 or abs(x1 - last_x) <= col_tolerance:
            current_col.append(box)
            last_x = x1
        else:
            columns.append(current_col)
            current_col = [box]
            last_x = x1
            
    if current_col:
        columns.append(current_col)

    # Step 3: Within each column, sort boxes from top-to-bottom
    sorted_boxes = []
    for col in columns:
        col.sort(key=lambda b: b[1])  # sort by y
        sorted_boxes.extend(col)

    # Step 4: Create output directory if it does not already exist
    os.makedirs(output_directory, exist_ok=True)

    # Step 5: Crop and save paragraph block as separate image
    for idx, box in enumerate(sorted_boxes):
        x1, y1, x2, y2 = box
        paragraph = image[y1:y2, x1:x2]
        filename = os.path.join(output_directory, f"paragraph_{idx+1}.png")
        cv2.imwrite(filename, paragraph)                                                               # Save the cropped paragraph image
    
# Full pipeline: Preprocess, Plot Projections, Extract and Save Paragraphs
def processImage(image_path, output_directory):
    image, gray, dilated = preprocessImage(image_path)                                                 # Preprocess the image
    plotProjections(image, dilated)                                                                    # Show projections
    extractAndSaveParagraphs(image, gray, dilated, output_directory, image_path)                       # Extract paragraphs and save them
    
# Process all image in the folder one by one
def main():
    image_folder = os.path.join("Converted Paper (8)")
    output_base = os.path.join(".")                                                                    # Base output folder
    image_filenames = [                                                                                # List of image filenames to process
        "001.png", "002.png", "003.png", "004.png", 
        "005.png", "006.png", "007.png", "008.png"
    ]
    
    for filename in image_filenames:                                                                   # Loop through all images
        image_path = os.path.join(image_folder, filename)                                              # Full path to image
        output_directory = os.path.join(output_base, f'extracted_paragraphs_{filename[:3]}')           # Output folder for this image
        print(f"Processing {filename}...")                                                             # Show progress
        processImage(image_path, output_directory)                                                     # Run the full pipeline
 
if __name__ == "__main__":                                                                             # Run main() only if this file is executed directly
    main()