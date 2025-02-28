import os
import csv
import json
import random
import cv2  # For reading image dimensions
from collections import defaultdict

# Configuration
input_csv = "/data/cardetection/data/trainbox.csv"
image_dir = "/data/cardetection/data/training_images/"  # Directory containing the images
output_train = "/data/cardetection/data/coco_train.json"
output_valid = "/data/cardetection/data/coco_valid.json"
train_ratio = 0.8
random_seed = 42
class_name = "car"

# Initialize data structures
image_groups = defaultdict(list)
image_dimensions = {}

# Read and group annotations
with open(input_csv, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["image"]
        image_path = os.path.join(image_dir, filename)
        
        # Get image dimensions once per file
        if filename not in image_dimensions:
            try:
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                image_dimensions[filename] = (w, h)
            except:
                print(f"Warning: Could not read dimensions for {filename}, using defaults")
                image_dimensions[filename] = (676, 380)  # Fallback dimensions
        
        # Store annotation
        image_groups[filename].append({
            "xmin": float(row["xmin"]),
            "ymin": float(row["ymin"]),
            "xmax": float(row["xmax"]),
            "ymax": float(row["ymax"])
        })

# Prepare for splitting
image_files = list(image_groups.keys())
random.seed(random_seed)
random.shuffle(image_files)
split_idx = int(len(image_files) * train_ratio)
train_files = image_files[:split_idx]
valid_files = image_files[split_idx:]

def generate_coco_dataset(filenames, annotation_id_start=1):
    dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": class_name}]
    }
    
    image_id = 1
    annotation_id = annotation_id_start
    
    for filename in filenames:
        # Get dimensions
        w, h = image_dimensions[filename]
        
        # Add image entry
        dataset["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": w,
            "height": h
        })
        
        # Add all annotations for this image
        for box in image_groups[filename]:
            xmin = box["xmin"]
            ymin = box["ymin"]
            width = box["xmax"] - xmin
            height = box["ymax"] - ymin
            
            dataset["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": []
            })
            annotation_id += 1
        
        image_id += 1
    
    return dataset, annotation_id

# Generate datasets with continuous annotation IDs
train_dataset, last_id = generate_coco_dataset(train_files)
valid_dataset, _ = generate_coco_dataset(valid_files, last_id)

# Save files
with open(output_train, "w") as f:
    json.dump(train_dataset, f, indent=4)

with open(output_valid, "w") as f:
    json.dump(valid_dataset, f, indent=4)

print(f"Generated COCO datasets:")
print(f"Training: {len(train_files)} images, {len(train_dataset['annotations'])} annotations")
print(f"Validation: {len(valid_files)} images, {len(valid_dataset['annotations'])} annotations")