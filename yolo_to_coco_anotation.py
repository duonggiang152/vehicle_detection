import os
import json
import glob
import cv2

CLASS_NAMES = {0: "car"}

def yolo_to_coco_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = x_center - (width / 2)
    y_min = y_center - (height / 2)
    return [x_min, y_min, width, height]

def process(YOLO_ANNOTATIONS_DIR, IMAGES_DIR, OUTPUT_COCO_JSON):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for class_id, class_name in CLASS_NAMES.items():
        coco_output["categories"].append({
            "id": class_id + 1,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    # Fix: Check for common image extensions
    image_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']

    for txt_file in glob.glob(os.path.join(YOLO_ANNOTATIONS_DIR, "*.txt")):
        base_name = os.path.basename(txt_file).replace(".txt", "")
        img_path = None

        # Look for image with any common extension
        for ext in image_extensions:
            possible_path = os.path.join(IMAGES_DIR, f"{base_name}.{ext}")
            if os.path.exists(possible_path):
                img_path = possible_path
                break

        if not img_path:
            print(f"Warning: Image for {base_name} not found, skipping...")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue
        img_height, img_width, _ = img.shape

        # Add image metadata
        coco_output["images"].append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "width": img_width,
            "height": img_height
        })

        # Read YOLO annotations
        with open(txt_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                bbox = list(map(float, parts[1:5]))

                coco_bbox = yolo_to_coco_bbox(bbox, img_width, img_height)

                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1
    os.makedirs(os.path.dirname(OUTPUT_COCO_JSON), exist_ok=True)
    with open(OUTPUT_COCO_JSON, "w") as json_file:
        json.dump(coco_output, json_file, indent=4)

    print(f"Conversion completed! Saved to {OUTPUT_COCO_JSON}")

BASE_DIR = "/home/giang/data/topviewcar/Vehicle_Detection_Image_Dataset"
TRAIN_YOLO_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "train/labels")
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "train/images")
TRAIN_OUTPUT_COCO_JSON = os.path.join(BASE_DIR, "train/coco_train.json")

VALID_YOLO_ANNOTATIONS_DIR = os.path.join(BASE_DIR, "valid/labels")
VALID_IMAGES_DIR = os.path.join(BASE_DIR, "valid/images")
VALID_OUTPUT_COCO_JSON = os.path.join(BASE_DIR, "valid/coco_valid.json")
process(TRAIN_YOLO_ANNOTATIONS_DIR, TRAIN_IMAGES_DIR, TRAIN_OUTPUT_COCO_JSON)
process(VALID_YOLO_ANNOTATIONS_DIR, VALID_IMAGES_DIR, VALID_OUTPUT_COCO_JSON)