import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json

PROPS_DATASET_PATH = "datasets/PROPS-Detection-Dataset"

# Define class names and their indices
PROPS_CLASSES = [
    "master_chef_can", "cracker_box", "sugar_box",
    "tomato_soup_can", "mustard_bottle", "tuna_fish_can",
    "gelatin_box", "potted_meat_can", "mug", "large_marker"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(PROPS_CLASSES)}

def convert_props_to_coco(annotations):
    """Convert PROPS format annotations to COCO format."""
    # PROPS format is a list of [image_path, annotations]
    # COCO format needs to be a dict with images, annotations, categories
    
    # Create categories list
    categories = [
        {"id": i, "name": name} for i, name in enumerate(PROPS_CLASSES)
    ]
    
    # Convert annotations
    coco_annotations = []
    coco_images = []
    ann_id = 1
    
    for img_id, (img_path, anns) in enumerate(annotations, 1):
        # Extract just the filename from the path
        img_filename = os.path.basename(img_path)
        
        # Add image info
        coco_images.append({
            "id": img_id,
            "file_name": img_filename,  # Just use the filename, not the full path
            "height": 480,  # PROPS images are 640x480
            "width": 640
        })
        
        # Add annotations
        for ann in anns:
            x1, y1, x2, y2 = ann["xyxy"]
            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_TO_IDX[ann["name"]],  # Use class index instead of name
                "bbox": [x1, y1, x2-x1, y2-y1],
                "area": (x2-x1) * (y2-y1),
                "iscrowd": 0
            })
            ann_id += 1
    
    return {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }

def register_props(name, json_file, image_root):
    # Load the original PROPS format annotations
    with open(json_file, 'r') as f:
        props_annotations = json.load(f)
    
    # Convert to COCO format
    coco_annotations = convert_props_to_coco(props_annotations)
    
    # Save converted annotations to a temporary file
    temp_json_file = json_file + ".coco.json"
    with open(temp_json_file, 'w') as f:
        json.dump(coco_annotations, f)
    
    # Register using the converted file
    DatasetCatalog.register(name, lambda: load_coco_json(temp_json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=temp_json_file,
        image_root=image_root,
        evaluator_type="coco",
    )

def register_props_instances():
    register_props(
        "props_train",
        os.path.join(PROPS_DATASET_PATH, "train.json"),
        os.path.join(PROPS_DATASET_PATH, "PROPS-Detection"),
    )
    register_props(
        "props_val",
        os.path.join(PROPS_DATASET_PATH, "val.json"),
        os.path.join(PROPS_DATASET_PATH, "PROPS-Detection"),
    ) 