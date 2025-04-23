import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
import os

# Import the function to add custom config
from diffusiondet.config import add_diffusiondet_config
from diffusiondet.predictor import VisualizationDemo

def setup_cfg():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)  # Add custom config for DiffusionDet
    cfg.merge_from_file("configs/diffdet.props.res50.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  
    cfg.freeze()
    return cfg

def visualize_detections(image_path, demo, metadata):
    im = cv2.imread(image_path)
    
    predictions, visualized_output = demo.run_on_image(im)
    
    output_path = os.path.join("output/visualization", os.path.basename(image_path))
    os.makedirs("output/visualization", exist_ok=True)
    cv2.imwrite(output_path, visualized_output.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_path}")
    print(f"Detected {len(predictions['instances'])} instances")

def main():
    cfg = setup_cfg()
    
    demo = VisualizationDemo(cfg)

    metadata = MetadataCatalog.get("props_val")
    
    test_images = [
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004981.jpg",
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004982.jpg",
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004983.jpg",
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/000026.jpg"
    ]
    
    for img_path in test_images:
        visualize_detections(img_path, demo, metadata)

if __name__ == "__main__":
    main() 