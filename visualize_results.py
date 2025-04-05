import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os

# Import the function to add custom config
from diffusiondet.config import add_diffusiondet_config

def setup_cfg():
    cfg = get_cfg()
    add_diffusiondet_config(cfg)  # Add custom config for DiffusionDet
    cfg.merge_from_file("configs/diffdet.props.res50.yaml")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def visualize_detections(image_path, predictor, metadata):
    # 读取图像
    im = cv2.imread(image_path)
    
    # 进行预测
    outputs = predictor(im)
    
    # 可视化结果
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # 保存结果
    output_path = os.path.join("output/visualization", os.path.basename(image_path))
    os.makedirs("output/visualization", exist_ok=True)
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"Saved visualization to {output_path}")

def main():
    # 设置配置
    cfg = setup_cfg()
    
    # 创建预测器
    predictor = DefaultPredictor(cfg)
    
    # 获取元数据
    metadata = MetadataCatalog.get("props_val")
    
    # 可视化测试集中的一些图像
    test_images = [
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004981.jpg",
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004982.jpg",
        "datasets/PROPS-Detection-Dataset/PROPS-Detection/004983.jpg"
    ]
    
    for img_path in test_images:
        visualize_detections(img_path, predictor, metadata)

if __name__ == "__main__":
    main() 