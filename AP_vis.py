import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the JSON data
with open('output/props_res50/inference/coco_instances_results.json') as f:
    data = json.load(f)

# Organize detections by category
category_detections = defaultdict(list)
for detection in data:
    category_detections[detection['category_id']].append(detection)

# Calculate precision-recall for each category
category_metrics = {}
for cat_id, detections in category_detections.items():
    # Sort detections by confidence score (descending)
    sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
    
    # For simplicity, we'll assume ground truth counts (in real implementation, you'd need GT data)
    # This is a placeholder - in practice you'd compare with ground truth to calculate TP/FP
    num_gt = 50  # Example value - should be actual number of GT instances per category
    
    tp = []
    fp = []
    for i, det in enumerate(sorted_detections):
        # In real implementation, you'd check if this detection matches a GT box
        # Here we'll simulate some true/false positives for demonstration
        if np.random.rand() > 0.3:  # 70% chance of being TP for this demo
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recall = tp_cumsum / num_gt
    
    # Store metrics
    category_metrics[cat_id] = {
        'precision': precision,
        'recall': recall,
        'ap': np.trapz(precision, recall)  # Approximate AP
    }

# Plot AP curves for each category
plt.figure(figsize=(12, 8))
for cat_id, metrics in category_metrics.items():
    plt.plot(metrics['recall'], metrics['precision'], label=f'Category {cat_id} (AP={metrics["ap"]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves by Category(Res50)')
plt.legend()
plt.grid()
save_path = "output/visualization/AP_curve_Res50.png"
plt.savefig(save_path)
print(f"保存成功: {save_path}")