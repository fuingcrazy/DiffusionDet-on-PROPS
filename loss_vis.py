import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# 日志路径
log_file = "output/props_res50/metrics.json"

# 创建保存图像的目录
os.makedirs("output/visualization", exist_ok=True)

# 读取 metrics.json 并提取有用数据
records = [json.loads(line) for line in open(log_file, "r") if line.strip() and "total_loss" in line]
df = pd.DataFrame(records)

# 画 loss 曲线图
plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["total_loss"], label="Total Loss")
if "loss_cls" in df:
    plt.plot(df["iteration"], df["loss_cls"], label="Cls Loss", linestyle='--')
if "loss_box_reg" in df:
    plt.plot(df["iteration"], df["loss_box_reg"], label="Box Reg Loss", linestyle='--')
if "loss_giou" in df:
    plt.plot(df["iteration"], df["loss_giou"], label="GIoU Loss", linestyle='--')

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像
save_path = "output/visualization/loss_curve.png"
plt.savefig(save_path)
print(f"保存成功: {save_path}")
