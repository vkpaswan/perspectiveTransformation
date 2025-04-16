# perspectiveTransformation
# This repo can be helpful for transforming SOURCE TO TARGET coordinates i.e. from camera view point to world view point using perspective geometry.

# ---------------------------------------------
# 🚀 Perspective Transform with Visualization
# ---------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# 🔹 Load the input frame
frame = cv2.imread('/content/drive/MyDrive/ResearchWorks/VehicleSpeedEstimationAndTrafficAnalysis/Code/Input/pt.jpeg')

# 🔹 Function to draw a coordinate grid on the frame
def draw_grid(frame, spacing=50):
    height, width, _ = frame.shape
    print("Image Dimensions:", height, width)

    for y in range(0, height, spacing):
        cv2.line(frame, (0, y), (width, y), (0, 0, 255), 1)
        cv2.putText(frame, f'{y}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 0, 255), 2, cv2.LINE_AA)

    for x in range(0, width, spacing):
        cv2.line(frame, (x, 0), (x, height), (0, 0, 255), 1)
        cv2.putText(frame, f'{x}', (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# ✳️ Draw coordinate grid on image
draw_grid(frame)

# ---------------------------------------------
# 🟢 Define source and target points for transformation
# ---------------------------------------------

# These points are manually identified from the image
SOURCE = np.array([
    [190, 375],
    [540, 390],
    [760, 940],
    [100, 975]
])

# Define the size (in arbitrary units) of the top-down target space
TARGET_WIDTH = 15
TARGET_HEIGHT = 25

# These are the corner points of a rectangle representing the target (top-down) view
TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

# ---------------------------------------------
# 📐 Perspective Transformation Class
# ---------------------------------------------

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(source.astype(np.float32), target.astype(np.float32))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points is None or len(points) == 0:
            return points
        reshaped_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# 🔁 Initialize the transformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

# 📍 Points to transform from original frame (camera view point)
points_source = [(400, 535), (420, 650)]

# 🧮 Transform the points to top-down coordinates (world view point)
points_target = view_transformer.transform_points(points=points_source).astype(int)
print("Transformed Points (Target):", points_target)

# ---------------------------------------------
# 🖼️ Visualization on the frame
# ---------------------------------------------

# Draw the SOURCE polygon on the frame
cv2.polylines(frame, [SOURCE], isClosed=True, color=(0, 255, 0), thickness=2)

# Draw SOURCE corners
for (x, y) in SOURCE:
    cv2.circle(frame, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

# Draw the source points to be transformed
for i, (x, y) in enumerate(points_source):
    cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.putText(frame, f"S{i}", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Display transformed (target) coordinates on the image
for i, (tx, ty) in enumerate(points_target):
    label = f"T{i}: ({tx},{ty})"
    cv2.putText(frame, label, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# ---------------------------------------------
# 📊 Display Output with Matplotlib
# ---------------------------------------------

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.title("SOURCE Region and Transformed Points")
plt.axis('off')
plt.show()

