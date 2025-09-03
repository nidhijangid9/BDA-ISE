import torch
import cv2
import numpy as np

print("✅ Testing environment...")

# Test OpenCV
print("OpenCV version:", cv2.__version__)

# Test PyTorch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Load YOLOv5s model (will download weights first time)
print("Loading YOLOv5s model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create a dummy black image
dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

# Run detection
print("Running test inference...")
results = model(dummy_img)

# Print and show results
results.print()   # show detection results in terminal
results.show()    # show popup image with boxes (none for black image)

print("✅ All good! YOLOv5 loaded and inference ran successfully.")
