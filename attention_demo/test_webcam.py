import torch
import cv2

# Load YOLOv5s model (pretrained on COCO)
print("Loading YOLOv5s model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLOv5 inference
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Show frame
    cv2.imshow("YOLOv5 Webcam", annotated_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
