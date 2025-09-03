# attention_demo.py
import cv2
import numpy as np
import torch
import time
import pandas as pd
from tracker import CentroidTracker
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ---------------- CONFIG ----------------
VIDEO_SRC = 0  # webcam index; set to video file path to test on recorded video
SCORE_INIT = 50
SCORE_MAX = 100
SCORE_MIN = 0
SCORE_STEP_UP = 1
SCORE_STEP_DOWN = 1
CSV_LOG = "attention_log.csv"
PLOT_UPDATE_INTERVAL = 10  # frames between plot updates
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
# -----------------------------------------

# load YOLOv5s from torch.hub (will download weights first time)
print("Loading YOLOv5s model (this may take a few seconds the first time)...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.35  # confidence threshold
model.iou = 0.45

# COCO class indexes we care about
# person -> 'person', phone -> 'cell phone' (COCO name 'cell phone'), book 'book', laptop 'laptop'
COCO_NAMES = model.names  # mapping idx -> name
wanted_classes = {'person', 'cell phone', 'book', 'laptop'}

# helper: compute bbox intersection / IoU
def iou(boxA, boxB):
    # boxes are (x1,y1,x2,y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

# start video capture
cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

ct = CentroidTracker(max_disappeared=40, max_distance=100)

# store per-ID info
scores = defaultdict(lambda: SCORE_INIT)
last_state = {}  # 'attentive' or 'distracted'
history = []  # rows for CSV

frame_count = 0

# prepare plotting window
plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_ylim(0, SCORE_MAX + 5)
ax.set_xlim(0, 6)  # we'll plot up to 6 IDs
bar_container = None

print("Starting main loop. Press 'q' to quit.")
start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame, exiting.")
        break

# ✅ Ensure frame is writable
    frame = frame.copy()

    frame_count += 1
    orig = frame.copy()

    # run detection (model expects RGB)
    results = model(frame[..., ::-1], size=640)  # BGR to RGB
    detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

    persons = []
    objs = []  # list of tuples (class_name, bbox)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        name = COCO_NAMES[cls]
        if name in wanted_classes:
            bbox = (int(x1), int(y1), int(x2), int(y2))
            objs.append((name, bbox))
            if name == 'person':
                persons.append(bbox)

    # update tracker with person bboxes
    objects = ct.update(persons)  # dict id -> centroid
    # build mapping from id -> bbox by choosing nearest bbox to centroid
    id_to_bbox = {}
    # compute centroids for persons list
    person_centroids = []
    for b in persons:
        cx = int((b[0] + b[2]) / 2.0)
        cy = int((b[1] + b[3]) / 2.0)
        person_centroids.append((cx, cy, b))

    # for each tracked object, find nearest bbox
    for oid, centroid in objects.items():
        best = None
        best_dist = 1e9
        for (cx, cy, b) in person_centroids:
            d = (centroid[0] - cx) ** 2 + (centroid[1] - cy) ** 2
            if d < best_dist:
                best_dist = d
                best = b
        if best is not None:
            id_to_bbox[oid] = best

    # for each person id, check overlapping phone/book/laptop
    object_bboxes_by_class = {}
    for name, bbox in objs:
        if name != 'person':
            object_bboxes_by_class.setdefault(name, []).append(bbox)

    current_time = time.time()
    for oid, bbox in id_to_bbox.items():
        # default attentive
        state = 'attentive'
        # if any phone/book/laptop overlaps above a small threshold --> distracted
        for clsname in ('cell phone', 'book', 'laptop'):
            for other_bbox in object_bboxes_by_class.get(clsname, []):
                # check IoU between person bbox and object bbox
                overlap = iou(bbox, other_bbox)
                # threshold can be tuned; 0.01 is tiny; use 0.02 or 0.03 for more strict
                if overlap > 0.02:
                    state = 'distracted'
                    break
            if state == 'distracted':
                break

        # update score
        if state == 'attentive':
            scores[oid] = min(SCORE_MAX, scores[oid] + SCORE_STEP_UP)
        else:
            scores[oid] = max(SCORE_MIN, scores[oid] - SCORE_STEP_DOWN)

        last_state[oid] = state

        # overlay box and info
        startX, startY, endX, endY = bbox
        color = (0, 255, 0) if state == 'attentive' else (0, 0, 255)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        label = f"ID {oid} {scores[oid]}"
        cv2.putText(frame, label, (startX, startY - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # log (timestamp, id, state, score)
        history.append({
            'timestamp': current_time,
            'frame': frame_count,
            'id': oid,
            'state': state,
            'score': scores[oid],
            'bbox': bbox
        })

    # show objects (phones/books) as small boxes
    for name, bbox_list in object_bboxes_by_class.items():
        for b in bbox_list:
            sx, sy, ex, ey = b
            cv2.rectangle(frame, (sx, sy), (ex, ey), (200, 200, 0), 1)
            cv2.putText(frame, name, (sx, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)

    # display instructions
    cv2.putText(frame, "Press 'q' to quit. Tracking IDs shown with attention score.", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # show frame
    cv2.imshow("Attention Demo", frame)

    # update plot every few frames
    if frame_count % PLOT_UPDATE_INTERVAL == 0:
        # pick top 6 ids (sorted) to plot
        ids = sorted(list(scores.keys()))[:6]
        y = [scores[i] for i in ids]
        ax.clear()
        ax.set_ylim(0, SCORE_MAX + 5)
        ax.set_title("Attention Scores (sample IDs)")
        ax.set_xlabel("ID")
        ax.set_ylabel("Score")
        ax.bar([str(i) for i in ids], y)
        fig.canvas.draw()
        plt.pause(0.001)

    # handle key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# save CSV
df = pd.DataFrame(history)
if len(df) > 0:
    df.to_csv(CSV_LOG, index=False)
    print(f"Saved log to {CSV_LOG}")

cap.release()
cv2.destroyAllWindows()
print("Done. Ran for", frame_count, "frames. Time:", time.time() - start_time)

plt.savefig("final_attention_scores.png")
print("Final graph saved as final_attention_scores.png")
plt.show()  # keeps it open after program ends
