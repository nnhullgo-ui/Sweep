# -*- coding: utf-8 -*-
"""
Program 1 : The Brain

Input: Watches the drone footage and detects litter
Output: Detections saved to detections.json for ClearPath app
"""

from ultralytics import YOLO
import json
from datetime import datetime

# ---- YOUR DRONE VIDEO PATH (change this each flight) ----
image_path = "C:/Users/naila/Downloads/user/litter_test2.jpeg" # ---- This is where i input the footage from drone 

# ---- LOAD YOUR TRAINED MODEL ----
model = YOLO(r"C:\Users\naila\anaconda3\Lib\site-packages\ultralytics\data\runs\detect\litter_model-2\weights\best.pt")

# ---- RUN DETECTION ----
results = model(image_path, save=True) # can change this to video or image 

# ---- SAVE TO JSON ----
detections = []
for result in results:
    for box in result.boxes:
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 2),
            "timestamp": datetime.now().isoformat(),
            "lat": 43.0481,
            "lon": -76.1474
        })

with open("C:/Users/naila/detections.json", "w") as f:
    json.dump(detections, f, indent=2)

print(f"Done! {len(detections)} detections saved!")