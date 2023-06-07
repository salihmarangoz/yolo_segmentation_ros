import cv2
from ultralytics import YOLO
import torch
from mss import mss
import numpy as np

import os
SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))

left = 200
top = 200
right = 1280 + left
lower = 720 + top
bbox = (left, top, right, lower)

# Load model
model = YOLO(SCRIPT_PATH + "/../model/best.pt")

# a dirty hack to replace "pepper" with "orange"
for k,v in model.names.items():
  if v == "orange":
    model.names[k] = "pepper"
    break

cv_window_str = "YOLOv8. PRESS Q TO QUIT. PRESS OTHER KEYS TO CONTINUE."
cv2.namedWindow(cv_window_str, cv2.WINDOW_NORMAL)
with mss() as sct:
    while True:
        frame = np.array( sct.grab(bbox) )[:,:,:3]
        print(frame.shape)

        # Run inference on the frame
        results = model.track(frame, retina_masks=True, conf=0.75, iou=0.5, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow(cv_window_str, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()