import cv2
from ultralytics import YOLO
import torch
from mss import mss
import numpy as np

import os
SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))

images = ["1.png", 
          "2.png",
          "3.png",
          "4.png",
          "5.png"]

# Load model
model = YOLO(SCRIPT_PATH + "/../model/best.pt")

# a dirty hack to replace "pepper" with "orange"
for k,v in model.names.items():
  if v == "orange":
    model.names[k] = "pepper"
    break

cv_window_str = "YOLOv8. PRESS Q TO QUIT. PRESS OTHER KEYS TO CONTINUE."
cv2.namedWindow(cv_window_str, cv2.WINDOW_NORMAL)
for image_name in images:
    file_path = SCRIPT_PATH + "/../example_data/" + image_name
    print(file_path)
    frame = cv2.imread(file_path)

    # Run inference on the frame
    results = model.predict(frame, retina_masks=True, conf=0.75, iou=0.5)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow(cv_window_str, annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()