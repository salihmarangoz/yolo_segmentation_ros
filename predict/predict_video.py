import cv2
from ultralytics import YOLO
import torch
import os
SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))

SKIP_FRAMES = 5

# Load the YOLOv8 model
model_path = SCRIPT_PATH + "/../model/best.pt"
model = YOLO(model_path)

# a dirty hack to replace "pepper" with "orange"
for k,v in model.names.items():
  if v == "orange":
    model.names[k] = "pepper"
  if v == "person":
    model.names[k] = "pepper"

# Open the video file
video_path = SCRIPT_PATH + "/../example_data/video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
cv_window_str = "YOLOv8. PRESS Q TO QUIT"
cv2.namedWindow(cv_window_str, cv2.WINDOW_NORMAL)
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # Skip some frames to speed-up the video
    for i in range(SKIP_FRAMES):
        success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame, retina_masks=True, conf=0.75, iou=0.5, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow(cv_window_str, annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()