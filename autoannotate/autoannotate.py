# src: https://docs.ultralytics.com/models/sam/

from ultralytics.yolo.data.annotator import auto_annotate
import os

SCRIPT_PATH=os.path.dirname(os.path.realpath(__file__))

det_model = SCRIPT_PATH + "/../model/best.pt"
data = SCRIPT_PATH + "/data_to_be_annotated"

#auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model='sam_b.pt')
auto_annotate(data=data, det_model=det_model, sam_model='sam_l.pt')