from ultralytics import YOLO

# See: https://docs.ultralytics.com/tasks/segment/#models

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8s-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)


#model.train(data='sweet-pepper-seg.yaml', epochs=20, imgsz=640, val=True, single_cls=True, batch=8)
model.train(data='sweet-pepper-seg.yaml', epochs=60, imgsz=720, val=True, batch=48, single_cls=True)