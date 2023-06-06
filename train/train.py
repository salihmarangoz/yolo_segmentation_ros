from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8x-seg.pt')  # load a pretrained model (recommended for training)
#model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
#model.train(data='bup20-seg.yaml', epochs=20, imgsz=640, val=True, single_cls=True, batch=8)
model.train(data='bup20-seg.yaml', epochs=20, imgsz=640, val=True, batch=8)