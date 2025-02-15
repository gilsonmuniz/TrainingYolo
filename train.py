from ultralytics import YOLO

# Load a model
model = YOLO('yolo11n-seg.yaml')  # build a new model from YAML
model = YOLO('yolo11n-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolo11n-seg.yaml').load('yolo11n.pt')  # build from YAML and transfer weights

# Train the model
results_training1_1 = model.train(data='/content/drive/MyDrive/DataSets/dataset/dataset.yaml', epochs=100, imgsz=640)
