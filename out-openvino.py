from ultralytics import YOLO
# Load a YOLOv11n PyTorch model
model = YOLO('best.pt')

# Export the model
model.export(format="openvino") 






