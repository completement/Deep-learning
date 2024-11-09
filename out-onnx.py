from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model = YOLO('best.pt')  

model.export(
    format="onnx",    
    simplify=True    
)