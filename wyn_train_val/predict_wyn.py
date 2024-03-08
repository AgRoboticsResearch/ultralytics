from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('F:/doctor/ultralytics/runs\segment/train/weights/best.pt')  # load a custom model  
#model = YOLO('F:\doctor\ultralytics\weight\yolov8n-seg.pt')  # load a custom model  #

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image



#官方demo
#yolo predict model=yolov8n-seg.pt source='https://ultralytics.com/images/bus.jpg'
#你自己的图像
#yolo predict model=yolov8n-seg.pt source='F:\doctor\strawberry\StrawDI_Db1\StrawDI_Db1/train\img/1.png'