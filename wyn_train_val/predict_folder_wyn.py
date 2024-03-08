from ultralytics import YOLO
import glob
import os
# Load a model
model = YOLO(r'F:\doctor/ultralytics/runs\segment/train/weights/best.pt')  # load an official model
 
# Predict with the model
imgpath = r'F:\doctor\StrawDI_Data/images/test'
imgs = glob.glob(os.path.join(imgpath,'*.png'))
for img in imgs:
    model.predict(img, save=True)
 