from ultralytics import YOLO
 
if __name__ == '__main__':
    
    model = YOLO('F:/doctor/ultralytics/seg_data_transfer/config/yolov8s-seg.yaml')  # load a pretrained model (recommended for training)
    # Train the model
    model.train(epochs=300,data='F:/doctor/ultralytics/seg_data_transfer/config/strawberry.yaml')