from ultralytics import YOLO

def train():
    print("model is being trained...")
    # Load YOLO model
    model = YOLO('yolov8m.pt')  # You can use yolov8s.pt, yolov8m.pt, etc.

    # Train the model on CPU
    model.train(data='C:/Users/TUF/Desktop/Irene/ML/Project_test_change1_only_183/config.yaml', epochs = 100, patience = 15, device='0')
    
if __name__ == '__main__':
    train()



    