import cv2
import torch
import yolov5
import os
import sys

# Add a patch for PosixPath issue before loading the model
# This fixes the cross-platform compatibility problem
if sys.platform.startswith('win'):
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_yolo5_2024_10_18.pt')
model = yolov5.load(model_path)
model.conf = 0.6  # confidence threshold

# Check device and print it
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {'CUDA (GPU)' if device == 0 else 'CPU'}")
model.to(device)  # move model to device

print("Press 'q' to exit the webcam feed.")

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run prediction with YOLOv5
    results = model(frame)

    # Display results on webcam feed
    rendered_frame = results.render()[0]  # returns list of annotated images

    cv2.imshow('YOLOv5 Webcam Detection', rendered_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
