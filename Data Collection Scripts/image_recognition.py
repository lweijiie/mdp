import cv2
import torch
import requests
import numpy as np
from ultralytics import YOLO

# Load YOLOv5 model (pre-trained on COCO dataset)
model = YOLO("beste200v5.pt")

# URL of the Raspberry Pi camera feed (Update with your Pi’s IP)
RPI_CAMERA_URL = "http://192.168.19.1:5000/video_feed"

def get_frame():
    """ Fetch the latest frame from Raspberry Pi Camera """
    response = requests.get(RPI_CAMERA_URL, stream=True)
    bytes_data = b""
    for chunk in response.iter_content(chunk_size=1024):
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # Start of JPEG
        b = bytes_data.find(b'\xff\xd9')  # End of JPEG
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
    return None

while True:
    test_img = np.zeros((640, 480, 3), dtype=np.uint8)
    test_results = model(test_img)
    print("YOLOv5 Model Loaded Successfully!")
    frame = get_frame()
    if frame is None:
        continue
    
    # Run YOLOv5 inference
    results = model(frame)

    # Extract first result and plot detections
    frame_with_predictions = np.squeeze(results[0].plot())  # Fix: Use `.plot()` instead of `.render()`

    # Display results
    cv2.imshow("YOLOv5 Detection", frame_with_predictions)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()