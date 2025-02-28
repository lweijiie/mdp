import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('../Weights/bestv2.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit the webcam feed.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Run inference on the frame (color frame)
    results = model.predict(source=frame, verbose=False)

    # Display the annotated frame
    for result in results:
        annotated_frame = result.plot()  # Render bounding boxes
        cv2.imshow('YOLOv5 Webcam Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
