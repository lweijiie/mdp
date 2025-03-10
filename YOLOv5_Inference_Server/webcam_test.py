import cv2
from ultralytics import YOLO
import numpy as np

# Load two YOLO models
model1 = YOLO('../Weights/beste380.pt')  # First model
model2 = YOLO('../Weights/best_Aug.pt')  # Second model

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to exit the webcam feed.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Run inference on both models
    results1 = model1.predict(source=frame, verbose=False)
    results2 = model2.predict(source=frame, verbose=False)

    # Annotate frames
    annotated_frame1 = results1[0].plot()
    annotated_frame2 = results2[0].plot()

    # Resize images to the same size for side-by-side display
    height, width, _ = annotated_frame1.shape
    annotated_frame2 = cv2.resize(annotated_frame2, (width, height))

    # Combine both frames horizontally
    combined_frame = np.hstack((annotated_frame1, annotated_frame2))

    # Display side-by-side comparison
    cv2.imshow('YOLOv5 Model Comparison (Left: Model1 | Right: Model2)', combined_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
