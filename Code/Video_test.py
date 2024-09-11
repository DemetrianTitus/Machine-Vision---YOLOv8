import os
from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Define a color map for the classes using class IDs
# Each class ID is associated with a specific color used for bounding boxes in the output video
color_map = {
    0: (255, 0, 0),    # Red
    1: (0, 255, 0),    # Green
    2: (0, 0, 255),    # Blue
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
    5: (0, 255, 255),  # Yellow
    6: (128, 0, 128),  # Purple
    7: (255, 128, 0),  # Orange
    8: (128, 255, 0),  # Lime
    9: (0, 128, 255),  # Sky Blue
    10: (128, 0, 255), # Violet
    11: (255, 0, 128), # Pink
    12: (0, 255, 128), # Spring Green
    13: (128, 128, 0), # Olive
    14: (128, 0, 0),   # Maroon
    15: (0, 128, 0)    # Dark Green
}

# Set paths for input and output videos
# The input video is processed, and the output is saved with a suffix '_out' added to the filename
video_path = os.path.join('.', 'videos', 'Test_video.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

# Initialize video capture and writer objects
# Capture frames from the input video and set up the output video with the same properties
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Load the YOLO model and move it to the appropriate device (GPU if available, otherwise CPU)
# The model is used for object detection in the video frames
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Set the minimum confidence threshold for detecting objects
# Only objects with a confidence score above this threshold will be considered
threshold = 0.3

# Main loop to process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame from the video
    if not ret:
        break  # Exit loop if no frame is returned (end of video)

    # Prepare the frame for YOLO model inference
    # Convert frame to RGB format and resize it to fit the model's input size requirements
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (640, 640))
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    # Perform inference to detect objects in the frame
    results = model(frame_tensor)[0]

    # Process the detection results and draw bounding boxes and labels on the frame
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:  # Check if the detection is above the confidence threshold
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"
            color = color_map.get(int(class_id), (0, 255, 0))  # Default to green if class color is not found
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (int(x1 * width / 640), int(y1 * height / 640)), 
                          (int(x2 * width / 640), int(y2 * height / 640)), color, 2)
            cv2.putText(frame, label, 
                        (int(x1 * width / 640), int(y1 * height / 640 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Write the processed frame to the output video and display it
    out.write(frame)
    cv2.imshow('Frame', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources after processing is complete
cap.release()  # Release the video capture object
out.release()  # Release the video writer object
cv2.destroyAllWindows()  # Close all OpenCV windows
