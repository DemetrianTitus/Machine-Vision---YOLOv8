import os
from ultralytics import YOLO
import cv2

# Define paths section: Specify paths for the input image, output image, and model
IMAGE_PATH = r'C:\Users\User_name\...\test_image.jpg'
OUTPUT_IMAGE_PATH = r'C:\Users\User_name\...\test_image_out.jpg'
MODEL_PATH = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Model loading section: Load the YOLO model from the specified checkpoint
model = YOLO(MODEL_PATH)  # load a custom model

# Image loading section: Load the input image and check if it exists
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image file not found: {IMAGE_PATH}")

H, W, _ = image.shape

# Threshold setting section: Define the confidence threshold for predictions
threshold = 0.5

# Prediction section: Perform object detection on the input image
results = model(image)[0]

# Drawing bounding boxes section: Draw bounding boxes and labels on the detected objects
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Output section: Save and display the output image with bounding boxes
cv2.imwrite(OUTPUT_IMAGE_PATH, image)
cv2.imshow('Predicted Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
