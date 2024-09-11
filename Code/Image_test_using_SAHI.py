import os
import cv2
import torch
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sklearn.cluster import DBSCAN
import yaml

# Function to read class names from config file
def read_class_names(config_path):
    # Read the configuration file and extract class names
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return {int(k): v for k, v in config['names'].items()}

# Define a color map for the classes using class IDs
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

# Path setup section: Define paths for config file, input directory, and output directory
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
class_names = read_class_names(config_path)

input_dir = os.path.join('.', 'Pictures')
output_dir = os.path.join('.', 'Pictures_out')
os.makedirs(output_dir, exist_ok=True)

# Model loading section: Load the YOLO model from the specified checkpoint
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')  # Ensure correct path to best.pt
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.7,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Function to merge overlapping bounding boxes
def merge_boxes(boxes):
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    clustering = DBSCAN(eps=50, min_samples=1).fit(boxes[:, :2])
    
    merged_boxes = []
    for cluster in np.unique(clustering.labels_):
        cluster_boxes = boxes[clustering.labels_ == cluster]
        x1 = np.min(cluster_boxes[:, 0])
        y1 = np.min(cluster_boxes[:, 1])
        x2 = np.max(cluster_boxes[:, 2])
        y2 = np.max(cluster_boxes[:, 3])
        score = np.mean(cluster_boxes[:, 4])
        class_id = cluster_boxes[0, 5]
        merged_boxes.append([x1, y1, x2, y2, score, class_id])
    
    return merged_boxes

# Function to calculate Intersection over Union (IoU) between two boxes
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    iou_value = inter_area / min(box1_area, box2_area)
    return iou_value

# Function to filter overlapping boxes based on IoU and confidence score
def filter_boxes(boxes):
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        keep = True
        for j, box2 in enumerate(boxes):
            if i != j and iou(box1[:4], box2[:4]) > 0.7:
                if box1[4] < box2[4]:
                    keep = False
                    break
        if keep:
            filtered_boxes.append(box1)
    return filtered_boxes

# Loop through all test images
for i in range(12, 13):
    image_path = os.path.join(input_dir, f'test{i}.jpg')

    # Read the image
    image = read_image(image_path)

    # Perform SAHI inference
    results = get_sliced_prediction(
        image=image,
        detection_model=detection_model,
        slice_height=480,
        slice_width=480,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5
    )

    # Collect all detections
    detections = []
    for detection in results.object_prediction_list:
        x1, y1, x2, y2 = detection.bbox.minx, detection.bbox.miny, detection.bbox.maxx, detection.bbox.maxy
        score = detection.score.value
        class_id = detection.category.id
        if score > 0.3:
            detections.append([x1, y1, x2, y2, score, class_id])
    
    # Merge close bounding boxes
    merged_boxes = merge_boxes(detections)
    
    # Filter boxes based on IoU and detection precision
    filtered_boxes = filter_boxes(merged_boxes)

    # Draw filtered bounding boxes
    for (x1, y1, x2, y2, score, class_id) in filtered_boxes:
        class_name = class_names.get(class_id, str(class_id))
        label = f"{class_name} {score:.2f}"
        color = color_map.get(class_id, (0, 255, 0))  # Default to green if class is not in the color map
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save the processed image
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Display the resulting frame (commented out)
    # cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Frame', 1920, 1080)  # Set the window size to 1600x1200
    # cv2.imshow('Frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # # Wait for 'q' key to close the window (commented out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
