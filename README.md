# Object Detection in Cluttered Environments - Machine Vision

## Overview
This project provides a comprehensive guide on ***object detection in cluttered environments*** using *YOLOv8*. It demonstrates how to identify and classify objects in both still images and video streams. The techniques outlined can be applied to various fields, including surveillance, traffic monitoring, and other applications requiring object detection.

### Purpose
This project primarily demonstrates the capabilities of object detection in cluttered environments using YOLOv8. It serves as a comprehensive guide to correctly setting up data, training the model, and testing on images and videos. The aim is to provide a clear and practical guide for implementing object detection in various real-world scenarios.


## Features
- **Real-Time Object Detection:** Detection and classification of objects in scenes with high accuracy using YOLOv8.

- **Multi-Class Support:** Identifies a variety of objects, including cars, trucks, people, and more, based on your custom classes. 

- **Bounding Box Visualization:** Draws bounding boxes around detected objects with labels and confidence scores.

- **Configurable Detection Threshold:** Allows you to set a confidence threshold to filter out less certain detections.

- **GPU Acceleration:** Utilizes NVIDIA GPU acceleration for faster inference and processing times (if available).

- **Patch Analysis:** Provides detailed patch analysis for images using SAHI, improving detection quality on large-scale and complex scenes.

## Instalation

### Prerequisites
Ensure you have the following installed:
- **Python 3.9.13**: [Download Python](https://www.python.org/downloads/)

### Installation Steps
1. **Clone the Repository:**

   ```bash
   git clone https://github.com/DemetrianTitus/Machine-Vision---YOLOv8.git
2. **Navigate to the Project Directory and Create a Virtual Environment (Optional but recommended):**
   ```bash
   cd "Machine Vision - YOLOv8"
   ```
   ```
   python -m venv env
   ```

3. **Install the Required Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

This project consists of two primary components:

1. **Model Training**
2. **Model Testing**

### Model Training

To train a model, follow these steps:

1. **Data Preparation:** Gather and organize the necessary data as outlined in [Preparing_data.md](Preparing_data.md). Ensure that the data is sorted in the specified order and configure the `config.yaml` file with the desired classes.

    ***Note:*** Ensure to include the correct directory path in the configuration.

2. **Training the Model:** Use the `Model_training.py` script to train your model.

For additional details on object detection and YOLOv8 models, please refer to the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics.git).

If you need to resume training after stopping, use the `Continue_model_training.py` script, which supports checkpoints. Point to your model's checkpoint file, such as `runs/detect/train/weights/last.pt`, to continue training.

***Note:*** The `NVIDIA_GPU_monitoring.py` script is used to monitor your NVIDIA GPU in a separate terminal. The terminal content is refreshed every 10 seconds to avoid clutter and provide up-to-date GPU monitoring data.

### Model Testing

1. **Image Testing:** To test the model on images, use the `Image_test.py` or `Image_test_using_SAHI.py` script. For more information on the SAHI algorithm, refer to the [SAHI GitHub repository](https://github.com/obss/sahi.git).

2. **Video Testing:** To test the model on videos, use the `Video_test.py` script.

## Configuration

Proper configuration is crucial for successful model training and testing. Ensure that the following files and settings are correctly configured:

### 1. `config.yaml`

The `config.yaml` file contains important parameters for training your model. Verify and adjust the following settings as needed:

- **Classes:** List all the classes that the model should detect. Ensure that this list matches the classes in your dataset.
- **Paths:** Set the correct paths for training and validation data directories.
- **Hyperparameters:** Configure hyperparameters such as learning rate, batch size, and number of epochs according to your requirements. These settings apply to both the `Model_training.py` and `Continue_model_training.py` scripts.

### 2. Directory Structure

Ensure that the directory structure of your dataset follows the expected format:

- **Training Data:**
  - `Dataset/train/images/` (contains image files)
  - `Dataset/train/labels/` (contains corresponding label files)
- **Validation Data:**
  - `Dataset/val/images/` (contains image files)
  - `Dataset/val/labels/` (contains corresponding label files)

### 3. File Paths

Make sure all file paths specified in your scripts are correct. This includes paths to:
- Training and validation images and labels.
- Checkpoints for resuming training.
- Monitoring scripts, if applicable.

### 4. Additional Settings

Depending on your setup and requirements, additional configuration settings may include:
- **GPU Settings:** Ensure proper settings for GPU utilization, such as specifying GPU device IDs if using multiple GPUs.
- **Logging:** Configure logging options to monitor training progress and performance metrics.
- **Data Augmentation:** If applicable, adjust data augmentation parameters in your configuration files.

For further assistance and details, refer to the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics.git) and the [Preparing_data.md](Preparing_data.md) for guidance on data preparation.

# Model Training and Testing

After a detailed guide on collecting data, training a model, and testing it, a short presentation of models trained for two distinct environments: `indoor` and `urban` is provided.

## Indoor Environment

### Indoor Environment Dataset: Training and Validation Data

<div align="center">

| **Class**       | **Training Images** | **Validation Images** |
|-----------------|----------------------|------------------------|
| Ball            | 2,436                | 1,044                  |
| Book            | 2,800                | 1,200                  |
| Bottle          | 2,800                | 1,200                  |
| Doll            | 2,368                | 1,015                  |
| Headphones      | 745                  | 319                    |
| Mug             | 1,119                | 479                    |
| Teddy bear      | 701                  | 300                    |
| **Total**       | **12,969**           | **5,557**              |

</div>

### Indoor Environment Model Testing

The `YOLOv8s model` was trained on `640x640` image sizes with approximately `500 epochs`. The initial results showed only ***3 out of 9 objects*** detected.

To enhance detection, `SAHI algorithm` was applied, and a `clustering algorithm (DBSCAN)` was used to merge overlapping bounding boxes. This approach significantly improved detection quality.

![alt text](<Indoor enviroment/Example pictures/test_before_after.gif>)

With the `SAHI` framework applied, ***8 out of 9 objects*** were detected, with minimal accuracy of 70% for the Mug and 71% for the Teddy bear. Accuracy for other objects improved to 80% or above. Additional training with more complex models, larger image sizes, and more training images could further improve accuracy. Another example where the middle object was not detected is shown below.

![Indoor Environment Example 3](<Indoor enviroment/Example pictures/test7.jpg>)

Note that not all examples are "*perfect*". For instance, in the image below, the model failed to detect a book in the far-right corner but incorrectly identified a pillow as a book.

![Indoor Environment Example 4](<Indoor enviroment/Example pictures/test6.jpg>)

## Urban Environment

### Urban Environment Dataset: Training and Validation Data

<div align="center">

| **Class**       | **Training Images** | **Validation Images** |
|-----------------|----------------------|------------------------|
| Car             | 40,000               | 10,000                 |
| Truck           | 6,500                | 1,578                  |
| Train           | 8,000                | 2,081                  |
| Motorcycle      | 5,500                | 1,444                  |
| Bicycle         | 15,000               | 2,631                  |
| Traffic light   | 1,500                | 274                    |
| Traffic sign    | 2,400                | 417                    |
| Person          | 40,000               | 10,000                 |
| Street light    | 9,000                | 2,226                  |
| Van             | 4,500                | 878                    |
| **Total**       | **132,400**          | **31,529**             |

</div>

### Urban Environment Model Testing

Testing the model in an urban environment presented several challenges due to the complexity of the scene. The performance of the model was influenced significantly by the image resolution and the configuration of the YOLOv8 model used for training.

In the first example, the model successfully detects cars, traffic lights, traffic signs, and people. However, due to the use of `640x640` image size and the lightweight `YOLOv8n model`, the detection probabilities are not optimal, and the model struggles to maintain consistent tracking of objects. This suggests that there is room for improvement through the use of more robust models, higher-resolution images, and an increased volume of training data that is meticulously curated.

<div align="center">
    <img src="Urban enviroment/Model_training_results/GIF_1.gif" alt="Urban Environment Detection - Example 1">
</div>

In the second example, the model detects street lights, traffic lights, traffic signs, and cars. Similar to the first example, we observe challenges in detection consistency, further emphasizing the need for additional enhancements. These findings suggest that further training with more diverse and high-quality data, as well as adjustments to model complexity, could significantly enhance detection performance.

<div align="center">
    <img src="Urban enviroment/Model_training_results/GIF_2.gif" alt="Urban Environment Detection - Example 2">
</div>

---

**Note:** See additional examples inside the `Indoor environment` and `Urban environment` folders.


## License Information

### MIT License

This project is licensed under the MIT License. See the `LICENSE` file for details.



