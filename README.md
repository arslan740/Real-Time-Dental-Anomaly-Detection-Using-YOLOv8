# YOLOv8 for Dental Caries Detection

This project implements a YOLOv8-based model to detect dental anomalies such as cavities, caries, and cracks. The objective is to assist dental professionals by automating anomaly detection through deep learning and computer vision techniques.

---

## Features

- Real-time dental anomaly detection using the YOLOv8 model.
- Detection classes: **tooth**, **cavity**, **caries**, and **crack**.
- End-to-end workflow from data preparation to model inference.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Install necessary dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Clone the Repository

```bash
git clone https://github.com/your-username/yolov8-dental-detection.git
cd yolov8-dental-detection
```

### Install YOLOv8

Install the official YOLOv8 library for object detection:
```bash
pip install ultralytics
```

---

## Usage

### 1. Data Preparation

- Prepare a dataset containing annotated images of dental anomalies.
- The dataset should follow the YOLO format:
  - Images in the `images` directory.
  - Corresponding label files in the `labels` directory.

### 2. Training the Model

Train the YOLOv8 model with your prepared dataset:
```bash
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data='path_to_your_dataset.yaml', epochs=100, imgsz=640)
```

### 3. Inference

Perform inference on new dental images:
```bash
# Perform inference
results = model.predict(source='path_to_images_or_video', save=True)
```

### 4. Evaluation

Evaluate the model's performance using metrics like mAP:
```bash
metrics = model.val()
print(metrics)
```

---

## Results

The trained model demonstrates accurate detection of the specified classes:
- **Detection Classes**: Tooth, Cavity, Caries, Crack
- High precision and recall on dental datasets.

---

## File Structure

```
.
|-- data/
|   |-- images/       # Training and testing images
|   |-- labels/       # Annotations in YOLO format
|-- models/           # Saved YOLOv8 model weights
|-- notebook.ipynb    # Jupyter notebook with implementation
|-- requirements.txt  # Dependencies
|-- scripts/
|   |-- train.py      # Training script
|   |-- infer.py      # Inference script
```

---

## Acknowledgments

- Built with [YOLOv8](https://github.com/ultralytics/ultralytics).
- Thanks to the dental professionals who provided data and guidance.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
