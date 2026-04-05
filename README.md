# 🌿 Plant Disease Detection — ESP32-CAM + TinyML

> Real-time plant leaf disease classification on a $5 microcontroller, no cloud required.

A complete end-to-end IoT pipeline: a custom CNN trained on the PlantVillage dataset, compressed to an **18 KB INT8 TFLite model**, deployed on an **ESP32-CAM (AI Thinker)** via Edge Impulse and Arduino IDE, with a live Python inference client and a Streamlit web app.

**Course Project — Internet of Things**  
B.Tech ECE (Rail Engineering), Gati Shakti Vishwavidyalaya  
Faculty Mentor: Dr. Sagar

**Team:**
| Roll No. | Name |
|----------|------|
| 23EC005 | Aditya Kumar Jha |
| 23EC039 | [Nikhil Rathaur](https://github.com/nike14rath) |
| 23EC040 | [Parth Sidhu](https://github.com/Parth-Sidhu-4) |
| 23EC046 | [Roshan Gupta](https://github.com/r0shan-git) |
| 23EC048 | Shashikant Sargam |

---

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Repository Structure](#repository-structure)
- [Setup & Usage](#setup--usage)
  - [1. Training the Model](#1-training-the-model)
  - [2. Running the Streamlit App](#2-running-the-streamlit-app)
  - [3. Real-Time ESP32-CAM Detection](#3-real-time-esp32-cam-detection)
  - [4. Flashing the ESP32-CAM](#4-flashing-the-esp32-cam)
- [Results](#results)
- [Supported Classes](#supported-classes)
- [References](#references)

---

## Overview

Plant diseases cause an estimated **20–40% of global crop losses** annually. Early diagnosis is critical but constrained by the availability of agricultural experts, especially in rural areas.

This project demonstrates that a fully functional **38-class plant disease classifier** can run on hardware costing under **USD 7**, entirely offline, with no cloud dependency. Key highlights:

- **18 KB** INT8 quantised TFLite model (vs. ~100 MB for ResNet-50)
- **~94% validation accuracy** on 17,572 images
- Runs on ESP32-CAM with **520 KB SRAM**
- Live MJPEG stream inference with HSV-based leaf detection
- Also accessible via a Streamlit web interface for image upload

---

## System Architecture

```
PlantVillage Dataset (87K images, 38 classes)
        │
        ▼
  Tiny CNN Training
  (TensorFlow / Keras, 64×64 input)
        │
        ▼
  Float32 Keras Model (.h5)
        │
        ▼
  INT8 Post-Training Quantisation
  (TFLite Converter + representative dataset)
        │
        ▼
  18 KB plant_disease_tiny_int8.tflite
        │
        ├──────────────────────────────┐
        ▼                              ▼
  Edge Impulse Export           Streamlit Web App
  (Arduino C library)           (app.py — image upload)
        │
        ▼
  Flash to ESP32-CAM
  (Arduino IDE, AI Thinker board)
        │
        ▼
  MJPEG Stream  ──►  Python Client (Final_Disease_Detection.py)
                      HSV leaf detection + TFLite inference
                      OpenCV overlay: label + confidence
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [PlantVillage (Mohanty et al., 2016)](https://github.com/spMohanty/PlantVillage-Dataset) |
| Augmented version | [Kaggle — New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) |
| Total images | ~87,000 RGB |
| Training split | 70,295 images |
| Validation split | 17,572 images |
| Classes | 38 (14 crop species, 26 diseases + healthy states) |
| Input size used | 64 × 64 pixels |

Download the dataset from Kaggle and place it as:
```
train/
valid/
```
in the project root before running training.

---

## Model

### Architecture

| Layer | Config | Output Shape |
|-------|--------|-------------|
| Input | 64×64×3 uint8 | 64×64×3 |
| Rescaling | ÷255 | 64×64×3 |
| Conv2D | 8 filters, 3×3, ReLU | 62×62×8 |
| MaxPooling2D | 2×2 | 31×31×8 |
| Conv2D | 16 filters, 3×3, ReLU | 29×29×16 |
| MaxPooling2D | 2×2 | 14×14×16 |
| Conv2D | 32 filters, 3×3, ReLU | 12×12×32 |
| MaxPooling2D | 2×2 | 6×6×32 |
| Flatten | — | 1152 |
| Dense | 64 units, ReLU | 64 |
| Dropout | 0.3 | 64 |
| Dense | 38 units, Softmax | 38 |

### Size Comparison

| Format | Size | Compression |
|--------|------|-------------|
| Float32 Keras (.h5) | ~72 KB | 1× |
| Float32 TFLite | ~48 KB | 1.5× |
| **INT8 TFLite** | **18 KB** | **4×** |

The pre-trained INT8 model is included at:
```
pretrained_final_light_version/plant_disease_tiny_int8.tflite
```

---

## Repository Structure

```
Plant-Disease-Monitoring/
│
├── ESP32_model.ipynb                  # Model design, training & TFLite export
├── ESP32_testing.ipynb                # Model testing & validation
├── Final_Disease_Detection.py         # Real-time ESP32-CAM stream inference
├── app.py                             # Streamlit web application
│
├── pretrained_final_light_version/
│   └── plant_disease_tiny_int8.tflite # Pre-trained INT8 model (18 KB)
│
├── ei-nikhilrathaur-project-1-arduino-1_0_3.zip
│                                      # Edge Impulse Arduino library
│
└── README.md
```

---

## Setup & Usage

### Prerequisites

```bash
pip install tensorflow numpy opencv-python streamlit pillow
```

### 1. Training the Model

Open and run `ESP32_model.ipynb` in Jupyter or Google Colab.

The notebook will:
- Load the PlantVillage dataset from `train/` and `valid/`
- Train the tiny CNN for 25 epochs
- Save `plant_disease_model_small.h5`
- Run INT8 quantisation and export `plant_disease_tiny_int8.tflite`
- Save `training_history.json` for accuracy/loss plots

### 2. Running the Streamlit App

For image-upload based inference (no ESP32 needed):

```bash
streamlit run app.py
```

Navigate to the **Disease Recognition** page, upload a leaf image (JPG/PNG), and click **Predict**.

### 3. Real-Time ESP32-CAM Detection

Once your ESP32-CAM is running and connected to Wi-Fi, update the stream URL in `Final_Disease_Detection.py`:

```python
stream_url = "http://<YOUR_ESP32_IP>:81/stream"
```

Then run:

```bash
python Final_Disease_Detection.py
```

A live window will open showing the camera feed with green bounding boxes around detected leaf regions and the predicted disease label + confidence overlaid. Press **ESC** to exit.

### 4. Flashing the ESP32-CAM

1. Extract `ei-nikhilrathaur-project-1-arduino-1_0_3.zip` and install the `Nikhilrathaur-project-1_inferencing` library in Arduino IDE (**Sketch → Include Library → Add .ZIP Library**).
2. Open the ESP32-CAM sketch (CameraWebServer or your custom sketch using the Edge Impulse library).
3. In Arduino IDE, set:
   - **Board:** AI Thinker ESP32-CAM
   - **Upload Speed:** 115200
   - **Flash Mode:** QIO
   - **Flash Frequency:** 80 MHz
4. Pull **GPIO0 to GND**, connect via USB-UART adapter, upload.
5. Release GPIO0 and press reset. The device IP will appear in the Serial Monitor.

---

## Results

| Metric | Value |
|--------|-------|
| Validation accuracy (Float32) | ~94% |
| Validation accuracy (INT8) | ~93–94% |
| Accuracy drop from quantisation | <1% |
| Model size (INT8 TFLite) | **18 KB** |
| Inference time on ESP32 (est.) | 50–200 ms |
| Classes supported | 38 |

---

## Supported Classes

<details>
<summary>Click to expand all 38 classes</summary>

| # | Class |
|---|-------|
| 1 | Apple — Apple Scab |
| 2 | Apple — Black Rot |
| 3 | Apple — Cedar Apple Rust |
| 4 | Apple — Healthy |
| 5 | Blueberry — Healthy |
| 6 | Cherry — Powdery Mildew |
| 7 | Cherry — Healthy |
| 8 | Corn — Cercospora Leaf Spot / Gray Leaf Spot |
| 9 | Corn — Common Rust |
| 10 | Corn — Northern Leaf Blight |
| 11 | Corn — Healthy |
| 12 | Grape — Black Rot |
| 13 | Grape — Esca (Black Measles) |
| 14 | Grape — Leaf Blight (Isariopsis Leaf Spot) |
| 15 | Grape — Healthy |
| 16 | Orange — Huanglongbing (Citrus Greening) |
| 17 | Peach — Bacterial Spot |
| 18 | Peach — Healthy |
| 19 | Pepper (Bell) — Bacterial Spot |
| 20 | Pepper (Bell) — Healthy |
| 21 | Potato — Early Blight |
| 22 | Potato — Late Blight |
| 23 | Potato — Healthy |
| 24 | Raspberry — Healthy |
| 25 | Soybean — Healthy |
| 26 | Squash — Powdery Mildew |
| 27 | Strawberry — Leaf Scorch |
| 28 | Strawberry — Healthy |
| 29 | Tomato — Bacterial Spot |
| 30 | Tomato — Early Blight |
| 31 | Tomato — Late Blight |
| 32 | Tomato — Leaf Mold |
| 33 | Tomato — Septoria Leaf Spot |
| 34 | Tomato — Spider Mites (Two-spotted) |
| 35 | Tomato — Target Spot |
| 36 | Tomato — Yellow Leaf Curl Virus |
| 37 | Tomato — Mosaic Virus |
| 38 | Tomato — Healthy |

</details>

---

## References

1. S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using Deep Learning for Image-Based Plant Disease Detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016. [doi:10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419)
2. Augmented PlantVillage Dataset on Kaggle: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
3. Original PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset
4. TensorFlow Lite: https://www.tensorflow.org/lite
5. Edge Impulse: https://edgeimpulse.com

---

*Project Repository: [github.com/nike14rath/Plant-Disease-Monitoring](https://github.com/nike14rath/Plant-Disease-Monitoring)*
