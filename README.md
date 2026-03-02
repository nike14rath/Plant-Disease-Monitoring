## ESP32 Plant Disease Detection Model

A **memory-efficient plant disease classification model** has been trained with an accuracy of **~94%**.  
The model is optimized for **TinyML deployment** and can run directly on an **ESP32 microcontroller**.

The final compiled model size is approximately **108 KB**, making it suitable for embedded systems with limited memory.

---

## Model Files

All model-related files are located in the following folder:



This folder contains:
`pretrained_final_light_model`

- **training_history.json** → Training history of the model
- **trained_model.h5 / .keras** → Trained TensorFlow model
- **plant_disease_tiny_int8.tflite** → Quantized TensorFlow Lite model
- Other artifacts generated during the training process

---

## Running the Model on ESP32

To run the model on an ESP32 device:

1. Use the generated **`model.h`** file (converted from the quantized `.tflite` model).
2. Open the **`plant_disease_detector.ino`** file in the **Arduino IDE**.
3. Place the **`model.h`** file in the same project directory.
4. Upload the code to the **ESP32 board**.

Once uploaded, the ESP32 will load the embedded model and perform **plant disease predictions** using **TensorFlow Lite for Microcontrollers**.

---

## Key Features

- Lightweight **TinyML model**
- **INT8 quantization** for reduced memory usage
- Compatible with **ESP32 microcontrollers**
- Designed for **edge AI and IoT applications**