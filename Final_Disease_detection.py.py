import tensorflow as tf
import numpy as np
import cv2
import urllib.request

IMG_SIZE = 64

# ==============================
# Load TFLite Model
# ==============================
interpreter = tf.lite.Interpreter(
    model_path="pretrained_final_light_version/plant_disease_tiny_int8.tflite"
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==============================
# Class Names
# ==============================
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# ==============================
# Leaf Detection Function
# ==============================
def detect_leaf_region(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return x, y, w, h

# ==============================
# ESP32-CAM STREAM (FIXED)
# ==============================
stream_url = "http://10.167.111.74:81/stream"

stream = urllib.request.urlopen(stream_url)

bytes_data = b''
frame_count = 0

print("Starting real-time detection... Press ESC to exit")

# ==============================
# REAL-TIME LOOP
# ==============================
while True:
    bytes_data += stream.read(1024)

    a = bytes_data.find(b'\xff\xd8')  # JPEG start
    b = bytes_data.find(b'\xff\xd9')  # JPEG end

    if a != -1 and b != -1:
        jpg = bytes_data[a:b+2]
        bytes_data = bytes_data[b+2:]

        frame = cv2.imdecode(
            np.frombuffer(jpg, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            continue

        frame_count += 1

        # Resize for speed
        frame = cv2.resize(frame, (640, 480))

        # Process every 5th frame
        if frame_count % 5 == 0:

            bbox = detect_leaf_region(frame)

            if bbox is not None:
                x, y, w, h = bbox

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                leaf = frame[y:y + h, x:x + w]

                try:
                    leaf_resized = cv2.resize(leaf, (IMG_SIZE, IMG_SIZE))
                except:
                    continue

                input_data = np.expand_dims(leaf_resized, axis=0).astype(np.uint8)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])

                predicted_index = np.argmax(output_data)
                predicted_class = class_names[predicted_index]
                confidence = output_data[0][predicted_index] / 255.0

                label = f"{predicted_class} ({confidence:.2f})"

                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        cv2.imshow("ESP32 Plant Disease Detection", frame)

        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()