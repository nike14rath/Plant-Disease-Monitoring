#include "esp_camera.h"
#include "board_config.h"
#include "model.h"

// EloquentTinyML library
// #include <EloquentTinyML.h>
#include <eloquent_tinyml.h>
// using namespace Eloquent::TinyML;

// Define input, output, and arena sizes
#define ARENA_SIZE 60 * 1024
#define INPUT_SIZE 64 * 64 * 3
#define OUTPUT_SIZE 38

// Eloquent::TinyML::TfLite<ARENA_SIZE, INPUT_SIZE, OUTPUT_SIZE> ml;
TfLite<ARENA_SIZE, INPUT_SIZE, OUTPUT_SIZE> ml;

// ================= CAMERA INITIALIZATION =================
void initCamera() {
  camera_config_t config;
  
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size = FRAMESIZE_96X96;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  
  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    while (1);
  }
}

// ================= IMAGE PREPROCESSING =================
// Convert RGB565 to RGB888 and resize from 96x96 to 64x64
void preprocess(camera_fb_t *fb, uint8_t *input) {
  int index = 0;
  int step = 96 / 64;  // 1.5, but we'll use integer logic
  
  for (int y = 0; y < 96 && index < INPUT_SIZE; y += step) {
    for (int x = 0; x < 96 && index < INPUT_SIZE; x += step) {
      int i = (y * 96 + x) * 2;
      
      // Convert RGB565 to RGB888
      uint16_t pixel = (fb->buf[i] << 8) | fb->buf[i + 1];
      
      uint8_t r = ((pixel >> 11) & 0x1F) << 3;
      uint8_t g = ((pixel >> 5) & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;
      
      input[index++] = r;
      input[index++] = g;
      input[index++] = b;
    }
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\nPlant Disease Monitoring System Starting...");
  
  // Initialize camera
  initCamera();
  Serial.println("Camera initialized successfully");
  
  // Load the model
  Serial.println("Loading TensorFlow Lite model...");
  if (!ml.begin(pretrained_final_light_version_plant_disease_tiny_int8_tflite)) {
    Serial.println("ERROR: Failed to load model!");
    Serial.println("Check if model array exists and arena size is sufficient");
    while (1);
  }
  
  Serial.println("Model loaded successfully!");
  Serial.println("Ready to classify plant diseases");
  delay(1000);
}

// ================= LOOP =================
void loop() {
  // Capture image
  camera_fb_t *fb = esp_camera_fb_get();
  
  if (!fb) {
    Serial.println("ERROR: Camera capture failed");
    delay(1000);
    return;
  }
  
  Serial.println("\n=== New Classification ===");
  
  // Prepare input buffer
  static uint8_t input[INPUT_SIZE];
  
  // Preprocess image
  preprocess(fb, input);
  
  // Run inference
  Serial.println("Running inference...");
  ml.predict(input);
  
  // Get predictions
  float *output = ml.getPredictions();
  
  // Find the class with highest probability
  int predicted_class = 0;
  float max_probability = output[0];
  
  Serial.println("\nTop predictions:");
  for (int i = 0; i < OUTPUT_SIZE; i++) {
    if (output[i] > max_probability) {
      max_probability = output[i];
      predicted_class = i;
    }
    
    // Print only top 5 predictions to keep serial output clean
    if (output[i] > 0.1) {
      Serial.print("  Class ");
      Serial.print(i);
      Serial.print(": ");
      Serial.println(output[i], 4);
    }
  }
  
  // Print final result
  Serial.println("\n=== RESULT ===");
  Serial.print("Predicted class index: ");
  Serial.print(predicted_class);
  Serial.print(" (Confidence: ");
  Serial.print(max_probability * 100, 2);
  Serial.println("%)");
  
  // Add your class names here if you have them
  // For example:
  // const char* disease_names[] = {"Healthy", "Apple Scab", "Black Rot", ...};
  // Serial.print("Disease: ");
  // Serial.println(disease_names[predicted_class]);
  
  // Return frame buffer
  esp_camera_fb_return(fb);
  
  // Wait before next capture
  delay(3000);
}