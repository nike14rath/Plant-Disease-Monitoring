// this code needs to be pasted in the ESP32 Arduino IDE to run the plant disease detector model. It initializes the TensorFlow Lite Micro interpreter, allocates tensors, and runs inference on the input data. The results can be printed to the Serial Monitor for debugging purposes.

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

const int tensor_arena_size = 120 * 1024;
uint8_t tensor_arena[tensor_arena_size];

void setup() {

  Serial.begin(115200);

  const tflite::Model* model =
      tflite::GetModel(pretrained_final_light_version_plant_disease_tiny_int8_tflite);

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, tensor_arena_size);

  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);

  interpreter.Invoke();

  TfLiteTensor* output = interpreter.output(0);

  Serial.println("Prediction done");
}

void loop() {}