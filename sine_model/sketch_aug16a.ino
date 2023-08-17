#include "TensorFlowLite.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"

#include "sine_model.h"

#define DEBUG 1

constexpr int led_pin = 2;
constexpr float pi = 3.14159265;                  // Some pi
constexpr float freq = 0.5;                       // Frequency (Hz) of sinewave
constexpr float period = (1 / freq) * (1000000);  // Period (microseconds)


// TFLite globals, used for compatibility with Arduino-style sketches
namespace{
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}



void setup() {

#if DEBUG
while (!Serial);
#endif

pinMode(led_pin, OUTPUT);

//  logging
static tflite::MicroErrorReporter micro_error_reporter;
error_reporter = &micro_error_reporter;

// map the model
model = tflite::GetModel(sine_model);
if (model-> version() != TFLITE_SCHEMA_VERSION){
  error_reporter -> Report("Model version does not match Schema");
  return 1;



// ops resolver
static tflite::MicroMutableOpResolver resolver;
resolver.AddBuiltin(
  tflite::BuiltinOperator_FULLY_CONNECTED,
  tflite::ops::micro::Register_FULLY_CONNECTED(),
  1, 3);


// build an interpreter to run the model
static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
interpreter = &static_interpreter;


// allocate tensor to the interpreter
interpreter.AllocateTensors();


// Assign model input output buffers 
model_input = interpreter->input(0);
model_output = interpreter->output(0);

#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif

}



}

void loop() {
  #if DEBUG
  unsigned long start_timestamp = micros();
  #endif

  unsigned long timestamp = micros();
  timestamp = timestamp % (unsigned long) period;

  
}
