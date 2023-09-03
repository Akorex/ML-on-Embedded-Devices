#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
//#include "tensorflow/lite/version.h"

// our model
#include "sine_model.h"

#define DEBUG 1

// some settings
constexpr int led_pin = 2;
constexpr float pi = 3.14159265;
constexpr float freq = 0.5;
constexpr float period = (1/freq) * (1000000);


// TFLite globals

namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;

  // tensor arena
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}



void setup() {

    // Wait for Serial to connect
#if DEBUG
  while(!Serial);
#endif

  // put your setup code here, to run once:
  // set up error reporter for debugging
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // map the model into a useable data structure
  model = tflite::GetModel(sine_model);

  // ops resolver
  tflite::AllOpsResolver resolver;

  //tflite::MicroAllocator* allocator = tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize);
  //tflite::MicroInterpreter interpreter(model, resolver, allocator);


  // build an interpreter
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);

  // allocate tensors
  
  interpreter.AllocateTensors();

  // assign model input and output buffers
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);

#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(input->type);
#endif


}

void loop() {
  // put your main code here, to run repeatedly:

}
