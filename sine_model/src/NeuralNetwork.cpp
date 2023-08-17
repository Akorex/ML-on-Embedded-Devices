#include "NeuralNetwork.h"
#include "sine_model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "TensorFlowLite_ESP32.h"

const int kArenaSize = 5 * 1024;

NeuralNetwork::NeuralNetwork(){
    error_reporter = new tflite::MicroErrorReporter();
    model = tflite::GetModel(sine_model_tflite);
    if (model -> version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided's Schema version %d not equal to
        supported version %d", model-> version(), TFLITE_SCHEMA_VERSION);

        return;
    }

    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver-> AddFullyConnected();
    resolver-> AddMul();
    resolver-> AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

    tensor_arena = (uint8_t *)malloc(kArenaSize);

    if (!tensor_arena){
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // build the interpreter

    interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena, kArenaSize, error_reporter);
}