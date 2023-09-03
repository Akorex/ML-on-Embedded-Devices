#include "NeuralNetwork.h"
#include "sine_model.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


NeuralNetwork::NeuralNetwork()
{
    // error reporter for debugging
    error_reporter = new tflite::MicroErrorReporter();

    // map the model and check the version
    model = tflite::GetModel(sine_model_tflite);

    if (model-> version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "Model provided's schema version %d not equal to supported version%d.", model-> version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // resolver
    resolver = new tflite::MicroMutableOpResolver<10>();
    resolver->AddFullyConnected();
    resolver->AddMul();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddReshape();
    resolver->AddQuantize();
    resolver->AddDequantize();

    // tensor arena
    const int tensor_arena_size = 2 * 1024;
    uint8_t tensor_arena[tensor_arena_size];

    if (!tensor_arena){
        TF_LITE_REPORT_ERROR(error_reporter, "Could not allocate arena");
        return;
    }

    // build the interpreter
    interpreter = new tflite::MicroInterpreter(model, *resolver, tensor_arena, tensor_arena_size, error_reporter);

    // allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk){
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return;
    }

    // inspect input tensor
    input = interpreter->input(0);
    output = interpreter->output(0);

}
