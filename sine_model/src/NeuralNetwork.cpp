#include "NeuralNetwork.h"
#include "sine_model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

const int kArenaSize = 5 * 1024;

NeuralNetwork::NeuralNetwork(){
    error_reporter = 
}