#ifndef __NeuralNetwork__
#define __NeuralNetwork__


namespace tflite{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
}

struct TfLiteTensor;

class NeuralNetwork
{
    private:
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    tflite::MicroMutableOpResolver<10> *resolver;
    TfLiteTensor *input;
    TfLiteTensor *output;

    public:
    float *getInputBuffer();
    NeuralNetwork();
    float predict();

};



#endif