namespace tflite {
    namespace ops{
        namespace micro{
            TfLiteRegistration* Register_DEPTHWISE_CONV_2D();
            TfLiteRegistration* Register_FULLY_CONNECTED();
            TfLiteRegistration* Register_SOFTMAX();

        }
    }
}

// setup logging
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;