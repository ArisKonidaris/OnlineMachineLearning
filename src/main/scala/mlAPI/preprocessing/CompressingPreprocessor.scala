package mlAPI.preprocessing

/**
 * A trait for preprocessors that compress the incoming data point. The number of output features
 * should be less than the number of features of the incoming data point.
 */
trait CompressingPreprocessor extends Preprocessor
