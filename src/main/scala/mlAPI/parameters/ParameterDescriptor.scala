package mlAPI.parameters

import ControlAPI.CountableSerial
import mlAPI.math.{DenseVector, Vector}

/**
 * A Serializable POJO case class for sending the parameters over the Network.
 * This class contains all the necessary information to reconstruct the parameters
 * on the receiver side.
 *
 * @param paramSizes    The vector sizes of the parameters.
 * @param params        The vector parameters.
 * @param bucket        The range bucket of a vectorized model.
 * @param dataStructure A data structure of a model.
 * @param miscellaneous Extra serializable misc.
 * @param fitted        The number of data points fitted.
 */
case class ParameterDescriptor(var paramSizes: Array[Int],
                               var params: Vector,
                               var bucket: Bucket,
                               var dataStructure: CountableSerial,
                               var miscellaneous: CountableSerial,
                               var fitted: Long)
  extends java.io.Serializable {

  def this() = this(Array(1), DenseVector(Array(0.0)), new Bucket(), null, null, 0)

  // =================================== Getters ===================================================

  def getParamSizes: Array[Int] = paramSizes

  def getParams: Vector = params

  def getBucket: Bucket = bucket

  def getDataStructure: CountableSerial = dataStructure

  def getMiscellaneous: CountableSerial = miscellaneous

  def getFitted: Long = fitted

  // =================================== Setters ===================================================

  def setParamSizes(paramSizes: Array[Int]): Unit = this.paramSizes = paramSizes

  def setParams(params: Vector): Unit = this.params = params

  def setBucket(bucket: Bucket): Unit = this.bucket = bucket

  def setDataStructure(dataStructure: CountableSerial): Unit = this.dataStructure = dataStructure

  def setMiscellaneous(miscellaneous: CountableSerial): Unit = this.miscellaneous = miscellaneous

  def setFitted(fitted: Long): Unit = this.fitted = fitted

  def getSize: Int = {
    { if (paramSizes != null) 4 * paramSizes.length else 0 } +
      { if (params != null) params.getSize else 0 } +
      { if (bucket != null) bucket.getSize else 0 } +
      { if (dataStructure != null) dataStructure.getSize else 0 } +
      { if (miscellaneous != null) miscellaneous.getSize else 0 } +
      { if (fitted != null) 8 else 0 }
  }

}

