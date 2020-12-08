package mlAPI.parameters

import mlAPI.math.{DenseVector, Vector}
import mlAPI.utils.Sizable

import java.io.Serializable

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
                               var dataStructure: Serializable,
                               var miscellaneous: Serializable,
                               var fitted: Long)
  extends java.io.Serializable {

  def this() = this(Array(1), DenseVector(Array(0.0)), new Bucket(), null, null, 0)

  // =================================== Getters ===================================================

  def getParamSizes: Array[Int] = paramSizes

  def getParams: Vector = params

  def getBucket: Bucket = bucket

  def getDataStructure: Serializable = dataStructure

  def getMiscellaneous: Serializable = miscellaneous

  def getFitted: Long = fitted

  // =================================== Setters ===================================================

  def setParamSizes(paramSizes: Array[Int]): Unit = this.paramSizes = paramSizes

  def setParams(params: Vector): Unit = this.params = params

  def setBucket(bucket: Bucket): Unit = this.bucket = bucket

  def setDataStructure(dataStructure: Serializable): Unit = this.dataStructure = dataStructure

  def setMiscellaneous(miscellaneous: Serializable): Unit = this.miscellaneous = miscellaneous

  def setFitted(fitted: Long): Unit = this.fitted = fitted

}

