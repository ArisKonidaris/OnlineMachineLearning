package mlAPI.parameters

import mlAPI.math.{DenseVector, Vector}

/** A Serializable POJO case class for sending the parameters over the Network.
  * This class contains all the necessary information to reconstruct the parameters
  * on the receiver side.
  *
  */
case class ParameterDescriptor(var paramSizes: Array[Int],
                               var params: Vector,
                               var bucket: Bucket,
                               var fitted: Long)
  extends java.io.Serializable {

  def this() = this(Array(1), DenseVector(Array(0.0)), new Bucket(), 0)

  // =================================== Getters ===================================================

  def getParamSizes: Array[Int] = paramSizes

  def getParams: Vector = params

  def getFitted: Long = fitted

  def getBucket: Bucket = bucket

  // =================================== Setters ===================================================

  def setParamSizes(paramSizes: Array[Int]): Unit = this.paramSizes = paramSizes

  def setParams(params: Vector): Unit = this.params = params

  def setFitted(fitted: Long): Unit = this.fitted = fitted

  def setBucket(bucket: Bucket): Unit = this.bucket = bucket

}

