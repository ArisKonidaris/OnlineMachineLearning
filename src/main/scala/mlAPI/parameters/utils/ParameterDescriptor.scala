package mlAPI.parameters.utils

import ControlAPI.CountableSerial
import mlAPI.math.Vector
import mlAPI.protocols.LongWrapper

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
                               var miscellaneous: Array[CountableSerial],
                               var fitted: LongWrapper)
  extends java.io.Serializable {

  def this() = this(null, null, null, null, null, null)

  // =================================== Getters ===================================================

  def getParamSizes: Array[Int] = paramSizes

  def getParams: Vector = params

  def getBucket: Bucket = bucket

  def getDataStructure: CountableSerial = dataStructure

  def getMiscellaneous: Array[CountableSerial] = miscellaneous

  def getFitted: LongWrapper = fitted

  // =================================== Setters ===================================================

  def setParamSizes(paramSizes: Array[Int]): Unit = this.paramSizes = paramSizes

  def setParams(params: Vector): Unit = this.params = params

  def setBucket(bucket: Bucket): Unit = this.bucket = bucket

  def setDataStructure(dataStructure: CountableSerial): Unit = this.dataStructure = dataStructure

  def setMiscellaneous(miscellaneous: Array[CountableSerial]): Unit = this.miscellaneous = miscellaneous

  def setFitted(fitted: LongWrapper): Unit = this.fitted = fitted

  def getSize: Int = {
    {
      if (paramSizes != null) 4 * paramSizes.length else 0
    } + {
      if (params != null) params.getSize else 0
    } + {
      if (bucket != null) bucket.getSize else 0
    } + {
      if (dataStructure != null) dataStructure.getSize else 0
    } + {
      if (miscellaneous != null) (for (misc <- miscellaneous) yield misc.getSize).sum else 0
    } + {
      if (fitted != null) fitted.getSize else 0
    }
  }

}
