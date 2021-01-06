package mlAPI.parameters

import ControlAPI.CountableSerial

/**
 * The base trait representing the learning hyper parameters of a machine learning algorithm.
 */
trait LearningParameters extends Serializable {

  var size: Int = _
  var bytes: Int = _

  def getSize: Int = size

  def getBytes: Int = bytes

  def setSize(size: Int): Unit = this.size = size

  def setBytes(bytes: Int): Unit = this.bytes = bytes

  def toString: String

  def getCopy: LearningParameters

  def generateSerializedParams: (LearningParameters, Array[_]) => SerializedParameters

  def generateParameters(pDesc: ParameterDescriptor): LearningParameters

  def sliceRequirements(range: Bucket): Unit = require(range.getEnd <= getSize - 1)

}