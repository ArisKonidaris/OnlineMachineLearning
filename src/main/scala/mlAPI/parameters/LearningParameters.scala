package mlAPI.parameters

import mlAPI.parameters.utils.{Bucket, ParameterDescriptor, SerializableParameters}

/**
 * The base trait representing the learning hyper parameters of a machine learning algorithm.
 */
trait LearningParameters extends Serializable {

  var size: Int = _
  var bytes: Int = _
  var sizes: Array[Int] = _

  def getSize: Int = size

  def setSize(size: Int): Unit = this.size = size

  def getBytes: Int = bytes

  def setBytes(bytes: Int): Unit = this.bytes = bytes

  def getSizes: Array[Int] = sizes

  def setSizes(sizes: Array[Int]): Unit = this.sizes = sizes

  def toString: String

  def getCopy: LearningParameters

  def extractParams: (LearningParameters, Boolean) => SerializableParameters

  def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]]

  def generateParameters(pDesc: ParameterDescriptor): LearningParameters

  def sliceRequirements(range: Bucket): Unit = require(range.getEnd <= getSize - 1)

}