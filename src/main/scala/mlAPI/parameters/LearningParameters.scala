package mlAPI.parameters

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

  def equals(obj: Any): Boolean

  def toString: String

  def getCopy: LearningParameters

  def generateSerializedParams: (LearningParameters, Array[_]) => java.io.Serializable

  def generateParameters(pDesc: ParameterDescriptor): LearningParameters


}