package mlAPI.math

import ControlAPI.DataInstance
import com.fasterxml.jackson.databind.ObjectMapper

/**
 * A trait representing a data point required for
 * machine learning tasks.
 */
trait Point extends Serializable {

  var numericVector: Vector
  var discreteVector: Vector
  var categoricalVector: Array[String]
  var dataInstance: String

  def validDiscreteVector: Boolean = {
    if (discreteVector.size == 0)
      true
    else
      (
        for (value: Double <- discreteVector.asInstanceOf[DenseVector].data) yield value == Math.floor(value)
        ).reduce((x,y) => x && y)
  }

  def setNumericVector(vector: Vector): Unit = this.numericVector = vector

  def getNumericVector: Vector = numericVector

  def setDiscreteVector(vector: Vector): Unit = this.discreteVector = vector

  def getDiscreteVector: Vector = discreteVector

  def setCategoricalVector(vector: Array[String]): Unit = this.categoricalVector = vector

  def getCategoricalVector: Array[String] = categoricalVector

  def getDataInstance: String = dataInstance

  def setDataInstance(dataInstance: String): Unit = this.dataInstance = dataInstance

  def numericToList: List[Double] = numericVector.toList

  def discreteToList: List[Double] = discreteVector.toList

  def categoricalToList: List[String] = categoricalVector.toList

  def marshal(): (Array[Int], Array[Double], Array[String], String) = {
    (
      Array[Int](numericVector.size, discreteVector.size),
      {
        val ar1: Array[Double] = numericVector.toList.toArray
        val ar2: Array[Double] = discreteVector.toList.toArray
        ar1 ++ ar2
      },
      categoricalVector,
      dataInstance
    )
  }

  def asUnlabeledPoint: UnlabeledPoint = UnlabeledPoint(numericVector, discreteVector, categoricalVector, dataInstance)

  def asTrainingPoint: TrainingPoint

  def asForecastingPoint: ForecastingPoint = ForecastingPoint(asUnlabeledPoint)

  def toDataInstance: DataInstance = {
    val mapper: ObjectMapper = new ObjectMapper()
    val instance = mapper.readValue(dataInstance, classOf[DataInstance])
    if (instance.isValid)
      instance
    else
      throw new RuntimeException("Cannot convert Point to DataInstance.")
  }

}
