package mlAPI.preprocessing

import ControlAPI.PreprocessorPOJO
import mlAPI.math.{DenseVector, LabeledPoint, LearningPoint, UnlabeledPoint, Vector}
import breeze.linalg.{DenseVector => BreezeDenseVector}

import scala.collection.mutable.ListBuffer
import mlAPI.math.Breeze._
import scala.collection.JavaConverters._

import scala.collection.mutable

/**
 * A min max scaler.
 *
 * @param min The minimum possible value(s) of the training data points.
 * @param max The maximum possible value(s) of the training data points.
 */
case class MinMaxScaler(private var min: BreezeDenseVector[Double] = BreezeDenseVector[Double](0D),
                        private var max: BreezeDenseVector[Double] = BreezeDenseVector[Double](255D))
  extends Preprocessor {

  require(min.length == max.length)

  def this(min: Double, max: Double) = this(BreezeDenseVector[Double](min), BreezeDenseVector[Double](max))

  def getMin: BreezeDenseVector[Double] = min

  def getMax: BreezeDenseVector[Double] = max

  def setMin(min: BreezeDenseVector[Double]): Unit = this.min = min

  def setMax(max: BreezeDenseVector[Double]): Unit = this.max = max

  override def transform(point: LearningPoint): LearningPoint = {
    point match {
      case UnlabeledPoint(_, _, _, di) => UnlabeledPoint(scale(point), DenseVector(), Array[String](), di)
      case LabeledPoint(label, _, _, _, di) => LabeledPoint(label, scale(point), DenseVector(), Array[String](), di)
    }
  }

  private def scale(point: LearningPoint): Vector = {
    if (min.length == 1)
      ((point.getNumericVector.asBreeze - min(0)) / (max(0) - min(0))).fromBreeze
    else
      ((point.getNumericVector.asBreeze - min) / (max - min)).fromBreeze
  }

  override def transform(dataSet: ListBuffer[LearningPoint]): ListBuffer[LearningPoint] = {
    val transformedBuffer = ListBuffer[LearningPoint]()
    for (point <- dataSet) transformedBuffer.append(transform(point))
    transformedBuffer
  }

  override def generatePOJOPreprocessor: PreprocessorPOJO = {
    new PreprocessorPOJO("MinMaxScaler",
      null,
      Map[String, AnyRef](
        ("min", if (min == null) null else min.data.asInstanceOf[AnyRef]),
        ("max", if (max == null) null else max.data.asInstanceOf[AnyRef])
      ).asJava,
      null
    )
  }

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): MinMaxScaler = {
    for ((parameter, value) <- parameterMap) {
      parameter match {
        case "min" =>
          try {
            val newMin = BreezeDenseVector[Double](value.asInstanceOf[java.util.List[Double]].asScala.toArray)
            if (min == null || min.size == newMin.size)
              setMin(newMin)
            else
              throw new RuntimeException("Invalid size of new min vector for the MinMaxScaler.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the min vector of MinMaxScaler.")
              e.printStackTrace()
          }
        case "max" =>
          try {
            val newMax = BreezeDenseVector[Double](value.asInstanceOf[java.util.List[Double]].asScala.toArray)
            if (max == null || max.size == newMax.size)
              setMax(newMax)
            else
              throw new RuntimeException("Invalid size of new max vector for the MinMaxScaler.")
          } catch {
            case e: Exception =>
              println("Error while trying to update the max vector of MinMaxScaler.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

}
