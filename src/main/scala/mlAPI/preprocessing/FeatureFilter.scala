package mlAPI.preprocessing

import ControlAPI.PreprocessorPOJO
import mlAPI.math.{DenseVector, LabeledPoint, Point, UnlabeledPoint}

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

/**
 * A method that filter attributes of data point.
 *
 * @param dropNumeric     The indexes of the numerical features that are to be dropped.
 * @param dropDiscrete    The indexes of the discrete features that are to be dropped.
 * @param dropCategorical The indexes of the categorical features that are to be dropped.
 */
case class FeatureFilter(private var dropNumeric: List[Int],
                         private var dropDiscrete: List[Int],
                         private var dropCategorical: List[Int])
  extends CompressingPreprocessor {

  import FeatureFilter._

  require(
    dropNumeric != null &&
      dropDiscrete != null &&
      dropCategorical != null &&
      dropNumeric.size == dropNumeric.distinct.size &&
      dropDiscrete.size == dropDiscrete.distinct.size &&
      dropCategorical.size == dropCategorical.distinct.size
  )

  def setDropNumeric(dropNumeric: List[Int]): FeatureFilter = {
    this.dropNumeric = dropNumeric
    this
  }

  def setDropDiscrete(dropDiscrete: List[Int]): FeatureFilter = {
    this.dropDiscrete = dropDiscrete
    this
  }

  def setDropCategorical(dropCategorical: List[Int]): FeatureFilter = {
    this.dropCategorical = dropCategorical
    this
  }

  override def transform(point: Point): Point = {

    assert(
      point.numericVector.size > dropNumeric.length &&
        point.discreteVector.size > dropDiscrete.length &&
        point.categoricalVector.length > dropCategorical.length
    )

    point match {
      case LabeledPoint(label, numericalFeatures, discreteFeatures, categoricalFeatures) =>
        LabeledPoint(label,
          DenseVector(drop(dropNumeric, numericalFeatures.toList).toArray.asInstanceOf[Array[Double]]),
          DenseVector(drop(dropDiscrete, discreteFeatures.toList).toArray.asInstanceOf[Array[Double]]),
          drop(dropCategorical, categoricalFeatures.toList).toArray.asInstanceOf[Array[String]])
      case UnlabeledPoint(numericalFeatures, discreteFeatures, categoricalFeatures) =>
        UnlabeledPoint(DenseVector(drop(dropNumeric, numericalFeatures.toList).toArray.asInstanceOf[Array[Double]]),
          DenseVector(drop(dropDiscrete, discreteFeatures.toList).toArray.asInstanceOf[Array[Double]]),
          drop(dropCategorical, categoricalFeatures.toList).toArray.asInstanceOf[Array[String]])
    }
  }

  override def transform(dataSet: ListBuffer[Point]): ListBuffer[Point] = {
    val transformedSet = ListBuffer[Point]()
    for (data <- dataSet) transformedSet.append(transform(data))
    transformedSet
  }

  def setDropList(dropList: List[Int], listIndex: Int): Unit = {
    try {
      if (listIndex == 0)
        setDropNumeric(dropList)
      else if (listIndex == 1)
        setDropDiscrete(dropList)
      else if (listIndex == 2)
        setDropCategorical(dropList)
      else throw new RuntimeException("Non recognizable drop list.")
    } catch {
      case e: Exception =>
        println("Error while trying to update a drop list of a feature filter.")
        e.printStackTrace()
      case _: Throwable =>
    }
  }

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Preprocessor = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      val dropList: List[Int] = {
        try {
          value.asInstanceOf[java.util.List[Double]].asScala.map(_.toInt).toList
        } catch {
          case _: Exception => List()
        }
      }
      if (dropList.nonEmpty)
        hyperparameter match {
          case "dropNumeric" => setDropList(dropList, 0)
          case "dropDiscrete" => setDropList(dropList, 1)
          case "dropCategorical" => setDropList(dropList, 2)
        }
    }
    this
  }

  override def generatePOJOPreprocessor: PreprocessorPOJO = {
    new PreprocessorPOJO("FeatureFilter",
      Map[String, AnyRef](("dropNumeric", dropNumeric.asInstanceOf[AnyRef]),
        ("dropDiscrete", dropDiscrete.asInstanceOf[AnyRef]),
        ("dropCategorical", dropCategorical.asInstanceOf[AnyRef])
      ).asJava,
      null,
      null
    )
  }
}

object FeatureFilter {


  // =================================== Factory methods ===========================================

  def apply(): FeatureFilter = {
    new FeatureFilter(List(), List(), List())
  }

  def apply(dropList: List[Int], index: Int): FeatureFilter = {
    require(dropList.nonEmpty && index >= 0 && index <= 2)
    if (index == 0)
      new FeatureFilter(dropList, List(), List())
    else if (index == 1)
      new FeatureFilter(List(), dropList, List())
    else
      new FeatureFilter(List(), List(), dropList)
  }

  // ====================================== Operations =============================================

  @tailrec
  def drop(dropList: List[Int], attributes: List[Any]): List[Any] = {
    if (dropList.length > attributes.length || dropList.isEmpty)
      attributes
    else
      drop(
        dropList.tail.flatMap(x => if (x == dropList.head) None else if (x < dropList.head) Some(x) else Some(x - 1)),
        for ((attr, ind) <- attributes zipWithIndex; if ind != dropList.head) yield attr
      )
  }

}