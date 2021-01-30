package mlAPI.preprocessing

import ControlAPI.PreprocessorPOJO
import mlAPI.math.Point
import mlAPI.parameters.utils.WithParams

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * The basic trait for data pre processing methods.
 * All those methods contain hyper-parameters.
 */
trait Preprocessor extends Serializable with WithParams {

  // =============================== Data transformation methods ===================================

  def transform(point: Point): Point

  def transform(dataSet: ListBuffer[Point]): ListBuffer[Point]

  def generatePOJOPreprocessor: PreprocessorPOJO

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Preprocessor = this

  override def addHyperParameter(key: String, value: AnyRef): Preprocessor = this

  override def removeHyperParameter(key: String, value: AnyRef): Preprocessor = this

  override def setParametersFromMap(parameterMap: mutable.Map[String, AnyRef]): Preprocessor = this

  override def addParameter(key: String, value: AnyRef): Preprocessor = this

  override def removeParameter(key: String, value: AnyRef): Preprocessor = this

  def setStructureFromMap(structureMap: mutable.Map[String, AnyRef]): Preprocessor = this

}
