package mlAPI.learners.classification

import ControlAPI.LearnerPOJO
import com.fasterxml.jackson.databind.ObjectMapper
import mlAPI.learners.classification.trees.HoeffdingTree
import mlAPI.learners.classification.trees.serializable.HTDescriptor
import mlAPI.learners.Learner
import mlAPI.math.{LabeledPoint, LearningPoint}
import mlAPI.parameters.utils.{ParameterDescriptor, SerializableParameters}
import mlAPI.parameters.{HTParameters, LearningParameters}
import mlAPI.scores.Scores
import mlAPI.utils.Parsing

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * A Hoeffding tree classifier.
 */
case class HoeffdingTreeClassifier() extends Classifier with Serializable {

  override protected var targetLabel: Double = 1.0
  override protected var miniBatchSize: Int = 1
  override protected val parallelizable: Boolean = false
  var tree: HTParameters = HTParameters(new HoeffdingTree())

  override def predict(data: LearningPoint): Option[Double] = Some(tree.ht.predict(data.asUnlabeledPoint)._1)

  override def predict(batch: ListBuffer[LearningPoint]): Array[Option[Double]] = {
    val predictions: ListBuffer[Option[Double]] = ListBuffer[Option[Double]]()
    for (point <- batch)
      predictions append predict(point)
    predictions.toArray
  }

  override def fit(data: LearningPoint): Unit = {
    data match {
      case labeledPoint: LabeledPoint => tree.ht.fit(labeledPoint)
      case _ =>
        throw new RuntimeException("LABEL MISSING ERROR: Cannot train the Hoeffding Tree on an unlabeled data point.")
    }
  }

  override def fitLoss(data: LearningPoint): Double = {
    data match {
      case labeledPoint: LabeledPoint => 1.0 * tree.ht.fit(labeledPoint)
      case _ =>
        throw new RuntimeException("LABEL MISSING ERROR: Cannot train the Hoeffding Tree on an unlabeled data point.")
    }
  }

  override def fit(batch: ListBuffer[LearningPoint]): Unit = {
    fitLoss(batch)
    ()
  }

  override def fitLoss(batch: ListBuffer[LearningPoint]): Double = (for (point <- batch) yield fitLoss(point)).sum

  override def score(testSet: ListBuffer[LearningPoint]): Double =
    Scores.F1Score(testSet.asInstanceOf[ListBuffer[LabeledPoint]], this)

  override def getParameters: Option[LearningParameters] = Some(tree)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[HTParameters])
    tree = params.asInstanceOf[HTParameters]
    this
  }

  override def setHyperParametersFromMap(hyperParameterMap: mutable.Map[String, AnyRef]): Learner = {
    for ((hyperparameter, value) <- hyperParameterMap) {
      hyperparameter match {
        case "miniBatchSize" =>
          try {
            miniBatchSize = Parsing.IntegerParsing(hyperParameterMap, "miniBatchSize", 1)
          } catch {
            case e: Exception =>
              println("Error while trying to update the miniBatchSize hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "discrete" =>
          try {
            tree.ht.setDiscrete(value.asInstanceOf[Boolean])
          } catch {
            case e: Exception =>
              println("Error while trying to update the discrete hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "maxByteSize" =>
          try {
            tree.ht.setMaxByteSize(Parsing.IntegerParsing(hyperParameterMap, "maxByteSize", 33554432))
          } catch {
            case e: Exception =>
              println("Error while trying to update the maxByteSize hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "n_min" =>
          try {
            tree.ht.setNMin(Parsing.DoubleParsing(hyperParameterMap, "n_min", 200).toLong)
          } catch {
            case e: Exception =>
              println("Error while trying to update the n_min hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "tau" =>
          try {
            tree.ht.setTau(Parsing.DoubleParsing(hyperParameterMap, "tau", 0.05))
          } catch {
            case e: Exception =>
              println("Error while trying to update the tau hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "delta" =>
          try {
            tree.ht.setDelta(Parsing.DoubleParsing(hyperParameterMap, "delta", 1.0E-7D))
          } catch {
            case e: Exception =>
              println("Error while trying to update the delta hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "num_of_classes" =>
          try {
            tree.ht.setNumOfClasses(Parsing.DoubleParsing(hyperParameterMap, "num_of_classes", 2))
          } catch {
            case e: Exception =>
              println("Error while trying to update the num_of_classes hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "splits" =>
          try {
            tree.ht.setSplits(Parsing.IntegerParsing(hyperParameterMap, "splits", 10))
          } catch {
            case e: Exception =>
              println("Error while trying to update the splits hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case "mem_period" =>
          try {
            tree.ht.setMemPeriod(Parsing.IntegerParsing(hyperParameterMap, "mem_period", 100000))
          } catch {
            case e: Exception =>
              println("Error while trying to update the mem_period hyper parameter of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def setStructureFromMap(structureMap: mutable.Map[String, AnyRef]): Learner = {
    for ((parameter, value) <- structureMap) {
      parameter match {
        case "serializedHT" =>
          try {
            tree.ht.deserialize(new ObjectMapper().readValue(value.asInstanceOf[String], classOf[HTDescriptor]))
          } catch {
            case e: Exception =>
              println("Error while trying to update the structure of the Hoeffding Tree classifier.")
              e.printStackTrace()
          }
        case _ =>
      }
    }
    this
  }

  override def generateParameters: ParameterDescriptor => LearningParameters = tree.generateParameters

  override def extractParams: (LearningParameters, Boolean) => SerializableParameters = tree.extractParams

  override def extractDivParams: (LearningParameters, Array[_]) => Array[Array[SerializableParameters]] =
    tree.extractDivParams

  override def generatePOJOLearner: LearnerPOJO = {
    new LearnerPOJO("HoeffdingTree",
      null,
      null,
      Map[String, AnyRef](("serializedHT",tree.ht.serialize)).asJava
    )
  }

}

object HoeffdingTreeClassifier {

  def apply(discrete: Boolean,
            maxByteSize: Int,
            n_min: Long,
            tau: Double,
            delta: Double,
            num_of_classes: Double,
            splits: Int,
            mem_period: Int): HoeffdingTreeClassifier = {
    val newHT = new HoeffdingTreeClassifier()
    newHT.tree.ht.setDiscrete(discrete)
    newHT.tree.ht.setMaxByteSize(maxByteSize)
    newHT.tree.ht.setNMin(n_min)
    newHT.tree.ht.setTau(tau)
    newHT.tree.ht.setDelta(delta)
    newHT.tree.ht.setNumOfClasses(num_of_classes)
    newHT.tree.ht.setSplits(splits)
    newHT.tree.ht.setMemPeriod(mem_period)
    newHT
  }

}
