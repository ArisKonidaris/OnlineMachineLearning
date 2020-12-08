package mlAPI.learners.classification

import ControlAPI.LearnerPOJO
import mlAPI.learners.classification.trees.HoeffdingTree
import mlAPI.learners.{Learner, OnlineLearner}
import mlAPI.math.{LabeledPoint, Point}
import mlAPI.parameters.{HTParameters, LearningParameters, ParameterDescriptor}
import mlAPI.scores.Scores

import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

case class HoeffdingTreeClassifier(override protected var targetLabel: Double = 1.0)
  extends OnlineLearner with Classifier with Serializable {

  var tree: HTParameters = HTParameters(new HoeffdingTree())

  override protected val parallelizable: Boolean = false

  override def predict(data: Point): Option[Double] = Some(tree.ht.predict(data.asUnlabeledPoint)._1)

  override def fit(data: Point): Unit = {
    data match {
      case labeledPoint: LabeledPoint => tree.ht.fit(labeledPoint)
      case _ =>
        throw new RuntimeException("LABEL MISSING ERROR: Cannot train the Hoeffding Tree on an unlabeled data point.")
    }
  }

  override def fitLoss(data: Point): Double = {
    data match {
      case labeledPoint: LabeledPoint => 1.0 * tree.ht.fit(labeledPoint)
      case _ =>
        throw new RuntimeException("LABEL MISSING ERROR: Cannot train the Hoeffding Tree on an unlabeled data point.")
    }
  }

  override def score(test_set: ListBuffer[Point]): Double =
    Scores.F1Score(test_set.asInstanceOf[ListBuffer[LabeledPoint]], this)

  override def getParameters: Option[LearningParameters] = Some(tree)

  override def setParameters(params: LearningParameters): Learner = {
    assert(params.isInstanceOf[HTParameters])
    tree = params.asInstanceOf[HTParameters]
    this
  }

  override def generateParameters: ParameterDescriptor => LearningParameters = new HTParameters().generateParameters

  override def getSerializedParams: (LearningParameters, Array[_]) => java.io.Serializable =
    tree.generateSerializedParams

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
