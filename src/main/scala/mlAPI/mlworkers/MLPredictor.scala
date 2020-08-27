package mlAPI.mlworkers

import BipartiteTopologyAPI.NodeInstance
import BipartiteTopologyAPI.annotations.{DefaultOp, InitOp, MergeOp, ProcessOp, QueryOp}
import ControlAPI.{DataInstance, Prediction, Request}
import mlAPI.math.{DenseVector, SparseVector, UnlabeledPoint, Vector}
import mlAPI.mlParameterServers.PullPush
import mlAPI.mlpipeline.MLPipeline
import mlAPI.mlworkers.interfaces.{MLPredictorRemote, RemoteLearner}
import mlAPI.parameters.{LearningParameters, ParameterDescriptor}

import scala.collection.mutable
import scala.collection.JavaConverters._

/**
 * An ML predictor.
 */
class MLPredictor() extends NodeInstance[PullPush, MLPredictorRemote] with RemoteLearner {

  /** A TreeMap with the parameter splits. */
  protected var parameter_tree: mutable.TreeMap[(Int, Int), Vector] = _

  /** The local machine learning pipeline predictor */
  protected var ml_pipeline: MLPipeline = new MLPipeline()

  // =================================== Getters ===================================================

  def getMLPipeline: MLPipeline = ml_pipeline

  def getLearnerParams: Option[LearningParameters] = ml_pipeline.getLearner.getParameters

  // =================================== Setters ===================================================

  def setMLPipeline(ml_pipeline: MLPipeline): Unit = this.ml_pipeline = ml_pipeline

  def setLearnerParams(params: LearningParameters): Unit = ml_pipeline.getLearner.setParameters(params)

  // =================================== Periodic ML workers basic operations =======================

  /** This method configures an Online Machine Learning predictor by using a creation Request.
   *
   * @param request The creation request provided.
   * @return An [[MLPredictor]] instance.
   */
  def configureWorker(request: Request): MLPredictor = {

    // Setting the ML node parameters
    val config: mutable.Map[String, AnyRef] = request.getTraining_configuration.asScala
    if (config == null) throw new RuntimeException("Empty configuration map.")

    // Setting the ML pipeline
    ml_pipeline.configureMLPipeline(request)

    this
  }

  /** Clear the Machine Learning predictor. */
  def clear(): MLPredictor = {
    ml_pipeline.clear()
    this
  }

  /** Initialization method of the Machine Learning predictor node. */
  @InitOp
  def init(): Unit = {

  }

  /** A method called when merging two ML predictors.
   *
   * @param predictors The ML predictors to merge this one with.
   * @return An [[MLPredictor]] instance.
   */
  @MergeOp
  def merge(predictors: Array[MLPredictor]): MLPredictor = {
    for (predictor <- predictors) ml_pipeline.merge(predictor.getMLPipeline)
    this
  }

  /** This method responds to a query for the Machine Learning predictor.
   *
   * @param test_set The test set that the predictive performance of the model should be calculated on.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, test_set: Array[java.io.Serializable]): Unit = {

  }

  /** The data point to be predicted.
   *
   * @param data An unlabeled data point.
   */
  @ProcessOp
  def receiveTuple(data: DataInstance): Unit = {

    val features: (Vector, Vector, Array[String]) = {
      (if (data.getNumericalFeatures == null)
        DenseVector()
      else
        DenseVector(data.getNumericalFeatures.asInstanceOf[java.util.List[Double]].asScala.toArray),
        if (data.getDiscreteFeatures == null)
          DenseVector()
        else
          DenseVector(data.getDiscreteFeatures.asInstanceOf[java.util.List[Int]].asScala.toArray.map(x => x.toDouble)),
        if (data.getCategoricalFeatures == null)
          Array[String]()
        else
          data.getCategoricalFeatures.asScala.toArray
      )
    }

    val unlabeledPoint = UnlabeledPoint(features._1, features._2, features._3)

    val prediction = {
      ml_pipeline.predict(unlabeledPoint) match {
        case Some(prediction: Double) => prediction
        case None => Double.MaxValue
      }
    }

    getQuerier.sendPrediction(new Prediction(getNetworkID(), data, prediction))

  }

  /** A method called each type the new global model
   * (or a slice of it) arrives from the parameter server.
   */
  @DefaultOp
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    if (getNumberOfHubs == 1) {
      ml_pipeline.getLearner.setParameters(ml_pipeline.getLearner.generateParameters(mDesc).getCopy)
    } else {
      parameter_tree.put((mDesc.getBucket.getStart.toInt, mDesc.getBucket.getEnd.toInt), mDesc.getParams)
      if (parameter_tree.size == getNumberOfHubs) {
        mDesc.setParams(
          DenseVector(
            parameter_tree.values
              .map(
                {
                  case dense: DenseVector => dense
                  case sparse: SparseVector => sparse.toDenseVector
                })
              .fold(Array[Double]())(
                (accum, vector) => accum.asInstanceOf[Array[Double]] ++ vector.asInstanceOf[DenseVector].data)
              .asInstanceOf[Array[Double]]
          )
        )
        ml_pipeline.getLearner.setParameters(ml_pipeline.getLearner.generateParameters(mDesc).getCopy)
      }
    }
  }

}
