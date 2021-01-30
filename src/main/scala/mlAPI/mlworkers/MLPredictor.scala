package mlAPI.mlworkers

import BipartiteTopologyAPI.annotations.{InitOp, ProcessOp, QueryOp}
import ControlAPI.{DataInstance, Prediction}
import mlAPI.math.{DenseVector, Point, UnlabeledPoint, Vector}
import mlAPI.mlworkers.interfaces.MLPredictorRemote
import mlAPI.mlworkers.worker.VectoredWorker
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.protocols.IntWrapper
import mlAPI.protocols.periodic.{PushPull, RemoteLearner}

import scala.collection.JavaConverters._

/**
 * An ML predictor.
 */
class MLPredictor(override protected var maxMsgParams: Int = 10000)
  extends VectoredWorker[PushPull, MLPredictorRemote] with RemoteLearner {

  /** Initialization method of the Machine Learning predictor node. */
  @InitOp
  def init(): Unit = setWarmed(true)

  /** This method responds to a query for the Machine Learning worker.
   *
   * @param predicates The predicated of the query.
   */
  @QueryOp
  def query(queryId: Long, queryTarget: Int, predicates: (Double, Array[Point])): Unit = ()

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
      mlPipeline.predict(unlabeledPoint) match {
        case Some(prediction: Double) => prediction
        case None => Double.MaxValue
      }
    }

    getQuerier.sendPrediction(new Prediction(getNetworkID(), data, prediction))

  }

  /**
   * A method called each type the new global model
   * (or a slice of it) arrives from a parameter server.
   */
  override def updateModel(mDesc: ParameterDescriptor): Unit = {
    try {
      val spt: Int = {
        try {
          splits(getCurrentCaller)
        } catch {
          case _: Throwable =>
            assert(mDesc.getMiscellaneous != null, mDesc.getMiscellaneous.head.isInstanceOf[IntWrapper])
            splits.put(getCurrentCaller, mDesc.getMiscellaneous.head.asInstanceOf[IntWrapper].getInt)
            splits(getCurrentCaller)
        }
      }
      if (getNumberOfHubs == 1)
        if (spt == 1)
          updateModels(mDesc)
        else
          updateParameterTree(spt, mDesc, updateModels)
      else
        updateParameterTree(spt, mDesc, updateModels)
    } catch {
      case e: Throwable =>
        e.printStackTrace()
        throw new RuntimeException("Something went wrong while updating the local predictor " +
          getNodeId + " of MLPipeline " + getNetworkID + ".")
    }
  }

  override def updateModels(mDesc: ParameterDescriptor): Unit = {
    if (mDesc.getParamSizes == null)
      mDesc.setParamSizes(mlPipeline.getLearner.getParameters.get.sizes)
    setMLPipelineParams(mDesc)
  }

}
