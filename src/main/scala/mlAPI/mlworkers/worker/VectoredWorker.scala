package mlAPI.mlworkers.worker

import mlAPI.math.Vector
import mlAPI.parameters.{Bucket, LearningParameters, ParameterDescriptor, SerializedVectoredParameters, VectoredParameters}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/** An abstract base class of an Online Machine Learning worker with a vectored model.
 *
 * @tparam ProxyIfc The remote interface of the Parameter Server.
 * @tparam QueryIfc The remote interface of the querier.
 */
abstract class VectoredWorker[ProxyIfc, QueryIfc]() extends MLWorker[ProxyIfc, QueryIfc] {

  /** The boundaries used to split the model into pieces. */
  protected var quantiles: ListBuffer[Bucket] = _

  /** A TreeMap with the parameter splits. */
  protected var parameterTree: mutable.TreeMap[(Int, Int), Vector] = _

  // =============================================== Getters ===========================================================

  def getQuantiles: ListBuffer[Bucket] = quantiles

  def getParameterTree: mutable.TreeMap[(Int, Int), Vector] = parameterTree

  // =============================================== Setters ===========================================================

  def setQuantiles(quantiles: ListBuffer[Bucket]): Unit = this.quantiles = quantiles

  def setParameterTree(parameterTree: mutable.TreeMap[(Int, Int), Vector]): Unit = this.parameterTree = parameterTree

  // ================================= ML vectored worker basic operations =============================================

  /** Calculates the drift of the local model compared to
   * the last global model received by the ML worker.
   *
   * @return The delta parameters.
   */
  def getDeltaVector: LearningParameters = {
    if (globalModel.getFittedData == 0)
      getMLPipelineParams.get.asInstanceOf[VectoredParameters]
    else
      getMLPipelineParams.get.asInstanceOf[VectoredParameters] - getGlobalParams.get.asInstanceOf[VectoredParameters]
  }

  /** This method creates the bucket for the splitting of the model. */
  def generateQuantiles(): Unit = {
    require(mlPipeline.getLearner.getParameters.isDefined)

    val numberOfBuckets: Int = getNumberOfHubs
    val bucketSize: Int = mlPipeline.getLearner.getParameters.get.getSize / numberOfBuckets
    val remainder: Int = mlPipeline.getLearner.getParameters.get.getSize % numberOfBuckets

    @scala.annotation.tailrec
    def createRanges(index: Int, remainder: Int, quantiles: ListBuffer[Bucket]): ListBuffer[Bucket] = {
      if (index == numberOfBuckets) return quantiles
      if (index == 0)
        quantiles.append(Bucket(0, if (remainder > 0) bucketSize else bucketSize - 1))
      else {
        val previousQ = quantiles(index - 1).getEnd
        quantiles.append(Bucket(previousQ + 1, previousQ + {
          if (remainder > 0) bucketSize + 1 else bucketSize
        }))
      }
      createRanges(index + 1, if (remainder > 0) remainder - 1 else remainder, quantiles)
    }

    quantiles = createRanges(0, remainder, ListBuffer[Bucket]())
  }

  /** Converts the model into a Serializable POJO case class to be send over the Network. */
  override def ModelMarshalling(sparse: Boolean = false, drift: Boolean = true): Array[ParameterDescriptor] = {
    try {
      split(sparse, if (drift) getDeltaVector else getMLPipelineParams.get)
    } catch {
      case _: NullPointerException =>
        generateQuantiles()
        split(sparse, if (drift) getDeltaVector else getMLPipelineParams.get)
    }
  }

  def split(sparse: Boolean = false, params: LearningParameters): Array[ParameterDescriptor] = {
    (for (bucket <- quantiles) yield {
      val (sizes, parameters) = {
        val sp = params.generateSerializedParams(params, Array(sparse, bucket)).asInstanceOf[SerializedVectoredParameters]
        (sp.sizes, sp.data)
      }
      ParameterDescriptor(sizes, parameters, bucket, null, null, processedData)
    }).toArray
  }

}
