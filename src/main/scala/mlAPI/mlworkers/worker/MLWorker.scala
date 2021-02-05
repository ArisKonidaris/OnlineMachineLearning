package mlAPI.mlworkers.worker

import ControlAPI.Request
import BipartiteTopologyAPI.NodeInstance
import BipartiteTopologyAPI.annotations.MergeOp
import mlAPI.math.LearningPoint
import mlAPI.pipelines.MLPipeline
import mlAPI.parameters.LearningParameters
import mlAPI.parameters.utils.ParameterDescriptor
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

/** An abstract base class of an Online Machine Learning worker.
 *
 * @tparam ProxyIfc The remote interface of the Parameter Server.
 * @tparam QueryIfc The remote interface of the querier.
 * @param maxMsgParams The maximum number of parameters that can be transmitted in a single message over the network.
 */
abstract class MLWorker[ProxyIfc, QueryIfc]() extends NodeInstance[ProxyIfc, QueryIfc] {

  /** The distributed training protocol. */
  protected var protocol: String = _

  /** The total number of data points fitted to the local Machine Learning pipeline since the last synchronization. */
  protected var processedData: Long = 0

  /** The number of mini-batches fitted by the worker before checking
   * if it should push its parameters to the parameter server.
   */
  protected var miniBatches: Int = 256

  /** The warmup size of the distributed learning procedure. */
  protected var warmupSize: Int = 256

  /** The local Machine Learning pipeline to train in on streaming data. */
  protected var mlPipeline: MLPipeline = new MLPipeline()

  /** The global model. */
  protected var globalModel: MLPipeline = new MLPipeline()

  /** A flag determining if the local worker is warmed up. */
  private var warmed: Boolean = false

  /** The maximum number of parameters that can be transmitted over the network. */
  protected var maxMsgParams: Int

  // =============================================== Getters ===========================================================

  def getProtocol: String = protocol

  def getProcessedData: Long = processedData

  def getMiniBatchSize: Int = getMLPipeline.getLearner.getMiniBatchSize

  def getMiniBatches: Int = miniBatches

  def getWarmUpSize: Int = warmupSize

  def getMLPipeline: MLPipeline = mlPipeline

  def getMLPipelineParams: Option[LearningParameters] = mlPipeline.getLearner.getParameters

  def getGlobalModel: MLPipeline = globalModel

  def getGlobalParams: Option[LearningParameters] = globalModel.getLearner.getParameters

  def getMaxMsgParams: Int = maxMsgParams

  def isWarmedUp: Boolean = warmed

  // =============================================== Setters ===========================================================

  def setProtocol(protocol: String): Unit = this.protocol = protocol

  def setProcessedData(processedData: Long): Unit = this.processedData = processedData

  def setMiniBatchSize(miniBatchSize: Int): Unit = getMLPipeline.getLearner.setMiniBatchSize(miniBatchSize)

  def setMiniBatches(miniBatches: Int): Unit = this.miniBatches = miniBatches

  def setWarmupSize(warmUpSize: Int): Unit = this.warmupSize = warmUpSize

  def setMLPipeline(mlPipeline: MLPipeline): Unit = this.mlPipeline = mlPipeline

  def setMLPipelineParams(params: LearningParameters): Unit = mlPipeline.getLearner.setParameters(params)

  def setMLPipelineParams(mDesc: ParameterDescriptor): Unit = setMLPipelineParams(deserializeParams(mDesc))

  def setGlobalModel(globalModel: MLPipeline): Unit = this.globalModel = globalModel

  def setGlobalModelParams(params: LearningParameters): Unit = globalModel.getLearner.setParameters(params)

  def setGlobalModelParams(mDesc: ParameterDescriptor): Unit = setGlobalModelParams(deserializeParams(mDesc))

  def setMaxMsgParams(maxMsgParams: Int): Unit = this.maxMsgParams = maxMsgParams

  def setWarmed(warmed: Boolean): Unit = this.warmed = warmed

  // ======================================== ML worker basic operations ===============================================

  /** This method configures an Online Machine Learning worker by using a creation Request.
   *
   * @param request The creation request provided.
   * @return An [[MLWorker]] instance with Parameter Server
   *         proxies of type [[ProxyIfc]] and querier proxy type of [[QueryIfc]].
   */
  def configureWorker(request: Request): MLWorker[ProxyIfc, QueryIfc] = {

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTrainingConfiguration.asScala

    if (config == null)
      throw new RuntimeException("Empty training configuration map.")

    if (config.contains("miniBatches"))
      try {
        setMiniBatches(Parsing.IntegerParsing(config, "miniBatches", 256))
      } catch {
        case e: Throwable => e.printStackTrace()
      }

    if (config.contains("protocol"))
      try {
        setProtocol(config("protocol").asInstanceOf[String])
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    else
      setProtocol("Asynchronous")

    if (config.contains("warmupSize"))
      try {
        setWarmupSize(Parsing.IntegerParsing(config, "warmupSize", 256))
      } catch {
        case e: Throwable => e.printStackTrace()
      }

    if (config.contains("maxMsgParams"))
      try {
        setMaxMsgParams(Parsing.IntegerParsing(config, "maxMsgParams", 10000))
      } catch {
        case e: Throwable => e.printStackTrace()
      }

    // Setting the ML pipeline and the global model.
    mlPipeline.configureMLPipeline(request)
    globalModel.configureMLPipeline(request)

    this
  }

  /** Clears the Machine Learning worker. */
  def clear(): MLWorker[ProxyIfc, QueryIfc] = {
    processedData = 0
    miniBatches = 4
    mlPipeline.clear()
    globalModel.clear()
    this
  }

  /** The method called on a data point to train the ML Pipeline. */
  def fit(data: LearningPoint): Unit = {
    if ((processedData + 1) % getMiniBatchSize == 0)
      mlPipeline.fitLoss(data)
    else
      mlPipeline.fit(data)
    processedData += 1
  }

  /** A method called when merging two Machine Learning workers.
   *
   * @param workers The Machine Learning workers to merge this one with.
   * @return An array of [[MLWorker]] instances.
   */
  @MergeOp
  def merge(workers: Array[MLWorker[ProxyIfc, QueryIfc]]): MLWorker[ProxyIfc, QueryIfc] = {
    setProcessedData(0)
    setMiniBatchSize(workers(0).getMiniBatchSize)
    setMiniBatches(workers(0).getMiniBatches)
    for (worker <- workers) mlPipeline.merge(worker.getMLPipeline)
    setGlobalModel(workers(0).getGlobalModel)
    this
  }

  /** A method for calculating the performance of the local model.
   *
   * @param testSet The test set to calculate the performance on.
   * @return A String representation of the performance of the model.
   */
  def getPerformance(testSet: ListBuffer[LearningPoint]): Double = mlPipeline.score(testSet)

  /** A method for calculating the performance of the global model.
   *
   * @param testSet The test set to calculate the performance on.
   * @return A String representation of the performance of the model.
   */
  def getGlobalPerformance(testSet: ListBuffer[LearningPoint]): Double = globalModel.score(testSet)

  /** Converts the model into a Serializable POJO case class to be send over the Network. */
  def ModelMarshalling(sparse: Boolean, warm: Boolean, model: LearningParameters): Array[Array[ParameterDescriptor]]

  /** A method for deserializing the parameters received by the coordinaor. */
  def deserializeParams(mDesc: ParameterDescriptor): LearningParameters = mlPipeline.getLearner.generateParameters(mDesc)

  /** A method for updating the local and global machine learning models. */
  def updateModels(mDesc: ParameterDescriptor): Unit = {
    if (mDesc.getParamSizes == null)
      mDesc.setParamSizes(getMLPipeline.getLearner.getParameters.get.sizes)
    if (mDesc.getFitted != null)
      mlPipeline.setFittedData(mDesc.getFitted.getLong)
    setGlobalModelParams(mDesc)
    setMLPipelineParams(getGlobalParams.get.getCopy)
  }

  def warmModel(mDesc: ParameterDescriptor): Unit = {
    assert(!warmed)
    warmed = true
    setGlobalModelParams(mDesc)
    setMLPipelineParams(mDesc)
    assert(isBlocked)
    if (mDesc.getFitted != null)
      mlPipeline.setFittedData(mDesc.getFitted.getLong)
    unblockStream()
  }

}
