package mlAPI.mlworkers.worker

import ControlAPI.Request
import BipartiteTopologyAPI.NodeInstance
import BipartiteTopologyAPI.annotations.MergeOp
import mlAPI.math.Point
import mlAPI.pipelines.MLPipeline
import mlAPI.parameters.{LearningParameters, ParameterDescriptor}
import mlAPI.utils.Parsing

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.collection.mutable.ListBuffer

/** An abstract base class of an Online Machine Learning worker.
 *
 * @tparam ProxyIfc The remote interface of the Parameter Server.
 * @tparam QueryIfc The remote interface of the querier.
 */
abstract class MLWorker[ProxyIfc, QueryIfc]() extends NodeInstance[ProxyIfc, QueryIfc] {

  /** The distributed training protocol. */
  protected var protocol: String = _

  /** The total number of data points fitted to the local Machine Learning pipeline since the last synchronization. */
  protected var processedData: Long = 0

  /** The size of the mini batch, or else, the number of distinct data points
   * that are fitted to the Machine Learning pipeline in single fit operation.
   */
  protected var miniBatchSize: Int = 64

  /** The number of mini-batches fitted by the worker before checking
   * if it should push its parameters to the parameter server.
   */
  protected var miniBatches: Int = 4

  /** The warmup size of the distributed learning procedure. */
  protected var warmupSize: Int = 64

  /** The local Machine Learning pipeline to train in on streaming data. */
  protected implicit var mlPipeline: MLPipeline = new MLPipeline()

  /** The global model. */
  protected var globalModel: LearningParameters = _

  /** The grace period between the calculation of the loss. */
  protected var nMinLoss: Long = 10

  // =============================================== Getters ===========================================================

  def getProtocol: String = protocol

  def getProcessedData: Long = processedData

  def getMiniBatchSize: Int = miniBatchSize

  def getMiniBatches: Int = miniBatches

  def getWarmUpSize: Int = warmupSize

  def getMLPipeline: MLPipeline = mlPipeline

  def getLearnerParams: Option[LearningParameters] = mlPipeline.getLearner.getParameters

  def getGlobalModel: LearningParameters = globalModel

  def getNMinLoss: Long = nMinLoss

  // =============================================== Setters ===========================================================

  def setProtocol(protocol: String): Unit = this.protocol = protocol

  def setProcessedData(processed_data: Long): Unit = this.processedData = processed_data

  def setMiniBatchSize(mini_batch_size: Int): Unit = this.miniBatchSize = mini_batch_size

  def setMiniBatches(mini_batches: Int): Unit = this.miniBatches = mini_batches

  def setWarmupSize(warmUpSize: Int): Unit = this.warmupSize = warmUpSize

  def setMLPipeline(ml_pipeline: MLPipeline): Unit = this.mlPipeline = ml_pipeline

  def setLearnerParams(params: LearningParameters): Unit = mlPipeline.getLearner.setParameters(params)

  def setGlobalModel(global_model: LearningParameters): Unit = this.globalModel = global_model

  def setDeepGlobalModel(global_model: LearningParameters): Unit = this.globalModel = global_model.getCopy

  def setNMinLoss(nMinLoss: Long): Unit = this.nMinLoss = nMinLoss

  // ======================================== ML worker basic operations ===============================================



  /** This method configures an Online Machine Learning worker by using a creation Request.
   *
   * @param request The creation request provided.
   * @return An [[MLWorker]] instance with Parameter Server
   *         proxies of type [[ProxyIfc]] and querier proxy type of [[QueryIfc]].
   */
  def configureWorker(request: Request): MLWorker[ProxyIfc, QueryIfc] = {

    // Setting the ML node parameters.
    val config: mutable.Map[String, AnyRef] = request.getTraining_configuration.asScala

    if (config == null) throw new RuntimeException("Empty training configuration map.")

    if (config.contains("mini_batch_size")) {
      try {
        setMiniBatchSize(Parsing.IntegerParsing(config, "mini_batch_size", 64))
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    if (config.contains("mini_batches")) {
      try {
        setMiniBatches(Parsing.IntegerParsing(config, "mini_batches", 4))
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    if (config.contains("protocol")) {
      try {
        setProtocol(config("protocol").asInstanceOf[String])
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    } else setProtocol("Asynchronous")

    if (config.contains("n_min_loss")) {
      try {
        setNMinLoss(config("n_min_loss").asInstanceOf[Double].toLong)
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    if (config.contains("warmup_size")) {
      try {
        setWarmupSize(Parsing.IntegerParsing(config, "warmup_size", 64))
      } catch {
        case e: Throwable => e.printStackTrace()
      }
    }

    // Setting the ML pipeline
    mlPipeline.configureMLPipeline(request)

    this
  }

  /** Clears the Machine Learning worker. */
  def clear(): MLWorker[ProxyIfc, QueryIfc] = {
    processedData = 0
    miniBatchSize = 64
    miniBatches = 4
    mlPipeline.clear()
    globalModel = null
    this
  }

  /** The method called on a data point to train the ML Pipeline. */
  def fit(data: Point): Unit = {
    if (processedData % nMinLoss == 0) mlPipeline.fit(data) else mlPipeline.fitLoss(data)
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
   * @param test_set The test set to calculate the performance on.
   * @return A String representation of the performance of the model.
   */
  def getPerformance(test_set: ListBuffer[Point]): String = mlPipeline.score(test_set).toString

  /** Converts the model into a Serializable POJO case class to be send over the Network. */
  def ModelMarshalling(sparse: Boolean, drift: Boolean): Array[ParameterDescriptor]

}
