package mlAPI.mlParameterServers

import BipartiteTopologyAPI.NodeInstance
import ControlAPI.Request
import mlAPI.dataBuffers.DataSet
import mlAPI.preprocessing.RunningMean
import mlAPI.protocols.statistics.{AsynchronousStatistics, ProtocolStatistics}

/**
 * An abstract base class of a Machine Learning Parameter Server.
 *
 * @tparam WorkerIfc The remote interface of the Machine Learning worker.
 * @tparam QueryIfc  The remote interface of the querier.
 */
abstract class MLParameterServer[WorkerIfc, QueryIfc](protected var maxMsgParams: Int = 10000)
  extends NodeInstance[WorkerIfc, QueryIfc] {

  /** An object holding the statistics of the distributed training procedure. */
  protected var protocolStatistics: ProtocolStatistics = AsynchronousStatistics()

  /**
   * The cumulative loss of the distributed Machine Learning training.
   */
  protected var cumulativeLoss: Double = 0D

  /**
   * The number of data fitted to the distributed Machine Learning algorithm.
   */
  protected var fitted: Long = 0L

  /** The learning curve for the ML pipeline. */
  protected var learningCurve: DataSet[(Double, Long)] = new DataSet[(Double, Long)](100000)

  /** The mean loss of the learning round(s). */
  protected var roundLoss: RunningMean = RunningMean()

  /** An index counter. */
  protected var idx: Int = 0

  // ================================================= Getters =========================================================

  def getProtocolStatistics: ProtocolStatistics = {
    val value = protocolStatistics
    value
  }

  def getCumulativeLoss: Double = cumulativeLoss

  def getNumberOfFittedData: Long = fitted

  def getMaxMsgParams: Int = maxMsgParams

  def getLearningCurve: DataSet[(Double, Long)] = learningCurve

  def getRoundLoss: RunningMean = roundLoss

  def getIdx: Int = idx

  // ================================================= Setters =========================================================

  def setProtocolStatistics(protocolStatistics: ProtocolStatistics): Unit = this.protocolStatistics = protocolStatistics

  def setCumulativeLoss(cumulativeLoss: Double): Unit = this.cumulativeLoss = cumulativeLoss

  def setNumberOfFittedData(fitted: Long): Unit = this.fitted = fitted

  def setMaxMsgParams(maxMsgParams: Int): Unit = this.maxMsgParams = maxMsgParams

  def setLearningCurveY(learningCurve: DataSet[(Double, Long)]): Unit = this.learningCurve = learningCurve

  def setRoundLoss(roundLoss: RunningMean): Unit = this.roundLoss = roundLoss

  def setIdx(idx: Int): Unit = this.idx = idx

  // ============================== Machine Learning Parameter Server Basic Operations =================================

  /** This method configures the Parameter Server Node by using a creation Request.
   * Right now this method does not provide any functionality. It exists for configuring
   * more complex parameter server that may be developed later on.
   *
   * @param request The [[Request]] instance to configure the parameter server.
   * */
  def configureParameterServer(request: Request): MLParameterServer[WorkerIfc, QueryIfc] = {
    this
  }

  /** A method called when the Parameter Server needs to be cleared. */
  def clear(): MLParameterServer[WorkerIfc, QueryIfc] = {
    protocolStatistics.clear()
    cumulativeLoss = 0D
    fitted = 0L
    this
  }

  /** A counter used for keeping track of the number of data points fitted on the global model. */
  def incrementNumberOfFittedData(size: Long): Unit = fitted = incrementLong(fitted, size)

  def incrementLong(variable: Long, size: Long): Long = {
    if (variable != Long.MaxValue)
      if (variable < Long.MaxValue - size)
        variable + size
      else
        Long.MaxValue
    else
      variable
  }

}