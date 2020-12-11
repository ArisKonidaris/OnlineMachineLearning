package mlAPI.mlParameterServers

import BipartiteTopologyAPI.NodeInstance
import ControlAPI.Request
import mlAPI.parameters.ParameterDescriptor

/**
 * An abstract base class of a Machine Learning Parameter Server.
 *
 * @tparam WorkerIfc The remote interface of the Machine Learning worker.
 * @tparam QueryIfc  The remote interface of the querier.
 */
abstract class MLParameterServer[WorkerIfc, QueryIfc] extends NodeInstance[WorkerIfc, QueryIfc] {

  /**
   * The total number of models that the parameter server received.
   */
  protected var modelsReceived: Long = 0

  /**
   * The total number of models sent to the workers.
   */
  protected var modelsSent: Long = 0

  /**
   * The cumulative loss of the distributed Machine Learning training.
   */
  protected var cumulativeLoss: Double = 0D

  /**
   * The number of data fitted to the distributed Machine Learning algorithm.
   */
  protected var fitted: Long = 0L

  /**
   * The range of parameters that the current parameter server is responsible for.
   */
  protected var parametersDescription: ParameterDescriptor = _

  // ================================================= Getters =========================================================

  def getCumulativeLoss: Double = cumulativeLoss

  def getNumberOfFittedData: Long = fitted

  def getParameterRange: ParameterDescriptor = parametersDescription

  // ================================================= Setters =========================================================

  def setCumulativeLoss(cumulativeLoss: Double): Unit = this.cumulativeLoss = cumulativeLoss

  def setNumberOfFittedData(fitted: Long): Unit = this.fitted = fitted

  def setParameterRange(parametersDescription: ParameterDescriptor): Unit =
    this.parametersDescription = parametersDescription

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
    cumulativeLoss = 0D
    fitted = 0L
    parametersDescription = null
    this
  }

  /** A counter used for keeping track of the number of data points fitted on the global model. */
  def incrementNumberOfFittedData(size: Long): Unit = fitted = incrementLong(fitted, size)

  /** A counter used for keeping track of the number of data points fitted on the global model. */
  def incrementNumberOfReceivedModels(size: Long = 1): Unit = modelsReceived = incrementLong(modelsReceived, size)

  /** A counter used for keeping track of the number of data points fitted on the global model. */
  def incrementNumberOfShippedModels(size: Long = 1): Unit = modelsSent = incrementLong(modelsSent, size)

  def incrementLong(variable: Long, size: Long): Long = {
    if (variable != Long.MaxValue)
      if (variable < Long.MaxValue - size)
        variable + size
      else
        Long.MaxValue
    else
      variable
  }

  def printStatistics(): Unit = {
    println("Fitted: " + fitted)
    println("Models Received: " + modelsReceived)
    println("Models Sent: " + modelsSent)
    println("Communication: " + (modelsReceived + modelsSent))
  }

}